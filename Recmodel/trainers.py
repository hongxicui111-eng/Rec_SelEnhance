# -*- coding: utf-8 -*-
# @Time    : 2020/3/30 11:06
# @Author  : Hui Wang

import numpy as np
import tqdm
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from utils import recall_at_k, ndcg_k, get_metric, generate_scaled_fx




class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = True and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        if self.cuda_condition:
            self.model.cuda()

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader

        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
        self.criterion = nn.BCELoss()

    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)

    def valid(self, epoch, full_sort=False):
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)

    def test(self, epoch, full_sort=False):
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError

    def _DNS(self, batch_neg_candidates, seq_out, M):
        """
        使用动态负采样（DNS）从候选负样本中选择
        
        Args:
            batch_neg_candidates: 形状为 [B, N] 的负样本候选项
            seq_out: 形状为 [B, hidden_size] 的序列输出
            model: 推荐模型
            M: 考虑的顶部候选项数量
        
        Returns:
            形状为 [B] 的选定负样本
        """
        batch_size, N = batch_neg_candidates.size()
        device = seq_out.device
        
        # 获取模型中的项目嵌入
        with torch.no_grad():
            item_emb = self.model.item_embeddings.weight
            
            # 为每个批次计算负样本的分数
            selected_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
            
            for i in range(batch_size):
                # 获取当前序列的负样本候选项
                neg_candidates = batch_neg_candidates[i]  # [N]
                
                # 获取这些负样本候选项的嵌入
                neg_emb = item_emb[neg_candidates]  # [N, hidden_size]
                
                # 计算预测分数
                neg_scores = torch.matmul(seq_out[i].unsqueeze(0), neg_emb.transpose(0, 1)).squeeze(0)  # [N]
                
                # 获取得分最高的M个负样本
                _, top_indices = torch.topk(neg_scores, min(M, N))
                top_neg_candidates = neg_candidates[top_indices]
                
                # 从前M个中随机选择一个
                selected_idx = random.randint(0, min(M, N) - 1)
                selected_neg[i] = top_neg_candidates[selected_idx]
        
        return selected_neg
    

    def _random_neg_sampling(self, batch_neg_candidates, seq_out, M=None):
        """
        从候选负样本中随机选择一个
        
        Args:
            batch_neg_candidates: 形状为 [B, N] 的负样本候选项
            seq_out: 形状为 [B, hidden_size] 的序列输出
            M: 不使用，保留参数以保持接口一致
        
        Returns:
            形状为 [B] 的选定负样本
        """
        batch_size, N = batch_neg_candidates.size()
        device = seq_out.device
        
        # 为每个批次随机选择一个负样本
        selected_neg = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            # 获取当前序列的负样本候选项
            neg_candidates = batch_neg_candidates[i]  # [N]
            
            # 随机选择一个索引
            selected_idx = random.randint(0, N - 1)
            selected_neg[i] = neg_candidates[selected_idx]
        
        return selected_neg
        
    def _CL_Gentle(self, rating_pred: torch.Tensor, 
                        target_pos: torch.Tensor, 
                        item_sequence: torch.Tensor, 
                        top_k: int = 100) -> torch.Tensor:
        """
        从 sequence_output 中选出排名最高的负例（排除正例和已出现的物品），
        然后从前100个分数高的负例中随机选一个。

        :param sequence_output: [batch_size, num_items] 每个样本对所有物品的打分
        :param target_pos: [batch_size] 当前 batch 中每个样本的正例物品下标
        :param item_sequence: [batch_size, seq_length] 每个样本中已经出现过的物品下标
        :param top_k: 选择前 k 个高分的负例，默认值为 100
        :return: target_neg: [batch_size] 随机选出的负例物品下标
        """
        # 拷贝一份得分，避免修改原始张量
        scores = rating_pred.clone()
        batch_size, num_items = scores.shape
        
        # 构建一个与 scores 相同形状的布尔 mask
        mask = torch.zeros_like(scores, dtype=torch.bool)
        
        # 将正例物品标记到 mask 里
        mask[torch.arange(batch_size), target_pos] = True

        # 将已出现过的物品也标记到 mask 里
        for i in range(batch_size):
            mask[i, item_sequence[i]] = True

        # 对 mask 为 True 的位置赋值 -1e9，避免被选中为负例
        scores[mask] = -1e9

        # 获取每行（每个样本）前 top_k 个高分的索引
        top_k_scores, top_k_indices = torch.topk(scores, top_k, dim=1, largest=True, sorted=False)

        # 从每行的前 top_k 个负例中随机选择一个
        random_indices = torch.randint(0, top_k, (batch_size,))
        target_neg = top_k_indices[torch.arange(batch_size), random_indices]
        
        return target_neg



    # length: [length_lower_bound, length_upper_bound)
    def get_sample_scores_length(self, epoch, answers, pred_list, original_input_length, length_lower_bound, length_upper_bound):
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        filter_pred_list = []
        for i in range(len(original_input_length)):  # length filter
            if length_lower_bound <= original_input_length[i] and original_input_length[i] < length_upper_bound:
                filter_pred_list.append(pred_list[i])
        pred_list = np.array(filter_pred_list)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        print(str(length_lower_bound) + " " + str(post_fix))
        with open(self.args.log_file, 'a') as f:
            f.write(str(length_lower_bound) + " " + str(post_fix) + '\n')
        return str(post_fix)

    def get_sample_scores(self, epoch, answers, pred_list, original_input_length):
        length_lower_bound = [0, 20, 30, 40]
        length_upper_bound = [20, 30, 40, 51]
        for i in range(len(length_lower_bound)):
            self.get_sample_scores_length(epoch, answers, pred_list, original_input_length, length_lower_bound[i], length_upper_bound[i])
        # print(post_fix)
        # with open(self.args.log_file, 'a') as f:
        #     f.write(str(post_fix) + '\n')
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        # HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        # R_20 = recall_at_k(answers, pred_list, 20)
        # R_50 = recall_at_k(answers, pred_list, 50)
        R_5, NDCG_5, MRR_5 = get_metric(pred_list, 5)
        R_10, NDCG_10, MRR_10 = get_metric(pred_list, 10)
        R_20, NDCG_20, MRR_20 = get_metric(pred_list, 20)

        post_fix = {
            "Epoch": epoch,
            "HR_5": '{:.7f}'.format(R_5), "HR_10": '{:.7f}'.format(R_10), "HR_20": '{:.7f}'.format(R_20),
            "NDCG@5": '{:.7f}'.format(NDCG_5), "NDCG@10": '{:.7f}'.format(NDCG_10), "NDCG@20": '{:.7f}'.format(NDCG_20),
            "MRR@5": '{:.7f}'.format(MRR_5), "MRR@10": '{:.7f}'.format(MRR_10), "MRR@20": '{:.7f}'.format(MRR_20)
        }
        return [R_5, R_10, R_20, NDCG_5, NDCG_10, NDCG_20, MRR_5, MRR_10, MRR_20], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)

    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)

    def load(self, file_name):
        self.model.load_state_dict(torch.load(file_name))

    def cross_entropy(self,seq_out, pos_ids, neg_ids):
        # [batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        pos = pos_emb.view(-1, pos_emb.size(1))
        neg = neg_emb.view(-1, neg_emb.size(1))
        seq_emb = seq_out.view(-1, self.args.hidden_size)  # [batch hidden_size]
        pos_logits = torch.sum(pos * seq_emb, -1)  # [batch]
        neg_logits = torch.sum(neg * seq_emb, -1)
        istarget = (pos_ids > 0).view(pos_ids.size(0)).float()  # [batch]
        loss = torch.sum(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) * istarget -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24) * istarget
        ) / torch.sum(istarget) 

        return loss
    
    def bpr_loss(self, seq_out, pos_ids, neg_ids):
        """
        实现BPR (Bayesian Personalized Ranking) Loss
        
        Args:
            seq_out: 序列输出, [batch, hidden_size]
            pos_ids: 正样本ID, [batch]
            neg_ids: 负样本ID, [batch]
            
        Returns:
            loss: BPR损失值
        """
        # 获取正样本和负样本的嵌入
        pos_emb = self.model.item_embeddings(pos_ids)  # [batch, hidden_size]
        neg_emb = self.model.item_embeddings(neg_ids)  # [batch, hidden_size]
        
        # 计算序列输出和物品嵌入的内积
        pos_logits = torch.sum(pos_emb * seq_out, -1)  # [batch]
        neg_logits = torch.sum(neg_emb * seq_out, -1)  # [batch]
        
        # 确定哪些位置是有效的（pos_ids > 0表示有效位置）
        istarget = (pos_ids > 0).float()  # [batch]
        
        # 计算BPR损失: -log(sigmoid(pos_logits - neg_logits))
        # 只考虑有效位置的损失
        loss = -torch.log(torch.sigmoid(pos_logits - neg_logits) + 1e-24) * istarget
        
        # 对有效位置的损失取平均
        loss = torch.sum(loss) / torch.sum(istarget)
        
        return loss

    def info_nce_loss(self, seq_out, pos_ids, neg_ids=None):
        """
        InfoNCE (Information Noise Contrastive Estimation) Loss
        
        利用 batch 内所有样本作为负例进行对比学习，相比 BCE/BPR 只用单个负例，
        InfoNCE 提供了更强的梯度信号，能显著加速收敛。
        
        公式: L = -log( exp(sim(q, k+) / τ) / Σ_j exp(sim(q, k_j) / τ) )
        其中 q=序列输出, k+=正例物品嵌入, k_j=batch内所有物品嵌入, τ=温度系数
        
        Args:
            seq_out: 序列输出, [batch, hidden_size]
            pos_ids: 正样本ID, [batch]
            neg_ids: 不使用（InfoNCE用batch内所有样本作为负例），保留参数以保持接口一致
            
        Returns:
            loss: InfoNCE损失值
        """
        # 温度系数 τ：越小则对比越严格（梯度越尖锐），越大则越宽松
        # 经验值一般在 0.05 ~ 0.5 之间，推荐默认 0.1
        temperature = self.args.temperature if hasattr(self.args, 'temperature') else 0.1
        
        # 获取所有物品的嵌入矩阵作为负例候选池
        # [item_num, hidden_size]
        all_item_emb = self.model.item_embeddings.weight
        
        # 计算序列输出与所有物品的内积得分
        # [batch, item_num]
        logits = torch.matmul(seq_out, all_item_emb.transpose(0, 1))
        
        # 温度缩放：使softmax分布更尖锐或更平滑
        logits = logits / temperature
        
        # 构建目标标签：正例物品的索引
        # pos_ids 是每个样本正例物品的ID，直接作为 softmax 的目标类别
        istarget = (pos_ids > 0).float()  # [batch] 标记有效位置
        
        # InfoNCE loss = 交叉熵形式，将正例视为正确类别
        # batch内每个样本的正例物品是"正确答案"，其余所有物品都是"干扰项"
        loss = F.cross_entropy(logits, pos_ids, reduction='none')  # [batch]
        
        # 只计算有效位置的损失并取平均
        loss = (loss * istarget).sum() / istarget.sum()
        
        return loss

    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred


class FinetuneTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FinetuneTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )
        if self.args.loss_type=="BCE":
            self.loss = self.cross_entropy
        elif self.args.loss_type=="BPR":
            self.loss = self.bpr_loss
        elif self.args.loss_type=="InfoNCE":
            self.loss = self.info_nce_loss
        
    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar

        rec_data_iter = tqdm.tqdm(enumerate(dataloader),
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}", colour="#00ff00")
        if train:
            self.model.train()
            avg_loss = 0.0
            rec_avg_loss = 0.0
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                batch = tuple(t.to(self.device) for t in batch)
                user_id, input_ids, target_pos, target_neg, _, _ = batch
                sequence_output = self.model.finetune(input_ids)[:, -1, :]
                if self.args.CL_type=='Radical':
                    if self.args.neg_sampler=="DNS":
                        
                        if epoch >= self.args.start_epoch:
                            with torch.no_grad():
                                target_neg = self._DNS(target_neg, sequence_output, self.args.M)
                        else:
                            with torch.no_grad():
                                target_neg = self._random_neg_sampling(target_neg, sequence_output)
                                
                    elif self.args.neg_sampler=="Uniform":
                        target_neg = target_neg
                elif self.args.CL_type=="Gentle":
                    _, s = generate_scaled_fx(k=self.args.K, size=200)
                    topk = int(self.args.item_size * (1 - s[epoch]))   
                    if topk <= int(self.args.item_size * 0.005):
                        topk = int(self.args.item_size * 0.005)
                    rating_pred = self.predict_full(sequence_output)
                    with torch.no_grad():
                        target_neg = self._CL_Gentle(rating_pred, target_pos, input_ids, topk)
                
                loss = self.loss(sequence_output, target_pos, target_neg) 
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
                avg_loss += loss.item()

            post_fix = {
                "epoch": epoch,
                "loss": '{:.4f}'.format(avg_loss / len(rec_data_iter))
            }

            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, target_pos, _, answers, _ = batch
                    recommend_output = self.model.finetune(input_ids)[:, -1, :]

                    # 推荐的结果
                    rating_pred = self.predict_full(recommend_output)

                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    ind = np.argpartition(rating_pred, -20)[:, -20:]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # 再取一次 从ind中取回 原来的下标
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)
