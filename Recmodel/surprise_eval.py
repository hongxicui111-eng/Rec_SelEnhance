#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
惊喜子集评估器 — 在"惊喜"交互子集上评估模型性能，并与整体指标对比

核心功能:
1. 加载训练好的模型，在整体测试集上评估 (NDCG, Recall)
2. 在"惊喜"子集 (surprise_subset) 上单独评估
3. 在"训练子集" (从训练数据中提取的子集) 上评估
4. 计算训练子集与测试集之间的指标差距 (过拟合检测)
5. 生成对比报告，分析模型在不同子集上的性能差异
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from typing import Dict, List, Optional

from utils import get_user_seqs, generate_rating_matrix


class SurpriseEvaluator:
    """
    惊喜子集评估器
    
    在三个维度上评估模型:
    1. 整体测试集 (NDCG@K, Recall@K)
    2. 惊喜子集 (与历史行为模式差异大的交互)
    3. 训练子集 (检测过拟合程度)
    
    对比这三个维度，为 LLM 提供诊断信息
    """

    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.cuda_condition = True and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

    def evaluate_on_subset(self, user_seq, rating_matrix, 
                           subset_indices: Optional[List[int]] = None,
                           full_sort: bool = True) -> Dict:
        """
        在指定的用户子集上评估模型
        
        Args:
            user_seq: 全部用户序列数据
            rating_matrix: 评分矩阵 (用于过滤已出现的物品)
            subset_indices: 要评估的用户索引列表 (None = 全部评估)
            full_sort: 是否全排序评估
            
        Returns:
            Dict containing metrics and detailed results
        """
        from datasets import SRDataset, DNSDataset
        from trainers import FinetuneTrainer
        
        # 如果指定了子集，则只选取这些用户
        if subset_indices is not None:
            subset_seq = [user_seq[i] for i in subset_indices]
        else:
            subset_seq = user_seq
        
        neg_sampler_dict = {'Uniform': SRDataset, "DNS": DNSDataset}
        dataset_class = neg_sampler_dict.get(self.args.neg_sampler, SRDataset)
        
        dataset = dataset_class(self.args, subset_seq)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        
        self.model.eval()
        
        pred_list = None
        answer_list = None
        
        rec_data_iter = enumerate(dataloader)
        
        for i, batch in rec_data_iter:
            batch = tuple(t.to(self.device) for t in batch)
            user_ids, input_ids, target_pos, _, answers, original_input_length = batch
            
            recommend_output = self.model.finetune(input_ids)[:, -1, :]
            rating_pred = torch.matmul(
                recommend_output,
                self.model.item_embeddings.weight.transpose(0, 1)
            )
            
            rating_pred = rating_pred.cpu().data.numpy().copy()
            batch_user_index = user_ids.cpu().numpy()
            
            # 对子集用户需要映射回原始 user_seq 的索引
            if subset_indices is not None:
                original_indices = [subset_indices[idx] for idx in range(len(batch_user_index))]
                batch_original_indices = np.array(original_indices[:len(batch_user_index)])
                rating_pred[rating_matrix[batch_original_indices].toarray() > 0] = 0
            else:
                rating_pred[rating_matrix[batch_user_index].toarray() > 0] = 0
            
            ind = np.argpartition(rating_pred, -20)[:, -20:]
            arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
            arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
            batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
            
            if i == 0:
                pred_list = batch_pred_list
                answer_list = answers.cpu().data.numpy()
            else:
                pred_list = np.append(pred_list, batch_pred_list, axis=0)
                answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
        
        # 计算指标
        metrics = self._compute_metrics(answer_list, pred_list)
        
        # 收集逐用户结果 (用于惊喜度分析)
        per_user_results = self._compute_per_user_results(answer_list, pred_list)
        
        return {
            "metrics": metrics,
            "per_user": per_user_results,
            "num_users": len(subset_seq),
        }

    def _compute_metrics(self, answer_list, pred_list) -> Dict:
        """
        计算 NDCG@K, Recall@K, HR@K, MRR@K 等指标
        
        ⚠ 关键修复: pred_list 包含的是 **物品ID** (top-20 predicted item IDs),
        不是 **分数**! 不能像 trainers.py 的 get_sample_scores那样
        用 (-pred_list).argsort().argsort() 来计算 rank.
        
        正确做法: 对每个用户, 找到目标物品在 top-20 预测列表中的位置.
        
        在单目标序列推荐中:
        - Recall@K ≡ HR@K (只有 1 个目标物品, recall = hit rate)
        - NDCG@K ≤ HR@K (NDCG 考虑排名位置, 更严格)
        - MRR@K 是所有命中的 reciprocal rank 平均值
        """
        # ── 正确计算目标物品的排名 ──
        # pred_list[i] 是 top-20 predicted item IDs, 已按分数降序排列
        # answer_list[i] 是目标物品 ID
        # 目标排名 = 目标在 pred_list[i] 中的位置 (0-based)
        pred_ranks = np.full(len(answer_list), 9999, dtype=np.float64)  # 默认不在 top-20
        
        for i in range(len(answer_list)):
            target_item = int(answer_list[i])
            # 在 top-20 预测列表中查找目标物品的位置
            for rank_pos, pred_item in enumerate(pred_list[i]):
                if int(pred_item) == target_item:
                    pred_ranks[i] = rank_pos
                    break
        
        results = {}
        for k in [5, 10, 20]:
            # ── HR@K (Hit Rate): 目标是否出现在 top-K ──
            hits = (pred_ranks < k)
            hr = np.sum(hits) / len(pred_ranks)
            
            # ── NDCG@K: 考虑目标位置的折扣增益 ──
            # 对于单个目标物品: DCG = 1/log2(rank+2), IDCG = 1/log2(2) = 1
            # 所以 NDCG@K = 1/log2(rank+2) for rank < k, else 0
            ndcg_values = np.where(
                pred_ranks < k,
                1.0 / np.log2(pred_ranks + 2.0),
                0.0
            )
            ndcg = np.mean(ndcg_values)
            
            # ── MRR@K: 所有命中用户的 reciprocal rank 平均 ──
            rr_values = np.where(
                pred_ranks < k,
                1.0 / (pred_ranks + 1.0),
                0.0
            )
            mrr = np.mean(rr_values)
            
            # ── Recall@K: 单目标场景下 ≡ HR@K ──
            # Recall = |{relevant items in top-K}| / |{all relevant items}|
            # 对于单目标: Recall@K = HR@K (因为 relevant set = {target}, size = 1)
            recall = hr
            
            results[f"Recall@{k}"] = float(recall)
            results[f"NDCG@{k}"] = float(ndcg)
            results[f"HR@{k}"] = float(hr)
            results[f"MRR@{k}"] = float(mrr)
        
        return results

    def _compute_per_user_results(self, answer_list, pred_list) -> List[Dict]:
        """计算逐用户的结果 (目标物品的排名)"""
        results = []
        
        for i in range(len(answer_list)):
            target_item = int(answer_list[i])
            # 在 top-20 预测列表中查找目标物品的位置
            target_rank = 9999  # 默认不在 top-20
            for rank_pos, pred_item in enumerate(pred_list[i]):
                if int(pred_item) == target_item:
                    target_rank = rank_pos
                    break
            
            top_k_preds = pred_list[i].tolist()
            results.append({
                "user_index": i,
                "target": target_item,
                "target_rank": target_rank,
                "in_top5": target_rank < 5,
                "in_top10": target_rank < 10,
                "in_top20": target_rank < 20,
                "top20_preds": top_k_preds,
            })
        
        return results

    def full_evaluation(self, train_data, valid_data, test_data,
                        train_matrix, valid_matrix, test_matrix,
                        surprise_subset_path: str = None,
                        item_text_map: Dict = None,
                        num_train_subset: int = 500) -> Dict:
        """
        完整评估流程
        
        1. 整体测试集评估
        2. 惊喜子集评估
        3. 训练子集评估 (随机选取 num_train_subset 个训练用户)
        4. 对比分析
        
        Returns:
            Full evaluation report dict
        """
        report = {}
        
        # --- 1. 整体测试集评估 ---
        print("=== Evaluating on full test set ===")
        test_result = self.evaluate_on_subset(test_data, test_matrix)
        report["test_full"] = test_result["metrics"]
        print(f"  Test metrics: {test_result['metrics']}")
        
        # --- 2. 整体验证集评估 ---
        print("=== Evaluating on full validation set ===")
        val_result = self.evaluate_on_subset(valid_data, valid_matrix)
        report["val_full"] = val_result["metrics"]
        print(f"  Val metrics: {val_result['metrics']}")
        
        # --- 3. 训练子集评估 ---
        print(f"=== Evaluating on training subset ({num_train_subset} users) ===")
        import random
        train_subset_indices = random.sample(range(len(train_data)), 
                                              min(num_train_subset, len(train_data)))
        train_subset_result = self.evaluate_on_subset(
            train_data, train_matrix, subset_indices=train_subset_indices
        )
        report["train_subset"] = train_subset_result["metrics"]
        report["train_subset_indices"] = train_subset_indices
        print(f"  Train subset metrics: {train_subset_result['metrics']}")
        
        # --- 4. 惊喜子集评估 ---
        # 加载惊喜子集 (从之前提取的 wrong_cases 中选取高惊喜度的)
        surprise_indices = None
        if surprise_subset_path and os.path.exists(surprise_subset_path):
            with open(surprise_subset_path, 'r', encoding='utf-8') as f:
                surprise_data = json.load(f)
            # 惊喜子集的用户在 test split 中
            surprise_indices = [c["user_id"] for c in surprise_data]
            # 过滤掉不在 test_data 范围内的
            surprise_indices = [idx for idx in surprise_indices if idx < len(test_data)]
            if surprise_indices:
                print(f"=== Evaluating on surprise subset ({len(surprise_indices)} users) ===")
                surprise_result = self.evaluate_on_subset(
                    test_data, test_matrix, subset_indices=surprise_indices
                )
                report["surprise_subset"] = surprise_result["metrics"]
                print(f"  Surprise subset metrics: {surprise_result['metrics']}")
            else:
                print("  No surprise indices found in test data")
                report["surprise_subset"] = None
        
        # --- 5. 对比分析 ---
        report["comparison"] = self._compute_comparison(report)
        
        # --- 6. 生成诊断摘要 ---
        report["diagnosis"] = self._generate_diagnosis(report, item_text_map)
        
        return report

    def _compute_comparison(self, report: Dict) -> Dict:
        """
        计算各子集之间的指标差距
        
        重点指标:
        - train_subset vs test_full: 过拟合程度
        - surprise_subset vs test_full: 模型对惊喜交互的捕获能力
        - val_full vs test_full: 验证集与测试集的一致性
        """
        comparison = {}
        
        test_metrics = report.get("test_full", {})
        train_metrics = report.get("train_subset", {})
        surprise_metrics = report.get("surprise_subset", {})
        val_metrics = report.get("val_full", {})
        
        # 过拟合差距: train_subset - test_full
        if test_metrics and train_metrics:
            overfit_gap = {}
            for key in ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]:
                if key in train_metrics and key in test_metrics:
                    gap = train_metrics[key] - test_metrics[key]
                    overfit_gap[key] = {
                        "train": train_metrics[key],
                        "test": test_metrics[key],
                        "gap": gap,
                        "gap_pct": gap / test_metrics[key] * 100 if test_metrics[key] > 0 else 0,
                    }
            comparison["overfit_gap"] = overfit_gap
        
        # 惊喜差距: surprise_subset - test_full
        if test_metrics and surprise_metrics:
            surprise_gap = {}
            for key in ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]:
                if key in surprise_metrics and key in test_metrics:
                    gap = surprise_metrics[key] - test_metrics[key]
                    surprise_gap[key] = {
                        "surprise": surprise_metrics[key],
                        "test": test_metrics[key],
                        "gap": gap,
                        "gap_pct": gap / test_metrics[key] * 100 if test_metrics[key] > 0 else 0,
                    }
            comparison["surprise_gap"] = surprise_gap
        
        # 验证集一致性: val - test
        if test_metrics and val_metrics:
            val_test_consistency = {}
            for key in ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]:
                if key in val_metrics and key in test_metrics:
                    gap = val_metrics[key] - test_metrics[key]
                    val_test_consistency[key] = {
                        "val": val_metrics[key],
                        "test": test_metrics[key],
                        "gap": gap,
                    }
            comparison["val_test_consistency"] = val_test_consistency
        
        return comparison

    def _generate_diagnosis(self, report: Dict, item_text_map: Dict = None) -> Dict:
        """
        根据对比结果生成诊断摘要
        
        诊断维度:
        1. 过拟合检测: train_subset >> test_full → 可能过拟合
        2. 惊喜捕获能力: surprise_subset << test_full → 模型难以捕获惊喜交互
        3. 指标短板: NDCG vs Recall 的差异
        4. 序列长度效应: 短序列 vs 长序列的性能差异
        """
        diagnosis = {}
        
        comparison = report.get("comparison", {})
        
        # --- 过拟合检测 ---
        overfit_gap = comparison.get("overfit_gap", {})
        if overfit_gap:
            avg_ndcg_gap = np.mean([
                v["gap_pct"] for k, v in overfit_gap.items() 
                if "NDCG" in k and isinstance(v["gap_pct"], (int, float))
            ]) if overfit_gap else 0
            
            avg_recall_gap = np.mean([
                v["gap_pct"] for k, v in overfit_gap.items() 
                if "Recall" in k and isinstance(v["gap_pct"], (int, float))
            ]) if overfit_gap else 0
            
            if avg_ndcg_gap > 10 or avg_recall_gap > 10:
                diagnosis["overfitting"] = "SEVERE"
                diagnosis["overfit_note"] = f"训练子集指标显著高于测试集 (NDCG差{avg_ndcg_gap:.1f}%, Recall差{avg_recall_gap:.1f}%), 表明模型可能存在过拟合"
            elif avg_ndcg_gap > 5 or avg_recall_gap > 5:
                diagnosis["overfitting"] = "MODERATE"
                diagnosis["overfit_note"] = f"训练子集指标略高于测试集 (NDCG差{avg_ndcg_gap:.1f}%, Recall差{avg_recall_gap:.1f}%), 需注意防止过拟合加剧"
            else:
                diagnosis["overfitting"] = "LOW"
                diagnosis["overfit_note"] = f"训练与测试指标差距不大 (NDCG差{avg_ndcg_gap:.1f}%, Recall差{avg_recall_gap:.1f}%), 过拟合风险低"
        
        # --- 惊喜捕获能力 ---
        surprise_gap = comparison.get("surprise_gap", {})
        if surprise_gap:
            avg_surprise_ndcg_gap = np.mean([
                v["gap_pct"] for k, v in surprise_gap.items() 
                if "NDCG" in k and isinstance(v["gap_pct"], (int, float))
            ]) if surprise_gap else 0
            
            avg_surprise_recall_gap = np.mean([
                v["gap_pct"] for k, v in surprise_gap.items() 
                if "Recall" in k and isinstance(v["gap_pct"], (int, float))
            ]) if surprise_gap else 0
            
            if avg_surprise_ndcg_gap < -20 or avg_surprise_recall_gap < -20:
                diagnosis["surprise_capture"] = "VERY_POOR"
                diagnosis["surprise_note"] = f"模型在惊喜交互上表现极差 (NDCG差{avg_surprise_ndcg_gap:.1f}%, Recall差{avg_surprise_recall_gap:.1f}%), 模型几乎无法捕获与历史模式差异大的交互"
            elif avg_surprise_ndcg_gap < -10 or avg_surprise_recall_gap < -10:
                diagnosis["surprise_capture"] = "POOR"
                diagnosis["surprise_note"] = f"模型在惊喜交互上表现较差 (NDCG差{avg_surprise_ndcg_gap:.1f}%, Recall差{avg_surprise_recall_gap:.1f}%), 需增强模型对新颖交互的感知能力"
            elif avg_surprise_ndcg_gap < -5 or avg_surprise_recall_gap < -5:
                diagnosis["surprise_capture"] = "MODERATE"
                diagnosis["surprise_note"] = f"模型在惊喜交互上有一定不足 (NDCG差{avg_surprise_ndcg_gap:.1f}%, Recall差{avg_surprise_recall_gap:.1f}%), 可以进一步优化"
            else:
                diagnosis["surprise_capture"] = "GOOD"
                diagnosis["surprise_note"] = f"模型在惊喜交互上表现接近整体水平 (NDCG差{avg_surprise_ndcg_gap:.1f}%, Recall差{avg_surprise_recall_gap:.1f}%)"
        
        # --- NDCG vs Recall 差异 ---
        test_metrics = report.get("test_full", {})
        if test_metrics:
            ndcg_10 = test_metrics.get("NDCG@10", 0)
            recall_10 = test_metrics.get("Recall@10", 0)
            
            if ndcg_10 > recall_10 + 0.05:
                diagnosis["metric_balance"] = "NDCG_DOMINANT"
                diagnosis["metric_note"] = f"NDCG@10={ndcg_10:.4f} >> Recall@10={recall_10:.4f}, 模型排序较好但覆盖面不足, 建议增加推荐多样性"
            elif recall_10 > ndcg_10 + 0.05:
                diagnosis["metric_balance"] = "RECALL_DOMINANT"
                diagnosis["metric_note"] = f"Recall@10={recall_10:.4f} >> NDCG@10={ndcg_10:.4f}, 模型覆盖面好但排序不够精确, 建议优化排序能力"
            else:
                diagnosis["metric_balance"] = "BALANCED"
                diagnosis["metric_note"] = f"NDCG@10={ndcg_10:.4f}, Recall@10={recall_10:.4f}, 指标较为平衡"
        
        return diagnosis

    def save_report(self, report: Dict, output_path: str):
        """保存评估报告"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        # 移除不可序列化的内容
        saveable_report = {
            "test_full": report.get("test_full"),
            "val_full": report.get("val_full"),
            "train_subset": report.get("train_subset"),
            "surprise_subset": report.get("surprise_subset"),
            "comparison": report.get("comparison"),
            "diagnosis": report.get("diagnosis"),
        }
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(saveable_report, f, indent=2, ensure_ascii=False)
        print(f"Saved evaluation report to {output_path}")


def run_surprise_evaluation(args, model, checkpoint_path: str = None,
                            surprise_subset_path: str = None,
                            item_text_map_path: str = None,
                            output_dir: str = "analysis_output",
                            num_train_subset: int = 500) -> Dict:
    """
    完整的惊喜评估流程
    
    Args:
        args: 模型参数
        model: 模型实例 (如果 None, 会从 checkpoint 加载)
        checkpoint_path: 模型 checkpoint 路径 (如果 model 未提供)
        surprise_subset_path: 惊喜子集 JSON 文件路径
        item_text_map_path: 物品文本映射 JSON 文件路径
        output_dir: 输出目录
        num_train_subset: 训练子集用户数量
    """
    from utils import get_user_seqs, generate_rating_matrix
    from models import SASRec
    
    # 加载物品文本映射 (id_meta_data.json 格式: nested dict)
    item_text_map = {}
    if item_text_map_path and os.path.exists(item_text_map_path):
        with open(item_text_map_path, 'r', encoding='utf-8') as f:
            item_text_map = json.load(f)
    else:
        # 查找默认路径: id_meta_data.json
        default_path = os.path.join(os.path.dirname(args.data_dir.rstrip('/')), "id_meta_data.json")
        if not os.path.exists(default_path):
            default_path = os.path.join(args.data_dir, "id_meta_data.json")
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                item_text_map = json.load(f)
    
    # 加载模型
    if model is None:
        model = SASRec(args=args)
        if checkpoint_path:
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Loaded model from {checkpoint_path}")
    
    # 加载数据
    data_file = args.data_dir + args.data_name
    train_data, max_item_train, _ = get_user_seqs(data_file + "_train.txt")
    valid_data, max_item_val, _ = get_user_seqs(data_file + "_val.txt")
    test_data, max_item_test, _ = get_user_seqs(data_file + "_test.txt")
    max_item = max(max_item_train, max_item_val, max_item_test)
    args.item_size = max_item + 2
    
    train_matrix = generate_rating_matrix(train_data, args.item_size)
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)
    
    # 创建评估器
    evaluator = SurpriseEvaluator(args, model)
    
    # 执行评估
    report = evaluator.full_evaluation(
        train_data, valid_data, test_data,
        train_matrix, valid_matrix, test_matrix,
        surprise_subset_path=surprise_subset_path,
        item_text_map=item_text_map,
        num_train_subset=num_train_subset,
    )
    
    # 保存报告
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"surprise_eval_report_{args.data_name}.json")
    evaluator.save_report(report, report_path)
    
    return report