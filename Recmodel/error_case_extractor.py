#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
错误案例提取器 — 从推荐模型推理结果中提取预测错误的交互，
并将其转化为文本交互格式供 LLM 分析

核心功能:
1. 加载训练好的模型，对 train/val/test 数据进行全量推理
2. 找出每个用户预测排名中目标物品不在 Top-K 的错误案例
3. 将错误案例的 (用户序列 → 目标物品) 转化为文本描述
4. 随机选择 N 个错误案例供 LLM 分析
5. 构建"惊喜"子集 (surprise subset): 用户序列中与历史行为模式差异大的交互
"""

import os
import json
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional


class ErrorCaseExtractor:
    """
    从推荐模型推理结果中提取预测错误的交互并转化为文本
    
    用法:
        extractor = ErrorCaseExtractor(args, model, item_text_map)
        wrong_cases = extractor.extract_wrong_cases(train_data, train_matrix, topk=20)
        text_cases = extractor.convert_to_text(wrong_cases, num_samples=500)
    """

    def __init__(self, args, model, item_text_map: Dict = None):
        self.args = args
        self.model = model
        self.item_text_map = item_text_map or {}  # 支持 flat str 或 nested dict 格式
        
        self.cuda_condition = True and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")
        
    def predict_full_batch(self, dataloader, rating_matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        对整个 dataloader 进行全量推理
        
        Returns:
            pred_list: [num_users, topk] 每个用户的 top-K 预测物品列表
            answer_list: [num_users] 每个用户的目标物品
            user_ids: [num_users] 每个用户的 ID
        """
        self.model.eval()
        pred_list = None
        answer_list = None
        user_ids_list = None
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                batch = tuple(t.to(self.device) for t in batch)
                user_ids, input_ids, target_pos, _, answers, original_input_length = batch
                
                recommend_output = self.model.finetune(input_ids)[:, -1, :]
                rating_pred = torch.matmul(
                    recommend_output, 
                    self.model.item_embeddings.weight.transpose(0, 1)
                )
                
                rating_pred = rating_pred.cpu().data.numpy().copy()
                batch_user_index = user_ids.cpu().numpy()
                
                # 将训练集中已出现的物品分数设为 0
                rating_pred[rating_matrix[batch_user_index].toarray() > 0] = 0
                
                # 取 top-20
                ind = np.argpartition(rating_pred, -20)[:, -20:]
                arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]
                
                if i == 0:
                    pred_list = batch_pred_list
                    answer_list = answers.cpu().data.numpy()
                    user_ids_list = batch_user_index
                    orig_lengths = original_input_length.cpu().numpy()
                else:
                    pred_list = np.append(pred_list, batch_pred_list, axis=0)
                    answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                    user_ids_list = np.append(user_ids_list, batch_user_index, axis=0)
                    orig_lengths = np.append(orig_lengths, original_input_length.cpu().numpy(), axis=0)
        
        return pred_list, answer_list, user_ids_list, orig_lengths

    def extract_wrong_cases(self, user_seq, rating_matrix, topk: int = 20,
                            dataset_class=None) -> List[Dict]:
        """
        从数据中提取模型预测错误的案例
        
        Args:
            user_seq: 用户序列数据 (list of lists)
            rating_matrix: 训练矩阵 (用于过滤已出现物品)
            topk: 如果目标物品不在 top-K 中，视为错误案例
            
        Returns:
            List of dicts, each containing:
            - user_id: 用户 ID
            - history: 用户历史交互序列 (item ids)
            - target: 目标物品 ID
            - predictions: 模型 top-K 预测物品列表
            - target_rank: 目标物品在预测中的排名 (若不在 top-K, 则为 -1)
            - original_length: 原始序列长度
        """
        # 创建 dataset 和 dataloader
        from datasets import SRDataset, DNSDataset
        if dataset_class is None:
            neg_sampler_dict = {'Uniform': SRDataset, "DNS": DNSDataset}
            dataset_class = neg_sampler_dict.get(self.args.neg_sampler, SRDataset)
        
        dataset = dataset_class(self.args, user_seq)
        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.batch_size)
        
        # 推理
        pred_list, answer_list, user_ids, orig_lengths = self.predict_full_batch(
            dataloader, rating_matrix
        )
        
        # 提取错误案例
        wrong_cases = []
        for i in range(len(user_ids)):
            target = int(answer_list[i])
            predictions = pred_list[i].tolist()
            
            # 计算目标物品在预测中的排名
            target_rank = -1
            for rank, pred_item in enumerate(predictions):
                if pred_item == target:
                    target_rank = rank
                    break
            
            # 如果目标不在 top-K 中，就是错误案例
            if target_rank == -1 or target_rank >= topk:
                history = user_seq[i][:-1]  # 去掉最后一个 (目标)
                wrong_cases.append({
                    "user_id": int(user_ids[i]),
                    "history": [int(x) for x in history if x > 0],  # 过滤 padding
                    "target": target,
                    "predictions": predictions[:topk],
                    "target_rank": target_rank,
                    "original_length": int(orig_lengths[i]),
                })
        
        return wrong_cases

    def extract_wrong_cases_from_splits(self, train_data, valid_data, test_data,
                                         train_matrix, valid_matrix, test_matrix,
                                         topk: int = 20) -> Dict[str, List[Dict]]:
        """
        从 train/val/test 三个 split 中分别提取错误案例
        
        Returns:
            Dict with keys 'train', 'val', 'test', each containing list of wrong cases
        """
        results = {}
        
        print(f"Extracting wrong cases from train split (topk={topk})...")
        results['train'] = self.extract_wrong_cases(train_data, train_matrix, topk)
        print(f"  Train: {len(results['train'])} wrong cases out of {len(train_data)} users")
        
        print(f"Extracting wrong cases from val split (topk={topk})...")
        results['val'] = self.extract_wrong_cases(valid_data, valid_matrix, topk)
        print(f"  Val: {len(results['val'])} wrong cases out of {len(valid_data)} users")
        
        print(f"Extracting wrong cases from test split (topk={topk})...")
        results['test'] = self.extract_wrong_cases(test_data, test_matrix, topk)
        print(f"  Test: {len(results['test'])} wrong cases out of {len(test_data)} users")
        
        return results

    def item_id_to_text(self, item_id: int) -> str:
        """将物品 ID 转化为文本描述
        
        支持 id_meta_data.json 格式 (nested dict with title/categories/description)
        和 flat string 格式 (id -> text)
        """
        str_id = str(item_id)
        if str_id in self.item_text_map:
            entry = self.item_text_map[str_id]
            # id_meta_data.json 格式: dict with title/categories/description
            if isinstance(entry, dict):
                title = entry.get("title", "")
                categories = entry.get("categories", "")
                # 构造简洁文本: "Title [Category]"
                parts = []
                if title:
                    parts.append(title)
                if categories:
                    # 取类别链的最细粒度 (最后一个)
                    cat_list = categories.split(" > ")
                    leaf_cat = cat_list[-1] if cat_list else categories
                    parts.append(f"[{leaf_cat}]")
                return " | ".join(parts) if parts else f"Item_{item_id}"
            # flat string 格式: 直接返回
            elif isinstance(entry, str):
                return entry
        # fallback: 用 ID 本身
        return f"Item_{item_id}"

    def convert_case_to_text(self, case: Dict) -> Dict:
        """
        将一个错误案例转化为文本交互描述
        
        Returns:
            Dict containing:
            - user_id: 用户 ID
            - history_text: 用户历史交互的文本列表
            - target_text: 目标物品的文本
            - predictions_text: 模型预测的文本列表
            - target_rank: 目标排名
            - surprise_score: 惊喜度评分 (目标与历史行为的差异度)
        """
        # 转化历史序列
        history_text = [self.item_id_to_text(item) for item in case["history"]]
        target_text = self.item_id_to_text(case["target"])
        predictions_text = [self.item_id_to_text(item) for item in case["predictions"]]
        
        # 计算惊喜度: 目标物品与历史行为的差异程度
        surprise_score = self._compute_surprise_score(case)
        
        return {
            "user_id": case["user_id"],
            "history_ids": case["history"],
            "history_text": history_text,
            "target_id": case["target"],
            "target_text": target_text,
            "predictions_ids": case["predictions"],
            "predictions_text": predictions_text,
            "target_rank": case["target_rank"],
            "original_length": case["original_length"],
            "surprise_score": surprise_score,
        }

    def _compute_surprise_score(self, case: Dict) -> float:
        """
        计算惊喜度评分 — 衡量目标物品与用户历史行为模式的差异
        
        惊喜度定义: 目标物品是否偏离了用户的历史交互模式
        - 目标物品的类别在历史中从未出现 → 高惊喜度
        - 目标物品与历史物品类别高度相似 → 低惊喜度
        
        计算方式:
        1. 基于类别的差异度 (利用 id_meta_data.json 中的 categories)
        2. 基于预测排名的补充 (排名越低越难预测 → 更惊喜)
        """
        history = case["history"]
        target = case["target"]
        
        if len(history) == 0:
            return 1.0  # 无历史，完全惊喜
        
        # 基于物品 ID 的简单差异度: 目标是否在历史中出现过
        if target in history:
            return 0.0  # 已出现过，不是惊喜
        
        # --- 基于类别的惊喜度 (核心改进) ---
        # 获取目标物品的叶类别
        target_leaf_cat = self._get_leaf_category(target)
        history_leaf_cats = set()
        for h_item in history:
            history_leaf_cats.add(self._get_leaf_category(h_item))
        
        # 如果目标类别从未出现在历史中 → 高惊喜度
        if target_leaf_cat and target_leaf_cat not in history_leaf_cats:
            category_surprise = 1.0
        elif target_leaf_cat in history_leaf_cats:
            category_surprise = 0.0
        else:
            category_surprise = 0.5  # 无类别信息，中等惊喜度
        
        # --- 基于预测排名的惊喜度 ---
        rank = case["target_rank"]
        if rank == -1:
            rank_surprise = 1.0  # 完全没预测到
        else:
            rank_surprise = min(rank / 20.0, 1.0)
        
        # 综合惊喜度: 类别差异占 70%, 排名差异占 30%
        surprise = category_surprise * 0.7 + rank_surprise * 0.3
        return surprise

    def _get_leaf_category(self, item_id: int) -> str:
        """获取物品的叶类别 (类别链中最细粒度的类别)
        
        利用 id_meta_data.json 中的 categories 字段
        格式: "Sports & Outdoors > Outdoor Gear > Camping & Hiking > ..."
        """
        str_id = str(item_id)
        if str_id in self.item_text_map:
            entry = self.item_text_map[str_id]
            if isinstance(entry, dict):
                categories = entry.get("categories", "")
                if categories:
                    cat_list = categories.split(" > ")
                    return cat_list[-1] if cat_list else categories
        return ""  # 无类别信息

    def convert_to_text(self, wrong_cases: List[Dict], 
                        num_samples: int = 500,
                        prioritize_surprise: bool = True) -> List[Dict]:
        """
        将错误案例转化为文本交互描述，并随机选取指定数量
        
        Args:
            wrong_cases: 错误案例列表
            num_samples: 要选取的案例数量
            prioritize_surprise: 是否优先选取高惊喜度的案例
            
        Returns:
            List of text-converted cases
        """
        # 先全部转化为文本
        text_cases = [self.convert_case_to_text(case) for case in wrong_cases]
        
        if prioritize_surprise:
            # 按惊喜度排序，优先选取高惊喜度的案例
            text_cases.sort(key=lambda x: x["surprise_score"], reverse=True)
            # 从高惊喜度中随机选取，确保多样性
            high_surprise = [c for c in text_cases if c["surprise_score"] >= 0.5]
            low_surprise = [c for c in text_cases if c["surprise_score"] < 0.5]
            
            # 70% 来自高惊喜度，30% 来自低惊喜度
            n_high = min(int(num_samples * 0.7), len(high_surprise))
            n_low = min(num_samples - n_high, len(low_surprise))
            
            selected = random.sample(high_surprise, n_high) + random.sample(low_surprise, n_low)
        else:
            selected = random.sample(text_cases, min(num_samples, len(text_cases)))
        
        return selected

    def build_surprise_subset(self, wrong_cases: List[Dict],
                              surprise_threshold: float = 0.5) -> List[Dict]:
        """
        构建"惊喜"子集 — 只保留高惊喜度的错误案例
        
        这些案例的目标物品与用户历史行为模式差异大，
        是模型最难预测的部分，也是核心优化方向
        """
        text_cases = [self.convert_case_to_text(case) for case in wrong_cases]
        surprise_subset = [c for c in text_cases if c["surprise_score"] >= surprise_threshold]
        return surprise_subset

    def compute_surprise_metrics(self, wrong_cases: List[Dict],
                                 all_cases_count: int) -> Dict:
        """
        计算惊喜相关的统计指标
        
        Returns:
            Dict containing:
            - wrong_ratio: 错误率
            - surprise_wrong_ratio: 惊喜案例中的错误率
            - avg_surprise_score: 平均惊喜度
            - surprise_distribution: 惊喜度分布统计
        """
        text_cases = [self.convert_case_to_text(case) for case in wrong_cases]
        
        wrong_ratio = len(wrong_cases) / all_cases_count if all_cases_count > 0 else 0
        
        surprise_cases = [c for c in text_cases if c["surprise_score"] >= 0.5]
        non_surprise_cases = [c for c in text_cases if c["surprise_score"] < 0.5]
        
        avg_surprise = np.mean([c["surprise_score"] for c in text_cases]) if text_cases else 0
        
        # 惊喜度分布
        surprise_bins = defaultdict(int)
        for c in text_cases:
            bucket = round(c["surprise_score"], 1)
            surprise_bins[bucket] += 1
        
        return {
            "total_wrong_cases": len(wrong_cases),
            "total_users": all_cases_count,
            "wrong_ratio": wrong_ratio,
            "surprise_wrong_count": len(surprise_cases),
            "non_surprise_wrong_count": len(non_surprise_cases),
            "avg_surprise_score": avg_surprise,
            "surprise_distribution": dict(surprise_bins),
        }

    def save_text_cases(self, text_cases: List[Dict], output_path: str):
        """保存文本案例到 JSON 文件"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(text_cases, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(text_cases)} text cases to {output_path}")

    def load_text_cases(self, input_path: str) -> List[Dict]:
        """从 JSON 文件加载文本案例"""
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)


def run_extraction(args, model, item_text_map_path: str = None,
                   output_dir: str = "analysis_output",
                   num_samples: int = 500,
                   topk: int = 20):
    """
    完整的错误案例提取流程
    
    1. 加载数据
    2. 构建矩阵
    3. 对每个 split 提取错误案例
    4. 转化为文本并选取 500 个
    5. 保存结果
    """
    from utils import get_user_seqs, generate_rating_matrix
    
    # 加载 id-to-text 映射 (id_meta_data.json 格式: nested dict with title/categories/description)
    item_text_map = {}
    if item_text_map_path and os.path.exists(item_text_map_path):
        with open(item_text_map_path, 'r', encoding='utf-8') as f:
            item_text_map = json.load(f)
        print(f"Loaded item text mapping: {len(item_text_map)} items from {item_text_map_path}")
    else:
        # 查找默认路径: id_meta_data.json
        default_path = os.path.join(args.data_dir, "id_meta_data.json")
        if os.path.exists(default_path):
            with open(default_path, 'r', encoding='utf-8') as f:
                item_text_map = json.load(f)
            print(f"Loaded item text mapping: {len(item_text_map)} items from {default_path}")
    
    # 加载数据
    data_file = args.data_dir + args.data_name
    train_data, max_item_train, _ = get_user_seqs(data_file + "_train.txt")
    valid_data, max_item_val, _ = get_user_seqs(data_file + "_val.txt")
    test_data, max_item_test, _ = get_user_seqs(data_file + "_test.txt")
    max_item = max(max_item_train, max_item_val, max_item_test)
    args.item_size = max_item + 2
    
    # 构建评分矩阵
    train_matrix = generate_rating_matrix(train_data, args.item_size)
    valid_matrix = generate_rating_matrix(valid_data, args.item_size)
    test_matrix = generate_rating_matrix(test_data, args.item_size)
    
    # 创建提取器
    extractor = ErrorCaseExtractor(args, model, item_text_map)
    
    # 从三个 split 提取错误案例
    wrong_cases_by_split = extractor.extract_wrong_cases_from_splits(
        train_data, valid_data, test_data,
        train_matrix, valid_matrix, test_matrix,
        topk=topk
    )
    
    # 合并所有错误案例
    all_wrong_cases = []
    for split_name, cases in wrong_cases_by_split.items():
        all_wrong_cases.extend(cases)
    
    print(f"\nTotal wrong cases across all splits: {len(all_wrong_cases)}")
    
    # 转化为文本并选取 500 个
    text_cases = extractor.convert_to_text(all_wrong_cases, num_samples=num_samples)
    
    # 构建惊喜子集
    surprise_subset = extractor.build_surprise_subset(all_wrong_cases)
    
    # 计算惊喜指标
    surprise_metrics = {}
    for split_name, cases in wrong_cases_by_split.items():
        split_count = len(train_data) if split_name == 'train' else \
                      len(valid_data) if split_name == 'val' else len(test_data)
        surprise_metrics[split_name] = extractor.compute_surprise_metrics(cases, split_count)
    
    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    
    extractor.save_text_cases(
        text_cases, 
        os.path.join(output_dir, f"wrong_cases_text_{args.data_name}_{num_samples}.json")
    )
    extractor.save_text_cases(
        surprise_subset,
        os.path.join(output_dir, f"surprise_subset_{args.data_name}.json")
    )
    
    # 保存惊喜指标
    metrics_path = os.path.join(output_dir, f"surprise_metrics_{args.data_name}.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(surprise_metrics, f, indent=2, ensure_ascii=False)
    print(f"Saved surprise metrics to {metrics_path}")
    
    # 保存按 split 分的错误案例
    for split_name, cases in wrong_cases_by_split.items():
        text_split = [extractor.convert_case_to_text(c) for c in cases]
        extractor.save_text_cases(
            text_split,
            os.path.join(output_dir, f"wrong_cases_{split_name}_{args.data_name}.json")
        )
    
    return {
        "text_cases": text_cases,
        "surprise_subset": surprise_subset,
        "surprise_metrics": surprise_metrics,
        "wrong_cases_by_split": wrong_cases_by_split,
    }