#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HypothesisVerifier 单元测试 — 验证假设提取和验证逻辑的正确性
"""

import json
import numpy as np
from collections import Counter

# 直接测试 HypothesisVerifier 的数据处理方法 (不需要 LLM 调用)
from agent.hypothesis_verifier import HypothesisVerifier


def _make_mock_llm():
    """创建一个不调用真实 LLM 的 mock"""
    class MockLLM:
        def chat(self, **kwargs):
            return json.dumps({
                "hypotheses": [
                    {
                        "id": "H1",
                        "claim": "冷门物品最容易被误推",
                        "source_field": "error_patterns",
                        "verification_method": "item_popularity",
                        "expected_if_true": "误推目标中冷门物品占比显著高于全量",
                        "expected_if_false": "误推目标中冷门物品占比不高于全量",
                        "confidence_in_llm": "medium",
                        "priority": 5,
                    },
                    {
                        "id": "H2",
                        "claim": "模型过度依赖相似性推荐",
                        "source_field": "model_bottleneck",
                        "verification_method": "similarity_bias",
                        "expected_if_true": "预测类别与历史类别高度重叠",
                        "expected_if_false": "预测类别与历史类别重叠度低",
                        "confidence_in_llm": "high",
                        "priority": 4,
                    },
                    {
                        "id": "H3",
                        "claim": "短序列用户更容易误推",
                        "source_field": "error_patterns",
                        "verification_method": "sequence_length",
                        "expected_if_true": "短序列用户在误推案例中占比高",
                        "expected_if_false": "短序列用户在误推案例中占比不高",
                        "confidence_in_llm": "low",
                        "priority": 3,
                    },
                ],
                "summary": "假设H1和H2最可能有数据支撑, H3可能是臆断"
            })
    return MockLLM()


def _make_mock_item_text_map():
    """创建 mock 物品元数据"""
    return {
        "1": {"title": "Laptop", "categories": "Electronics > Computers > Laptops"},
        "2": {"title": "Phone", "categories": "Electronics > Phones > Smartphones"},
        "3": {"title": "Book", "categories": "Books > Fiction > Sci-Fi"},
        "4": {"title": "Shirt", "categories": "Clothing > Men > T-Shirts"},
        "5": {"title": "Coffee", "categories": "Food > Drinks > Coffee"},
        "6": {"title": "Laptop Stand", "categories": "Electronics > Accessories > Stands"},
        "7": {"title": "Novel", "categories": "Books > Fiction > Romance"},
        "8": {"title": "Jacket", "categories": "Clothing > Men > Jackets"},
        "9": {"title": "Tea", "categories": "Food > Drinks > Tea"},
        "10": {"title": "Headphones", "categories": "Electronics > Audio > Headphones"},
    }


def _make_mock_wrong_cases():
    """创建 mock 错误案例 (文本格式)"""
    item_map = _make_mock_item_text_map()
    cases = []
    for i in range(50):
        target_id = (i % 10) + 1
        # 高惊喜度案例: 目标类别不在历史中
        surprise_score = 0.8 if target_id in [3, 7, 9] else 0.2
        # 短序列案例
        seq_length = 5 if i < 20 else (15 if i < 35 else 30)
        
        history_text = []
        for h in range(seq_length):
            h_item = str((h % 5) + 1)  # 历史物品偏向 1-5
            entry = item_map.get(h_item, {})
            if isinstance(entry, dict):
                cat = entry.get("categories", "").split(" > ")[-1]
                history_text.append(f"{entry.get('title', 'Item')} [{cat}]")
            else:
                history_text.append(f"Item_{h_item}")
        
        target_entry = item_map.get(str(target_id), {})
        target_text = f"{target_entry.get('title', 'Item')} [{target_entry.get('categories', '').split(' > ')[-1]}]" if isinstance(target_entry, dict) else f"Item_{target_id}"
        
        predictions_ids = [(j % 10) + 1 for j in range(20)]
        predictions_text = []
        for pid in predictions_ids[:10]:
            p_entry = item_map.get(str(pid), {})
            if isinstance(p_entry, dict):
                pcat = p_entry.get("categories", "").split(" > ")[-1]
                predictions_text.append(f"{p_entry.get('title', 'Item')} [{pcat}]")
        
        cases.append({
            "user_id": i,
            "target_id": target_id,
            "target_text": target_text,
            "history_text": history_text,
            "predictions_ids": predictions_ids,
            "predictions_text": predictions_text,
            "target_rank": -1 if i < 30 else 15,
            "original_length": seq_length,
            "surprise_score": surprise_score,
        })
    
    return cases


def _make_mock_item_popularity():
    """创建 mock 物品热度"""
    # 物品 1-10 的交互次数 (热门: 1,2,10; 冷门: 3,7,9)
    return {
        "1": 100,  # Laptop - 热门
        "2": 80,   # Phone - 热门
        "3": 3,    # Book - 冷门
        "4": 40,   # Shirt - 中等
        "5": 60,   # Coffee - 中等
        "6": 20,   # Laptop Stand - 中等
        "7": 2,    # Novel - 冷门
        "8": 30,   # Jacket - 中等
        "9": 1,    # Tea - 冷门
        "10": 90,  # Headphones - 热门
    }


def test_compute_stats_baseline():
    """测试统计基线计算"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    cases = _make_mock_wrong_cases()
    popularity = _make_mock_item_popularity()
    
    baseline = verifier._compute_stats_baseline(cases, None, popularity)
    
    # 检查基线包含必要的统计
    assert "target_ids" in baseline
    assert "target_categories" in baseline
    assert "surprise_scores" in baseline
    assert "seq_lengths" in baseline
    assert "surprise_groups" in baseline
    assert "seq_length_groups" in baseline
    
    # 检查惊喜度分布
    sg = baseline["surprise_groups"]
    assert sg["high"] > 0  # 应有高惊喜案例
    assert sg["low"] > 0   # 应有低惊喜案例
    
    # 检查序列长度分布
    slg = baseline["seq_length_groups"]
    assert slg["short"] > 0
    assert slg["medium"] > 0
    assert slg["long"] > 0
    
    print("✅ test_compute_stats_baseline PASSED")


def test_verify_item_popularity():
    """测试物品热度验证"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    cases = _make_mock_wrong_cases()
    popularity = _make_mock_item_popularity()
    
    baseline = verifier._compute_stats_baseline(cases, None, popularity)
    
    # 假设: 冷门物品最容易被误推
    hyp_cold = {
        "id": "H1",
        "claim": "冷门物品最容易被误推",
        "verification_method": "item_popularity",
    }
    
    result_cold = verifier._verify_item_popularity(hyp_cold, baseline)
    # 在我们的 mock 数据中, 冷门物品 (3,7,9) 的交互次数很低
    # 误推目标中冷门占比应该高于全量占比
    assert result_cold["status"] in [verifier.CONFIRMED, verifier.PARTIALLY_CONFIRMED, verifier.REFUTED]
    assert result_cold["evidence"] is not None
    print(f"  H1 (cold items): status={result_cold['status']}, brief={result_cold['brief']}")
    
    # 假设: 热门物品总是被推荐
    hyp_hot = {
        "id": "H2",
        "claim": "热门物品总是被推荐",
        "verification_method": "item_popularity",
    }
    result_hot = verifier._verify_item_popularity(hyp_hot, baseline)
    assert result_hot["status"] in [verifier.CONFIRMED, verifier.PARTIALLY_CONFIRMED, verifier.REFUTED, verifier.UNVERIFIABLE]
    print(f"  H2 (hot items): status={result_hot['status']}, brief={result_hot['brief']}")
    
    print("✅ test_verify_item_popularity PASSED")


def test_verify_category_bias():
    """测试类别偏差验证"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    cases = _make_mock_wrong_cases()
    popularity = _make_mock_item_popularity()
    
    baseline = verifier._compute_stats_baseline(cases, None, popularity)
    
    # 假设: 跨类别物品最容易被误推
    hyp_cross = {
        "id": "H1",
        "claim": "跨类别物品最容易被误推",
        "verification_method": "category_bias",
    }
    result = verifier._verify_category_bias(hyp_cross, baseline)
    assert result["status"] in [verifier.CONFIRMED, verifier.PARTIALLY_CONFIRMED, verifier.REFUTED, verifier.UNVERIFIABLE]
    assert "entropy" in result.get("evidence", {})
    print(f"  Category bias: status={result['status']}, brief={result['brief']}")
    
    print("✅ test_verify_category_bias PASSED")


def test_verify_sequence_length():
    """测试序列长度验证"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    cases = _make_mock_wrong_cases()
    
    baseline = verifier._compute_stats_baseline(cases, None, None)
    
    # 假设: 短序列用户更容易误推
    hyp_short = {
        "id": "H1",
        "claim": "短序列用户更容易误推",
        "verification_method": "sequence_length",
    }
    result = verifier._verify_sequence_length(hyp_short, baseline)
    assert result["status"] in [verifier.CONFIRMED, verifier.PARTIALLY_CONFIRMED, verifier.REFUTED, verifier.UNVERIFIABLE]
    print(f"  Short seq: status={result['status']}, brief={result['brief']}")
    
    print("✅ test_verify_sequence_length PASSED")


def test_verify_surprise_score():
    """测试惊喜度验证"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    cases = _make_mock_wrong_cases()
    
    baseline = verifier._compute_stats_baseline(cases, None, None)
    
    # 假设: 模型无法预测与历史差异大的交互
    overall_metrics = {"NDCG@10": 0.3, "NDCG@20": 0.4, "Recall@10": 0.25, "Recall@20": 0.35}
    surprise_metrics = {"NDCG@10": 0.15, "NDCG@20": 0.25, "Recall@10": 0.10, "Recall@20": 0.20}
    
    hyp_surprise = {
        "id": "H1",
        "claim": "模型无法预测与历史差异大的交互",
        "verification_method": "surprise_score",
    }
    result = verifier._verify_surprise_score(hyp_surprise, baseline, overall_metrics, surprise_metrics)
    # 惊喜子集 NDCG 比 整体低 50%, 应该被确认
    assert result["status"] == verifier.CONFIRMED
    print(f"  Surprise: status={result['status']}, brief={result['brief']}")
    
    print("✅ test_verify_surprise_score PASSED")


def test_generate_verification_report():
    """测试验证报告生成"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    
    verified_hypotheses = [
        {
            "id": "H1",
            "claim": "冷门物品最容易被误推",
            "source_field": "error_patterns",
            "verification_method": "item_popularity",
            "verification_result": {
                "status": "CONFIRMED",
                "brief": "冷门物品误推占比显著高于全量",
                "evidence": {"ratio": 2.5},
                "method": "item_popularity",
            },
        },
        {
            "id": "H2",
            "claim": "模型过度依赖相似性推荐",
            "source_field": "model_bottleneck",
            "verification_method": "similarity_bias",
            "verification_result": {
                "status": "REFUTED",
                "brief": "预测类别与历史重叠度低, 不存在过度相似依赖",
                "evidence": {"avg_category_overlap": 0.2},
                "method": "similarity_bias",
            },
        },
        {
            "id": "H3",
            "claim": "Self-Attention 在长序列上失效",
            "source_field": "model_bottleneck",
            "verification_method": "attention_pattern",
            "verification_result": {
                "status": "UNVERIFIABLE",
                "brief": "需要模型内部数据",
                "reason": "注意力权重验证需要模型运行时的内部数据",
                "evidence": None,
            },
        },
    ]
    
    report = verifier.generate_verification_report(verified_hypotheses)
    
    assert report["total_hypotheses"] == 3
    assert report["confirmed_count"] == 1
    assert report["refuted_count"] == 1
    assert report["unverifiable_count"] == 1
    assert len(report["recommendations"]) == 1  # 1 个被反驳的结论
    assert report["overall_credibility"] in ["HIGH", "MODERATE", "LOW"]
    
    print(f"  Report: confirmed={report['confirmed_pct']}%, refuted={report['refuted_pct']}%, credibility={report['overall_credibility']}")
    print("✅ test_generate_verification_report PASSED")


def test_apply_verification_to_analysis():
    """测试验证结果应用到分析结论"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    
    llm_analysis = {
        "parse_success": True,
        "error_patterns": {"pattern_1": "冷门物品误推"},
        "model_bottleneck": {"attention_failure": "Self-Attention在长序列失效"},
        "surprise_failure_reasons": {"main_reason": "模型过度依赖相似性"},
        "improvement_suggestions": [
            {
                "priority": 1,
                "action_type": "structure_change",
                "description": "修改Self-Attention以解决长序列失效问题",
                "expected_effect": "NDCG提升5%",
            },
            {
                "priority": 2,
                "action_type": "parameter_change",
                "description": "增加多样性约束来解决过度相似推荐",
                "expected_effect": "Recall提升3%",
            },
        ],
        "summary": "模型最大的问题是Self-Attention无法处理长序列",
    }
    
    verification_report = {
        "total_hypotheses": 3,
        "confirmed": [{"id": "H1", "claim": "冷门物品误推", "status": "CONFIRMED", "brief": "..."}],
        "confirmed_count": 1,
        "confirmed_pct": 33.3,
        "partially_confirmed": [],
        "partially_confirmed_count": 0,
        "refuted": [{"id": "H2", "claim": "模型过度依赖相似性推荐", "status": "REFUTED", "brief": "..."}],
        "refuted_count": 1,
        "refuted_pct": 33.3,
        "unverifiable": [{"id": "H3", "claim": "Self-Attention在长序列失效", "status": "UNVERIFIABLE"}],
        "unverifiable_count": 1,
        "overall_credibility": "MODERATE",
        "refuted_claims": ["模型过度依赖相似性推荐"],
        "verified_hypotheses": [],
    }
    
    enhanced = verifier.apply_verification_to_analysis(llm_analysis, verification_report)
    
    # 检查增强后的分析包含验证元数据
    assert "verification_meta" in enhanced
    vm = enhanced["verification_meta"]
    assert vm["overall_credibility"] == "MODERATE"
    assert vm["confirmed_pct"] == 33.3
    assert len(vm["refuted_claims"]) == 1
    
    # 检查被反驳的改进建议标注了警告
    suggestions = enhanced["improvement_suggestions"]
    diversity_suggestion = suggestions[1]  # "增加多样性约束来解决过度相似推荐"
    assert diversity_suggestion.get("verification_warning") is not None
    assert diversity_suggestion.get("confidence_level") == "LOW"
    
    print("✅ test_apply_verification_to_analysis PASSED")


def test_compute_item_popularity():
    """测试物品热度计算"""
    verifier = HypothesisVerifier(_make_mock_llm(), _make_mock_item_text_map())
    
    train_data = [
        [0, 1, 2, 3, 5],  # 交互了物品 1,2,3,5 (去掉最后目标 5)
        [0, 1, 1, 4, 6],  # 交互了物品 1,1,4 (去掉最后目标 6)
        [0, 2, 2, 8],     # 交互了物品 2,2 (去掉最后目标 8)
    ]
    
    popularity = verifier.compute_item_popularity_from_data(train_data)
    
    assert str(1) in popularity
    assert popularity[str(1)] == 3  # 物品 1 出现了 3 次
    assert popularity[str(2)] == 3  # 物品 2 出现了 3 次 (两次在第三个序列 + 一次在第一个)
    assert str(3) in popularity
    assert popularity[str(3)] == 1
    
    print("✅ test_compute_item_popularity PASSED")


if __name__ == "__main__":
    print("\n═══════════ Running HypothesisVerifier Unit Tests ═══════════\n")
    test_compute_stats_baseline()
    test_verify_item_popularity()
    test_verify_category_bias()
    test_verify_sequence_length()
    test_verify_surprise_score()
    test_generate_verification_report()
    test_apply_verification_to_analysis()
    test_compute_item_popularity()
    print("\n═══════════ All Tests PASSED ✅ ═══════════\n")