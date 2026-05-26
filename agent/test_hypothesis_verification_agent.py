#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HypothesisVerificationAgent 单元测试 — 验证自主验证 Agent 的核心功能

测试重点:
1. DataInventory 数据发现和格式化
2. 假设提取 (新版 — 不限制验证类型)
3. 代码生成和清理
4. 脚本执行和修正循环
5. Fallback 验证 (Agent 失败时回退到旧版)
6. 验证报告生成 (与旧版接口一致)
7. apply_verification_to_analysis (与旧版接口一致)
"""

import json
import os
import sys
import tempfile
import shutil

# 直接测试 Agent 的数据处理方法 (不需要 LLM 调用)
from agent.hypothesis_verification_agent import (
    HypothesisVerificationAgent, DataInventory,
    HYPOTHESIS_EXTRACTION_PROMPT_V2,
    VERIFICATION_PLAN_PROMPT,
    VERIFICATION_CODE_PROMPT,
    VERIFICATION_CODE_FIX_PROMPT,
    RESULT_ANALYSIS_PROMPT,
)


def _make_mock_llm():
    """创建一个不调用真实 LLM 的 mock"""
    class MockLLM:
        def __init__(self):
            self._call_count = 0
            self._call_types = []
        
        def chat(self, messages, **kwargs):
            self._call_count += 1
            # 根据调用上下文返回不同响应
            system_msg = messages[0].get("content", "")
            user_msg = messages[1].get("content", "")
            
            if "从分析结论中提取" in system_msg or "可验证的假设" in system_msg:
                self._call_types.append("extract")
                return json.dumps({
                    "hypotheses": [
                        {
                            "id": "H1",
                            "claim": "冷门物品最容易被误推",
                            "source_field": "error_patterns",
                            "verification_thought": "对比误推目标中冷门物品占比与全量占比, 计算比率",
                            "data_needed": ["物品热度分布", "误推案例物品ID"],
                            "expected_if_true": "误推目标中冷门物品占比 > 30%",
                            "expected_if_false": "误推目标中冷门物品占比 < 10%",
                            "confidence_in_llm": "medium",
                            "priority": 5,
                        },
                        {
                            "id": "H2",
                            "claim": "模型倾向于推荐近期交互的物品",
                            "source_field": "model_bottleneck",
                            "verification_thought": "计算预测物品与最近交互物品的时间距离分布",
                            "data_needed": ["用户交互序列", "预测物品列表"],
                            "expected_if_true": "预测物品中近5步交互的物品占比 > 60%",
                            "expected_if_false": "预测物品中近5步交互的物品占比 < 30%",
                            "confidence_in_llm": "low",
                            "priority": 3,
                        },
                    ],
                    "summary": "H1最可能有数据支撑"
                })
            
            elif "设计验证方案" in system_msg or "verification_plan" in user_msg[:200]:
                self._call_types.append("plan")
                return json.dumps({
                    "hypothesis_id": "H1",
                    "verification_plan": {
                        "method_name": "item_popularity_comparison",
                        "method_description": "对比误推目标中冷门物品占比与全量占比",
                        "data_sources": ["item_popularity", "wrong_text_cases"],
                        "analysis_steps": [
                            "Step 1: 从误推案例中提取所有目标物品ID",
                            "Step 2: 根据 item_popularity 判断每个目标物品是冷门/中等/热门",
                            "Step 3: 计算误推目标中冷门占比",
                            "Step 4: 计算全量物品中冷门占比",
                            "Step 5: 比较两个比率",
                        ],
                        "statistical_method": "proportion_comparison",
                        "code_outline": "load data → classify items → compute ratios → compare",
                        "confirm_criteria": "冷门占比比率 > 1.5",
                        "refute_criteria": "冷门占比比率 < 1.0",
                        "partial_criteria": "冷门占比比率在 1.0-1.5 之间",
                        "expected_output_format": "JSON with ratio, cold_pct_target, cold_pct_overall",
                    }
                })
            
            elif "数据科学家" in system_msg or "验证脚本" in system_msg:
                self._call_types.append("code")
                # 返回一个简单的验证脚本
                return '''
import json
import numpy as np

# 加载 item_popularity
popularity = _preloaded.get("item_popularity", {})
wrong_cases = _preloaded.get("wrong_text_cases_sample", [])

# 计算误推目标的热度分布
cold_count = 0
medium_count = 0
hot_count = 0
total_targets = 0

for case in wrong_cases:
    tid = str(case.get("target_id", ""))
    if tid in popularity:
        pop = popularity[tid]
        if pop < 5:
            cold_count += 1
        elif pop < 50:
            medium_count += 1
        else:
            hot_count += 1
        total_targets += 1

# 全量物品热度分布
all_cold = sum(1 for v in popularity.values() if v < 5)
all_medium = sum(1 for v in popularity.values() if 5 <= v < 50)
all_hot = sum(1 for v in popularity.values() if v >= 50)
total_all = len(popularity)

# 计算比率
cold_pct_target = cold_count / total_targets * 100 if total_targets > 0 else 0
cold_pct_overall = all_cold / total_all * 100 if total_all > 0 else 0
ratio = cold_pct_target / cold_pct_overall if cold_pct_overall > 0 else 0

result = {
    "hypothesis_id": "H1",
    "statistics": {
        "cold_pct_target": cold_pct_target,
        "cold_pct_overall": cold_pct_overall,
        "ratio": ratio,
        "cold_count_target": cold_count,
        "total_targets": total_targets,
    },
    "interpretation": f"误推目标中冷门物品占{cold_pct_target:.1f}%, 全量中冷门占{cold_pct_overall:.1f}%, 比率={ratio:.2f}",
}

save_result(result)
'''
            
            elif "修正代码" in system_msg or "修正" in system_msg:
                self._call_types.append("fix")
                return '''
import json
import numpy as np

popularity = _preloaded.get("item_popularity", {})
wrong_cases_sample = _preloaded.get("wrong_text_cases_sample", [])

cold_count = 0
total_targets = 0

for case in wrong_cases_sample:
    try:
        tid = str(case.get("target_id", ""))
        if tid in popularity:
            pop = popularity[tid]
            if pop < 5:
                cold_count += 1
            total_targets += 1
    except Exception:
        continue

all_cold = sum(1 for v in popularity.values() if isinstance(v, (int, float)) and v < 5)
total_all = len(popularity)

cold_pct_target = cold_count / total_targets * 100 if total_targets > 0 else 0
cold_pct_overall = all_cold / total_all * 100 if total_all > 0 else 0
ratio = cold_pct_target / cold_pct_overall if cold_pct_overall > 0 else 0

result = {
    "hypothesis_id": "H1",
    "statistics": {
        "cold_pct_target": cold_pct_target,
        "cold_pct_overall": cold_pct_overall,
        "ratio": ratio,
    },
    "interpretation": f"比率={ratio:.2f}",
}

save_result(result)
'''
            
            elif "判断假设" in system_msg or "数据结果与期望" in user_msg[:200]:
                self._call_types.append("analyze")
                return json.dumps({
                    "hypothesis_id": "H1",
                    "status": "CONFIRMED",
                    "brief": "冷门物品误推占比显著高于全量",
                    "detailed_reasoning": "误推目标中冷门物品占比远高于全量占比, 比率>1.5",
                    "evidence_summary": {
                        "key_statistic": "ratio=2.5",
                        "comparison": "冷门占比误推30% vs 全量10%",
                        "confidence": 0.85,
                    },
                    "limitations": "仅基于50个误推案例样本",
                })
            
            else:
                self._call_types.append("unknown")
                return json.dumps({"error": "Unknown request type"})
    
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
    """创建 mock 错误案例"""
    item_map = _make_mock_item_text_map()
    cases = []
    for i in range(50):
        target_id = (i % 10) + 1
        surprise_score = 0.8 if target_id in [3, 7, 9] else 0.2
        seq_length = 5 if i < 20 else (15 if i < 35 else 30)
        
        history_text = []
        for h in range(seq_length):
            h_item = str((h % 5) + 1)
            entry = item_map.get(h_item, {})
            if isinstance(entry, dict):
                cat = entry.get("categories", "").split(" > ")[-1]
                history_text.append(f"{entry.get('title', 'Item')} [{cat}]")
        
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
    return {
        "1": 100, "2": 80, "3": 3, "4": 40, "5": 60,
        "6": 20, "7": 2, "8": 30, "9": 1, "10": 90,
    }


# ════════════════════════════════════════
# 测试 DataInventory
# ════════════════════════════════════════

def test_data_inventory():
    """测试数据发现和盘点"""
    # 使用临时目录模拟项目结构
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建数据目录
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 创建数据文件
        with open(os.path.join(data_dir, "Beauty_train.txt"), 'w') as f:
            f.write("1 2 3 4 5\n6 7 8\n")
        
        with open(os.path.join(data_dir, "id_meta_data.json"), 'w') as f:
            json.dump({"1": {"title": "Item1", "categories": "Cat1"}}, f)
        
        # 创建日志目录和已计算统计量
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        with open(os.path.join(log_dir, "item_popularity_Beauty.json"), 'w') as f:
            json.dump({"1": 100, "2": 50}, f)
        
        # 执行盘点
        inventory = DataInventory(project_root=tmpdir, data_dir=data_dir, log_dir=log_dir)
        discovered = inventory.discover()
        
        # 检查发现的数据文件
        assert len(discovered["data_files"]) >= 2  # train.txt + meta_data.json
        
        # 检查发现了 item_popularity
        assert len(discovered["computed_stats"]) >= 1
        assert discovered["computed_stats"][0]["name"] == "item_popularity"
        
        # 格式化为 prompt 文本
        prompt_text = inventory.format_inventory_for_prompt()
        assert len(prompt_text) > 0
        assert "Beauty_train.txt" in prompt_text
        assert "item_popularity" in prompt_text
        
        print(f"  Discovered: {len(discovered['data_files'])} data files, "
              f"{len(discovered['computed_stats'])} computed stats")
        print("✅ test_data_inventory PASSED")


def test_data_inventory_load_data():
    """测试数据加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建 item_popularity
        pop_data = {"1": 100, "2": 50, "3": 3}
        with open(os.path.join(log_dir, "item_popularity_Beauty.json"), 'w') as f:
            json.dump(pop_data, f)
        
        # 创建元数据
        meta_data = {"1": {"title": "Laptop", "categories": "Electronics"}}
        with open(os.path.join(data_dir, "id_meta_data.json"), 'w') as f:
            json.dump(meta_data, f)
        
        inventory = DataInventory(project_root=tmpdir, data_dir=data_dir, log_dir=log_dir)
        
        # 测试加载热度数据
        loaded = inventory.load_data_for_verification(["物品热度分布"])
        assert "item_popularity" in loaded
        assert loaded["item_popularity"]["1"] == 100
        
        # 测试加载元数据
        loaded2 = inventory.load_data_for_verification(["物品元数据", "类别信息"])
        assert "item_metadata" in loaded2
        
        # 测试格式化已加载摘要
        summary = inventory.format_loaded_data_summary(loaded)
        assert "item_popularity" in summary
        
        print("✅ test_data_inventory_load_data PASSED")


# ════════════════════════════════════════
# 测试 Agent 核心流程
# ════════════════════════════════════════

def test_agent_initialization():
    """测试 Agent 初始化"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            item_text_map=_make_mock_item_text_map(),
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        # 验证初始化
        assert agent.llm is mock_llm
        assert agent.CONFIRMED == "CONFIRMED"
        assert agent.REFUTED == "REFUTED"
        assert agent.MAX_CODE_FIX_ROUNDS == 3
        
        # 验证 fallback verifier 可用
        assert agent._fallback_verifier is not None
        
        print("✅ test_agent_initialization PASSED")


def test_extract_hypotheses_v2():
    """测试新版假设提取 (不限制验证类型)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            item_text_map=_make_mock_item_text_map(),
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        llm_analysis = {
            "parse_success": True,
            "error_patterns": {"pattern_1": "冷门物品误推"},
            "model_bottleneck": {"issue": "模型倾向于推荐近期交互的物品"},
            "summary": "模型最大的问题是对冷门物品推荐效果差",
        }
        
        hypotheses = agent.extract_hypotheses(llm_analysis)
        
        assert hypotheses is not None
        assert len(hypotheses) == 2
        
        # 新版假设应该有 verification_thought (而非固定 verification_method)
        h1 = hypotheses[0]
        assert h1["id"] == "H1"
        assert "verification_thought" in h1  # 新版字段
        assert "data_needed" in h1  # 新版字段
        
        # 验证 LLM 被调用
        assert mock_llm._call_count == 1
        assert mock_llm._call_types[0] == "extract"
        
        print(f"  Extracted {len(hypotheses)} hypotheses with verification_thought")
        print("✅ test_extract_hypotheses_v2 PASSED")


def test_prepare_preloaded_data():
    """测试数据预加载"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        wrong_cases = _make_mock_wrong_cases()
        popularity = _make_mock_item_popularity()
        metrics = {"NDCG@10": 0.3, "Recall@10": 0.25}
        surprise_metrics = {"NDCG@10": 0.15, "Recall@10": 0.10}
        
        preloaded = agent._prepare_preloaded_data(
            wrong_cases, None, popularity, metrics, surprise_metrics
        )
        
        assert "wrong_text_cases" in preloaded
        assert "item_popularity" in preloaded
        assert "overall_metrics" in preloaded
        assert "surprise_metrics" in preloaded
        
        # 验证数据格式
        assert len(preloaded["wrong_text_cases"]) == 50
        assert preloaded["item_popularity"]["1"] == 100
        
        print("✅ test_prepare_preloaded_data PASSED")


def test_clean_code_response():
    """测试代码清理"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        # 测试 markdown code block 提取
        md_code = """这是解释文字
```python
import json
result = {"a": 1}
save_result(result)
```
这是结尾解释"""
        
        cleaned = agent._clean_code_response(md_code)
        assert "import json" in cleaned
        assert "save_result" in cleaned
        assert "解释文字" not in cleaned
        
        # 测试纯代码 (无 markdown)
        pure_code = """import json
import numpy as np

result = {"key": "value"}
save_result(result)"""
        
        cleaned2 = agent._clean_code_response(pure_code)
        assert "import json" in cleaned2
        
        print("✅ test_clean_code_response PASSED")


def test_infer_verification_method():
    """测试从自由描述推断旧版验证方法"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        # 测试关键词映射
        assert agent._infer_verification_method("冷门物品误推", "对比热度分布") == "item_popularity"
        assert agent._infer_verification_method("类别偏差问题", "跨类别分析") == "category_bias"
        assert agent._infer_verification_method("短序列用户", "序列长度效应") == "sequence_length"
        assert agent._infer_verification_method("相似性依赖", "余弦相似度分析") == "similarity_bias"
        assert agent._infer_verification_method("惊喜度问题", "高惊喜vs低惊喜对比") == "surprise_score"
        assert agent._infer_verification_method("位置编码失效", "注意力模式分析") == "custom"
        
        print("✅ test_infer_verification_method PASSED")


def test_generate_verification_report():
    """测试验证报告生成 (与旧版接口一致)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        verified_hypotheses = [
            {
                "id": "H1",
                "claim": "冷门物品最容易被误推",
                "source_field": "error_patterns",
                "verification_result": {
                    "status": "CONFIRMED",
                    "brief": "冷门物品误推占比显著高于全量",
                    "evidence": {"ratio": 2.5},
                    "method": "agent_autonomous",
                },
            },
            {
                "id": "H2",
                "claim": "模型过度依赖相似性推荐",
                "source_field": "model_bottleneck",
                "verification_result": {
                    "status": "REFUTED",
                    "brief": "不存在过度相似依赖",
                    "evidence": {"avg_category_overlap": 0.2},
                    "method": "agent_autonomous",
                },
            },
            {
                "id": "H3",
                "claim": "位置编码无法捕捉时间衰减",
                "source_field": "model_bottleneck",
                "verification_result": {
                    "status": "UNVERIFIABLE",
                    "brief": "需要模型内部数据",
                    "evidence": None,
                    "method": "agent_autonomous",
                },
            },
        ]
        
        report = agent.generate_verification_report(verified_hypotheses)
        
        # 检查报告结构 (与旧版一致)
        assert report["total_hypotheses"] == 3
        assert report["confirmed_count"] == 1
        assert report["refuted_count"] == 1
        assert report["unverifiable_count"] == 1
        assert report["overall_credibility"] in ["HIGH", "MODERATE", "LOW"]
        assert len(report["recommendations"]) == 1
        
        # 检查新版标记
        assert report["verification_agent_used"] == True
        
        print(f"  Report: confirmed={report['confirmed_pct']}%, "
              f"refuted={report['refuted_pct']}%, credibility={report['overall_credibility']}")
        print("✅ test_generate_verification_report PASSED")


def test_apply_verification_to_analysis():
    """测试将验证结果应用到分析 (与旧版接口一致)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        llm_analysis = {
            "parse_success": True,
            "error_patterns": {"pattern_1": "冷门物品误推"},
            "improvement_suggestions": [
                {
                    "priority": 1,
                    "action_type": "parameter_change",
                    "description": "增加多样性约束来解决过度相似推荐",
                    "expected_effect": "Recall提升3%",
                },
            ],
            "summary": "模型最大的问题是过度依赖相似性",
        }
        
        verification_report = {
            "total_hypotheses": 2,
            "confirmed": [{"id": "H1", "claim": "冷门物品误推", "status": "CONFIRMED"}],
            "confirmed_count": 1,
            "confirmed_pct": 50.0,
            "partially_confirmed": [],
            "refuted": [{"id": "H2", "claim": "模型过度依赖相似性推荐", "status": "REFUTED"}],
            "refuted_count": 1,
            "refuted_pct": 50.0,
            "unverifiable": [],
            "unverifiable_count": 0,
            "overall_credibility": "MODERATE",
            "refuted_claims": ["模型过度依赖相似性推荐"],
            "verified_hypotheses": [],
        }
        
        enhanced = agent.apply_verification_to_analysis(llm_analysis, verification_report)
        
        # 检查增强后的分析包含验证元数据
        assert "verification_meta" in enhanced
        vm = enhanced["verification_meta"]
        assert vm["overall_credibility"] == "MODERATE"
        assert vm["confirmed_pct"] == 50.0
        
        # 检查被反驳的改进建议标注了警告
        suggestions = enhanced["improvement_suggestions"]
        diversity_suggestion = suggestions[0]
        assert diversity_suggestion.get("verification_warning") is not None
        assert diversity_suggestion.get("confidence_level") == "LOW"
        
        print("✅ test_apply_verification_to_analysis PASSED")


def test_save_verification_report():
    """测试保存验证报告"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        report = {
            "total_hypotheses": 2,
            "confirmed_count": 1,
            "refuted_count": 1,
            "verification_agent_used": True,
        }
        
        output_path = os.path.join(tmpdir, "verification_report.json")
        agent.save_verification_report(report, output_path)
        
        # 检查文件存在且内容正确
        assert os.path.exists(output_path)
        with open(output_path, 'r', encoding='utf-8') as f:
            saved = json.load(f)
        assert saved["verification_agent_used"] == True
        
        print("✅ test_save_verification_report PASSED")


def test_fallback_verification():
    """测试 fallback 验证 (Agent 失败时回退到旧版)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        wrong_cases = _make_mock_wrong_cases()
        popularity = _make_mock_item_popularity()
        
        # 测试 fallback 验证
        hyp_cold = {
            "id": "H1",
            "claim": "冷门物品最容易被误推",
            "verification_thought": "对比误推目标中冷门物品占比与全量占比",
        }
        
        result = agent._try_fallback_verification(
            hyp_cold, wrong_cases, None, popularity,
            {"NDCG@10": 0.3}, {"NDCG@10": 0.15}
        )
        
        # 应该返回有效结果 (不崩溃)
        assert result["status"] in ["CONFIRMED", "PARTIALLY_CONFIRMED", "REFUTED", "UNVERIFIABLE"]
        assert result["method"] == "fallback_fixed"
        assert "fallback_reason" in result
        
        print(f"  Fallback result: status={result['status']}, method={result['method']}")
        print("✅ test_fallback_verification PASSED")


def test_prompts_exist():
    """测试所有 Prompt 模板存在且包含必要的占位符"""
    # 检查 HYPOTHESIS_EXTRACTION_PROMPT_V2
    assert "{llm_analysis_json}" in HYPOTHESIS_EXTRACTION_PROMPT_V2
    assert "{data_inventory}" in HYPOTHESIS_EXTRACTION_PROMPT_V2
    assert "verification_thought" in HYPOTHESIS_EXTRACTION_PROMPT_V2
    assert "data_needed" in HYPOTHESIS_EXTRACTION_PROMPT_V2
    
    # 检查 VERIFICATION_PLAN_PROMPT
    assert "{hypothesis_id}" in VERIFICATION_PLAN_PROMPT
    assert "{hypothesis_claim}" in VERIFICATION_PLAN_PROMPT
    assert "{verification_thought}" in VERIFICATION_PLAN_PROMPT
    assert "{data_inventory}" in VERIFICATION_PLAN_PROMPT
    assert "{loaded_data_summary}" in VERIFICATION_PLAN_PROMPT
    
    # 检查 VERIFICATION_CODE_PROMPT
    assert "{hypothesis_claim}" in VERIFICATION_CODE_PROMPT
    assert "{verification_plan_json}" in VERIFICATION_CODE_PROMPT
    assert "{available_data_description}" in VERIFICATION_CODE_PROMPT
    assert "{output_file_path}" in VERIFICATION_CODE_PROMPT
    assert "{hypothesis_id}" in VERIFICATION_CODE_PROMPT
    
    # 检查 VERIFICATION_CODE_FIX_PROMPT
    assert "{original_code}" in VERIFICATION_CODE_FIX_PROMPT
    assert "{error_output}" in VERIFICATION_CODE_FIX_PROMPT
    assert "{output_file_path}" in VERIFICATION_CODE_FIX_PROMPT
    
    # 检查 RESULT_ANALYSIS_PROMPT
    assert "{hypothesis_id}" in RESULT_ANALYSIS_PROMPT
    assert "{hypothesis_claim}" in RESULT_ANALYSIS_PROMPT
    assert "{expected_if_true}" in RESULT_ANALYSIS_PROMPT
    assert "{expected_if_false}" in RESULT_ANALYSIS_PROMPT
    assert "{verification_plan_json}" in RESULT_ANALYSIS_PROMPT
    assert "{execution_result_json}" in RESULT_ANALYSIS_PROMPT
    
    print("✅ test_prompts_exist PASSED")


def test_format_available_data_for_code():
    """测试数据格式化为代码描述"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        verification_data = {
            "wrong_text_cases": _make_mock_wrong_cases(),
            "item_popularity": _make_mock_item_popularity(),
            "overall_metrics": {"NDCG@10": 0.3},
        }
        
        desc = agent._format_available_data_for_code(verification_data)
        
        assert "wrong_text_cases" in desc
        assert "item_popularity" in desc
        assert "overall_metrics" in desc
        assert "List" in desc or "长度" in desc
        
        print("✅ test_format_available_data_for_code PASSED")


def test_parse_json_from_response():
    """测试从 LLM 回复中解析 JSON"""
    with tempfile.TemporaryDirectory() as tmpdir:
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        # 测试 JSON code block
        response_with_block = """```json
{"hypothesis_id": "H1", "status": "CONFIRMED"}
```"""
        parsed = agent._parse_json_from_response(response_with_block)
        assert parsed is not None
        assert parsed["hypothesis_id"] == "H1"
        
        # 测试纯 JSON
        response_pure = '{"key": "value"}'
        parsed2 = agent._parse_json_from_response(response_pure)
        assert parsed2 is not None
        assert parsed2["key"] == "value"
        
        # 测试嵌入文本中的 JSON
        response_embedded = """Here is the result:
{
  "status": "REFUTED",
  "brief": "test"
}
End of analysis."""
        parsed3 = agent._parse_json_from_response(response_embedded)
        assert parsed3 is not None
        assert parsed3["status"] == "REFUTED"
        
        # 测试无效响应
        response_invalid = "This is not JSON at all"
        parsed4 = agent._parse_json_from_response(response_invalid)
        assert parsed4 is None
        
        print("✅ test_parse_json_from_response PASSED")


# ════════════════════════════════════════
# 测试全流程 (仅验证数据流和接口, 不需要真实 LLM)
# ════════════════════════════════════════

def test_full_flow_with_mock_llm():
    """测试完整 Agent 流程 (使用 mock LLM)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, "logs"), exist_ok=True)
        
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            item_text_map=_make_mock_item_text_map(),
            project_root=tmpdir,
            log_dir=os.path.join(tmpdir, "logs"),
        )
        
        wrong_cases = _make_mock_wrong_cases()
        popularity = _make_mock_item_popularity()
        metrics = {"NDCG@10": 0.3, "Recall@10": 0.25}
        surprise_metrics = {"NDCG@10": 0.15, "Recall@10": 0.10}
        
        # Step 1: 提取假设
        llm_analysis = {
            "parse_success": True,
            "error_patterns": {"pattern_1": "冷门物品误推"},
            "summary": "模型最大的问题是对冷门物品推荐效果差",
        }
        
        hypotheses = agent.extract_hypotheses(llm_analysis)
        assert hypotheses is not None
        assert len(hypotheses) >= 1
        
        # Step 2: 验证假设
        verified = agent.verify_hypotheses(
            hypotheses=hypotheses,
            wrong_text_cases=wrong_cases,
            all_wrong_cases=None,
            model_config=None,
            item_popularity=popularity,
            overall_metrics=metrics,
            surprise_metrics=surprise_metrics,
        )
        
        assert verified is not None
        assert len(verified) >= 1
        
        # Step 3: 生成报告
        report = agent.generate_verification_report(verified)
        assert report["total_hypotheses"] >= 1
        assert "verification_agent_used" in report
        assert report["verification_agent_used"] == True
        
        # Step 4: 保存报告
        report_path = os.path.join(tmpdir, "logs", "verification_test.json")
        agent.save_verification_report(report, report_path)
        assert os.path.exists(report_path)
        
        # 检查 mock LLM 被多次调用
        assert mock_llm._call_count >= 3  # extract + plan + code at minimum
        
        print(f"  Full flow: {mock_llm._call_count} LLM calls, "
              f"call types: {mock_llm._call_types}")
        print(f"  Report: {report['total_hypotheses']} hypotheses, "
              f"credibility={report['overall_credibility']}")
        print("✅ test_full_flow_with_mock_llm PASSED")


def test_interface_compatibility():
    """测试与旧版 HypothesisVerifier 的接口兼容性"""
    # 确保新版 Agent 的所有公共方法与旧版一致
    from agent.hypothesis_verifier import HypothesisVerifier
    
    old_methods = set(dir(HypothesisVerifier))
    new_methods = set(dir(HypothesisVerificationAgent))
    
    # 新版必须实现的关键接口方法
    required_methods = [
        "extract_hypotheses",
        "verify_hypotheses",
        "generate_verification_report",
        "apply_verification_to_analysis",
        "save_verification_report",
        "compute_item_popularity_from_data",
    ]
    
    for method in required_methods:
        assert method in new_methods, f"Missing method: {method}"
        assert callable(getattr(HypothesisVerificationAgent, method)), f"Not callable: {method}"
    
    # 验证状态常量一致
    assert HypothesisVerificationAgent.CONFIRMED == HypothesisVerifier.CONFIRMED
    assert HypothesisVerificationAgent.PARTIALLY_CONFIRMED == HypothesisVerifier.PARTIALLY_CONFIRMED
    assert HypothesisVerificationAgent.REFUTED == HypothesisVerifier.REFUTED
    assert HypothesisVerificationAgent.UNVERIFIABLE == HypothesisVerifier.UNVERIFIABLE
    
    print("✅ test_interface_compatibility PASSED")


if __name__ == "__main__":
    print("\n═══════════ Running HypothesisVerificationAgent Unit Tests ═══════════\n")
    
    test_data_inventory()
    test_data_inventory_load_data()
    test_agent_initialization()
    test_extract_hypotheses_v2()
    test_prepare_preloaded_data()
    test_clean_code_response()
    test_infer_verification_method()
    test_generate_verification_report()
    test_apply_verification_to_analysis()
    test_save_verification_report()
    test_fallback_verification()
    test_prompts_exist()
    test_format_available_data_for_code()
    test_parse_json_from_response()
    test_full_flow_with_mock_llm()
    test_interface_compatibility()
    
    print("\n═══════════ All Tests PASSED ✅ ═══════════\n")