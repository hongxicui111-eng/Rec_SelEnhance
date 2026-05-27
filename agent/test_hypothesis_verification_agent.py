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
    HypothesisVerificationAgent, DataInventory, DataComputationEngine,
    ModelProbingEngine,
    HYPOTHESIS_EXTRACTION_PROMPT_V2,
    VERIFICATION_PLAN_PROMPT,
    VERIFICATION_CODE_PROMPT,
    VERIFICATION_CODE_FIX_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    DATA_COMPUTATION_PROMPT,
    MODEL_PROBING_ANALYSIS_PROMPT,
    MODEL_PROBING_SCRIPT_PROMPT,
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


# ════════════════════════════════════════
# 测试 DataComputationEngine
# ════════════════════════════════════════

def test_data_computation_engine_init():
    """测试 DataComputationEngine 初始化"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            data_dir=data_dir,
            log_dir=log_dir,
        )
        
        assert engine.project_root == tmpdir
        assert engine.llm is None  # no LLM by default
        assert engine.MAX_COMPUTE_TIMEOUT == 120
        assert engine.MAX_COMPUTE_FIX_ROUNDS == 2
        
        # Check cache dir exists
        assert os.path.exists(engine.cache_dir)
        
        print("✅ test_data_computation_engine_init PASSED")


def test_compute_item_interaction_freq():
    """测试内置方法: 物品交互频率计算"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建训练数据文件
        train_file = os.path.join(data_dir, "Beauty_train.txt")
        with open(train_file, 'w') as f:
            f.write("1 2 3 4 5\n6 7 8\n1 2 3\n")
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            data_dir=data_dir,
            log_dir=log_dir,
        )
        
        # 计算物品交互频率
        preloaded = {"train_data_path": train_file}
        result = engine._compute_item_interaction_freq(preloaded)
        
        assert result is not None
        assert "freq_dict" in result
        assert "statistics" in result
        
        # 检查统计值
        freq = result["freq_dict"]
        assert freq["1"] == 2  # item 1 appears in lines 1 and 3
        assert freq["2"] == 2  # item 2 appears in lines 1 and 3
        assert freq["5"] == 1  # item 5 appears in line 1
        
        stats = result["statistics"]
        assert stats["total_items"] == 8  # 8 unique items
        assert stats["total_sequences"] == 3
        
        print(f"  Item freq: {result['statistics']['total_items']} unique items, "
              f"{result['statistics']['total_sequences']} sequences")
        print("✅ test_compute_item_interaction_freq PASSED")


def test_compute_category_overlap_stats():
    """测试内置方法: 类别重叠统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            log_dir=log_dir,
        )
        
        # 使用 mock 数据
        wrong_cases = _make_mock_wrong_cases()
        item_map = _make_mock_item_text_map()
        
        preloaded = {
            "wrong_text_cases": wrong_cases,
            "item_text_map": item_map,
        }
        
        result = engine._compute_category_overlap_stats(
            {"id": "H1", "claim": "跨类别跳跃"}, preloaded
        )
        
        assert result is not None
        assert "per_case_overlap" in result
        assert "statistics" in result
        assert "sample" in result
        
        stats = result["statistics"]
        assert "total_wrong_cases" in stats
        assert "no_category_overlap_pct" in stats
        
        # 检查样本有正确的结构
        if result["sample"]:
            sample_entry = result["sample"][0]
            assert "target_id" in sample_entry
            assert "overlap_size" in sample_entry
        
        print(f"  Category overlap stats: total={stats['total_wrong_cases']}, "
              f"no_overlap_pct={stats['no_category_overlap_pct']}%")
        print("✅ test_compute_category_overlap_stats PASSED")


def test_compute_category_distribution():
    """测试内置方法: 类别分布统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建元数据文件
        meta_file = os.path.join(data_dir, "id_meta_data.json")
        meta_data = {
            "1": {"title": "Laptop", "categories": "Electronics > Computers > Laptops"},
            "2": {"title": "Phone", "categories": "Electronics > Phones > Smartphones"},
            "3": {"title": "Book", "categories": "Books > Fiction > Sci-Fi"},
            "4": {"title": "Shirt", "categories": "Clothing > Men > T-Shirts"},
            "5": {"title": "Coffee", "categories": "Food > Drinks > Coffee"},
        }
        with open(meta_file, 'w') as f:
            json.dump(meta_data, f)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            data_dir=data_dir,
            log_dir=log_dir,
        )
        
        result = engine._compute_category_distribution({"item_metadata": meta_data})
        
        assert result is not None
        assert "statistics" in result
        
        stats = result["statistics"]
        assert stats["total_items"] == 5
        assert stats["unique_categories"] == 4  # Electronics, Books, Clothing, Food
        
        # Electronics should be the top category (2 items)
        assert stats["top_category"] == "Electronics"
        
        print(f"  Category distribution: {stats['unique_categories']} categories, "
              f"top={stats['top_category']} ({stats['top_category_pct']}%)")
        print("✅ test_compute_category_distribution PASSED")


def test_compute_recommendation_frequency():
    """测试内置方法: 推荐频率统计"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            log_dir=log_dir,
        )
        
        wrong_cases = _make_mock_wrong_cases()
        result = engine._compute_recommendation_frequency({"wrong_text_cases": wrong_cases})
        
        assert result is not None
        assert "frequency_dict" in result
        assert "statistics" in result
        
        stats = result["statistics"]
        assert "total_prediction_slots" in stats
        assert "top_20_recommendations" in stats
        
        print(f"  Rec frequency: {stats['unique_items_in_predictions']} unique items in predictions")
        print("✅ test_compute_recommendation_frequency PASSED")


def test_compute_sequence_target_mapping():
    """测试内置方法: 序列-目标关联数据"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            log_dir=log_dir,
        )
        
        wrong_cases = _make_mock_wrong_cases()
        result = engine._compute_sequence_target_mapping({"wrong_text_cases": wrong_cases})
        
        assert result is not None
        assert "per_case_mapping" in result
        assert "statistics" in result
        
        stats = result["statistics"]
        assert "total_cases" in stats
        assert "avg_sequence_length" in stats
        
        print(f"  Sequence-target mapping: {stats['total_cases']} cases, "
              f"avg_seq_len={stats['avg_sequence_length']}")
        print("✅ test_compute_sequence_target_mapping PASSED")


def test_identify_missing_data():
    """测试缺失数据识别"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        inventory = DataInventory(project_root=tmpdir, data_dir=data_dir, log_dir=log_dir)
        
        # 测试 1: 所有数据都已加载 → 无缺失
        loaded_all = {
            "category_overlap_stats": {"some": "data"},
            "item_popularity": {"1": 100},
            "wrong_text_cases": [{"target_id": 1}],
        }
        missing = inventory.identify_missing_data(
            ["类别重叠统计", "物品热度分布"], loaded_all
        )
        assert len(missing) == 0
        
        # 测试 2: 类别重叠缺失
        loaded_partial = {"item_popularity": {"1": 100}}
        missing = inventory.identify_missing_data(
            ["类别重叠统计", "跨类别跳跃分析"], loaded_partial
        )
        assert len(missing) > 0
        assert any("category_overlap_stats" in m for m in missing)
        
        # 测试 3: 推荐频率缺失
        loaded_empty = {}
        missing = inventory.identify_missing_data(
            ["模型推荐频率", "推荐频次对比"], loaded_empty
        )
        assert len(missing) > 0
        assert any("recommendation_frequency" in m for m in missing)
        
        print("✅ test_identify_missing_data PASSED")


def test_compute_needed_data_integration():
    """测试 compute_needed_data 完整流程"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建训练数据
        train_file = os.path.join(data_dir, "Beauty_train.txt")
        with open(train_file, 'w') as f:
            f.write("1 2 3 4 5\n6 7 8\n1 2 3\n")
        
        # 创建元数据
        meta_file = os.path.join(data_dir, "id_meta_data.json")
        with open(meta_file, 'w') as f:
            json.dump(_make_mock_item_text_map(), f)
        
        engine = DataComputationEngine(
            project_root=tmpdir,
            data_dir=data_dir,
            log_dir=log_dir,
        )
        
        wrong_cases = _make_mock_wrong_cases()
        item_map = _make_mock_item_text_map()
        
        preloaded = {
            "wrong_text_cases": wrong_cases,
            "item_text_map": item_map,
        }
        
        # 模拟缺失数据列表
        missing = [
            "category_overlap_stats: 目标物品与用户历史序列的类别重叠统计",
            "recommendation_frequency: 模型推荐结果中各物品的推荐频次",
        ]
        
        hypothesis = {"id": "H1", "claim": "跨类别跳跃"}
        
        computed = engine.compute_needed_data(missing, hypothesis, preloaded)
        
        # 应至少计算成功一部分
        assert len(computed) > 0
        
        # category_overlap_stats 应被计算
        if "category_overlap_stats" in computed:
            assert "per_case_overlap" in computed["category_overlap_stats"]
            assert "statistics" in computed["category_overlap_stats"]
        
        # recommendation_frequency 应被计算
        if "recommendation_frequency" in computed:
            assert "frequency_dict" in computed["recommendation_frequency"]
            assert "statistics" in computed["recommendation_frequency"]
        
        print(f"  Computed {len(computed)} data items: {list(computed.keys())}")
        print("✅ test_compute_needed_data_integration PASSED")


def test_data_computation_prompt():
    """测试 DATA_COMPUTATION_PROMPT 模板"""
    # 检查关键占位符
    assert "{missing_data_description}" in DATA_COMPUTATION_PROMPT
    assert "{raw_data_sources}" in DATA_COMPUTATION_PROMPT
    assert "{hypothesis_id}" in DATA_COMPUTATION_PROMPT
    assert "{hypothesis_claim}" in DATA_COMPUTATION_PROMPT
    assert "{verification_thought}" in DATA_COMPUTATION_PROMPT
    assert "{data_needed}" in DATA_COMPUTATION_PROMPT
    assert "{output_file_path}" in DATA_COMPUTATION_PROMPT
    
    # 检查提示了数据格式 (英文 keywords)
    assert "categories" in DATA_COMPUTATION_PROMPT
    assert "训练数据格式" in DATA_COMPUTATION_PROMPT
    
    print("✅ test_data_computation_prompt PASSED")


def test_data_inventory_get_computation_engine():
    """测试 DataInventory 获取计算引擎"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        inventory = DataInventory(project_root=tmpdir, data_dir=data_dir, log_dir=log_dir)
        
        # 获取引擎 (无 LLM)
        engine = inventory.get_computation_engine()
        assert engine is not None
        assert engine.llm is None
        
        # 再次获取 → 应返回相同实例
        engine2 = inventory.get_computation_engine()
        assert engine2 is engine
        
        # 传入 LLM → 应更新
        mock_llm = _make_mock_llm()
        engine3 = inventory.get_computation_engine(llm_client=mock_llm)
        assert engine3.llm is mock_llm
        
        print("✅ test_data_inventory_get_computation_engine PASSED")


def test_prepare_verification_data_with_computation():
    """测试 _prepare_verification_data 自动计算缺失数据"""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = os.path.join(tmpdir, "Recmodel", "data")
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建训练数据
        train_file = os.path.join(data_dir, "Beauty_train.txt")
        with open(train_file, 'w') as f:
            f.write("1 2 3 4 5\n6 7 8\n1 2 3\n")
        
        # 创建元数据
        meta_file = os.path.join(data_dir, "id_meta_data.json")
        with open(meta_file, 'w') as f:
            json.dump(_make_mock_item_text_map(), f)
        
        mock_llm = _make_mock_llm()
        agent = HypothesisVerificationAgent(
            llm_client=mock_llm,
            item_text_map=_make_mock_item_text_map(),
            project_root=tmpdir,
            data_dir=data_dir,
            log_dir=log_dir,
        )
        
        wrong_cases = _make_mock_wrong_cases()
        
        # 假设需要类别重叠数据
        hypothesis = {
            "id": "H1",
            "claim": "目标物品与用户历史序列在类别上完全不相关",
            "verification_thought": "统计误推案例中目标物品与历史序列的类别交集",
            "data_needed": ["误推案例列表", "物品元数据", "类别重叠统计"],
            "expected_if_true": "误推中无类别重叠的比例显著高于正确推荐",
            "expected_if_false": "误推与正确推荐在类别重叠比例上无显著差异",
        }
        
        verification_plan = {
            "verification_plan": {
                "data_sources": ["wrong_text_cases", "item_metadata", "category_overlap_stats"],
            }
        }
        
        preloaded = agent._prepare_preloaded_data(
            wrong_cases, None, _make_mock_item_popularity(),
            {"NDCG@10": 0.3}, {"NDCG@10": 0.15}
        )
        
        # 调用 _prepare_verification_data
        verification_data = agent._prepare_verification_data(
            hypothesis, verification_plan, preloaded
        )
        
        # 应包含 wrong_text_cases (原始传入)
        assert "wrong_text_cases" in verification_data
        
        # 应包含 item_text_map
        assert "item_text_map" in verification_data
        
        # 检查是否有计算生成的数据 (可能不会完全成功, 但流程应该运行不崩溃)
        # 关键: 不会因为缺失数据而崩溃
        assert verification_data is not None
        
        print(f"  Verification data keys: {list(verification_data.keys())}")
        print("✅ test_prepare_verification_data_with_computation PASSED")


def test_format_computed_data_for_prompt():
    """测试格式化计算数据为 LLM prompt"""
    with tempfile.TemporaryDirectory() as tmpdir:
        log_dir = os.path.join(tmpdir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        engine = DataComputationEngine(project_root=tmpdir, log_dir=log_dir)
        
        computed_data = {
            "category_overlap_stats": {
                "statistics": {
                    "total_wrong_cases": 50,
                    "no_category_overlap_pct": 65.0,
                    "avg_overlap_ratio": 0.12,
                },
                "sample": {"1": "Electronics"},
                "computation_method": "builtin: category overlap",
            },
            "recommendation_frequency": {
                "frequency_dict": {"1": 100, "2": 80},
                "statistics": {
                    "total_prediction_slots": 1000,
                    "unique_items_in_predictions": 20,
                },
                "sample": {"1": 100},
            },
        }
        
        formatted = engine.format_computed_data_for_prompt(computed_data)
        
        assert "category_overlap_stats" in formatted
        assert "recommendation_frequency" in formatted
        assert "65.0" in formatted  # no_overlap_pct
        # Check for the known description from KNOWN_COMPUTED_DATA
        assert "目标物品与用户历史序列的类别重叠统计" in formatted
        
        print(f"  Formatted prompt text length: {len(formatted)} chars")
        print("✅ test_format_computed_data_for_prompt PASSED")


def test_inject_data_loading_truncated_variable_alias():
    """验证 _inject_data_loading 在大列表截断时, 原始变量名仍可用"""
    tmp_dir = tempfile.mkdtemp()
    try:
        agent = HypothesisVerificationAgent.__new__(HypothesisVerificationAgent)
        agent.log_dir = tmp_dir
        
        # 构造超过100项的列表, 模拟 wrong_text_cases 被截断的情况
        big_list = [{"user_id": f"u{i}", "item": f"item{i}"} for i in range(200)]
        small_dict = {"metric1": 0.5, "metric2": 0.3}
        
        verification_data = {
            "wrong_text_cases": big_list,   # >100, 会被截断为 wrong_text_cases_sample
            "overall_metrics": small_dict,  # dict, 不会被截断
        }
        
        output_file = os.path.join(tmp_dir, "verification_scripts", "result_H2.json")
        dummy_code = "# dummy verification code\npass"
        
        result_code = agent._inject_data_loading(dummy_code, verification_data, output_file)
        
        # 验证: 原始变量名 wrong_text_cases 必须出现在注入代码中
        assert "wrong_text_cases = " in result_code, \
            "原始变量名 wrong_text_cases 必须被注入, 否则 LLM 生成的代码引用该变量时会报 NameError"
        
        # 验证: wrong_text_cases 应指向 wrong_text_cases_sample (截断数据)
        assert "wrong_text_cases = wrong_text_cases_sample" in result_code, \
            "截断后, 原始变量名应指向样本数据"
        
        # 验证: wrong_text_cases_sample 和 wrong_text_cases_count 也要存在
        assert "wrong_text_cases_sample = " in result_code
        assert "wrong_text_cases_count = " in result_code
        
        # 验证: 小字典保持原始变量名直接映射
        assert "overall_metrics = _preloaded.get" in result_code
        
        # 验证: JSON 数据文件已创建且包含截断后的数据
        data_file = os.path.join(tmp_dir, "verification_scripts", "data", "preloaded_data.json")
        assert os.path.exists(data_file)
        with open(data_file, 'r') as f:
            saved_data = json.load(f)
        assert "wrong_text_cases_sample" in saved_data
        assert len(saved_data["wrong_text_cases_sample"]) == 50  # 前50条样本
        assert saved_data["wrong_text_cases_count"] == 200
        assert "wrong_text_cases" not in saved_data  # 原始key不在JSON中(已截断)
        assert "overall_metrics" in saved_data
        
        # 验证: 生成的代码可以正确加载变量 (模拟执行)
        # 提取注入部分 (不含 dummy_code), 验证变量赋值语法正确
        injection_only = result_code.split("# dummy verification code")[0]
        # 确保没有语法错误 — 变量赋值顺序正确 (sample 在 alias 之前)
        lines = [l.strip() for l in injection_only.split('\n') if l.strip() and '=' in l and not l.strip().startswith('#')]
        sample_line_idx = next(i for i, l in enumerate(lines) if 'wrong_text_cases_sample = _preloaded' in l)
        alias_line_idx = next(i for i, l in enumerate(lines) if 'wrong_text_cases = wrong_text_cases_sample' in l)
        assert alias_line_idx > sample_line_idx, \
            "别名赋值必须在样本赋值之后, 否会 NameError"
        
        print("✅ test_inject_data_loading_truncated_variable_alias PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_execute_script_detects_error_result_in_output_file():
    """验证 _execute_script 能检测结果文件中的 'error' 字段, 不将其误判为成功
    
    这是关键 Bug 修复的测试:
    - LLM 生成的验证脚本用 try/except 捕获异常后写入 {"error": "..."} 
    - 脚本退出码为 0, _execute_script 读到文件后返回 True — BUG!
    - 修复后应返回 False + error 信息, 让修正循环生效
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        agent = HypothesisVerificationAgent.__new__(HypothesisVerificationAgent)
        agent.log_dir = tmp_dir
        agent.project_root = tmp_dir
        agent.MAX_EXECUTION_TIMEOUT = 30
        
        # 创建一个脚本: 退出码为 0, 但结果文件包含 error 字段
        script_dir = os.path.join(tmp_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        result_path = os.path.join(script_dir, "result_H2.json")
        script_path = os.path.join(script_dir, "verify_H2.py")
        
        # 脚本内容: 退出码0, 写入 {"hypothesis_id": "H2", "error": "Missing required variable: wrong_text_cases"}
        error_result = {"hypothesis_id": "H2", "error": "Missing required variable: wrong_text_cases"}
        error_script = (
            "import json, os\n"
            f"result = {json.dumps(error_result)}\n"
            f"os.makedirs(os.path.dirname('{result_path}'), exist_ok=True)\n"
            f"with open('{result_path}', 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        with open(script_path, 'w') as f:
            f.write(error_script)
        
        # 执行并检查结果
        success, result, error = agent._execute_script(script_path)
        
        # 关键: 应返回 False, 不应误判为成功
        assert success is False, \
            "包含 'error' 字段的结果应返回 False, 否则会跳过 LLM 修正循环"
        assert result is None, \
            "error-result 不应作为 result_dict 返回"
        assert error is not None, \
            "error 信息应被返回, 用于 LLM 修正"
        assert "Missing required variable" in error, \
            f"错误信息应包含原始错误描述, got: {error}"
        assert "Script error-result" in error, \
            f"应标注为 error-result 类型, got: {error}"
        
        print("✅ test_execute_script_detects_error_result_in_output_file PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_execute_script_detects_error_result_in_stdout():
    """验证 _execute_script 能检测 stdout 中的 error-result"""
    tmp_dir = tempfile.mkdtemp()
    try:
        agent = HypothesisVerificationAgent.__new__(HypothesisVerificationAgent)
        agent.log_dir = tmp_dir
        agent.project_root = tmp_dir
        agent.MAX_EXECUTION_TIMEOUT = 30
        
        script_dir = os.path.join(tmp_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_path = os.path.join(script_dir, "verify_H3.py")
        
        # 脚本内容: 退出码0, 无输出文件, 但 stdout 输出 error JSON
        stdout_error_script = '''
import json
result = {"hypothesis_id": "H3", "error": "Script execution failed: division by zero"}
print(json.dumps(result))
'''
        with open(script_path, 'w') as f:
            f.write(stdout_error_script)
        
        success, result, error = agent._execute_script(script_path)
        
        assert success is False, "stdout 中的 error-result 也应返回 False"
        assert result is None
        assert error is not None
        assert "division by zero" in error
        
        print("✅ test_execute_script_detects_error_result_in_stdout PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_execute_script_normal_result_still_succeeds():
    """验证正常结果 (无 error 字段) 仍然返回 True — 防止误杀"""
    tmp_dir = tempfile.mkdtemp()
    try:
        agent = HypothesisVerificationAgent.__new__(HypothesisVerificationAgent)
        agent.log_dir = tmp_dir
        agent.project_root = tmp_dir
        agent.MAX_EXECUTION_TIMEOUT = 30
        
        script_dir = os.path.join(tmp_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        result_path = os.path.join(script_dir, "result_H1.json")
        script_path = os.path.join(script_dir, "verify_H1.py")
        
        # 正常结果: 包含 statistics 和 interpretation, 没有 error 字段
        normal_result = {
            "hypothesis_id": "H1",
            "statistics": {"mean": 0.5, "p_value": 0.03},
            "interpretation": "假设被确认"
        }
        normal_script = (
            "import json, os\n"
            f"result = {json.dumps(normal_result)}\n"
            f"os.makedirs(os.path.dirname('{result_path}'), exist_ok=True)\n"
            f"with open('{result_path}', 'w') as f:\n"
            "    json.dump(result, f)\n"
        )
        with open(script_path, 'w') as f:
            f.write(normal_script)
        
        success, result, error = agent._execute_script(script_path)
        
        assert success is True, "正常结果应返回 True"
        assert result is not None
        assert result["hypothesis_id"] == "H1"
        assert "statistics" in result
        assert error is None
        
        print("✅ test_execute_script_normal_result_still_succeeds PASSED")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ════════════════════════════════════════
# ModelProbingEngine Tests (v3 新增)
# ════════════════════════════════════════

def test_model_probing_engine_init():
    """测试 ModelProbingEngine 初始化"""
    from agent.hypothesis_verification_agent import ModelProbingEngine
    
    tmp_dir = tempfile.mkdtemp()
    try:
        engine = ModelProbingEngine(project_root=tmp_dir)
        assert engine.project_root == tmp_dir
        assert engine.recmodel_dir == tmp_dir  # 因为没有 models.py
        assert engine.MAX_PROBE_TIMEOUT == 180
        assert engine.MAX_PROBE_FIX_ROUNDS == 3
        assert engine.MAX_PROBE_SAMPLES == 50
        assert engine.llm is None
        
        # KNOWN_MODEL_DATA_TYPES 应包含关键字映射
        assert "attention_weights" in engine.KNOWN_MODEL_DATA_TYPES
        assert "hidden_states" in engine.KNOWN_MODEL_DATA_TYPES
        assert "item_embeddings" in engine.KNOWN_MODEL_DATA_TYPES
        assert "model_predictions" in engine.KNOWN_MODEL_DATA_TYPES
        
        print("  ✅ ModelProbingEngine init")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_model_probing_engine_known_data_types():
    """测试 ModelProbingEngine 的已知数据类型关键词检测"""
    from agent.hypothesis_verification_agent import ModelProbingEngine
    
    engine = ModelProbingEngine(project_root=tempfile.mkdtemp())
    
    # 测试注意力权重关键词
    assert engine.is_model_internal_data("注意力权重矩阵")
    assert engine.is_model_internal_data("attention weights output")
    assert engine.is_model_internal_data("注意力坍缩现象")
    assert engine.is_model_internal_data("attention entropy")
    assert engine.is_model_internal_data("attention_probs")
    
    # 测试隐藏状态关键词
    assert engine.is_model_internal_data("hidden states of encoder")
    assert engine.is_model_internal_data("编码器中间表示")
    
    # 测试嵌入关键词
    assert engine.is_model_internal_data("item embeddings")
    assert engine.is_model_internal_data("嵌入向量")
    
    # 测试模型预测关键词
    assert engine.is_model_internal_data("模型预测分数")
    assert engine.is_model_internal_data("prediction scores")
    
    # 测试非模型内部数据 (不应匹配)
    assert not engine.is_model_internal_data("类别重叠统计")
    assert not engine.is_model_internal_data("item popularity distribution")
    assert not engine.is_model_internal_data("用户交互频率")
    
    shutil.rmtree(engine.project_root, ignore_errors=True)
    print("  ✅ ModelProbingEngine known data types")


def test_identify_model_internal_data():
    """测试 DataInventory.identify_model_internal_data 方法"""
    from agent.hypothesis_verification_agent import DataInventory
    
    tmp_dir = tempfile.mkdtemp()
    inv = DataInventory(project_root=tmp_dir)
    
    missing_data = [
        "category_overlap_stats: 目标物品与用户历史序列的类别重叠统计",
        "attention_weights: 模型自注意力权重矩阵 (需通过模型探测提取)",
        "item_interaction_freq: 训练数据中各物品的交互频率统计",
        "hidden_states: Transformer编码器各层的隐藏状态输出 (需通过模型探测提取)",
    ]
    
    model_internal = inv.identify_model_internal_data(missing_data)
    assert len(model_internal) == 2
    assert "attention_weights: 模型自注意力权重矩阵 (需通过模型探测提取)" in model_internal
    assert "hidden_states: Transformer编码器各层的隐藏状态输出 (需通过模型探测提取)" in model_internal
    
    print("  ✅ identify_model_internal_data")


def test_identify_computable_data():
    """测试 DataInventory.identify_computable_data 方法"""
    from agent.hypothesis_verification_agent import DataInventory
    
    tmp_dir = tempfile.mkdtemp()
    inv = DataInventory(project_root=tmp_dir)
    
    missing_data = [
        "category_overlap_stats: 目标物品与用户历史序列的类别重叠统计",
        "attention_weights: 模型自注意力权重矩阵 (需通过模型探测提取)",
        "item_interaction_freq: 训练数据中各物品的交互频率统计",
    ]
    
    computable = inv.identify_computable_data(missing_data)
    assert len(computable) == 2
    assert "category_overlap_stats: 目标物品与用户历史序列的类别重叠统计" in computable
    assert "item_interaction_freq: 训练数据中各物品的交互频率统计" in computable
    assert "attention_weights" not in [c.split(":")[0].strip() for c in computable]
    
    print("  ✅ identify_computable_data")


def test_data_inventory_get_model_probing_engine():
    """测试 DataInventory.get_model_probing_engine lazy init"""
    from agent.hypothesis_verification_agent import DataInventory
    
    tmp_dir = tempfile.mkdtemp()
    inv = DataInventory(project_root=tmp_dir)
    
    # 第一次调用 — 创建引擎
    engine = inv.get_model_probing_engine()
    assert engine is not None
    assert engine.project_root == tmp_dir
    assert engine.llm is None
    
    # 第二次调用 — 返回相同引擎
    engine2 = inv.get_model_probing_engine()
    assert engine2 is engine
    
    # 传入 llm_client — 更新引擎
    mock_llm = _make_mock_llm()
    engine3 = inv.get_model_probing_engine(llm_client=mock_llm)
    assert engine3.llm is mock_llm
    
    print("  ✅ DataInventory.get_model_probing_engine")


def test_model_probing_engine_discover_model_info():
    """测试 ModelProbingEngine._discover_model_info"""
    from agent.hypothesis_verification_agent import ModelProbingEngine
    
    # 使用真实的 Recmodel 目录
    recmodel_dir = os.path.join(os.path.dirname(__file__), "..", "Recmodel")
    if os.path.exists(os.path.join(recmodel_dir, "models.py")):
        engine = ModelProbingEngine(project_root=os.path.dirname(recmodel_dir))
        model_info = engine._discover_model_info()
        
        assert "model_source_files" in model_info
        assert "models.py" in model_info["model_source_files"]
        assert "modules.py" in model_info["model_source_files"]
        assert "model_args" in model_info
        assert model_info["model_args"]["hidden_size"] == 64
        assert model_info["model_args"]["num_attention_heads"] == 2
        
        # checkpoint 可能不存在 (如果未训练)
        # 但信息结构应该正确
        assert "checkpoint_path" in model_info
        
        print("  ✅ ModelProbingEngine._discover_model_info (with real Recmodel)")
    else:
        # 创建临时目录模拟
        tmp_dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
        
        # 创建假的 models.py 和 modules.py
        with open(os.path.join(tmp_dir, "models.py"), 'w') as f:
            f.write("class SASRec: pass\n")
        with open(os.path.join(tmp_dir, "modules.py"), 'w') as f:
            f.write("class SelfAttention: pass\n")
        
        # 创建假的训练数据
        with open(os.path.join(tmp_dir, "data", "Beauty_train.txt"), 'w') as f:
            f.write("1 2 3 4 5\n6 7 8\n")
        
        engine = ModelProbingEngine(project_root=tmp_dir)
        model_info = engine._discover_model_info()
        
        assert "model_source_files" in model_info
        assert "models.py" in model_info["model_source_files"]
        assert model_info["checkpoint_path"] is None  # 没有 .pt 文件
        
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print("  ✅ ModelProbingEngine._discover_model_info (mock)")


def test_model_probing_engine_is_model_internal_data():
    """测试 ModelProbingEngine.is_model_internal_data"""
    from agent.hypothesis_verification_agent import ModelProbingEngine
    
    engine = ModelProbingEngine(project_root=tempfile.mkdtemp())
    
    # 测试各种关键词
    test_cases = [
        ("SASRec模型在长序列中注意力权重集中在最近几个物品上", True),
        ("导致远距离但语义相关的惊喜物品被忽略（注意力坍缩现象）", True),
        ("从模型推理过程中提取自注意力权重矩阵", True),
        ("分析每个预测样本中注意力分布的集中程度", True),
        ("物品热度分布", False),
        ("类别重叠统计", False),
    ]
    
    for desc, expected in test_cases:
        result = engine.is_model_internal_data(desc)
        assert result == expected, f"Failed for '{desc}': got {result}, expected {expected}"
    
    shutil.rmtree(engine.project_root, ignore_errors=True)
    print("  ✅ ModelProbingEngine.is_model_internal_data")


def test_prepare_verification_data_with_model_probing():
    """测试 _prepare_verification_data 中的模型探测流程"""
    from agent.hypothesis_verification_agent import HypothesisVerificationAgent, DataInventory
    
    tmp_dir = tempfile.mkdtemp()
    os.makedirs(os.path.join(tmp_dir, "data"), exist_ok=True)
    
    # 创建假的训练数据文件
    train_file = os.path.join(tmp_dir, "data", "Beauty_train.txt")
    with open(train_file, 'w') as f:
        f.write("1 2 3 4 5\n6 7 8\n")
    
    # 创建假设 — 包含注意力权重需求
    hypothesis = {
        "id": "H1",
        "claim": "SASRec模型在长序列中注意力权重集中在最近几个物品上",
        "verification_thought": "从模型推理过程中提取自注意力权重矩阵",
        "data_needed": [
            "模型推理时的自注意力权重输出",
            "用户交互序列",
            "真实下一个物品标签",
        ],
    }
    
    # 创建 mock LLM
    mock_llm = _make_mock_llm()
    
    # 创建 Agent
    agent = HypothesisVerificationAgent(
        llm_client=mock_llm,
        project_root=tmp_dir,
    )
    
    # 测试 identify_missing_data — 应识别出注意力权重需求
    missing = agent.data_inventory.identify_missing_data(
        hypothesis["data_needed"], {}
    )
    
    # 应包含 attention_weights
    has_attention = any("attention" in m.lower() or "注意力" in m for m in missing)
    assert has_attention, f"Missing data should include attention_weights, got: {missing}"
    
    # 测试 identify_model_internal_data
    model_internal = agent.data_inventory.identify_model_internal_data(missing)
    assert len(model_internal) > 0, "Should have model-internal data"
    
    # 测试 identify_computable_data
    computable = agent.data_inventory.identify_computable_data(missing)
    # 注意力权重不应在可计算数据中
    assert not any("attention" in c.lower() or "注意力" in c for c in computable)
    
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("  ✅ _prepare_verification_data with model probing detection")


def test_model_probing_analysis_prompt():
    """测试模型探测分析 Prompt 模板"""
    from agent.hypothesis_verification_agent import MODEL_PROBING_ANALYSIS_PROMPT, MODEL_PROBING_SCRIPT_PROMPT
    
    # 检查 Prompt 模板存在且包含必要字段
    assert "{missing_data_description}" in MODEL_PROBING_ANALYSIS_PROMPT
    assert "{hypothesis_id}" in MODEL_PROBING_ANALYSIS_PROMPT
    assert "{models_source}" in MODEL_PROBING_ANALYSIS_PROMPT
    assert "{modules_source}" in MODEL_PROBING_ANALYSIS_PROMPT
    
    assert "{missing_data_description}" in MODEL_PROBING_SCRIPT_PROMPT
    assert "{checkpoint_path}" in MODEL_PROBING_SCRIPT_PROMPT
    assert "{recmodel_dir}" in MODEL_PROBING_SCRIPT_PROMPT
    assert "{modules_source}" in MODEL_PROBING_SCRIPT_PROMPT
    assert "{model_analysis_json}" in MODEL_PROBING_SCRIPT_PROMPT
    
    print("  ✅ Model probing prompts")


def test_identify_missing_data_with_attention_keywords():
    """测试 identify_missing_data 新增的模型内部数据关键词检测"""
    from agent.hypothesis_verification_agent import DataInventory
    
    tmp_dir = tempfile.mkdtemp()
    inv = DataInventory(project_root=tmp_dir)
    
    # 测试注意力权重关键词
    missing = inv.identify_missing_data(
        ["模型推理时的自注意力权重输出", "注意力分布"],
        {}
    )
    assert any("attention_weights" in m for m in missing)
    
    # 测试隐藏状态关键词
    missing = inv.identify_missing_data(
        ["编码器中间表示", "hidden states"],
        {}
    )
    assert any("hidden_states" in m for m in missing)
    
    # 测试嵌入向量关键词
    missing = inv.identify_missing_data(
        ["物品嵌入向量", "item embeddings"],
        {}
    )
    assert any("item_embeddings" in m for m in missing)
    
    # 测试混合 — 既有数据计算又有模型探测需求
    missing = inv.identify_missing_data(
        ["类别重叠统计", "注意力权重矩阵", "交互频率"],
        {}
    )
    model_internal = inv.identify_model_internal_data(missing)
    computable = inv.identify_computable_data(missing)
    
    assert len(model_internal) > 0  # 应包含注意力权重
    assert len(computable) > 0      # 应包含类别重叠和交互频率
    
    print("  ✅ identify_missing_data with attention keywords")
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
    
    # DataComputationEngine tests
    test_data_computation_engine_init()
    test_compute_item_interaction_freq()
    test_compute_category_overlap_stats()
    test_compute_category_distribution()
    test_compute_recommendation_frequency()
    test_compute_sequence_target_mapping()
    test_identify_missing_data()
    test_compute_needed_data_integration()
    test_data_computation_prompt()
    test_data_inventory_get_computation_engine()
    test_prepare_verification_data_with_computation()
    test_format_computed_data_for_prompt()
    test_inject_data_loading_truncated_variable_alias()
    test_execute_script_detects_error_result_in_output_file()
    test_execute_script_detects_error_result_in_stdout()
    test_execute_script_normal_result_still_succeeds()
    
    # ModelProbingEngine tests (v3)
    test_model_probing_engine_init()
    test_model_probing_engine_known_data_types()
    test_identify_model_internal_data()
    test_identify_computable_data()
    test_data_inventory_get_model_probing_engine()
    test_model_probing_engine_discover_model_info()
    test_model_probing_engine_is_model_internal_data()
    test_prepare_verification_data_with_model_probing()
    test_model_probing_analysis_prompt()
    
    print("\n═══════════ All Tests PASSED ✅ ═══════════\n")