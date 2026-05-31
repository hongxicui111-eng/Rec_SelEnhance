#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 案例分析器 — 让 LLM 分析推荐模型推理错误的文本案例，
推理出模型存在的问题，并给出具体的改进建议

核心功能:
1. 将错误案例的文本描述格式化为 LLM Prompt
2. LLM 分析错误模式，推理出模型瓶颈
3. 结合惊喜评估指标，给出针对性的改进建议
4. 生成结构化的分析报告供 Agent 循环使用
"""

import os
import json
import logging
import random
from typing import Dict, List, Optional

from .llm_utils import LLMRetryHelper, parse_json_from_response

logger = logging.getLogger("rec_self_evolve.llm_analyzer")


# ════════════════════════════════════════
# Prompt 模板
# ════════════════════════════════════════

CASE_ANALYSIS_PROMPT = """你是一位顶尖的推荐系统算法工程师，正在深入分析一个序列推荐模型的推理错误案例。

## 项目背景
这是一个**序列推荐系统**
- 模型通过用户的历史交互序列来预测下一个可能交互的物品
- 评估指标: NDCG@K, Recall@K (K=5,10,20)

## 当前模型配置
```
数据集: {data_name}
模型: {backbone}
损失: {loss_type}
负采样: {neg_sampler} (N={N}, M={M})
对比学习: {CL_type}
学习率: {lr}
Batch Size: {batch_size}
隐藏层: {hidden_size} (层数={num_hidden_layers})
序列长度: {max_seq_length}
Dropout: {hidden_dropout_prob}
```

## 当前模型源码结构 (供你参考结构修改)
{source_code_summary}

## 整体评估指标
```json
{overall_metrics}
```

## 惊喜评估指标 (模型对"惊喜"交互的捕获能力)
```json
{surprise_metrics}
```

## 诊断信息
```json
{diagnosis}
```

## 错误案例分析 (从推理错误的 500 个案例中精选)
以下是模型预测错误的典型案例，每个案例包含:
- 用户的历史交互序列 (文本描述)
- 模型应该预测的目标物品 (文本描述)
- 模型实际预测的 Top-20 物品 (文本描述)
- 目标物品在预测中的排名 (如果不在 Top-20 中，标记为 "未命中")
- 惊喜度评分 (越高表示越偏离历史模式)

```json
{sample_cases}
```

## 任务

请深入分析这些错误案例，推理出模型为什么推不对，并给出具体的改进建议。

**⚠ 重要: 你的改进建议必须包含结构修改方案，不仅仅是调参数!**
如果模型瓶颈是架构性的，仅调参数无法解决。

### 分析维度 (必须覆盖每个维度)

1. **错误模式分析**: 
   - 这些错误案例有什么共同特征？
   - 是哪类物品最容易被误推？
   - 用户的哪种行为模式最难被捕捉？

2. **模型瓶颈推理**: 
   - 从错误模式出发，自由推理模型架构中可能存在的瓶颈
   - 不要局限于常见套路，可以从任何角度深入思考（如信息流、梯度、表达能力、归纳偏置等）
   - **如果瓶颈是架构性的，需要提出具体的代码修改方案**

3. **惊喜交互失败原因**:
   - 为什么模型无法预测与历史行为模式差异大的交互？
   - 模型是否过度依赖"相似性"(总是推荐与历史相似的物品)?
   - 如何增强模型对"惊喜"交互的感知能力？(需要什么结构性改动？)

4. **改进建议 (必须包含结构修改!)**:
   - 针对发现的问题，提出**具体可实施的**改进方案
   - 每个改进方案需要说明: 改什么参数/结构 → 预期效果 → 风险评估
   - **结构修改方案**需要指出: 修改哪个文件的哪个类/函数 → 具体改法 → 理论依据
   - 优先级排序: 最可能带来提升的改进排在前面

### 输出格式 (严格遵守)

```json
{{
  "error_patterns": {{
    "pattern_1": "描述第一种错误模式",
    "pattern_2": "描述第二种错误模式",
    "pattern_3": "描述第三种错误模式"
  }},
  "model_bottleneck": {{
    "瓶颈1": "你推理出的第一个瓶颈及其原因",
    "瓶颈2": "你推理出的第二个瓶颈及其原因",
    "瓶颈3": "你推理出的第三个瓶颈及其原因"
  }},
  "surprise_failure_reasons": {{
    "main_reason": "惊喜交互失败的主要原因",
    "secondary_reason": "次要原因",
    "structural_cause": "导致惊喜失败的架构性原因"
  }},
  "improvement_suggestions": [
    {{
      "priority": 1,
      "action_type": "parameter_change | structure_change | data_change",
      "description": "具体改进描述",
      "param_changes": {{}},
      "structural_change_detail": {{
        "target_file": "models.py | modules.py | trainers.py",
        "target_class_or_function": "要修改的类名.方法名",
        "description": "具体修改什么、为什么要这样改",
        "approach": "修改方法的简述 (如: 在attention_scores上加时间衰减偏置)",
        "expected_effect": "预期提升的指标和幅度",
        "risk": "可能的风险和副作用",
        "theoretical_basis": "理论依据或相关工作引用"
      }},
      "expected_effect": "预期提升的指标和幅度",
      "risk": "可能的风险和副作用"
    }},
    {{
      "priority": 2,
      "action_type": "structure_change",
      "description": "...",
      "structural_change_detail": {{
        "target_file": "...",
        "target_class_or_function": "...",
        "description": "...",
        "approach": "...",
        "expected_effect": "...",
        "risk": "...",
        "theoretical_basis": "..."
      }},
      "expected_effect": "...",
      "risk": "..."
    }}
  ],
  "summary": "一句话总结模型最大的问题和最关键的改进方向 (必须提到是否需要结构修改)"
}}
```"""

SURPRISE_OPTIMIZATION_PROMPT = """你是一位推荐系统专家，专门研究如何让推荐模型更好地捕捉"惊喜"交互。

## 背景
在推荐系统中，"惊喜"交互是指用户与历史行为模式差异大的交互。
例如: 一个一直购买美妆产品的用户突然购买了一个电子产品。
一个好的推荐系统不仅要推荐用户可能喜欢的 (准确性)，还要能捕捉用户意想不到但会感兴趣的新领域 (惊喜性)。

## 当前模型性能
- 整体: {overall_summary}
- 惊喜子集: {surprise_summary}
- 差距: 惊喜子集的 NDCG/Recall 比整体低 {gap_pct}%

## 模型配置
```
{config_summary}
```

## 当前模型源码结构
{source_code_summary}

## 问题
模型在惊喜交互上的表现远低于整体水平，这表明模型过度依赖"相似性推荐"，
总是推荐与用户历史相似的物品，无法跳出历史模式。
**仅靠调参数无法解决这个问题**——需要结构性的代码修改!

## 任务
请提出**专门针对提升惊喜交互捕获能力**的改进方案，必须包含结构修改!

改进方向 (每项都涉及代码修改，不是调参数):
请从你推理出的瓶颈出发，自由提出结构性改进方案。不要局限于上述常见套路，可以从模型架构、训练策略、数据建模等任何角度出发。

### 输出格式

```json
{{
  "root_cause_analysis": "为什么模型无法捕获惊喜交互的根本原因 (架构层面)",
  "structural_cause": "导致惊喜失败的架构性瓶颈",
  "suggestions": [
    {{
      "priority": 1,
      "approach": "方法名称",
      "description": "具体描述",
      "action_type": "structure_change | parameter_change",
      "structural_change_detail": {{
        "target_file": "models.py | modules.py | trainers.py",
        "target_class_or_function": "要修改的类名.方法名",
        "description": "具体修改什么、为什么要这样改",
        "approach": "修改方法的简述",
        "theoretical_basis": "理论依据"
      }},
      "param_changes": {{}},
      "expected_surprise_improvement": "预期惊喜指标提升幅度",
      "expected_overall_effect": "对整体指标的影响 (正面/负面/中性)",
      "implementation_complexity": "低/中/高"
    }}
  ],
  "risk_assessment": "过度追求惊喜性可能带来的风险 (如准确性下降)"
}}
```"""


class LLMCaseAnalyzer:
    """
    LLM 案例分析器
    
    将错误案例和评估指标格式化为 LLM Prompt，
    让 LLM 分析模型瓶颈并给出改进建议
    """

    def __init__(self, llm_client, item_text_map: Dict = None):
        """
        Args:
            llm_client: LLMClient 实例 (来自 agent.llm_client)
            item_text_map: 物品 ID → 元数据 dict (id_meta-data.json 格式) 或 flat str 映射
        """
        self.llm = llm_client
        self.item_text_map = item_text_map or {}
        self.llm_retry = LLMRetryHelper(llm_client)

    def item_id_to_text(self, item_id: int) -> str:
        """将物品 ID 转化为文本描述，支持 id_meta_data.json 的 nested dict 格式"""
        str_id = str(item_id)
        if str_id in self.item_text_map:
            entry = self.item_text_map[str_id]
            if isinstance(entry, dict):
                title = entry.get("title", "")
                categories = entry.get("categories", "")
                parts = []
                if title:
                    parts.append(title)
                if categories:
                    cat_list = categories.split(" > ")
                    leaf_cat = cat_list[-1] if cat_list else categories
                    parts.append(f"[{leaf_cat}]")
                return " | ".join(parts) if parts else f"Item_{item_id}"
            elif isinstance(entry, str):
                return entry
        return f"Item_{item_id}"

    def format_cases_for_llm(self, text_cases: List[Dict], 
                              max_cases: int = 30,
                              prioritize_surprise: bool = True) -> str:
        """
        将文本案例格式化为 LLM 可理解的格式
        
        选取策略:
        - 优先选取高惊喜度的案例 (更能揭示模型对惊喜交互的问题)
        - 从不同的惊喜度等级中都选取一些案例 (保证覆盖面)
        - 限制案例数量避免 Token 过长
        
        Args:
            text_cases: 错误案例列表 (已转为文本格式)
            max_cases: 最多展示给 LLM 的案例数量
            prioritize_surprise: 是否优先选取高惊喜度案例
        """
        if not text_cases:
            return "[]"
        
        # 按惊喜度排序
        sorted_cases = sorted(text_cases, 
                              key=lambda x: x.get("surprise_score", 0),
                              reverse=prioritize_surprise)
        
        # 分层采样: 高/中/低惊喜度各取一些
        high = [c for c in sorted_cases if c.get("surprise_score", 0) >= 0.7]
        medium = [c for c in sorted_cases if 0.3 <= c.get("surprise_score", 0) < 0.7]
        low = [c for c in sorted_cases if c.get("surprise_score", 0) < 0.3]
        
        n_high = min(int(max_cases * 0.5), len(high))
        n_medium = min(int(max_cases * 0.3), len(medium))
        n_low = min(max_cases - n_high - n_medium, len(low))
        
        selected = random.sample(high, n_high) + \
                   random.sample(medium, n_medium) + \
                   random.sample(low, n_low)
        
        # 格式化每个案例
        formatted_cases = []
        for i, case in enumerate(selected):
            # 截断历史序列 (避免过长)
            history_text = case.get("history_text", [])
            if len(history_text) > 15:
                history_text = history_text[-15:]  # 只保留最近 15 个
            
            formatted = {
                "case_id": i + 1,
                "history": history_text,
                "target": case.get("target_text", f"Item_{case.get('target_id', '?')}"),
                "top20_predictions": case.get("predictions_text", [])[:10],  # 只展示 top10
                "target_rank": case.get("target_rank", -1),
                "hit_status": "未命中Top20" if case.get("target_rank", -1) == -1 or case.get("target_rank", -1) >= 20
                              else f"排名#{case.get('target_rank', '?')}",
                "surprise_score": case.get("surprise_score", 0),
                "sequence_length": case.get("original_length", 0),
            }
            formatted_cases.append(formatted)
        
        return json.dumps(formatted_cases, indent=2, ensure_ascii=False)

    def analyze_wrong_cases(self, text_cases: List[Dict],
                             model_config: Dict,
                             overall_metrics: Dict,
                             surprise_metrics: Dict = None,
                             diagnosis: Dict = None,
                             max_cases: int = 30,
                             source_code_summary: str = "") -> Optional[Dict]:
        """
        让 LLM 分析错误案例并给出改进建议 (参数 + 结构修改)
        
        Args:
            text_cases: 错误案例列表 (文本格式)
            model_config: 当前模型配置
            overall_metrics: 整体评估指标
            surprise_metrics: 惊喜子集评估指标
            diagnosis: 诊断信息 (来自 SurpriseEvaluator)
            max_cases: 展示给 LLM 的最大案例数量
            source_code_summary: 模型源码结构摘要
            
        Returns:
            Dict: LLM 的分析结果 (解析后的 JSON)
        """
        # 格式化案例
        cases_str = self.format_cases_for_llm(text_cases, max_cases=max_cases)
        
        # 格式化指标
        metrics_str = json.dumps(overall_metrics, indent=2, ensure_ascii=False)
        surprise_str = json.dumps(surprise_metrics or {}, indent=2, ensure_ascii=False)
        diagnosis_str = json.dumps(diagnosis or {}, indent=2, ensure_ascii=False)
        
        # 构建 Prompt
        prompt = CASE_ANALYSIS_PROMPT.format(
            data_name=model_config.get("data_name", "?"),
            backbone=model_config.get("backbone", "SASRec"),
            loss_type=model_config.get("loss_type", "?"),
            neg_sampler=model_config.get("neg_sampler", "?"),
            N=model_config.get("N", "?"),
            M=model_config.get("M", "?"),
            CL_type=model_config.get("CL_type", "?"),
            lr=model_config.get("lr", "?"),
            batch_size=model_config.get("batch_size", "?"),
            hidden_size=model_config.get("hidden_size", "?"),
            num_hidden_layers=model_config.get("num_hidden_layers", "?"),
            max_seq_length=model_config.get("max_seq_length", "?"),
            hidden_dropout_prob=model_config.get("hidden_dropout_prob", "?"),
            overall_metrics=metrics_str,
            surprise_metrics=surprise_str,
            diagnosis=diagnosis_str,
            sample_cases=cases_str,
            source_code_summary=source_code_summary or "SASRec: models.py (SRModel, SASRec) + modules.py (SelfAttention, Intermediate, Encoder, EncoderLayer) + trainers.py",
        )
        
        system_content = (
            "你是一位严谨的推荐系统算法专家，擅长从错误案例中推理模型瓶颈。"
            "你不仅分析指标问题，更重要的是能识别出模型架构层面的根本瓶颈，"
            "并提出具体的代码结构修改方案 (不仅仅是调参数)。"
        )
        
        # 使用 LLMRetryHelper: LLM 调用 + 健壮解析 + JSON 格式失败自动重试
        result = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content=system_content,
            temperature=0.7,
            max_retries=2,
            additional_instructions=(
                "5. 保持 JSON 结构包含以下字段: "
                "error_patterns, model_bottleneck, surprise_failure_reasons, "
                "improvement_suggestions (每个包含 priority, action_type, description, "
                "param_changes, structural_change_detail, expected_effect, risk), "
                "summary"
            ),
        )
        
        if result is None:
            logger.error("LLM case analysis failed - all retries exhausted")
            return {
                "raw_response": "",
                "parse_success": False,
                "parse_error": "All LLM retries exhausted (robust_parse + 2 LLM fix retries)",
            }
        
        result["parse_success"] = True
        return result

    def analyze_surprise_optimization(self, overall_metrics: Dict,
                                       surprise_metrics: Dict,
                                       model_config: Dict,
                                       diagnosis: Dict = None,
                                       source_code_summary: str = "") -> Optional[Dict]:
        """
        专门针对惊喜交互优化的 LLM 分析
        
        与 analyze_wrong_cases 不同，这个方法更聚焦于
        "如何提升模型对惊喜交互的捕获能力"，并强调结构修改
        """
        # 计算差距
        gap_pct = self._compute_gap_percentage(overall_metrics, surprise_metrics)
        
        overall_summary = self._format_metrics_summary(overall_metrics)
        surprise_summary = self._format_metrics_summary(surprise_metrics or {})
        config_summary = json.dumps(model_config, indent=2, ensure_ascii=False)
        
        prompt = SURPRISE_OPTIMIZATION_PROMPT.format(
            overall_summary=overall_summary,
            surprise_summary=surprise_summary,
            gap_pct=gap_pct,
            config_summary=config_summary,
            source_code_summary=source_code_summary or "SASRec: models.py (SRModel, SASRec) + modules.py (SelfAttention, Intermediate, Encoder, EncoderLayer) + trainers.py",
        )
        
        response = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content="你是一位推荐系统专家，专注于惊喜性(Serendipity)研究。",
            temperature=0.7,
            max_retries=2,
            additional_instructions=(
                "5. 保持 JSON 结构包含以下字段: "
                "surprise_patterns, surprise_bottleneck, optimization_suggestions, "
                "summary"
            ),
        )
        
        if response is None:
            logger.error("LLM surprise analysis failed - all retries exhausted")
            return {
                "raw_response": "",
                "parse_success": False,
                "parse_error": "All LLM retries exhausted (robust_parse + 2 LLM fix retries)",
            }
        
        response["parse_success"] = True
        return response

    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 的回复为结构化 JSON (使用 llm_utils 的健壮解析器)
        
        注意: 主要的分析方法 (analyze_wrong_cases, analyze_surprise_optimization)
        已改用 LLMRetryHelper.call_and_parse_with_retry() 完成调用+解析+重试。
        此方法作为备用/直接调用场景的解析工具保留。
        
        改进: 使用 parse_json_from_response (5策略健壮解析) 替代原来的
        json.loads + ast.literal_eval 两策略解析。
        """
        parsed = parse_json_from_response(response)
        if parsed is not None:
            parsed["parse_success"] = True
            return parsed
        
        # 无法提取 JSON block
        logger.warning(f"Cannot parse LLM response with robust parser")
        return {
            "raw_response": response,
            "parse_success": False,
            "parse_error": "All 5 robust parsing strategies failed",
        }

    def _compute_gap_percentage(self, overall: Dict, surprise: Dict) -> str:
        """计算惊喜子集与整体的差距百分比"""
        if not overall or not surprise:
            return "N/A"
        
        gaps = []
        for key in ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20"]:
            if key in overall and key in surprise:
                gap = (surprise[key] - overall[key]) / overall[key] * 100
                gaps.append(f"{key}: {gap:.1f}%")
        
        return "; ".join(gaps) if gaps else "N/A"

    def _format_metrics_summary(self, metrics: Dict) -> str:
        """格式化指标为简洁摘要"""
        if not metrics:
            return "N/A"
        parts = []
        for key in ["NDCG@5", "NDCG@10", "NDCG@20", "Recall@5", "Recall@10", "Recall@20"]:
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        return " | ".join(parts) if parts else "N/A"

    def generate_combined_report(self, case_analysis: Dict,
                                  surprise_analysis: Dict,
                                  evaluation_report: Dict) -> Dict:
        """
        将案例分析和惊喜分析合并成一份综合报告
        
        这份报告将作为 Agent 循环中 LLM 分析阶段的输入，
        与之前仅基于指标的分析不同，这份报告包含了对错误案例的深入分析
        """
        combined = {
            "case_analysis": case_analysis,
            "surprise_analysis": surprise_analysis,
            "evaluation_report": {
                "overall_metrics": evaluation_report.get("test_full", {}),
                "surprise_metrics": evaluation_report.get("surprise_subset", {}),
                "train_subset_metrics": evaluation_report.get("train_subset", {}),
                "diagnosis": evaluation_report.get("diagnosis", {}),
            },
            "combined_suggestions": self._merge_suggestions(
                case_analysis, surprise_analysis
            ),
        }
        
        # 添加摘要
        case_summary = case_analysis.get("summary", "") if case_analysis and case_analysis.get("parse_success") else ""
        surprise_summary = surprise_analysis.get("root_cause_analysis", "") if surprise_analysis and surprise_analysis.get("parse_success") else ""
        
        combined["meta_summary"] = f"案例分析: {case_summary}\n惊喜分析: {surprise_summary}"
        
        return combined

    def _merge_suggestions(self, case_analysis: Dict, surprise_analysis: Dict) -> List[Dict]:
        """
        合并两份分析中的改进建议，按优先级排序
        
        去重: 如果两个分析都建议了相同的参数变更，只保留一个
        """
        all_suggestions = []
        
        # 从案例分析中提取
        if case_analysis and case_analysis.get("parse_success"):
            for s in case_analysis.get("improvement_suggestions", []):
                all_suggestions.append({
                    "source": "case_analysis",
                    "priority": s.get("priority", 99),
                    **s,
                })
        
        # 从惊喜分析中提取
        if surprise_analysis and surprise_analysis.get("parse_success"):
            for s in surprise_analysis.get("suggestions", []):
                all_suggestions.append({
                    "source": "surprise_analysis",
                    "priority": s.get("priority", 99),
                    **s,
                })
        
        # 按优先级排序
        all_suggestions.sort(key=lambda x: x.get("priority", 99))
        
        # 去重 (基于 param_changes 的 key)
        seen_keys = set()
        deduplicated = []
        for s in all_suggestions:
            param_changes = s.get("param_changes", {})
            key_set = set(param_changes.keys())
            if key_set and key_set & seen_keys:
                continue  # 已有相同参数的建议
            deduplicated.append(s)
            seen_keys.update(key_set)
        
        return deduplicated

    def save_analysis_report(self, report: Dict, output_path: str):
        """保存分析报告"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved analysis report to {output_path}")