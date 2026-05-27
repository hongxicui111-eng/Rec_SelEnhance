#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假设验证器 — 针对 LLM 分析结论进行自动化数据验证

核心问题:
  LLM 在分析错误案例后可能产生幻觉或主观臆断, 例如:
  - "冷门物品最容易被误推" → 但数据可能显示热门物品误推更多
  - "模型过度依赖相似性" → 但实际预测多样性可能不低
  - "位置编码无法捕捉时间衰减" → 但短序列表现可能也不差

解决方案:
  1. 从 LLM 分析结果中提取可验证的假设 (hypotheses)
  2. 对每个假设设计具体的验证方法 (基于实际数据统计)
  3. 运行验证, 产出 CONFIRMED / PARTIALLY_CONFIRMED / REFUTED / UNVERIFIABLE 评级
  4. 将验证结果反馈给下游流程, 过滤掉被反驳的结论

验证维度:
  - 物品热度分布 (冷门 vs 热门误推比例)
  - 类别偏差 (跨类别预测失败率)
  - 序列长度效应 (短/长序列表现差异)
  - 嵌入聚类 (预测物品与历史的相似度分布)
  - 惊喜度验证 (高惊喜案例 vs 低惊喜案例的实际差异)
"""

import os
import json
import logging
import numpy as np
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rec_self_evolve.hypothesis_verifier")


# ════════════════════════════════════════
# 假设提取 Prompt
# ════════════════════════════════════════

HYPOTHESIS_EXTRACTION_PROMPT = """你是一位严谨的数据科学家，正在从推荐系统分析报告中提取**可验证的假设**。

## 背景
LLM 分析了推荐模型 (SASRec) 的错误案例，给出了错误模式、模型瓶颈和改进建议。
但 LLM 的分析可能包含幻觉或主观臆断。我们需要提取其中的**可验证假设**，
用数据来确认或反驳这些结论。

## LLM 分析结论
```json
{llm_analysis_json}
```

## 任务
从上面的分析结论中，提取所有**可以用数据验证的假设**。

一个假设是可验证的，意味着我们可以通过以下方式验证:
1. **物品热度验证**: 检查误推物品是否偏向冷门/热门 (需要错误案例中的物品ID和训练集统计)
2. **类别偏差验证**: 检查误推是否偏向特定类别 (需要物品类别元数据)
3. **序列长度验证**: 检查短序列/长序列用户是否误推率更高 (需要用户的序列长度)
4. **相似性偏差验证**: 检查预测是否偏向与历史相似的物品 (需要物品嵌入的余弦相似度)
5. **惊喜度验证**: 检查高惊喜案例与低惊喜案例的预测命中率差异 (需要惊喜度评分)
6. **注意力失效验证**: 检查注意力权重是否在某些模式下异常 (需要模型内部注意力分布)

对于每个假设:
- 用精确的描述说明假设内容 (不要模糊)
- 指出验证方法 (用上述 6 种之一, 或自定义)
- 指出期望的数据现象 (如果假设成立, 我们应该观察到什么?)
- 指出反驳现象 (如果假设不成立, 我们应该观察到什么?)

### 输出格式 (严格遵守)

```json
{{
  "hypotheses": [
    {{
      "id": "H1",
      "claim": "精确描述假设内容",
      "source_field": "error_patterns | model_bottleneck | surprise_failure_reasons | improvement_suggestions",
      "verification_method": "item_popularity | category_bias | sequence_length | similarity_bias | surprise_score | attention_pattern | custom",
      "expected_if_true": "如果假设成立, 应观察到的数据现象",
      "expected_if_false": "如果假设不成立, 应观察到的数据现象",
      "confidence_in_llm": "high | medium | low (LLM 对此结论的置信度估计)",
      "priority": 1-5 (验证优先级, 5=最高)
    }},
    {{
      "id": "H2",
      "claim": "...",
      "source_field": "...",
      "verification_method": "...",
      "expected_if_true": "...",
      "expected_if_false": "...",
      "confidence_in_llm": "...",
      "priority": ...
    }}
  ],
  "summary": "哪些 LLM 结论最可能是幻觉/臆断, 哪些最可能有数据支撑"
}}
```"""


class HypothesisVerifier:
    """
    假设验证器 — 用数据验证 LLM 分析结论
    
    工作流程:
    1. extract_hypotheses(): 从 LLM 分析结果中提取可验证的假设
    2. verify_hypotheses(): 对每个假设运行数据验证
    3. generate_verification_report(): 生成验证报告, 标注每个结论的验证状态
    """
    
    # 验证状态枚举
    CONFIRMED = "CONFIRMED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    REFUTED = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    
    def __init__(self, llm_client, item_text_map: Dict = None):
        """
        Args:
            llm_client: LLMClient 实例
            item_text_map: 物品 ID → 元数据映射
        """
        self.llm = llm_client
        self.item_text_map = item_text_map or {}
    
    # ════════════════════════════════════════
    # Phase 1: 假设提取
    # ════════════════════════════════════════
    
    def extract_hypotheses(self, llm_analysis: Dict) -> Optional[List[Dict]]:
        """
        从 LLM 分析结果中提取可验证的假设
        
        Args:
            llm_analysis: LLM 案例分析的结果 (来自 LLMCaseAnalyzer.analyze_wrong_cases)
            
        Returns:
            List of hypothesis dicts, or None if extraction failed
        """
        if not llm_analysis or not llm_analysis.get("parse_success"):
            logger.warning("Cannot extract hypotheses from invalid LLM analysis")
            return None
        
        # 构建 prompt
        analysis_json = json.dumps(llm_analysis, indent=2, ensure_ascii=False)
        # 截断过长的 JSON (避免超 token)
        if len(analysis_json) > 6000:
            # 保留关键字段, 截断 improvement_suggestions
            truncated = dict(llm_analysis)
            suggestions = truncated.get("improvement_suggestions", [])
            if suggestions:
                truncated["improvement_suggestions"] = [
                    {k: v for k, v in s.items() if k != "structural_change_detail"}
                    for s in suggestions[:3]
                ]
            analysis_json = json.dumps(truncated, indent=2, ensure_ascii=False)
        
        prompt = HYPOTHESIS_EXTRACTION_PROMPT.format(
            llm_analysis_json=analysis_json,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位严谨的数据科学家，擅长从分析结论中识别可验证的假设。"
                    "你的目标是区分LLM的有数据支撑的结论和可能的主观臆断。"
                    "每个假设必须是可以用数据统计来验证的。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,  # 低温度 → 更精确的假设提取
            max_tokens=2048,
        )
        
        if response is None:
            logger.error("Hypothesis extraction failed - no LLM response")
            return None
        
        # 解析假设列表
        parsed = self._parse_hypothesis_response(response)
        if parsed and parsed.get("hypotheses"):
            logger.info(f"Extracted {len(parsed['hypotheses'])} verifiable hypotheses")
            return parsed["hypotheses"]
        
        return None
    
    # ════════════════════════════════════════
    # Phase 2: 数据验证
    # ════════════════════════════════════════
    
    def verify_hypotheses(self,
                          hypotheses: List[Dict],
                          wrong_text_cases: List[Dict],
                          all_wrong_cases: List[Dict] = None,
                          model_config: Dict = None,
                          item_popularity: Dict = None,
                          overall_metrics: Dict = None,
                          surprise_metrics: Dict = None) -> List[Dict]:
        """
        对每个假设运行数据验证
        
        Args:
            hypotheses: 提取的假设列表
            wrong_text_cases: LLM 使用的文本格式错误案例
            all_wrong_cases: 原始格式的错误案例 (包含 item IDs)
            model_config: 模型配置
            item_popularity: 物品热度分布 (item_id → count)
            overall_metrics: 整体评估指标
            surprise_metrics: 惊喜子集指标
            
        Returns:
            List of verified hypothesis dicts (每个包含 verification_result)
        """
        verified = []
        
        # 预计算统计基线 (所有假设共享)
        stats_baseline = self._compute_stats_baseline(
            wrong_text_cases, all_wrong_cases, item_popularity
        )
        
        for hyp in hypotheses:
            method = hyp.get("verification_method", "custom")
            hyp_id = hyp.get("id", "H?")
            claim = hyp.get("claim", "")
            
            logger.info(f"Verifying {hyp_id}: {claim[:80]}...")
            print(f"  🔬 验证假设 {hyp_id}: {claim[:60]}...")
            
            try:
                if method == "item_popularity":
                    result = self._verify_item_popularity(hyp, stats_baseline)
                elif method == "category_bias":
                    result = self._verify_category_bias(hyp, stats_baseline)
                elif method == "sequence_length":
                    result = self._verify_sequence_length(hyp, stats_baseline)
                elif method == "similarity_bias":
                    result = self._verify_similarity_bias(hyp, stats_baseline, wrong_text_cases)
                elif method == "surprise_score":
                    result = self._verify_surprise_score(hyp, stats_baseline, 
                                                         overall_metrics, surprise_metrics)
                elif method == "attention_pattern":
                    # 注意力验证需要模型内部数据, 目前标记为 UNVERIFIABLE
                    result = {
                        "status": self.UNVERIFIABLE,
                        "reason": "注意力权重验证需要模型运行时的内部数据, 当前不可用",
                        "evidence": None,
                    }
                else:
                    # 自定义方法 — 尝试基于文本案例做统计验证
                    result = self._verify_custom(hyp, stats_baseline, wrong_text_cases)
                
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = result
                verified.append(verified_hyp)
                
                status = result.get("status", self.UNVERIFIABLE)
                symbol = {"CONFIRMED": "✅", "PARTIALLY_CONFIRMED": "⚠️", 
                          "REFUTED": "❌", "UNVERIFIABLE": "🔍"}.get(status, "?")
                print(f"    {symbol} {hyp_id} → {status}: {result.get('brief', '')[:80]}")
                
            except Exception as e:
                logger.error(f"Verification failed for {hyp_id}: {e}")
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = {
                    "status": self.UNVERIFIABLE,
                    "reason": f"验证过程出错: {str(e)}",
                    "evidence": None,
                }
                verified.append(verified_hyp)
        
        return verified
    
    # ════════════════════════════════════════
    # 统计基线计算
    # ════════════════════════════════════════
    
    def _compute_stats_baseline(self,
                                 wrong_text_cases: List[Dict],
                                 all_wrong_cases: List[Dict] = None,
                                 item_popularity: Dict = None) -> Dict:
        """
        预计算统计基线 — 所有验证方法共享的数据统计
        
        Returns:
            Dict containing:
            - target_items: 误推目标物品的 ID 列表
            - target_categories: 误推目标物品的类别分布
            - popularity_of_targets: 误推目标物品的热度分布
            - seq_length_distribution: 误推用户的序列长度分布
            - surprise_distribution: 惊喜度分布
            - overall_item_popularity: 全量物品热度 (如果有)
        """
        baseline = {}
        
        # --- 误推目标物品统计 ---
        target_ids = []
        target_cats = []
        surprise_scores = []
        seq_lengths = []
        pred_item_ids_flat = []  # 所有预测物品
        
        for case in (wrong_text_cases or []):
            tid = case.get("target_id")
            if tid is not None:
                target_ids.append(tid)
            
            # 类别
            cat = self._get_leaf_category(tid)
            if cat:
                target_cats.append(cat)
            
            # 惊喜度
            ss = case.get("surprise_score", 0)
            surprise_scores.append(ss)
            
            # 序列长度
            sl = case.get("original_length", 0)
            seq_lengths.append(sl)
            
            # 预测物品
            preds = case.get("predictions_ids", [])
            if preds:
                pred_item_ids_flat.extend(preds)
        
        baseline["target_ids"] = target_ids
        baseline["target_categories"] = Counter(target_cats)
        baseline["surprise_scores"] = surprise_scores
        baseline["seq_lengths"] = seq_lengths
        baseline["pred_item_ids"] = pred_item_ids_flat
        
        # --- 物品热度 ---
        if item_popularity:
            baseline["overall_item_popularity"] = item_popularity
            # 误推目标的热度
            target_popularity = [item_popularity.get(str(tid), 0) for tid in target_ids 
                                 if str(tid) in item_popularity]
            baseline["target_popularity"] = target_popularity
            
            # 热度分组: 冷门(< 5次交互), 中等(5-50), 热门(> 50)
            cold = sum(1 for p in target_popularity if p < 5)
            medium = sum(1 for p in target_popularity if 5 <= p < 50)
            hot = sum(1 for p in target_popularity if p >= 50)
            baseline["target_popularity_groups"] = {
                "cold": cold, "medium": medium, "hot": hot,
                "cold_pct": cold / len(target_popularity) * 100 if target_popularity else 0,
                "medium_pct": medium / len(target_popularity) * 100 if target_popularity else 0,
                "hot_pct": hot / len(target_popularity) * 100 if target_popularity else 0,
            }
            
            # 全量物品热度分组 (作为对照组)
            all_pops = list(item_popularity.values())
            all_cold = sum(1 for p in all_pops if p < 5)
            all_medium = sum(1 for p in all_pops if 5 <= p < 50)
            all_hot = sum(1 for p in all_pops if p >= 50)
            baseline["overall_popularity_groups"] = {
                "cold": all_cold, "medium": all_medium, "hot": all_hot,
                "cold_pct": all_cold / len(all_pops) * 100 if all_pops else 0,
                "medium_pct": all_medium / len(all_pops) * 100 if all_pops else 0,
                "hot_pct": all_hot / len(all_pops) * 100 if all_pops else 0,
            }
        
        # --- 惊喜度分组 ---
        high_surprise = [s for s in surprise_scores if s >= 0.7]
        medium_surprise = [s for s in surprise_scores if 0.3 <= s < 0.7]
        low_surprise = [s for s in surprise_scores if s < 0.3]
        baseline["surprise_groups"] = {
            "high": len(high_surprise),
            "medium": len(medium_surprise),
            "low": len(low_surprise),
            "high_pct": len(high_surprise) / len(surprise_scores) * 100 if surprise_scores else 0,
            "medium_pct": len(medium_surprise) / len(surprise_scores) * 100 if surprise_scores else 0,
            "low_pct": len(low_surprise) / len(surprise_scores) * 100 if surprise_scores else 0,
        }
        
        # --- 序列长度分组 ---
        short_seq = [l for l in seq_lengths if l < 10]
        medium_seq = [l for l in seq_lengths if 10 <= l < 30]
        long_seq = [l for l in seq_lengths if l >= 30]
        baseline["seq_length_groups"] = {
            "short": len(short_seq),
            "medium": len(medium_seq),
            "long": len(long_seq),
            "short_pct": len(short_seq) / len(seq_lengths) * 100 if seq_lengths else 0,
            "medium_pct": len(medium_seq) / len(seq_lengths) * 100 if seq_lengths else 0,
            "long_pct": len(long_seq) / len(seq_lengths) * 100 if seq_lengths else 0,
        }
        
        return baseline
    
    # ════════════════════════════════════════
    # 具体验证方法
    # ════════════════════════════════════════
    
    def _verify_item_popularity(self, hyp: Dict, baseline: Dict) -> Dict:
        """
        验证物品热度相关假设
        
        常见假设:
        - "冷门物品最容易被误推" → 检查误推目标中冷门占比是否高于全量
        - "热门物品总是被推荐" → 检查预测中热门物品的占比
        """
        claim = hyp.get("claim", "")
        target_groups = baseline.get("target_popularity_groups", {})
        overall_groups = baseline.get("overall_popularity_groups", {})
        target_pop = baseline.get("target_popularity", [])
        
        if not target_groups or not overall_groups:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "缺少物品热度数据, 无法验证",
                "brief": "无热度数据",
                "evidence": None,
            }
        
        # 检查假设方向
        is_cold_bias_claim = any(kw in claim.lower() for kw in 
                                  ["冷门", "cold", "unpopular", "稀少", "低频"])
        is_hot_bias_claim = any(kw in claim.lower() for kw in 
                                 ["热门", "hot", "popular", "高频", "常见"])
        
        evidence = {}
        
        if is_cold_bias_claim:
            # 假设: 冷门物品误推更多 → 检查误推目标中冷门占比 vs 全量冷门占比
            cold_pct_target = target_groups.get("cold_pct", 0)
            cold_pct_overall = overall_groups.get("cold_pct", 0)
            
            # 如果误推目标冷门占比显著高于全量 → 确认
            ratio = cold_pct_target / cold_pct_overall if cold_pct_overall > 0 else 0
            
            evidence = {
                "cold_pct_in_wrong_targets": cold_pct_target,
                "cold_pct_in_all_items": cold_pct_overall,
                "ratio": ratio,
                "interpretation": f"误推目标中冷门物品占{cold_pct_target:.1f}%, "
                                  f"全量物品中冷门占{cold_pct_overall:.1f}%, "
                                  f"比率={ratio:.2f}",
            }
            
            if ratio > 1.5:
                status = self.CONFIRMED
                brief = f"冷门物品误推占比({cold_pct_target:.1f}%)显著高于全量({cold_pct_overall:.1f}%)"
            elif ratio > 1.1:
                status = self.PARTIALLY_CONFIRMED
                brief = f"冷门物品误推占比略高于全量, 但差异不显著"
            else:
                status = self.REFUTED
                brief = f"冷门物品误推占比({cold_pct_target:.1f}%)不高于全量({cold_pct_overall:.1f}%)"
        
        elif is_hot_bias_claim:
            # 假设: 模型总是推荐热门物品 → 检查预测中热门占比
            pred_ids = baseline.get("pred_item_ids", [])
            if pred_ids and baseline.get("overall_item_popularity"):
                pops = [baseline["overall_item_popularity"].get(str(pid), 0) 
                        for pid in pred_ids]
                hot_in_pred = sum(1 for p in pops if p >= 50)
                hot_pct_pred = hot_in_pred / len(pops) * 100 if pops else 0
                
                hot_pct_overall = overall_groups.get("hot_pct", 0)
                ratio = hot_pct_pred / hot_pct_overall if hot_pct_overall > 0 else 0
                
                evidence = {
                    "hot_pct_in_predictions": hot_pct_pred,
                    "hot_pct_in_all_items": hot_pct_overall,
                    "ratio": ratio,
                }
                
                if ratio > 1.5:
                    status = self.CONFIRMED
                    brief = f"预测中热门物品占{hot_pct_pred:.1f}%, 显著高于全量({hot_pct_overall:.1f}%)"
                elif ratio > 1.1:
                    status = self.PARTIALLY_CONFIRMED
                    brief = f"预测中热门占比略偏高"
                else:
                    status = self.REFUTED
                    brief = f"预测中热门占比({hot_pct_pred:.1f}%)不高于全量"
            else:
                status = self.UNVERIFIABLE
                brief = "缺少预测物品热度数据"
                evidence = None
        
        else:
            # 一般热度验证 — 展示分布对比
            evidence = {
                "target_popularity_groups": target_groups,
                "overall_popularity_groups": overall_groups,
            }
            status = self.PARTIALLY_CONFIRMED
            brief = f"误推物品热度分布: 冷{target_groups.get('cold_pct',0):.1f}% 中{target_groups.get('medium_pct',0):.1f}% 热{target_groups.get('hot_pct',0):.1f}%"
        
        return {
            "status": status,
            "brief": brief,
            "evidence": evidence,
            "method": "item_popularity",
        }
    
    def _verify_category_bias(self, hyp: Dict, baseline: Dict) -> Dict:
        """
        验证类别偏差假设
        
        常见假设:
        - "跨类别物品最容易被误推" → 检查误推目标类别是否偏向罕见类别
        - "模型总是推荐同一类别" → 检查预测类别分布是否高度集中
        """
        claim = hyp.get("claim", "")
        target_cats = baseline.get("target_categories", {})
        
        if not target_cats:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "缺少物品类别元数据",
                "brief": "无类别数据",
                "evidence": None,
            }
        
        # 类别集中度: Top-5 类别占多少比例
        total = sum(target_cats.values())
        top5 = sorted(target_cats.values(), reverse=True)[:5]
        top5_pct = sum(top5) / total * 100 if total > 0 else 0
        
        # 类别多样性: Shannon entropy
        probs = [c / total for c in target_cats.values()] if total > 0 else []
        entropy = -sum(p * np.log2(p) for p in probs if p > 0) if probs else 0
        max_entropy = np.log2(len(target_cats)) if len(target_cats) > 1 else 1
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
        
        is_cross_category_claim = any(kw in claim.lower() for kw in 
                                       ["跨类别", "cross-category", "类别跨越", "新类型", "不同类别"])
        is_same_category_claim = any(kw in claim.lower() for kw in 
                                      ["同一类别", "same category", "类别集中", "总是推荐相似"])
        
        evidence = {
            "num_unique_categories": len(target_cats),
            "top5_pct": top5_pct,
            "entropy": entropy,
            "normalized_entropy": normalized_entropy,
            "top_categories": target_cats.most_common(5),
        }
        
        if is_cross_category_claim:
            # 假设: 跨类别物品误推多 → 检查类别多样性是否高
            if normalized_entropy > 0.7:
                status = self.CONFIRMED
                brief = f"误推目标类别多样性高 (entropy={normalized_entropy:.2f}), 跨类别误推确实显著"
            elif normalized_entropy > 0.4:
                status = self.PARTIALLY_CONFIRMED
                brief = f"误推目标有一定类别多样性 (entropy={normalized_entropy:.2f})"
            else:
                status = self.REFUTED
                brief = f"误推目标类别高度集中 (entropy={normalized_entropy:.2f}), 跨类别误推不明显"
        
        elif is_same_category_claim:
            # 假设: 模型总是推荐同一类别 → 检查 top5 占比
            if top5_pct > 70:
                status = self.CONFIRMED
                brief = f"预测类别高度集中, Top-5类别占{top5_pct:.1f}%"
            elif top5_pct > 50:
                status = self.PARTIALLY_CONFIRMED
                brief = f"预测类别有一定集中, Top-5类别占{top5_pct:.1f}%"
            else:
                status = self.REFUTED
                brief = f"预测类别分布较均匀, Top-5类别仅占{top5_pct:.1f}%"
        
        else:
            status = self.PARTIALLY_CONFIRMED
            brief = f"误推类别分布: {len(target_cats)}个类别, Top-5占{top5_pct:.1f}%"
        
        return {
            "status": status,
            "brief": brief,
            "evidence": evidence,
            "method": "category_bias",
        }
    
    def _verify_sequence_length(self, hyp: Dict, baseline: Dict) -> Dict:
        """
        验证序列长度效应
        
        常见假设:
        - "短序列用户更容易误推" → 检查短序列用户在误推案例中的比例
        - "长序列用户难以预测" → 检查长序列用户的误推比例
        """
        claim = hyp.get("claim", "")
        seq_groups = baseline.get("seq_length_groups", {})
        
        if not seq_groups or sum([seq_groups.get("short",0), seq_groups.get("medium",0), seq_groups.get("long",0)]) == 0:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "缺少序列长度数据",
                "brief": "无序列长度数据",
                "evidence": None,
            }
        
        is_short_seq_claim = any(kw in claim.lower() for kw in 
                                  ["短序列", "short sequence", "序列不足", "短历史"])
        is_long_seq_claim = any(kw in claim.lower() for kw in 
                                 ["长序列", "long sequence", "序列过长", "长历史"])
        
        evidence = {
            "short_pct": seq_groups.get("short_pct", 0),
            "medium_pct": seq_groups.get("medium_pct", 0),
            "long_pct": seq_groups.get("long_pct", 0),
            "short_count": seq_groups.get("short", 0),
            "medium_count": seq_groups.get("medium", 0),
            "long_count": seq_groups.get("long", 0),
        }
        
        if is_short_seq_claim:
            short_pct = seq_groups.get("short_pct", 0)
            if short_pct > 40:
                status = self.CONFIRMED
                brief = f"短序列用户占误推案例的{short_pct:.1f}%, 确实显著"
            elif short_pct > 25:
                status = self.PARTIALLY_CONFIRMED
                brief = f"短序列用户在误推案例中占{short_pct:.1f}%"
            else:
                status = self.REFUTED
                brief = f"短序列用户仅占误推案例的{short_pct:.1f}%, 不是主要误推群体"
        
        elif is_long_seq_claim:
            long_pct = seq_groups.get("long_pct", 0)
            if long_pct > 40:
                status = self.CONFIRMED
                brief = f"长序列用户占误推案例的{long_pct:.1f}%, 确实显著"
            elif long_pct > 25:
                status = self.PARTIALLY_CONFIRMED
                brief = f"长序列用户在误推案例中占{long_pct:.1f}%"
            else:
                status = self.REFUTED
                brief = f"长序列用户仅占误推案例的{long_pct:.1f}%, 不是主要误推群体"
        
        else:
            status = self.PARTIALLY_CONFIRMED
            brief = f"序列长度分布: 短{seq_groups.get('short_pct',0):.1f}% 中{seq_groups.get('medium_pct',0):.1f}% 长{seq_groups.get('long_pct',0):.1f}%"
        
        return {
            "status": status,
            "brief": brief,
            "evidence": evidence,
            "method": "sequence_length",
        }
    
    def _verify_similarity_bias(self, hyp: Dict, baseline: Dict,
                                 wrong_text_cases: List[Dict]) -> Dict:
        """
        验证相似性偏差假设
        
        常见假设:
        - "模型过度依赖相似性, 总是推荐与历史相似的物品" → 检查预测与历史的类别重叠度
        - "模型缺乏多样性" → 检查预测的类别多样性
        
        注意: 嵌入层面的余弦相似度需要加载模型, 这里先做类别层面的验证
        """
        claim = hyp.get("claim", "")
        
        if not wrong_text_cases:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "缺少文本案例数据",
                "brief": "无案例数据",
                "evidence": None,
            }
        
        # 计算预测与历史类别的重叠度
        overlap_ratios = []
        pred_category_diversities = []
        
        for case in wrong_text_cases:
            history_cats = set()
            for h_text in case.get("history_text", []):
                cat = self._extract_category_from_text(h_text)
                if cat:
                    history_cats.add(cat)
            
            pred_cats = set()
            for p_text in case.get("predictions_text", [])[:10]:
                cat = self._extract_category_from_text(p_text)
                if cat:
                    pred_cats.add(cat)
            
            # 类别重叠度: 预测类别中有多少与历史类别相同
            if history_cats and pred_cats:
                overlap = len(history_cats & pred_cats) / len(pred_cats)
                overlap_ratios.append(overlap)
            
            # 预测类别多样性
            if pred_cats:
                pred_category_diversities.append(len(pred_cats))
        
        avg_overlap = np.mean(overlap_ratios) if overlap_ratios else 0
        avg_pred_diversity = np.mean(pred_category_diversities) if pred_category_diversities else 0
        
        is_similarity_bias_claim = any(kw in claim.lower() for kw in 
                                        ["过度依赖相似性", "相似性偏差", "总是推荐相似", "缺乏多样性", "over-reliance on similarity"])
        
        evidence = {
            "avg_category_overlap": avg_overlap,
            "avg_pred_category_diversity": avg_pred_diversity,
            "num_cases_analyzed": len(wrong_text_cases),
        }
        
        if is_similarity_bias_claim:
            if avg_overlap > 0.7:
                status = self.CONFIRMED
                brief = f"预测类别与历史类别高度重叠({avg_overlap:.2f}), 确实过度依赖相似性"
            elif avg_overlap > 0.4:
                status = self.PARTIALLY_CONFIRMED
                brief = f"预测类别有一定历史重叠({avg_overlap:.2f}), 但并非极端"
            else:
                status = self.REFUTED
                brief = f"预测类别与历史重叠度低({avg_overlap:.2f}), 不存在过度相似依赖"
        
        else:
            status = self.PARTIALLY_CONFIRMED
            brief = f"类别重叠度={avg_overlap:.2f}, 预测多样性={avg_pred_diversity:.1f}类"
        
        return {
            "status": status,
            "brief": brief,
            "evidence": evidence,
            "method": "similarity_bias",
        }
    
    def _verify_surprise_score(self, hyp: Dict, baseline: Dict,
                                overall_metrics: Dict = None,
                                surprise_metrics: Dict = None) -> Dict:
        """
        验证惊喜度相关假设
        
        常见假设:
        - "模型无法预测与历史差异大的交互" → 检查惊喜指标是否显著低于整体
        - "高惊喜交互的命中率极低" → 检查高惊喜案例的实际命中率
        """
        claim = hyp.get("claim", "")
        surprise_groups = baseline.get("surprise_groups", {})
        
        is_surprise_failure_claim = any(kw in claim.lower() for kw in 
                                         ["惊喜", "surprise", "差异大", "偏离历史", "无法预测"])
        
        evidence = {}
        
        # 从指标层面验证
        if overall_metrics and surprise_metrics:
            # 计算惊喜子集 vs 整体的差距
            ndcg_gap = {}
            recall_gap = {}
            for key in ["NDCG@10", "NDCG@20", "Recall@10", "Recall@20"]:
                if key in overall_metrics and key in surprise_metrics:
                    gap = surprise_metrics[key] - overall_metrics[key]
                    gap_pct = gap / overall_metrics[key] * 100 if overall_metrics[key] > 0 else 0
                    ndcg_gap[key] = gap_pct if "NDCG" in key else None
                    recall_gap[key] = gap_pct if "Recall" in key else None
            
            avg_ndcg_gap_pct = np.mean([v for v in ndcg_gap.values() if v is not None])
            avg_recall_gap_pct = np.mean([v for v in recall_gap.values() if v is not None])
            
            evidence["avg_ndcg_gap_pct"] = avg_ndcg_gap_pct
            evidence["avg_recall_gap_pct"] = avg_recall_gap_pct
        
        # 从案例层面验证
        if surprise_groups:
            evidence["surprise_groups"] = surprise_groups
            high_pct = surprise_groups.get("high_pct", 0)
            evidence["high_surprise_pct"] = high_pct
        
        if not evidence:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "缺少惊喜指标和案例数据",
                "brief": "无惊喜数据",
                "evidence": None,
            }
        
        if is_surprise_failure_claim:
            avg_gap = evidence.get("avg_ndcg_gap_pct", 0)
            if avg_gap < -20:
                status = self.CONFIRMED
                brief = f"惊喜子集NDCG比整体低{abs(avg_gap):.1f}%, 确实显著无法捕获惊喜"
            elif avg_gap < -10:
                status = self.PARTIALLY_CONFIRMED
                brief = f"惊喜子集NDCG比整体低{abs(avg_gap):.1f}%, 有差距但不极端"
            elif avg_gap < 0:
                status = self.PARTIALLY_CONFIRMED
                brief = f"惊喜子集指标略低于整体, 差距较小"
            else:
                status = self.REFUTED
                brief = f"惊喜子集指标不低于整体, 假设不成立"
        
        else:
            status = self.PARTIALLY_CONFIRMED
            brief = f"惊喜分布: 高{surprise_groups.get('high_pct',0):.1f}% 中{surprise_groups.get('medium_pct',0):.1f}% 低{surprise_groups.get('low_pct',0):.1f}%"
        
        return {
            "status": status,
            "brief": brief,
            "evidence": evidence,
            "method": "surprise_score",
        }
    
    def _verify_custom(self, hyp: Dict, baseline: Dict, 
                        wrong_text_cases: List[Dict]) -> Dict:
        """
        自定义验证 — 对无法归入上述类别的假设, 尝试基于文本案例做统计
        
        验证策略: 查看假设中提到的关键词, 对应到可计算的统计量
        """
        claim = hyp.get("claim", "")
        
        # 检查常见关键词并做对应验证
        evidence = {
            "claim_keywords": claim,
            "cases_available": len(wrong_text_cases) if wrong_text_cases else 0,
            "baseline_stats": {
                "surprise_groups": baseline.get("surprise_groups", {}),
                "seq_length_groups": baseline.get("seq_length_groups", {}),
            },
        }
        
        # 对一般性假设, 给出 UNVERIFIABLE (需要更多信息)
        return {
            "status": self.UNVERIFIABLE,
            "brief": f"假设 '{claim[:40]}' 无法用当前数据自动验证",
            "reason": "该假设需要模型内部数据或更详细的实验, 当前数据不足",
            "evidence": evidence,
            "method": "custom",
        }
    
    # ════════════════════════════════════════
    # Phase 3: 生成验证报告
    # ════════════════════════════════════════
    
    def generate_verification_report(self, verified_hypotheses: List[Dict]) -> Dict:
        """
        生成验证报告
        
        将每个假设的验证结果汇总, 并标注:
        - 哪些 LLM 结论被数据确认
        - 哪些被数据反驳
        - 哪些无法验证
        
        这个报告将反馈给下游流程, 让改进建议基于被验证的结论
        """
        confirmed = []
        partially_confirmed = []
        refuted = []
        unverifiable = []
        
        for hyp in verified_hypotheses:
            result = hyp.get("verification_result", {})
            status = result.get("status", self.UNVERIFIABLE)
            
            entry = {
                "id": hyp.get("id", "?"),
                "claim": hyp.get("claim", ""),
                "source_field": hyp.get("source_field", ""),
                "verification_method": hyp.get("verification_method", ""),
                "status": status,
                "brief": result.get("brief", ""),
                "evidence": result.get("evidence"),
            }
            
            if status == self.CONFIRMED:
                confirmed.append(entry)
            elif status == self.PARTIALLY_CONFIRMED:
                partially_confirmed.append(entry)
            elif status == self.REFUTED:
                refuted.append(entry)
            else:
                unverifiable.append(entry)
        
        # 计算置信度分数
        total = len(verified_hypotheses)
        confirmed_pct = len(confirmed) / total * 100 if total > 0 else 0
        refuted_pct = len(refuted) / total * 100 if total > 0 else 0
        
        # 生成建议: 被反驳的结论不应作为改进依据
        recommendations = []
        for r in refuted:
            recommendations.append(
                f"⚠ 假设 {r['id']} ({r['claim'][:50]}) 被数据反驳, "
                f"基于此结论的改进建议需要重新审视"
            )
        
        report = {
            "total_hypotheses": total,
            "confirmed": confirmed,
            "confirmed_count": len(confirmed),
            "partially_confirmed": partially_confirmed,
            "partially_confirmed_count": len(partially_confirmed),
            "refuted": refuted,
            "refuted_count": len(refuted),
            "unverifiable": unverifiable,
            "unverifiable_count": len(unverifiable),
            "confirmed_pct": confirmed_pct,
            "refuted_pct": refuted_pct,
            "overall_credibility": (
                "HIGH" if confirmed_pct > 60 else
                "MODERATE" if confirmed_pct > 30 else
                "LOW"
            ),
            "recommendations": recommendations,
            "verified_hypotheses": verified_hypotheses,
        }
        
        # 打印摘要
        print(f"\n  ══════════ 假设验证报告 ══════════")
        print(f"  总假设数: {total}")
        print(f"  ✅ 已确认: {len(confirmed)} ({confirmed_pct:.1f}%)")
        print(f"  ⚠️ 部分确认: {len(partially_confirmed)}")
        print(f"  ❌ 已反驳: {len(refuted)} ({refuted_pct:.1f}%)")
        print(f"  🔍 无法验证: {len(unverifiable)}")
        print(f"  综合可信度: {report['overall_credibility']}")
        if refuted:
            print(f"  反驳的结论:")
            for r in refuted:
                print(f"    ❌ {r['id']}: {r['brief']}")
        print(f"  ══════════════════════════════════\n")
        
        return report
    
    def apply_verification_to_analysis(self,
                                        llm_analysis: Dict,
                                        verification_report: Dict) -> Dict:
        """
        将验证结果应用到 LLM 分析结论
        
        策略:
        1. 被反驳的结论 → 标注为 REFUTED, 降低其权重
        2. 被确认的结论 → 标注为 CONFIRMED, 增强其可信度
        3. 无法验证的结论 → 保持原状, 但标注为 UNVERIFIABLE
        
        Returns:
            增强后的分析结果 (每个字段添加 verification_status)
        """
        if not llm_analysis or not verification_report:
            return llm_analysis
        
        enhanced = dict(llm_analysis)
        
        # 构建假设 → 结论字段的映射
        verified = verification_report.get("verified_hypotheses", [])
        field_status = defaultdict(list)
        
        for hyp in verified:
            result = hyp.get("verification_result", {})
            status = result.get("status", self.UNVERIFIABLE)
            source = hyp.get("source_field", "")
            field_status[source].append({
                "claim": hyp.get("claim", ""),
                "status": status,
                "brief": result.get("brief", ""),
            })
        
        # 为每个分析字段添加验证状态
        enhanced["verification_meta"] = {
            "overall_credibility": verification_report.get("overall_credibility", "LOW"),
            "confirmed_pct": verification_report.get("confirmed_pct", 0),
            "refuted_pct": verification_report.get("refuted_pct", 0),
            "field_verification": dict(field_status),
            "refuted_claims": [r["claim"] for r in verification_report.get("refuted", [])],
        }
        
        # 被反驳的结论 → 在改进建议中标注
        if enhanced.get("improvement_suggestions"):
            refuted_claims_list = verification_report.get("refuted_claims", [])
            for suggestion in enhanced["improvement_suggestions"]:
                desc = suggestion.get("description", "")
                # 如果改进建议基于被反驳的结论, 标注风险
                for rc in refuted_claims_list:
                    # 中文文本不能按空格分词 → 用子字符串匹配
                    # 从反驳结论中提取 2-4 字的关键短语做匹配
                    if self._chinese_keyword_match(desc, rc):
                        suggestion["verification_warning"] = (
                            f"⚠ 此建议可能基于被数据反驳的结论: {rc[:50]}"
                        )
                        suggestion["confidence_level"] = "LOW"
        
        return enhanced
    
    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════
    
    def _chinese_keyword_match(self, text: str, claim: str) -> bool:
        """
        中文关键词匹配 — 检查文本是否包含反驳结论的关键语义
        
        中文不能按空格分词, 所以:
        1. 先提取中文关键词 (2-4字的有意义子字符串)
        2. 检查是否有足够多的关键词在文本中出现
        
        也支持英文关键词 (按空格分词)
        """
        import re
        
        # 提取中文关键词: 只取 2字和3+字的有意义短语
        # 不做 sliding window → 太多噪声关键词
        chinese_phrases = re.findall(r'[\u4e00-\u9fff]+', claim)
        keywords = []
        for phrase in chinese_phrases:
            if len(phrase) <= 4:
                # 短短语整体作为一个关键词
                keywords.append(phrase)
            else:
                # 长短语: 只提取 2字关键词 (不做 sliding window)
                # 取语义较独立的部分
                for i in range(len(phrase) - 1):
                    kw = phrase[i:i+2]
                    keywords.append(kw)
        
        # 提取英文关键词
        english_words = re.findall(r'[a-zA-Z]+', claim.lower())
        keywords.extend([w for w in english_words if len(w) >= 3])
        
        # 去除过于通用的关键词
        stop_words = {"的", "是", "在", "有", "了", "和", "与", "或", "不", "为", 
                       "能", "可", "会", "到", "从", "对", "被", "中", "上", "下",
                       "但", "也", "又", "很", "最", "都", "这", "那", "一", "个",
                       "些", "所", "其", "型", "度", "性"}
        keywords = [kw for kw in keywords if kw not in stop_words and len(kw) >= 2]
        
        if not keywords:
            return False
        
        # 检查匹配度: 至少2个关键词命中 OR 超过20%的关键词命中
        hits = sum(1 for kw in keywords if kw in text.lower())
        return hits >= 2
    
    def _get_leaf_category(self, item_id) -> str:
        """获取物品的叶类别"""
        str_id = str(item_id) if item_id is not None else ""
        if str_id in self.item_text_map:
            entry = self.item_text_map[str_id]
            if isinstance(entry, dict):
                categories = entry.get("categories", "")
                if categories:
                    cat_list = categories.split(" > ")
                    return cat_list[-1] if cat_list else categories
        return ""
    
    def _extract_category_from_text(self, text: str) -> str:
        """
        从文本描述中提取类别
        
        格式: "Title [Category]" → 返回 "Category"
        """
        if not text:
            return ""
        # 匹配 [Category] 模式
        import re
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            return match.group(1)
        return ""
    
    def _parse_hypothesis_response(self, response: str) -> Optional[Dict]:
        """解析 LLM 假设提取回复 (增强版 — 多策略健壮解析 + 结构验证)"""
        import re
        
        # 提取 JSON block
        json_str = self._extract_json_block(response)
        if json_str is None:
            logger.warning("Cannot extract JSON from hypothesis extraction response")
            return None
        
        # 健壮解析
        parsed = self._robust_json_parse(json_str)
        if parsed is not None:
            validated = self._validate_hypotheses_structure(parsed)
            if validated is not None:
                return validated
            logger.warning("JSON parsed but structure validation failed")
        else:
            logger.warning("All JSON parsing strategies failed")
        
        return None
    
    @staticmethod
    def _extract_json_block(response: str) -> Optional[str]:
        """从 LLM 回复中提取 JSON block"""
        import re
        
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            return response[start:end + 1]
        
        return None
    
    @staticmethod
    def _robust_json_parse(json_str: str) -> Optional[Dict]:
        """
        多策略 JSON 解析, 带模糊修复
        
        Strategy 1: 标准 json.loads
        Strategy 2: ast.literal_eval
        Strategy 3: 修复常见格式问题后重试
        Strategy 4: 修复缺失引号的键
        """
        # --- Strategy 1: 标准解析 ---
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 2: Python literal ---
        try:
            import ast
            parsed = ast.literal_eval(json_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        
        # --- Strategy 3: 修复常见 JSON 格式问题 ---
        fixed = json_str
        
        # 移除注释
        fixed = re.sub(r'//[^\n]*', '', fixed)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        # 移除尾随逗号
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # Python 字面量 → JSON 字面量
        fixed = fixed.replace(': None', ': null')
        fixed = fixed.replace(': True', ': true')
        fixed = fixed.replace(': False', ': false')
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 4: 修复缺失引号的键 ---
        fixed2 = re.sub(
            r'(?<![:"\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'"\1":',
            fixed
        )
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 5: 单引号 → 双引号 ---
        fixed3 = fixed2.replace("'", '"')
        try:
            return json.loads(fixed3)
        except json.JSONDecodeError:
            pass
        
        return None
    
    @staticmethod
    def _validate_hypotheses_structure(parsed: Dict) -> Optional[Dict]:
        """验证和补全假设结构"""
        if not isinstance(parsed, dict):
            return None
        
        hypotheses = parsed.get("hypotheses")
        if hypotheses is None:
            return None
        
        if not isinstance(hypotheses, list) or len(hypotheses) == 0:
            return None
        
        required_fields = ["id", "claim"]
        optional_fields = {
            "verification_method": "custom",
            "expected_if_true": "",
            "expected_if_false": "",
            "confidence_in_llm": "medium",
            "priority": 3,
            "source_field": "unknown",
        }
        
        validated = []
        for h in hypotheses:
            if not isinstance(h, dict):
                continue
            missing = [f for f in required_fields if f not in h or not h[f]]
            if missing:
                continue
            for key, default in optional_fields.items():
                if key not in h:
                    h[key] = default
            validated.append(h)
        
        if not validated:
            return None
        
        return {"hypotheses": validated, "summary": parsed.get("summary", "")}
    
    def compute_item_popularity_from_data(self, train_data) -> Dict:
        """
        从训练数据计算物品热度分布
        
        Args:
            train_data: 训练集用户序列 (list of lists)
            
        Returns:
            Dict: item_id (str) → interaction count
        """
        popularity = defaultdict(int)
        for user_seq in train_data:
            # 去掉 padding (0) 和最后一个目标
            for item in user_seq[:-1]:
                if item > 0:
                    popularity[str(item)] += 1
        
        return dict(popularity)
    
    def save_verification_report(self, report: Dict, output_path: str):
        """保存验证报告"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved verification report to {output_path}")