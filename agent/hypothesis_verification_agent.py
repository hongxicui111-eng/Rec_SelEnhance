#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假设验证 Agent — 自主式假设验证框架 (重构版 v4)

重构要点:
  - prompts 从 prompts.py 导入, 不再在本文件硬编码
  - JSON 解析/清理/重试 使用 llm_utils.py 共享模块
  - 代码执行/数据注入 使用 script_executor.py 共享模块
  - 移除 KNOWN_COMPUTED_DATA / KNOWN_MODEL_PROBED_DATA 等预定义数据类型描述
    (LLM 应根据实际数据结构动态理解, 不需要硬编码映射)
  - 核心职责聚焦: 验证调度 (Plan → Data → Code → Execute → Analyze → Reflect → Retry)

工作流程:
  1. extract_hypotheses(): 从 LLM 分析中提取可验证假设
  2. verify_hypotheses(): 对每个假设执行完整的 Agent 验证流程:
     a. generate_verification_plan() — LLM 设计验证方案
     b. discover_and_load_data() — 发现并加载所需数据
     c. generate_verification_code() — LLM 写验证脚本
     d. execute_verification_code() — 执行脚本 (失败则修正)
     e. analyze_results() — LLM 解读结果, 判断假设
  3. generate_verification_report() — 生成汇总报告
  4. apply_verification_to_analysis() — 将验证结果反馈到分析中

与旧版兼容:
  - 保持与 HypothesisVerifier 相同的外部接口
  - core.py 只需替换 import, 无需修改调用方式
"""

import os
import json
import logging
import traceback
import re
import subprocess
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

# 导入基础设施模块
from .data_infrastructure import DataInfrastructure
from .task_scheduler import TaskScheduler, TaskStep, StepStatus, TaskStatus

# 导入共享工具模块
from .llm_utils import (
    extract_json_block, robust_json_parse, diagnose_json_error,
    parse_json_from_response, clean_code_response, clean_markdown_wrapper,
    LLMRetryHelper,
)
from .script_executor import (
    extract_output_path, DataInjector, ScriptExecutor,
)

# 导入 prompts (从 prompts.py, 不再在本文件硬编码)
from .prompts import (
    HYPOTHESIS_EXTRACTION_PROMPT_V2,
    VERIFICATION_PLAN_PROMPT,
    VERIFICATION_CODE_PROMPT,
    VERIFICATION_CODE_FIX_PROMPT,
    RESULT_ANALYSIS_PROMPT,
    HYPOTHESIS_JSON_FIX_PROMPT,
    DATA_ANALYSIS_PROMPT,
    DATA_ACQUISITION_STRATEGY_PROMPT,
    DATA_ACQUISITION_SCRIPT_PROMPT,
    DATA_ACQUISITION_FIX_PROMPT,
)

logger = logging.getLogger("rec_self_evolve.hypothesis_verification_agent")


class HypothesisVerificationAgent:
    """
    假设验证 Agent — 自主式假设验证框架
    
    核心职责: 验证调度 (Plan → Data → Code → Execute → Analyze → Reflect → Retry)
    
    不包含:
    - JSON 解析逻辑 (使用 llm_utils.py)
    - 代码执行逻辑 (使用 script_executor.py)
    - 预定义数据类型描述 (LLM 动态理解数据结构)
    
    工作流程:
    1. extract_hypotheses() — 从 LLM 分析提取假设
    2. verify_hypotheses() — 对每个假设执行完整的 Agent 验证流程:
       a. _generate_verification_plan() — LLM 设计验证方案
       b. _prepare_verification_data() — 发现并加载所需数据
       c. _generate_verification_code() — LLM 写验证脚本
       d. _execute_with_correction_loop() — 执行脚本 (失败则修正)
       e. _analyze_results() — LLM 解读结果, 判断假设
       f. _reflect_and_adjust() — 结果不理想时反思并调整
    3. generate_verification_report() — 生成汇总报告
    4. apply_verification_to_analysis() — 将验证结果反馈到分析中
    """
    
    # 验证状态枚举 (与旧版一致)
    CONFIRMED = "CONFIRMED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    REFUTED = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    
    # Agent 参数
    MAX_VERIFICATION_ROUNDS = 3  # 验证循环最大轮数
    MAX_CODE_FIX_ROUNDS = 5     # 代码修正最大轮次
    MAX_EXECUTION_TIMEOUT = 60  # 验证脚本执行超时 (秒)
    
    def __init__(self, llm_client, item_text_map: Dict = None,
                 project_root: str = None, data_dir: str = None,
                 log_dir: str = None):
        """
        Args:
            llm_client: LLMClient 实例
            item_text_map: 物品 ID → 元数据映射
            project_root: 项目根目录路径
            data_dir: 数据目录路径
            log_dir: 日志目录路径
        """
        self.llm = llm_client
        self.item_text_map = item_text_map or {}
        self.project_root = project_root or os.getcwd()
        self.log_dir = log_dir or os.path.join(self.project_root, "logs")
        
        # 数据基础设施
        self.data_infra = DataInfrastructure(
            project_root=self.project_root,
            data_dir=data_dir,
            log_dir=self.log_dir,
            llm_client=self.llm,
        )
        
        # 共享工具实例
        self.llm_retry = LLMRetryHelper(llm_client)
        self.executor = ScriptExecutor(
            project_root=self.project_root,
            timeout=self.MAX_EXECUTION_TIMEOUT,
        )
        self.injector = DataInjector(log_dir=self.log_dir)
        
        # 旧版 verifier 作为 fallback
        from .hypothesis_verifier import HypothesisVerifier
        self._fallback_verifier = HypothesisVerifier(llm_client, item_text_map)
    
    # ════════════════════════════════════════
    # Phase 1: 假设提取
    # ════════════════════════════════════════
    
    def extract_hypotheses(self, llm_analysis: Dict) -> Optional[List[Dict]]:
        """
        从 LLM 分析结果中提取可验证的假设
        
        Args:
            llm_analysis: LLM 案例分析的结果
            
        Returns:
            List of hypothesis dicts, or None if extraction failed
        """
        if not llm_analysis or not llm_analysis.get("parse_success"):
            logger.warning("Cannot extract hypotheses from invalid LLM analysis")
            return None
        
        # 构建 prompt
        analysis_json = json.dumps(llm_analysis, indent=2, ensure_ascii=False)
        if len(analysis_json) > 6000:
            truncated = dict(llm_analysis)
            suggestions = truncated.get("improvement_suggestions", [])
            if suggestions:
                truncated["improvement_suggestions"] = [
                    {k: v for k, v in s.items() if k != "structural_change_detail"}
                    for s in suggestions[:3]
                ]
            analysis_json = json.dumps(truncated, indent=2, ensure_ascii=False)
        
        # 数据盘点
        _preloaded_for_inventory = {}
        if self.item_text_map:
            _preloaded_for_inventory["item_text_map"] = self.item_text_map
        data_inventory_text = self.data_infra.format_inventory_for_prompt(
            preloaded_data=_preloaded_for_inventory if _preloaded_for_inventory else None
        )
        
        prompt = HYPOTHESIS_EXTRACTION_PROMPT_V2.format(
            llm_analysis_json=analysis_json,
            data_inventory=data_inventory_text,
        )
        
        # 主请求
        response = self.llm_retry.call_llm(
            prompt=prompt,
            system_content=(
                "你是一位严谨的数据科学家，擅长从分析结论中识别可验证的假设。"
                "你的目标是区分LLM的有数据支撑的结论和可能的主观臆断。"
                "每个假设必须是可以用数据统计来验证的。"
                "不要局限于固定验证类型，任何可以用数据回答的问题都是可验证假设。"
            ),
            temperature=0.3,
            max_tokens=2048,
        )
        if response is None:
            logger.error("Hypothesis extraction failed - no LLM response")
            return None
        
        # 解析 + 自动重试
        result = self._try_parse_hypotheses_with_retry(response)
        if result is not None:
            return result
        
        # --- 最后手段: 尝试旧版 verifier 作为 fallback ---
        logger.info("All extraction + retry strategies failed, trying fallback verifier")
        try:
            old_hypotheses = self._fallback_verifier.extract_hypotheses(llm_analysis)
            if old_hypotheses:
                logger.info(f"Fallback verifier recovered {len(old_hypotheses)} hypotheses")
                for h in old_hypotheses:
                    if "verification_thought" not in h:
                        h["verification_thought"] = h.get("verification_method", "custom")
                    if "data_needed" not in h:
                        h["data_needed"] = ["item_popularity", "category_metadata", "sequence_data"]
                return old_hypotheses
        except Exception as e2:
            logger.error(f"Fallback verifier also failed: {e2}")
        
        logger.warning("All hypothesis extraction strategies exhausted — returning None")
        return None
    
    def _try_parse_hypotheses_with_retry(self, first_response: str,
                                           max_retries: int = 2) -> Optional[List[Dict]]:
        """
        解析 LLM 响应, 失败时自动重试带错误反馈
        
        Args:
            first_response: 首次 LLM 响应
            max_retries: 最大重试次数
            
        Returns:
            List of hypothesis dicts, or None
        """
        response = first_response
        
        for attempt in range(1 + max_retries):
            parsed = self._parse_hypothesis_response(response)
            
            if parsed and parsed.get("hypotheses"):
                hypotheses = parsed["hypotheses"]
                if attempt > 0:
                    logger.info(f"Hypotheses recovered on retry #{attempt}: {len(hypotheses)} hypotheses")
                else:
                    logger.info(f"Extracted {len(hypotheses)} verifiable hypotheses")
                return hypotheses
            
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries + 1} attempts exhausted. "
                    "Raw response (first 500 chars): " +
                    (response[:500] if response else "None")
                )
                return None
            
            # 准备重试: 提取错误信息
            raw_truncated = (response[:3000] + "..." 
                             if response and len(response) > 3000 
                             else (response or "None"))
            parse_error = diagnose_json_error(response)
            
            # 提取部分假设用于保留
            partial_text = "无部分解析结果"
            if response:
                partial = self._extract_partial_hypotheses(response)
                if partial:
                    partial_text = json.dumps(
                        [{"id": p.get("id", "?"), "claim": p.get("claim", "")[:60]}
                         for p in partial],
                        ensure_ascii=False
                    )
            
            fix_prompt = HYPOTHESIS_JSON_FIX_PROMPT.format(
                raw_response_truncated=raw_truncated,
                parse_error=parse_error,
                partial_hypotheses_text=partial_text,
            )
            
            logger.info(f"Retry #{attempt + 1} with JSON fix prompt (parse error: {parse_error[:80]})")
            
            response = self.llm_retry.call_llm(
                prompt=fix_prompt,
                system_content=(
                    "你是一位严谨的数据科学家。你之前输出的假设 JSON 格式有误，"
                    "请重新输出，确保是严格合法的 JSON 格式。"
                ),
                temperature=0.2,
                max_tokens=2048,
            )
            
            if response is None:
                logger.error("No response from LLM during retry")
                return None
        
        return None
    
    def _extract_partial_hypotheses(self, response: str) -> Optional[List[Dict]]:
        """
        从完全不可解析的响应中尝试提取部分假设 (正则匹配假设级别 JSON 对象)
        """
        hypothesis_blocks = re.findall(
            r'\{\s*"id"\s*:\s*"[^"]*"\s*.*?"claim"\s*:\s*"[^"]*"[^}]*\}',
            response,
            re.DOTALL
        )
        
        recovered = []
        for block in hypothesis_blocks:
            parsed = robust_json_parse(block)
            if parsed and isinstance(parsed, dict) and "id" in parsed and "claim" in parsed:
                recovered.append(parsed)
        
        if recovered:
            logger.info(f"Partial extraction recovered {len(recovered)} hypotheses via regex")
        
        return recovered if recovered else None
    
    def _parse_hypothesis_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 假设提取回复 (使用 llm_utils 共享解析工具)
        """
        json_str = extract_json_block(response)
        if json_str is None:
            logger.warning("Cannot extract JSON from hypothesis extraction response")
            return None
        
        parsed = robust_json_parse(json_str)
        if parsed is not None:
            validated = self._validate_hypotheses_structure(parsed)
            if validated is not None:
                return validated
            else:
                logger.warning("JSON parsed but structure validation failed")
        else:
            logger.warning("All JSON parsing strategies failed")
        
        # 尝试从 raw response 中提取部分假设
        partial = self._extract_partial_hypotheses(response)
        if partial:
            logger.info(f"Recovered {len(partial)} hypotheses from partial extraction")
            return {"hypotheses": partial, "summary": "部分恢复的假设 (存在格式问题)"}
        
        return None
    
    @staticmethod
    def _validate_hypotheses_structure(parsed: Dict) -> Optional[Dict]:
        """
        验证和补全假设结构
        
        检查每个 hypothesis 的必要字段, 为缺失的字段设置默认值
        """
        if not isinstance(parsed, dict):
            return None
        
        hypotheses = parsed.get("hypotheses", parsed.get("hypothesis"))
        if hypotheses is None:
            logger.warning("No 'hypotheses' key in parsed response")
            return None
        
        if not isinstance(hypotheses, list):
            logger.warning("'hypotheses' is not a list")
            return None
        
        if len(hypotheses) == 0:
            logger.warning("Hypotheses list is empty")
            return None
        
        required_fields = ["id", "claim"]
        optional_fields = {
            "verification_thought": "",
            "verification_method": "custom",
            "data_needed": [],
            "expected_if_true": "",
            "expected_if_false": "",
            "confidence_in_llm": "medium",
            "priority": 3,
            "source_field": "unknown",
        }
        
        validated_hypotheses = []
        for h in hypotheses:
            if not isinstance(h, dict):
                continue
            
            missing = [f for f in required_fields if f not in h or not h[f]]
            if missing:
                logger.warning(f"Hypothesis missing required fields: {missing}")
                continue
            
            for key, default in optional_fields.items():
                if key not in h:
                    h[key] = default
            
            if isinstance(h.get("data_needed"), str):
                h["data_needed"] = [h["data_needed"]]
            elif not isinstance(h.get("data_needed"), list):
                h["data_needed"] = []
            
            validated_hypotheses.append(h)
        
        if not validated_hypotheses:
            logger.warning("No valid hypotheses after structure validation")
            return None
        
        result = {
            "hypotheses": validated_hypotheses,
            "summary": parsed.get("summary", ""),
        }
        
        logger.info(f"Structure validation passed: {len(validated_hypotheses)} hypotheses")
        return result
    
    # ════════════════════════════════════════
    # Phase 2: 验证执行 (Agent 核心流程)
    # ════════════════════════════════════════
    
    def verify_hypotheses(self,
                          hypotheses: List[Dict],
                          wrong_text_cases: List[Dict],
                          all_wrong_cases: List[Dict] = None,
                          model_config: Dict = None,
                          item_popularity: Dict = None,
                          overall_metrics: Dict = None,
                          surprise_metrics: Dict = None,
                          iteration_number: int = None) -> List[Dict]:
        """
        对每个假设运行自主验证
        
        Args:
            hypotheses: 提取的假设列表
            wrong_text_cases: LLM 使用的文本格式错误案例
            all_wrong_cases: 原始格式的错误案例
            model_config: 模型配置
            item_popularity: 物品热度分布
            overall_metrics: 整体评估指标
            surprise_metrics: 惊喜子集指标
            iteration_number: 主流程迭代轮次 (用于数据隔离)
            
        Returns:
            List of verified hypothesis dicts (每个包含 verification_result)
        """
        verified = []
        self._iteration_number = iteration_number
        
        # 预加载可用数据
        preloaded_data = self._prepare_preloaded_data(
            wrong_text_cases, all_wrong_cases, item_popularity,
            overall_metrics, surprise_metrics
        )
        
        for hyp in hypotheses:
            hyp_id = hyp.get("id", "H?")
            claim = hyp.get("claim", "")
            
            logger.info(f"Verifying {hyp_id}: {claim[:80]}...")
            print(f"  🔬 [Agent] 验证假设 {hyp_id}: {claim[:60]}...")
            
            try:
                result = self._verify_single_hypothesis(
                    hyp, preloaded_data, self._iteration_number
                )
                
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = result
                verified.append(verified_hyp)
                
                status = result.get("status", self.UNVERIFIABLE)
                symbol = {"CONFIRMED": "✅", "PARTIALLY_CONFIRMED": "⚠️",
                          "REFUTED": "❌", "UNVERIFIABLE": "🔍"}.get(status, "?")
                print(f"    {symbol} {hyp_id} → {status}: {result.get('brief', '')[:80]}")
                
            except Exception as e:
                logger.error(f"Agent verification failed for {hyp_id}: {e}")
                traceback.print_exc()
                
                # Fallback: 尝试旧版验证
                print(f"    ⚠ Agent verification failed, trying fallback...")
                result = self._try_fallback_verification(
                    hyp, wrong_text_cases, all_wrong_cases,
                    item_popularity, overall_metrics, surprise_metrics
                )
                
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = result
                verified.append(verified_hyp)
        
        return verified
    
    def _verify_single_hypothesis(self,
                                   hypothesis: Dict,
                                   preloaded_data: Dict,
                                   iteration_number: int = None) -> Dict:
        """
        对单个假设执行完整的 Agent 验证流程
        
        流程: 验证循环 (Plan → Data → Code → Execute → Analyze → Reflect → Retry)
        
        Args:
            iteration_number: 主流程迭代轮次 (用于数据隔离)
        
        Returns:
            verification result dict
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        verification_data = dict(preloaded_data)
        last_verification_plan = None
        last_execution_error = None
        
        for round_num in range(self.MAX_VERIFICATION_ROUNDS):
            print(f"\n    ════ 验证循环第 {round_num + 1}/{self.MAX_VERIFICATION_ROUNDS} 轮 ════")
            
            # ── 为本轮创建独立工作目录 (数据隔离) ──
            iter_prefix = f"iter_{iteration_number}" if iteration_number is not None else "iter_shared"
            workspace_dir = os.path.join(
                self.log_dir, "verification_scripts", 
                iter_prefix, hyp_id, f"round_{round_num + 1}"
            )
            os.makedirs(workspace_dir, exist_ok=True)
            print(f"    📁 工作目录: {workspace_dir}")
            
            # --- Step 1: 生成验证方案 ---
            print(f"    📋 [Step 1] 生成验证方案...")
            verification_plan = self._generate_verification_plan(
                hypothesis, verification_data, round_num, last_verification_plan, last_execution_error
            )
            if not verification_plan:
                return {
                    "status": self.UNVERIFIABLE,
                    "reason": "无法生成验证方案",
                    "brief": "验证方案生成失败",
                    "evidence": None,
                }
            last_verification_plan = verification_plan
            
            # --- Step 2: 准备数据 ---
            print(f"    📊 [Step 2] 准备验证数据...")
            verification_data = self._prepare_verification_data(
                hypothesis, verification_plan, verification_data
            )
            
            # --- Step 3: 生成验证代码 ---
            print(f"    💻 [Step 3] 生成验证代码...")
            code = self._generate_verification_code(
                hypothesis, verification_plan, verification_data, workspace_dir
            )
            if not code:
                last_execution_error = "代码生成失败"
                continue
            
            # --- Step 4: 执行验证代码 (带修正循环) ---
            print(f"    ⚡ [Step 4] 执行验证代码...")
            execution_result, exec_error = self.executor.execute_with_correction_loop(
                initial_code=code,
                hypothesis_id=hyp_id,
                llm_retry_helper=self.llm_retry,
                fix_prompt_template=VERIFICATION_CODE_FIX_PROMPT,
                injector=self.injector,
                verification_data=verification_data,
                output_file=os.path.join(workspace_dir, f"result_{hyp_id}.json"),
                max_rounds=self.MAX_CODE_FIX_ROUNDS,
                script_dir=workspace_dir,
            )
            
            if not execution_result:
                last_execution_error = exec_error or "代码执行失败"
                
                # 检测是否是缺失数据导致的错误
                missing_data_detected = self._detect_missing_data_from_error(
                    exec_error, hypothesis, verification_data
                )
                if missing_data_detected:
                    print(f"    🔄 检测到缺失数据，尝试获取后重试...")
                    acquired_data = self._acquire_missing_data_v2(
                        missing_data_detected, hypothesis, verification_data, workspace_dir
                    )
                    if acquired_data:
                        verification_data = acquired_data
                        continue
                
                print(f"    ⚠️ 执行失败且无法获取缺失数据: {(exec_error or '')[:100]}...")
                continue
            
            # --- Step 5: 分析结果 ---
            print(f"    🔍 [Step 5] 分析验证结果...")
            analysis_result = self._analyze_results(
                hypothesis, verification_plan, execution_result
            )
            
            # --- Step 6: 反思结果 ---
            status = analysis_result.get("status", self.UNVERIFIABLE)
            
            if status in [self.CONFIRMED, self.REFUTED]:
                print(f"    ✅ 假设验证完成: {status}")
                return analysis_result
            
            if round_num < self.MAX_VERIFICATION_ROUNDS - 1:
                print(f"    🤔 结果不理想 ({status})，反思原因并重试...")
                last_execution_error = f"结果分析: {analysis_result.get('brief', '')}"
                
                adjusted_plan = self._reflect_and_adjust(
                    hypothesis, verification_plan, analysis_result, verification_data
                )
                if adjusted_plan:
                    last_verification_plan = adjusted_plan
                    continue
            
            if status == self.UNVERIFIABLE:
                analysis_result["reason"] = analysis_result.get("reason", "") + \
                    f" (验证{round_num + 1}轮后仍无法确定)"
            
            return analysis_result
        
        return {
            "status": self.UNVERIFIABLE,
            "reason": f"验证{self.MAX_VERIFICATION_ROUNDS}轮后仍无法确定结论",
            "brief": "验证循环达到最大轮次",
            "evidence": {"last_plan": last_verification_plan, "last_error": last_execution_error},
        }
    
    # ════════════════════════════════════════
    # Step 1: 验证方案生成
    # ════════════════════════════════════════
    
    def _generate_verification_plan(self,
                                     hypothesis: Dict,
                                     preloaded_data: Dict,
                                     round_num: int = 0,
                                     last_plan: Optional[Dict] = None,
                                     last_error: Optional[str] = None) -> Optional[Dict]:
        """
        LLM 为假设生成验证方案 (支持反思重试)
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        data_inventory_text = self.data_infra.format_inventory_for_prompt(
            preloaded_data=preloaded_data
        )
        
        # 格式化已加载数据摘要
        loaded_data_summary = "## 已加载数据\n"
        for key, value in preloaded_data.items():
            if value is not None:
                if isinstance(value, list):
                    loaded_data_summary += f"- {key}: {len(value)} 项\n"
                elif isinstance(value, dict):
                    loaded_data_summary += f"- {key}: {len(value)} 个键\n"
                else:
                    loaded_data_summary += f"- {key}: {type(value).__name__}\n"
        
        # 构建反思上下文
        reflection_context = ""
        if round_num > 0 and last_error:
            reflection_context = f"""
## 上一轮验证信息 (第 {round_num} 轮尝试)
"""
            if last_plan:
                reflection_context += f"- 上一轮验证方案: {json.dumps(last_plan, ensure_ascii=False)[:500]}\n"
            reflection_context += f"- 上一轮问题/错误: {last_error}\n"
            reflection_context += """
请根据上一轮的问题调整验证方案，确保:
1. 如果是数据缺失问题: 明确列出需要的数据，使用可获取的数据或替代方案
2. 如果是代码执行问题: 调整验证方法，使用更稳健的实现
3. 如果是结果不理想: 调整统计方法或分析步骤
"""
        
        prompt = VERIFICATION_PLAN_PROMPT.format(
            hypothesis_id=hyp_id,
            hypothesis_claim=claim,
            verification_thought=hypothesis.get("verification_thought", ""),
            data_needed=json.dumps(hypothesis.get("data_needed", []), ensure_ascii=False),
            expected_if_true=hypothesis.get("expected_if_true", ""),
            expected_if_false=hypothesis.get("expected_if_false", ""),
            data_inventory=data_inventory_text,
            loaded_data_summary=loaded_data_summary,
        ) + reflection_context
        
        def _validate_plan(parsed: Dict) -> bool:
            if not isinstance(parsed, dict):
                return False
            plan = parsed.get("verification_plan", parsed)
            if not isinstance(plan, dict):
                return False
            if not plan.get("method_name") and not plan.get("analysis_steps"):
                logger.warning(f"Plan for {hyp_id} missing method_name and analysis_steps")
                return False
            return True
        
        result = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位数据科学家，擅长设计严谨的统计验证方案。"
                "方案必须具体、可执行，使用项目中可用的数据。"
                "如果某些数据不可用，方案应该说明替代方法。"
            ),
            temperature=0.3,
            max_tokens=2048,
            max_retries=2,
            additional_instructions=(
                "输出格式必须包含: verification_plan 对象 "
                "包含 method_name, method_description, data_sources, "
                "analysis_steps, statistical_method, confirm_criteria, refute_criteria"
            ),
            validate_func=_validate_plan,
        )
        
        if result is None:
            logger.error(f"Verification plan generation failed for {hyp_id}")
            return None
        
        if "verification_plan" not in result:
            result = {"verification_plan": result}
        
        return result
    
    # ════════════════════════════════════════
    # Step 2: 数据准备
    # ════════════════════════════════════════
    
    def _prepare_verification_data(self,
                                    hypothesis: Dict,
                                    verification_plan: Dict,
                                    preloaded_data: Dict) -> Dict:
        """
        为验证准备数据
        
        将已加载的数据 + 数据发现信息 + 模型信息 合为统一的数据描述,
        供代码生成使用
        
        注意: 不预定义任何数据处理逻辑。
        所有具体的数据获取由 LLM 在运行时动态生成代码实现。
        """
        verification_data = dict(preloaded_data)
        
        plan = verification_plan.get("verification_plan", {})
        data_needed = hypothesis.get("data_needed", [])
        
        # 将数据发现信息注入
        inventory_text = self.data_infra.format_inventory_for_prompt(
            preloaded_data=verification_data
        )
        verification_data["_data_inventory"] = inventory_text
        
        # 将模型信息注入
        model_info = self.data_infra.discover_model_info()
        verification_data["_model_info"] = model_info
        
        # 检查缓存中是否有之前计算/探测的数据
        for data_name in data_needed:
            cached = self.data_infra.load_from_cache(data_name)
            if cached is not None:
                verification_data[data_name] = cached
        
        return verification_data
    
    def _prepare_preloaded_data(self,
                                 wrong_text_cases: List[Dict],
                                 all_wrong_cases: List[Dict],
                                 item_popularity: Dict,
                                 overall_metrics: Dict,
                                 surprise_metrics: Dict) -> Dict:
        """准备从 core.py 传入的预加载数据"""
        preloaded = {}
        
        if wrong_text_cases:
            preloaded["wrong_text_cases"] = wrong_text_cases
        if item_popularity:
            preloaded["item_popularity"] = item_popularity
        if overall_metrics:
            preloaded["overall_metrics"] = overall_metrics
        if surprise_metrics:
            preloaded["surprise_metrics"] = surprise_metrics
        if self.item_text_map:
            preloaded["item_text_map"] = self.item_text_map
        
        return preloaded
    
    # ════════════════════════════════════════
    # Step 3: 验证代码生成
    # ════════════════════════════════════════
    
    def _generate_verification_code(self,
                                    hypothesis: Dict,
                                    verification_plan: Dict,
                                    verification_data: Dict,
                                    workspace_dir: str = None) -> Optional[str]:
        """
        LLM 根据验证方案和可用数据生成验证脚本
        
        新流程 (多步迭代):
        1. 分析数据需求 - 检查需要哪些数据，数据是否可用
        2. 如果有缺失数据，先获取数据
        3. 生成验证代码
        
        新增: 使用 DataInjector 进行数据注入 (不在本文件硬编码注入逻辑)
        新增: 移除 KNOWN_COMPUTED_DATA / KNOWN_MODEL_PROBED_DATA
              (LLM 应根据实际数据结构动态理解)
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        # 创建输出文件路径
        output_dir = workspace_dir or os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"result_{hyp_id}.json")
        
        # ── Step 1: 分析数据需求 ──
        print(f"    🔍 [Step 3a] 分析数据需求...")
        data_analysis = self._analyze_data_requirements(
            hypothesis, verification_plan, verification_data
        )
        
        missing_data = data_analysis.get("missing_data", []) if data_analysis else []
        can_proceed = data_analysis.get("can_proceed", True) if data_analysis else True
        
        # ── Step 2: 如果有缺失数据，先获取数据 ──
        if missing_data and not can_proceed:
            print(f"    📥 [Step 3b] 获取缺失数据: {missing_data}...")
            acquired_data = self._acquire_missing_data_v2(
                missing_data, hypothesis, verification_data, output_dir
            )
            if acquired_data:
                verification_data = acquired_data
                print(f"    ✅ 成功获取 {len(missing_data)} 项数据")
            else:
                print(f"    ⚠️ 无法获取缺失数据，将尝试使用可用数据继续...")
        
        # ── Step 3: 生成验证代码 ──
        print(f"    💻 [Step 3c] 生成验证代码...")
        
        # 构建可用数据描述 (动态, 不硬编码数据类型映射)
        available_data_desc = self._format_available_data_for_code(verification_data)
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        
        prompt = VERIFICATION_CODE_PROMPT.format(
            hypothesis_claim=claim,
            verification_plan_json=plan_json,
            available_data_description=available_data_desc,
            output_file_path=output_file,
            hypothesis_id=hyp_id,
        )
        
        system_content = (
            "你是一位 Python 数据科学家，擅长编写独立的数据验证脚本。"
            "代码必须稳健、能处理异常、只依赖标准库和 numpy/scipy。"
            "结果必须以 JSON 格式输出到指定文件。"
            "确保你使用 save_result() 函数将结果写入输出文件。"
        )
        
        max_attempts = 3
        last_error = ""
        
        for attempt in range(max_attempts):
            response = self.llm_retry.call_llm(
                prompt=prompt + (f"\n\n## 上一次尝试的问题\n{last_error}" if last_error else ""),
                system_content=system_content,
                temperature=0.2 if attempt == 0 else 0.3,
                max_tokens=4096,
                suppress_response_log=True,  # 代码响应不输出到日志
            )
            
            if response is None:
                logger.error(f"Verification code generation failed (attempt {attempt+1})")
                continue
            
            # 清理代码
            code = clean_code_response(response)
            
            # 代码质量验证
            validation_errors = self._validate_verification_code(code, output_file)
            
            if not validation_errors:
                # 使用 DataInjector wrap_script (最小注入: DATA_FILE + OUTPUT_FILE + save_result)
                # LLM 自主决定 import 和数据加载逻辑
                code = self.injector.wrap_script(code, verification_data, output_file)
                return code
            
            last_error = "; ".join(validation_errors)
            logger.warning(f"Code validation failed (attempt {attempt+1}): {last_error}")
            
            if attempt < max_attempts - 1:
                prompt += (
                    f"\n\n## 修正要求\n上一版代码存在以下问题:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                )
        
        logger.error(f"Verification code generation failed after {max_attempts} attempts")
        return None
    
    @staticmethod
    def _validate_verification_code(code: str, expected_output_file: str) -> List[str]:
        """验证生成的代码是否满足基本质量要求"""
        errors = []
        
        # LLM 必须自主编写数据加载逻辑 (使用 DATA_FILE)
        has_data_loading = ("DATA_FILE" in code or "json.load" in code)
        if not has_data_loading:
            errors.append("代码中未包含数据加载逻辑 (DATA_FILE / json.load)")
        
        # LLM 必须使用 save_result() 输出结果
        has_save = ("save_result" in code)
        if not has_save:
            errors.append("代码中未调用 save_result() 保存结果")
        
        # 代码必须有统计计算
        if not any(kw in code for kw in ["Counter", "statistics", "mean", "sum", "len",
                                          "count", "ratio", "histogram", "统计"]):
            if "statistics" not in code and "analysis" not in code.lower():
                errors.append("代码中未检测到统计计算逻辑")
        
        # 代码必须有异常处理
        if "try:" not in code or "except" not in code:
            errors.append("代码缺少异常处理 (try/except)")
        
        # 代码必须包含 import 语句 (LLM 自主决定导入)
        if "import " not in code:
            errors.append("代码缺少 import 语句")
        
        return errors
    
    def _format_available_data_for_code(self, verification_data: Dict) -> str:
        """
        格式化可用数据描述 — 动态, 不硬编码数据类型映射
        
        关键变化: 移除 KNOWN_COMPUTED_DATA / KNOWN_MODEL_PROBED_DATA
        LLM 应根据数据的实际内容 (统计摘要、样本、字段名) 来理解数据,
        不需要预定义的"类别重叠统计"等类型描述。
        
        策略: 对每个数据变量, 展示其:
        - 类型 (list/dict/str/...)
        - 大小 (长度/键数)
        - 样本 (前2条记录或前5个键)
        - 如果是 dict, 展示 top-level keys 和部分值
        """
        lines = ["以下数据已经作为 Python 变量注入到脚本中, 直接使用即可:", ""]
        
        for key, value in verification_data.items():
            if key.startswith("_"):  # 内部字段 (如 _data_inventory, _model_info)
                lines.append(f"- `{key}`: 内部元数据 (仅供参考)")
                continue
            
            if isinstance(value, list):
                lines.append(f"- `{key}`: List, 长度 {len(value)}")
                if value and isinstance(value[0], dict):
                    sample_keys = list(value[0].keys())[:8]
                    lines.append(f"  每个元素的 keys: {sample_keys}")
                    if len(value[0]) > 0:
                        sample_str = json.dumps(value[0], ensure_ascii=False)[:200]
                        lines.append(f"  样本: {sample_str}")
            elif isinstance(value, dict):
                lines.append(f"- `{key}`: Dict, {len(value)} 条记录")
                if value:
                    # 展示 top-level keys
                    top_keys = list(value.keys())[:8]
                    lines.append(f"  顶层 keys: {top_keys}")
                    # 展示每个 key 的类型和大小
                    for sk in top_keys[:5]:
                        sv = value[sk]
                        if isinstance(sv, list):
                            lines.append(f"  - {sk}: List ({len(sv)} 项)")
                        elif isinstance(sv, dict):
                            lines.append(f"  - {sk}: Dict ({len(sv)} 条)")
                        elif isinstance(sv, (int, float, str)):
                            lines.append(f"  - {sk}: {sv}")
                        else:
                            lines.append(f"  - {sk}: {type(sv).__name__}")
            elif isinstance(value, str):
                lines.append(f"- `{key}`: 文件路径字符串 `{value}`")
            elif value is None:
                lines.append(f"- `{key}`: None (不可用)")
            else:
                lines.append(f"- `{key}`: {type(value).__name__}")
        
        return "\n".join(lines)
    
    # ════════════════════════════════════════
    # Step 3a: 数据需求分析 (新增)
    # ════════════════════════════════════════
    
    def _analyze_data_requirements(self,
                                    hypothesis: Dict,
                                    verification_plan: Dict,
                                    verification_data: Dict) -> Optional[Dict]:
        """
        分析验证所需的数据需求，检查数据是否可用
        
        返回:
        - data_requirements: 数据需求列表
        - missing_data: 缺失数据列表
        - can_proceed: 是否可以继续生成验证代码
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        available_data_desc = self._format_available_data_for_code(verification_data)
        
        # 获取数据盘点信息
        data_inventory = verification_data.get("_data_inventory", "无数据盘点信息")
        
        prompt = DATA_ANALYSIS_PROMPT.format(
            hypothesis_claim=claim,
            verification_plan_json=plan_json,
            available_data_description=available_data_desc,
            data_inventory=data_inventory,
        )
        
        def _validate_analysis(parsed: Dict) -> bool:
            if not isinstance(parsed, dict):
                return False
            if "data_requirements" not in parsed and "missing_data" not in parsed:
                return False
            return True
        
        result = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位数据分析师，擅长分析验证假设需要哪些数据。"
                "请仔细分析验证方案，确定需要哪些数据，并检查这些数据是否可用。"
            ),
            temperature=0.3,
            max_tokens=2048,
            max_retries=2,
            validate_func=_validate_analysis,
        )
        
        if result is None:
            logger.warning(f"Data analysis failed for {hyp_id}")
            # 如果分析失败，假设可以继续，让 LLM 自己处理
            return {"can_proceed": True, "missing_data": [], "data_requirements": []}
        
        return result
    
    # ════════════════════════════════════════
    # Step 3b: 数据获取 (多步迭代版本)
    # ════════════════════════════════════════
    
    def _acquire_missing_data_v2(self,
                                  missing_data: List[str],
                                  hypothesis: Dict,
                                  verification_data: Dict,
                                  workspace_dir: str) -> Optional[Dict]:
        """
        获取缺失的数据 - 策略驱动 + 单脚本版本
        
        流程:
        1. 规划获取策略 (分步蓝图，不生成代码)
        2. 为每个策略生成一个**完整的连贯脚本** (而非多个碎片脚本)
        3. 保存脚本到工作目录
        4. 执行脚本 (subprocess, 支持 GPU/CUDA)
        5. 修正循环 (如果执行失败)
        6. 将获取到的数据添加到 verification_data
        
        重构要点 (vs 旧版 per-step 模式):
        - 旧版: 为每个子步骤生成独立脚本，subprocess 之间无状态传递
        - 新版: 为整个策略生成一个连贯脚本，Python 变量自然传递
        - 好处: model_instance 可在 step2 创建 → step3 加载权重 → step6 注册 hook
                 不再需要每个脚本重复构建模型/数据
        """
        if not missing_data:
            return verification_data
        
        updated_data = dict(verification_data)
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        # ── 创建数据获取专用工作目录 ──
        acquire_dir = os.path.join(workspace_dir, "data_acquisition")
        os.makedirs(acquire_dir, exist_ok=True)
        print(f"    📁 数据获取工作目录: {acquire_dir}")
        
        # ── Step 1: 规划获取策略 ──
        print(f"    🎯 [Step 1] 规划数据获取策略...")
        strategy = self._plan_acquisition_strategy(
            missing_data, hypothesis, verification_data, acquire_dir
        )
        
        if not strategy:
            logger.warning(f"Acquisition strategy planning failed for {hyp_id}")
            # 回退: 每个数据一个策略 (但仍然生成单脚本，不分步)
            strategy = {"strategies": [
                {"data_name": d, "data_type": "unknown", "difficulty": "medium",
                 "acquisition_steps": [{"step_name": f"acquire_{d}",
                                        "description": f"获取数据 {d}",
                                        "output": d,
                                        "dependencies": []}],
                 "total_steps": 1, "priority": "中"}
                for d in missing_data
            ], "execution_order": missing_data, "estimated_total_steps": len(missing_data)}
        
        # ── Step 2: 按策略顺序为每个策略生成单脚本 ──
        strategies = strategy.get("strategies", [])
        execution_order = strategy.get("execution_order", missing_data)
        
        # 按 execution_order 排序
        ordered_strategies = []
        for data_name in execution_order:
            for s in strategies:
                if s.get("data_name") == data_name:
                    ordered_strategies.append(s)
                    break
        
        # 如果排序后有遗漏，补充
        for s in strategies:
            if s not in ordered_strategies:
                ordered_strategies.append(s)
        
        # ── Step 2: 按策略顺序获取数据 (每策略一个脚本) ──
        failed_data_names = []   # 记录失败的数据名 (用于重规划)
        failed_reasons = {}      # 记录失败原因
        
        for strat_idx, strat in enumerate(ordered_strategies):
            data_name = strat.get("data_name", f"data_{strat_idx}")
            data_type = strat.get("data_type", "unknown")
            difficulty = strat.get("difficulty", "medium")
            total_steps = strat.get("total_steps", len(strat.get("acquisition_steps", [])))
            
            print(f"    📥 [Step 2.{strat_idx+1}] 获取数据 '{data_name}' "
                  f"(类型: {data_type}, 难度: {difficulty}, "
                  f"策略含 {total_steps} 个子步骤 → 合并为单脚本)")
            
            # ── 为整个策略生成一个连贯脚本 ──
            script_file = os.path.join(acquire_dir, f"acquire_{data_name}.py")
            output_file = os.path.join(acquire_dir, f"acquired_{data_name}.json")
            
            code = self._generate_acquisition_script(
                strategy=strat,
                verification_data=updated_data,
                output_file=output_file,
            )
            
            if not code:
                print(f"    ❌ 获取脚本生成失败: {data_name}")
                continue
            
            # ── 保存脚本到文件 ──
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)
            logger.info(f"Saved acquisition script: {script_file}")
            
            # ── 执行获取脚本 ──
            success, result, error = self._execute_acquisition_script(
                script_file, output_file
            )
            
            # ── 修正循环 ──
            max_fix_rounds = 3 if difficulty in ["complex", "very_complex"] else 2
            
            for fix_round in range(max_fix_rounds):
                if success and result is not None:
                    break
                
                if not error:
                    break
                
                print(f"    🔄 修正获取脚本 (第 {fix_round+1} 次)...")
                
                fix_prompt = DATA_ACQUISITION_FIX_PROMPT.format(
                    step_name=data_name,
                    original_code=self.injector.extract_core_code(code),
                    error_output=error[:1500],
                    available_data_description=self._format_available_data_for_code(updated_data),
                )
                
                fixed_response = self.llm_retry.call_llm(
                    prompt=fix_prompt,
                    system_content=(
                        "你是一位 Python 数据工程师，擅长根据错误信息修正数据获取脚本。"
                        "只修正导致错误的部分，保持获取逻辑不变。"
                        "如果数据确实无法获取，设置 success=false 并说明原因。"
                        "绝对不要模拟或假设数据。"
                    ),
                    temperature=0.1,
                    max_tokens=4096,
                    suppress_response_log=True,  # 代码响应不输出到日志
                )
                
                if not fixed_response:
                    continue
                
                from .llm_utils import clean_code_response
                code = clean_code_response(fixed_response)
                code = self.injector.wrap_script(code, updated_data, output_file)
                
                # 保存修正后的脚本
                with open(script_file, 'w', encoding='utf-8') as f:
                    f.write(code)
                
                success, result, error = self._execute_acquisition_script(
                    script_file, output_file
                )
            
            # ── 处理执行结果 ──
            acquired_ok = False
            validation_issues = []
            
            if success and result is not None:
                # ── Bug fix: 原逻辑 `not step_data.get("success", False)` 是反转的 ──
                # 正确逻辑: success=False 或 success 不存在时视为失败;
                #           success=True 或 success 不存在但数据非空时视为成功
                step_data = result.get("data", result)
                
                # 判断脚本是否明确标记失败
                explicitly_failed = False
                if isinstance(step_data, dict):
                    success_flag = step_data.get("success")
                    if success_flag is False:
                        explicitly_failed = True
                
                if not explicitly_failed:
                    # ── 数据质量校验 ──
                    validation_issues = self._validate_acquired_data(
                        data_name, step_data, data_type
                    )
                    
                    if validation_issues:
                        print(f"    ⚠️ 数据 '{data_name}' 获取成功但质量有问题:")
                        for issue in validation_issues:
                            print(f"      - {issue}")
                        # 仍然接受数据，但标记问题
                        acquired_ok = True
                    else:
                        acquired_ok = True
                    
                    if acquired_ok:
                        # 尝试提取策略预期的 output 名称
                        expected_output = data_name
                        acquisition_steps = strat.get("acquisition_steps", [])
                        if acquisition_steps:
                            expected_output = acquisition_steps[-1].get("output", data_name)
                        
                        updated_data[expected_output] = step_data
                        
                        # 保存到 DataInfrastructure 缓存
                        self.data_infra.save_to_cache(data_name, step_data)
                        
                        print(f"    ✅ 成功获取数据 '{expected_output}'")
                else:
                    # 脚本明确标记 success=False
                    error_msg = step_data.get("error", "未知原因") if isinstance(step_data, dict) else "数据格式错误"
                    print(f"    ❌ 数据 '{data_name}' 获取失败 (脚本返回 success=False): {error_msg}")
                    failed_data_names.append(data_name)
                    failed_reasons[data_name] = error_msg
            else:
                # 脚本执行本身就失败了 (subprocess 返回非零)
                print(f"    ❌ 获取脚本执行失败: {(error or '')[:100]}")
                failed_data_names.append(data_name)
                failed_reasons[data_name] = (error or "执行失败")[:200]
        
        # ── Step 3: 失败数据的重规划 ──
        if failed_data_names:
            print(f"    🔁 [Step 3] 重规划失败数据: {failed_data_names}")
            replan_strategy = self._plan_acquisition_strategy(
                failed_data_names, hypothesis, updated_data, acquire_dir
            )
            
            if replan_strategy:
                replan_strategies = replan_strategy.get("strategies", [])
                for strat in replan_strategies:
                    data_name = strat.get("data_name", "")
                    if data_name in updated_data:
                        # 已经获取成功，跳过
                        continue
                    
                    print(f"    📥 [Step 3-重试] 用替代策略获取 '{data_name}'...")
                    
                    script_file = os.path.join(acquire_dir, f"acquire_{data_name}_replan.py")
                    output_file = os.path.join(acquire_dir, f"acquired_{data_name}_replan.json")
                    
                    code = self._generate_acquisition_script(
                        strategy=strat,
                        verification_data=updated_data,
                        output_file=output_file,
                    )
                    
                    if not code:
                        continue
                    
                    with open(script_file, 'w', encoding='utf-8') as f:
                        f.write(code)
                    
                    success, result, error = self._execute_acquisition_script(
                        script_file, output_file
                    )
                    
                    # 修正循环 (1次)
                    if not success and error:
                        fix_prompt = DATA_ACQUISITION_FIX_PROMPT.format(
                            step_name=data_name,
                            original_code=self.injector.extract_core_code(code),
                            error_output=error[:1500],
                            available_data_description=self._format_available_data_for_code(updated_data),
                        )
                        fixed_response = self.llm_retry.call_llm(
                            prompt=fix_prompt,
                            system_content=(
                                "你是一位 Python 数据工程师，擅长根据错误信息修正数据获取脚本。"
                                "只修正导致错误的部分，保持获取逻辑不变。"
                                "绝对不要模拟或假设数据。"
                            ),
                            temperature=0.1,
                            max_tokens=4096,
                            suppress_response_log=True,  # 代码响应不输出到日志
                        )
                        if fixed_response:
                            from .llm_utils import clean_code_response
                            code = clean_code_response(fixed_response)
                            code = self.injector.wrap_script(code, updated_data, output_file)
                            with open(script_file, 'w', encoding='utf-8') as f:
                                f.write(code)
                            success, result, error = self._execute_acquisition_script(
                                script_file, output_file
                            )
                    
                    if success and result is not None:
                        step_data = result.get("data", result)
                        explicitly_failed = isinstance(step_data, dict) and step_data.get("success") is False
                        if not explicitly_failed:
                            validation_issues = self._validate_acquired_data(
                                data_name, step_data, strat.get("data_type", "unknown")
                            )
                            updated_data[data_name] = step_data
                            self.data_infra.save_to_cache(data_name, step_data)
                            print(f"    ✅ 重规划成功获取 '{data_name}'")
                        else:
                            error_msg = step_data.get("error", "重规划也失败")
                            print(f"    ❌ 重规划获取 '{data_name}' 也失败: {error_msg}")
        
        # ── 检查获取结果 ──
        acquired_count = sum(1 for d in missing_data if d in updated_data 
                           and updated_data[d] is not None)
        
        # 输出详细的结果摘要
        result_summary_lines = []
        for d in missing_data:
            if d in updated_data and updated_data[d] is not None:
                result_summary_lines.append(f"  ✅ {d}")
            else:
                reason = failed_reasons.get(d, "未知原因")
                result_summary_lines.append(f"  ❌ {d} — 原因: {reason}")
        print(f"    📊 数据获取结果 ({acquired_count}/{len(missing_data)}):\n" + 
              "\n".join(result_summary_lines))
        
        if acquired_count > 0:
            return updated_data
        else:
            logger.warning(f"No data acquired for {hyp_id}")
            return None
    
    # ════════════════════════════════════════
    # 数据获取辅助方法
    # ════════════════════════════════════════
    
    def _plan_acquisition_strategy(self,
                                    missing_data: List[str],
                                    hypothesis: Dict,
                                    verification_data: Dict,
                                    acquire_dir: str) -> Optional[Dict]:
        """
        规划数据获取策略 — 不直接生成代码，先分析获取路径
        """
        claim = hypothesis.get("claim", "")
        hyp_id = hypothesis.get("id", "H?")
        
        # 构建缺失数据描述
        missing_desc = "\n".join([f"- {d}: 需要获取 (原因: 验证假设需要此数据)" for d in missing_data])
        
        available_data_desc = self._format_available_data_for_code(verification_data)
        data_inventory = verification_data.get("_data_inventory", "无数据盘点信息")
        model_info = verification_data.get("_model_info", {})
        
        prompt = DATA_ACQUISITION_STRATEGY_PROMPT.format(
            hypothesis_claim=claim,
            missing_data_description=missing_desc,
            data_inventory=data_inventory,
            model_info=json.dumps(model_info, ensure_ascii=False)[:2000],
            available_data_description=available_data_desc,
        )
        
        def _validate_strategy(parsed: Dict) -> bool:
            if not isinstance(parsed, dict):
                return False
            if "strategies" not in parsed:
                return False
            return True
        
        result = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位数据获取策略规划师，擅长分析如何从项目数据和模型中获取所需数据。"
                "请先理解数据来源和获取路径，再制定策略。"
                "对于模型内部数据，需要考虑: 加载模型 → 注册 hooks → 运行推理 → 提取输出。"
                "不要直接生成代码，只规划获取步骤。"
            ),
            temperature=0.3,
            max_tokens=2048,
            max_retries=2,
            validate_func=_validate_strategy,
        )
        
        if result is None:
            logger.warning(f"Acquisition strategy planning failed for {hyp_id}")
            return None
        
        # 保存策略到文件
        strategy_file = os.path.join(acquire_dir, "acquisition_strategy.json")
        with open(strategy_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved acquisition strategy: {strategy_file}")
        
        return result
    
    def _generate_acquisition_script(self,
                                      strategy: Dict,
                                      verification_data: Dict,
                                      output_file: str) -> Optional[str]:
        """
        为整个策略生成一个连贯的获取脚本
        
        与旧版 _generate_acquisition_step_code 的区别:
        - 旧版: 为每个子步骤生成独立脚本，步骤间无法传递 Python 变量
        - 新版: 将策略的所有步骤合并为一个脚本，变量自然流转
        
        策略蓝图的 acquisition_steps 作为 LLM 的参考，帮助它理解步骤顺序，
        但生成的代码是一个连贯的 pipeline，不是多个独立脚本。
        
        返回经过 DataInjector wrap_script 的完整脚本
        """
        data_name = strategy.get("data_name", "unknown")
        data_type = strategy.get("data_type", "unknown")
        difficulty = strategy.get("difficulty", "medium")
        
        available_data_desc = self._format_available_data_for_code(verification_data)
        data_inventory = verification_data.get("_data_inventory", "无数据盘点信息")
        model_info = verification_data.get("_model_info", {})
        
        prompt = DATA_ACQUISITION_SCRIPT_PROMPT.format(
            data_name=data_name,
            data_type=data_type,
            difficulty=difficulty,
            strategy_context=json.dumps(strategy, ensure_ascii=False),
            data_inventory=data_inventory,
            model_info=json.dumps(model_info, ensure_ascii=False)[:2000],
            available_data_description=available_data_desc,
        )
        
        response = self.llm_retry.call_llm(
            prompt=prompt,
            system_content=(
                "你是一位 Python 数据工程师，擅长编写数据获取脚本。"
                "你需要将策略规划中的多个步骤合并为一个连贯的 Python 脚本。"
                "步骤之间通过 Python 变量传递中间结果 (如 model_instance, test_dataset)，"
                "而不是分成多个独立脚本。"
                "对于模型内部数据，使用 torch.load() + PyTorch hooks。"
                "绝对不要模拟或假设数据 — 如果无法获取，标记失败。"
                "确保使用 save_result() 将结果保存到 JSON 文件。"
            ),
            temperature=0.2,
            max_tokens=4096,
            suppress_response_log=True,  # 代码响应不输出到日志
        )
        
        if response is None:
            return None
        
        from .llm_utils import clean_code_response
        code = clean_code_response(response)
        code = self.injector.wrap_script(code, verification_data, output_file)
        
        return code
    
    def _validate_acquired_data(self,
                                 data_name: str,
                                 data_value: Any,
                                 data_type: str) -> List[str]:
        """
        校验获取到的数据质量
        
        检查项:
        1. 是否为 None / 空
        2. 是否为明显的模拟/伪造数据 (全零、全相同值、固定小数)
        3. 是否缺少关键字段 (对于 model_internal 类型)
        4. 数据量是否异常 (list 长度为 0 或 1)
        
        Returns:
            List[str] — 校验问题列表 (空列表 = 校验通过)
        """
        issues = []
        
        # ── 检查 1: None / 空 ──
        if data_value is None:
            issues.append("数据值为 None")
            return issues
        
        if isinstance(data_value, dict):
            # 检查空 dict
            if len(data_value) == 0:
                issues.append("数据为空字典")
            
            # ── 检查 2: 模拟数据特征 ──
            # 检查是否所有数值都相同 (典型的模拟数据)
            numeric_values = []
            for v in data_value.values():
                if isinstance(v, (int, float)):
                    numeric_values.append(v)
                elif isinstance(v, list) and len(v) > 0:
                    # 检查列表内是否全相同
                    first = v[0]
                    if all(x == first for x in v) and len(v) > 3:
                        issues.append(f"列表数据全为相同值 {first} (疑似模拟数据)")
            
            if numeric_values and len(numeric_values) > 3:
                if all(v == numeric_values[0] for v in numeric_values):
                    issues.append(f"所有数值均为 {numeric_values[0]} (疑似模拟数据)")
            
            # ── 检查 3: 关键字段缺失 ──
            if data_type == "model_internal":
                # 模型内部数据应有 prediction_scores / embeddings 等字段
                expected_keys = ["prediction_scores", "scores", "embeddings", 
                                 "attention_weights", "item_scores", "outputs"]
                if not any(k in data_value for k in expected_keys):
                    issues.append(f"模型内部数据缺少预期字段 (期望含: {expected_keys})")
            
            # ── 检查 success 字段 ──
            if data_value.get("success") is True:
                # success=True 但 data 字段为空
                inner_data = data_value.get("data")
                if inner_data is None or (isinstance(inner_data, dict) and len(inner_data) == 0):
                    issues.append("success=True 但 data 字段为空")
        
        elif isinstance(data_value, list):
            # ── 检查 4: 空列表 ──
            if len(data_value) == 0:
                issues.append("数据为空列表")
            elif len(data_value) == 1:
                issues.append(f"数据列表只有 1 条记录 (可能不完整)")
            
            # ── 检查 2: 全相同值 ──
            if len(data_value) > 3 and all(v == data_value[0] for v in data_value):
                issues.append(f"列表所有元素均为 {data_value[0]} (疑似模拟数据)")
        
        elif isinstance(data_value, str):
            # 字符串型数据 — 检查是否为错误信息伪装成数据
            if "模拟" in data_value or "mock" in data_value.lower() or "dummy" in data_value.lower():
                issues.append(f"数据含模拟/mock/dummy 关键词 (疑似非真实数据)")
        
        # ── 检查 5: 数据类型不匹配 ──
        if data_type == "model_internal" and not isinstance(data_value, (dict, list)):
            issues.append(f"模型内部数据应为 dict/list，实际为 {type(data_value).__name__}")
        
        return issues
    
    def _execute_acquisition_script(self,
                                     script_path: str,
                                     output_file: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        执行数据获取脚本 — 使用 subprocess (不用 exec)
        
        使用 DataInfrastructure.execute_script() 或直接 subprocess.run
        """
        try:
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=self.data_infra.DEFAULT_SCRIPT_TIMEOUT,
                cwd=str(self.project_root),
            )
            
            if result.returncode == 0:
                # 从输出文件读取结果
                if os.path.exists(output_file):
                    try:
                        with open(output_file, 'r') as f:
                            data = json.load(f)
                        return True, data, None
                    except json.JSONDecodeError as e:
                        return False, None, f"JSON 解析失败: {e}"
                
                # 尝试从 stdout 解析 JSON
                if result.stdout.strip():
                    try:
                        data = json.loads(result.stdout)
                        return True, data, None
                    except json.JSONDecodeError:
                        return True, {"success": True, "data": result.stdout}, None
                
                return True, None, None
            else:
                error_msg = result.stderr or "未知错误"
                return False, None, error_msg
                
        except subprocess.TimeoutExpired:
            return False, None, f"执行超时 ({self.data_infra.DEFAULT_SCRIPT_TIMEOUT}秒)"
        except Exception as e:
            return False, None, str(e)
    
    def _format_acquired_data_desc(self,
                                    step_outputs: Dict,
                                    updated_data: Dict) -> str:
        """
        格式化已获取数据描述 — 供后续步骤参考
        """
        lines = ["已获取的数据:"]
        
        for key, value in step_outputs.items():
            if isinstance(value, list):
                lines.append(f"- `{key}`: List, 长度 {len(value)}")
            elif isinstance(value, dict):
                lines.append(f"- `{key}`: Dict, {len(value)} 条记录")
            elif isinstance(value, str):
                lines.append(f"- `{key}`: 字符串")
            else:
                lines.append(f"- `{key}`: {type(value).__name__}")
        
        # 也列出 updated_data 中非 None 的数据
        for key, value in updated_data.items():
            if key not in step_outputs and value is not None and not key.startswith("_"):
                if isinstance(value, list):
                    lines.append(f"- `{key}`: List, 长度 {len(value)} (已有)")
                elif isinstance(value, dict):
                    lines.append(f"- `{key}`: Dict, {len(value)} 条记录 (已有)")
        
        return "\n".join(lines)
    
    # ════════════════════════════════════════
    # Step 5: 结果分析
    # ════════════════════════════════════════
    
    def _analyze_results(self,
                         hypothesis: Dict,
                         verification_plan: Dict,
                         execution_result: Dict) -> Dict:
        """
        LLM 解读验证代码的执行结果, 判断假设是否成立
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        expected_true = hypothesis.get("expected_if_true", "")
        expected_false = hypothesis.get("expected_if_false", "")
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        result_json = json.dumps(execution_result, indent=2, ensure_ascii=False)
        
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "\n... (截断)"
        
        prompt = RESULT_ANALYSIS_PROMPT.format(
            hypothesis_id=hyp_id,
            hypothesis_claim=claim,
            expected_if_true=expected_true,
            expected_if_false=expected_false,
            verification_plan_json=plan_json,
            execution_result_json=result_json,
        )
        
        VALID_STATUSES = {"CONFIRMED", "PARTIALLY_CONFIRMED", "REFUTED", "UNVERIFIABLE"}
        
        def _validate_result(parsed: Dict) -> bool:
            if not isinstance(parsed, dict):
                return False
            status = parsed.get("status", "")
            if status not in VALID_STATUSES:
                logger.warning(f"Invalid status '{status}' in result analysis")
                return False
            if not parsed.get("brief") and not parsed.get("detailed_reasoning"):
                logger.warning("Result analysis missing brief and detailed_reasoning")
                return False
            return True
        
        parsed = self.llm_retry.call_and_parse_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位严谨的数据科学家，擅长根据统计结果判断假设是否成立。"
                "你必须给出明确判断 (不能模棱两可)。"
                "判断基于数据，而非直觉。如果数据不支持假设, 就判为 REFUTED。"
            ),
            temperature=0.2,
            max_tokens=1024,
            max_retries=2,
            additional_instructions=(
                "输出格式必须包含: status (CONFIRMED|PARTIALLY_CONFIRMED|REFUTED|UNVERIFIABLE), "
                "brief, detailed_reasoning, evidence_summary"
            ),
            validate_func=_validate_result,
        )
        
        if parsed is not None:
            if "evidence" not in parsed:
                parsed["evidence"] = execution_result.get("statistics", execution_result)
            parsed["method"] = "agent_autonomous"
            return parsed
        
        # 重试耗尽 → 尝试从执行结果中提取部分状态
        logger.warning(f"Result analysis failed for {hyp_id}, trying partial extraction")
        statistics = execution_result.get("statistics", execution_result)
        if isinstance(statistics, dict) and statistics:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "LLM 分析结果格式错误, 但原始统计数据可用",
                "brief": "结果分析格式错误, 请查看原始统计数据",
                "evidence": statistics,
                "method": "agent_autonomous",
            }
        
        return {
            "status": self.UNVERIFIABLE,
            "reason": "LLM 分析结果解析失败 (重试耗尽)",
            "brief": "分析结果解析失败",
            "evidence": execution_result,
            "method": "agent_autonomous",
        }
    
    # ════════════════════════════════════════
    # Step 6: 反思与调整
    # ════════════════════════════════════════
    
    def _reflect_and_adjust(self, hypothesis: Dict, verification_plan: Dict,
                             analysis_result: Dict, verification_data: Dict) -> Optional[Dict]:
        """
        结果不理想时反思原因并调整验证方案
        
        让 LLM 分析失败原因并提出改进方案
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        status = analysis_result.get("status", self.UNVERIFIABLE)
        brief = analysis_result.get("brief", "")
        
        reflection_prompt = f"""
## 反思任务
假设 {hyp_id} 的验证结果为 {status}: {brief}

### 假设
- Claim: {claim}
- 验证思路: {hypothesis.get('verification_thought', '')}

### 当前验证方案
{json.dumps(verification_plan.get('verification_plan', verification_plan), ensure_ascii=False)[:500]}

### 分析结果
{json.dumps(analysis_result, ensure_ascii=False)[:500]}

### 已有数据
{json.dumps(list(verification_data.keys()), ensure_ascii=False)}

请分析验证失败的原因并提出调整方案:
1. 如果数据不足以支撑验证, 说明需要什么新数据
2. 如果统计方法不合适, 建议替代方法
3. 如果验证方案设计有缺陷, 建议如何改进

输出 JSON 格式:
```json
{{{{
  "failure_analysis": "失败原因分析",
  "adjustment_suggestion": "调整方案建议",
  "new_data_needed": ["需要的新数据"],
  "alternative_method": "替代验证方法"
}}}}
```
"""
        
        result = self.llm_retry.call_and_parse_with_retry(
            prompt=reflection_prompt,
            system_content=(
                "你是一位经验丰富的数据科学家，擅长分析验证失败的原因并提出改进方案。"
                "请给出具体可执行的调整建议，而不是泛泛而谈。"
            ),
            temperature=0.3,
            max_tokens=1024,
            max_retries=1,
        )
        
        if result is None:
            return None
        
        # 基于反思结果调整验证方案
        adjusted_plan = dict(verification_plan)
        plan = adjusted_plan.get("verification_plan", adjusted_plan)
        
        if result.get("alternative_method"):
            plan["method_name"] = result["alternative_method"]
        if result.get("adjustment_suggestion"):
            plan["method_description"] = result["adjustment_suggestion"]
        if result.get("new_data_needed"):
            plan["data_sources"] = result["new_data_needed"]
        
        return adjusted_plan
    
    # ════════════════════════════════════════
    # 缺失数据检测与获取
    # ════════════════════════════════════════
    
    def _detect_missing_data_from_error(self, error: Optional[str],
                                          hypothesis: Dict,
                                          verification_data: Dict) -> Optional[List[str]]:
        """
        从代码执行错误中检测缺失的数据
        
        查找 NameError 或 KeyError 等暗示数据缺失的错误
        """
        if not error:
            return None
        
        missing = []
        
        # 查找 NameError (变量未定义)
        name_errors = re.findall(r"NameError:\s*name\s+'(\w+)'\s+is\s+not\s+defined", error)
        for var_name in name_errors:
            if var_name not in verification_data and not var_name.startswith("_"):
                missing.append(var_name)
        
        # 查找 KeyError (字典键不存在)
        key_errors = re.findall(r"KeyError:\s*'(\w+)'", error)
        for key_name in key_errors:
            if key_name not in verification_data and not key_name.startswith("_"):
                missing.append(key_name)
        
        # 查找 FileNotFoundError
        file_errors = re.findall(r"FileNotFoundError:\s*\[Errno \d+\]\s+No\s+such\s+file.*?'([^']+)'", error)
        for file_path in file_errors:
            missing.append(f"file:{file_path}")
        
        return missing if missing else None
    
    def _acquire_missing_data(self, missing_data: List[str],
                               hypothesis: Dict,
                               verification_data: Dict) -> Optional[Dict]:
        """
        尝试获取缺失的数据
        
        策略:
        1. 从数据缓存中查找
        2. 从数据发现信息中查找对应的文件路径
        3. 对于模型内部数据, 让 DataInfrastructure 动态生成探测脚本
        """
        updated_data = dict(verification_data)
        acquired_any = False
        
        for data_name in missing_data:
            # 策略 1: 检查缓存
            cached = self.data_infra.load_from_cache(data_name)
            if cached is not None:
                updated_data[data_name] = cached
                acquired_any = True
                logger.info(f"Acquired missing data '{data_name}' from cache")
                continue
            
            # 策略 2: 从数据发现中查找文件路径
            if data_name.startswith("file:"):
                file_path = data_name[5:]
                # 让 LLM 动态生成加载脚本 (由 DataInfrastructure 处理)
                logger.info(f"Missing file: {file_path} — will attempt to load via DataInfrastructure")
                continue
            
            # 策略 3: 对于模型内部数据, 尝试探测
            model_internal_keywords = ["attention", "embedding", "hidden", "gradient", "weight"]
            if any(kw in data_name.lower() for kw in model_internal_keywords):
                logger.info(f"Missing model internal data '{data_name}' — will attempt probing")
                # DataInfrastructure 可以执行探测脚本获取这些数据
                # (具体探测脚本由 LLM 动态生成)
                continue
        
        if acquired_any:
            return updated_data
        
        return None
    
    # ════════════════════════════════════════
    # Fallback: 旧版验证方法
    # ══════════════════════════
    
    def _try_fallback_verification(self,
                                    hypothesis: Dict,
                                    wrong_text_cases: List[Dict],
                                    all_wrong_cases: List[Dict],
                                    item_popularity: Dict,
                                    overall_metrics: Dict,
                                    surprise_metrics: Dict) -> Dict:
        """当 Agent 验证完全失败时, 尝试旧版固定方法验证"""
        claim = hypothesis.get("claim", "")
        thought = hypothesis.get("verification_thought", "")
        
        method = self._infer_verification_method(claim, thought)
        hypothesis_copy = dict(hypothesis)
        hypothesis_copy["verification_method"] = method
        
        baseline = self._fallback_verifier._compute_stats_baseline(
            wrong_text_cases, all_wrong_cases, item_popularity
        )
        
        try:
            if method == "item_popularity":
                result = self._fallback_verifier._verify_item_popularity(hypothesis_copy, baseline)
            elif method == "category_bias":
                result = self._fallback_verifier._verify_category_bias(hypothesis_copy, baseline)
            elif method == "sequence_length":
                result = self._fallback_verifier._verify_sequence_length(hypothesis_copy, baseline)
            elif method == "similarity_bias":
                result = self._fallback_verifier._verify_similarity_bias(hypothesis_copy, baseline, wrong_text_cases)
            elif method == "surprise_score":
                result = self._fallback_verifier._verify_surprise_score(
                    hypothesis_copy, baseline, overall_metrics, surprise_metrics)
            else:
                result = self._fallback_verifier._verify_custom(hypothesis_copy, baseline, wrong_text_cases)
            
            result["method"] = "fallback_fixed"
            result["fallback_reason"] = "Agent autonomous verification failed, using fixed method"
            return result
        
        except Exception as e:
            logger.error(f"Fallback verification also failed: {e}")
            return {
                "status": self.UNVERIFIABLE,
                "reason": f"Both agent and fallback verification failed: {str(e)}",
                "brief": "验证完全失败",
                "evidence": None,
            }
    
    def _infer_verification_method(self, claim: str, thought: str) -> str:
        """从假设的自由描述推断最匹配的旧版固定方法"""
        text = (claim + " " + thought).lower()
        
        if any(kw in text for kw in ["热度", "popularity", "冷门", "热门", "频次", "交互次数", "cold", "hot", "unpopular", "popular"]):
            return "item_popularity"
        if any(kw in text for kw in ["类别", "category", "跨类别", "类别集中", "类型偏差"]):
            return "category_bias"
        if any(kw in text for kw in ["序列长度", "sequence length", "短序列", "长序列", "历史长度"]):
            return "sequence_length"
        if any(kw in text for kw in ["相似性", "similarity", "相似度", "嵌入", "余弦"]):
            return "similarity_bias"
        if any(kw in text for kw in ["惊喜", "surprise", "差异大", "偏离"]):
            return "surprise_score"
        
        return "custom"
    
    # ════════════════════════════════════════
    # Phase 3: 生成验证报告 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def generate_verification_report(self, verified_hypotheses: List[Dict]) -> Dict:
        """生成验证报告 (与旧版 HypothesisVerifier.generate_verification_report 一致)"""
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
                "verification_method": result.get("method", "agent_autonomous"),
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
        
        total = len(verified_hypotheses)
        confirmed_pct = len(confirmed) / total * 100 if total > 0 else 0
        refuted_pct = len(refuted) / total * 100 if total > 0 else 0
        
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
            "verification_agent_used": True,
        }
        
        print(f"\n  ══════════ 假设验证报告 (Agent 自主验证) ══════════")
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
        print(f"  ════════════════════════════════════\n")
        
        return report
    
    # ════════════════════════════════════════
    # 将验证结果应用到分析 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def apply_verification_to_analysis(self,
                                        llm_analysis: Dict,
                                        verification_report: Dict) -> Dict:
        """将验证结果应用到 LLM 分析结论 (与旧版接口一致)"""
        return self._fallback_verifier.apply_verification_to_analysis(
            llm_analysis, verification_report
        )
    
    # ════════════════════════════════════════
    # 保存验证报告 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def save_verification_report(self, report: Dict, output_path: str):
        """保存验证报告"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved verification report to {output_path}")
    
    # ════════════════════════════════════════
    # 兼容方法
    # ════════════════════════════════════════
    
    def compute_item_popularity_from_data(self, train_data) -> Dict:
        """从训练数据计算物品热度分布 (与旧版一致)"""
        return self._fallback_verifier.compute_item_popularity_from_data(train_data)