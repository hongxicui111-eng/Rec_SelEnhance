"""
RecSelfEvolveAgent — 推荐系统自增强 Agent 主循环

整合:
- Google 论文: 双循环架构 + 专业化 Persona + Think-Code-Verify
- Self-EvolveRec: 方向性反馈 + 诊断-模型协同进化
- **你的项目适配**: 通过 ProjectAdapter 理解具体运行方式
"""
import ast
import json
import re
import logging
import sys
import os
import shutil
from typing import Optional

from .config import AgentConfig
from .llm_client import LLMClient
from .error_handler import ProposalParser, LLMFixer
from .train_runner import FaultTolerantTrainRunner
from .code_applier import CodeApplier
from .quality_guard import EvolutionQualityGuard, SafetyGuardrails
from .journal import ExperimentJournal
from .project_adapter import create_adapter, SeqRecAdapter
from .prompts import (
    MLE_ANALYSIS_PROMPT, STRUCTURE_OPTIMIZATION_PROMPT,
    ERROR_FEEDBACK_PROMPT, STRUCTURE_FIX_PROMPT,
    TRAIN_DIAGNOSIS_PROMPT, CODE_FIX_PROMPT, PREFLIGHT_FIX_PROMPT,
    # 多角色工作流 prompt 模板
    PLANNER_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
    REFLECTION_INSTRUCTIONS, SEARCH_INSTRUCTIONS,
    CODER_INSTRUCTIONS, DEBUGGER_INSTRUCTIONS,
    # 代码查询模式 prompt 模板 (核心新增!)
    QUERY_BASED_PHASE1_PROMPT, QUERY_BASED_PHASE2_PROMPT, QUERY_BASED_FINAL_PROMPT,
    # 纠错查询模式 prompt 模板 (核心新增!)
    QUERY_BASED_FIX_PHASE1_PROMPT, QUERY_BASED_FIX_PHASE2_PROMPT,
    QUERY_BASED_FIX_FINAL_PROMPT, QUERY_BASED_PREFLIGHT_FIX_PROMPT,
    # SEARCH/REPLACE 匹配失败重试 prompt
    SEARCH_REPLACE_FIX_PROMPT,
)
from .llm_analyzer import LLMCaseAnalyzer
from .hypothesis_verification_agent import HypothesisVerificationAgent
from .structure_applier import StructureApplier
from .iterative_memory import IterativeMemory
from .llm_utils import parse_json_from_response, clean_markdown_wrapper as llm_clean_markdown_wrapper
from .context_compressor import LLMContextCompressor
from .code_query_tool import CodeQueryTool

logger = logging.getLogger("rec_self_evolve.core")


class RecSelfEvolveAgent:
    """
    推荐系统自增强 Agent
    
    如何工作:
    1. 通过 ProjectAdapter 理解你的项目(训练命令、参数格式、评估指标、模型源码)
    2. 循环: 训练 → 评估 → LLM 分析瓶颈 → LLM 提出改进 (参数+结构) → 应用变更 → 重训练验证
    3. **核心升级**: LLM 不仅调参数，还可以修改模型结构 (如改attention机制、加新模块等)
    4. 自动处理 OOM、NaN、语法错误、指标退化等异常
    """

    def __init__(self, config: Optional[AgentConfig] = None,
                 adapter: Optional[SeqRecAdapter] = None):
        self.config = config or AgentConfig()
        self._setup_logging()

        # ---- 创建或使用适配器 ----
        self.adapter = adapter or self._create_adapter()
        logger.info(f"Using adapter for: {self.adapter.backbone} @ {self.adapter.data_name}")

        # ---- 初始化各模块 ----
        logger.info("Initializing RecSelfEvolveAgent...")
        self.llm = LLMClient(
            api_url=self.config.llm_api_url,
            api_key=self.config.llm_api_key,
            model=self.config.llm_model,
            timeout=self.config.llm_timeout,
            max_retries=self.config.llm_max_retries,
            max_context_tokens=self.config.llm_max_context_tokens,
            prompt_safety_ratio=self.config.llm_prompt_safety_ratio,
        )
        self.trainer = FaultTolerantTrainRunner(
            adapter=self.adapter,
            timeout=self.config.train_timeout,
            oom_reduce_factor=self.config.oom_reduce_factor,
            nan_reduce_factor=self.config.nan_reduce_factor,
        )
        self.applier = CodeApplier(project_root=self.config.project_root)
        # ---- 结构修改应用器 (核心新增!) ----
        self.struct_applier = StructureApplier(
            project_root=self.config.project_root,
            adapter=self.adapter,
            log_dir=self.config.log_dir,
        )
        # ---- 迭代修改记忆系统 (让LLM感知历史修改!) ----
        self.iter_memory = IterativeMemory(
            project_root=self.config.project_root,
            log_dir=self.config.log_dir,
            source_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
        )
        self.guard = EvolutionQualityGuard(
            window_size=self.config.quality_window,
            degrade_threshold=self.config.degrade_threshold,
            plateau_threshold=self.config.plateau_threshold,
        )
        self.safety = SafetyGuardrails(self.config.metric_guardrails)
        self.journal = ExperimentJournal(
            file_path=os.path.join(self.config.log_dir, self.config.journal_file)
        )
        self.parser = ProposalParser()
        self.fixer = LLMFixer(self.llm)
        self.context_compressor = LLMContextCompressor(
            self.llm,
            enable_cache=self.config.llm_compression_enable_cache,
            cache_ttl_seconds=self.config.llm_compression_cache_ttl_seconds,
            cache_path=self.config.llm_compression_cache_path,
        )

        # ---- 惊喜评估与案例分析 ----
        self.item_text_map = self._load_item_text_map()
        self.case_analyzer = LLMCaseAnalyzer(self.llm, self.item_text_map)

        # ---- 假设验证 Agent (自主式: LLM生成方案 → 写代码 → 执行 → 分析) ----
        self.hypothesis_verifier = HypothesisVerificationAgent(
            self.llm, self.item_text_map,
            project_root=self.config.project_root,
            data_dir=os.path.join(self.config.project_root, "Recmodel", "data"),
            log_dir=self.config.log_dir,
        )

        # ---- 代码查询工具 (核心新增!) ----
        self._code_query_enabled = self.config.enable_code_query
        if self._code_query_enabled:
            self._code_query_tool = CodeQueryTool(
                project_root=self.config.project_root,
                source_file_map=dict(self.adapter.SOURCE_FILE_MAP),
                adapter=self.adapter,
            )
            logger.info("Code query tool enabled — LLM will query code on-demand instead of receiving all source code upfront")

        self.current_iteration = 0
        self.current_strategy = "balanced"
        self.llm_health_ok = False
        self._last_surprise_report = None  # 上次惊喜评估报告
        self._last_case_analysis = None     # 上次案例分析结果
        self._last_verification_report = None   # 上次假设验证报告
        self._last_wrong_text_cases = None  # 上次提取的错误文本案例
        self._last_structural_changes = None   # 上次结构修改提案
        # _structural_change_history 已迁移到 IterativeMemory — 不再使用 list
        self._max_self_correction_rounds = 5  # 自纠错最大轮数 (从3增加到5!)
        self._max_code_fix_rounds = 10        # 源码bug修复最大轮数 (核心新增!)
        self._consecutive_fail_count = 0      # 连续失败计数 (用于判断是否需要强制修复)

        # ---- 多角色工作流 (Planner→Researcher→Coder→Debugger) ----
        self._multi_role_enabled = self.config.enable_multi_role_workflow
        if self._multi_role_enabled:
            from .researcher import ResearcherAgent
            from .coder import CoderAgent
            
            self._researcher_agent = ResearcherAgent(
                api_url=self.config.llm_api_url,
                api_key=self.config.llm_api_key,
                model=self.config.researcher_model or self.config.llm_model,
                temperature=self.config.researcher_temperature,
                max_reflection_times=self.config.max_reflection_rounds,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            self._coder_agent = CoderAgent(
                api_url=self.config.llm_api_url,
                api_key=self.config.llm_api_key,
                model=self.config.coder_model or self.config.llm_model,
                temperature=self.config.coder_temperature,
                max_reflection_times=self.config.max_reflection_rounds,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
            )
            # Debugger 使用独立 LLM 调用 (温度更低, 更精确)
            self._debugger_llm = LLMClient(
                api_url=self.config.llm_api_url,
                api_key=self.config.llm_api_key,
                model=self.config.debugger_model or self.config.llm_model,
                timeout=self.config.llm_timeout,
                max_retries=self.config.llm_max_retries,
                max_context_tokens=self.config.llm_max_context_tokens,
                prompt_safety_ratio=self.config.llm_prompt_safety_ratio,
            )
            logger.info("Multi-role workflow enabled: Planner→Researcher→Coder→Debugger")

        logger.info("RecSelfEvolveAgent initialized")

    def _create_adapter(self) -> SeqRecAdapter:
        """根据配置创建项目适配器"""
        return create_adapter(
            project_root=self.config.project_root,
            script_name=self.config.script_name,
            data_name=self.config.data_name,
            backbone=self.config.backbone,
            gpu_id=self.config.gpu_id,
            output_dir=self.config.output_dir,
            extra_args=self.config.extra_args,
        )

    def _load_item_text_map(self) -> dict:
        """加载物品 ID → 元数据映射 (id_meta_data.json 格式: nested dict with title/categories/description)"""
        # 优先使用用户指定的路径
        map_path = self.config.item_text_map_path
        if map_path and os.path.exists(map_path):
            with open(map_path, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            logger.info(f"Loaded item text map: {len(mapping)} items from {map_path}")
            return mapping
        # 查找默认路径: Recmodel/data/id_meta_data.json
        default_paths = [
            os.path.join(self.config.project_root, 'data', 'id_meta_data.json'),
            os.path.join(self.config.project_root, 'Recmodel', 'data', 'id_meta_data.json'),
        ]
        for path in default_paths:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    mapping = json.load(f)
                logger.info(f"Loaded item text map: {len(mapping)} items from {path}")
                return mapping
        logger.info("No id_meta_data.json found, will use item IDs as fallback")
        return {}

    # ════════════════════════════════════════
    # 主入口
    # ════════════════════════════════════════

    def evolve(self, max_iterations: Optional[int] = None) -> dict:
        """
        启动自进化循环 (v0.5 — 增强自纠错闭环)
        
        核心改进:
        1. 每次训练后立即检查是否成功, 失败则进入自纠错内循环
        2. 自纠错内循环: 提取错误 → 判断类型 → 修源码/调参数 → 重训 → 重复
        3. 基线训练失败不再简单跳过, 而是尝试修复源码中的 bug
        4. 自纠错轮数从 2-3 轮增加到 5-10 轮
        """
        max_iters = max_iterations or self.config.max_iterations
        logger.info(f"Starting evolution: {max_iters} max iterations")

        print(f"\n{'='*60}")
        print(f"  RecSelfEvolve — 推荐系统自增强 Agent (v0.6)")
        print(f"  核心改进: 代码查询模式 — LLM 按需获取代码, 不塞全部源码+截断")
        print(f"  LLM: {self.config.llm_model} @ {self.config.llm_api_url}")
        print(f"  Project: {self.config.project_root}")
        print(f"  Backbone: {self.adapter.backbone}")
        print(f"  Dataset: {self.adapter.data_name}")
        print(f"  Max iterations: {max_iters}")
        print(f"  Max code fix rounds: {self._max_code_fix_rounds}")
        print(f"  Max self correction rounds: {self._max_self_correction_rounds}")
        mode_str = "QUERY (LLM按需获取代码)" if self._code_query_enabled else "TRADITIONAL (塞全部源码)"
        if self._multi_role_enabled:
            mode_str = "MULTI-ROLE (Planner→Researcher→Coder)"
        print(f"  Code analysis mode: {mode_str}")
        print(f"  Max query rounds: {self.config.max_query_rounds}")
        print(f"{'='*60}\n")

        # ---- Step 0: 检查 LLM 健康状态 ----
        self._check_llm_health()

        # ---- 记录上一轮 retrain 结果 (用于后续迭代跳过 baseline 训练) ----
        last_retrain_metrics = None  # 上一轮 retrain 的指标
        last_retrain_result = None   # 上一轮 retrain 的完整结果

        # ---- 主循环 ----
        for i in range(max_iters):
            self.current_iteration = i
            print(f"\n{'─'*50}")
            print(f"  [Iteration {i+1}/{max_iters}]  strategy={self.current_strategy}  consecutive_fail={self._consecutive_fail_count}")
            print(f"{'─'*50}")

            # ── Phase 0: 预检 — 训练前检查源码是否有语法/导入错误 ──
            preflight = self._phase_preflight_check()
            if preflight["status"] == "FAIL":
                print(f"  ⚠ Preflight check FAILED: {len(preflight['errors'])} issues found")
                for err in preflight["errors"]:
                    print(f"    ✗ {err['file']}: {err['error_type']} @ line {err.get('line', '?')} — {err.get('error_message', '')[:80]}")
                
                # ── 预检发现问题 → 先修复源码语法错误, 再训练 ──
                print(f"  🔁 触发预检修复自纠错...")
                fix_result = self._preflight_fix_and_retry(preflight, iteration=i)
                if fix_result is None:
                    # 预检修复也失败 → 跳过本轮
                    print(f"  ⏭ Preflight fix failed, skipping iteration")
                    self._consecutive_fail_count += 1
                    self.journal.record({
                        "iteration": i, "status": "PREFLIGHT_FAILED",
                        "preflight_errors": preflight["errors"],
                    })
                    continue
                # 预检修复成功 → 继续后续流程
                print(f"  ✓ Preflight fix successful, proceeding to training")

            # --- Phase 0.5: 保存当前轮次的源码快照 ---
            self.iter_memory.save_source_snapshot(i)

            # ── Phase 1: 获取当前指标 ──
            # ★★★ 关键优化: 只有第一轮需要 baseline 训练 ★★★
            # 从第二轮开始, 上一轮的 retrain 已经训练了修改后的代码并得到指标,
            # 当前轮次的代码已经是修改后的版本, 不需要重新训练 baseline!
            if i == 0 or last_retrain_metrics is None:
                # 第一轮 / 上一轮 retrain 失败无指标 → 需要跑 baseline 训练
                print(f"  🔄 Running baseline training (iteration {i+1} needs fresh metrics)")
                train_result = self._phase_run_verify_fix_retry(
                    iteration=i,
                    param_overrides=None,
                    phase_name="baseline",
                )

                if train_result["status"] != "SUCCESS":
                    # run-verify-fix-retry 闭环已经尝试了所有可能的修复
                    # 如果仍然失败 → 只有真正无法修复的问题才跳过
                    print(f"  ⏭ Baseline training failed after all self-correction attempts")
                    print(f"     Final error: {train_result.get('error', '')[:100]}")
                    print(f"     Error category: {train_result.get('error_category', 'UNKNOWN')}")
                    
                    self._consecutive_fail_count += 1
                    self.journal.record({
                        "iteration": i,
                        "status": train_result["status"],
                        "error": train_result.get("error", ""),
                        "error_category": train_result.get("error_category", ""),
                        "fix_attempts": train_result.get("fix_attempts", 0),
                    })
                    
                    # 如果连续失败太多 → 尝试回滚到原始源码状态
                    if self._consecutive_fail_count >= 3:
                        print(f"  ⚠ 连续失败 {self._consecutive_fail_count} 次 → 回滚所有源码修改!")
                        self._rollback_all_source_changes()
                        self._consecutive_fail_count = 0
                        print(f"  ↩ 所有源码修改已回滚, 下一轮使用原始代码")
                    continue
                
                # 训练成功 → 清零连续失败计数
                self._consecutive_fail_count = 0

                # --- Phase 2: 提取指标 ---
                metrics = train_result.get("metrics", {})
                print(f"  📊 {self.adapter.format_metrics_for_llm(metrics)}")
            else:
                # 第二轮+ → 直接使用上一轮 retrain 的指标, 跳过 baseline 训练!
                metrics = last_retrain_metrics
                print(f"  ⏭ Skipping baseline training — using metrics from previous iteration's retrain")
                print(f"  📊 {self.adapter.format_metrics_for_llm(metrics)}")

            # --- Phase 2.5: 惊喜评估 + 错误案例分析 ---
            surprise_and_analysis = self._phase_surprise_analysis(metrics)
            if surprise_and_analysis:
                self._last_surprise_report = surprise_and_analysis.get("evaluation_report")
                self._last_case_analysis = surprise_and_analysis.get("case_analysis")
                self._last_wrong_text_cases = surprise_and_analysis.get("wrong_text_cases")
                
                diagnosis = surprise_and_analysis.get("evaluation_report", {}).get("diagnosis", {})
                if diagnosis:
                    print(f"  🎯 Diagnosis:")
                    if "overfitting" in diagnosis:
                        print(f"    Overfitting: {diagnosis['overfitting']} — {diagnosis.get('overfit_note', '')}")
                    if "surprise_capture" in diagnosis:
                        print(f"    Surprise: {diagnosis['surprise_capture']} — {diagnosis.get('surprise_note', '')}")
                    if "metric_balance" in diagnosis:
                        print(f"    Balance: {diagnosis['metric_balance']} — {diagnosis.get('metric_note', '')}")
                
                self.journal.record({
                    "iteration": i,
                    "phase": "surprise_analysis",
                    "status": "SURPRISE_ANALYSIS",
                    "surprise_report": surprise_and_analysis.get("evaluation_report_summary"),
                    "case_analysis_summary": surprise_and_analysis.get("case_analysis_summary"),
                })

                # --- Phase 2.6: 假设验证 (验证 LLM 分析结论是否有数据支撑) ---
                verification_result = self._phase_verify_analysis(
                    surprise_and_analysis, metrics
                )
                if verification_result:
                    self._last_verification_report = verification_result
                    # 将验证结果应用到案例分析, 过滤被反驳的结论
                    self._last_case_analysis = self.hypothesis_verifier.apply_verification_to_analysis(
                        self._last_case_analysis, verification_result
                    )
                    self.journal.record({
                        "iteration": i,
                        "phase": "hypothesis_verification",
                        "status": "VERIFICATION_DONE",
                        "overall_credibility": verification_result.get("overall_credibility", "LOW"),
                        "confirmed_pct": verification_result.get("confirmed_pct", 0),
                        "refuted_pct": verification_result.get("refuted_pct", 0),
                        "refuted_claims": [r.get("claim", "") for r in verification_result.get("refuted", [])],
                    })

            # --- Phase 3: 安全护栏 ---
            violations = self.safety.check_metrics(metrics)
            if violations:
                print(f"  ⚠ Safety: {violations}")
                # 安全护栏拦截 → 代码未修改, 下一轮仍可使用当前指标
                last_retrain_metrics = metrics
                print(f"  ⚠ Metrics preserved for next iteration (code unchanged)")
                self.journal.record({
                    "iteration": i, "status": "SAFETY_VIOLATION",
                    "metrics": metrics, "error": "; ".join(violations),
                })
                continue

            # --- Phase 4: LLM 分析 + 提案 ---
            if self._multi_role_enabled:
                # 多角色工作流: Planner → Researcher → Coder → Debugger
                proposal_result = self._phase_multi_role_analyze_and_propose(metrics)
            elif self._code_query_enabled:
                # 查询模式: LLM 按需获取代码, 不塞全部源码 (核心新增!)
                proposal_result = self._phase_query_based_analyze_and_propose(metrics)
            else:
                # 传统模式: 塞全部源码 + 截断
                proposal_result = self._phase_analyze_and_propose(metrics)
            if proposal_result is None:
                print(f"  ⚠ LLM analysis failed, skipping")
                # LLM分析失败 → 代码未修改, 下一轮仍可使用当前指标
                last_retrain_metrics = metrics
                print(f"  ⚠ Metrics preserved for next iteration (code unchanged)")
                continue

            param_changes = proposal_result.get("param_changes", {})
            structural_changes = proposal_result.get("structural_changes", [])
            explanation = proposal_result.get("explanation", "")

            print(f"  💡 Analysis: {explanation[:120]}...")
            print(f"  🔧 Param changes: {json.dumps(param_changes, ensure_ascii=False)[:200]}")
            if structural_changes:
                print(f"  🏗️ Structural changes: {len(structural_changes)} modifications")
                for sc in structural_changes:
                    print(f"    → [{sc.get('target_file', '?')}] "
                          f"{sc.get('target_class_or_function', '?')}: "
                          f"{sc.get('description', '?')[:60]}...")
            else:
                print(f"  🏗️ No structural changes proposed")

            # --- Phase 5: 应用结构修改 (如果有) ---
            struct_result = None
            if structural_changes:
                print(f"  🏗️ Applying {len(structural_changes)} structural changes...")
                struct_result = self.struct_applier.apply_structural_changes(structural_changes)

                if struct_result["status"] == "ROLLBACK":
                    print(f"  ↩ Structural changes rolled back: {struct_result.get('rollback_reason', '')[:100]}")
                    self.iter_memory.record_rollback(
                        iteration=i,
                        reason=struct_result.get("rollback_reason", "validation failed"),
                    )
                    
                    # ── 自纠错: 让 LLM 修正校验失败的结构修改代码 ──
                    print(f"  🔁 触发结构修改代码自纠错...")
                    validation_failures = struct_result.get("failed_changes", [])
                    if not validation_failures:
                        validation_failures = [{
                            "description": sc.get("description", "unknown"),
                            "error_type": "validation_error",
                            "error": struct_result.get("rollback_reason", "validation failed"),
                            "target_class_or_function": sc.get("target_class_or_function", "?"),
                        } for sc in structural_changes]
                    
                    fix_result = self._phase_structure_validation_retry(
                        iteration=i,
                        original_structural_changes=structural_changes,
                        validation_failures=validation_failures,
                    )
                    
                    if fix_result and fix_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                        struct_result = fix_result
                        revised_structural_changes = fix_result.get("applied_changes", structural_changes)
                        structural_changes = revised_structural_changes
                        print(f"  ✓ 结构修改自纠错成功: {fix_result['status']}")
                        self._last_structural_changes = fix_result.get("applied_changes", [])
                    else:
                        print(f"  ✗ 结构修改自纠错也失败，仅尝试参数修改")
                        structural_changes = []
                        struct_result = None
                    
                    self.journal.record({
                        "iteration": i, "status": "STRUCTURE_ROLLBACK_WITH_RETRY",
                        "metrics": metrics,
                        "structural_changes": structural_changes,
                        "error": struct_result.get("rollback_reason", "") if struct_result else "",
                    })
                elif struct_result["status"] == "SUCCESS" or struct_result["status"] == "PARTIAL_SUCCESS":
                    # 防御性兜底：状态显示成功但没有任何实际应用，按失败处理
                    applied_count = len(struct_result.get("applied_changes", []))
                    if applied_count == 0:
                        print(f"  ✗ Structural changes reported {struct_result['status']} but applied_changes=0; treat as failed")
                        structural_changes = []
                        struct_result = {
                            **struct_result,
                            "status": "ALL_FAILED",
                            "error": "No structural changes were actually applied",
                        }
                    else:
                        print(f"  ✓ Structural changes applied: {struct_result['status']}")
                        print(f"    Files modified: {struct_result.get('files_modified', [])}")
                        self._last_structural_changes = struct_result.get("applied_changes", [])
                elif struct_result["status"] == "ALL_FAILED":
                    # ── ALL_FAILED 重试机制: 将匹配失败诊断反馈给 LLM, 让它修正 ──
                    print(f"  ✗ All structural changes failed — triggering SEARCH/REPLACE retry...")
                    detailed_failure = struct_result.get("detailed_failure_info", [])

                    # 打印详细失败诊断
                    for df in detailed_failure:
                        print(f"     问题: [{df.get('target_file', '?')}] "
                              f"{df.get('description', '?')[:60]}")
                        for ed in df.get("failed_edit_details", []):
                            diag = ed.get("match_diagnostic", {})
                            ratio = diag.get("best_fuzzy_ratio", "N/A")
                            non_match = diag.get("level1_non_matching_lines", "N/A")
                            print(f"       Edit {ed.get('edit_idx', '?')}: "
                                  f"best_fuzzy_ratio={ratio}, "
                                  f"non_matching_lines={non_match}, "
                                  f"search_preview={ed.get('search_text_preview', '?')[:60]}...")

                    # ── 自纠错: 让 LLM 根据 实际源码 + 失败诊断 重新编写 search 文本 ──
                    fix_result = self._phase_search_replace_retry(
                        iteration=i,
                        original_structural_changes=structural_changes,
                        detailed_failure_info=detailed_failure,
                    )

                    if fix_result and fix_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                        # 重试成功 → 使用修正后的结果
                        struct_result = fix_result
                        revised_structural_changes = fix_result.get("applied_changes", structural_changes)
                        structural_changes = revised_structural_changes
                        print(f"  ✓ SEARCH/REPLACE 重试成功: {fix_result['status']}")
                        print(f"    修改文件: {fix_result.get('files_modified', [])}")
                        self._last_structural_changes = fix_result.get("applied_changes", [])
                        self.journal.record({
                            "iteration": i, "status": "STRUCTURE_ALL_FAILED_WITH_RETRY_SUCCESS",
                            "metrics": metrics,
                            "structural_changes": structural_changes,
                            "original_structural_changes": proposal_result.get("structural_changes", []),
                            "detailed_failure_info": detailed_failure,
                        })
                    else:
                        # 重试也失败 → 不可恢复
                        # ── 关键决策: 是否跳过本轮 ──
                        # 如果提案的核心意图是结构修改 (代码变更), 用原代码训练没有意义
                        # 但如果提案还有独立的参数修改, 仍然可以仅做参数修改训练
                        if param_changes:
                            # 有独立的参数修改 → 仅保留参数修改, 跳过结构修改
                            print(f"  ⚠ 结构修改失败但有参数修改 → 仅尝试参数修改")
                            structural_changes = []
                            self.journal.record({
                                "iteration": i, "status": "STRUCTURE_ALL_FAILED_RETRY_ALSO_FAILED_WITH_PARAM_FALLBACK",
                                "metrics": metrics,
                                "structural_changes": proposal_result.get("structural_changes", []),
                                "failed_changes": struct_result.get("failed_changes", []),
                                "detailed_failure_info": detailed_failure,
                                "retry_result": fix_result,
                                "param_changes": param_changes,
                                "error": "Structural changes failed, retry also failed; falling back to param-only",
                            })
                        else:
                            # 没有参数修改 → 本轮完全无法产生有效变更, 跳过
                            print(f"  ✗ SEARCH/REPLACE 重试也失败且无参数修改 — 本轮完全无效，跳过")
                            structural_changes = []
                            self.journal.record({
                                "iteration": i, "status": "STRUCTURE_ALL_FAILED_RETRY_ALSO_FAILED",
                                "metrics": metrics,
                                "structural_changes": proposal_result.get("structural_changes", []),
                                "failed_changes": struct_result.get("failed_changes", []),
                                "detailed_failure_info": detailed_failure,
                                "retry_result": fix_result,
                                "error": "All structural changes failed, retry also failed, no param changes",
                            })

            # --- Phase 6: 应用参数变更 + 训练验证 (使用新的 run-verify-fix-retry!) ---
            has_any_change = bool(param_changes) or bool(structural_changes)
            if not has_any_change:
                # 区分两种 NO_CHANGES 情况：
                # (1) 提案本身就无变更 (param_changes={}, structural_changes=[])
                # (2) 提案有变更但被验证/应用环节清空了
                had_changes_in_proposal = bool(proposal_result.get("structural_changes", [])) or bool(proposal_result.get("param_changes", {}))
                reason = "changes_proposed_but_lost_after_validation" if had_changes_in_proposal else "no_changes_in_proposal"
                print(f"  ⚠ No valid changes extracted (reason: {reason})")
                # NO_CHANGES → 代码未修改, 下一轮仍可使用当前指标
                last_retrain_metrics = metrics
                print(f"  ⚠ Metrics preserved for next iteration (code unchanged)")
                self.journal.record({
                    "iteration": i, "status": "NO_CHANGES",
                    "metrics": metrics, "proposal_result": proposal_result,
                    "no_changes_reason": reason,
                    "struct_result_status": struct_result.get("status") if struct_result else None,
                })
                continue

            # ── 使用 run-verify-fix-retry 闭环训练验证! ──
            # 如果有结构修改，训练会使用修改后的模型代码
            new_train = self._phase_run_verify_fix_retry(
                iteration=i,
                param_overrides=param_changes if param_changes else None,
                phase_name="retrain",
                structural_changes=structural_changes,
                struct_result=struct_result,
                metrics_before=metrics,
            )

            if new_train["status"] == "SUCCESS":
                new_metrics = new_train.get("metrics", {})
                print(f"  → After: {self.adapter.format_metrics_for_llm(new_metrics)}")

                # ★★★ 保存 retrain 指标, 供下一轮迭代跳过 baseline 训练 ★★★
                last_retrain_metrics = new_metrics
                last_retrain_result = new_train
                print(f"  ✓ Metrics saved for next iteration (will skip baseline training)")

                # ── 记录修改因果链到 IterativeMemory ──
                if structural_changes and struct_result:
                    self.iter_memory.record_modification(
                        iteration=i,
                        structural_changes=structural_changes,
                        apply_result=struct_result,
                        metrics_before=metrics,
                        metrics_after=new_metrics,
                    )

                # 质量检查
                guard_decision = self.guard.update(
                    iteration=i,
                    metrics=new_metrics,
                    config=param_changes,
                )

                self.journal.record({
                    "iteration": i, "status": "SUCCESS",
                    "metrics": new_metrics,
                    "param_changes": param_changes,
                    "structural_changes": structural_changes,
                    "structural_result": struct_result,
                    "explanation": explanation,
                    "guard_decision": guard_decision,
                })

                # 执行守卫决策
                if guard_decision["action"] == "REVERT_TO_BEST":
                    print(f"  ↩ Reverting to iter {guard_decision.get('best_iteration', '?')}")
                    best_iter = guard_decision.get('best_iteration', 0)
                    if structural_changes and struct_result and struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                        self.struct_applier.rollback_last_changes()
                        print(f"  ↩ Structural changes also rolled back")
                        self.iter_memory.record_rollback(
                            iteration=i,
                            reason=f"Quality guard reverted to best iter {best_iter}",
                            rollback_to_iteration=best_iter,
                        )
                    self.trainer.run(param_overrides=self.guard.best_config)
                elif guard_decision["action"] == "SWITCH_STRATEGY":
                    self.current_strategy = guard_decision.get("strategy", "aggressive")
                    print(f"  🔄 Strategy → {self.current_strategy}")
                else:
                    print(f"  ✓ Guard: {guard_decision['action']}")

            else:
                # run-verify-fix-retry 已经尝试了所有可能的修复
                print(f"  ✗ Retrain failed after all self-correction attempts")
                print(f"     Final error: {new_train.get('error', '')[:100]}")
                print(f"     Error category: {new_train.get('error_category', 'UNKNOWN')}")
                
                # ★★★ retrain 失败 → 清空 last_retrain_metrics, 下一轮需要重新 baseline 训练 ★★★
                last_retrain_metrics = None
                last_retrain_result = None
                print(f"  ⚠ Retrain failed — next iteration will run baseline training")
                
                self._consecutive_fail_count += 1
                self.journal.record({
                    "iteration": i, "status": "RETRAIN_FAILED_AFTER_ALL_CORRECTIONS",
                    "metrics": metrics,
                    "param_changes": param_changes,
                    "structural_changes": structural_changes,
                    "error": new_train.get("error", ""),
                    "error_category": new_train.get("error_category", ""),
                    "fix_attempts": new_train.get("fix_attempts", 0),
                })
                
                # 回退到 best config
                if self.guard.best_config:
                    print(f"  ↩ Falling back to best config")
                    self.trainer.run(param_overrides=self.guard.best_config)

        # ---- 完成 ----
        summary = self.guard.get_summary()
        mem_stats = self.iter_memory.get_summary_stats()
        result = {
            "journal": self.journal.records,
            "best_metrics": summary.get("best_metrics", {}),
            "best_iteration": summary.get("best_iteration", -1),
            "total_iterations": self.current_iteration + 1,
            "termination_reason": "max_iterations_reached",
            "structural_change_summary": mem_stats,
            "consecutive_fail_count": self._consecutive_fail_count,
        }
        self._print_final_report(result)
        return result

    # ════════════════════════════════════════
    # 新增核心方法: 预检 + run-verify-fix-retry 闭环
    # ════════════════════════════════════════

    def _phase_preflight_check(self) -> dict:
        """
        预检: 在训练前检查源码文件是否语法正确
        
        这是最关键的改进之一 — 在每次训练前先检查源码是否有语法错误,
        如果有就先修复, 避免浪费训练时间在明知有 bug 的代码上!
        
        Returns:
            {
                "status": "PASS" | "FAIL",
                "errors": [dict],  # 发现的语法错误
                "files_checked": [str],
            }
        """
        logger.info("Running preflight check...")
        print(f"  🔍 Preflight check: validating source code...")
        
        source_files = list(self.adapter.SOURCE_FILE_MAP.keys())
        result = self.trainer.preflight_check(source_files)
        
        if result["status"] == "PASS":
            print(f"  ✓ Preflight check passed — source code OK")
        else:
            print(f"  ✗ Preflight check FAILED — {len(result['errors'])} issues found")
        
        return result

    # ════════════════════════════════════════════════════════
    # 通用: 纠错阶段的查询模式 (核心新增!)
    # ════════════════════════════════════════════════════════
    #
    # 让纠错过程也用 query 模式, 不再塞全部源码+截断!
    # 所有纠错方法 (_preflight_fix, _fix_code_error, etc.) 共用此方法
    #

    def _fix_with_query_mode(
        self,
        phase1_prompt_template: str,
        phase2_prompt_template: str,
        final_prompt_template: str,
        phase1_prompt_kwargs: dict,
        phase2_extra_kwargs: dict = None,
        final_extra_kwargs: dict = None,
        system_prompt: str = None,
        temperature: float = 0.2,
        max_query_rounds: int = 3,
    ) -> Optional[dict]:
        """
        纠错阶段的通用 query 模式方法
        
        让 LLM 通过按需查询代码来修复 bug, 而不是塞全部源码+截断。
        工作流程:
          1. 第一轮: 给 LLM 错误信息 + 代码索引 → LLM 查询出错位置代码
          2. 后续轮: LLM 继续查询或输出修复方案
          3. 最终轮: 强制输出修复方案
        
        Args:
            phase1_prompt_template: 第一轮 prompt 模板 (含 {_error_type} 等占位符)
            phase2_prompt_template: 第二轮 prompt 模板
            final_prompt_template: 最终轮 prompt 模板
            phase1_prompt_kwargs: 第一轮 prompt 的格式化参数
            phase2_extra_kwargs: 第二轮额外参数 (如 {_error_type})
            final_extra_kwargs: 最终轮额外参数
            system_prompt: system message 内容
            temperature: LLM 温度
            max_query_rounds: 最大查询轮数
        
        Returns:
            Dict: 修复方案 (与 _parse_proposal_response 格式一致)
            None: 查询失败或无法解析
        """
        if system_prompt is None:
            system_prompt = (
                "你是一位 Python 代码调试专家。"
                "你可以通过代码查询工具按需获取源码详情, 然后基于精确的代码提出修复方案。"
                "⚠ 重要: 如果你想修改代码, 请先查询要修改的代码, "
                "确保 SEARCH/REPLACE 的 search 文本与源码完全匹配!"
            )
        
        if phase2_extra_kwargs is None:
            phase2_extra_kwargs = {}
        if final_extra_kwargs is None:
            final_extra_kwargs = {}
        
        # ── 累积查询结果 ──
        all_queried_code = ""
        observation_summary = ""
        reasoning_summary = ""
        analysis_direction = ""
        query_results_this_round = ""
        
        for round_idx in range(max_query_rounds):
            print(f"  🔍 [Fix Query Round {round_idx + 1}/{max_query_rounds}]")
            
            # ── 构建当前轮的 prompt ──
            if round_idx == 0:
                # 第一轮: 使用 phase1 prompt
                kwargs = dict(phase1_prompt_kwargs)
                kwargs["queried_code"] = kwargs.get("queried_code", 
                    "(暂无 — 这是第一轮, 请提出你想查询的代码)")
                prompt = phase1_prompt_template.format(**kwargs)
            else:
                # 后续轮: 使用 phase2 prompt
                kwargs = {
                    "query_results": query_results_this_round,
                    "previous_observation": observation_summary,
                    "previous_analysis_direction": analysis_direction or "待确认",
                }
                kwargs.update(phase2_extra_kwargs)
                prompt = phase2_prompt_template.format(**kwargs)
            
            # ── 添加回滚黑名单 ──
            rollback_warning = self.iter_memory.build_rollback_aware_context()
            if rollback_warning:
                prompt += rollback_warning
            
            # ── 调用 LLM ──
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            
            if response is None:
                print(f"     ✗ [Fix Query] LLM 调用失败 at round {round_idx + 1}")
                if round_idx == 0:
                    return None  # 第一轮就失败 → 直接返回
                break
            
            # ── 解析 LLM 回复 ──
            parsed = self._parse_query_response(response)
            
            if parsed is None:
                # 解析失败 → 尝试直接作为提案解析
                proposal = self._parse_proposal_response(response)
                if proposal:
                    print(f"     ✓ [Fix Query] Parsed as proposal directly at round {round_idx + 1}")
                    return proposal
                print(f"     ✗ [Fix Query] Response parse failed at round {round_idx + 1}")
                if round_idx == 0:
                    return None
                break
            
            phase = parsed.get("phase", "")
            
            # ── 记录观察和分析 ──
            observation_summary = parsed.get("observation", observation_summary)
            reasoning_summary = parsed.get("reasoning", reasoning_summary)
            analysis_direction = parsed.get("analysis_direction", "")
            
            if phase == "proposal":
                # ── LLM 已经做出修复提案! ──
                print(f"     ✓ [Fix Query] LLM produced fix proposal after {round_idx + 1} round(s)")
                print(f"       Diagnosis: {reasoning_summary[:120]}")
                return {
                    "structural_changes": parsed.get("structural_changes", []),
                    "param_changes": parsed.get("param_changes", {}),
                    "explanation": parsed.get("rationale", "") or parsed.get("message", "") or reasoning_summary,
                    "analysis": observation_summary + "\n" + reasoning_summary,
                    "_query_mode_used": True,
                    "_query_rounds": round_idx + 1,
                    "_queried_code_chars": len(all_queried_code),
                    "_raw_response": response,  # 保留原始 LLM 响应, 用于回退解析
                }
            
            elif phase == "query":
                # ── LLM 提出查询请求 ──
                queries = parsed.get("queries", [])
                if not queries:
                    print(f"     ⚠ No queries in response, forcing next round")
                    continue
                
                print(f"     🔎 LLM queries: {len(queries)} requests")
                # ── 兼容 LLM 输出的简化格式: strings → dicts ──
                # LLM 有时会输出 queries 为纯字符串列表 (如 ["SASRec.__init__"]),
                # 而不是结构化的 [{"action": "search_function", "args": {"name": "SASRec.__init__}}]
                normalized_queries = []
                for q in queries:
                    if isinstance(q, str):
                        # ── 清洗 LLM 格式污染 ──
                        # LLM 有时将 SEARCH/REPLACE edit 格式与查询格式混淆,
                        # 输出 "SEARCH: class SASRec" 或 "SEARCH: SASRec.__init__" 等
                        import re as regex_module
                        cleaned_q = q.strip()
                        # 移除 "SEARCH:" / "REPLACE:" / "class " / "def " 等前缀
                        cleaned_q = regex_module.sub(
                            r'^\s*(SEARCH|REPLACE)\s*:\s*(class\s+|def\s+)?', '',
                            cleaned_q, flags=regex_module.IGNORECASE
                        ).strip()
                        # 移除包裹的引号
                        cleaned_q = cleaned_q.strip('"\'`<>')
                        # 自动将字符串转为 search_function 查询
                        normalized_queries.append({"action": "search_function", "args": {"name": cleaned_q}})
                        print(f"       → search_function(name={cleaned_q})  [auto-converted from string{' (cleaned from: '+q+')' if cleaned_q != q else ''}]")
                    elif isinstance(q, dict):
                        action = q.get("action", "?")
                        args = q.get("args", {})
                        print(f"       → {action}({args})")
                        normalized_queries.append(q)
                    else:
                        print(f"       ⚠ Skipping invalid query item: {type(q)} = {q}")
                queries = normalized_queries
                if not queries:
                    print(f"     ⚠ No valid queries after normalization, forcing next round")
                    continue
                
                # ── 执行查询 ──
                query_results_this_round = self._code_query_tool.execute_queries(queries)
                
                # 限制单次查询结果大小
                max_chars = self.config.code_query_max_chars_per_result
                if len(query_results_this_round) > max_chars:
                    query_results_this_round = query_results_this_round[:max_chars] + \
                        f"\n\n⚠ 查询结果超过 {max_chars} 字符, 已截断。如需更多细节, 请使用 get_region 获取具体行范围。"
                
                all_queried_code += "\n\n" + query_results_this_round
                
                # 刷新缓存 (源码可能已被修改)
                self._code_query_tool.refresh_cache()
                
                print(f"       Results: {len(query_results_this_round)} chars returned")
            
            else:
                # 未知 phase → 尝试作为提案解析
                print(f"     ⚠ Unknown phase '{phase}' in fix query response")
                proposal = self._parse_proposal_response(response)
                if proposal:
                    return proposal
                if round_idx == 0:
                    return None
                break
        
        # ── 查询轮数用尽 → 强制输出修复方案 ──
        print(f"  📢 [Fix Query] Max rounds reached, forcing final proposal")
        kwargs = {
            "all_queried_code": all_queried_code,
            "observation_summary": observation_summary,
            "reasoning_summary": reasoning_summary,
            "current_hidden_size": self.adapter.base_args.get("hidden_size", 64),
        }
        kwargs.update(final_extra_kwargs)
        final_prompt = final_prompt_template.format(**kwargs)
        
        # 添加回滚黑名单
        rollback_warning = self.iter_memory.build_rollback_aware_context()
        if rollback_warning:
            final_prompt += rollback_warning
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt},
            ],
            temperature=temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        
        if response is None:
            return None
        
        # 尝试解析为提案
        proposal = self._parse_proposal_response(response)
        if proposal:
            proposal["_query_mode_used"] = True
            proposal["_query_rounds"] = max_query_rounds
            proposal["_queried_code_chars"] = len(all_queried_code)
            return proposal
        
        # 尝试从 query 格式中提取 proposal
        parsed = self._parse_query_response(response)
        if parsed and parsed.get("phase") == "proposal":
            return {
                "structural_changes": parsed.get("structural_changes", []),
                "param_changes": parsed.get("param_changes", {}),
                "explanation": parsed.get("rationale", ""),
                "analysis": observation_summary + "\n" + reasoning_summary,
                "_query_mode_used": True,
                "_query_rounds": max_query_rounds,
                "_queried_code_chars": len(all_queried_code),
            }
        
        return None

    def _preflight_fix_and_retry(self, preflight_result: dict, iteration: int) -> Optional[dict]:
        """
        预检修复: 当源码有语法错误时, 让 LLM 修复并验证
        
        Args:
            preflight_result: 预检结果 (包含语法错误详情)
            iteration: 当前迭代轮次
            
        Returns:
            Dict: 修复后的训练结果 (如果修复成功且训练通过)
            None: 修复失败
        """
        errors = preflight_result.get("errors", [])
        if not errors:
            return None
        
        max_fix_rounds = self._max_self_correction_rounds
        
        for round_idx in range(1, max_fix_rounds + 1):
            print(f"\n  🔁 预检修复第 {round_idx}/{max_fix_rounds} 轮")
            
            # ── 构建错误详情 ──
            errors_detail = ""
            for err in errors:
                errors_detail += f"\n### 错误: {err.get('file', '?')}\n"
                errors_detail += f"- **错误类型**: {err.get('error_type', 'SyntaxError')}\n"
                errors_detail += f"- **行号**: {err.get('line', '?')}\n"
                errors_detail += f"- **错误消息**: {err.get('error_message', '')[:300]}\n"
                if err.get('text'):
                    errors_detail += f"- **出错行代码**: {err.get('text', '')[:100]}\n"
            
            # ── 根据 query 模式选择不同的修复流程 ──
            if self._code_query_enabled:
                fix_proposal = self._fix_with_query_mode(
                    phase1_prompt_template=QUERY_BASED_PREFLIGHT_FIX_PROMPT,
                    phase2_prompt_template=QUERY_BASED_FIX_PHASE2_PROMPT,
                    final_prompt_template=QUERY_BASED_FIX_FINAL_PROMPT,
                    phase1_prompt_kwargs={
                        "_preflight_errors": errors_detail,
                        "code_index": self._code_query_tool.build_code_index(),
                        "queried_code": "(暂无 — 这是第一轮, 请提出你想查询的代码)",
                        "rollback_warning": "",
                    },
                    phase2_extra_kwargs={
                        "_error_type": "SyntaxError",
                        "_error_message": errors_detail[:500],
                    },
                    final_extra_kwargs={
                        "_error_type": "SyntaxError",
                        "_error_message": errors_detail[:500],
                    },
                    system_prompt="你是一位 Python 语法修复专家。你可以通过代码查询工具按需获取源码详情, 然后基于精确的代码提出修复方案。修复后的代码必须语法正确, 不能有省略号或占位符。⚠ 重要: 如果你想修改代码, 请先查询要修改的代码, 确保 SEARCH/REPLACE 的 search 文本与源码完全匹配!",
                    temperature=0.2,
                    max_query_rounds=3,  # 语法修复通常简单, 只允许3轮查询
                )
            else:
                # ── 传统模式: 塞全部源码 ──
                source_code_ctx = self.adapter.build_source_code_context(
                    include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                    max_total_chars=15000,
                )
                
                prompt = PREFLIGHT_FIX_PROMPT.format(
                    _preflight_errors=errors_detail,
                    _current_source_code=source_code_ctx,
                )
                
                response = self.llm.chat(
                    messages=[
                        {"role": "system", "content": (
                            "你是一位 Python 语法修复专家。"
                            "你需要修复模型源码中的语法错误, 确保代码可以正常运行。"
                            "修复后的代码必须语法正确, 不能有省略号或占位符。"
                        )},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=self.config.llm_max_tokens,
                    suppress_response_log=True,  # 代码响应不输出到日志
                )
                
                if response is None:
                    print(f"     ✗ LLM 调用失败")
                    continue
                
                fix_proposal = self._parse_proposal_response(response)
            
            if fix_proposal is None:
                print(f"     ✗ 无法解析 LLM 修复方案")
                continue
            
            structural_fixes = fix_proposal.get("structural_changes", [])
            if not structural_fixes:
                print(f"     ⚠ LLM 没有给出结构修改方案")
                continue
            
            # ── 应用修复 ──
            fix_result = self.struct_applier.apply_structural_changes(structural_fixes)
            
            if fix_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                print(f"     ✓ 语法修复已应用!")
                
                # ── 重新预检 ──
                new_preflight = self.trainer.preflight_check(
                    list(self.adapter.SOURCE_FILE_MAP.keys())
                )
                
                if new_preflight["status"] == "PASS":
                    print(f"     ✓ 修复后预检通过! 可以开始训练")
                    
                    # ── 尝试训练验证 ──
                    train_result = self.trainer.run()
                    if train_result["status"] == "SUCCESS":
                        print(f"     ✓ 修复后训练成功!")
                        self.journal.record({
                            "iteration": iteration,
                            "status": "PREFLIGHT_FIX_SUCCESS",
                            "self_correction_round": round_idx,
                            "errors_fixed": errors,
                        })
                        return train_result
                    else:
                        # 修复语法后训练仍然失败 → 可能还有其他错误
                        # 交给后续的 run-verify-fix-retry 处理
                        print(f"     ✓ 语法修复成功, 但训练仍失败: {train_result.get('error', '')[:80]}")
                        print(f"     (将在训练阶段继续自纠错)")
                        self.journal.record({
                            "iteration": iteration,
                            "status": "PREFLIGHT_FIX_BUT_TRAIN_FAILED",
                            "self_correction_round": round_idx,
                            "errors_fixed": errors,
                            "train_error": train_result.get("error", "")[:200],
                        })
                        return train_result  # 返回训练结果, 让后续 run-verify-fix-retry 处理
                
                else:
                    # 修复后还有语法错误 → 继续下一轮修复
                    print(f"     ✗ 修复后仍有语法错误: {len(new_preflight['errors'])} issues")
                    errors = new_preflight["errors"]
                    # 回滚这次修复
                    self.struct_applier.rollback_last_changes()
                    # 刷新 query 缓存 (源码已回滚)
                    if self._code_query_enabled:
                        self._code_query_tool.refresh_cache()
                    continue
            
            elif fix_result["status"] == "ROLLBACK":
                print(f"     ↩ 语法修复被回滚: {fix_result.get('rollback_reason', '')[:80]}")
                if self._code_query_enabled:
                    self._code_query_tool.refresh_cache()
                continue
            
            else:
                print(f"     ✗ 语法修复全部失败")
                continue
        
        print(f"\n  ✗ 预检修复: 所有 {max_fix_rounds} 轮均失败")
        return None

    def _phase_run_verify_fix_retry(
        self,
        iteration: int,
        param_overrides: Optional[dict] = None,
        phase_name: str = "baseline",
        structural_changes: list = None,
        struct_result: dict = None,
        metrics_before: dict = None,
    ) -> dict:
        """
        ════════════════════════════════════════════════════════
        核心: 运行 → 验证 → 修复 → 重试 自纠错内循环
        ════════════════════════════════════════════════════════
        
        这是最关键的改进! 对应你说的:
        "每次运行结束不是先判断运行是否成功，从终端获取错误信息，
         如果错误的话，就修改代码，运行，不应该是这样吗"
        
        工作流程:
        1. 运行训练
        2. 立即检查是否成功
        3. 如果失败 → 从终端输出提取错误信息
        4. 分类错误: CODE_ERROR(修源码) / CONFIG_ERROR(调参数) / DATA_ERROR(修路径)
        5. 根据错误类型, 让 LLM 修复:
           - CODE_ERROR → 修改源码文件中的 bug → 重训
           - CONFIG_ERROR → 调整超参数 → 重训
           - DATA_ERROR → 修正路径/配置 → 重训
        6. 重复 1-5, 最多 max_rounds 次
        7. 全部失败 → 回退并返回最终错误
        
        Args:
            iteration: 当前迭代轮次
            param_overrides: 参数覆盖 (可选)
            phase_name: 阶段名称 (baseline/retrain, 用于日志)
            structural_changes: 结构修改 (仅用于 retrain 阶段)
            struct_result: 结构修改结果 (仅用于 retrain 阶段)
            metrics_before: 修改前指标 (仅用于 retrain 阶段)
            
        Returns:
            Dict: 最终的训练结果 (可能成功, 也可能失败)
                  如果成功, 格式与 trainer.run() 一致
                  如果失败, 包含 fix_attempts 字段记录自纠错尝试次数
        """
        max_rounds = self._max_code_fix_rounds
        
        # ── 第一次运行 ──
        print(f"  🔄 Running {phase_name} training...")
        train_result = self.trainer.run(param_overrides=param_overrides)
        
        if train_result["status"] == "SUCCESS":
            print(f"  ✓ {phase_name} training SUCCESS")
            return train_result
        
        # ── 训练失败 → 进入自纠错内循环! ──
        print(f"\n╔══════════════════════════════════════╗")
        print(f"  ║  🔁 进入自纠错内循环                    ║")
        print(f"  ║  训练失败 → 提取错误 → 修复 → 重试       ║")
        print(f"  ║  最多 {max_rounds} 轮自纠错            ║")
        print(f"  ╚══════════════════════════════════════╝")
        
        error_category = train_result.get("error_category", "UNKNOWN")
        fixable = train_result.get("fixable", True)
        
        # ── 如果 train_runner 未做分类 (error_category="UNKNOWN")，让 LLM 判断 ──
        if error_category == "UNKNOWN":
            print(f"  🔍 错误分类由 LLM 判断 (不再使用硬编码规则)")
            error_category = self._classify_error_with_llm(train_result)
            train_result["error_category"] = error_category
            print(f"  LLM 判断的错误分类: {error_category}")
        else:
            print(f"  错误分类: {error_category}")
        
        print(f"  是否可修复: {fixable}")
        print(f"  错误摘要: {train_result.get('error', '')[:150]}")
        
        # 如果 traceback 里有文件信息, 打印出来
        tb_details = train_result.get("traceback_details", {})
        if tb_details.get("files"):
            print(f"  Traceback 涉及的文件:")
            for f, l in zip(tb_details.get("files", []), tb_details.get("line_numbers", [])):
                print(f"    → {f} @ line {l}")
        if tb_details.get("offending_code_snippets"):
            print(f"  出错代码:")
            for snippet in tb_details.get("offending_code_snippets", []):
                print(f"    {snippet}")
        
        if not fixable:
            # 不可修复 → 直接返回失败结果
            print(f"  ⏭ 错误不可自动修复 (SYSTEM_ERROR), 跳过自纠错")
            train_result["fix_attempts"] = 0
            return train_result
        
        # ── 自纠错内循环 ──
        # ── BUG FIX #6: 添加策略适应机制 ──
        # 当连续多轮使用同一种修复方式都失败时, 自动切换策略:
        # - 连续 3 轮 code_fix_failed → 切换到 replace_class 策略
        # - 连续 5 轮同一种方式失败 → 尝试整文件重写
        consecutive_fix_failures = 0  # 连续修复失败计数
        last_fix_action = None        # 上次的修复 action

        for round_idx in range(1, max_rounds + 1):
            print(f"\n  ── 自纠错第 {round_idx}/{max_rounds} 轮 ──")

            # ── 如果 error_category 仍为 UNKNOWN，让 LLM 重新分类 ──
            if error_category == "UNKNOWN":
                print(f"  🔍 错误分类仍为 UNKNOWN, 让 LLM 重新判断")
                error_category = self._classify_error_with_llm(train_result)
                train_result["error_category"] = error_category
                print(f"  LLM 重新分类: {error_category}")

            # ── BUG FIX #6: 策略适应 — 连续失败时切换修复方式 ──
            if consecutive_fix_failures >= 3 and error_category == "CODE_ERROR":
                print(f"  🔄 策略适应: 连续 {consecutive_fix_failures} 轮 code_fix_failed → 尝试 replace_class 策略")
                fix_result = self._fix_code_error_with_class_replacement(
                    iteration=iteration,
                    train_error=train_result,
                    phase_name=phase_name,
                    round_idx=round_idx,
                    param_overrides=param_overrides,
                )
                if fix_result["action"] in ("code_fixed_and_retrained", "code_fixed_but_retrain_failed"):
                    consecutive_fix_failures = 0  # 成功 → 清零
                    if fix_result["action"] == "code_fixed_and_retrained":
                        print(f"  ✓✓ 策略切换成功! 用 replace_class 方式修复!")
                        fix_result["fix_attempts"] = round_idx
                        return fix_result
                    train_result = fix_result
                    error_category = fix_result.get("error_category", error_category)
                    continue
                else:
                    consecutive_fix_failures += 1
                    print(f"  ✗ replace_class 策略也失败")
                    # 继续下一轮, 但会尝试更激进的策略

            if consecutive_fix_failures >= 5 and error_category == "CODE_ERROR":
                print(f"  🔄 策略适应: 连续 {consecutive_fix_failures} 轮都失败 → 尝试整文件重写策略")
                fix_result = self._fix_code_error_whole_file(
                    iteration=iteration,
                    train_error=train_result,
                    phase_name=phase_name,
                    round_idx=round_idx,
                    param_overrides=param_overrides,
                )
                if fix_result["action"] in ("code_fixed_and_retrained", "code_fixed_but_retrain_failed"):
                    consecutive_fix_failures = 0
                    if fix_result["action"] == "code_fixed_and_retrained":
                        print(f"  ✓✓ 整文件重写策略成功!")
                        fix_result["fix_attempts"] = round_idx
                        return fix_result
                    train_result = fix_result
                    error_category = fix_result.get("error_category", error_category)
                    continue
                else:
                    consecutive_fix_failures += 1

            # ── 根据错误类型选择修复策略 ──
            if error_category == "CODE_ERROR":
                # ════════════════════════════════════
                # CODE_ERROR: 源码有 bug → 让 LLM 修复源码!
                # ════════════════════════════════════
                print(f"  🔧 修复策略: 修改源码中的代码 bug")

                fix_result = self._fix_code_error(
                    iteration=iteration,
                    train_error=train_result,
                    phase_name=phase_name,
                    round_idx=round_idx,
                    param_overrides=param_overrides,
                )

                # ── BUG FIX #6: 跟踪连续失败 ──
                if fix_result["action"] == "code_fix_failed":
                    if last_fix_action == "code_fix_failed":
                        consecutive_fix_failures += 1
                    else:
                        consecutive_fix_failures = 1
                elif fix_result["action"] in ("code_fixed_and_retrained", "code_fixed_but_retrain_failed"):
                    consecutive_fix_failures = 0
                last_fix_action = fix_result["action"]

                if fix_result["action"] == "code_fixed_and_retrained":
                    # 源码修复成功 + 重新训练成功!
                    print(f"  ✓✓ 自纠错成功! 代码bug修复 + 重训通过!")
                    fix_result["fix_attempts"] = round_idx
                    return fix_result

                elif fix_result["action"] == "code_fixed_but_retrain_failed":
                    # 源码修复成功, 但重新训练仍然失败 → 用新错误信息继续自纠错
                    print(f"  ✓ 代码bug已修复, 但重训仍失败")
                    print(f"     新错误: {fix_result.get('error', '')[:100]}")
                    train_result = fix_result
                    error_category = fix_result.get("error_category", error_category)
                    continue

                elif fix_result["action"] == "code_fix_failed":
                    # LLM 修复源码失败 → 回滚, 继续自纠错
                    print(f"  ✗ 代码bug修复失败 (连续失败 {consecutive_fix_failures} 次)")
                    continue

                elif fix_result["action"] == "skip":
                    # 无法修复 → 退出自纠错
                    print(f"  ⏭ 无法修复此错误")
                    train_result["fix_attempts"] = round_idx
                    return train_result

            elif error_category == "CONFIG_ERROR":
                # ════════════════════════════════════
                # CONFIG_ERROR: 参数配置有问题 → 让 LLM 调参!
                # ════════════════════════════════════
                print(f"  🔧 修复策略: 调整训练参数配置")

                fix_result = self._fix_config_error(
                    iteration=iteration,
                    train_error=train_result,
                    phase_name=phase_name,
                    param_overrides=param_overrides,
                    round_idx=round_idx,
                )

                if fix_result["action"] == "config_fixed_and_retrained":
                    print(f"  ✓✓ 自纠错成功! 参数调整 + 重训通过!")
                    fix_result["fix_attempts"] = round_idx
                    return fix_result

                elif fix_result["action"] == "config_fixed_but_retrain_failed":
                    print(f"  ✓ 参数已调整, 但重训仍失败")
                    train_result = fix_result
                    error_category = fix_result.get("error_category", error_category)
                    continue

                elif fix_result["action"] == "config_fix_failed":
                    print(f"  ✗ 参数调整失败")
                    continue

            elif error_category == "DATA_ERROR":
                # ════════════════════════════════════
                # DATA_ERROR: 数据路径有问题 → 让 LLM 修正路径!
                # ════════════════════════════════════
                print(f"  🔧 修复策略: 修正数据路径/配置")

                fix_result = self._fix_data_error(
                    iteration=iteration,
                    train_error=train_result,
                    phase_name=phase_name,
                    round_idx=round_idx,
                )

                if fix_result["action"] == "data_fixed_and_retrained":
                    print(f"  ✓✓ 自纠错成功! 数据路径修复 + 重训通过!")
                    fix_result["fix_attempts"] = round_idx
                    return fix_result

                elif fix_result["action"] == "data_fixed_but_retrain_failed":
                    print(f"  ✓ 数据路径已修复, 但重训仍失败")
                    train_result = fix_result
                    error_category = fix_result.get("error_category", error_category)
                    continue

                else:
                    print(f"  ✗ 数据路径修复失败")
                    continue

            else:
                # UNKNOWN / SYSTEM_ERROR → 无法自动修复
                print(f"  ⏭ 错误类型无法自动修复: {error_category}")
                train_result["fix_attempts"] = round_idx
                return train_result
        
        # ── 所有自纠错轮次都失败了 ──
        print(f"\n  ✗ 自纠错内循环: 所有 {max_rounds} 轮均失败")
        train_result["fix_attempts"] = max_rounds
        
        # 如果是 retrain 阶段, 回滚结构修改
        if phase_name == "retrain" and structural_changes and \
           struct_result and struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
            print(f"  ↩ 回滚结构修改")
            self.struct_applier.rollback_last_changes()
            self.iter_memory.record_rollback(
                iteration=iteration,
                reason=f"All {max_rounds} self-correction rounds failed in retrain",
            )
        
        self.journal.record({
            "iteration": iteration,
            "status": f"RUN_VERIFY_FIX_RETRY_ALL_FAILED_{phase_name}",
            "max_rounds": max_rounds,
            "final_error_category": error_category,
            "final_error": train_result.get("error", "")[:500],
        })
        
        return train_result

    # ════════════════════════════════════════
    # 错误修复子方法 (按错误类型分类)
    # ════════════════════════════════════════

    def _fix_code_error(self, iteration: int, train_error: dict, 
                        phase_name: str, round_idx: int,
                        param_overrides: Optional[dict] = None) -> dict:
        """
        修复 CODE_ERROR: 源码中有 bug (语法错误、运行时错误、维度不匹配等)
        
        让 LLM 根据 traceback 信息直接修改源码文件, 然后重新训练验证。
        当 enable_code_query=True 时, 使用查询模式让 LLM 按需获取出错代码。
        
        Returns:
            {
                "action": "code_fixed_and_retrained" | "code_fixed_but_retrain_failed" | "code_fix_failed" | "skip",
                ...其他字段同 train_result 格式
            }
        """
        # ── 构建 traceback 详情 ──
        tb_details = train_error.get("traceback_details", {})
        traceback_text = tb_details.get("traceback_text", "")
        error_type = tb_details.get("error_type", train_error.get("error_category", "UNKNOWN"))
        error_message = tb_details.get("error_message", train_error.get("error", ""))
        files = tb_details.get("files", [])
        line_numbers = tb_details.get("line_numbers", [])
        offending_snippets = tb_details.get("offending_code_snippets", [])
        
        # 构建结构化的 traceback 展示
        tb_display = ""
        if traceback_text:
            tb_display += f"\n### 完整 Traceback:\n```\n{traceback_text[:2000]}\n```"
        if files:
            tb_display += f"\n### 出错文件和行号:"
            for f, l in zip(files, line_numbers):
                tb_display += f"\n- **文件**: `{f}` @ **行 {l}**"
        if offending_snippets:
            tb_display += f"\n### 出错代码片段:"
            for snippet in offending_snippets:
                tb_display += f"\n```python\n{snippet}\n```"
        if not tb_display:
            tb_display = f"\n### 错误信息:\n```\n{train_error.get('error', '')[:1500]}\n```"
        
        # ── 根据 query 模式选择不同的修复流程 ──
        if self._code_query_enabled:
            # ── 查询模式: LLM 按需获取出错代码 (核心新增!) ──
            fix_proposal = self._fix_with_query_mode(
                phase1_prompt_template=QUERY_BASED_FIX_PHASE1_PROMPT,
                phase2_prompt_template=QUERY_BASED_FIX_PHASE2_PROMPT,
                final_prompt_template=QUERY_BASED_FIX_FINAL_PROMPT,
                phase1_prompt_kwargs={
                    "_error_type": error_type,
                    "_error_category": train_error.get("error_category", "CODE_ERROR"),
                    "_error_message": error_message[:2000],
                    "_traceback_details": tb_display,
                    "code_index": self._code_query_tool.build_code_index(),
                    "queried_code": "(暂无 — 这是第一轮, 请提出你想查询的代码)",
                    "rollback_warning": self.iter_memory.build_rollback_aware_context(),
                    "current_hidden_size": self.adapter.base_args.get("hidden_size", 64),
                },
                phase2_extra_kwargs={
                    "_error_type": error_type,
                    "_error_message": error_message[:500],
                },
                final_extra_kwargs={
                    "_error_type": error_type,
                    "_error_message": error_message[:500],
                },
                system_prompt=(
                    "你是一位 Python 代码调试专家，擅长从 traceback 信息中精准定位 bug 并修复。"
                    "你可以通过代码查询工具按需获取源码详情, 然后基于精确的代码提出修复方案。"
                    "修正后的代码必须: 1)语法正确 2)维度与 hidden_size 对齐 "
                    "3)import完整 4)与现有代码兼容 5)是可执行的完整代码，无省略号。"
                    "⚠ 重要: 你必须根据 traceback 信息精准定位, 不要猜测! "
                    "如果你想修改代码, 请先查询要修改的代码, 确保 SEARCH/REPLACE 的 search 文本与源码完全匹配!"
                ),
                temperature=0.2,
                max_query_rounds=3,
            )
            
            if fix_proposal is None:
                return {"action": "code_fix_failed", "error": "Query mode LLM call failed or unparseable"}
            
            # 尝试用 LLMFixer 修复格式 (如果 query 模式返回的不标准)
            if not fix_proposal.get("structural_changes") and not fix_proposal.get("param_changes"):
                # 可能 query 模式的输出格式有小问题
                # 检查是否有原始的顶层 "edits" 被 _parse_query_response 遗漏
                print(f"     ⚠ Query mode returned no fixes, trying format fix...")
                print(f"       Available keys in fix_proposal: {list(fix_proposal.keys())}")
                # 尝试直接把整个 proposal 当作 _parse_proposal_response 解析
                raw_response = fix_proposal.get("_raw_response", "")
                if raw_response:
                    print(f"       Trying _parse_proposal_response on raw response...")
                    fallback_proposal = self._parse_proposal_response(raw_response)
                    if fallback_proposal and (fallback_proposal.get("structural_changes") or fallback_proposal.get("param_changes")):
                        print(f"       ✓ Fallback parse succeeded!")
                        fix_proposal = fallback_proposal
        
        else:
            # ── 传统模式: 塞全部源码 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
                iterative_memory=self.iter_memory,
            )
            
            # ── 构建 CODE_FIX_PROMPT ──
            prompt = CODE_FIX_PROMPT.format(
                _error_type=error_type,
                _error_category=train_error.get("error_category", "CODE_ERROR"),
                _error_message=error_message[:2000],
                _traceback_details=tb_display,
                _offending_code_snippets="\n".join(offending_snippets) if offending_snippets else "(未提取到出错代码片段)",
                _current_source_code=source_code_ctx,
                current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
            )
            
            # ── 添加回滚黑名单 ──
            rollback_warning = self.iter_memory.build_rollback_aware_context()
            if rollback_warning:
                prompt += rollback_warning
            
            # ── 调用 LLM ──
            logger.info(f"Code fix round {round_idx}: sending traceback to LLM")
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位 Python 代码调试专家，擅长从 traceback 信息中精准定位 bug 并修复。"
                        "你需要根据 traceback 的文件名和行号定位出错的代码，然后给出修正后的完整代码。"
                        "修正后的代码必须: 1)语法正确 2)维度与 hidden_size 对齐 "
                        "3)import完整 4)与现有代码兼容 5)是可执行的完整代码，无省略号。"
                        "特别注意: 你必须根据 traceback 信息精准定位, 不要猜测!"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # 代码修复用极低温度
                max_tokens=self.config.llm_max_tokens,
                suppress_response_log=True,  # 代码响应不输出到日志
            )
            
            if response is None:
                return {"action": "code_fix_failed", "error": "LLM call failed"}
            
            # ── 解析修复方案 ──
            fix_proposal = self._parse_proposal_response(response)
            if fix_proposal is None:
                print(f"     ✗ 无法解析 LLM 代码修复方案")
                # 尝试用 LLMFixer 修复格式
                fixed_response = self.fixer.fix_format(response, "无法从回复中提取有效JSON")
                fix_proposal = self._parse_proposal_response(fixed_response)
                if fix_proposal is None:
                    return {"action": "code_fix_failed", "error": "Cannot parse fix proposal"}
        
        structural_fixes = fix_proposal.get("structural_changes", [])
        param_fixes = fix_proposal.get("param_changes", {})
        fix_explanation = fix_proposal.get("explanation", "")
        
        print(f"     💡 修复诊断: {fix_explanation}")
        if structural_fixes:
            print(f"     🏗️ 源码修复: {len(structural_fixes)} 处修改")
            for sc in structural_fixes:
                print(f"       → [{sc.get('target_file', '?')}] "
                      f"{sc.get('target_class_or_function', '?')}: "
                      f"{sc.get('description', '?')}")
        if param_fixes:
            print(f"     🔧 参数修复: {json.dumps(param_fixes, ensure_ascii=False)}")
        
        if not structural_fixes and not param_fixes:
            print(f"     ⚠ LLM 没有给出任何修复方案")
            return {"action": "code_fix_failed", "error": "No fixes proposed"}
        
        # ── 应用源码修复 ──
        # ── BUG FIX #4: 收集结构应用器的内部诊断信息 ──
        apply_diagnostics = []  # 收集每个 change 的诊断信息
        if structural_fixes:
            apply_result = self.struct_applier.apply_structural_changes(structural_fixes)
           
            # ── 收集诊断信息 ──
            for applied_entry in apply_result.get("applied_changes", []):
                diag = applied_entry.get("result", {}).get("diagnostics", {})
                if diag:
                    apply_diagnostics.append(diag)
            for failed_entry in apply_result.get("failed_changes", []):
                diag = failed_entry.get("error", "")
                apply_diagnostics.append({"failure": diag})

            if apply_result["status"] == "ROLLBACK":
                rollback_reason = apply_result.get("rollback_reason", "")
                validation_errors = apply_result.get("validation_results", {}).get("errors", [])
                
                # ── BUG FIX #4: 回滚原因中包含结构应用器的内部诊断 ──
                detailed_reason = f"Code fix round {round_idx} rolled back: {rollback_reason}"
                if apply_diagnostics:
                    diag_summary = " | ".join([
                        f"{d.get('target_name', '?')}: method={d.get('replacement_method', '?')}, "
                        f"reason={d.get('failure_reason', d.get('failure', '?'))}"
                        for d in apply_diagnostics[:3]
                    ])
                    detailed_reason += f" | Diagnostics: {diag_summary}"
                if validation_errors:
                    detailed_reason += f" | Validation errors: {'; '.join(validation_errors)}"
                
                print(f"     ↩ 代码修复被回滚: {rollback_reason}")
                # 记录到 IterativeMemory — 包含详细诊断!
                self.iter_memory.record_rollback(
                    iteration=iteration,
                    reason=detailed_reason,
                )
                return {"action": "code_fix_failed",
                        "error": apply_result.get("rollback_reason", "validation failed"),
                        "apply_diagnostics": apply_diagnostics}
            
            elif apply_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                print(f"     ✓ 源码修复已应用!")
                # 记录修改到 IterativeMemory
                self.iter_memory.record_modification(
                    iteration=iteration,
                    structural_changes=structural_fixes,
                    apply_result=apply_result,
                    metrics_before=None,
                    metrics_after=None,  # 还没训练, 不知道效果
                    note=f"Code fix round {round_idx} for {phase_name}",
                )
            
            elif apply_result["status"] == "ALL_FAILED":
                print(f"     ✗ 所有源码修复都失败")
                return {"action": "code_fix_failed", "error": "All structural fixes failed"}
        
        # ── 重新训练验证 ──
        print(f"     🔄 修复后重新训练验证...")
        
        # 合并参数修复
        combined_params = {}
        if param_overrides:
            combined_params.update(param_overrides)
        if param_fixes:
            validated_fixes = self._validate_param_changes(param_fixes)
            if validated_fixes:
                combined_params.update(validated_fixes)
        
        retrain_result = self.trainer.run(
            param_overrides=combined_params if combined_params else None,
        )
        
        if retrain_result["status"] == "SUCCESS":
            # 修复成功! 返回训练结果
            retrain_result["action"] = "code_fixed_and_retrained"
            retrain_result["fix_explanation"] = fix_explanation
            retrain_result["structural_fixes"] = structural_fixes
            retrain_result["param_fixes"] = param_fixes
            
            self.journal.record({
                "iteration": iteration,
                "status": f"CODE_FIX_SUCCESS_{phase_name}",
                "self_correction_round": round_idx,
                "fix_explanation": fix_explanation,
                "structural_fixes_summary": json.dumps(structural_fixes, ensure_ascii=False),
                "retrain_metrics": retrain_result.get("metrics", {}),
            })
            
            return retrain_result
        
        else:
            # 修复后训练仍然失败 → 用新错误信息继续自纠错
            print(f"     ✗ 修复后重训仍失败: {retrain_result.get('error', '')}")
            print(f"     新错误分类: {retrain_result.get('error_category', 'UNKNOWN')}")
            
            # 如果新错误也是 CODE_ERROR → 可能需要修复不同的 bug
            # 如果新错误是 CONFIG_ERROR → 切换到参数修复策略
            
            retrain_result["action"] = "code_fixed_but_retrain_failed"
            retrain_result["fix_explanation"] = fix_explanation
            
            self.journal.record({
                "iteration": iteration,
                "status": f"CODE_FIX_BUT_RETRAIN_FAILED_{phase_name}",
                "self_correction_round": round_idx,
                "fix_explanation": fix_explanation,
                "retrain_error": retrain_result.get("error", ""),
                "retrain_error_category": retrain_result.get("error_category", ""),
            })
            
            return retrain_result

    def _fix_code_error_with_class_replacement(
        self, iteration: int, train_error: dict,
        phase_name: str, round_idx: int,
        param_overrides: Optional[dict] = None) -> dict:
        """
        BUG FIX #6: 策略适应 — 用 replace_class 代替 replace_function
        
        当连续多轮 replace_function 失败时 (通常是因为 structure_applier
        无法找到目标方法), 切换到 replace_class 策略, 直接替换整个类定义。
        这绕过了方法级定位的困难。
        """
        print(f"  🏗️ 策略切换: 用 replace_class 替代 replace_function")
        
        tb_details = train_error.get("traceback_details", {})
        error_message = tb_details.get("error_message", train_error.get("error", ""))
        traceback_text = tb_details.get("traceback_text", "")
        error_type = tb_details.get("error_type", "CODE_ERROR")
        
        # ── 根据 query 模式选择不同的修复流程 ──
        if self._code_query_enabled:
            fix_proposal = self._fix_with_query_mode(
                phase1_prompt_template=QUERY_BASED_FIX_PHASE1_PROMPT,
                phase2_prompt_template=QUERY_BASED_FIX_PHASE2_PROMPT,
                final_prompt_template=QUERY_BASED_FIX_FINAL_PROMPT,
                phase1_prompt_kwargs={
                    "_error_type": error_type,
                    "_error_category": "CODE_ERROR",
                    "_error_message": error_message[:2000],
                    "_traceback_details": f"\n### Traceback:\n```\n{traceback_text[:2000]}\n```",
                    "code_index": self._code_query_tool.build_code_index(),
                    "queried_code": "(暂无 — 这是第一轮, 请提出你想查询的代码)",
                    "rollback_warning": self.iter_memory.build_rollback_aware_context(),
                    "current_hidden_size": self.adapter.base_args.get("hidden_size", 64),
                },
                phase2_extra_kwargs={
                    "_error_type": error_type,
                    "_error_message": error_message[:500],
                },
                final_extra_kwargs={
                    "_error_type": error_type,
                    "_error_message": error_message[:500],
                },
                system_prompt=(
                    "你是一位 Python 代码调试专家。"
                    "连续多轮尝试替换单个方法失败, 现在需要你输出完整的类定义代码。"
                    "你可以通过代码查询工具按需获取源码详情, 然后基于精确的代码提出修复方案。"
                    "你必须输出包含类头 (class XXX(nn.Module):) 和所有方法的完整代码。"
                    "代码必须语法正确、维度对齐、import完整、与现有代码兼容。"
                    "⚠ 重要: 先查询你要修改的类代码, 确保 SEARCH/REPLACE 的 search 文本与源码完全匹配!"
                ),
                temperature=0.2,
                max_query_rounds=3,
            )
            
            if fix_proposal is None:
                return {"action": "code_fix_failed", "error": "Query mode LLM call failed or unparseable"}
        else:
            # ── 传统模式: 塞全部源码 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
                iterative_memory=self.iter_memory,
            )
            
            # 构建 prompt — 明确告知 LLM 输出完整的 class 定义
            prompt = CODE_FIX_PROMPT.format(
                _error_type=error_type,
                _error_category="CODE_ERROR",
                _error_message=error_message[:2000],
                _traceback_details=f"\n### Traceback:\n```\n{traceback_text[:2000]}\n```",
                _offending_code_snippets="\n".join(tb_details.get("offending_code_snippets", [])) or "(未提取到)",
                _current_source_code=source_code_ctx,
                current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
            )
            
            rollback_warning = self.iter_memory.build_rollback_aware_context()
            if rollback_warning:
                prompt += rollback_warning
            
            # 明确告知 LLM: 使用 SEARCH/REPLACE 格式, 输出完整的类定义
            prompt += (
                "\n\n### ⚠ 重要修改: 请使用 SEARCH/REPLACE 格式, 输出完整类定义!"
                "\n之前的修改策略连续失败, 因为代码定位困难。"
                "\n这次请**使用 SEARCH/REPLACE 格式**, search 部分写出原始的完整类定义代码, "
                "\nreplace 部分写出修改后的完整类定义代码。"
                "\n确保类定义中的所有方法都完整、语法正确、无省略号。"
            )
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位 Python 代码调试专家。"
                        "连续多轮尝试替换单个方法失败, 现在需要你输出完整的类定义代码。"
                        "你必须输出包含类头 (class XXX(nn.Module):) 和所有方法的完整代码。"
                        "代码必须语法正确、维度对齐、import完整、与现有代码兼容。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=self.config.llm_max_tokens,
                suppress_response_log=True,  # 代码响应不输出到日志
            )
            
            if response is None:
                return {"action": "code_fix_failed", "error": "LLM call failed"}
            
            fix_proposal = self._parse_proposal_response(response)
            if fix_proposal is None:
                return {"action": "code_fix_failed", "error": "Cannot parse fix proposal"}
        
        structural_fixes = fix_proposal.get("structural_changes", [])
        fix_explanation = fix_proposal.get("explanation", "")
        
        # 强制将所有 fix 的 insert_position 设置为 replace_class
        for sc in structural_fixes:
            target_name = sc.get("target_class_or_function", "")
            if "." in target_name:
                # "SelfAttention.__init__" → 只取类名部分用于 replace_class
                sc["target_class_or_function"] = target_name.split(".", 1)[0]
            sc["insert_position"] = "replace_class"
        
        print(f"     💡 策略切换修复: {fix_explanation[:150]}")
        print(f"     🏗️ 使用 replace_class 策略: {len(structural_fixes)} 处修改")
        for sc in structural_fixes:
            print(f"       → [{sc.get('target_file', '?')}] "
                  f"{sc.get('target_class_or_function', '?')}: "
                  f"{sc.get('description', '?')[:60]}...")
        
        if not structural_fixes:
            return {"action": "code_fix_failed", "error": "No fixes proposed"}
        
        apply_result = self.struct_applier.apply_structural_changes(structural_fixes)
        
        if apply_result["status"] == "ROLLBACK":
            print(f"     ↩ replace_class 策略也被回滚: {apply_result.get('rollback_reason', '')[:100]}")
            self.iter_memory.record_rollback(
                iteration=iteration,
                reason=f"replace_class strategy round {round_idx} rolled back: {apply_result.get('rollback_reason', '')[:200]}",
            )
            return {"action": "code_fix_failed", "error": apply_result.get("rollback_reason", "")}
        
        elif apply_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
            print(f"     ✓ replace_class 策略修复已应用!")
        
        elif apply_result["status"] == "ALL_FAILED":
            return {"action": "code_fix_failed", "error": "replace_class: All fixes failed"}
        
        # 重新训练验证
        retrain_result = self.trainer.run(param_overrides=param_overrides)
        
        if retrain_result["status"] == "SUCCESS":
            retrain_result["action"] = "code_fixed_and_retrained"
            retrain_result["fix_explanation"] = f"[replace_class策略] {fix_explanation}"
            return retrain_result
        
        retrain_result["action"] = "code_fixed_but_retrain_failed"
        retrain_result["fix_explanation"] = f"[replace_class策略] {fix_explanation}"
        return retrain_result

    def _fix_code_error_whole_file(
        self, iteration: int, train_error: dict,
        phase_name: str, round_idx: int,
        param_overrides: Optional[dict] = None) -> dict:
        """
        BUG FIX #6: 策略适应 — 整文件重写
        
        当所有其他策略都失败时, 让 LLM 输出整个文件的完整代码,
        直接替换整个文件。这是最激进但也是最可靠的方式。
        
        当 enable_code_query=True 时, 不塞截断的全部源码, 
        而是通过 read_file 查询获取出错文件的完整内容,
        并允许 LLM 查询其他相关文件来辅助理解。
        """
        print(f"  📝 策略切换: 整文件重写")
        
        tb_details = train_error.get("traceback_details", {})
        error_message = tb_details.get("error_message", train_error.get("error", ""))
        traceback_text = tb_details.get("traceback_text", "")
        
        # 从 traceback 中确定出错的是哪个文件
        error_files = tb_details.get("files", [])
        # 默认取 SOURCE_FILE_MAP 中的第一个文件
        source_keys = list(self.adapter.SOURCE_FILE_MAP.keys())
        target_file = source_keys[0] if source_keys else "modules.py"
        if error_files:
            for ef in error_files:
                # 遍历所有可修改的源码文件，匹配 traceback 中出现的文件名
                for src_key in source_keys:
                    if src_key.replace(".py", "") in ef:
                        target_file = src_key
                        break
        
        # ── 读取当前文件内容 ──
        file_path = self.struct_applier._resolve_file_path(target_file)
        if not file_path:
            return {"action": "code_fix_failed", "error": f"Cannot find {target_file}"}
        
        if self._code_query_enabled:
            # 查询模式: 通过 query 工具获取完整文件内容 (不受截断!)
            # 同时自动查询其他相关文件来辅助 LLM 理解上下文
            current_file_content = self._code_query_tool.read_file(target_file)
            if not current_file_content:
                # fallback: 直接文件读取
                with open(file_path, 'r', encoding='utf-8') as f:
                    current_file_content = f.read()
            
            # 构建轻量索引, 供 LLM 查看其他文件的结构
            code_index = self._code_query_tool.build_code_index()
            # 在 prompt 中包含索引, 让 LLM 可以决定是否需要查询其他文件
            other_files_hint = f"\n## 其他源码文件索引 (你可以查询任何文件来理解上下文)\n{code_index}\n"
        else:
            # 传统模式: 直接读取 + 塞截断的全部源码 (虽然有 current_file_content 但还塞了截断版本)
            with open(file_path, 'r', encoding='utf-8') as f:
                current_file_content = f.read()
            other_files_hint = ""
            # 传统模式塞全部源码作额外上下文 (虽然主要用 current_file_content)
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
                iterative_memory=self.iter_memory,
            )
        
        prompt = (
            f"训练运行失败, 错误来自**{target_file}**中的代码 bug。\n"
            f"所有之前的修复策略 (replace_function, replace_class) 都连续失败。\n"
            f"现在需要你**输出 {target_file} 的完整修复后的代码**。\n\n"
            f"## 错误信息\n```\n{error_message[:1500]}\n```\n\n"
            f"## Traceback\n```\n{traceback_text[:2000]}\n```\n\n"
            f"## 当前 {target_file} 的内容\n```python\n{current_file_content}\n```\n\n"
            f"{other_files_hint}"
            f"## 修复要求\n"
            f"输出 {target_file} 的**完整代码** (不能有任何省略号)。\n"
            f"修复上述错误, 但保持所有其他类/函数不变。\n"
            f"确保: 1)语法正确 2)import完整 3)维度与 hidden_size={self.adapter.base_args.get('hidden_size', 64)} 对齐\n"
            f"4)所有关键类仍然存在\n\n"
            f"### 输出格式\n"
            f"输出完整的 Python 代码, 不要用 JSON 包裹, 不要有省略号:\n"
            f"```python\n完整的代码...\n```\n"
        )
        
        rollback_warning = self.iter_memory.build_rollback_aware_context()
        if rollback_warning:
            prompt += rollback_warning
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位 Python 代码修复专家。"
                    "请输出修复后的完整文件代码, 不能有省略号。"
                    "保持所有没有 bug 的类/函数不变, 只修复出错的部分。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=self.config.llm_max_tokens,
            suppress_response_log=True,  # 代码响应不输出到日志
        )
        
        if response is None:
            return {"action": "code_fix_failed", "error": "LLM call failed"}
        
        # 提取代码 (从 ```python ... ``` 或直接提取)
        new_file_content = StructureApplier.clean_new_code(response)
        
        # 语法检查
        try:
            ast.parse(new_file_content)
        except SyntaxError as e:
            logger.warning(f"Whole-file rewrite has syntax error: line {e.lineno}: {e.msg}")
            return {"action": "code_fix_failed", "error": f"Syntax error in rewritten file: line {e.lineno}: {e.msg}"}
        
        # 关键符号检查
        missing = []
        for sym in ["class SelfAttention", "class Intermediate", "class EncoderLayer"]:
            if target_file == "modules.py" and sym not in new_file_content:
                missing.append(sym)
        if missing:
            return {"action": "code_fix_failed", "error": f"Missing critical symbols: {missing}"}
        
        # 创建快照 + 写入
        snapshot = self.struct_applier._create_local_snapshot()
        if not snapshot["ok"]:
            return {"action": "code_fix_failed", "error": "Snapshot creation failed"}
        self.struct_applier._current_snapshot_id = snapshot["snapshot_id"]
        
        # 写入文件
        shutil_copy = shutil.copy2  # 先备份
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_file_content)
            logger.info(f"Whole-file rewrite applied to {file_path}")
        except Exception as e:
            self.struct_applier._local_rollback(snapshot["snapshot_id"])
            return {"action": "code_fix_failed", "error": f"Cannot write file: {e}"}
        
        # 验证 (语法 + 关键符号)
        validation = self.struct_applier._validate_all_modified_files([file_path])
        if not validation["all_passed"]:
            logger.warning(f"Whole-file rewrite validation failed: {validation['errors']}")
            self.struct_applier._local_rollback(snapshot["snapshot_id"])
            return {"action": "code_fix_failed", "error": f"Validation failed: {validation['errors']}"}
        
        print(f"     ✓ 整文件重写已应用并通过验证!")
        
        # 重新训练验证
        retrain_result = self.trainer.run(param_overrides=param_overrides)
        
        if retrain_result["status"] == "SUCCESS":
            retrain_result["action"] = "code_fixed_and_retrained"
            retrain_result["fix_explanation"] = f"[整文件重写策略] Fixed {target_file}"
            return retrain_result
        
        # 重训失败 → 回滚文件
        self.struct_applier._local_rollback(snapshot["snapshot_id"])
        retrain_result["action"] = "code_fixed_but_retrain_failed"
        retrain_result["fix_explanation"] = f"[整文件重写策略] Fixed {target_file} but retrain failed"
        return retrain_result

    def _fix_config_error(self, iteration: int, train_error: dict,
                          phase_name: str, param_overrides: Optional[dict],
                          round_idx: int) -> dict:
        """
        修复 CONFIG_ERROR: 参数配置有问题 (OOM、NaN等)
        
        使用现有的 _phase_train_diagnosis_and_retry 机制, 但增加了轮数。
        """
        diagnosis_result = self._phase_train_diagnosis_and_retry(
            iteration=iteration,
            train_error=train_error,
            max_rounds=2,  # 每轮 config fix 内部最多 2 次尝试
        )
        
        if diagnosis_result and diagnosis_result["status"] == "SUCCESS":
            diagnosis_result["action"] = "config_fixed_and_retrained"
            return diagnosis_result
        
        # 诊断也失败 → 返回失败结果
        return {
            "action": "config_fix_failed",
            "status": "FAILED",
            "error": train_error.get("error", ""),
            "error_category": "CONFIG_ERROR",
        }

    def _fix_data_error(self, iteration: int, train_error: dict,
                        phase_name: str, round_idx: int) -> dict:
        """
        修复 DATA_ERROR: 数据路径或格式有问题
        
        让 LLM 诊断数据路径问题并给出修复建议。
        """
        # ── 使用 train diagnosis prompt, 但着重数据路径问题 ──
        error_msg = train_error.get("error", "")
        
        # 构建项目配置
        project_config = json.dumps(self.adapter.base_args, indent=2, ensure_ascii=False)
        train_cmd = self.adapter.build_train_command()
        
        prompt = TRAIN_DIAGNOSIS_PROMPT.format(
            _error_type=f"DATA_ERROR — {train_error.get('error_category', 'DATA_ERROR')}",
            _error_message=error_msg[:2000],
            _project_config=project_config,
            _train_command=train_cmd,
        )
        
        # ── 调用 LLM ──
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位数据工程师，擅长诊断数据路径和格式问题。"
                    "请分析错误信息，判断是数据文件路径错误、文件格式错误还是配置参数错误。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        
        if response is None:
            return {"action": "data_fix_failed", "error": "LLM call failed"}
        
        # ── 解析诊断结果 ──
        try:
            json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
            if json_match:
                diagnosis = json.loads(json_match.group(1))
            else:
                start = response.find('{')
                end = response.rfind('}')
                if start >= 0 and end > start:
                    diagnosis = json.loads(response[start:end + 1])
                else:
                    return {"action": "data_fix_failed", "error": "Cannot parse diagnosis"}
        except json.JSONDecodeError:
            return {"action": "data_fix_failed", "error": "JSON parse error"}
        
        param_fixes = diagnosis.get("param_changes", {})
        
        if param_fixes:
            validated_fixes = self._validate_param_changes(param_fixes)
            if validated_fixes:
                print(f"     🔧 数据路径参数修复: {json.dumps(validated_fixes, ensure_ascii=False)[:200]}")
                retry_result = self.trainer.run(param_overrides=validated_fixes)
                
                if retry_result["status"] == "SUCCESS":
                    retry_result["action"] = "data_fixed_and_retrained"
                    return retry_result
                else:
                    retry_result["action"] = "data_fixed_but_retrain_failed"
                    return retry_result
        
        return {"action": "data_fix_failed", "error": "No data path fixes proposed"}

    def _rollback_all_source_changes(self):
        """
        回滚所有源码修改 — 当连续失败次数过多时, 回滚到初始状态
        
        从 IterativeMemory 的快照目录恢复第 0 轮的源码文件。
        """
        print(f"  ↩↩↩ 回滚所有源码修改到初始状态!")
        
        # 从 IterativeMemory 的快照目录恢复 iter_000 的源码
        snapshot_dir = self.iter_memory.snapshot_dir / "iter_000"
        if snapshot_dir.exists():
            for file_key in self.iter_memory.source_files:
                snapshot_path = snapshot_dir / file_key
                if snapshot_path.exists():
                    # 找到当前源码文件的实际路径
                    actual_path = self.iter_memory._find_source_file(file_key)
                    if actual_path and os.path.exists(actual_path):
                        try:
                            shutil.copy2(str(snapshot_path), actual_path)
                            logger.info(f"Restored {file_key} to initial state: {snapshot_path} → {actual_path}")
                            print(f"    ✓ {file_key} restored")
                        except Exception as e:
                            logger.warning(f"Failed to restore {file_key}: {e}")
                    else:
                        logger.warning(f"Source file not found for {file_key}")
        else:
            logger.warning("No iter_000 snapshot found for rollback")
            # 也使用 StructureApplier 的回滚作为后备
            self.struct_applier.rollback_last_changes()

    # ════════════════════════════════════════
    # 原有阶段实现 (保留)
    # ════════════════════════════════════════

    def _phase_train(self) -> dict:
        """训练阶段"""
        return self.trainer.run()

    def _phase_surprise_analysis(self, metrics: dict) -> Optional[dict]:
        """
        惊喜评估 + 错误案例分析
        
        这个阶段在每次训练完成后执行:
        1. 加载最新训练的模型 checkpoint
        2. 在训练子集、整体测试集、惊喜子集上评估
        3. 提取推理错误的案例并转化为文本
        4. LLM 分析错误案例，推理出模型瓶颈
        5. 返回完整的分析结果
        
        Returns:
            Dict containing evaluation_report, case_analysis, wrong_text_cases
            如果无法执行 (如 GPU 不可用)，返回 None
        """
        import sys
        import importlib
        
        logger.info("Starting surprise analysis phase...")
        
        try:
            # 动态导入推荐模型项目中的模块
            project_root = self.config.project_root
            sys.path.insert(0, project_root)
            
            # 添加子模块路径
            extractor_path = os.path.join(project_root, 'error_case_extractor.py')
            surprise_eval_path = os.path.join(project_root, 'surprise_eval.py')
            
            if not os.path.exists(extractor_path) or not os.path.exists(surprise_eval_path):
                logger.warning("Surprise analysis modules not found in project root, skipping")
                return None
            
            # 导入模块 (动态)
            import importlib.util
            
            # 导入 error_case_extractor
            spec_extract = importlib.util.spec_from_file_location("error_case_extractor", extractor_path)
            mod_extract = importlib.util.module_from_spec(spec_extract)
            spec_extract.loader.exec_module(mod_extract)
            
            # 导入 surprise_eval
            spec_eval = importlib.util.spec_from_file_location("surprise_eval", surprise_eval_path)
            mod_eval = importlib.util.module_from_spec(spec_eval)
            spec_eval.loader.exec_module(mod_eval)
            
            # 导入 utils (也需要从 project_root)
            utils_path = os.path.join(project_root, 'utils.py')
            if os.path.exists(utils_path):
                spec_utils = importlib.util.spec_from_file_location("utils", utils_path)
                mod_utils = importlib.util.module_from_spec(spec_utils)
                spec_utils.loader.exec_module(mod_utils)
            
            # 导入 models
            models_path = os.path.join(project_root, 'models.py')
            if os.path.exists(models_path):
                spec_models = importlib.util.spec_from_file_location("models", models_path)
                mod_models = importlib.util.module_from_spec(spec_models)
                # 需要先导入 modules
                modules_path = os.path.join(project_root, 'modules.py')
                if os.path.exists(modules_path):
                    spec_modules = importlib.util.spec_from_file_location("modules", modules_path)
                    mod_modules = importlib.util.module_from_spec(spec_modules)
                    spec_modules.loader.exec_module(mod_modules)
                    sys.modules['modules'] = mod_modules
                spec_models.loader.exec_module(mod_models)
            
            # --- 加载模型和数据 ---
            # 构建 args
            import argparse
            args = argparse.Namespace(**self.adapter.base_args)
            args.data_dir = os.path.join(project_root, 'data') + '/'
            args.no_cuda = False
            
            # 加载用户序列数据
            data_file = args.data_dir + args.data_name
            train_data, max_item_train, _ = mod_utils.get_user_seqs(data_file + "_train.txt")
            valid_data, max_item_val, _ = mod_utils.get_user_seqs(data_file + "_val.txt")
            test_data, max_item_test, _ = mod_utils.get_user_seqs(data_file + "_test.txt")
            max_item = max(max_item_train, max_item_val, max_item_test)
            args.item_size = max_item + 2
            
            # 构建评分矩阵
            train_matrix = mod_utils.generate_rating_matrix(train_data, args.item_size)
            valid_matrix = mod_utils.generate_rating_matrix(valid_data, args.item_size)
            test_matrix = mod_utils.generate_rating_matrix(test_data, args.item_size)
            
            # 加载模型
            import torch
            model = mod_models.SASRec(args=args)
            checkpoint_path = self._find_latest_checkpoint(project_root)
            if checkpoint_path:
                model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
                logger.info(f"Loaded model checkpoint: {checkpoint_path}")
            else:
                logger.warning("No checkpoint found, using current model state")
            
            cuda_condition = True and not args.no_cuda
            device = torch.device("cuda" if cuda_condition else "cpu")
            if cuda_condition:
                model.cuda()
            
            # --- Step 1: 惊喜评估 ---
            print(f"  🔍 Running surprise evaluation...")
            evaluator = mod_eval.SurpriseEvaluator(args, model)
            
            # 查找已有的惊喜子集
            surprise_subset_path = os.path.join(
                self.config.log_dir, 
                f"surprise_subset_{args.data_name}.json"
            )
            # 也查找 analysis_output 目录
            if not os.path.exists(surprise_subset_path):
                surprise_subset_path = os.path.join(
                    project_root, "analysis_output",
                    f"surprise_subset_{args.data_name}.json"
                )
            
            eval_report = evaluator.full_evaluation(
                train_data, valid_data, test_data,
                train_matrix, valid_matrix, test_matrix,
                surprise_subset_path=surprise_subset_path,
                item_text_map=self.item_text_map,
                num_train_subset=500,
            )
            
            # 保存评估报告
            report_path = os.path.join(
                self.config.log_dir, 
                f"surprise_eval_{args.data_name}_iter{self.current_iteration}.json"
            )
            evaluator.save_report(eval_report, report_path)
            
            # --- Step 2: 错误案例提取 ---
            print(f"  📝 Extracting wrong prediction cases...")
            extractor = mod_extract.ErrorCaseExtractor(args, model, self.item_text_map)
            
            # 从训练集提取错误案例 (用于过拟合检测)
            train_wrong = extractor.extract_wrong_cases(
                train_data, train_matrix, topk=20
            )
            
            # 从测试集提取错误案例
            test_wrong = extractor.extract_wrong_cases(
                test_data, test_matrix, topk=20
            )
            
            # 合并并选取 500 个
            all_wrong = train_wrong + test_wrong
            wrong_text_cases = extractor.convert_to_text(all_wrong, num_samples=500)
            
            # 保存错误案例
            cases_path = os.path.join(
                self.config.log_dir,
                f"wrong_cases_{args.data_name}_iter{self.current_iteration}.json"
            )
            extractor.save_text_cases(wrong_text_cases, cases_path)
            
            # --- Step 3: LLM 分析错误案例 ---
            print(f"  🧠 LLM analyzing wrong cases (with source code context)...")
            # 获取模型源码摘要供案例分析使用
            source_summary = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
            )
            case_analysis = self.case_analyzer.analyze_wrong_cases(
                text_cases=wrong_text_cases,
                model_config=self.adapter.base_args,
                overall_metrics=metrics,
                surprise_metrics=eval_report.get("surprise_subset"),
                diagnosis=eval_report.get("diagnosis"),
                max_cases=30,
                source_code_summary=source_summary,
            )
            
            # 保存案例分析
            if case_analysis:
                analysis_path = os.path.join(
                    self.config.log_dir,
                    f"case_analysis_{args.data_name}_iter{self.current_iteration}.json"
                )
                self.case_analyzer.save_analysis_report(case_analysis, analysis_path)
            
            # --- Step 4: LLM 惊喜优化分析 ---
            print(f"  🎯 LLM analyzing surprise optimization (with source code context)...")
            surprise_analysis = self.case_analyzer.analyze_surprise_optimization(
                overall_metrics=metrics,
                surprise_metrics=eval_report.get("surprise_subset"),
                model_config=self.adapter.base_args,
                diagnosis=eval_report.get("diagnosis"),
                source_code_summary=source_summary,
            )
            
            # --- Step 5: 合并报告 ---
            combined_report = self.case_analyzer.generate_combined_report(
                case_analysis or {},
                surprise_analysis or {},
                eval_report,
            )
            
            combined_path = os.path.join(
                self.config.log_dir,
                f"combined_analysis_{args.data_name}_iter{self.current_iteration}.json"
            )
            self.case_analyzer.save_analysis_report(combined_report, combined_path)
            
            # 清理 GPU
            model.cpu()
            torch.cuda.empty_cache()
            
            return {
                "evaluation_report": eval_report,
                "evaluation_report_summary": {
                    "test_full": eval_report.get("test_full"),
                    "train_subset": eval_report.get("train_subset"),
                    "surprise_subset": eval_report.get("surprise_subset"),
                    "diagnosis": eval_report.get("diagnosis"),
                },
                "wrong_text_cases": wrong_text_cases,
                "case_analysis": case_analysis,
                "case_analysis_summary": case_analysis.get("summary", "") if case_analysis and case_analysis.get("parse_success") else "",
                "surprise_analysis": surprise_analysis,
                "combined_report": combined_report,
            }
            
        except Exception as e:
            logger.error(f"Surprise analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _phase_verify_analysis(self, surprise_and_analysis: dict, metrics: dict) -> Optional[dict]:
        """
        假设验证阶段 — 验证 LLM 分析结论是否有数据支撑
        
        工作流程 (新版 Agent 流程):
        1. 从 LLM 案例分析结果中提取可验证的假设 (不限制验证类型)
        2. 为每个假设: LLM生成验证方案 → 发现数据 → 写代码 → 执行 → LLM解读结果
        3. 生成验证报告 (CONFIRMED / REFUTED / UNVERIFIABLE)
        4. 将验证结果反馈给下游流程
        
        与旧版区别:
        - 旧版: 固定6种验证方法 + 硬编码阈值
        - 新版: LLM 自主生成验证方案 + 写代码执行 + 解读结果
        
        Args:
            surprise_and_analysis: _phase_surprise_analysis 的返回结果
            metrics: 当前评估指标
            
        Returns:
            验证报告 dict, or None if 验证无法执行
        """
        case_analysis = surprise_and_analysis.get("case_analysis")
        wrong_text_cases = surprise_and_analysis.get("wrong_text_cases")
        
        if not case_analysis or not case_analysis.get("parse_success"):
            logger.info("Skipping hypothesis verification: no valid case analysis available")
            return None
        
        print(f"  🔬 [Phase 2.6] Verifying LLM analysis hypotheses (Agent autonomous mode)...")
        
        try:
            # --- Step 1: 从 LLM 分析中提取可验证的假设 ---
            hypotheses = self.hypothesis_verifier.extract_hypotheses(case_analysis)
            if not hypotheses:
                logger.info("No verifiable hypotheses extracted from LLM analysis")
                return None
            
            print(f"  📋 Extracted {len(hypotheses)} hypotheses to verify")
            for h in hypotheses:
                thought = h.get('verification_thought', h.get('verification_method', '?'))
                print(f"    {h.get('id', '?')}: {h.get('claim', '')[:60]}... "
                      f"[thought={thought[:40]}]")
            
            # --- Step 2: 准备验证所需的数据 ---
            # 物品热度分布 (从训练数据中计算)
            item_popularity = {}
            eval_report = surprise_and_analysis.get("evaluation_report", {})
            
            # 尝试从已有数据中获取热度信息
            # 如果 surprise_and_analysis 中有 train_data, 直接计算
            # 否则从日志中加载
            popularity_path = os.path.join(
                self.config.log_dir,
                f"item_popularity_{self.adapter.data_name}.json"
            )
            if os.path.exists(popularity_path):
                with open(popularity_path, 'r', encoding='utf-8') as f:
                    item_popularity = json.load(f)
                logger.info(f"Loaded item popularity from {popularity_path}: {len(item_popularity)} items")
            
            overall_metrics = metrics
            surprise_metrics = eval_report.get("surprise_subset")
            
            # --- Step 3: 运行验证 ---
            verified_hypotheses = self.hypothesis_verifier.verify_hypotheses(
                hypotheses=hypotheses,
                wrong_text_cases=wrong_text_cases or [],
                all_wrong_cases=None,  # 原始格式案例暂不可用
                model_config=self.adapter.base_args,
                item_popularity=item_popularity,
                overall_metrics=overall_metrics,
                surprise_metrics=surprise_metrics,
                iteration_number=self.current_iteration,  # 主流程迭代轮次，用于数据隔离
            )
            
            # --- Step 4: 生成验证报告 ---
            verification_report = self.hypothesis_verifier.generate_verification_report(
                verified_hypotheses
            )
            
            # --- Step 5: 保存验证报告 ---
            verification_path = os.path.join(
                self.config.log_dir,
                f"verification_{self.adapter.data_name}_iter{self.current_iteration}.json"
            )
            self.hypothesis_verifier.save_verification_report(
                verification_report, verification_path
            )
            
            return verification_report
            
        except Exception as e:
            logger.error(f"Hypothesis verification failed: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _find_latest_checkpoint(self, project_root: str) -> Optional[str]:
        """
        查找最新的模型 checkpoint
        
        搜索策略:
        1. project_root/output/  (训练脚本默认的输出目录)
        2. project_root/         (也可能在根目录)
        3. project_root/Recmodel/output/  (如果 project_root 不是 Recmodel 本身)
        4. config.output_dir 相对路径
        
        checkpoint 文件名格式: {backbone}-{data_name}-{ckp}{timestamp}.pt
        """
        # 可能的搜索目录
        search_dirs = [
            os.path.join(project_root, self.config.output_dir),
            project_root,
            os.path.join(project_root, "output"),
            os.path.join(project_root, "output/"),
        ]
        # 如果 project_root 不是 Recmodel, 也搜索 Recmodel 子目录
        if not project_root.endswith("Recmodel"):
            search_dirs.extend([
                os.path.join(project_root, "Recmodel", self.config.output_dir),
                os.path.join(project_root, "Recmodel", "output"),
                os.path.join(project_root, "Recmodel", "output/"),
            ])
        
        # 查找 .pt 文件
        pt_files = []
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for f in os.listdir(search_dir):
                if f.endswith('.pt'):
                    fpath = os.path.join(search_dir, f)
                    pt_files.append((fpath, os.path.getmtime(fpath)))
        
        if not pt_files:
            logger.warning(f"No checkpoint (.pt) found in: {search_dirs}")
            return None
        
        # 按修改时间排序，选最新的
        pt_files.sort(key=lambda x: x[1], reverse=True)
        latest = pt_files[0][0]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest

    def _phase_analyze_and_propose(self, metrics: dict) -> Optional[dict]:
        """
        LLM 分析 + 提案阶段 (参数 + 结构修改)
        
        向 LLM 提供:
        - 项目上下文 (包含可修改的源码结构)
        - 模型源码 (让 LLM 看到当前代码)
        - 当前指标 + 历史日志
        - 惊喜评估结果 + 错误案例分析
        
        LLM 输出包含两类改进:
        - param_changes: 超参数修改
        - structural_changes: 模型结构修改 (代码级别的改动)
        
        Returns:
            Dict: {
                "param_changes": {...},
                "structural_changes": [...],
                "explanation": "...",
            }
            或 None (LLM 调用失败)
        """
        # 构建项目上下文 (包含结构修改信息)
        project_ctx = self.adapter.build_llm_context()

        # ── 构建模型源码上下文 (使用 IterativeMemory 的智能截断!) ──
        # 不再使用旧的截断方式，而是让 IterativeMemory 优先展示修改区域
        source_code_ctx = self.adapter.build_source_code_context(
            include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
            max_total_chars=15000,
            iterative_memory=self.iter_memory,  # 传入 IterativeMemory → 智能截断!
        )

        # 构建历史摘要（先取再限长，避免历史日志无限膨胀）
        journal_summary = self.journal.summarize(n=8)
        if self.config.llm_enable_semantic_compression:
            journal_summary = self.context_compressor.compress_text(
                journal_summary,
                chunk_chars=self.config.llm_compression_chunk_chars,
                target_chars=self.config.llm_compression_target_chars,
                section_name="journal_summary",
                profile="journal",
            )
        journal_summary = self._clip_text_by_chars(journal_summary, max_chars=4500)

        # ── 构建结构修改历史 (使用 IterativeMemory 的完整因果链!) ──
        # 不再是旧的简短 60 字描述，而是完整的修改→效果→是否回滚 的因果链
        structural_history = self.iter_memory.build_history_context_for_llm(
            current_iteration=self.current_iteration,
            current_metrics=metrics,
        )
        if self.config.llm_enable_semantic_compression:
            structural_history = self.context_compressor.compress_text(
                structural_history,
                chunk_chars=self.config.llm_compression_chunk_chars,
                target_chars=max(4200, self.config.llm_compression_target_chars),
                section_name="structural_history",
                profile="history",
            )
        structural_history = self._clip_text_by_chars(structural_history, max_chars=6000)

        # ── 构建回滚黑名单 (如果有的话，这是让 LLM 不再踩坑的关键!) ──
        rollback_warning = self.iter_memory.build_rollback_aware_context()

        # 构建惊喜评估信息 (如果有)
        surprise_info = ""
        if self._last_surprise_report:
            surprise_info = f"""
## 惊喜评估结果
```json
{json.dumps(self._last_surprise_report.get("evaluation_report_summary", {}), indent=2, ensure_ascii=False)}
```

## 诊断信息
```json
{json.dumps(self._last_surprise_report.get("evaluation_report_summary", {}).get("diagnosis", {}), indent=2, ensure_ascii=False)}
```
"""

        # 构建案例分析信息 (如果有) — 包含验证状态标注
        case_info = ""
        if self._last_case_analysis:
            ca = self._last_case_analysis
            if ca.get("parse_success"):
                case_info = f"""
## LLM 对错误案例的分析结论
- **错误模式**: {json.dumps(ca.get("error_patterns", {}), ensure_ascii=False)}
- **模型瓶颈**: {json.dumps(ca.get("model_bottleneck", {}), ensure_ascii=False)}
- **惊喜失败原因**: {json.dumps(ca.get("surprise_failure_reasons", {}), ensure_ascii=False)}
- **改进建议**: {json.dumps(ca.get("improvement_suggestions", []), ensure_ascii=False)[:800]}
- **总结**: {ca.get("summary", "")}
"""
                # ── 添加验证状态 (如果有) ──
                if self._last_verification_report:
                    vm = ca.get("verification_meta", {})
                    if vm:
                        refuted = vm.get("refuted_claims", [])
                        credibility = vm.get("overall_credibility", "LOW")
                        field_verification = vm.get("field_verification", {})
                        
                        verification_info = f"""
## ⚠ 假设验证结果 (数据对 LLM 结论的验证)
**综合可信度**: {credibility} (已确认{vm.get('confirmed_pct', 0):.0f}%, 被反驳{vm.get('refuted_pct', 0):.0f}%)

"""
                        if refuted:
                            verification_info += "**❌ 以下结论被数据反驳, 不要基于这些结论提出改进**:\n"
                            for rc in refuted:
                                verification_info += f"- ❌ {rc[:100]}\n"
                            verification_info += "\n"
                        
                        # 逐字段标注验证状态
                        if field_verification:
                            verification_info += "**逐字段验证状态**:\n"
                            for field, verifications in field_verification.items():
                                statuses = [v.get("status", "?") for v in verifications]
                                confirmed_count = sum(1 for s in statuses if s == "CONFIRMED")
                                refuted_count = sum(1 for s in statuses if s == "REFUTED")
                                verification_info += f"- {field}: {confirmed_count}确认 / {refuted_count}反驳\n"
                        
                        case_info += verification_info

        # 构建 Prompt
        prompt = MLE_ANALYSIS_PROMPT.format(
            training_log=project_ctx,
            current_metrics=json.dumps(metrics, indent=2, ensure_ascii=False),
            experiment_journal=journal_summary,
            source_code_context=source_code_ctx,
            current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
        )

        # 添加源码上下文 (如果没有在 MLE_ANALYSIS_PROMPT 中包含)
        if "source_code_context" not in prompt and source_code_ctx:
            prompt += f"\n\n## 当前模型源码 (你可以直接修改)\n{source_code_ctx}"

        # 添加惊喜评估信息
        if surprise_info:
            prompt += surprise_info

        # 添加案例分析信息
        if case_info:
            prompt += case_info

        # 添加结构修改历史 (由 IterativeMemory 生成 — 完整因果链!)
        if structural_history:
            prompt += structural_history

        # 添加回滚黑名单警告 (让 LLM 不再踩坑!)
        if rollback_warning:
            prompt += rollback_warning

        strategy_instruction = self._get_strategy_instruction()
        if strategy_instruction:
            prompt += f"\n\n## 当前探索策略\n{strategy_instruction}"

        # 调用 LLM
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位推荐系统算法研究员，擅长从实验现象中观察、推理并提出改进方案。"
                    "你可以自由选择改进方式 — 调参、改代码、或两者结合 — 只要有充分的实验依据。"
                    "如果你提出代码修改，必须是可执行的完整 Python 代码，不能有省略号或占位符。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=self._get_temperature(),
            max_tokens=self.config.llm_max_tokens,
        )

        if response is None:
            logger.error("LLM analysis failed - no response")
            return None

        # 解析 LLM 的回复为结构化数据
        return self._parse_proposal_response(response)

    # ════════════════════════════════════════
    # 代码查询模式: LLM 按需获取代码, 而不是塞全部源码
    # ════════════════════════════════════════

    def _phase_query_based_analyze_and_propose(self, metrics: dict) -> Optional[dict]:
        """
        查询模式分析 — LLM 按需获取代码, 替代塞全部源码
        
        工作流程:
        1. 构建轻量索引 (文件名 + 类/函数签名, 通常 2000-4000 字符)
        2. 第一轮 LLM: 收到索引 + 指标 → 输出查询请求或直接提案
        3. 执行查询 → 返回精确代码片段
        4. 第二轮 LLM: 收到查询结果 → 继续查询或输出提案
        5. 循环 2-4, 最多 max_query_rounds 轮
        6. 最后强制输出提案
        
        核心优势:
        - 无截断! LLM 看到的是**精确**的代码, 不是截断后的残缺代码
        - SEARCH/REPLACE 的 search 文本一定能匹配源码
        - 索引只占 2000-4000 字符, 留出更多空间给指标和历史
        
        Returns:
            Dict: 同 _phase_analyze_and_propose 的格式
        """
        print(f"  🔍 [Query Mode] LLM 将按需查询代码...")
        
        # ── 构建轻量索引 ──
        code_index = self._code_query_tool.build_code_index()
        print(f"  📇 Code index built: {len(code_index)} chars")
        
        # ── 构建项目上下文 (不含全部源码!) ──
        project_ctx = self.adapter.build_llm_context()
        
        # ── 构建历史摘要 ──
        journal_summary = self.journal.summarize(n=8)
        if self.config.llm_enable_semantic_compression:
            journal_summary = self.context_compressor.compress_text(
                journal_summary,
                chunk_chars=self.config.llm_compression_chunk_chars,
                target_chars=self.config.llm_compression_target_chars,
                section_name="journal_summary",
                profile="journal",
            )
        journal_summary = self._clip_text_by_chars(journal_summary, max_chars=4500)
        
        # ── 构建结构修改历史 ──
        structural_history = self.iter_memory.build_history_context_for_llm(
            current_iteration=self.current_iteration,
            current_metrics=metrics,
        )
        if self.config.llm_enable_semantic_compression:
            structural_history = self.context_compressor.compress_text(
                structural_history,
                chunk_chars=self.config.llm_compression_chunk_chars,
                target_chars=max(4200, self.config.llm_compression_target_chars),
                section_name="structural_history",
                profile="history",
            )
        structural_history = self._clip_text_by_chars(structural_history, max_chars=6000)
        
        # ── 构建回滚黑名单 ──
        rollback_warning = self.iter_memory.build_rollback_aware_context()
        
        # ── 构建惊喜评估和案例分析信息 ──
        surprise_info = ""
        if self._last_surprise_report:
            surprise_info = f"""
## 惊喜评估结果
```json
{json.dumps(self._last_surprise_report.get("evaluation_report_summary", {}), indent=2, ensure_ascii=False)}
```
"""
        
        case_info = ""
        if self._last_case_analysis:
            ca = self._last_case_analysis
            if ca.get("parse_success"):
                case_info = f"""
## LLM 对错误案例的分析结论
- **错误模式**: {json.dumps(ca.get("error_patterns", {}), ensure_ascii=False)}
- **模型瓶颈**: {json.dumps(ca.get("model_bottleneck", {}), ensure_ascii=False)}
- **改进建议**: {json.dumps(ca.get("improvement_suggestions", []), ensure_ascii=False)[:800]}
"""
        
        # ── 累积查询结果 ──
        all_queried_code = ""
        observation_summary = ""
        reasoning_summary = ""
        
        # ── 多轮查询循环 ──
        max_rounds = self.config.max_query_rounds
        
        for round_idx in range(max_rounds):
            print(f"  🔍 [Query Round {round_idx + 1}/{max_rounds}]")
            
            # 构建当前轮的 prompt
            if round_idx == 0:
                # 第一轮: 使用 Phase1 prompt
                prompt = QUERY_BASED_PHASE1_PROMPT.format(
                    project_context=project_ctx,
                    current_metrics=json.dumps(metrics, indent=2, ensure_ascii=False),
                    experiment_journal=journal_summary,
                    code_index=code_index,
                    queried_code="(暂无 — 这是第一轮, 请提出你想查询的代码)",
                    structural_history=structural_history,
                    rollback_warning=rollback_warning,
                    surprise_info=surprise_info + case_info,
                    current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
                )
            else:
                # 后续轮: 使用 Phase2 prompt
                prompt = QUERY_BASED_PHASE2_PROMPT.format(
                    query_results=query_results_this_round,
                    previous_observation=observation_summary,
                    previous_analysis_direction=analysis_direction or "待确认",
                    current_metrics=json.dumps(metrics, indent=2, ensure_ascii=False),
                )
            
            # 添加策略指令
            strategy_instruction = self._get_strategy_instruction()
            if strategy_instruction:
                prompt += f"\n\n## 当前探索策略\n{strategy_instruction}"
            
            # 调用 LLM
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位推荐系统算法研究员。"
                        "你可以通过代码查询工具按需获取源码详情, 然后基于精确的代码提出改进方案。"
                        "⚠ 重要: 如果你想修改代码, 请先查询要修改的代码, 确保 SEARCH/REPLACE 的 search 文本与源码完全匹配!"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=self._get_temperature(),
                max_tokens=self.config.llm_max_tokens,
            )
            
            if response is None:
                logger.error(f"Query round {round_idx + 1} failed - no LLM response")
                if round_idx == 0:
                    # 第一轮就失败 → 降级回传统流程
                    print(f"  ⚠ Query mode failed, falling back to traditional mode")
                    return self._phase_analyze_and_propose(metrics)
                break
            
            # ── 解析 LLM 回复 ──
            parsed = self._parse_query_response(response)
            
            if parsed is None:
                # 解析失败 → 降级
                print(f"  ⚠ Query response parse failed at round {round_idx + 1}")
                # 尝试直接作为提案解析
                proposal = self._parse_proposal_response(response)
                if proposal:
                    print(f"  ✓ Parsed as proposal directly")
                    return proposal
                if round_idx == 0:
                    return self._phase_analyze_and_propose(metrics)
                break
            
            phase = parsed.get("phase", "")
            
            # ── 记录观察和分析 ──
            observation_summary = parsed.get("observation", observation_summary)
            reasoning_summary = parsed.get("reasoning", reasoning_summary)
            analysis_direction = parsed.get("analysis_direction", "")
            
            if phase == "proposal":
                # ── LLM 已经做出提案! ──
                print(f"  ✓ [Query Mode] LLM produced proposal after {round_idx + 1} round(s)")
                print(f"    Observation: {observation_summary[:120]}")
                
                # 将查询模式的回复转换为标准提案格式
                return {
                    "param_changes": parsed.get("param_changes", {}),
                    "structural_changes": parsed.get("structural_changes", []),
                    "explanation": parsed.get("rationale", "") or reasoning_summary,
                    "analysis": observation_summary + "\n" + reasoning_summary,
                    "_query_mode_used": True,
                    "_query_rounds": round_idx + 1,
                    "_queried_code_chars": len(all_queried_code),
                }
            
            elif phase == "query":
                # ── LLM 提出查询请求 ──
                queries = parsed.get("queries", [])
                if not queries:
                    print(f"  ⚠ No queries in response, forcing proposal")
                    continue  # 下一轮强制
                
                print(f"  🔎 LLM queries: {len(queries)} requests")
                # ── 兼容 LLM 输出的简化格式: strings → dicts ──
                # LLM 有时会输出 queries 为纯字符串列表 (如 ["Encoder.forward"]),
                # 而不是结构化的 [{"action": "search_function", "args": {"name": "Encoder.forward"}}]
                normalized_queries = []
                import re as regex_module
                for q in queries:
                    if isinstance(q, str):
                        # ── 清洗 LLM 格式污染 ──
                        cleaned_q = q.strip()
                        # 移除 "SEARCH:" / "REPLACE:" / "class " / "def " 等前缀
                        cleaned_q = regex_module.sub(
                            r'^\s*(SEARCH|REPLACE)\s*:\s*(class\s+|def\s+)?', '',
                            cleaned_q, flags=regex_module.IGNORECASE
                        ).strip()
                        # 移除包裹的引号
                        cleaned_q = cleaned_q.strip('"\'`<>')
                        # 自动将字符串转为 search_function 查询
                        normalized_queries.append({"action": "search_function", "args": {"name": cleaned_q}})
                        print(f"    → search_function(name={cleaned_q})  [auto-converted from string{' (cleaned from: '+q+')' if cleaned_q != q else ''}]")
                    elif isinstance(q, dict):
                        action = q.get("action", "?")
                        args = q.get("args", {})
                        print(f"    → {action}({args})")
                        normalized_queries.append(q)
                    else:
                        print(f"    ⚠ Skipping invalid query item: {type(q)} = {q}")
                queries = normalized_queries
                if not queries:
                    print(f"  ⚠ No valid queries after normalization, forcing next round")
                    continue
                
                # ── 执行查询 ──
                query_results_this_round = self._code_query_tool.execute_queries(queries)
                
                # 限制单次查询结果大小
                max_chars = self.config.code_query_max_chars_per_result
                if len(query_results_this_round) > max_chars:
                    query_results_this_round = query_results_this_round[:max_chars] + \
                        f"\n\n⚠ 查询结果超过 {max_chars} 字符, 已截断。如需更多细节, 请使用 get_region 获取具体行范围。"
                
                all_queried_code += "\n\n" + query_results_this_round
                
                # 刷新缓存 (源码可能已被修改)
                self._code_query_tool.refresh_cache()
                
                print(f"    Results: {len(query_results_this_round)} chars returned")
            
            else:
                # 未知 phase → 降级
                print(f"  ⚠ Unknown phase '{phase}' in query response")
                proposal = self._parse_proposal_response(response)
                if proposal:
                    return proposal
                continue
        
        # ── 达到最大查询轮数 → 强制输出最终提案 ──
        print(f"  📝 [Query Mode] Max rounds reached, forcing final proposal")
        
        final_prompt = QUERY_BASED_FINAL_PROMPT.format(
            all_queried_code=all_queried_code if all_queried_code else "(未查询任何代码)",
            current_metrics=json.dumps(metrics, indent=2, ensure_ascii=False),
            observation_summary=observation_summary,
            reasoning_summary=reasoning_summary,
            current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
        )
        
        # 添加策略指令
        strategy_instruction = self._get_strategy_instruction()
        if strategy_instruction:
            final_prompt += f"\n\n## 当前探索策略\n{strategy_instruction}"
        
        final_response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位推荐系统算法研究员。"
                    "你已通过多轮查询获取了精确的源码, 现在必须输出最终的改进方案。"
                    "⚠ SEARCH/REPLACE 的 search 文本必须与你查询到的代码完全匹配!"
                )},
                {"role": "user", "content": final_prompt},
            ],
            temperature=self._get_temperature(),
            max_tokens=self.config.llm_max_tokens,
        )
        
        if final_response is None:
            logger.error("Final query-based proposal failed")
            return self._phase_analyze_and_propose(metrics)  # 降级
        
        return self._parse_proposal_response(final_response)

    def _parse_query_response(self, response: str) -> Optional[dict]:
        """解析 LLM 在查询模式中的回复 (可能是查询请求或提案)
        
        兼容 LLM 输出的多种 proposal 格式:
        1. 标准格式: {"phase": "proposal", "structural_changes": [...], "param_changes": {...}}
        2. 扁平格式: {"phase": "proposal", "edits": [{"filename": "...", "action": "modify", "search": "...", "replace": "..."}]}
           (LLM 直接把 edits 放在顶层, 而不是嵌套在 structural_changes 里)
        3. 混合格式: {"phase": "proposal", "edits": [...], "structural_changes": [...]}
        
        统一转换为标准格式 (structural_changes + param_changes)
        """
        # 提取 JSON
        import re as regex_module
        json_match = regex_module.search(
            r'```(?:json)?\s*\n?(.*?)\n?```', response, regex_module.DOTALL
        )
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end + 1]
            else:
                return None
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_str)
                if not isinstance(data, dict):
                    return None
            except Exception:
                return None
        
        # ── 格式转换: 顶层 "edits" → "structural_changes" ──
        # LLM 有时会输出 {"phase": "proposal", "edits": [...]},
        # 但下游代码期望 {"structural_changes": [{"edits": [...]}]} 格式
        if "edits" in data and "structural_changes" not in data:
            top_level_edits = data.pop("edits")
            if isinstance(top_level_edits, list) and top_level_edits:
                # 将每个 edit 转换为 structural_changes 条目
                # LLM 扁平格式的 edit 通常包含 "file"/"instruction" + "search"/"replace"
                structural_entries = []
                for edit in top_level_edits:
                    if isinstance(edit, dict):
                        # 提取 target_file (可能是 "file" 或 "filename" 或 "target_file")
                        target_file = edit.get("file", edit.get("filename", edit.get("target_file", "")))
                        instruction = edit.get("instruction", edit.get("description", ""))
                        search_text = edit.get("search", "")
                        replace_text = edit.get("replace", "")
                        action = edit.get("action", edit.get("action_type", "modify"))
                        
                        if search_text or replace_text:
                            entry = {
                                "target_file": target_file,
                                "description": instruction,
                                "edits": [{"search": search_text, "replace": replace_text}],
                            }
                            # 保留其他有用字段
                            for key in ["expected_effect", "confidence"]:
                                if key in edit:
                                    entry[key] = edit[key]
                            # action / action_type → action_type
                            entry["action_type"] = action
                            structural_entries.append(entry)
                
                if structural_entries:
                    data["structural_changes"] = structural_entries
                    logger.info(f"Normalized top-level 'edits' ({len(top_level_edits)} items) "
                                f"→ 'structural_changes' ({len(structural_entries)} entries)")
        
        # 检查是否包含 "phase" 字段
        if "phase" not in data:
            # 可能是直接提案格式 (没有 phase 字段)
            # 如果有 structural_changes 或 param_changes → 视为提案
            if "structural_changes" in data or "param_changes" in data:
                data["phase"] = "proposal"
            elif "queries" in data:
                data["phase"] = "query"
            else:
                # 无法判断 → 假设是提案
                data["phase"] = "proposal"
        
        return data

    # ════════════════════════════════════════
    # 多角色工作流: Planner → Researcher → Coder → Debugger
    # ════════════════════════════════════════

    def _phase_multi_role_analyze_and_propose(self, metrics: dict) -> Optional[dict]:
        """
        多角色工作流的分析与提案阶段
        
        工作流:
        1. Planner: 规划研究方向 (使用 PLANNER_INSTRUCTIONS)
        2. Researcher: 深度研究提出方案 (使用 RESEARCHER_INSTRUCTIONS)
        3. Coder: 代码修改 (使用 CODER_INSTRUCTIONS)
        4. Debugger: 代码验证 (使用 DEBUGGER_INSTRUCTIONS) — 可选
        
        同时在每轮迭代后执行 Reflection (使用 REFLECTION_INSTRUCTIONS)
        
        Returns:
            与 _phase_analyze_and_propose 相同格式的 dict
        """
        import asyncio
        
        # ── 构建公共上下文 ──
        journal_summary = self.journal.summarize(n=8)
        if self.config.llm_enable_semantic_compression:
            journal_summary = self.context_compressor.compress_text(
                journal_summary,
                chunk_chars=self.config.llm_compression_chunk_chars,
                target_chars=self.config.llm_compression_target_chars,
                section_name="journal_summary",
                profile="journal",
            )
        journal_summary = self._clip_text_by_chars(journal_summary, max_chars=4500)
        # 使用与 _phase_analyze_and_propose 一致的源码上下文构建方式
        source_code_ctx = self.adapter.build_source_code_context(
            include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
            max_total_chars=15000,
            iterative_memory=self.iter_memory,
        )
        project_ctx = self.adapter.format_metrics_for_llm(metrics)
        
        metrics_str = json.dumps(metrics, indent=2, ensure_ascii=False)
        hidden_size = self.adapter.base_args.get("hidden_size", 64)
        research_topic = f"{self.adapter.backbone} on {self.adapter.data_name}"
        
        # ── Step 1: Planner 规划研究方向 ──
        print(f"  🎯 [Planner] 规划研究方向...")
        planner_prompt = PLANNER_INSTRUCTIONS.format(
            research_topic=research_topic,
            current_metrics=metrics_str,
            experiment_journal=journal_summary,
        )
        
        # 添加额外上下文
        planner_prompt += f"\n\n## 当前模型源码\n{source_code_ctx}"
        
        # 添加惊喜评估信息
        if self._last_surprise_report:
            surprise_info = f"""
## 惊喜评估结果
```json
{json.dumps(self._last_surprise_report.get("evaluation_report_summary", {}), indent=2, ensure_ascii=False)}
```

## 诊断信息
```json
{json.dumps(self._last_surprise_report.get("evaluation_report_summary", {}).get("diagnosis", {}), indent=2, ensure_ascii=False)}
```
"""
            planner_prompt += surprise_info
        
        strategy_instruction = self._get_strategy_instruction()
        if strategy_instruction:
            planner_prompt += f"\n\n## 当前探索策略\n{strategy_instruction}"
        
        planner_response = self.llm.chat(
            messages=[
                {"role": "system", "content": "你是一位资深推荐系统研究规划师，擅长从实验数据中识别瓶颈并规划研究方向。"},
                {"role": "user", "content": planner_prompt},
            ],
            temperature=self.config.planner_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        
        if planner_response is None:
            logger.error("Planner step failed - no response")
            return self._phase_analyze_and_propose(metrics)  # 降级回传统流程
        
        # 解析 Planner 的规划结果
        planner_result = self._parse_json_response(planner_response)
        if planner_result is None:
            logger.warning("Planner output not valid JSON, falling back to traditional workflow")
            return self._phase_analyze_and_propose(metrics)
        
        research_plan = planner_result.get("research_plan", {})
        primary_direction = research_plan.get("primary_direction", "")
        hypothesis = research_plan.get("hypothesis", "")
        
        print(f"  📋 [Planner] 方向: {primary_direction[:80]}")
        print(f"  📋 [Planner] 假设: {hypothesis[:80]}")
        
        # ── Step 2: Researcher 深度研究 ──
        print(f"  🔍 [Researcher] 深度研究...")
        
        # 更新研究 Agent 的主题
        self._researcher_agent.update_topic(
            query=primary_direction,
            problem_name=research_topic,
            problem_description=hypothesis,
        )
        
        researcher_prompt = RESEARCHER_INSTRUCTIONS.format(
            research_direction=primary_direction,
            current_metrics=metrics_str,
            experiment_journal=journal_summary,
            source_code_context=source_code_ctx[:4000],  # 限制长度
        )
        
        researcher_response = self.llm.chat(
            messages=[
                {"role": "system", "content": "你是一位专业的推荐系统研究员，擅长分析问题并提出创新性改进方案。"},
                {"role": "user", "content": researcher_prompt},
            ],
            temperature=self.config.researcher_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        
        if researcher_response is None:
            logger.error("Researcher step failed - no response")
            return self._phase_analyze_and_propose(metrics)
        
        researcher_result = self._parse_json_response(researcher_response)
        if researcher_result is None:
            logger.warning("Researcher output not valid JSON, falling back to traditional workflow")
            return self._phase_analyze_and_propose(metrics)
        
        # 提取推荐方案
        recommended = researcher_result.get("recommended_solution", {})
        chosen_solution = recommended.get("choice", primary_direction)
        proposed_solutions = researcher_result.get("proposed_solutions", [])
        
        # 选择推荐方案或第一个方案
        best_solution = None
        for sol in proposed_solutions:
            if sol.get("solution_name") == chosen_solution:
                best_solution = sol
                break
        if best_solution is None and proposed_solutions:
            best_solution = proposed_solutions[0]
        
        if best_solution is None:
            logger.warning("No valid solution from Researcher, falling back")
            return self._phase_analyze_and_propose(metrics)
        
        print(f"  📝 [Researcher] 方案: {best_solution.get('solution_name', '?')[:80]}")
        print(f"  📝 [Researcher] 理论: {best_solution.get('theoretical_basis', '?')[:80]}")
        
        # ── Step 2.5: Reflection (可选) ──
        if self.current_iteration > 0 and self._last_surprise_report:
            print(f"  🔄 [Reflection] 反思研究进展...")
            reflection_prompt = REFLECTION_INSTRUCTIONS.format(
                iteration_count=self.current_iteration,
                previous_direction=primary_direction,
                previous_results=metrics_str,
                current_metrics=metrics_str,
            )
            
            reflection_response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一位经验丰富的AI研究顾问，擅长反思和评估研究进展。"},
                    {"role": "user", "content": reflection_prompt},
                ],
                temperature=self.config.planner_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            
            if reflection_response:
                reflection_result = self._parse_json_response(reflection_response)
                if reflection_result:
                    recommendations = reflection_result.get("recommendations", {})
                    if not recommendations.get("continue_current_direction", True):
                        # 反思建议转向 — 重新规划
                        pivot = recommendations.get("suggested_pivot", "")
                        print(f"  🔀 [Reflection] 建议转向: {pivot[:80]}")
                        # 用转向方向覆盖当前方案的理论依据
                        if pivot:
                            best_solution["theoretical_basis"] = f"Reflection pivot: {pivot}"
        
        # ── Step 3: Coder 代码修改 ──
        print(f"  💻 [Coder] 代码修改...")
        research_idea = (
            f"{best_solution.get('solution_name', chosen_solution)}: "
            f"{best_solution.get('theoretical_basis', '')}"
        )
        
        coder_prompt = CODER_INSTRUCTIONS.format(
            research_idea=research_idea,
            target_metrics=metrics_str,
            source_code_context=source_code_ctx,
        )
        
        coder_response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位具有强大软件工程能力的研究者，擅长使用SEARCH/REPLACE格式精确修改代码。"
                    "所有修改必须使用 Self_EvolveRec-BLOCK-START/END 标记追踪。"
                )},
                {"role": "user", "content": coder_prompt},
            ],
            temperature=self.config.coder_temperature,
            max_tokens=self.config.llm_max_tokens,
        )
        
        if coder_response is None:
            logger.error("Coder step failed - no response")
            return self._phase_analyze_and_propose(metrics)
        
        # ── Step 3.5: 解析 Coder 输出 (支持 SEARCH/REPLACE 格式和 JSON 格式) ──
        coder_result = self._parse_multi_role_coder_output(coder_response, research_idea)
        
        if coder_result is None:
            logger.warning("Coder output could not be parsed, falling back to traditional workflow")
            return self._phase_analyze_and_propose(metrics)
        
        # ── Step 4: Debugger 代码验证 (可选) ──
        if coder_result.get("structural_changes"):
            print(f"  🔧 [Debugger] 验证代码修改...")
            
            # 构建当前修改后的代码用于验证
            current_code = source_code_ctx[:6000]  # 限制长度
            
            debugger_prompt = DEBUGGER_INSTRUCTIONS.format(
                current_code=current_code,
                error_info="请在提交前验证代码正确性",  # 预检模式
            )
            
            debugger_response = self._debugger_llm.chat(
                messages=[
                    {"role": "system", "content": "你是一位专家开发者，负责验证代码修改的正确性和一致性。"},
                    {"role": "user", "content": debugger_prompt},
                ],
                temperature=self.config.debugger_temperature,
                max_tokens=self.config.llm_max_tokens,
            )
            
            if debugger_response:
                debug_result = self._parse_json_response(debugger_response)
                if debug_result and debug_result.get("issues_found"):
                    # Debugger 发现问题 → 标记低信心
                    for sc in coder_result.get("structural_changes", []):
                        sc["confidence"] = "低"
                    print(f"  ⚠ [Debugger] 发现潜在问题: {debug_result.get('issues_found', [])}")
        
        # ── 合并结果为统一格式 ──
        result = {
            "param_changes": coder_result.get("param_changes", {}),
            "structural_changes": coder_result.get("structural_changes", []),
            "explanation": (
                f"[Multi-Role] Planner方向={primary_direction}, "
                f"Researcher方案={chosen_solution}, "
                f"Coder修改={len(coder_result.get('structural_changes', []))}处"
            ),
            "observation": planner_result.get("analysis", {}).get("current_bottlenecks", []),
            "reasoning": best_solution.get("theoretical_basis", ""),
            "rationale": planner_result.get("rationale", ""),
            "workflow_trace": {
                "planner": planner_result,
                "researcher": researcher_result,
                "coder": coder_result,
            },
        }
        
        self.journal.record({
            "iteration": self.current_iteration,
            "phase": "multi_role_analyze_and_propose",
            "status": "MULTI_ROLE_PROPOSAL",
            "planner_direction": primary_direction,
            "researcher_solution": chosen_solution,
            "num_structural_changes": len(result["structural_changes"]),
        })
        
        return result

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """
        尝试从 LLM 输出中解析 JSON
        
        使用 llm_utils 共享模块, 增加回退逻辑保留旧版兼容
        """
        result = parse_json_from_response(response)
        if result is not None:
            return result
        
        # 回退: 旧版 regex 匹配 (覆盖 llm_utils 不处理的格式)
        if not response:
            return None
        for pattern in [r'```json\s*\n(.*?)```', r'```(.*?)```']:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    continue
        
        logger.warning(f"Could not parse JSON from response (first 200 chars): {response[:200]}")
        return None

    def _parse_multi_role_coder_output(self, response: str, research_idea: str) -> Optional[dict]:
        """
        解析 Coder 的输出 — 支持 SEARCH/REPLACE 格式和传统 JSON 格式
        
        SEARCH/REPLACE 格式:
          <<<<<<< SEARCH
          # original code
          =======
          ### >>> Self_EvolveRec-BLOCK-START: <idea>
          # new code
          ### <<< Self_EvolveRec-BLOCK-END
          >>>>>>> REPLACE
        
        传统 JSON 格式:
          { "structural_changes": [...], "param_changes": {...} }
        """
        if not response:
            return None
        
        # ── 尝试解析 SEARCH/REPLACE 格式 ──
        sr_result = self._parse_search_replace_diff(response, research_idea)
        if sr_result:
            # SEARCH/REPLACE 格式也需要验证 (与 _parse_proposal_response 保持一致)
            sc = sr_result.get("structural_changes", [])
            if sc:
                validated = self._validate_structural_changes(sc)
                sr_result["structural_changes"] = validated
                if validated:
                    logger.info(f"[_parse_multi_role_coder_output] SEARCH/REPLACE: "
                                f"{len(sc)} changes → {len(validated)} validated")
                else:
                    logger.warning(f"[_parse_multi_role_coder_output] SEARCH/REPLACE: "
                                   f"{len(sc)} changes → 0 after validation")
            return sr_result
        
        # ── 降级: 尝试解析 JSON 格式 ──
        json_result = self._parse_json_response(response)
        if json_result:
            # JSON 格式可能直接包含 structural_changes 和 param_changes
            sc = json_result.get("structural_changes", [])
            if sc:
                validated = self._validate_structural_changes(sc)
                json_result["structural_changes"] = validated
                if not validated:
                    logger.warning(f"[_parse_multi_role_coder_output] JSON: "
                                   f"{len(sc)} structural_changes → 0 after validation")
            return {
                "structural_changes": json_result.get("structural_changes", []),
                "param_changes": json_result.get("param_changes", {}),
            }
        
        # ── 最后降级: 使用传统 _parse_proposal_response ──
        # (已经包含 _validate_structural_changes 调用)
        traditional_result = self._parse_proposal_response(response)
        if traditional_result:
            return traditional_result
        
        return None

    def _parse_search_replace_diff(self, response: str, research_idea: str) -> Optional[dict]:
        """
        从 LLM 输出中提取 SEARCH/REPLACE diff 块并转换为 structural_changes
        
        每个 SEARCH/REPLACE 块转换为新格式:
        {
            "target_file": "...",
            "description": "...",
            "edits": [{"search": "...", "replace": "..."}],
            "expected_effect": "...",
            "confidence": "中",
        }
        """
        # 提取所有 SEARCH/REPLACE 块
        diff_pattern = r'<<<<<<< SEARCH\s*\n(.*?)=======\s*\n(.*?)>>>>>>> REPLACE'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        if not matches:
            return None
        
        # 按目标文件分组 edits
        file_edits = {}  # target_file -> list of (search, replace, description)
        
        for search_code, replace_code in matches:
            search_code = search_code.strip()
            replace_code = replace_code.strip()
            
            # 提取 Self_EvolveRec 块描述 (如果有)
            block_idea = research_idea
            idea_match = re.search(
                r'Self_EvolveRec-BLOCK-START:\s*(.*?)(?:\n|$)',
                replace_code,
            )
            if idea_match:
                block_idea = idea_match.group(1).strip()
            
            # 尝试确定目标文件
            target_file, _ = self._identify_target_from_code(search_code)
            
            # 提取实际新代码 (去掉 Self_EvolveRec 标记行)
            clean_replace = self._strip_evolve_markers(replace_code)
            clean_search = self._strip_evolve_markers(search_code)
            
            if target_file not in file_edits:
                file_edits[target_file] = []
            file_edits[target_file].append({
                "search": clean_search,
                "replace": clean_replace,
                "description": f"[Self_EvolveRec] {block_idea}",
            })
        
        # 构建新格式的 structural_changes
        structural_changes = []
        for target_file, edits_list in file_edits.items():
            change = {
                "target_file": target_file,
                "description": edits_list[0]["description"],
                "edits": [{"search": e["search"], "replace": e["replace"]} for e in edits_list],
                "expected_effect": f"Implement: {research_idea}",
                "confidence": "中",
            }
            structural_changes.append(change)
        
        if not structural_changes:
            return None
        
        return {
            "structural_changes": structural_changes,
            "param_changes": {},  # SEARCH/REPLACE 格式不含参数变更
        }

    def _identify_target_from_code(self, code_snippet: str) -> tuple:
        """
        从代码片段中推断目标文件和函数/类名
        
        三级推断策略:
        1. 类定义匹配 — 在源码文件中查找唯一包含该类名的文件
        2. 函数定义匹配 — 在源码文件中查找包含该函数名的文件
           * 对于常见函数名 (__init__, forward, train 等)，逐文件验证
           * 先尝试精确匹配 search text 行, 再退到函数名匹配
        3. 内容逐行匹配 — 将 search text 的每一行在所有源码文件中搜索,
           选匹配行最多的文件 (解决 "betas=..." "elif..." 等无定义的片段)
        
        Returns:
            (target_file, target_class_or_function)
        """
        # ── Level 1: 类定义匹配 ──
        class_match = re.search(r'class\s+(\w+)', code_snippet)
        if class_match:
            class_name = class_match.group(1)
            target_file = self._find_file_containing_class(class_name)
            return (target_file, class_name)
        
        # ── Level 2: 函数定义匹配 (改进: 验证多文件冲突) ──
        func_match = re.search(r'def\s+(\w+)', code_snippet)
        if func_match:
            func_name = func_match.group(1)
            # 先尝试用 search text 的行内容精确定位文件
            content_match_file = self._find_best_file_by_content_match(code_snippet)
            if content_match_file:
                logger.info(f"_identify_target: content match → {content_match_file} "
                            f"(func_name={func_name} found in multiple files)")
                return (content_match_file, func_name)
            # 退到函数名查找
            target_file = self._find_file_containing_function(func_name)
            return (target_file, func_name)
        
        # ── Level 3: 内容逐行匹配 (无 class/def 时的关键策略) ──
        # 例如 "betas = ...", "elif self.args...", "self.criterion = ..."
        content_match_file = self._find_best_file_by_content_match(code_snippet)
        if content_match_file:
            logger.info(f"_identify_target: content match → {content_match_file} "
                        f"(no class/def in snippet)")
            return (content_match_file, "unknown")
        
        # ── 最终兜底 ──
        default_file = list(self.adapter.SOURCE_FILE_MAP.keys())[0] if self.adapter.SOURCE_FILE_MAP else "unknown"
        logger.warning(f"_identify_target: could not determine target file, "
                       f"falling back to default: {default_file}")
        return (default_file, "unknown")

    def _find_file_containing_class(self, class_name: str) -> str:
        """查找包含指定类名的源码文件"""
        for file_path in self.adapter.SOURCE_FILE_MAP.keys():
            full_path = os.path.join(self.config.project_root, file_path)
            if os.path.exists(full_path):
                try:
                    content = open(full_path).read()
                    if f"class {class_name}" in content:
                        return file_path
                except Exception:
                    continue
        return list(self.adapter.SOURCE_FILE_MAP.keys())[0] if self.adapter.SOURCE_FILE_MAP else "unknown"

    def _find_file_containing_function(self, func_name: str) -> str:
        """查找包含指定函数名的源码文件"""
        for file_path in self.adapter.SOURCE_FILE_MAP.keys():
            full_path = os.path.join(self.config.project_root, file_path)
            if os.path.exists(full_path):
                try:
                    content = open(full_path).read()
                    if f"def {func_name}" in content:
                        return file_path
                except Exception:
                    continue
        return list(self.adapter.SOURCE_FILE_MAP.keys())[0] if self.adapter.SOURCE_FILE_MAP else "unknown"

    def _find_best_file_by_content_match(self, code_snippet: str) -> Optional[str]:
        """
        用逐行内容匹配来确定 search text 最可能属于哪个文件
        
        算法:
        1. 取 search text 的所有非空行 (strip 后)
        2. 对每个源码文件, 计算有多少行的 stripped 版本出现在文件内容中
        3. 选匹配行数最多的文件 (至少 >= 1 行才算匹配)
        
        这解决了两个关键问题:
        - "betas = (self.args.adam_beta1, ...)" 这类无 def/class 的代码片段
        - "def __init__" 这类多个文件都有的函数名
        
        Returns:
            target_file (如 "trainers.py") 或 None (无任何匹配)
        """
        # 取 search text 的非空行 (stripped, 用于匹配)
        snippet_lines = [l.strip() for l in code_snippet.split('\n') if l.strip()]
        if not snippet_lines:
            return None
        
        best_file = None
        best_match_count = 0
        
        for file_path in self.adapter.SOURCE_FILE_MAP.keys():
            full_path = os.path.join(self.config.project_root, file_path)
            if not os.path.exists(full_path):
                continue
            try:
                content = open(full_path).read()
                # 统计该文件中有多少行匹配 search text 的行
                content_lines_stripped = [l.strip() for l in content.split('\n') if l.strip()]
                match_count = sum(1 for sl in snippet_lines if sl in content_lines_stripped)
                if match_count > best_match_count:
                    best_match_count = match_count
                    best_file = file_path
            except Exception:
                continue
        
        # 至少需要 1 行匹配才算有效 (避免完全无关的文件)
        if best_match_count >= 1 and best_file:
            logger.info(f"_find_best_file_by_content_match: {best_file} "
                        f"matched {best_match_count}/{len(snippet_lines)} lines")
            return best_file
        
        return None

    def _strip_evolve_markers(self, code: str) -> str:
        """从代码中移除 Self_EvolveRec-BLOCK-START/END 标记行"""
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('### >>> Self_EvolveRec-BLOCK-START:'):
                # 保留注释内容但去掉标记格式
                idea = stripped.replace('### >>> Self_EvolveRec-BLOCK-START:', '').strip()
                clean_lines.append(f"# [Self_EvolveRec] {idea}")
            elif stripped == '### <<< Self_EvolveRec-BLOCK-END':
                continue  # 移除结束标记
            else:
                clean_lines.append(line)
        return '\n'.join(clean_lines)

    def _parse_proposal_response(self, response: str) -> Optional[dict]:
        """
        解析 LLM 的提案回复
        
        支持多种格式:
        1. 完整 JSON (包含 param_changes + structural_changes)
        2. 旧格式 (只有 param_changes，没有 structural_changes)
        3. 自然语言 + JSON 混合
        
        Returns:
            Dict: {
                "param_changes": {...},
                "structural_changes": [...],
                "explanation": "...",
                "analysis": "...",
            }
        """
        import re as regex_module
        
        # 提取 JSON
        json_match = regex_module.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, regex_module.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end + 1]
            else:
                print(f"\n  ⚠ [PARSE DIAGNOSIS] Cannot extract JSON from LLM proposal response")
                print(f"     LLM response: {response}")
                logger.warning("Cannot extract JSON from LLM proposal response")
                # 尝试用旧格式解析
                parsed_old = self.parser.parse(response)
                if parsed_old["valid"]:
                    param_changes = self._parse_diff_to_params(parsed_old)
                    return {
                        "param_changes": param_changes or {},
                        "structural_changes": [],
                        "explanation": parsed_old.get("explanation", ""),
                        "analysis": "",
                    }
                return None

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"\n  ⚠ [PARSE DIAGNOSIS] JSON decode failed: {e}")
            print(f"     Extracted JSON string first 200 chars: {json_str[:200]}")
            logger.warning(f"JSON parse error: {e}")
            # 尝试宽松解析
            try:
                import ast
                data = ast.literal_eval(json_str)
                if not isinstance(data, dict):
                    return None
            except Exception:
                return None

        # ── 格式转换: 顶层 "edits" → "structural_changes" ──
        # LLM 有时会输出 {"phase": "proposal", "edits": [...]},
        # 但下游代码期望 {"structural_changes": [{"edits": [...]}]} 格式
        # 这与 _parse_query_response 中同一问题的修复一致 (Bug I)
        if "edits" in data and "structural_changes" not in data:
            top_level_edits = data.pop("edits")
            if isinstance(top_level_edits, list) and top_level_edits:
                structural_entries = []
                for edit in top_level_edits:
                    if isinstance(edit, dict):
                        target_file = edit.get("file", edit.get("filename", edit.get("target_file", "")))
                        instruction = edit.get("instruction", edit.get("description", ""))
                        search_text = edit.get("search", "")
                        replace_text = edit.get("replace", "")
                        action = edit.get("action", edit.get("action_type", "modify"))
                        
                        if search_text or replace_text:
                            entry = {
                                "target_file": target_file,
                                "description": instruction,
                                "edits": [{"search": search_text, "replace": replace_text}],
                            }
                            for key in ["expected_effect", "confidence"]:
                                if key in edit:
                                    entry[key] = edit[key]
                            entry["action_type"] = action
                            structural_entries.append(entry)
                
                if structural_entries:
                    data["structural_changes"] = structural_entries
                    logger.info(f"[_parse_proposal_response] Normalized top-level 'edits' "
                                f"({len(top_level_edits)} items) → 'structural_changes' ({len(structural_entries)} entries)")

        # 提取各部分 — 兼容新旧字段名
        # 新prompt使用: observation/reasoning/rationale, 旧prompt使用: analysis/explanation
        param_changes = data.get("param_changes", {})
        structural_changes = data.get("structural_changes", [])
        # 兼容: 新版 "rationale" 或旧版 "explanation"
        explanation = data.get("rationale", "") or data.get("explanation", "")
        # 兼容: 新版 "observation"+"reasoning" 或旧版 "analysis"
        analysis = data.get("observation", "") + "\n" + data.get("reasoning", "") if data.get("observation") else data.get("analysis", "")


        # 验证参数变更
        if param_changes:
            param_changes = self._validate_param_changes(param_changes)

        # 验证结构修改
        if structural_changes:
            structural_changes = self._validate_structural_changes(structural_changes)

        # 如果 LLM 没有提出结构修改但案例分析中有，从案例分析中提取
        if not structural_changes and self._last_case_analysis:
            structural_changes = self._extract_structural_from_case_analysis()

        return {
            "param_changes": param_changes or {},
            "structural_changes": structural_changes or [],
            "explanation": explanation,
            "analysis": analysis,
        }

    def _validate_param_changes(self, param_changes: dict) -> dict:
        """验证参数变更 — 完全开放模式: 默认 soft_limit=True，超出范围仅警告不拒绝"""
        valid_changes = {}
        for key, value in param_changes.items():
            if key in self.adapter.TUNABLE_PARAMS:
                param_info = self.adapter.TUNABLE_PARAMS[key]
                # 默认 soft_limit=True: 完全开放模式，不拒绝任何参数值
                soft_limit = param_info.get("soft_limit", True)
                try:
                    if param_info["type"] == "float":
                        value = float(value)
                    elif param_info["type"] == "int":
                        value = int(value)
                    if "range" in param_info and param_info["range"] is not None:
                        lo, hi = param_info["range"]
                        if lo <= value <= hi:
                            valid_changes[key] = value
                        elif soft_limit:
                            # soft_limit: 仍然接受，只给温和提示
                            logger.info(f"{key}={value} outside suggested range [{lo},{hi}], but accepted (soft limit)")
                            valid_changes[key] = value
                        else:
                            logger.warning(f"{key}={value} out of range [{lo},{hi}], rejected")
                    elif "choices" in param_info and param_info["choices"] is not None:
                        if value in param_info["choices"]:
                            valid_changes[key] = value
                        elif soft_limit:
                            # soft_limit: 接受任意值，LLM 可以自由命名
                            logger.info(f"{key}={value} not in suggested choices {param_info['choices']}, but accepted (soft limit)")
                            valid_changes[key] = value
                        else:
                            logger.warning(f"{key}={value} not in {param_info['choices']}, rejected")
                    else:
                        valid_changes[key] = value
                except (ValueError, TypeError):
                    logger.warning(f"Cannot convert {key}={value} to {param_info['type']}")
            else:
                # 不在可调参列表 — LLM 可能提出新参数，接受
                valid_changes[key] = value
        return valid_changes

    @staticmethod
    def _clean_markdown_wrapper(text: str) -> str:
        """
        只清理 markdown 代码块标记和多余空白, 不做"移除解释性前缀"处理
        
        使用 llm_utils 共享模块的 clean_markdown_wrapper
        """
        return llm_clean_markdown_wrapper(text)

    def _normalize_change_format(self, change: dict) -> Optional[dict]:
        """
        修复 Bug C: 格式兼容性 — 自动检测并转换 LLM 输出的多种格式
        
        支持的输入格式:
        1. 标准格式: {"target_file": "x.py", "edits": [{"search": "...", "replace": "..."}]}
        2. 扁平格式: {"filename": "x.py", "action": "modify", "search": "...", "replace": "..."}
           (缺少 edits 包裹, search/replace 在顶层)
        3. 混合格式: {"target_file": "x.py", "search": "...", "replace": "..."}
           (有 target_file 但缺少 edits 包裹)
        
        输出统一为标准格式.
        """
        # 格式1: 标准格式 (已有 target_file + edits) → 直接返回
        if change.get("target_file") and change.get("edits"):
            return change
        
        # 格式2: 扁平格式 (filename + action + search/replace)
        if not change.get("target_file") and change.get("filename"):
            change["target_file"] = change.pop("filename")
            # 扁平 search/replace → 转为 edits 数组
            if "search" in change or "replace" in change:
                search_text = change.pop("search", "")
                replace_text = change.pop("replace", "")
                action = change.pop("action", "modify")
                change["edits"] = [{"search": search_text, "replace": replace_text}]
                change["action_type"] = action
                # 删除不属于标准格式的多余字段
                for key in ["action"]:
                    change.pop(key, None)
            logger.info(f"Format normalized: filename→target_file, flat→edits for {change.get('target_file')}")
            return change
        
        # 格式3: 有 target_file 但 search/replace 在顶层 (缺少 edits 包裹)
        if change.get("target_file") and ("search" in change or "replace" in change):
            search_text = change.pop("search", "")
            replace_text = change.pop("replace", "")
            change["edits"] = [{"search": search_text, "replace": replace_text}]
            logger.info(f"Format normalized: flat search/replace→edits for {change.get('target_file')}")
            return change
        
        # 无法识别的格式
        if not change.get("target_file") and not change.get("filename"):
            logger.warning(f"Cannot normalize change format: missing both target_file and filename. "
                          f"Available keys: {list(change.keys())}")
            return None
        
        return change

    def _validate_structural_changes(self, structural_changes: list) -> list:
        """验证结构修改列表 — 支持 edits (SEARCH/REPLACE) 和 new_code (旧格式) 两种输入"""
        valid_changes = []
        for change in structural_changes:
            # ── 修复 Bug C: 格式兼容 — 支持 filename/action 扁平格式 ──
            # LLM 可能输出多种格式:
            #   标准: {"target_file": "x.py", "edits": [{"search": "...", "replace": "..."}]}
            #   扁平: {"filename": "x.py", "action": "modify", "search": "...", "replace": "..."}
            #   混合: {"target_file": "x.py", "search": "...", "replace": "..."} (缺少 edits 包裹)
            change = self._normalize_change_format(change)
            if not change:
                continue
            
            target_file = change.get("target_file", "")
            if not target_file:
                logger.warning(f"Structural change missing target_file: {change.get('description', '?')}")
                continue
            
            # 如果在 SOURCE_FILE_MAP 中，直接允许
            # 否则检查文件是否存在于项目目录中
            if target_file not in self.adapter.SOURCE_FILE_MAP:
                file_path = os.path.join(self.config.project_root, target_file)
                if not os.path.exists(file_path):
                    logger.warning(f"target_file {target_file} not found in project, skipping")
                    continue
            
            # ── 新格式: edits (SEARCH/REPLACE) ──
            edits = change.get("edits", [])
            if edits:
                # 验证每个 edit 都有 search 和 replace
                valid_edits = []
                for edit in edits:
                    search = edit.get("search", "")
                    replace = edit.get("replace", "")
                    # search 可以为空 (纯追加), 但 replace 必须有内容
                    if not replace:
                        logger.warning(f"Edit missing replace text, skipping")
                        continue
                    # ── 修复 Bug A: 不再使用 clean_new_code 处理 search/replace ──
                    # clean_new_code 的"移除解释性前缀"逻辑会删除不以 def/class/import 开头的行,
                    # 导致 parser.add_argument 等代码被清空 → 所有匹配必然失败!
                    # SEARCH/REPLACE 格式本身就是纯代码, 只需清理 markdown 包裹即可.
                    search = self._clean_markdown_wrapper(search)
                    replace = self._clean_markdown_wrapper(replace)
                    valid_edits.append({"search": search, "replace": replace})
                
                if valid_edits:
                    change["edits"] = valid_edits
                    change["action_type"] = change.get("action_type", "modify")
                    valid_changes.append(change)
                else:
                    logger.warning(f"All edits invalid for {target_file}, skipping")
                continue
            
            # ── 旧格式兼容: new_code + insert_position ──
            new_code = change.get("new_code", "")
            if not new_code:
                logger.warning(f"Structural change missing both edits and new_code: {change.get('description', '?')}")
                continue
            
            # 清理 new_code (移除 markdown 标记等)
            new_code = StructureApplier.clean_new_code(new_code)
            change["new_code"] = new_code
            
            # action_type: 不强制要求，自动推断
            if not change.get("action_type"):
                if "class " in new_code and "def " not in change.get("target_class_or_function", ""):
                    change["action_type"] = "add_module"
                else:
                    change["action_type"] = "modify"
            
            valid_changes.append(change)
        
        return valid_changes

    def _extract_structural_from_case_analysis(self) -> list:
        """
        从案例分析的 improvement_suggestions 中提取结构修改
        
        如果案例分析中有 action_type == "structure_change" 的建议，
        但 LLM 主分析没有生成 structural_changes，则从案例分析中提取
        """
        ca = self._last_case_analysis
        if not ca or not ca.get("parse_success"):
            return []
        
        suggestions = ca.get("improvement_suggestions", [])
        structural = []
        for s in suggestions:
            if s.get("action_type") == "structure_change":
                # 案例分析的结构修改建议通常是描述性的，需要让 LLM 进一步细化
                source_keys = list(self.adapter.SOURCE_FILE_MAP.keys())
                structural.append({
                    "action_type": s.get("action_type", "add_module"),
                    "target_file": source_keys[0] if source_keys else "modules.py",
                    "target_class_or_function": "",
                    "description": s.get("description", ""),
                    "new_code": "",  # 空的 — 需要后续让 LLM 生成具体代码
                    "expected_effect": s.get("expected_effect", ""),
                    "risk_level": s.get("risk", "中"),
                })
        
        if structural:
            logger.info(f"Extracted {len(structural)} structural suggestions from case analysis")
            # 如果有结构修改建议但没有具体代码，需要额外调用 LLM 生成代码
            # 这里暂时返回，让 _phase_analyze_and_propose 的 prompt 更有针对性
        
        return []  # 暂时不自动提取，让 LLM 主分析自己生成

    def _parse_diff_to_params(self, parsed: dict) -> Optional[dict]:
        """
        将 LLM 的提案解析为具体的参数变更
        支持两种格式:
        1. JSON 格式: {"param_changes": {"lr": 0.0005, ...}}
        2. 自然语言描述: "将学习率改为 0.0005"
        """
        diff = parsed.get("diff", "")
        explanation = parsed.get("explanation", "")

        param_changes = {}

        # 格式 1: 完整的 JSON
        if parsed["diff_type"] == "json":
            try:
                data = json.loads(diff)
                # 可能嵌套在 param_changes 下
                if "param_changes" in data:
                    param_changes = data["param_changes"]
                else:
                    # 直接就是参数字典
                    param_changes = data
            except json.JSONDecodeError:
                pass

        # 格式 2: Python dict 字符串
        if not param_changes and "{" in diff:
            try:
                # 尝试解析 {...} 中的内容
                import ast
                # 找第一个 { 到最后一个 }
                start = diff.find("{")
                end = diff.rfind("}")
                if start >= 0 and end > start:
                    dict_str = diff[start:end + 1]
                    param_changes = ast.literal_eval(dict_str)
            except (ValueError, SyntaxError, Exception):
                pass

        # 格式 3: 从自然语言中提取参数变更
        if not param_changes:
            param_changes = self._extract_params_from_text(diff + "\n" + explanation)

        # 验证参数是否合法
        if param_changes:
            valid_changes = {}
            for key, value in param_changes.items():
                if key in self.adapter.TUNABLE_PARAMS:
                    param_info = self.adapter.TUNABLE_PARAMS[key]
                    # 类型转换
                    try:
                        if param_info["type"] == "float":
                            value = float(value)
                        elif param_info["type"] == "int":
                            value = int(value)
                        # 检查范围
                        if "range" in param_info:
                            lo, hi = param_info["range"]
                            if lo <= value <= hi:
                                valid_changes[key] = value
                            else:
                                logger.warning(f"{key}={value} out of range [{lo},{hi}]")
                        elif "choices" in param_info:
                            if value in param_info["choices"]:
                                valid_changes[key] = value
                            else:
                                logger.warning(f"{key}={value} not in {param_info['choices']}")
                        else:
                            valid_changes[key] = value
                    except (ValueError, TypeError):
                        logger.warning(f"Cannot convert {key}={value} to {param_info['type']}")
                else:
                    # 不在可调参列表但可能合法（如新的 backbone 名）
                    if key != "param_changes" and key != "analysis" and key != "explanation":
                        valid_changes[key] = value

            return valid_changes if valid_changes else None

        return None

    @staticmethod
    def _extract_params_from_text(text: str) -> dict:
        """从自然语言中提取参数 = 值的对"""
        params = {}
        # 模式: "将学习率(learning rate/lr)改为0.0005" 或 "lr=0.0005"
        patterns = [
            r"(?:将|把)\s*(?:学习率|learning_rate|lr)\s*(?:改为|设为|设置为|调为)\s*([\d.e+\-]+)",
            r"(?:lr|learning_rate|batch_size|hidden_size|dropout|epochs|N|M|K)\s*[=:]\s*([\d.e+\-]+)",
            r"(?:loss_type|neg_sampler|CL_type|hidden_act|backbone|temperature)\s*[=:]\s*['\"]?(\w+)['\"]?",
        ]

        # 提取键值对
        kv_pattern = r"['\"]?(\w+)['\"]?\s*[=:]\s*['\"]?([\d.e+\-\w]+)['\"]?"
        matches = re.findall(kv_pattern, text)
        for k, v in matches:
            if k in ["lr", "loss_type", "neg_sampler", "hidden_size", "batch_size",
                     "epochs", "N", "M", "K", "backbone", "CL_type", "hidden_act",
                     "hidden_dropout_prob", "num_hidden_layers", "weight_decay",
                     "dropout", "seed", "start_epoch", "gpu_id", "max_seq_length",
                     "num_attention_heads", "d_state", "d_conv", "expand", "temperature",
                     "tau", "margin"]:
                try:
                    if "." in v:
                        params[k] = float(v)
                    else:
                        params[k] = int(v) if v.isdigit() else v
                except ValueError:
                    params[k] = v
        return params

    # ════════════════════════════════════════
    # 自纠错闭环 (核心新增!)
    # ════════════════════════════════════════

    def _phase_error_feedback_and_retry(
        self,
        iteration: int,
        original_proposal: dict,
        train_error: dict,
        metrics_before: dict,
        max_rounds: int = None,
    ) -> Optional[dict]:
        """
        自纠错闭环: 训练失败 → 把错误信息反馈给LLM → LLM修正方案 → 重新训练验证
        
        这是让 Agent 具备"Think-Code-Verify → Refinement"能力的关键环节!
        对应 Self-EvolveRec 论文中的诊断-模型协同进化机制。
        
        工作流程:
        1. 将原始提案 + 错误信息 + 当前源码发送给 LLM
        2. LLM 分析错误根因, 提出修正方案
        3. 解析修正方案 → 应用 → 重新训练验证
        4. 如果修正后仍然失败, 最多重试 max_rounds 次
        5. 全部失败则回滚所有修改
        
        Args:
            iteration: 当前迭代轮次
            original_proposal: 原始的 LLM 提案 (param_changes + structural_changes)
            train_error: 训练失败的详细信息 (status, error, log, action)
            metrics_before: 修改前的基线指标
            max_rounds: 最大自纠错轮数 (默认使用 self._max_self_correction_rounds)
            
        Returns:
            Dict: 修正后的最终结果 (与正常训练成功的返回格式一致)
            或 None: 自纠错全部失败
        """
        max_r = max_rounds or self._max_self_correction_rounds
        
        print(f"\n  🔁 自纠错闭环启动: 最多 {max_r} 轮修正")
        print(f"     错误类型: {train_error.get('status', 'UNKNOWN')}")
        print(f"     错误摘要: {train_error.get('error', '')}")
        
        for round_idx in range(1, max_r + 1):
            print(f"\n  🔁 自纠错第 {round_idx}/{max_r} 轮")
            
            # ── 1. 构建当前源码上下文 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
                iterative_memory=self.iter_memory,
            )
            
            # ── 2. 构建历史摘要 ──
            journal_summary = self.journal.summarize(n=5)
            if self.config.llm_enable_semantic_compression:
                journal_summary = self.context_compressor.compress_text(
                    journal_summary,
                    chunk_chars=self.config.llm_compression_chunk_chars,
                    target_chars=3000,
                    section_name="self_correction_journal",
                    profile="journal",
                )
            journal_summary = self._clip_text_by_chars(journal_summary, max_chars=3000)

            # ── 3. 构建回滚黑名单 ──
            rollback_warning = self.iter_memory.build_rollback_aware_context()
            if self.config.llm_enable_semantic_compression:
                rollback_warning = self.context_compressor.compress_text(
                    rollback_warning,
                    chunk_chars=self.config.llm_compression_chunk_chars,
                    target_chars=2200,
                    section_name="rollback_warning",
                    profile="rollback",
                )
            rollback_warning = self._clip_text_by_chars(rollback_warning, max_chars=3000)
            structural_history = self.iter_memory.build_history_context_for_llm(
                current_iteration=iteration,
                current_metrics=metrics_before,
            )
            if self.config.llm_enable_semantic_compression:
                structural_history = self.context_compressor.compress_text(
                    structural_history,
                    chunk_chars=self.config.llm_compression_chunk_chars,
                    target_chars=3600,
                    section_name="self_correction_struct_history",
                    profile="history",
                )
            structural_history = self._clip_text_by_chars(structural_history, max_chars=4500)
            
            # ── 4. 构建 ERROR_FEEDBACK_PROMPT ──
            # 直接将完整错误信息交给 LLM 分析，不做人工分类
            error_msg = train_error.get("error", "")
            
            prompt = ERROR_FEEDBACK_PROMPT.format(
                _original_proposal=json.dumps(original_proposal, indent=2, ensure_ascii=False),
                _error_message=error_msg[:1500],
                _error_log=train_error.get("log", "")[:2000],
                _current_source_code=source_code_ctx,
                _experiment_journal=journal_summary,
                current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
            )
            
            # 添加回滚黑名单警告
            if rollback_warning:
                prompt += rollback_warning
            if structural_history:
                prompt += structural_history
            
            # ── 5. 调用 LLM ──
            logger.info(f"Self-correction round {round_idx}: sending error feedback to LLM")
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的推荐系统算法专家，擅长诊断和修复代码错误。"
                        "你现在需要根据训练/运行错误信息，修正你之前提出的改进方案。"
                        "修正后的代码必须是可执行的完整 Python 代码，不能有省略号或占位符。"
                        "特别注意维度对齐、import完整性、变量名一致性。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # 自纠错时使用低温度，确保精确性
                max_tokens=self.config.llm_max_tokens,
                suppress_response_log=True,  # 代码响应不输出到日志
            )
            
            if response is None:
                logger.error(f"Self-correction round {round_idx}: LLM call failed")
                print(f"     ✗ LLM 调用失败")
                continue
            
            # ── 6. 解析修正后的提案 ──
            revised_proposal = self._parse_proposal_response(response)
            if revised_proposal is None:
                logger.warning(f"Self-correction round {round_idx}: failed to parse LLM response")
                print(f"     ✗ 无法解析 LLM 修正方案")
                # 尝试用 LLMFixer 修复格式
                fixed_response = self.fixer.fix_format(response, "无法从回复中提取有效JSON")
                revised_proposal = self._parse_proposal_response(fixed_response)
                if revised_proposal is None:
                    continue
            
            revised_param_changes = revised_proposal.get("param_changes", {})
            revised_structural_changes = revised_proposal.get("structural_changes", [])
            
            print(f"     💡 修正方案:")
            print(f"       参数修改: {json.dumps(revised_param_changes, ensure_ascii=False)[:200]}")
            if revised_structural_changes:
                print(f"       结构修改: {len(revised_structural_changes)} 项")
            print(f"       诊断: {revised_proposal.get('explanation', '')[:150]}")
            
            # ── 7. 应用修正后的结构修改 ──
            revised_struct_result = None
            if revised_structural_changes:
                # 先回滚之前可能有残留的结构修改
                self.struct_applier.rollback_last_changes()
                
                print(f"     🏗️ 应用修正后的结构修改...")
                revised_struct_result = self.struct_applier.apply_structural_changes(revised_structural_changes)
                
                if revised_struct_result["status"] == "ROLLBACK":
                    print(f"     ↩ 修正后的结构修改也被回滚: {revised_struct_result.get('rollback_reason', '')[:100]}")
                    # 结构修改全部失败，但可以尝试纯参数修改
                    revised_structural_changes = []
                    revised_struct_result = None
                elif revised_struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                    print(f"     ✓ 修正后的结构修改已应用")
                elif revised_struct_result["status"] == "ALL_FAILED":
                    print(f"     ✗ 修正后的结构修改全部失败")
                    revised_structural_changes = []
                    revised_struct_result = None
            
            # ── 8. 用修正后的参数重新训练 ──
            has_any_change = bool(revised_param_changes) or bool(revised_structural_changes)
            if not has_any_change:
                print(f"     ⚠ 修正方案也没有有效变更，放弃本轮自纠错")
                continue
            
            print(f"     🔄 用修正后的方案重新训练...")
            revised_train = self.trainer.run(
                param_overrides=revised_param_changes if revised_param_changes else None,
            )
            
            # ── 9. 检查训练结果 ──
            if revised_train["status"] == "SUCCESS":
                revised_metrics = revised_train.get("metrics", {})
                print(f"     ✓ 自纠错成功! 修正后的训练通过")
                print(f"     → 修正后: {self.adapter.format_metrics_for_llm(revised_metrics)}")
                
                # 记录修改因果链到 IterativeMemory
                if revised_structural_changes and revised_struct_result:
                    self.iter_memory.record_modification(
                        iteration=iteration,
                        structural_changes=revised_structural_changes,
                        apply_result=revised_struct_result,
                        metrics_before=metrics_before,
                        metrics_after=revised_metrics,
                    )
                
                # 记录自纠错成功到 journal
                self.journal.record({
                    "iteration": iteration,
                    "status": "SELF_CORRECTION_SUCCESS",
                    "self_correction_round": round_idx,
                    "original_proposal": original_proposal,
                    "revised_proposal": revised_proposal,
                    "error_type": error_msg[:200] or train_error.get("status", "UNKNOWN"),
                    "revised_metrics": revised_metrics,
                    "param_changes": revised_param_changes,
                    "structural_changes": revised_structural_changes,
                    "explanation": f"自纠错第{round_idx}轮修正成功 (原错误: {error_msg[:80]})",
                })
                
                # 返回修正后的成功结果
                return {
                    "status": "SELF_CORRECTION_SUCCESS",
                    "metrics": revised_metrics,
                    "param_changes": revised_param_changes,
                    "structural_changes": revised_structural_changes,
                    "struct_result": revised_struct_result,
                    "explanation": revised_proposal.get("explanation", ""),
                    "self_correction_round": round_idx,
                }
            
            else:
                # 修正后仍然失败 → 更新 error 信息用于下一轮自纠错
                train_error = revised_train  # 用新的错误信息继续自纠错
                print(f"     ✗ 修正后仍然失败: {revised_train.get('error', '')}")
                
                # 如果有结构修改导致失败，回滚
                if revised_structural_changes and revised_struct_result and \
                   revised_struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                    self.struct_applier.rollback_last_changes()
                    print(f"     ↩ 回滚修正后的结构修改")
                    self.iter_memory.record_rollback(
                        iteration=iteration,
                        reason=f"Self-correction round {round_idx} failed: {revised_train.get('error', 'unknown')[:200]}",
                    )
                
                # 记录自纠错失败到 journal
                self.journal.record({
                    "iteration": iteration,
                    "status": "SELF_CORRECTION_FAILED",
                    "self_correction_round": round_idx,
                    "revised_error": revised_train.get("error", "")[:500],
                    "revised_proposal_summary": json.dumps(revised_proposal, ensure_ascii=False)[:300],
                })
                
                # 更新 original_proposal 为修正后的方案 (下一轮基于修正版继续修正)
                original_proposal = revised_proposal
        
        # ── 所有自纠错轮次都失败了 ──
        print(f"\n  ✗ 自纠错闭环: 所有 {max_r} 轮修正均失败")
        logger.error(f"Self-correction loop exhausted {max_r} rounds, all failed")
        
        # 回滚所有残留的结构修改
        self.struct_applier.rollback_last_changes()
        
        # 回退参数到 best_config
        if self.guard.best_config:
            print(f"  ↩ 回退到最佳参数配置")
            self.trainer.run(param_overrides=self.guard.best_config)
        
        self.journal.record({
            "iteration": iteration,
            "status": "SELF_CORRECTION_ALL_FAILED",
            "max_rounds": max_r,
            "original_error_type": error_msg or train_error.get("status", "UNKNOWN"),
        })
        
        return None

    def _phase_structure_validation_retry(
        self,
        iteration: int,
        original_structural_changes: list,
        validation_failures: list,
        max_rounds: int = None,
    ) -> Optional[dict]:
        """
        自纠错闭环: 结构修改代码校验失败 → 让LLM修正代码 → 重新校验
        
        当 StructureApplier 检测到语法错误、维度不匹配等问题时，
        不是直接回滚跳过，而是把问题反馈给 LLM 让它修正代码。
        
        Args:
            iteration: 当前迭代轮次
            original_structural_changes: 原始的结构修改列表
            validation_failures: 校验失败的详细信息列表
            max_rounds: 最大自纠错轮数
            
        Returns:
            Dict: StructureApplier.apply_structural_changes 的返回格式
            或 None: 自纠错全部失败
        """
        max_r = max_rounds or self._max_self_correction_rounds
        
        print(f"\n  🔁 结构修改自纠错启动: 最多 {max_r} 轮修正")
        for vf in validation_failures:
            print(f"     问题: {vf.get('description', '?')} — {vf.get('error', '')[:100]}")
        
        for round_idx in range(1, max_r + 1):
            print(f"\n  🔁 结构修改自纠错第 {round_idx}/{max_r} 轮")
            
            # ── 1. 构建当前源码上下文 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=15000,
                iterative_memory=self.iter_memory,
            )
            
            # ── 2. 构建校验失败详情 ──
            failures_detail = ""
            for vf in validation_failures:
                failures_detail += f"\n### 失败项: {vf.get('description', '?')}\n"
                failures_detail += f"- **错误类型**: {vf.get('error_type', 'unknown')}\n"
                failures_detail += f"- **错误信息**: {vf.get('error', '')[:500]}\n"
                failures_detail += f"- **目标**: {vf.get('target_class_or_function', '?')}\n"
                if vf.get('original_code'):
                    failures_detail += f"- **原始代码片段**:\n```python\n{vf.get('original_code', '')[:300]}\n```\n"
            
            # ── 3. 构建 STRUCTURE_FIX_PROMPT ──
            prompt = STRUCTURE_FIX_PROMPT.format(
                _original_structural_changes=json.dumps(original_structural_changes, indent=2, ensure_ascii=False),
                _validation_failures=failures_detail,
                _current_source_code=source_code_ctx,
                current_hidden_size=self.adapter.base_args.get("hidden_size", 64),
            )
            
            # ── 4. 调用 LLM ──
            logger.info(f"Structure validation retry round {round_idx}: sending to LLM")
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的 Python 代码修复专家。"
                        "你需要根据代码校验失败的具体错误信息，修正之前提出的模型结构修改代码。"
                        "修正后的代码必须: 1) 语法正确 2) 维度与 hidden_size 对齐 "
                        "3) import 完整 4) 与现有代码兼容 5) 是可执行的完整代码，无省略号"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,  # 代码修正使用极低温度，确保精确性
                max_tokens=self.config.llm_max_tokens,
                suppress_response_log=True,  # 代码响应不输出到日志
            )
            
            if response is None:
                logger.error(f"Structure fix round {round_idx}: LLM call failed")
                print(f"     ✗ LLM 调用失败")
                continue
            
            # ── 5. 解析修正后的结构修改 ──
            revised_proposal = self._parse_proposal_response(response)
            if revised_proposal is None:
                print(f"     ✗ 无法解析 LLM 修正方案")
                fixed_response = self.fixer.fix_format(response, "无法从回复中提取有效JSON")
                revised_proposal = self._parse_proposal_response(fixed_response)
                if revised_proposal is None:
                    continue
            
            revised_structural_changes = revised_proposal.get("structural_changes", [])
            if not revised_structural_changes:
                print(f"     ⚠ LLM 修正方案中没有结构修改")
                continue
            
            print(f"     💡 修正后的结构修改: {len(revised_structural_changes)} 项")
            for sc in revised_structural_changes:
                print(f"       → [{sc.get('target_file', '?')}] "
                      f"{sc.get('target_class_or_function', '?')}: "
                      f"{sc.get('description', '?')[:60]}...")
            
            # ── 6. 重新应用修正后的结构修改 ──
            revised_struct_result = self.struct_applier.apply_structural_changes(revised_structural_changes)
            
            if revised_struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                print(f"     ✓ 修正后的结构修改校验通过!")
                print(f"       状态: {revised_struct_result['status']}")
                print(f"       修改文件: {revised_struct_result.get('files_modified', [])}")
                
                # 记录自纠错成功到 journal
                self.journal.record({
                    "iteration": iteration,
                    "status": "STRUCTURE_FIX_SUCCESS",
                    "self_correction_round": round_idx,
                    "original_failures": validation_failures,
                    "revised_structural_changes_summary": json.dumps(revised_structural_changes, ensure_ascii=False)[:500],
                })
                
                return revised_struct_result
            
            elif revised_struct_result["status"] == "ROLLBACK":
                # 修正后的代码仍然有校验问题，继续下一轮
                print(f"     ↩ 修正后的结构修改仍被回滚: {revised_struct_result.get('rollback_reason', '')[:100]}")
                # 更新 validation_failures 为新的失败信息
                validation_failures = revised_struct_result.get("failed_changes", validation_failures)
                # 更新 original_structural_changes 为修正后的版本
                original_structural_changes = revised_structural_changes
                continue
            
            else:
                print(f"     ✗ 修正后的结构修改全部失败")
                continue
        
        # ── 所有自纠错轮次都失败了 ──
        print(f"\n  ✗ 结构修改自纠错: 所有 {max_r} 轮修正均失败")
        logger.error(f"Structure fix loop exhausted {max_r} rounds, all failed")
        
        self.journal.record({
            "iteration": iteration,
            "status": "STRUCTURE_FIX_ALL_FAILED",
            "max_rounds": max_r,
        })
        
        return None

    def _phase_search_replace_retry(
        self,
        iteration: int,
        original_structural_changes: list,
        detailed_failure_info: list,
        max_rounds: int = None,
    ) -> Optional[dict]:
        """
        SEARCH/REPLACE 匹配失败重试机制
        
        当 StructureApplier 的 SEARCH/REPLACE 三级匹配全部失败时,
        不是直接放弃, 而是将详细的匹配失败诊断 + 实际文件源码
        反馈给 LLM, 让它根据真实源码重新编写 search 文本。
        
        这是确保每轮修改都成功应用的关键机制 — 修改失败不能
        直接跳过, 因为用原代码训练没有意义。
        
        Args:
            iteration: 当前迭代轮次
            original_structural_changes: 原始的结构修改列表 (LLM 第一次提出的)
            detailed_failure_info: StructureApplier 返回的详细失败诊断
            max_rounds: 最大重试轮数
            
        Returns:
            Dict: StructureApplier.apply_structural_changes 的返回格式
            或 None: 重试全部失败
        """
        max_r = max_rounds or self._max_self_correction_rounds
        
        print(f"\n  🔁 SEARCH/REPLACE 重试启动: 最多 {max_r} 轮修正")
        
        # ── 构建失败诊断文本 ──
        failure_diagnostic_text = self._format_match_failure_diagnostic(detailed_failure_info)
        
        # ── 计算最佳模糊匹配比率 (用于 prompt) ──
        best_fuzzy_ratio = 0.0
        fuzzy_threshold = 0.80
        for df in detailed_failure_info:
            for ed in df.get("failed_edit_details", []):
                diag = ed.get("match_diagnostic", {})
                ratio = diag.get("best_fuzzy_ratio", 0.0)
                if ratio > best_fuzzy_ratio:
                    best_fuzzy_ratio = ratio
                fuzzy_threshold = diag.get("threshold", 0.80)
        
        for round_idx in range(1, max_r + 1):
            print(f"\n  🔁 SEARCH/REPLACE 重试第 {round_idx}/{max_r} 轮")
            
            # ── 1. 构建当前源码上下文 ──
            # 对于匹配失败, 最重要的是让 LLM 看到 实际的源码
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=list(self.adapter.SOURCE_FILE_MAP.keys()),
                max_total_chars=20000,  # 给更多空间让 LLM 看到真实源码
                iterative_memory=self.iter_memory,
            )
            
            # ── 2. 构建 SEARCH_REPLACE_FIX_PROMPT ──
            prompt = SEARCH_REPLACE_FIX_PROMPT.format(
                _original_structural_changes=json.dumps(
                    original_structural_changes, indent=2, ensure_ascii=False
                ),
                _match_failure_diagnostic=failure_diagnostic_text,
                _current_source_code=source_code_ctx,
                _best_fuzzy_ratio=best_fuzzy_ratio,
                _fuzzy_threshold=fuzzy_threshold,
            )
            
            # ── 3. 调用 LLM ──
            logger.info(f"SEARCH/REPLACE retry round {round_idx}: sending to LLM")
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的 Python 代码修改专家。"
                        "你之前的 SEARCH/REPLACE 修改无法在目标文件中找到匹配。"
                        "这说明你对源码内容的记忆与实际文件不一致。"
                        "请仔细阅读下面提供的 实际源码, 从中精确复制你想修改的代码片段作为 search 文本, "
                        "然后编写修改后的代码作为 replace 文本。"
                        "search 文本必须与实际源码 逐行完全一致 (包括缩进、空行、注释), "
                        "不能凭记忆编写或省略任何字符。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 极低温度, 确保精确复制源码
                max_tokens=self.config.llm_max_tokens,
            )
            
            if response is None:
                logger.error(f"SEARCH/REPLACE retry round {round_idx}: LLM call failed")
                print(f"     ✗ LLM 调用失败")
                continue
            
            # ── 4. 解析修正后的结构修改 ──
            revised_proposal = self._parse_proposal_response(response)
            if revised_proposal is None:
                print(f"     ✗ 无法解析 LLM 修正方案")
                fixed_response = self.fixer.fix_format(response, "无法从回复中提取有效JSON")
                revised_proposal = self._parse_proposal_response(fixed_response)
                if revised_proposal is None:
                    continue
            
            revised_structural_changes = revised_proposal.get("structural_changes", [])
            if not revised_structural_changes:
                print(f"     ⚠ LLM 修正方案中没有结构修改")
                continue
            
            print(f"     💡 修正后的结构修改: {len(revised_structural_changes)} 项")
            for sc in revised_structural_changes:
                print(f"       → [{sc.get('target_file', '?')}] "
                      f"{sc.get('target_class_or_function', '?')}: "
                      f"{sc.get('description', '?')[:60]}...")
            
            # ── 5. 重新应用修正后的结构修改 ──
            revised_struct_result = self.struct_applier.apply_structural_changes(revised_structural_changes)
            
            if revised_struct_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                print(f"     ✓ 修正后的结构修改成功!")
                print(f"       状态: {revised_struct_result['status']}")
                print(f"       修改文件: {revised_struct_result.get('files_modified', [])}")
                
                self.journal.record({
                    "iteration": iteration,
                    "status": "SEARCH_REPLACE_RETRY_SUCCESS",
                    "self_correction_round": round_idx,
                    "original_structural_changes_summary": json.dumps(
                        original_structural_changes, ensure_ascii=False
                    )[:500],
                    "revised_structural_changes_summary": json.dumps(
                        revised_structural_changes, ensure_ascii=False
                    )[:500],
                    "detailed_failure_info": detailed_failure_info,
                })
                
                return revised_struct_result
            
            elif revised_struct_result["status"] == "ALL_FAILED":
                # 修正后仍然 ALL_FAILED → 更新诊断, 继续下一轮
                print(f"     ✗ 修正后仍然 ALL_FAILED — 继续下一轮重试")
                new_detailed_failure = revised_struct_result.get("detailed_failure_info", [])
                if new_detailed_failure:
                    detailed_failure_info = new_detailed_failure
                    failure_diagnostic_text = self._format_match_failure_diagnostic(detailed_failure_info)
                # 更新 original_structural_changes 为修正后的版本
                original_structural_changes = revised_structural_changes
                # 更新最佳模糊比率
                for df in detailed_failure_info:
                    for ed in df.get("failed_edit_details", []):
                        diag = ed.get("match_diagnostic", {})
                        ratio = diag.get("best_fuzzy_ratio", 0.0)
                        if ratio > best_fuzzy_ratio:
                            best_fuzzy_ratio = ratio
                continue
            
            elif revised_struct_result["status"] == "ROLLBACK":
                # 修正后的代码有校验问题 → 转交给 validation retry 处理
                print(f"     ↩ 修正后的结构修改被回滚: {revised_struct_result.get('rollback_reason', '')[:100]}")
                # 先尝试 validation retry
                validation_failures = revised_struct_result.get("failed_changes", [])
                if not validation_failures:
                    validation_failures = [{
                        "description": sc.get("description", "unknown"),
                        "error_type": "validation_error",
                        "error": revised_struct_result.get("rollback_reason", "validation failed"),
                        "target_class_or_function": sc.get("target_class_or_function", "?"),
                    } for sc in revised_structural_changes]
                
                val_fix_result = self._phase_structure_validation_retry(
                    iteration=iteration,
                    original_structural_changes=revised_structural_changes,
                    validation_failures=validation_failures,
                )
                if val_fix_result and val_fix_result["status"] in ("SUCCESS", "PARTIAL_SUCCESS"):
                    print(f"     ✓ Validation retry 成功!")
                    return val_fix_result
                else:
                    print(f"     ✗ Validation retry 也失败 — 继续下一轮 SEARCH/REPLACE 重试")
                    continue
            
            else:
                print(f"     ✗ 修正后的结构修改状态未知: {revised_struct_result['status']}")
                continue
        
        # ── 所有重试轮次都失败了 ──
        print(f"\n  ✗ SEARCH/REPLACE 重试: 所有 {max_r} 轮修正均失败")
        logger.error(f"SEARCH/REPLACE retry loop exhausted {max_r} rounds, all failed")
        
        self.journal.record({
            "iteration": iteration,
            "status": "SEARCH_REPLACE_RETRY_ALL_FAILED",
            "max_rounds": max_r,
            "detailed_failure_info": detailed_failure_info,
        })
        
        return None

    @staticmethod
    def _format_match_failure_diagnostic(detailed_failure_info: list) -> str:
        """
        将 detailed_failure_info 格式化为人类可读的文本, 用于发送给 LLM
        
        包含:
        - 每个失败项的目标文件和描述
        - 每个 edit 的 search 文本全文 (不截断!)
        - 模糊匹配的详细诊断 (相似度、最相似片段、不匹配行)
        - 实际文件中与 search 最相似的代码片段及上下文
        """
        text = ""
        for idx, df in enumerate(detailed_failure_info, 1):
            text += f"\n### 失败项 {idx}: [{df.get('target_file', '?')}] {df.get('description', '?')[:80]}\n"
            text += f"- **错误**: {df.get('error', 'unknown')[:200]}\n\n"
            
            for ed in df.get("failed_edit_details", []):
                edit_idx = ed.get("edit_idx", "?")
                text += f"\n#### Edit {edit_idx} 匹配失败详情\n"
                
                # ── 展示 LLM 提供的 search 文本全文 ──
                search_full = ed.get("search_text_full", "")
                if search_full:
                    text += f"- **LLM 提供的 search 文本** (全文):\n```\n{search_full}\n```\n"
                else:
                    search_preview = ed.get("search_text_preview", "")
                    text += f"- **LLM 提供的 search 文本** (截断): `{search_preview}`\n"
                
                # ── 展示匹配诊断 ──
                diag = ed.get("match_diagnostic", {})
                if diag:
                    text += f"- **Level 1 精确匹配**: ❌ 失败\n"
                    text += f"  - search 中与文件内容匹配的行数: {diag.get('level1_matching_lines', 'N/A')}\n"
                    text += f"  - search 中与文件内容不匹配的行数: {diag.get('level1_non_matching_lines', 'N/A')}\n"
                    non_match_samples = diag.get('level1_non_matching_samples', [])
                    if non_match_samples:
                        text += f"  - 不匹配行示例:\n"
                        for sample in non_match_samples[:3]:
                            text += f"    `{sample}`\n"
                    text += f"- **Level 2 去空白匹配**: {'✅ 成功' if diag.get('level2_whitespace_match') else '❌ 失败'}\n"
                    text += f"- **Level 3 模糊匹配**: ❌ 失败 (best_ratio={diag.get('best_fuzzy_ratio', 'N/A')}, "
                    text += f"threshold={diag.get('level3_fuzzy_threshold', 'N/A')})\n"
                    
                    # ── 展示文件中与 search 最相似的代码片段 ──
                    closest_context = diag.get("closest_match_segment_with_context")
                    if closest_context:
                        text += f"\n- **文件中与 search 最相似的代码片段** (上下文 ±3 行):\n```\n{closest_context}\n```\n"
                
                text += "\n"
        
        return text

    def _classify_error_with_llm(self, train_result: dict) -> str:
        """
        让 LLM 判断错误分类 (替代硬编码的字符串匹配)
        
        返回: "CODE_ERROR" | "CONFIG_ERROR" | "DATA_ERROR" | "SYSTEM_ERROR"
        
        这是最关键的改变: 不再用 if/elif 级联做字符串匹配来分类,
        而是把完整的错误信息交给 LLM, 让它推理判断。
        
        原因: 硬编码分类经常不准确, 比如:
        - RuntimeError: shape '[256, 1, -1]' 是维度不匹配(CODE_ERROR), 但硬编码会匹配为 "RuntimeError" → 可能被误标为 CONFIG_ERROR
        - CUDA error 可能是显存不足(CONFIG_ERROR), 也可能是代码中的内存访问错误(CODE_ERROR)
        - FileNotFoundError 可能是数据路径(DATA_ERROR), 也可能是项目源码缺失(CODE_ERROR)
        
        LLM 能看到完整的 traceback, 理解上下文, 给出更准确的分类。
        """
        error_msg = train_result.get("error", "")
        traceback_details = train_result.get("traceback_details", {})
        tb_text = traceback_details.get("traceback_text", "")
        returncode = train_result.get("returncode")
        
        # 构建 traceback 展示
        tb_display = ""
        if tb_text:
            tb_display += f"\n### 完整 Traceback:\n```\n{tb_text[:2000]}\n```"
        files = traceback_details.get("files", [])
        line_numbers = traceback_details.get("line_numbers", [])
        if files:
            tb_display += f"\n### 出错文件和行号:"
            for f, l in zip(files, line_numbers):
                tb_display += f"\n- **文件**: `{f}` @ **行 {l}**"
        if not tb_display:
            tb_display = f"\n### 错误信息:\n```\n{error_msg[:1500]}\n```"
        
        prompt = f"""训练运行失败，需要你判断错误的分类类别。

## 错误信息
```
{error_msg[:2000]}
```

{tb_display}

## 进程退出码: {returncode}

## 四种错误分类

请根据完整的错误信息 (包括 traceback 上下文), 判断这个错误属于哪一类:

- **CODE_ERROR**: 源码代码有问题 (语法错误、维度不匹配、变量名错误、运行时错误等)
  → 需要修改模型源码文件来修复
  → 判断依据: traceback 中的出错文件属于项目内部代码, 错误类型是语法/维度/变量等代码级问题
  
- **CONFIG_ERROR**: 参数配置有问题 (OOM显存溢出、NaN Loss、超时、指标格式不匹配等)
  → 需要调整超参数 (batch_size, lr, hidden_size 等) 或配置来修复
  → 判断依据: 错误与训练资源/参数有关, 如显存不足、训练发散、命令格式问题
  
- **DATA_ERROR**: 数据路径或格式有问题 (找不到数据文件、路径错误、格式不兼容等)
  → 需要修正数据路径或配置来修复
  → 判断依据: FileNotFoundError 且涉及 /data/ 路径, 或数据格式/编码问题
  
- **SYSTEM_ERROR**: 系统环境问题 (缺少外部包、GPU不可用、权限问题等)
  → 无法通过自纠错修复, 需要人工干预
  → 判断依据: 外部包缺失 (ModuleNotFoundError)、GPU不可用、权限被拒

### 输出格式

只输出一个 JSON, 格式如下:
```json
{{"error_category": "CODE_ERROR", "reason": "简短说明为什么是这个分类"}}
```

注意: 只输出这一个 JSON, 不要输出其他内容。"""

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位深度学习训练诊断专家, 擅长从错误信息中判断根因类别。"
                        "你需要根据完整的 traceback 上下文来判断错误分类, 不要只看错误类型名称。"
                        "例如: RuntimeError 可能是 CODE_ERROR (维度不匹配) 或 CONFIG_ERROR (OOM), "
                        "需要看 traceback 的具体内容和出错文件来判断。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # 分类判断用极低温度, 确保确定性
                max_tokens=150,   # 只需要输出分类, 不需要长回复
            )
            
            if response is None:
                logger.warning("LLM classification call failed, falling back to heuristic")
                return self._classify_error_heuristic(train_result)
            
            # 解析 LLM 回复
            # 尝试提取 JSON
            json_match = re.search(r'\{[^{}]+\}', response)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    category = result.get("error_category", "")
                    reason = result.get("reason", "")
                    
                    # 验证分类是合法的
                    valid_categories = ("CODE_ERROR", "CONFIG_ERROR", "DATA_ERROR", "SYSTEM_ERROR")
                    if category in valid_categories:
                        logger.info(f"LLM classified error as {category}: {reason}")
                        return category
                    else:
                        logger.warning(f"LLM returned invalid category '{category}', falling back to heuristic")
                        return self._classify_error_heuristic(train_result)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM classification JSON: {response[:200]}")
            
            # 尝试从文本中提取分类
            for cat in ("CODE_ERROR", "CONFIG_ERROR", "DATA_ERROR", "SYSTEM_ERROR"):
                if cat in response:
                    logger.info(f"LLM classification extracted from text: {cat}")
                    return cat
            
            # 所有尝试都失败 → 回退到启发式
            logger.warning("Could not extract classification from LLM response, falling back to heuristic")
            return self._classify_error_heuristic(train_result)
            
        except Exception as e:
            logger.error(f"Error during LLM classification: {e}")
            return self._classify_error_heuristic(train_result)

    @staticmethod
    def _classify_error_heuristic(train_result: dict) -> str:
        """
        启发式回退: 当 LLM 分类失败时的简单兜底
        
        只做最基本的判断:
        - 有 traceback → CODE_ERROR (源码问题)
        - 无 traceback → SYSTEM_ERROR (系统/环境问题)
        """
        traceback_details = train_result.get("traceback_details", {})
        if traceback_details.get("traceback_text"):
            return "CODE_ERROR"
        return "SYSTEM_ERROR"

    def _phase_train_diagnosis_and_retry(
        self,
        iteration: int,
        train_error: dict,
        max_rounds: int = None,
    ) -> Optional[dict]:
        """
        自纠错闭环: 基线训练失败 → 让LLM诊断 → 修复配置 → 重新训练
        
        当 Phase 1 的基线训练本身就失败时 (如配置参数有误、数据路径错误等),
        不是直接跳过迭代，而是让 LLM 诊断问题并修复。
        
        Args:
            iteration: 当前迭代轮次
            train_error: 训练失败的详细信息
            max_rounds: 最大自纠错轮数
            
        Returns:
            Dict: 训练成功的结果 (与 trainer.run() 的返回格式一致)
            或 None: 自纠错全部失败
        """
        max_r = max_rounds or 5  # 基线训练诊断增加到 5 轮!
        
        print(f"\n  🔁 基线训练诊断自纠错启动: 最多 {max_r} 轮")
        print(f"     错误: {train_error.get('status', 'UNKNOWN')} — {train_error.get('error', '')[:100]}")
        
        for round_idx in range(1, max_r + 1):
            print(f"\n  🔁 基线训练诊断第 {round_idx}/{max_r} 轮")
            
            # ── 构建诊断 Prompt ──
            error_msg = train_error.get("error", "")
            error_type = train_error.get("status", "UNKNOWN")
            
            # 构建当前训练命令
            train_cmd = self.adapter.build_train_command()
            
            # 构建项目配置
            project_config = json.dumps(self.adapter.base_args, indent=2, ensure_ascii=False)
            
            prompt = TRAIN_DIAGNOSIS_PROMPT.format(
                _error_type=error_type,
                _error_message=error_msg[:2000],
                _project_config=project_config,
                _train_command=train_cmd,
            )
            
            # ── 调用 LLM ──
            logger.info(f"Train diagnosis round {round_idx}: sending to LLM")
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位经验丰富的深度学习训练工程师，擅长诊断训练失败的根因。"
                        "你需要根据错误信息判断是配置问题、数据问题还是代码问题，并给出修复建议。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            
            if response is None:
                print(f"     ✗ LLM 调用失败")
                continue
            
            # ── 解析诊断结果 ──
            try:
                # 尝试提取 JSON
                json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
                if json_match:
                    diagnosis = json.loads(json_match.group(1))
                else:
                    start = response.find('{')
                    end = response.rfind('}')
                    if start >= 0 and end > start:
                        diagnosis = json.loads(response[start:end + 1])
                    else:
                        logger.warning("Cannot parse train diagnosis response")
                        continue
            except json.JSONDecodeError:
                logger.warning("Train diagnosis JSON parse error")
                continue
            
            param_fixes = diagnosis.get("param_changes", {})
            explanation = diagnosis.get("explanation", "")
            
            print(f"     💡 诊断: {explanation[:150]}")
            if param_fixes:
                print(f"       参数修复: {json.dumps(param_fixes, ensure_ascii=False)[:200]}")
            
            # ── 应用参数修复 ──
            if param_fixes:
                validated_fixes = self._validate_param_changes(param_fixes)
                if validated_fixes:
                    print(f"     🔄 用诊断修复的参数重新训练...")
                    retry_result = self.trainer.run(param_overrides=validated_fixes)
                    
                    if retry_result["status"] == "SUCCESS":
                        print(f"     ✓ 基线训练诊断修复成功!")
                        print(f"       指标: {self.adapter.format_metrics_for_llm(retry_result.get('metrics', {}))}")
                        
                        self.journal.record({
                            "iteration": iteration,
                            "status": "TRAIN_DIAGNOSIS_SUCCESS",
                            "self_correction_round": round_idx,
                            "diagnosis": diagnosis,
                            "param_fixes": validated_fixes,
                        })
                        
                        return retry_result
                    
                    else:
                        # 诊断修复后仍然失败
                        train_error = retry_result
                        print(f"     ✗ 诊断修复后仍然失败: {retry_result.get('error', '')[:100]}")
                        continue
            
            # LLM 只给了建议但没有参数修复 → 尝试直接重试 (可能是临时错误)
            print(f"     ⚠ 诊断只有建议，没有具体参数修复")
            continue
        
        # ── 所有诊断轮次都失败了 ──
        print(f"\n  ✗ 基线训练诊断: 所有 {max_r} 轮均失败")
        return None

    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════

    def _check_llm_health(self):
        ok = self.llm.check_health()
        self.llm_health_ok = ok
        print(f"  {'✓' if ok else '✗'} LLM health: {self.config.llm_api_url} [{self.config.llm_model}]")

    def _get_strategy_instruction(self) -> str:
        instructions = {
            "balanced": "当前策略: 自由探索 — 你可以观察现象后自由决定改进方向",
            "aggressive": "当前策略: 大胆探索 — 不受约束地尝试你认为最有潜力的方案",
            "conservative": "当前策略: 稳健优化 — 在已验证的基础上谨慎改进",
            "explorative": "当前策略: 探索驱动 — 从实验现象出发发现新的优化思路",
            "focused": "当前策略: 问题驱动 — 仔细观察实验数据中的短板并针对性地改进",
        }
        return instructions.get(self.current_strategy, "")

    def _get_temperature(self) -> float:
        temps = {"balanced": 0.7, "aggressive": 0.9,
                 "explorative": 0.85, "conservative": 0.4, "focused": 0.6}
        return temps.get(self.current_strategy, 0.7)

    @staticmethod
    def _clip_text_by_chars(text: str, max_chars: int) -> str:
        """按字符裁剪上下文，保留头尾，避免单段文本无限增长。"""
        if not text:
            return ""
        if len(text) <= max_chars:
            return text
        if max_chars < 120:
            return text[:max_chars]
        head = int(max_chars * 0.7)
        tail = max_chars - head - 30
        return text[:head] + "\n\n... [CLIPPED] ...\n\n" + text[-max(0, tail):]

    @staticmethod
    def _print_final_report(result: dict):
        print(f"\n{'='*60}")
        print(f"  进化完成 (参数优化 + 结构优化)")
        print(f"{'='*60}")
        print(f"  总迭代: {result['total_iterations']}")
        print(f"  最优轮次: {result['best_iteration']}")
        if result['best_metrics']:
            print(f"  最优指标:")
            for k, v in result['best_metrics'].items():
                print(f"    {k} = {v:.4f}" if isinstance(v, float) else f"    {k} = {v}")
        # 显示结构修改历史 (来自 IterativeMemory)
        struct_summary = result.get("structural_change_summary", {})
        if struct_summary and struct_summary.get("total_modifications", 0) > 0:
            print(f"\n  结构修改历史:")
            print(f"    总修改轮次: {struct_summary.get('total_modifications', 0)}")
            print(f"    成功修改: {struct_summary.get('successful_modifications', 0)}")
            print(f"    被回滚的修改: {struct_summary.get('rolled_back_modifications', 0)}")
            print(f"    源码快照数: {struct_summary.get('snapshots_available', 0)}")
        else:
            print(f"\n  (未执行任何结构修改)")
        print(f"{'='*60}")

    @staticmethod
    def _setup_logging():
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        root = logging.getLogger("rec_self_evolve")
        root.setLevel(logging.INFO)
        root.addHandler(handler)