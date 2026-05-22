"""
RecSelfEvolveAgent — 推荐系统自增强 Agent 主循环

整合:
- Google 论文: 双循环架构 + 专业化 Persona + Think-Code-Verify
- Self-EvolveRec: 方向性反馈 + 诊断-模型协同进化
- **你的项目适配**: 通过 ProjectAdapter 理解具体运行方式
"""
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
from .prompts import MLE_ANALYSIS_PROMPT, STRUCTURE_OPTIMIZATION_PROMPT, ERROR_FEEDBACK_PROMPT, STRUCTURE_FIX_PROMPT, TRAIN_DIAGNOSIS_PROMPT, CODE_FIX_PROMPT, PREFLIGHT_FIX_PROMPT
from .llm_analyzer import LLMCaseAnalyzer
from .structure_applier import StructureApplier
from .iterative_memory import IterativeMemory
from .context_compressor import LLMContextCompressor

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

        self.current_iteration = 0
        self.current_strategy = "balanced"
        self.llm_health_ok = False
        self._last_surprise_report = None  # 上次惊喜评估报告
        self._last_case_analysis = None     # 上次案例分析结果
        self._last_wrong_text_cases = None  # 上次提取的错误文本案例
        self._last_structural_changes = None   # 上次结构修改提案
        # _structural_change_history 已迁移到 IterativeMemory — 不再使用 list
        self._max_self_correction_rounds = 5  # 自纠错最大轮数 (从3增加到5!)
        self._max_code_fix_rounds = 10        # 源码bug修复最大轮数 (核心新增!)
        self._consecutive_fail_count = 0      # 连续失败计数 (用于判断是否需要强制修复)

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
        print(f"  RecSelfEvolve — 推荐系统自增强 Agent (v0.5)")
        print(f"  核心改进: 运行→验证→修复→重试 自纠错闭环")
        print(f"  LLM: {self.config.llm_model} @ {self.config.llm_api_url}")
        print(f"  Project: {self.config.project_root}")
        print(f"  Backbone: {self.adapter.backbone}")
        print(f"  Dataset: {self.adapter.data_name}")
        print(f"  Max iterations: {max_iters}")
        print(f"  Max code fix rounds: {self._max_code_fix_rounds}")
        print(f"  Max self correction rounds: {self._max_self_correction_rounds}")
        print(f"{'='*60}\n")

        # ---- Step 0: 检查 LLM 健康状态 ----
        self._check_llm_health()

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

            # ── Phase 1: 训练 Baseline (使用新的 run-verify-fix-retry 闭环!) ──
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

            # --- Phase 3: 安全护栏 ---
            violations = self.safety.check_metrics(metrics)
            if violations:
                print(f"  ⚠ Safety: {violations}")
                self.journal.record({
                    "iteration": i, "status": "SAFETY_VIOLATION",
                    "metrics": metrics, "error": "; ".join(violations),
                })
                continue

            # --- Phase 4: LLM 分析 + 提案 ---
            proposal_result = self._phase_analyze_and_propose(metrics)
            if proposal_result is None:
                print(f"  ⚠ LLM analysis failed, skipping")
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
                    print(f"  ✗ All structural changes failed")
                    structural_changes = []

            # --- Phase 6: 应用参数变更 + 训练验证 (使用新的 run-verify-fix-retry!) ---
            has_any_change = bool(param_changes) or bool(structural_changes)
            if not has_any_change:
                print(f"  ⚠ No valid changes extracted")
                self.journal.record({
                    "iteration": i, "status": "NO_CHANGES",
                    "metrics": metrics, "proposal_result": proposal_result,
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
            
            # ── 构建当前源码上下文 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=["models.py", "modules.py", "trainers.py"],
                max_total_chars=7000,
            )
            
            # ── 构建错误详情 ──
            errors_detail = ""
            for err in errors:
                errors_detail += f"\n### 错误: {err.get('file', '?')}\n"
                errors_detail += f"- **错误类型**: {err.get('error_type', 'SyntaxError')}\n"
                errors_detail += f"- **行号**: {err.get('line', '?')}\n"
                errors_detail += f"- **错误消息**: {err.get('error_message', '')[:300]}\n"
                if err.get('text'):
                    errors_detail += f"- **出错行代码**: {err.get('text', '')[:100]}\n"
            
            # ── 构建 PREFLIGHT_FIX_PROMPT ──
            prompt = PREFLIGHT_FIX_PROMPT.format(
                _preflight_errors=errors_detail,
                _current_source_code=source_code_ctx,
            )
            
            # ── 调用 LLM ──
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
            )
            
            if response is None:
                print(f"     ✗ LLM 调用失败")
                continue
            
            # ── 解析修复方案 ──
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
                    continue
            
            elif fix_result["status"] == "ROLLBACK":
                print(f"     ↩ 语法修复被回滚: {fix_result.get('rollback_reason', '')[:80]}")
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
        print(f"\n  ╔══════════════════════════════════════╗")
        print(f"  ║  🔁 进入自纠错内循环                  ║")
        print(f"  ║  训练失败 → 提取错误 → 修复 → 重试    ║")
        print(f"  ║  最多 {max_rounds} 轮自纠错              ║")
        print(f"  ╚══════════════════════════════════════╝")
        
        error_category = train_result.get("error_category", "UNKNOWN")
        fixable = train_result.get("fixable", True)
        
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
                print(f"    {snippet[:200]}")
        
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
        
        Returns:
            {
                "action": "code_fixed_and_retrained" | "code_fixed_but_retrain_failed" | "code_fix_failed" | "skip",
                ...其他字段同 train_result 格式
            }
        """
        # ── 构建源码上下文 ──
        source_code_ctx = self.adapter.build_source_code_context(
            include_files=["models.py", "modules.py", "trainers.py"],
            max_total_chars=7000,
            iterative_memory=self.iter_memory,
        )
        
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
        
        source_code_ctx = self.adapter.build_source_code_context(
            include_files=["models.py", "modules.py", "trainers.py"],
            max_total_chars=7000,
            iterative_memory=self.iter_memory,
        )
        
        tb_details = train_error.get("traceback_details", {})
        error_message = tb_details.get("error_message", train_error.get("error", ""))
        traceback_text = tb_details.get("traceback_text", "")
        
        # 构建 prompt — 明确告知 LLM 输出完整的 class 定义
        prompt = CODE_FIX_PROMPT.format(
            _error_type=tb_details.get("error_type", "CODE_ERROR"),
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
        
        # 明确告知 LLM: 用 replace_class, 输出完整的类定义
        prompt += (
            "\n\n### ⚠ 重要修改: 请使用 replace_class 策略!"
            "\n之前的 replace_function 策略连续失败, 因为代码定位困难。"
            "\n这次请**输出完整的类定义** (包含类头和所有方法), 使用 insert_position='replace_class'。"
            "\n不要只输出单个方法, 要输出整个类的完整代码。"
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
        """
        print(f"  📝 策略切换: 整文件重写")
        
        source_code_ctx = self.adapter.build_source_code_context(
            include_files=["models.py", "modules.py", "trainers.py"],
            max_total_chars=7000,
            iterative_memory=self.iter_memory,
        )
        
        tb_details = train_error.get("traceback_details", {})
        error_message = tb_details.get("error_message", train_error.get("error", ""))
        traceback_text = tb_details.get("traceback_text", "")
        
        # 从 traceback 中确定出错的是哪个文件
        error_files = tb_details.get("files", [])
        target_file = "modules.py"  # 默认
        if error_files:
            for ef in error_files:
                if "modules" in ef:
                    target_file = "modules.py"
                    break
                elif "models" in ef:
                    target_file = "models.py"
                    break
                elif "trainers" in ef:
                    target_file = "trainers.py"
                    break
        
        # 读取当前文件内容
        file_path = self.struct_applier._resolve_file_path(target_file)
        if not file_path:
            return {"action": "code_fix_failed", "error": f"Cannot find {target_file}"}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            current_file_content = f.read()
        
        prompt = (
            f"训练运行失败, 错误来自**{target_file}**中的代码 bug。\n"
            f"所有之前的修复策略 (replace_function, replace_class) 都连续失败。\n"
            f"现在需要你**输出 {target_file} 的完整修复后的代码**。\n\n"
            f"## 错误信息\n```\n{error_message[:1500]}\n```\n\n"
            f"## Traceback\n```\n{traceback_text[:2000]}\n```\n\n"
            f"## 当前 {target_file} 的内容\n```python\n{current_file_content}\n```\n\n"
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
                include_files=["models.py", "modules.py"],
                max_total_chars=5000,
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
            include_files=["models.py", "modules.py"],
            max_total_chars=6500,
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

        # 构建案例分析信息 (如果有)
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
        """验证参数变更 — soft_limit 模式: 超出范围仅警告不拒绝"""
        valid_changes = {}
        for key, value in param_changes.items():
            if key in self.adapter.TUNABLE_PARAMS:
                param_info = self.adapter.TUNABLE_PARAMS[key]
                soft_limit = param_info.get("soft_limit", False)
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

    def _validate_structural_changes(self, structural_changes: list) -> list:
        """验证结构修改列表 — 开放模式: 允许任意文件和类型"""
        valid_changes = []
        for change in structural_changes:
            # target_file: 允许任何项目中存在的 .py 文件
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
            
            # 必须有 new_code
            new_code = change.get("new_code", "")
            if not new_code:
                logger.warning(f"Structural change missing new_code: {change.get('description', '?')}")
                continue
            
            # 清理 new_code (移除 markdown 标记等)
            new_code = StructureApplier.clean_new_code(new_code)
            change["new_code"] = new_code
            
            # action_type: 不强制要求，如果缺失则根据内容自动推断或用 "modify"
            if not change.get("action_type"):
                # 简单推断: 有 class 定义 → add_module，否则 → modify
                if "class " in new_code and "def " not in change.get("target_class_or_function", ""):
                    change["action_type"] = "add_module"
                else:
                    change["action_type"] = "modify"
            
            # insert_position: 不强制要求，如果缺失则智能推断
            if not change.get("insert_position"):
                if change.get("target_class_or_function") and "." in change.get("target_class_or_function", ""):
                    change["insert_position"] = "replace_function"
                elif "class " in new_code:
                    change["insert_position"] = "append_to_file"
                else:
                    change["insert_position"] = "replace_function"
            
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
                structural.append({
                    "action_type": s.get("action_type", "add_module"),
                    "target_file": "modules.py",  # 默认
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
            r"(?:loss_type|neg_sampler|CL_type|hidden_act|backbone)\s*[=:]\s*['\"]?(\w+)['\"]?",
        ]

        # 提取键值对
        kv_pattern = r"['\"]?(\w+)['\"]?\s*[=:]\s*['\"]?([\d.e+\-\w]+)['\"]?"
        matches = re.findall(kv_pattern, text)
        for k, v in matches:
            if k in ["lr", "loss_type", "neg_sampler", "hidden_size", "batch_size",
                     "epochs", "N", "M", "K", "backbone", "CL_type", "hidden_act",
                     "hidden_dropout_prob", "num_hidden_layers", "weight_decay",
                     "dropout", "seed", "start_epoch", "gpu_id", "max_seq_length",
                     "num_attention_heads", "d_state", "d_conv", "expand"]:
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
        print(f"     错误摘要: {train_error.get('error', '')[:100]}")
        
        for round_idx in range(1, max_r + 1):
            print(f"\n  🔁 自纠错第 {round_idx}/{max_r} 轮")
            
            # ── 1. 构建当前源码上下文 ──
            source_code_ctx = self.adapter.build_source_code_context(
                include_files=["models.py", "modules.py", "trainers.py"],
                max_total_chars=6500,
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
            # 判断错误类型
            error_msg = train_error.get("error", "")
            if "CUDA out of memory" in error_msg or "OOM" in error_msg:
                error_type = "OOM (显存溢出)"
            elif "NaN" in error_msg or "nan" in error_msg:
                error_type = "NaN Loss (训练发散)"
            elif "SyntaxError" in error_msg or "IndentationError" in error_msg:
                error_type = "Syntax Error (语法错误)"
            elif "RuntimeError" in error_msg:
                error_type = "Runtime Error (运行时错误)"
            elif "FileNotFoundError" in error_msg:
                error_type = "File Not Found (文件缺失)"
            elif "ImportError" in error_msg or "ModuleNotFoundError" in error_msg:
                error_type = "Import Error (依赖缺失)"
            elif "DimensionMismatch" in error_msg or "size mismatch" in error_msg.lower() or "dimension" in error_msg.lower():
                error_type = "Dimension Mismatch (维度不匹配)"
            else:
                error_type = train_error.get("status", "UNKNOWN")
            
            prompt = ERROR_FEEDBACK_PROMPT.format(
                _original_proposal=json.dumps(original_proposal, indent=2, ensure_ascii=False),
                _error_type=error_type,
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
                    "error_type": error_type,
                    "revised_metrics": revised_metrics,
                    "param_changes": revised_param_changes,
                    "structural_changes": revised_structural_changes,
                    "explanation": f"自纠错第{round_idx}轮修正成功 (原错误: {error_type})",
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
                print(f"     ✗ 修正后仍然失败: {revised_train.get('error', '')[:100]}")
                
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
            "original_error_type": error_type,
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
                include_files=["models.py", "modules.py", "trainers.py"],
                max_total_chars=6500,
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