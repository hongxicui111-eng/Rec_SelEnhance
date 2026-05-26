"""
容错训练执行器 — 适配用户的推荐项目命令格式

核心改进 (v0.5):
- 错误分类系统: CODE_ERROR / CONFIG_ERROR / DATA_ERROR / SYSTEM_ERROR
- 从终端输出提取完整 traceback, 包含文件名+行号
- 对代码错误(语法/运行时/导入)不再简单跳过, 而是标记为 fixable=True
- 返回 error_category 和 traceback_details 供自纠错循环使用
"""
import subprocess
import re
import json
import time
import os
import logging
from typing import Optional

from .project_adapter import SeqRecAdapter

logger = logging.getLogger("rec_self_evolve.train_runner")


class FaultTolerantTrainRunner:
    """
    容错训练执行器 (v0.5 — 增强错误感知)
    通过 ProjectAdapter 构建命令，处理 OOM、NaN、Import 缺失等异常。
    
    关键改进:
    1. 训练失败后, 立即从终端提取完整错误信息 (traceback + 文件名 + 行号)
    2. 分类错误类型: CODE_ERROR(需修源码) / CONFIG_ERROR(需调参数) / DATA_ERROR(需修路径) / SYSTEM_ERROR(无法自动修)
    3. 标记 fixable=True/False — 告知上层是否可以通过自纠错修复
    4. 提供 traceback_details — 包含出错的文件路径和行号, 供 LLM 精准定位
    """

    def __init__(self, adapter: SeqRecAdapter,
                 timeout: int = 7200,
                 oom_reduce_factor: float = 0.5,
                 nan_reduce_factor: float = 0.5):
        self.adapter = adapter
        self.project_root = adapter.project_root
        self.timeout = timeout
        self.oom_reduce_factor = oom_reduce_factor
        self.nan_reduce_factor = nan_reduce_factor

    # ════════════════════════════════════════
    # 运行训练 (核心入口)
    # ════════════════════════════════════════

    def run(self, param_overrides: Optional[dict] = None,
            eval_only: bool = False) -> dict:
        """
        执行训练，带完整容错逻辑 + 错误分类

        返回:
        {
            "status": "SUCCESS" | "FAILED" | "TIMEOUT" | ...
            "metrics": {...},
            "log": str,
            "error": str,
            "error_category": "CODE_ERROR" | "CONFIG_ERROR" | "DATA_ERROR" | "SYSTEM_ERROR" | None,
            "fixable": bool,                    # 是否可通过自纠错修复
            "traceback_details": {...},          # traceback 详情 (文件路径+行号+错误行)
            "applied_overrides": {...},
            "action": str,
            "returncode": int,                   # 新增: 进程退出码
        }
        """
        overrides = param_overrides or {}
        working_overrides = dict(overrides)

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            logger.info(f"Train attempt {attempt}/{max_attempts}")

            # 通过 Adapter 构建命令
            cmd = self.adapter.build_train_command(
                param_overrides=working_overrides,
                eval_only=eval_only,
            )
            logger.info(f"Running:\n{cmd}")

            # ── 训练命令前置校验（fail-fast）──
            cmd_check = self.adapter.validate_train_command(cmd)
            if not cmd_check.get("ok", False):
                issue_text = " | ".join(cmd_check.get("issues", []))
                warn_text = " | ".join(cmd_check.get("warnings", []))
                if warn_text:
                    logger.warning(f"Train command warnings: {warn_text}")
                logger.error(f"Train command validation failed: {issue_text}")
                return {
                    "status": "FAILED",
                    "error": f"Train command validation failed: {issue_text}",
                    "error_category": "CONFIG_ERROR",
                    "fixable": True,
                    "traceback_details": {},
                    "log": "",
                    "applied_overrides": working_overrides,
                    "action": "fix_and_retry",
                    "returncode": None,
                    "diagnostics": {
                        "command": cmd,
                        "issues": cmd_check.get("issues", []),
                        "warnings": cmd_check.get("warnings", []),
                    },
                }

            if cmd_check.get("warnings"):
                logger.warning(f"Train command warnings: {' | '.join(cmd_check.get('warnings', []))}")

            try:
                # ── 使用 Popen 实时流式输出, 不再静默! ──
                # 这样用户能看到每个 epoch 的进度和指标
                # 同时捕获进程退出码用于判断训练是否成功
                combined, returncode = self._run_with_streaming(cmd, self.timeout)

                # ---------- 进程退出码检查 (核心新增!) ----------
                # 如果进程正常退出 (returncode==0) 且没有 Traceback → 很可能训练成功
                # 即使指标模式没匹配, 也应该尝试从完整输出中解析指标
                has_traceback = "Traceback (most recent call last):" in combined

                # ── 先从整个输出搜索指标 (不再只看最后500字符!) ──
                final_metrics_pattern = re.search(
                    r"(HIT@\d+|NDCG@\d+|Epoch.*HIT|HR_\d+|NDCG_\d+|Early stopping)", combined[-2000:]
                )
                # 如果最后2000字符没有, 再搜索整个输出
                if not final_metrics_pattern:
                    final_metrics_pattern = re.search(
                        r"(HIT@\d+|NDCG@\d+|HR_\d+|NDCG_\d+|'Epoch'.*'\d+)", combined
                    )

                # ── 判断训练是否成功的逻辑 (新增: 退出码 + 没有traceback + 有指标模式) ──
                likely_success = (returncode == 0 and not has_traceback)

                if likely_success or (not has_traceback and final_metrics_pattern):
                    # 尝试解析指标 (从整个输出解析, 不再只从最后部分)
                    metrics = self.adapter.parse_metrics_from_log(combined)

                    # ── 关键新增: 检测 NaN/Inf loss — 训练健康状态检查 ──
                    training_health = metrics.get("training_health", {})
                    health_status = training_health.get("status", "healthy")
                    
                    if health_status in ("fully_diverged", "partially_diverged"):
                        # 训练 loss 出现 NaN/Inf → 即使进程没崩溃, 模型也是废的!
                        nan_count = training_health.get("nan_loss_count", 0)
                        inf_count = training_health.get("inf_loss_count", 0)
                        logger.warning(f"NaN/Inf loss detected during training! NaN={nan_count}, Inf={inf_count}")
                        print(f"  ✗ Training loss diverged: {nan_count} NaN epochs, {inf_count} Inf epochs")
                        print(f"     health_status={health_status}")
                        # 返回失败, 让上层 LLM 判断是 CONFIG_ERROR (需要调参) 还是 CODE_ERROR (模型有bug)
                        return {
                            "status": "FAILED",
                            "error": f"Training loss diverged (NaN/Inf): {nan_count} NaN epochs, {inf_count} Inf epochs. "
                                     f"Training health: {health_status}. "
                                     f"Last few loss values: {training_health.get('loss_values', [])}",
                            "error_category": "UNKNOWN",  # 让 LLM 判断是代码bug还是参数问题
                            "fixable": True,
                            "traceback_details": {},
                            "log": combined[-3000:],
                            "metrics": metrics,  # 保留 metrics (含 training_health), 供上层参考
                            "applied_overrides": working_overrides,
                            "action": "fix_and_retry",
                            "returncode": returncode,
                        }

                    if metrics:
                        logger.info(f"Training OK. Metrics: {metrics}")
                        print(f"  ✓ Training converged. Metrics: {self.adapter.format_metrics_for_llm(metrics)}")
                        return {
                            "status": "SUCCESS",
                            "metrics": metrics,
                            "log": combined[-3000:],
                            "error_category": None,
                            "fixable": False,
                            "traceback_details": {},
                            "applied_overrides": working_overrides,
                            "action": "proceed",
                            "returncode": returncode,
                        }

                    # ── 指标无法解析, 但可能是训练成功 (格式不匹配) ──
                    has_training_keywords = any(
                        kw in combined for kw in ["Epoch", "epoch", "loss", "Loss", "train", "valid", "test"]
                    )
                    
                    # ── 即使 metrics 为空, 也要检查 NaN loss ──
                    # 因为 parse_metrics_from_log 可能因为格式问题没有返回 metrics dict
                    # 但 NaN loss 信息可以从原始输出直接扫描
                    if not training_health or training_health.get("status") == "healthy":
                        # training_health 没有或不完整 → 从原始输出直接扫描 NaN
                        direct_nan_count = len(re.findall(r"'loss':\s*'nan'", combined, re.IGNORECASE))
                        direct_inf_count = len(re.findall(r"'loss':\s*'inf'", combined, re.IGNORECASE))
                        if direct_nan_count > 0 or direct_inf_count > 0:
                            print(f"  ✗ NaN/Inf loss detected in raw output: {direct_nan_count} NaN, {direct_inf_count} Inf")
                            return {
                                "status": "FAILED",
                                "error": f"Training loss diverged: {direct_nan_count} NaN epochs, {direct_inf_count} Inf epochs detected in raw output",
                                "error_category": "UNKNOWN",
                                "fixable": True,
                                "traceback_details": {},
                                "log": combined[-3000:],
                                "metrics": metrics,
                                "applied_overrides": working_overrides,
                                "action": "fix_and_retry",
                                "returncode": returncode,
                            }
                    
                    if has_training_keywords and likely_success:
                        # 训练看起来是成功的, 只是指标格式不匹配
                        logger.warning("Training likely succeeded but metrics format not recognized")
                        print(f"  ⚠ Training appears successful (exit code 0, no traceback) but metrics format unrecognized")
                        print(f"     Output length: {len(combined)} chars, last 500: {combined[-500:][:200]}")
                        return {
                            "status": "SUCCESS",
                            "metrics": {},  # 无法解析指标, 但训练成功了
                            "log": combined[-3000:],
                            "raw_output": combined,
                            "error_category": None,
                            "fixable": False,
                            "traceback_details": {},
                            "applied_overrides": working_overrides,
                            "action": "proceed",
                            "returncode": returncode,
                            "metrics_parse_warning": "Metrics format not recognized in output",
                        }

                    # ── 修复 fall-through bug: 指标解析失败且无训练关键词 ──
                    # likely_success=True (returncode=0, 无 traceback) 但无法解析指标
                    # 且输出中也没有训练关键词 → 这说明训练脚本可能根本没运行!
                    # 不应继续 fall-through 到空输出检查或错误分类,
                    # 而应在这里明确判断为"训练未执行"并返回可修复错误
                    if likely_success and not combined.strip():
                        # 输出完全为空 + returncode=0 → 训练脚本没运行 (最可能是命令格式错误)
                        diagnostics = self._diagnose_empty_output(cmd, returncode)
                        error_msg = (
                            f"Training process exited normally (code 0) but produced NO output.\n"
                            f"This almost certainly means the training script never executed.\n"
                            f"{diagnostics}"
                        )
                        logger.error(error_msg)
                        print(f"  ✗ Training script appears not to have run (exit code 0, empty output)")
                        print(f"     {diagnostics}")
                        return {
                            "status": "FAILED",
                            "error": error_msg,
                            "error_category": "CONFIG_ERROR",  # 命令格式/路径问题 → 可修复
                            "fixable": True,  # 标记为可修复, 让自纠错循环可以处理
                            "traceback_details": {},
                            "log": "",
                            "applied_overrides": working_overrides,
                            "action": "fix_and_retry",  # 改为 fix_and_retry, 不再 skip
                            "returncode": returncode,
                            "diagnostics": diagnostics,
                        }

                    elif likely_success and combined.strip() and not has_training_keywords:
                        # 有输出, returncode=0, 无 traceback, 但没有训练关键词
                        # → 可能是脚本执行了但不是训练 (比如只打印了帮助信息)
                        logger.warning(f"Process exit code 0, output exists but no training keywords found")
                        print(f"  ⚠ Process exited normally but output doesn't look like training logs")
                        print(f"     Output length: {len(combined)} chars, first 300: {combined[:300]}")
                        return {
                            "status": "FAILED",
                            "error": f"Process exited normally (code 0) but output doesn't contain training keywords. Output: {combined[:1000]}",
                            "error_category": "CONFIG_ERROR",
                            "fixable": True,
                            "traceback_details": {},
                            "log": combined[-3000:],
                            "applied_overrides": working_overrides,
                            "action": "fix_and_retry",
                            "returncode": returncode,
                        }

                # ── 空输出检查 ──
                # 如果输出完全为空 (且 likely_success 分支没处理 → 说明有 traceback 或 returncode!=0)
                if not combined.strip():
                    diagnostics = self._diagnose_empty_output(cmd, returncode)
                    error_msg = (
                        f"Training process produced NO output (exit code: {returncode}).\n"
                        f"{diagnostics}"
                    )
                    logger.error(error_msg)
                    # 判断错误类别: 如果 returncode!=0 → 有错误码可参考
                    # 如果 returncode==0 但有 traceback标记 → 理论上不可能 (前面已过滤)
                    error_category = "CONFIG_ERROR" if returncode == 0 else "SYSTEM_ERROR"
                    fixable = True  # 空输出问题几乎都是命令格式/路径/环境问题 → 都可修复
                    return {
                        "status": "FAILED",
                        "error": error_msg,
                        "error_category": error_category,
                        "fixable": fixable,
                        "traceback_details": {},
                        "log": "",
                        "applied_overrides": working_overrides,
                        "action": "fix_and_retry",  # 改为 fix_and_retry, 不再 skip
                        "returncode": returncode,
                        "diagnostics": diagnostics,
                    }

                # ---------- 错误信息提取 (不做分类, 由上层 LLM 判断) ----------
                error_msg = combined[-3000:]

                # ── 提取完整 traceback 详情 ──
                traceback_details = self._extract_traceback_details(combined)

                # ── 缺失外部依赖 (SYSTEM_ERROR — 可尝试安装) ──
                # 这是唯一保留的自动恢复逻辑: pip install 缺失包是确定性操作, 不需要 LLM 判断
                missing_pkg = self._extract_missing_package(error_msg)
                if missing_pkg:
                    logger.info(f"Missing package: {missing_pkg}, auto-installing...")
                    install = subprocess.run(
                        f"pip install {missing_pkg}",
                        shell=True, capture_output=True, text=True, timeout=120
                    )
                    logger.info(f"Install result: {install.stdout[-200:]}")
                    continue  # 重试

                # ── 所有其他错误: 返回完整错误信息给上层, 由 LLM 判断分类和修复策略 ──
                # 不再做硬编码的 OOM/NaN 自动恢复, 也不做 _classify_error 分类
                # 上层 core.py 会调用 LLM 来判断 error_category
                
                fixable = True  # 默认可修复, 上层 LLM 可能推翻此判断
                if returncode in (134, 137, 139):
                    # abort/kill/segfault → 系统层异常, LLM 也难以修复
                    fixable = False
                
                # 如果有输出但没有 traceback, 且退出码为0 → 可能是指标解析问题
                if returncode == 0 and not has_traceback:
                    fixable = True
                
                return {
                    "status": "FAILED",
                    "error": error_msg[:1500],
                    "error_category": "UNKNOWN",  # 由上层 LLM 判断, 不再硬编码分类
                    "fixable": fixable,
                    "traceback_details": traceback_details,
                    "log": combined[-3000:],
                    "applied_overrides": working_overrides,
                    "action": "fix_and_retry" if fixable else "skip_iteration",
                    "returncode": returncode,
                }

            except subprocess.TimeoutExpired:
                logger.error(f"Training timeout after {self.timeout}s")
                return {
                    "status": "TIMEOUT",
                    "error_category": "SYSTEM_ERROR",
                    "fixable": False,
                    "traceback_details": {},
                    "log": f"Training exceeded {self.timeout}s timeout",
                    "action": "skip_iteration",
                }

            except Exception as e:
                logger.error(f"Unexpected training error: {e}")
                if attempt == max_attempts:
                    return {
                        "status": "CRASHED",
                        "error": str(e),
                        "error_category": "SYSTEM_ERROR",
                        "fixable": False,
                        "traceback_details": {},
                        "action": "skip_iteration",
                    }
                time.sleep(2 ** attempt)

        return {
            "status": "FAILED_AFTER_RETRY",
            "error": f"Failed after {max_attempts} attempts (missing package auto-install also failed)",
            "error_category": "UNKNOWN",  # 由上层 LLM 判断
            "fixable": True,
            "traceback_details": {},
            "action": "fix_and_retry",
        }

    # ════════════════════════════════════════
    # 实时流式输出训练进度 (核心改进!)
    # ════════════════════════════════════════

    def _run_with_streaming(self, cmd: str, timeout: int) -> tuple:
        """
        使用 Popen 运行训练命令, 实时流式输出到终端
        
        核心改进: 不再 capture_output=True 静默运行!
        用户可以看到每个 epoch 的进度、tqdm、loss、指标等
        
        同时保留完整的输出供后续 metrics 解析和错误提取
        
        ⚠ 新增: 返回 (combined_output, returncode) tuple
           returncode 是进程退出码, 对于判断训练是否成功至关重要:
           - 0: 正常退出 (即使指标格式不匹配, 训练本身是成功的)
           - 1: 一般错误
           - 其他: 特定错误
        
        Args:
            cmd: shell 命令字符串
            timeout: 超时时间 (秒)
            
        Returns:
            tuple: (combined_output_str, returncode_int)
        """
        env = self._build_env()
        process = subprocess.Popen(
            cmd, shell=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,  # 合并 stderr 到 stdout
            text=True, bufsize=1,  # 行缓冲
            env=env,
        )
        
        combined_output = []
        last_print_time = time.time()
        
        try:
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    combined_output.append(line)
                    # 实时打印训练输出到终端
                    # 每行都打印, 但过滤掉空的 tqdm 行以减少噪音
                    stripped = line.strip()
                    if stripped:
                        # 打印关键信息: epoch loss, validation metrics, early stopping, etc.
                        print(f"    {stripped}")
                    last_print_time = time.time()
                else:
                    # 没有输出时, 检查是否超时
                    if time.time() - last_print_time > timeout:
                        process.kill()
                        logger.error(f"Training timeout: no output for {timeout}s")
                        break
                    time.sleep(0.1)  # 短暂等待避免空转
            
            process.wait(timeout=10)  # 等待进程结束
            
        except subprocess.TimeoutExpired:
            process.kill()
            logger.error(f"Training process killed after timeout")
        
        combined = "".join(combined_output)
        returncode = process.returncode
        
        # ── 诊断信息: 如果输出非常短或为空, 打印警告 ──
        if len(combined) < 50:
            logger.warning(f"Training produced very short/empty output ({len(combined)} chars, exit code {returncode})")
            logger.warning(f"Command: {cmd[:200]}")
            if combined:
                logger.warning(f"Output: {combined[:200]}")
            else:
                logger.warning("Output: (empty — training script likely never executed)")
                # 立即诊断: 命令格式是否正确
                if "cd " in cmd and "&&" not in cmd.split("python")[0] if "python" in cmd else True:
                    logger.warning(
                        "⚠ Likely cause: cd command has no '&&' separator, "
                        "so all subsequent arguments became cd's arguments and were ignored"
                    )
        
        return (combined, returncode)

    # ════════════════════════════════════════
    # 空输出诊断 (核心新增!)
    # ════════════════════════════════════════

    def _diagnose_empty_output(self, cmd: str, returncode: int) -> str:
        """
        当训练输出为空时, 执行诊断检查以确定根本原因
        
        常见原因:
        1. cd 命令吞掉了后续所有参数 (缺少 && 连接符)
        2. 训练脚本文件不存在
        3. 项目目录路径不存在
        4. python 命令不在 PATH 中
        5. 环境变量配置问题
        
        Returns:
            str: 诊断信息, 供错误消息和日志使用
        """
        diagnostics = []
        diagnostics.append(f"Exit code: {returncode}")
        
        # ── 检查 1: 命令格式是否缺少 && ──
        # 如果 cd 命令后面没有 &&, 则后续的 CUDA_VISIBLE_DEVICES 和 python3 命令
        # 都会变成 cd 的额外参数, 导致 cd 成功 (exit 0) 但训练脚本根本不执行
        if "cd " in cmd and "&&" not in cmd.split("python")[0]:
            diagnostics.append(
                "⚠ CRITICAL: cd command has no '&&' separator! "
                "All subsequent arguments (CUDA_VISIBLE_DEVICES, python3, script, args) "
                "become extra arguments to cd and are IGNORED. "
                "This means the training script NEVER executed. "
                "Fix: add '&&' after the cd command."
            )
        
        # ── 检查 2: 项目目录是否存在 ──
        cd_match = re.search(r'cd\s+(\S+)', cmd)
        if cd_match:
            cd_path = cd_match.group(1)
            if os.path.exists(cd_path):
                diagnostics.append(f"✓ Directory exists: {cd_path}")
            else:
                diagnostics.append(f"✗ Directory NOT found: {cd_path}")
        
        # ── 检查 3: 训练脚本文件是否存在 ──
        script_match = re.search(r'python3\s+-u\s+(\S+)', cmd)
        if script_match:
            script_name = script_match.group(1)
            script_path = os.path.join(self.project_root, script_name)
            if os.path.exists(script_path):
                diagnostics.append(f"✓ Script exists: {script_path}")
            else:
                diagnostics.append(f"✗ Script NOT found: {script_path}")
                # 尝试其他常见路径
                alt_paths = [
                    os.path.join(self.project_root, "Recmodel", script_name),
                    os.path.join(os.getcwd(), script_name),
                ]
                for alt in alt_paths:
                    if os.path.exists(alt):
                        diagnostics.append(f"  → Found at alternative path: {alt}")
                        break
        
        # ── 检查 4: 命令是否用了 \ 续行符 (可能导致 cd 吞参数) ──
        if "\\n" in cmd or cmd.strip().endswith("\\"):
            diagnostics.append(
                "⚠ Command uses '\\' line continuation. "
                "When passed to shell=True, backslash-newline joins all lines into ONE command, "
                "making everything after 'cd' additional arguments to cd. "
                "Consider using '&&' to properly chain commands."
            )
        
        # ── 检查 5: python3 是否可用 ──
        try:
            py_check = subprocess.run(
                "which python3", shell=True, capture_output=True, text=True, timeout=5
            )
            if py_check.returncode == 0:
                diagnostics.append(f"✓ python3 available: {py_check.stdout.strip()}")
            else:
                diagnostics.append("✗ python3 NOT found in PATH")
        except Exception:
            diagnostics.append("? Could not check python3 availability")
        
        return "\n".join(diagnostics)

    # ════════════════════════════════════════
    # 错误分类系统 (核心新增!)
    # ════════════════════════════════════════

    def _classify_error(self, error_msg: str, traceback_details: dict) -> str:
        """
        不再硬编码分类错误 — 返回 "UNKNOWN" 让上层 LLM 判断
        
        错误分类 (CODE_ERROR / CONFIG_ERROR / DATA_ERROR / SYSTEM_ERROR)
        现在完全由 core.py 中调用 LLM 来决定，而不是字符串匹配。
        
        train_runner 只负责提取原始错误信息 (traceback, error_msg 等)，
        分类决策交给有推理能力的 LLM。
        """
        return "UNKNOWN"

    def _is_fixable(self, error_category: str, traceback_details: dict, returncode: Optional[int] = None) -> bool:
        """
        判断错误是否可以通过自纠错修复 — 简化版
        
        现在不再依赖硬编码的 error_category 分类，
        只做最基本的判断: returncode 为 134/137/139 → 不可修复
        其他情况默认 fixable=True, 让上层 LLM 来做最终决策
        """
        if returncode in (134, 137, 139):
            return False
        return True

    def _extract_traceback_details(self, combined_output: str) -> dict:
        """
        从终端输出提取完整的 traceback 详情
        
        返回:
        {
            "traceback_text": str,           # 完整 traceback 文本
            "error_line": str,               # 出错的那一行代码
            "error_type": str,               # 错误类型 (如 "RuntimeError")
            "error_message": str,            # 错误消息
            "files": [str],                  # traceback 中涉及的文件路径列表
            "line_numbers": [int],           # 出错的行号列表 (对应 files)
            "offending_code_snippets": [str], # 出错行的代码内容
        }
        """
        details = {
            "traceback_text": "",
            "error_line": "",
            "error_type": "",
            "error_message": "",
            "files": [],
            "line_numbers": [],
            "offending_code_snippets": [],
        }
        
        # ── 提取 traceback 块 ──
        # Python traceback 格式: "Traceback (most recent call last):" 开头
        tb_start = combined_output.find("Traceback (most recent call last):")
        if tb_start >= 0:
            # traceback 从 "Traceback..." 到错误类型的最后一行
            tb_text = combined_output[tb_start:]
            # 截取到合理长度
            tb_text = tb_text[:3000]
            details["traceback_text"] = tb_text
            
            # ── 提取最后一行 (真正的错误) ──
            # 格式: "RuntimeError: size mismatch ..."
            last_error_line = ""
            for line in tb_text.split("\n"):
                line = line.strip()
                if line and not line.startswith("File ") and not line.startswith("Traceback"):
                    if ":" in line and not line.startswith("During handling"):
                        last_error_line = line
            details["error_line"] = last_error_line
            
            # ── 提取错误类型 ──
            # 格式: "RuntimeError", "SyntaxError", "NameError" 等
            error_type_match = re.match(r'^(\w+Error|\w+Exception|AssertionError|TypeError|AttributeError|ValueError|KeyError|IndexError|RuntimeError|SyntaxError|IndentationError|NameError|ImportError|ModuleNotFoundError|FileNotFoundError|UnboundLocalError|StopIteration|OverflowError|ZeroDivisionError|FloatingPointError):', last_error_line)
            if error_type_match:
                details["error_type"] = error_type_match.group(1)
                details["error_message"] = last_error_line[len(error_type_match.group(1))+1:].strip()
            elif last_error_line:
                details["error_type"] = "Unknown"
                details["error_message"] = last_error_line
            
            # ── 提取文件路径和行号 ──
            # 格式: "File "/path/to/file.py", line 42, in <module>"
            file_line_pattern = r'File "([^"]+)", line (\d+), in (\S+)'
            for match in re.finditer(file_line_pattern, tb_text):
                file_path = match.group(1)
                line_num = int(match.group(2))
                func_name = match.group(3)
                details["files"].append(file_path)
                details["line_numbers"].append(line_num)
                
                # ── 尝试读取出错行的代码 ──
                code_snippet = self._read_error_line_from_file(file_path, line_num)
                if code_snippet:
                    details["offending_code_snippets"].append(code_snippet)
        else:
            # 没有 traceback → 可能是其他类型的错误输出
            # 尝试从 stderr 中提取任何错误信息
            if combined_output:
                # 取最后 1000 字符作为错误摘要
                error_lines = combined_output[-1000:].strip().split("\n")
                for line in reversed(error_lines):
                    line = line.strip()
                    if line and ("Error" in line or "error" in line or "Exception" in line):
                        details["error_line"] = line
                        details["traceback_text"] = combined_output[-2000:]
                        break
        
        return details

    def _read_error_line_from_file(self, file_path: str, line_num: int) -> Optional[str]:
        """
        读取出错文件中出错行附近的代码 (行号 ± 3 行, 共 7 行上下文)
        """
        try:
            if not os.path.exists(file_path):
                return None
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = f.readlines()
            # 提取 line_num-3 到 line_num+3
            start = max(0, line_num - 4)  # -3 行上下文
            end = min(len(lines), line_num + 3)  # +3 行上下文
            snippet_lines = lines[start:end]
            # 标注出错行
            result_lines = []
            for i, line in enumerate(snippet_lines):
                actual_line_num = start + i + 1
                marker = ">>> " if actual_line_num == line_num else "    "
                result_lines.append(f"{marker}{actual_line_num}: {line.rstrip()}")
            return "\n".join(result_lines)
        except Exception:
            return None

    # ════════════════════════════════════════
    # 快速评估 (仅 inference, 不训练)
    # ════════════════════════════════════════

    def evaluate(self, param_overrides: Optional[dict] = None) -> dict:
        """只做评估（使用已有的 checkpoint）"""
        return self.run(param_overrides=param_overrides, eval_only=True)

    # ════════════════════════════════════════
    # 预检: 在训练前先检查源码是否可导入
    # ════════════════════════════════════════

    def preflight_check(self, source_files: list = None) -> dict:
        """
        预检: 在训练前先检查源码文件是否语法正确、能否导入
        
        Args:
            source_files: 需要检查的源码文件列表 (默认: Recmodel 目录下所有 .py 文件)
        
        Returns:
            {
                "status": "PASS" | "FAIL",
                "errors": [dict],  # 检查发现的问题列表
                "files_checked": [str],  # 检查的文件列表
            }
        """
        if source_files is None:
            source_files = [
                "models.py", "modules.py", "trainers.py", "datasets.py",
                "utils.py", "error_case_extractor.py", "surprise_eval.py",
                "run_finetune_full.py",
            ]
        
        errors = []
        files_checked = []
        
        for fname in source_files:
            fpath = os.path.join(self.project_root, fname)
            if not os.path.exists(fpath):
                # 也尝试 Recmodel/ 子目录
                fpath = os.path.join(self.project_root, "Recmodel", fname)
                if not os.path.exists(fpath):
                    continue
            
            files_checked.append(fpath)
            
            # ── 语法检查 ──
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    code = f.read()
                import ast
                ast.parse(code)
            except SyntaxError as e:
                errors.append({
                    "file": fpath,
                    "error_type": "SyntaxError",
                    "error_message": str(e),
                    "line": e.lineno,
                    "text": e.text,
                    "fixable": True,
                })
                continue
            
            # ── 导入检查 ──
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(fname.replace('.py', ''), fpath)
                # 不实际执行模块 (避免 GPU 依赖等), 只检查能否 spec 解析
                if spec is None:
                    errors.append({
                        "file": fpath,
                        "error_type": "ImportSpecError",
                        "error_message": f"Cannot create import spec for {fpath}",
                        "fixable": True,
                    })
            except Exception as e:
                errors.append({
                    "file": fpath,
                    "error_type": "ImportCheckError",
                    "error_message": str(e),
                    "fixable": True,
                })
        
        return {
            "status": "PASS" if not errors else "FAIL",
            "errors": errors,
            "files_checked": files_checked,
        }

    # ════════════════════════════════════════
    # 辅助
    # ════════════════════════════════════════

    def _build_env(self) -> dict:
        """构建环境变量"""
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["CUDA_LAUNCH_BLOCKING"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        return env

    @staticmethod
    def _extract_missing_package(error_msg: str) -> Optional[str]:
        """从错误信息中提取缺失的包名"""
        patterns = [
            r"ModuleNotFoundError: No module named ['\"](.+?)['\"]",
            r"ImportError: No module named ['\"](.+?)['\"]",
            r"cannot import name ['\"](.+?)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_msg)
            if match:
                return match.group(1).split('.')[0]
        return None