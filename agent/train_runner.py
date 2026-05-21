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

                # ---------- 错误分类与自动恢复 ----------
                error_msg = combined[-3000:]

                # ── 提取完整 traceback 详情 ──
                traceback_details = self._extract_traceback_details(combined)

                # ── 分类错误 ──
                error_category = self._classify_error(error_msg, traceback_details)

                # ── CUDA OOM (CONFIG_ERROR — 可通过调参修复) ──
                if error_category == "CONFIG_ERROR" and any(
                    kw in error_msg for kw in ["CUDA out of memory", "out of memory", "OOM", "memory exhausted"]
                ):
                    current_batch = working_overrides.get("batch_size",
                                                          self.adapter.base_args.get("batch_size", 1024))
                    new_batch = max(int(current_batch * self.oom_reduce_factor), 1)
                    working_overrides["batch_size"] = new_batch
                    logger.warning(f"OOM! Reducing batch_size: {current_batch} → {new_batch}")
                    continue  # 重试 (这是配置错误, 自动调参即可)

                # ── NaN Loss (CONFIG_ERROR — 可通过调参修复) ──
                if error_category == "CONFIG_ERROR" and any(
                    kw in error_msg for kw in ["NaN", "nan", "inf", "division by zero", "unexpected EOF", "CUDA error"]
                ):
                    current_lr = working_overrides.get("lr",
                                                       self.adapter.base_args.get("lr", 1e-3))
                    new_lr = current_lr * self.nan_reduce_factor
                    working_overrides["lr"] = new_lr
                    logger.warning(f"NaN/Inf detected! Reducing lr: {current_lr} → {new_lr}")
                    continue  # 重试

                # ── 缺失外部依赖 (SYSTEM_ERROR — 可尝试安装) ──
                missing_pkg = self._extract_missing_package(error_msg)
                if missing_pkg:
                    logger.info(f"Missing package: {missing_pkg}, auto-installing...")
                    install = subprocess.run(
                        f"pip install {missing_pkg}",
                        shell=True, capture_output=True, text=True, timeout=120
                    )
                    logger.info(f"Install result: {install.stdout[-200:]}")
                    continue  # 重试

                # ── 其他所有错误: 返回详细错误信息给上层 ──
                # 关键改变: 不再简单返回 action="log_and_skip"
                # 而是返回 fixable=True/False 和 error_category, 让上层自纠错循环决定如何处理
                
                # 如果有输出但没有 traceback, 且退出码为0 → 可能是指标解析问题, 不一定是训练失败
                if returncode == 0 and not has_traceback:
                    # 训练进程正常退出但没有解析到指标
                    # 这更可能是指标格式问题, 而不是真正的训练失败
                    error_category = "CONFIG_ERROR"  # 改为 CONFIG_ERROR, 让 LLM 尝试分析
                    fixable = True
                else:
                    error_category = self._classify_error(error_msg, traceback_details)
                    fixable = self._is_fixable(error_category, traceback_details)
                
                return {
                    "status": "FAILED",
                    "error": error_msg[:1500],
                    "error_category": error_category,
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
            "error": f"Failed after {max_attempts} attempts (OOM/NaN auto-reduce also failed)",
            "error_category": "CONFIG_ERROR",
            "fixable": True,  # OOM/NaN 调参失败仍可能通过 LLM 诊断修复
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
        分类错误类型:
        - CODE_ERROR:  源码有问题 (语法错误、运行时错误、维度不匹配、变量名错误等)
                       → 需要修改源码文件来修复
        - CONFIG_ERROR: 参数配置有问题 (OOM、NaN、超时等)
                       → 需要调整超参数来修复
        - DATA_ERROR:   数据路径或格式有问题
                       → 需要修正路径或数据格式
        - SYSTEM_ERROR: 系统环境问题 (缺少外部包、GPU不可用等)
                       → 无法自动修复, 需要人工干预
        """
        # ── CODE_ERROR: 源码代码错误 ──
        code_error_patterns = [
            # 语法错误
            "SyntaxError", "IndentationError", "TabError",
            # 运行时错误 (通常来自模型代码中的 bug)
            "RuntimeError: size mismatch", "RuntimeError: dimension",
            "RuntimeError: Expected", "mat1 and mat2 shapes cannot be multiplied",
            "dimension mismatch", "DimensionMismatch",
            "size mismatch for", "shape mismatch",
            "not enough values to unpack", "too many values to unpack",
            # 变量/函数名错误
            "NameError:", "AttributeError:", "TypeError:",
            "UnboundLocalError:",
            # 导入错误 (项目内部的, 不是外部包)
            "cannot import name", "ImportError:",
        ]
        
        # 需要排除: 如果是外部包缺失, 则是 SYSTEM_ERROR
        for pattern in code_error_patterns:
            if pattern in error_msg:
                # 但如果是外部包 (如 torch/numpy 缺失), 不算 CODE_ERROR
                if "ModuleNotFoundError: No module named" in error_msg:
                    # 检查是否是项目内部模块
                    missing_pkg = self._extract_missing_package(error_msg)
                    if missing_pkg and missing_pkg not in ("torch", "numpy", "pandas", "sklearn",
                                                           "transformers", "scipy", "tensorflow"):
                        # 可能是项目内部模块导入路径有问题 → CODE_ERROR
                        # 也可能是外部包缺失 → SYSTEM_ERROR
                        # 检查 traceback 中的文件路径来判断
                        tb_files = traceback_details.get("files", [])
                        if any(self.project_root in f for f in tb_files):
                            return "CODE_ERROR"  # 项目内部代码导入错误
                        else:
                            return "SYSTEM_ERROR"  # 外部包缺失
                return "CODE_ERROR"
        
        # ── CONFIG_ERROR: 参数配置错误 ──
        config_error_patterns = [
            "CUDA out of memory", "out of memory", "OOM", "memory exhausted",
            "NaN", "nan", "inf", "division by zero",
            "CUDA error: an illegal memory access was encountered",
        ]
        for pattern in config_error_patterns:
            if pattern in error_msg:
                return "CONFIG_ERROR"
        
        # ── DATA_ERROR: 数据文件问题 ──
        data_error_patterns = [
            "FileNotFoundError", "No such file or directory",
            "Permission denied",
        ]
        for pattern in data_error_patterns:
            if pattern in error_msg:
                # 检查是否是数据文件路径
                if "/data/" in error_msg or "data_name" in error_msg or ".txt" in error_msg:
                    return "DATA_ERROR"
                # 可能是项目内部源码文件缺失 → CODE_ERROR
                return "CODE_ERROR"
        
        # ── 如果 traceback 出错文件在项目根目录 → CODE_ERROR ──
        tb_files = traceback_details.get("files", [])
        if any(self.project_root in f or "Recmodel" in f for f in tb_files):
            return "CODE_ERROR"
        
        # ── 默认: 如果有 traceback, 认为是 CODE_ERROR ──
        if traceback_details.get("traceback_text"):
            return "CODE_ERROR"
        
        # ── 无法分类 → SYSTEM_ERROR ──
        return "SYSTEM_ERROR"

    def _is_fixable(self, error_category: str, traceback_details: dict) -> bool:
        """
        判断错误是否可以通过自纠错修复:
        - CODE_ERROR: fixable=True → LLM 可以修改源码修复
        - CONFIG_ERROR: fixable=True → LLM 可以调整参数修复
        - DATA_ERROR: fixable=True → LLM 可以修正路径/配置修复
        - SYSTEM_ERROR: fixable=False → 需要人工干预
        """
        if error_category == "SYSTEM_ERROR":
            return False
        # CODE_ERROR, CONFIG_ERROR, DATA_ERROR 都可以通过 LLM 自纠错修复
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
            source_files: 需要检查的源码文件列表 (默认: models.py, modules.py, trainers.py)
        
        Returns:
            {
                "status": "PASS" | "FAIL",
                "errors": [dict],  # 检查发现的问题列表
                "files_checked": [str],  # 检查的文件列表
            }
        """
        if source_files is None:
            source_files = ["models.py", "modules.py", "trainers.py"]
        
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