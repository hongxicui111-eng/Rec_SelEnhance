#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IterativeMemory — 迭代修改记忆系统

核心职责:
1. 记录每次结构修改的完整因果链: 修改内容 → 应用结果 → 指标变化 → 是否回滚
2. 保存每轮迭代时的模型源码快照 (本地文件，不依赖 Git/GitHub)
3. 智能生成 LLM 可理解的"历史修改感知"上下文
4. 支持从本地 JSONL 文件持久化和恢复

这是让 Agent 从"盲目修改"升级到"基于历史感知的渐进修改"的关键模块。

设计理念:
- LLM 每次做结构修改时，必须知道:
  (a) 当前模型的完整代码状态 (不是截断的)
  (b) 之前做了哪些修改、每次修改的具体代码变更
  (c) 每次修改的效果 (指标变化)
  (d) 被回滚的修改及其原因 (避免重复踩坑)
"""

import os
import json
import ast
import shutil
import logging
import hashlib
import difflib
from datetime import datetime
from typing import Optional, List, Dict, Tuple
from pathlib import Path

logger = logging.getLogger("rec_self_evolve.iterative_memory")


class IterativeMemory:
    """
    迭代修改记忆系统
    
    存储:
    1. 每轮迭代的模型源码快照 (完整文件内容)
    2. 每次结构修改的详细记录 (含代码 diff)
    3. 修改效果与指标的关联 (因果链)
    4. 回滚记录 (避免重复失败修改)
    
    所有数据持久化到本地 JSONL + 源码快照目录，无需联网。
    """

    # ════════════════════════════════════════
    # 初始化
    # ════════════════════════════════════════

    def __init__(self, project_root: str, log_dir: str = "evolve_logs",
                 source_files: List[str] = None):
        """
        Args:
            project_root: 项目根目录 (包含 models.py, modules.py 等)
            log_dir: 日志/快照保存目录
            source_files: 需要跟踪的源码文件列表 (默认覆盖 Recmodel 目录下所有 .py 文件)
        """
        self.project_root = project_root
        self.log_dir = Path(log_dir)
        self.source_files = source_files or [
            "models.py", "modules.py", "trainers.py", "datasets.py",
            "utils.py", "error_case_extractor.py", "surprise_eval.py",
            "run_finetune_full.py",
        ]
        
        # 快照目录: 保存每轮迭代时的源码文件副本
        self.snapshot_dir = self.log_dir / "source_snapshots"
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # 记录文件: JSONL 格式，保存每次修改的详细记录
        self.record_file = self.log_dir / "iterative_memory.jsonl"
        
        # 内存中的记录 (也从文件加载历史)
        self.modification_records: List[Dict] = []
        self._load_history()
        
        # 当前活跃的修改记录 (本轮正在进行的)
        self._pending_modification: Optional[Dict] = None
        
        # 源码文件实际路径映射
        self._file_path_cache: Dict[str, str] = {}

    # ════════════════════════════════════════
    # 源码快照管理 (替代 Git)
    # ════════════════════════════════════════

    def save_source_snapshot(self, iteration: int) -> Dict:
        """
        保存当前轮次的模型源码快照
        
        将所有被跟踪的源码文件复制到快照目录:
        evolve_logs/source_snapshots/iter_000/models.py
        evolve_logs/source_snapshots/iter_000/modules.py
        ...
        
        Args:
            iteration: 当前迭代编号
        
        Returns:
            Dict: 保存结果摘要
        """
        snapshot_subdir = self.snapshot_dir / f"iter_{iteration:03d}"
        snapshot_subdir.mkdir(parents=True, exist_ok=True)
        
        saved_files = {}
        for file_key in self.source_files:
            src_path = self._find_source_file(file_key)
            if src_path and os.path.exists(src_path):
                dst_path = snapshot_subdir / file_key
                shutil.copy2(src_path, dst_path)
                saved_files[file_key] = {
                    "src_path": src_path,
                    "snapshot_path": str(dst_path),
                    "size": os.path.getsize(src_path),
                    "content_hash": self._hash_file(src_path),
                }
                logger.info(f"Snapshot saved: {file_key} → {dst_path}")
            else:
                logger.warning(f"Source file not found: {file_key}")
        
        # 保存快照元数据
        meta = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "files": saved_files,
        }
        meta_path = snapshot_subdir / "_snapshot_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        
        return meta

    def restore_source_snapshot(self, iteration: int) -> Dict:
        """
        从快照恢复模型源码
        
        将指定迭代轮次的源码文件复制回项目目录，覆盖当前版本。
        这替代了 Git rollback 功能。
        
        Args:
            iteration: 要恢复到的迭代编号
        
        Returns:
            Dict: 恢复结果摘要
        """
        snapshot_subdir = self.snapshot_dir / f"iter_{iteration:03d}"
        if not snapshot_subdir.exists():
            logger.error(f"Snapshot not found for iteration {iteration}")
            return {"ok": False, "error": f"No snapshot for iter {iteration}"}
        
        restored_files = []
        for file_key in self.source_files:
            snapshot_path = snapshot_subdir / file_key
            if snapshot_path.exists():
                target_path = self._find_source_file(file_key)
                if target_path:
                    shutil.copy2(snapshot_path, target_path)
                    restored_files.append(file_key)
                    logger.info(f"Restored: {file_key} from iter {iteration}")
                else:
                    logger.warning(f"Cannot find target path for {file_key}")
        
        return {
            "ok": True,
            "restored_files": restored_files,
            "restored_to_iteration": iteration,
        }

    def get_source_at_iteration(self, iteration: int, file_key: str) -> Optional[str]:
        """
        获取指定迭代轮次的源码文件内容
        
        Args:
            iteration: 迭代编号
            file_key: 文件标识 (如 "models.py")
        
        Returns:
            str: 该轮次时的文件内容，或 None
        """
        snapshot_path = self.snapshot_dir / f"iter_{iteration:03d}" / file_key
        if snapshot_path.exists():
            with open(snapshot_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        # 如果没有快照，尝试从当前文件读取 (说明是初始状态)
        current_path = self._find_source_file(file_key)
        if current_path and iteration <= 0:
            with open(current_path, 'r', encoding='utf-8') as f:
                return f.read()
        
        return None

    # ════════════════════════════════════════
    # 修改记录管理
    # ════════════════════════════════════════

    def record_modification(self, iteration: int, structural_changes: List[Dict],
                            apply_result: Dict, metrics_before: Dict,
                            metrics_after: Optional[Dict] = None,
                            note: Optional[str] = None) -> Dict:
        """
        记录一次完整的结构修改因果链
        
        Args:
            iteration: 当前迭代编号
            structural_changes: LLM 提出的结构修改列表
            apply_result: StructureApplier 的应用结果
            metrics_before: 修改前的评估指标
            metrics_after: 修改后重训练的评估指标 (可能为None，如果重训练失败)
            note: 可选的备注信息 (如"Code fix round 1 for phase_train")
        
        Returns:
            Dict: 完整的修改记录
        """
        # 计算代码 diff
        diffs = {}
        for file_key in self.source_files:
            before_content = self.get_source_at_iteration(iteration, file_key)
            current_path = self._find_source_file(file_key)
            if before_content and current_path:
                with open(current_path, 'r', encoding='utf-8') as f:
                    after_content = f.read()
                if before_content != after_content:
                    diffs[file_key] = self._compute_diff(before_content, after_content)

        # 计算指标变化
        metrics_delta = {}
        if metrics_before and metrics_after:
            for key in metrics_before:
                if key in metrics_after and isinstance(metrics_before[key], (int, float)) \
                        and isinstance(metrics_after[key], (int, float)):
                    delta = metrics_after[key] - metrics_before[key]
                    metrics_delta[key] = {
                        "before": metrics_before[key],
                        "after": metrics_after[key],
                        "delta": delta,
                        "direction": "↑" if delta > 0 else "↓" if delta < 0 else "→",
                    }

        # 判断修改的整体效果
        effect_summary = self._evaluate_effect(metrics_delta, apply_result)

        record = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            # 修改内容
            "structural_changes": structural_changes,
            # 应用结果
            "apply_status": apply_result.get("status", "UNKNOWN"),
            "files_modified": apply_result.get("files_modified", []),
            "applied_changes": apply_result.get("applied_changes", []),
            "failed_changes": apply_result.get("failed_changes", []),
            # 代码 diff (关键! 让 LLM 知道具体改了什么)
            "code_diffs": diffs,
            # 指标变化 (因果链核心!)
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "metrics_delta": metrics_delta,
            # 效果评估
            "effect_summary": effect_summary,
            # 是否回滚
            "rolled_back": False,
            "rollback_reason": None,
            # 备注 (可选)
            "note": note,
        }

        self.modification_records.append(record)
        self._save_record(record)
        
        logger.info(
            f"Modification recorded: iter={iteration}, "
            f"status={apply_result.get('status')}, "
            f"effect={effect_summary['overall']}"
        )
        
        return record

    def record_rollback(self, iteration: int, reason: str,
                        rollback_to_iteration: Optional[int] = None) -> None:
        """
        记录一次回滚操作
        
        将对应轮次的修改记录标记为"已回滚"，并保存回滚原因。
        这是让 LLM "不重复踩坑"的关键信息。
        
        Args:
            iteration: 被回滚的迭代编号
            reason: 回滚原因
            rollback_to_iteration: 回滚到的迭代编号 (如果执行了源码恢复)
        """
        for record in self.modification_records:
            if record.get("iteration") == iteration:
                record["rolled_back"] = True
                record["rollback_reason"] = reason
                if rollback_to_iteration is not None:
                    record["rollback_to_iteration"] = rollback_to_iteration
                # 更新效果评估
                record["effect_summary"]["overall"] = "ROLLBACK"
                record["effect_summary"]["note"] = f"回滚原因: {reason}"
                break
        
        # 持久化更新
        self._rewrite_all_records()
        
        # 如果指定了回滚目标，恢复源码
        if rollback_to_iteration is not None:
            self.restore_source_snapshot(rollback_to_iteration)
        
        logger.info(f"Rollback recorded: iter={iteration}, reason={reason[:100]}")

    # ════════════════════════════════════════
    # 为 LLM 生成历史感知上下文 (核心!)
    # ════════════════════════════════════════

    def build_history_context_for_llm(self, current_iteration: int,
                                       current_metrics: Optional[Dict] = None,
                                       max_detail_iterations: int = 5,
                                       max_chars: int = 9000) -> str:
        """
        为 LLM 生成完整的历史修改感知上下文
        
        这是解决"LLM 对历史修改没有感知"的核心方法。
        LLM 将看到:
        1. 所有历史修改的完整因果链 (修改 → 效果 → 是否回滚)
        2. 被回滚的修改及其原因 (避免重复踩坑)
        3. 成功修改的累积效果趋势
        4. 当前模型相对于原始模型的累积变更摘要
        
        Args:
            current_iteration: 当前迭代编号
            current_metrics: 当前指标 (用于与历史比较)
            max_detail_iterations: 详细展示的最近迭代次数
        
        Returns:
            str: 格式化的历史感知上下文，直接注入 LLM Prompt
        """
        if not self.modification_records:
            return ""
        
        parts = []
        
        # ─── Section 1: 修改历史因果链 ───
        parts.append("## 📋 结构修改历史因果链 (完整记录)\n")
        parts.append("以下是每次结构修改的完整因果链，包含: 修改内容 → 应用结果 → 指标变化 → 是否回滚。\n")
        parts.append("**务必仔细阅读!** 不要重复已做过但效果不好/被回滚的修改。\n\n")
        
        # 按效果分类展示
        successful = [r for r in self.modification_records if r["effect_summary"]["overall"] == "POSITIVE"]
        neutral = [r for r in self.modification_records if r["effect_summary"]["overall"] == "NEUTRAL"]
        negative = [r for r in self.modification_records if r["effect_summary"]["overall"] == "NEGATIVE"]
        rolled_back = [r for r in self.modification_records if r["rolled_back"]]
        
        # 1a: 回滚记录 (最重要! 告诉 LLM 别再犯了)
        if rolled_back:
            parts.append("### ❌ 被回滚的修改 (不要重复尝试!)\n")
            for r in rolled_back:
                iter_num = r["iteration"]
                reason = r.get("rollback_reason", "未知")
                changes_desc = self._summarize_changes_brief(r.get("structural_changes", []))
                parts.append(f"- **Iter {iter_num}**: {changes_desc}")
                parts.append(f"  回滚原因: {reason[:150]}")
                parts.append(f"  ⚠ **不要重复类似修改!**\n")
        
        # 1b: 效果不好的修改
        if negative:
            parts.append("### ⬇ 效果不好的修改 (谨慎考虑)\n")
            for r in negative:
                iter_num = r["iteration"]
                changes_desc = self._summarize_changes_brief(r.get("structural_changes", []))
                delta_desc = self._format_metrics_delta(r.get("metrics_delta", {}))
                parts.append(f"- **Iter {iter_num}**: {changes_desc} → {delta_desc}\n")
        
        # 1c: 效果好的修改 (可以在此基础上继续深化)
        if successful:
            parts.append("### ⬆ 效果好的修改 (可以在此基础上继续深化)\n")
            for r in successful:
                iter_num = r["iteration"]
                changes_desc = self._summarize_changes_brief(r.get("structural_changes", []))
                delta_desc = self._format_metrics_delta(r.get("metrics_delta", {}))
                parts.append(f"- **Iter {iter_num}**: {changes_desc} → {delta_desc}\n")
        
        # 1d: 效果中性的修改
        if neutral:
            parts.append("### → 效果中性的修改\n")
            for r in neutral:
                iter_num = r["iteration"]
                changes_desc = self._summarize_changes_brief(r.get("structural_changes", []))
                delta_desc = self._format_metrics_delta(r.get("metrics_delta", {}))
                parts.append(f"- **Iter {iter_num}**: {changes_desc} → {delta_desc}\n")
        
        # ─── Section 2: 最近几轮的详细修改记录 ───
        recent = self.modification_records[-max_detail_iterations:]
        if recent:
            parts.append("\n### 🔍 最近修改的详细记录\n\n")
            for r in recent:
                iter_num = r["iteration"]
                status = r.get("apply_status", "?")
                effect = r["effect_summary"]["overall"]
                rolled = r.get("rolled_back", False)
                
                status_icon = "✓" if status in ("SUCCESS", "PARTIAL_SUCCESS") else "✗"
                effect_icon = {"POSITIVE": "⬆", "NEUTRAL": "→", "NEGATIVE": "⬇", "ROLLBACK": "↩"}.get(effect, "?")
                rollback_tag = " [已回滚]" if rolled else ""
                
                parts.append(f"**Iter {iter_num}** {status_icon}{effect_icon}{rollback_tag}\n")
                
                # 展示每项修改的详细信息
                for sc in r.get("structural_changes", []):
                    parts.append(f"  - [{sc.get('target_file', '?')}] "
                                 f"{sc.get('target_class_or_function', '?')}: "
                                 f"{sc.get('description', '?')[:100]}\n")
                
                # 展示代码 diff 概要 (不是完整 diff，太长了)
                diffs = r.get("code_diffs", {})
                if diffs:
                    for file_key, diff_text in diffs.items():
                        # 只展示 diff 的统计和关键行
                        diff_summary = self._summarize_diff(diff_text)
                        parts.append(f"  - 代码变更 [{file_key}]: {diff_summary}\n")
                
                # 展示指标变化
                delta = r.get("metrics_delta", {})
                if delta:
                    parts.append(f"  - 指标变化: {self._format_metrics_delta(delta)}\n")
                
                if rolled:
                    parts.append(f"  - 回滚原因: {r.get('rollback_reason', '?')[:150]}\n")
                
                parts.append("\n")
        
        # ─── Section 3: 累积变更摘要 ───
        parts.append(self._build_cumulative_change_summary())
        
        # ─── Section 4: 指标趋势 ───
        if current_metrics:
            trend = self._compute_metric_trend(current_metrics)
            parts.append(f"\n### 📊 指标趋势\n{trend}\n")
        
        # ─── Section 5: 修改建议 ───
        parts.append(self._build_strategy_guidance())

        full_text = "\n".join(parts)
        if len(full_text) > max_chars:
            head = int(max_chars * 0.7)
            tail = max_chars - head - 30
            full_text = full_text[:head] + "\n\n... [HISTORY CLIPPED] ...\n\n" + full_text[-max(0, tail):]
        return full_text

    def build_rollback_aware_context(self) -> str:
        """
        生成专门针对回滚修改的上下文
        
        如果有被回滚的修改，生成一个醒目的警告，
        告诉 LLM 不要再犯同样的错误。
        
        Returns:
            str: 回滚警告上下文
        """
        rolled_back = [r for r in self.modification_records if r.get("rolled_back")]
        if not rolled_back:
            return ""
        
        parts = [
            "\n## ⛔ 回滚修改黑名单 (严禁重复!)\n",
            "以下修改已被尝试并回滚，**绝对不要再提出类似的修改**:\n\n",
        ]
        
        for r in rolled_back:
            iter_num = r["iteration"]
            changes = r.get("structural_changes", [])
            reason = r.get("rollback_reason", "未知")
            
            parts.append(f"### Iter {iter_num} 的修改 (已被回滚)\n")
            for sc in changes:
                parts.append(f"- **{sc.get('action_type', '?')}**: "
                             f"[{sc.get('target_file', '?')}] "
                             f"{sc.get('target_class_or_function', '?')}\n")
                parts.append(f"  描述: {sc.get('description', '?')[:200]}\n")
            parts.append(f"  回滚原因: {reason[:300]}\n\n")
        
        parts.append("⚠ 如果你认为某个回滚的修改方向仍然有价值，必须**显著改变实现方式**，")
        parts.append("而不是简单地重复相同的代码改动。\n")
        
        return "\n".join(parts)

    # ════════════════════════════════════════
    # 智能源码上下文 (替代截断方案)
    # ════════════════════════════════════════

    def build_smart_source_context(self, include_files: List[str] = None,
                                   max_total_chars: int = 15000) -> str:
        """
        构建智能源码上下文 — 确保 SEARCH/REPLACE 可用
        
        ⚠ 核心设计原则: LLM 生成 SEARCH/REPLACE 编辑时需要看到精确的源码文本!
        截断或总结化的源码会导致 LLM 无法生成匹配的 search 文本, 从而编辑失败。
        
        策略 (v2 — SEARCH/REPLACE 兼容):
        1. 训练脚本 (run_*.py) 和 argparse 文件 → 必须完整展示, 不截断
        2. 最近修改过的文件 → 展示修改区域 + 其上下文
        3. 核心模型文件 (models.py, modules.py) → 完整展示 (不超过限制)
        4. 其他文件 → 完整展示优先, 如果超过限制才签名化
        5. 总长度不超过 max_total_chars
        
        Args:
            include_files: 要包含的文件列表
            max_total_chars: 最大总字符数
        
        Returns:
            str: 格式化的源码上下文
        """
        include_files = include_files or self.source_files
        parts = []
        remaining_chars = max_total_chars
        
        # ── 优先级排序 (v2): 不可截断的文件优先 ──
        # 训练脚本和 argparse 文件必须完整展示, 否则 LLM 无法生成精确的 SEARCH/REPLACE
        priority_files = self._sort_files_by_display_priority(include_files)
        
        # ── 两阶段分配策略 ──
        # Phase 1: 完整展示不可截断的文件 (训练脚本 + argparse + 最近修改的)
        # Phase 2: 完整展示其他文件 (如果空间够) 或签名化展示
        
        shown_files = []
        omitted_files = []
        
        for file_key in priority_files:
            content = self._get_current_source(file_key)
            if not content:
                continue
            
            # 判断文件类型: 是否是不可截断的
            is_indispensable = self._is_indispensable_file(file_key, content)
            
            # 判断文件是否被修改过
            is_modified, modified_regions = self._detect_modified_regions(file_key)
            
            if is_indispensable:
                # ── 不可截断文件: 必须完整展示 ──
                # 训练脚本 (run_*.py)、包含 argparse 的文件 — LLM 必须
                # 看到精确文本才能生成有效的 SEARCH/REPLACE
                needed_chars = len(content) + 200  # markdown 包裹开销
                if needed_chars <= remaining_chars:
                    file_section = f"### 文件: {file_key} (完整展示 — 不可截断)\n```python\n{content}\n```\n\n"
                    parts.append(file_section)
                    remaining_chars -= len(file_section)
                    shown_files.append(file_key)
                else:
                    # 空间不够 — 仍然完整展示, 但警告 LLM
                    logger.warning(f"Indispensable file {file_key} ({len(content)} chars) "
                                   f"exceeds remaining budget ({remaining_chars} chars)")
                    file_section = f"### 文件: {file_key} (完整展示 — ⚠ 超出预算但仍展示)\n```python\n{content}\n```\n\n"
                    parts.append(file_section)
                    remaining_chars = max(0, remaining_chars - len(file_section))
                    shown_files.append(file_key)
            elif is_modified and modified_regions:
                # ── 已修改文件: 展示修改区域 ──
                file_section = self._build_modified_file_context(
                    file_key, content, modified_regions, remaining_chars
                )
                parts.append(file_section)
                remaining_chars -= len(file_section)
                shown_files.append(file_key)
            else:
                # ── 其他文件: 完整展示优先 ──
                if len(content) <= remaining_chars - 200:
                    file_section = f"### 文件: {file_key} (原始版本，未修改)\n```python\n{content}\n```\n\n"
                    parts.append(file_section)
                    remaining_chars -= len(file_section)
                    shown_files.append(file_key)
                else:
                    # 文件太长 → 签名化展示
                    file_section = self._build_unmodified_file_summary(
                        file_key, content, remaining_chars
                    )
                    if file_section:
                        parts.append(file_section)
                        remaining_chars -= len(file_section)
                        shown_files.append(file_key)
                    else:
                        omitted_files.append(file_key)
        
        # ── 添加 SEARCH/REPLACE 使用提示 ──
        parts.append(
            "\n⚠ **SEARCH/REPLACE 编辑注意**: 你的 edits 中的 search 文本必须与上面展示的源码"
            "**完全精确匹配** (包括空格、引号、换行)。不要猜测或凭记忆编写 search 文本 — "
            "直接复制上面展示的源码内容作为 search 文本!\n"
        )
        
        # ── 被省略的文件列表 ──
        omitted = [f for f in priority_files if f not in shown_files]
        if omitted:
            parts.append(f"\n⚠ 以下文件因空间限制未完整展示: {', '.join(omitted)}\n")
            parts.append("如需修改这些文件，请先请求系统展示其完整内容。\n")
        
        return "\n".join(parts)

    def _is_indispensable_file(self, file_key: str, content: str) -> bool:
        """
        判断文件是否不可截断 — 必须完整展示
        
        不可截断的文件类型:
        1. 训练脚本 (run_*.py) — LLM 需要精确的 argparse 定义
        2. 包含 parser.add_argument 的文件 — LLM 需要精确的参数定义文本
        3. 短文件 (< 500 chars) — 截断不如完整展示
        
        Args:
            file_key: 文件标识
            content: 文件内容
        
        Returns:
            bool: True = 不可截断, 必须完整展示
        """
        # 训练脚本
        if file_key.startswith("run_") or "train" in file_key.lower():
            return True
        
        # 包含 argparse 的文件
        if "parser.add_argument" in content or "ArgumentParser" in content:
            return True
        
        # 短文件 (< 500 chars) — 完整展示更高效
        if len(content) < 500:
            return True
        
        return False

    def _sort_files_by_display_priority(self, files: List[str]) -> List[str]:
        """
        按展示优先级排序文件 (v2 — 不可截断文件优先)
        
        优先级:
        1. 不可截断文件 (训练脚本 + argparse) — 必须完整展示
        2. 最近修改过的文件 — 需要展示修改区域
        3. 核心模型文件 (models.py, modules.py, trainers.py)
        4. 其他文件
        """
        # 计算每个文件的优先级分数
        scored_files = []
        for file_key in files:
            content = self._get_current_source(file_key)
            if not content:
                scored_files.append((file_key, 0))
                continue
            
            score = 0
            
            # 不可截断文件 → 最高优先级
            if self._is_indispensable_file(file_key, content):
                score += 100
            
            # 最近修改过的文件 → 高优先级
            last_modified_iter = -1
            for record in reversed(self.modification_records):
                if record.get("rolled_back"):
                    continue
                for sc in record.get("structural_changes", []):
                    if sc.get("target_file") == file_key:
                        last_modified_iter = max(last_modified_iter, record["iteration"])
                        break
            score += last_modified_iter + 1  # +1 so unmodified gets 0
            
            # 核心模型文件 → 中等优先级
            core_files = {"models.py": 10, "modules.py": 9, "trainers.py": 8}
            score += core_files.get(file_key, 0)
            
            # 短文件 → 优先展示 (性价比高)
            if content and len(content) < 1000:
                score += 5
            
            scored_files.append((file_key, score))
        
        # 按分数降序排序
        scored_files.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in scored_files]

    # ════════════════════════════════════════
    # 内部辅助方法
    # ════════════════════════════════════════

    def _find_source_file(self, file_key: str) -> Optional[str]:
        """查找源码文件的实际路径"""
        if file_key in self._file_path_cache:
            return self._file_path_cache[file_key]
        
        candidates = [
            os.path.join(self.project_root, file_key),
            os.path.join(self.project_root, "Recmodel", file_key),
        ]
        
        for path in candidates:
            if os.path.exists(path):
                self._file_path_cache[file_key] = path
                return path
        
        logger.warning(f"Source file not found: {file_key}")
        return None

    def _get_current_source(self, file_key: str) -> Optional[str]:
        """获取当前版本的源码文件内容"""
        path = self._find_source_file(file_key)
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def _hash_file(self, file_path: str) -> str:
        """计算文件内容的 hash (用于检测变更)"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:12]

    @staticmethod
    def _compute_diff(before: str, after: str) -> str:
        """计算两个版本之间的 diff"""
        before_lines = before.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        diff = difflib.unified_diff(
            before_lines, after_lines,
            fromfile="before", tofile="after",
            lineterm=""
        )
        return "\n".join(diff)

    @staticmethod
    def _summarize_diff(diff_text: str) -> str:
        """将完整 diff 概要化为统计信息"""
        lines = diff_text.split('\n')
        added = sum(1 for l in lines if l.startswith('+') and not l.startswith('+++'))
        removed = sum(1 for l in lines if l.startswith('-') and not l.startswith('---'))
        
        # 找出被修改的函数/类名
        modified_symbols = []
        for l in lines:
            if l.startswith('+') and ('def ' in l or 'class ' in l):
                # 提取名称
                for keyword in ('def ', 'class '):
                    if keyword in l:
                        idx = l.index(keyword) + len(keyword)
                        name_part = l[idx:].split('(')[0].split(':')[0].strip()
                        modified_symbols.append(name_part)
        
        summary = f"+{added}/-{removed} 行"
        if modified_symbols:
            summary += f", 涉及: {', '.join(modified_symbols[:5])}"
        return summary

    def _evaluate_effect(self, metrics_delta: Dict, apply_result: Dict) -> Dict:
        """
        评估修改的整体效果
        
        Returns:
            Dict: {
                "overall": "POSITIVE" | "NEUTRAL" | "NEGATIVE" | "ROLLBACK" | "UNKNOWN",
                "key_metric_delta": float,  # NDCG@5 的变化量
                "note": str,
            }
        """
        if apply_result.get("status") == "ROLLBACK":
            return {"overall": "ROLLBACK", "key_metric_delta": 0, "note": "修改被回滚"}
        
        if not metrics_delta:
            return {"overall": "UNKNOWN", "key_metric_delta": 0, "note": "无指标数据"}
        
        # 以 NDCG@5 作为主要判断指标 (如果有的话)
        key_metric = None
        for preferred in ["NDCG@5", "NDCG@10", "NDCG@20", "R@5", "R@10"]:
            if preferred in metrics_delta:
                key_metric = preferred
                break
        
        if key_metric is None:
            # 取第一个指标
            key_metric = list(metrics_delta.keys())[0] if metrics_delta else None
        
        if key_metric and key_metric in metrics_delta:
            delta = metrics_delta[key_metric]["delta"]
            # 判断阈值: NDCG 变化 < 0.005 视为中性
            if delta > 0.005:
                return {
                    "overall": "POSITIVE",
                    "key_metric_delta": delta,
                    "note": f"{key_metric} 提升 {delta:+.4f}",
                }
            elif delta < -0.005:
                return {
                    "overall": "NEGATIVE",
                    "key_metric_delta": delta,
                    "note": f"{key_metric} 下降 {delta:+.4f}",
                }
            else:
                return {
                    "overall": "NEUTRAL",
                    "key_metric_delta": delta,
                    "note": f"{key_metric} 变化微小 ({delta:+.4f})",
                }
        
        return {"overall": "UNKNOWN", "key_metric_delta": 0, "note": "无法判断"}

    @staticmethod
    def _summarize_changes_brief(changes: List[Dict]) -> str:
        """简述一组修改"""
        if not changes:
            return "无修改"
        parts = []
        for c in changes[:5]:
            parts.append(f"[{c.get('target_file', '?')}] "
                         f"{c.get('target_class_or_function', '?')}: "
                         f"{c.get('description', '?')[:80]}")
        return "; ".join(parts)

    @staticmethod
    def _format_metrics_delta(delta: Dict) -> str:
        """格式化指标变化"""
        if not delta:
            return "无数据"
        parts = []
        for key, val in delta.items():
            direction = val.get("direction", "?")
            before = val.get("before", "?")
            after_val = val.get("after", "?")
            d = val.get("delta", "?")
            if isinstance(d, float):
                parts.append(f"{key}: {before:.4f}→{after_val:.4f} ({direction}{d:+.4f})")
            else:
                parts.append(f"{key}: {before}→{after_val}")
        return " | ".join(parts[:5])

    def _detect_modified_regions(self, file_key: str) -> Tuple[bool, List[Dict]]:
        """
        检测文件中被修改的区域
        
        Returns:
            (is_modified, modified_regions)
            modified_regions 是一个列表，每项包含:
            {
                "symbol_name": "SelfAttention.forward",
                "start_line": 30,
                "end_line": 50,
                "modification_iter": 2,
            }
        """
        modified_regions = []
        
        # 检查最近的修改记录，看是否涉及这个文件
        for record in reversed(self.modification_records):
            if record.get("rolled_back"):
                continue  # 被回滚的修改不算
            
            for sc in record.get("structural_changes", []):
                if sc.get("target_file") == file_key:
                    modified_regions.append({
                        "symbol_name": sc.get("target_class_or_function", "?"),
                        "action_type": sc.get("action_type", "?"),
                        "description": sc.get("description", ""),
                        "modification_iter": record["iteration"],
                    })
        
        is_modified = len(modified_regions) > 0
        return is_modified, modified_regions

    def _sort_files_by_modification_priority(self, files: List[str]) -> List[str]:
        """
        按修改优先级排序文件
        
        最近被修改过的文件排在前面，让 LLM 优先看到。
        """
        priority = []
        for file_key in files:
            # 计算这个文件最近被修改的时间 (迭代编号)
            last_modified_iter = -1
            for record in reversed(self.modification_records):
                if record.get("rolled_back"):
                    continue
                for sc in record.get("structural_changes", []):
                    if sc.get("target_file") == file_key:
                        last_modified_iter = max(last_modified_iter, record["iteration"])
                        break
            
            priority.append((file_key, last_modified_iter))
        
        # 按最近修改时间降序排序
        priority.sort(key=lambda x: x[1], reverse=True)
        return [f[0] for f in priority]

    def _build_modified_file_context(self, file_key: str, content: str,
                                      modified_regions: List[Dict],
                                      max_chars: int) -> str:
        """
        为被修改过的文件构建上下文
        
        策略: 展示修改区域的完整代码 + 其上下文 (前后各5行)
        """
        import ast
        
        parts = [f"### 文件: {file_key} (已被修改 — 展示修改区域的完整代码)\n"]
        
        # 尝试 AST 解析，找到修改的符号对应的行号范围
        try:
            tree = ast.parse(content)
            symbol_ranges = self._find_symbol_ranges(tree, content)
        except SyntaxError:
            symbol_ranges = {}
        
        shown_sections = []
        remaining = max_chars - 200
        
        for region in modified_regions:
            symbol_name = region.get("symbol_name", "")
            
            # 找到符号对应的行号范围
            if symbol_name in symbol_ranges:
                start, end = symbol_ranges[symbol_name]
                # 展示上下文: 前后各扩展 5 行
                ctx_start = max(0, start - 5)
                ctx_end = min(len(content.split('\n')), end + 5)
                
                lines = content.split('\n')
                section = '\n'.join(lines[ctx_start:ctx_end])
                
                header = f"\n#### 修改区域: {symbol_name} (Iter {region.get('modification_iter', '?')} — {region.get('action_type', '?')})\n"
                header += f"描述: {region.get('description', '?')[:150]}\n"
                header += "```python\n"
                footer = "\n```\n"
                
                section_text = header + section + footer
                if len(section_text) <= remaining:
                    shown_sections.append(section_text)
                    remaining -= len(section_text)
            else:
                # 找不到具体行号，用描述替代
                desc_text = f"- 修改了 {symbol_name}: {region.get('description', '?')[:150]}\n"
                shown_sections.append(desc_text)
                remaining -= len(desc_text)
        
        # 如果还有空间，展示文件的其他关键定义 (签名)
        if remaining > 300:
            summary = self._build_file_signature_summary(file_key, content, symbol_ranges)
            if len(summary) <= remaining:
                shown_sections.append(summary)
        
        parts.extend(shown_sections)
        
        # 添加完整代码的获取提示
        full_code = self._get_current_source(file_key)
        if full_code and len(full_code) > remaining:
            parts.append(f"\n⚠ 文件 {file_key} 总共 {len(content)} 字符，以上只展示了修改区域。")
            parts.append(f"完整代码已在快照目录保存 (evolve_logs/source_snapshots/)。\n")
        
        return "\n".join(parts)

    def _build_unmodified_file_summary(self, file_key: str, content: str,
                                        max_chars: int) -> str:
        """
        为未修改过的文件构建摘要
        
        策略: 展示完整的类/函数定义签名 + 部分关键代码
        """
        import ast
        
        parts = [f"### 文件: {file_key} (原始版本，未修改 — 展示关键定义)\n"]
        
        try:
            tree = ast.parse(content)
            symbol_ranges = self._find_symbol_ranges(tree, content)
        except SyntaxError:
            # 降级: 展示前 max_chars 字符
            return f"### 文件: {file_key}\n```python\n{content[:max_chars-100]}\n```\n\n"
        
        remaining = max_chars - 200
        lines = content.split('\n')
        
        # 展示所有类定义的 __init__ 和关键方法的完整代码
        for symbol_name, (start, end) in symbol_ranges.items():
            # 类的 __init__ 方法
            if symbol_name.endswith(".__init__") or symbol_name.endswith(".forward") \
                    or symbol_name.endswith(".finetune") or "." not in symbol_name:
                # 关键方法/函数 → 展示完整代码
                section = '\n'.join(lines[start:end])
                section_text = f"#### {symbol_name}\n```python\n{section}\n```\n\n"
                if len(section_text) <= remaining:
                    parts.append(section_text)
                    remaining -= len(section_text)
            else:
                # 其他方法 → 只展示签名 (前几行)
                sig_lines = min(5, end - start)
                section = '\n'.join(lines[start:start + sig_lines])
                section_text = f"#### {symbol_name} (签名)\n```python\n{section}\n...\n```\n\n"
                if len(section_text) <= remaining:
                    parts.append(section_text)
                    remaining -= len(section_text)
        
        return "\n".join(parts)

    @staticmethod
    def _find_symbol_ranges(tree: ast.AST, content: str) -> Dict[str, Tuple[int, int]]:
        """
        通过 AST 找到所有符号的行号范围
        
        Returns:
            Dict: {"ClassName": (start, end), "ClassName.method": (start, end), "func_name": (start, end)}
                  行号是 0-based 的切片索引
        """
        ranges = {}
        lines_count = len(content.split('\n'))
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # 类定义的范围
                start = node.lineno - 1  # 0-based
                end = min(node.end_lineno, lines_count)  # 可以直接用 end_lineno
                ranges[node.name] = (start, end)
                
                # 类中的方法
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_start = item.lineno - 1
                        method_end = min(item.end_lineno, lines_count)
                        ranges[f"{node.name}.{item.name}"] = (method_start, method_end)
            
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start = node.lineno - 1
                end = min(node.end_lineno, lines_count)
                ranges[node.name] = (start, end)
        
        return ranges

    def _build_file_signature_summary(self, file_key: str, content: str,
                                        symbol_ranges: Dict) -> str:
        """构建文件的关键定义签名摘要"""
        lines = content.split('\n')
        parts = [f"\n#### {file_key} 的其他定义 (签名)\n"]
        
        for symbol_name, (start, end) in symbol_ranges.items():
            # 只展示前 3 行 (签名)
            sig = '\n'.join(lines[start:start + min(3, end - start)])
            parts.append(f"```python\n{sig}\n...\n```\n")
        
        return "\n".join(parts)

    def _build_cumulative_change_summary(self) -> str:
        """
        构建累积变更摘要
        
        总结从原始版本到当前版本的所有变更。
        """
        if not self.modification_records:
            return ""
        
        # 统计各类修改的次数
        action_counts = {}
        for r in self.modification_records:
            if r.get("rolled_back"):
                continue
            for sc in r.get("structural_changes", []):
                action_type = sc.get("action_type", "unknown")
                action_counts[action_type] = action_counts.get(action_type, 0) + 1
        
        # 统计成功/失败/回滚
        total = len(self.modification_records)
        successful_count = sum(1 for r in self.modification_records 
                               if r["effect_summary"]["overall"] == "POSITIVE")
        negative_count = sum(1 for r in self.modification_records 
                            if r["effect_summary"]["overall"] == "NEGATIVE")
        rollback_count = sum(1 for r in self.modification_records if r.get("rolled_back"))
        
        # 比较原始版本和当前版本的指标
        first_record = self.modification_records[0]
        last_non_rollback = None
        for r in reversed(self.modification_records):
            if not r.get("rolled_back") and r.get("metrics_after"):
                last_non_rollback = r
                break
        
        parts = [
            "\n### 📈 累积变更摘要\n",
            f"- 总修改轮次: {total}",
            f"- 有效修改 (未回滚): {total - rollback_count}",
            f"- 效果好的修改: {successful_count}",
            f"- 效果差的修改: {negative_count}",
            f"- 被回滚的修改: {rollback_count}",
            f"- 修改类型分布: {json.dumps(action_counts, ensure_ascii=False)}",
        ]
        
        # 累积指标变化
        if first_record.get("metrics_before") and last_non_rollback and last_non_rollback.get("metrics_after"):
            initial_metrics = first_record["metrics_before"]
            current_metrics = last_non_rollback["metrics_after"]
            cumulative_delta = {}
            for key in initial_metrics:
                if key in current_metrics and isinstance(initial_metrics[key], (int, float)) \
                        and isinstance(current_metrics[key], (int, float)):
                    cumulative_delta[key] = current_metrics[key] - initial_metrics[key]
            
            if cumulative_delta:
                parts.append(f"- 累积指标变化 (从初始到当前):")
                for key, delta in sorted(cumulative_delta.items()):
                    direction = "↑" if delta > 0 else "↓" if delta < 0 else "→"
                    parts.append(f"  {key}: {direction}{delta:+.4f}")
        
        return "\n".join(parts)

    def _compute_metric_trend(self, current_metrics: Dict) -> str:
        """计算指标趋势"""
        # 取最近 5 个未回滚的有效修改记录
        valid_records = [r for r in self.modification_records 
                        if not r.get("rolled_back") and r.get("metrics_after")]
        recent = valid_records[-5:]
        
        if not recent:
            return "无趋势数据"
        
        lines = []
        for key in ["NDCG@5", "NDCG@10", "R@5", "R@10"]:
            values = []
            for r in recent:
                val = r.get("metrics_after", {}).get(key)
                if val is not None:
                    values.append((r["iteration"], val))
            
            if len(values) >= 2:
                first_val = values[0][1]
                last_val = values[-1][1]
                trend = last_val - first_val
                direction = "↑" if trend > 0 else "↓" if trend < 0 else "→"
                lines.append(f"  {key}: {direction} (趋势 {trend:+.4f}, 最近值 {last_val:.4f})")
        
        # 加入当前指标
        if current_metrics:
            lines.append(f"\n  当前指标:")
            for key in ["NDCG@5", "NDCG@10", "R@5", "R@10", "MRR@10"]:
                if key in current_metrics:
                    lines.append(f"    {key} = {current_metrics[key]:.4f}")
        
        return "\n".join(lines)

    def _build_strategy_guidance(self) -> str:
        """基于历史修改记录，为 LLM 构建策略指导"""
        parts = ["\n### 💡 修改策略指导\n"]
        
        # 分析哪些类型的修改效果最好
        type_effects = {}
        for r in self.modification_records:
            if r.get("rolled_back"):
                continue
            for sc in r.get("structural_changes", []):
                action_type = sc.get("action_type", "unknown")
                effect = r["effect_summary"]["overall"]
                if action_type not in type_effects:
                    type_effects[action_type] = {"positive": 0, "negative": 0, "neutral": 0}
                type_effects[action_type][effect.lower()] = \
                    type_effects[action_type].get(effect.lower(), 0) + 1
        
        # 推荐效果好的类型
        good_types = sorted(
            [(t, e.get("positive", 0) - e.get("negative", 0)) for t, e in type_effects.items()],
            key=lambda x: x[1], reverse=True
        )
        
        if good_types:
            parts.append("- 效果较好的修改类型:")
            for t, score in good_types[:3]:
                if score > 0:
                    parts.append(f"  {t} (净效果 +{score})")
        
        # 警告效果差的类型
        bad_types = [t for t, score in good_types if score < 0]
        if bad_types:
            parts.append("- 效果较差的修改类型 (谨慎尝试或换方向):")
            for t in bad_types:
                parts.append(f"  {t}")
        
        # 通用建议
        parts.append("\n- 如果连续 2-3 轮修改效果都不好，考虑:")
        parts.append("  1. 回退到效果最好的版本，尝试完全不同的修改方向")
        parts.append("  2. 不做结构修改，只微调超参数 (有时候调参就够了)")
        parts.append("  3. 在已有成功修改的基础上深化，而不是另起炉灶")
        
        return "\n".join(parts)

    # ════════════════════════════════════════
    # 持久化
    # ════════════════════════════════════════

    def _save_record(self, record: Dict):
        """追加保存一条记录到 JSONL"""
        self.record_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.record_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _load_history(self):
        """从 JSONL 文件加载历史记录"""
        if self.record_file.exists():
            with open(self.record_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.modification_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Loaded {len(self.modification_records)} modification records from history")

    def _rewrite_all_records(self):
        """重写所有记录 (用于更新回滚状态等)"""
        with open(self.record_file, 'w', encoding='utf-8') as f:
            for record in self.modification_records:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def get_summary_stats(self) -> Dict:
        """获取记忆系统的统计信息"""
        total = len(self.modification_records)
        rolled_back = sum(1 for r in self.modification_records if r.get("rolled_back"))
        positive = sum(1 for r in self.modification_records 
                       if r["effect_summary"]["overall"] == "POSITIVE")
        
        return {
            "total_modifications": total,
            "successful_modifications": positive,
            "rolled_back_modifications": rolled_back,
            "snapshots_available": len(list(self.snapshot_dir.glob("iter_*"))),
        }