#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StructureApplier — 模型结构修改应用器 (v2)

基于开源 Agent 项目的最佳实践重写:
- Aider 的 SEARCH/REPLACE diff 格式 — LLM 输出结构化编辑指令, 而非自由代码
- SWE-Agent 的 lint + 执行验证 — 不仅检查语法, 还验证代码能否 import/运行
- OpenHands 的 post-edit 反馈 — 编辑后展示上下文, 让 LLM 检查结果

核心变更 (vs v1):
1. 删除 7 种替换策略 + 多层回退, 只保留 SEARCH/REPLACE + whole_file 两种
2. 删除 _strip_class_wrapper 等预处理, LLM 必须输出精确的 search/replace 块
3. 新增模糊匹配 (difflib.SequenceMatcher) — 容忍 LLM 输出的小偏差
4. 新增执行验证 (subprocess import check)
5. 新增 post-edit 上下文展示 (供 LLM 自纠错)

兼容性:
- 新格式: {"edits": [{"search": "...", "replace": "..."}]} — 推荐
- 旧格式: {"new_code": "...", "insert_position": "..."} — 自动转换为新格式后处理
"""

import os
import ast
import re
import json
import shutil
import hashlib
import logging
import subprocess
import difflib
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple

logger = logging.getLogger("rec_self_evolve.structure_applier")


class StructureApplier:
    """
    模型结构修改应用器 (v2 — SEARCH/REPLACE 方案)

    工作流程:
    1. 本地文件快照 (创建备份目录)
    2. 逐个应用 SEARCH/REPLACE 编辑块
    3. 语法校验 (ast.parse) + 执行验证 (subprocess import check)
    4. 校验失败 → 从本地快照回滚
    5. 校验成功 → 保留修改 + 生成 post-edit 反馈
    """

    def __init__(self, project_root: str, adapter=None,
                 log_dir: str = "evolve_logs",
                 source_files: List[str] = None):
        """
        Args:
            project_root: 项目根目录路径
            adapter: SeqRecAdapter 实例 (用于获取文件路径映射)
            log_dir: 快照保存目录
            source_files: 需要跟踪的源码文件列表
        """
        self.project_root = project_root
        self.adapter = adapter
        self._applied_changes = []
        # ── 本地快照系统 ──
        self._snapshot_dir = Path(log_dir) / "rollback_snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._current_snapshot_id = None
        self._source_files = source_files or [
            "models.py", "modules.py", "trainers.py", "datasets.py",
            "utils.py", "error_case_extractor.py", "surprise_eval.py",
            "run_finetune_full.py",
        ]
        self._pre_snapshot_hashes = {}
        # 保留字段兼容旧代码引用
        self._backup_branch = None
        # ── 模糊匹配参数 ──
        self.fuzzy_min_similarity = 0.80  # 模糊匹配最低相似度阈值

    # ════════════════════════════════════════
    # 主入口 — 应用一组结构修改
    # ════════════════════════════════════════

    def apply_structural_changes(self, structural_changes: List[Dict]) -> Dict:
        """
        应用一组 LLM 提出的结构修改

        支持两种输入格式:
        - 新格式 (推荐): 每项包含 "edits" 字段 (SEARCH/REPLACE 列表)
        - 旧格式 (兼容): 每项包含 "new_code" + "insert_position" 等字段

        Args:
            structural_changes: LLM 输出的结构修改列表

        Returns:
            Dict: {
                "status": "SUCCESS" | "PARTIAL_SUCCESS" | "ALL_FAILED" | "ROLLBACK",
                "applied_changes": [...],
                "failed_changes": [...],
                "validation_results": {...},
                "files_modified": [...],
                "post_edit_context": {...},  # 新增: post-edit 反馈
            }
        """
        if not structural_changes:
            return {
                "status": "SUCCESS",
                "applied_changes": [],
                "failed_changes": [],
                "validation_results": {},
                "files_modified": [],
                "post_edit_context": {},
            }

        logger.info(f"Applying {len(structural_changes)} structural changes...")

        # Step 1: 本地文件快照
        snapshot = self._create_local_snapshot()
        if not snapshot["ok"]:
            return {
                "status": "ALL_FAILED",
                "failed_changes": structural_changes,
                "error": f"Local snapshot failed: {snapshot.get('error', 'unknown')}",
            }
        self._current_snapshot_id = snapshot["snapshot_id"]
        self._backup_branch = snapshot["snapshot_id"]

        applied = []
        failed = []
        files_modified = set()
        post_edit_context = {}

        # Step 2: 逐个应用修改
        for change in structural_changes:
            result = self._apply_single_change(change)
            if result["status"] == "APPLIED":
                applied.append({"change": change, "result": result})
                files_modified.add(result.get("file_path", ""))
                # 收集 post-edit 反馈
                if result.get("post_edit_window"):
                    post_edit_context[result["file_path"]] = result["post_edit_window"]
            else:
                failed.append({"change": change, "error": result.get("error", "unknown")})
                logger.warning(f"Change failed: {result.get('error', 'unknown')}")

        # 没有一个修改成功
        if not applied and failed:
            logger.warning("No structural changes were successfully applied")
            return {
                "status": "ALL_FAILED",
                "applied_changes": [],
                "failed_changes": failed,
                "validation_results": {
                    "all_passed": False, "results": {},
                    "errors": ["No structural changes were applied"],
                    "summary": "No files modified",
                },
                "files_modified": [],
                "post_edit_context": {},
            }

        # Step 3: 校验所有修改后的文件
        validation = self._validate_all_modified_files(list(files_modified))

        if not validation["all_passed"]:
            logger.warning(f"Validation failed: {validation['errors']}")
            self._local_rollback(self._current_snapshot_id)
            self._current_snapshot_id = None
            self._backup_branch = None
            return {
                "status": "ROLLBACK",
                "applied_changes": applied,
                "failed_changes": failed + [{
                    "change": {},
                    "error": f"Validation failed: {validation['errors']}",
                }],
                "validation_results": validation,
                "files_modified": [],
                "rollback_reason": validation.get("summary", ""),
                "post_edit_context": {},
            }

        # Step 4: 成功 → 保留修改
        logger.info(
            f"Structural changes finished: applied={len(applied)}, "
            f"failed={len(failed)}, files_modified={len(files_modified)}"
        )
        self._applied_changes.extend(applied)

        return {
            "status": "SUCCESS" if not failed else "PARTIAL_SUCCESS",
            "applied_changes": applied,
            "failed_changes": failed,
            "validation_results": validation,
            "files_modified": list(files_modified),
            "post_edit_context": post_edit_context,
        }

    # ════════════════════════════════════════
    # 应用单个修改 — 核心逻辑
    # ════════════════════════════════════════

    def _apply_single_change(self, change: Dict) -> Dict:
        """
        应用单个结构修改

        自动检测输入格式:
        - 新格式: change["edits"] 存在 → 使用 SEARCH/REPLACE
        - 旧格式: change["new_code"] 存在 → 先转换为 SEARCH/REPLACE 再处理
        """
        target_file = change.get("target_file", "")
        if not target_file:
            return {"status": "FAILED", "error": "Missing target_file"}

        file_path = self._resolve_file_path(target_file)
        if not file_path:
            return {"status": "FAILED", "error": f"Cannot find file: {target_file}"}

        # 读取当前文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return {"status": "FAILED", "error": f"Cannot read file {file_path}: {e}"}

        # ── 自动检测格式 ──
        edits = change.get("edits", [])
        if not edits:
            # 旧格式 → 转换为 SEARCH/REPLACE
            edits = self._convert_legacy_format(change, original_content, file_path)
            if not edits:
                return {
                    "status": "FAILED",
                    "error": f"Cannot convert legacy format for {target_file}",
                }

        # ── 应用所有编辑块 ──
        content = original_content
        edit_results = []
        for edit_idx, edit in enumerate(edits):
            search_text = edit.get("search", "")
            replace_text = edit.get("replace", "")
            if not search_text:
                # 空 search → 纯插入 (在文件末尾)
                content = content + "\n\n" + replace_text
                edit_results.append({
                    "edit_idx": edit_idx,
                    "method": "append",
                    "success": True,
                })
                continue

            # 尝试 SEARCH/REPLACE
            result_content, method = self._str_replace(content, search_text, replace_text)
            if result_content is not None:
                content = result_content
                edit_results.append({
                    "edit_idx": edit_idx,
                    "method": method,
                    "success": True,
                })
            else:
                edit_results.append({
                    "edit_idx": edit_idx,
                    "method": "failed",
                    "success": False,
                    "search_text_preview": search_text[:80],
                })

        # 检查是否所有编辑都失败
        successful_edits = [r for r in edit_results if r["success"]]
        if not successful_edits and edit_results:
            failed_methods = [r["method"] for r in edit_results if not r["success"]]
            return {
                "status": "FAILED",
                "error": f"All SEARCH/REPLACE edits failed for {target_file} "
                         f"(methods tried: {failed_methods})",
                "edit_results": edit_results,
            }

        # ── 写入前语法检查 ──
        syntax_check = self._check_syntax(content, file_path)
        if not syntax_check["passed"]:
            return {
                "status": "FAILED",
                "error": f"Generated code has syntax error: {syntax_check.get('error', '')}",
                "edit_results": edit_results,
            }

        # ── 写入文件 ──
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return {"status": "FAILED", "error": f"Cannot write file {file_path}: {e}"}

        # ── 生成 post-edit 反馈 ──
        post_edit_window = self._generate_post_edit_context(
            original_content, content, file_path, context_lines=5
        )

        logger.info(f"Applied {len(successful_edits)} edits to {file_path}")
        return {
            "status": "APPLIED",
            "file_path": file_path,
            "edit_results": edit_results,
            "post_edit_window": post_edit_window,
        }

    # ════════════════════════════════════════
    # SEARCH/REPLACE 核心方法 (受 Aider 启发)
    # ════════════════════════════════════════

    def _str_replace(self, content: str, search_text: str, replace_text: str) -> Tuple[Optional[str], str]:
        """
        在 content 中查找 search_text 并替换为 replace_text

        三级匹配策略:
        1. 精确匹配 — 直接字符串查找
        2. 去空白匹配 — 忽略空行/多余空格差异
        3. 模糊匹配 — difflib.SequenceMatcher, 最低相似度 0.80

        Returns:
            (new_content, method_used) 或 (None, "failed")
        """
        # ── Level 1: 精确匹配 ──
        if search_text in content:
            new_content = content.replace(search_text, replace_text, 1)
            logger.info("SEARCH/REPLACE: exact match succeeded")
            return new_content, "exact_match"

        # ── Level 2: 去空白匹配 ──
        # LLM 经常多写或少写空行, 这是最常见的匹配失败原因
        result = self._strip_whitespace_match(content, search_text, replace_text)
        if result is not None:
            logger.info("SEARCH/REPLACE: whitespace-normalized match succeeded")
            return result, "whitespace_match"

        # ── Level 3: 模糊匹配 ──
        # 容忍 LLM 输出中少量行偏差 (多/少注释、行顺序微调等)
        result = self._fuzzy_match(content, search_text, replace_text)
        if result is not None:
            logger.info("SEARCH/REPLACE: fuzzy match succeeded")
            return result, "fuzzy_match"

        logger.warning(f"SEARCH/REPLACE: all 3 levels failed for search text "
                       f"(first 80 chars): {search_text[:80]}")
        return None, "failed"

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        规范化空白: 去除连续空行, 去除行尾空格, 去除首尾空行
        用于 Level 2 去空白匹配
        """
        lines = text.split('\n')
        # 去除行尾空格
        lines = [line.rstrip() for line in lines]
        # 去除首尾空行
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        # 压缩连续空行为单个空行
        result = []
        prev_empty = False
        for line in lines:
            if not line.strip():
                if not prev_empty:
                    result.append('')
                prev_empty = True
            else:
                result.append(line)
                prev_empty = False
        return '\n'.join(result)

    def _strip_whitespace_match(self, content: str, search_text: str, replace_text: str) -> Optional[str]:
        """
        Level 2: 去空白后匹配

        将 content 和 search_text 都做空白规范化, 然后在规范化后的 content 中查找.
        找到后, 映射回原始 content 中的位置, 执行替换.
        """
        norm_search = self._normalize_whitespace(search_text)
        norm_content = self._normalize_whitespace(content)

        if norm_search not in norm_content:
            return None

        # 找到匹配位置 — 需要映射回原始 content
        # 策略: 用行号映射. 规范化后的行号 → 原始行号
        norm_content_lines = norm_content.split('\n')
        norm_search_lines = norm_search.split('\n')

        # 找到 norm_search 在 norm_content 中开始的行号
        match_start_line = None
        for i in range(len(norm_content_lines) - len(norm_search_lines) + 1):
            segment = '\n'.join(norm_content_lines[i:i + len(norm_search_lines)])
            if segment == norm_search:
                match_start_line = i
                break

        if match_start_line is None:
            return None

        # 映射回原始行号
        # 构建 norm_line_idx → original_line_idx 映射
        original_lines = content.split('\n')
        norm_to_orig = self._build_line_mapping(original_lines)

        orig_start = norm_to_orig.get(match_start_line)
        # end: match_start_line + len(norm_search_lines) - 1 在 norm 中的行号
        norm_end_line = match_start_line + len(norm_search_lines) - 1
        orig_end = norm_to_orig.get(norm_end_line)

        if orig_start is None or orig_end is None:
            return None

        # 执行替换: 替换 original_lines[orig_start:orig_end+1] 为 replace_text
        new_lines = original_lines[:orig_start] + replace_text.split('\n') + original_lines[orig_end + 1:]
        return '\n'.join(new_lines)

    @staticmethod
    def _build_line_mapping(original_lines: List[str]) -> Dict[int, int]:
        """
        构建规范化行号 → 原始行号的映射

        规范化会压缩连续空行和去除首尾空行, 所以行号会偏移.
        """
        mapping = {}
        norm_idx = 0
        prev_empty = False
        leading_empty_done = False

        for orig_idx, line in enumerate(original_lines):
            stripped = line.rstrip().strip()
            if not stripped:
                # 空行
                if not leading_empty_done:
                    # 首部空行 → 跳过, 不映射
                    continue
                if prev_empty:
                    # 连续空行 → 跳过, 不映射
                    continue
                # 单个空行 → 映射
                mapping[norm_idx] = orig_idx
                norm_idx += 1
                prev_empty = True
            else:
                leading_empty_done = True
                mapping[norm_idx] = orig_idx
                norm_idx += 1
                prev_empty = False

        return mapping

    def _fuzzy_match(self, content: str, search_text: str, replace_text: str) -> Optional[str]:
        """
        Level 3: 模糊匹配 — 在 content 中找与 search_text 最相似的代码片段

        使用 difflib.SequenceMatcher 做行级比较.
        只在原始行和规范行都找不到时才启用 (最宽松的匹配级别).

        算法:
        1. 将 content 和 search_text 都按行分割
        2. 用滑动窗口在 content 中找与 search_text 最相似的片段
        3. 相似度 >= fuzzy_min_similarity (默认 0.80) 才接受匹配
        """
        content_lines = content.split('\n')
        search_lines = search_text.split('\n')

        if not search_lines or not content_lines:
            return None

        # 去除 search_lines 中的纯空行 (LLM 经常多写空行)
        search_lines_stripped = [l for l in search_lines if l.strip()]
        if not search_lines_stripped:
            return None

        n_search = len(search_lines_stripped)
        n_content = len(content_lines)

        # 滑动窗口大小: 从 n_search 向上下浮动 ±30%
        min_window = max(1, n_search - max(1, int(n_search * 0.3)))
        max_window = min(n_content, n_search + max(1, int(n_search * 0.3)))

        best_ratio = 0.0
        best_start = None
        best_end = None

        for window_size in range(min_window, max_window + 1):
            for start in range(n_content - window_size + 1):
                segment = content_lines[start:start + window_size]
                # 去除 segment 中的纯空行用于比较
                segment_stripped = [l for l in segment if l.strip()]
                if not segment_stripped:
                    continue

                # 计算行级相似度
                matcher = difflib.SequenceMatcher(None, segment_stripped, search_lines_stripped)
                ratio = matcher.ratio()

                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = start
                    best_end = start + window_size

        if best_ratio < self.fuzzy_min_similarity or best_start is None:
            logger.info(f"Fuzzy match: best ratio {best_ratio:.2f} < threshold "
                        f"{self.fuzzy_min_similarity}")
            return None

        logger.info(f"Fuzzy match: found segment at lines {best_start+1}-{best_end+1} "
                    f"with similarity {best_ratio:.2f}")

        # 替换匹配的行范围
        new_lines = content_lines[:best_start] + replace_text.split('\n') + content_lines[best_end:]
        return '\n'.join(new_lines)

    # ════════════════════════════════════════
    # 旧格式转换 (兼容性)
    # ════════════════════════════════════════

    def _convert_legacy_format(self, change: Dict, original_content: str,
                                file_path: str) -> List[Dict]:
        """
        将旧格式 (new_code + insert_position) 转换为 SEARCH/REPLACE 格式

        旧格式:
          {"new_code": "...", "insert_position": "replace_function",
           "target_class_or_function": "SelfAttention.forward"}

        转换策略:
        1. 用 AST 在 original_content 中找到目标代码片段
        2. 将找到的代码片段作为 search_text, new_code 作为 replace_text
        3. 如果找不到 → 尝试 whole_file 替换 (如果 new_code 是完整文件)

        注意: 这是一种降级策略 — 旧格式的可靠性远低于 SEARCH/REPLACE.
        建议所有 prompt 都迁移到新格式后, 删除此方法.
        """
        new_code = change.get("new_code", "")
        insert_position = change.get("insert_position", "replace_function")
        target_name = change.get("target_class_or_function", "")

        if not new_code:
            return []

        # 清理 new_code
        new_code = self.clean_new_code(new_code)

        # ── 尝试在文件中找到目标代码片段 ──
        search_text = self._find_target_code(original_content, target_name, insert_position)

        if search_text:
            # 成功找到 → 构建 SEARCH/REPLACE
            return [{"search": search_text, "replace": new_code}]

        # ── 找不到 → 尝试 whole_file 模式 ──
        # 如果 new_code 长度 >= 文件原始长度的 50%, 可能是完整文件替换
        if len(new_code) >= len(original_content) * 0.5:
            logger.info(f"Legacy format: falling back to whole_file for {target_name}")
            return [{"search": original_content, "replace": new_code}]

        # ── 最后兜底: 纯追加 ──
        logger.info(f"Legacy format: falling back to append for {target_name}")
        return [{"search": "", "replace": new_code}]  # 空 search → 追加到文件末尾

    def _find_target_code(self, content: str, target_name: str,
                          insert_position: str) -> Optional[str]:
        """
        在文件内容中查找 target_name 对应的代码片段

        用于旧格式转换时, 构建 SEARCH/REPLACE 的 search 部分.
        """
        if not target_name:
            return None

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._find_target_regex(content, target_name)

        if "." in target_name:
            class_name, method_name = target_name.split(".", 1)
            method_name = method_name.split("(")[0]  # 剥离参数签名

            # 查找类中的方法
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == method_name:
                                lines = content.split('\n')
                                return '\n'.join(lines[item.lineno - 1:item.end_lineno])
            # 类找到了但方法没找到 → 返回整个类
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    lines = content.split('\n')
                    return '\n'.join(lines[node.lineno - 1:node.end_lineno])
        else:
            # 查找顶层函数/类
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == target_name:
                        lines = content.split('\n')
                        return '\n'.join(lines[node.lineno - 1:node.end_lineno])
                if isinstance(node, ast.ClassDef):
                    if node.name == target_name:
                        lines = content.split('\n')
                        return '\n'.join(lines[node.lineno - 1:node.end_lineno])

        # AST 查找失败 → regex 后备
        return self._find_target_regex(content, target_name)

    @staticmethod
    def _find_target_regex(content: str, target_name: str) -> Optional[str]:
        """
        用 regex 在文件内容中查找目标定义的代码片段

        简化版 regex — 只处理顶层 def/class, 不处理嵌套.
        """
        if "." in target_name:
            # "ClassName.method_name" → 查找类定义
            class_name = target_name.split(".", 1)[0]
            pattern = rf'^class\s+{class_name}\s*\([^)]*\)\s*:'
            match = re.search(pattern, content, re.MULTILINE)
            if not match:
                return None
            # 找到类定义头 → 用缩进确定范围
            lines = content.split('\n')
            start_idx = content[:match.start()].count('\n')
            # 找到类的结束
            class_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            end_idx = len(lines)
            for idx in range(start_idx + 1, len(lines)):
                stripped = lines[idx].strip()
                if not stripped:
                    continue
                line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                if line_indent <= class_indent:
                    end_idx = idx
                    break
            return '\n'.join(lines[start_idx:end_idx])
        else:
            # 顶层函数
            pattern = rf'^def\s+{target_name}\s*\('
            match = re.search(pattern, content, re.MULTILINE)
            if not match:
                return None
            lines = content.split('\n')
            start_idx = content[:match.start()].count('\n')
            func_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            end_idx = len(lines)
            for idx in range(start_idx + 1, len(lines)):
                stripped = lines[idx].strip()
                if not stripped:
                    continue
                line_indent = len(lines[idx]) - len(lines[idx].lstrip())
                if line_indent <= func_indent:
                    end_idx = idx
                    break
            return '\n'.join(lines[start_idx:end_idx])

    # ════════════════════════════════════════
    # Post-Edit 反馈 (受 SWE-Agent/OpenHands 启发)
    # ════════════════════════════════════════

    def _generate_post_edit_context(self, original_content: str, new_content: str,
                                      file_path: str, context_lines: int = 5) -> Optional[str]:
        """
        生成 post-edit 反馈 — 展示修改区域前后 N 行的上下文

        让 LLM 检查修改是否正确, 发现重复代码/缩进错误等问题.
        这是 SWE-Agent 和 OpenHands 证明有效的关键机制.

        Args:
            original_content: 修改前的文件内容
            new_content: 修改后的文件内容
            file_path: 文件路径
            context_lines: 上下文行数

        Returns:
            str: 包含行号的修改区域上下文, 或 None
        """
        # 用 diff 找到修改的行范围
        orig_lines = original_content.split('\n')
        new_lines = new_content.split('\n')

        diff = difflib.SequenceMatcher(None, orig_lines, new_lines)
        changes = diff.get_opcodes()

        # 找到所有修改行的范围
        modified_ranges = []
        for tag, i1, i2, j1, j2 in changes:
            if tag != 'equal':
                modified_ranges.append((j1, j2))

        if not modified_ranges:
            return None

        # 合合连续的修改范围, 加上上下文
        min_line = max(0, modified_ranges[0][0] - context_lines)
        max_line = min(len(new_lines), modified_ranges[-1][1] + context_lines)

        # 展示带行号的上下文
        window_lines = []
        for idx in range(min_line, max_line):
            marker = "►" if any(j1 <= idx < j2 for _, j1, j2 in modified_ranges) else " "
            window_lines.append(f"{idx + 1:4d}{marker}| {new_lines[idx]}")

        filename = os.path.basename(file_path)
        context_str = f"=== Post-edit context for {filename} ===\n"
        context_str += '\n'.join(window_lines)
        context_str += f"\n=== End of context (total {len(new_lines)} lines) ==="

        return context_str

    # ════════════════════════════════════════
    # 校验 — 语法 + 执行验证
    # ════════════════════════════════════════

    def _validate_all_modified_files(self, file_paths: List[str]) -> Dict:
        """
        校验所有修改后的文件

        三级验证:
        1. Python 语法 (ast.parse)
        2. Import 完整性 (subprocess import check)
        3. 关键符号存在性 (class/function 定义检查)
        """
        results = {}
        all_passed = True
        errors = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                results[file_path] = {"passed": False, "error": "File not found"}
                all_passed = False
                errors.append(f"{file_path}: not found")
                continue

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查 1: Python 语法
            syntax_result = self._check_syntax(content, file_path)
            if not syntax_result["passed"]:
                results[file_path] = syntax_result
                all_passed = False
                errors.append(f"{file_path}: syntax error — {syntax_result.get('error', '')}")
                continue

            # 检查 2: 执行验证 (import check)
            exec_result = self._validate_by_execution(file_path)
            if not exec_result["passed"]:
                results[file_path] = exec_result
                all_passed = False
                errors.append(f"{file_path}: import error — {exec_result.get('error', '')}")
                continue

            # 检查 3: 关键符号存在性
            symbol_result = self._check_critical_symbols(content, file_path)
            if not symbol_result["passed"]:
                results[file_path] = symbol_result
                all_passed = False
                errors.append(f"{file_path}: missing critical symbols — {symbol_result.get('missing', [])}")
                continue

            results[file_path] = {
                "passed": True,
                "syntax_check": syntax_result,
                "exec_check": exec_result,
                "symbol_check": symbol_result,
            }

        summary = " | ".join([
            f"{os.path.basename(k)}: {'✓' if v['passed'] else '✗'}"
            for k, v in results.items()
        ]) if results else "No files to validate"

        return {
            "all_passed": all_passed,
            "results": results,
            "errors": errors,
            "summary": summary,
        }

    @staticmethod
    def _check_syntax(content: str, file_path: str) -> Dict:
        """Python 语法检查 (ast.parse)"""
        try:
            ast.parse(content)
            return {"passed": True, "detail": "AST parse OK"}
        except SyntaxError as e:
            return {
                "passed": False,
                "error": f"SyntaxError at line {e.lineno}: {e.msg}",
                "line": e.lineno,
            }

    def _validate_by_execution(self, file_path: str) -> Dict:
        """
        执行验证: 尝试 import 修改后的模块

        在 subprocess 中运行 python -c "import module",
        检查模块能否被成功加载 (不抛 ImportError/NameError 等).

        这是比 ast.parse 更严格的验证 — 语法正确不代表代码能跑!
        """
        filename = os.path.basename(file_path)
        module_name = filename.replace('.py', '')

        # 构建项目根目录和 Recmodel 子目录作为 Python path
        normalized_root = os.path.normpath(self.project_root)
        recmodel_dir = os.path.join(normalized_root, "Recmodel")
        python_path = f"{normalized_root}:{recmodel_dir}"

        # 尝试 import
        import_cmd = (
            f"import sys; "
            f"sys.path.insert(0, '{normalized_root}'); "
            f"sys.path.insert(0, '{recmodel_dir}'); "
            f"import {module_name}; "
            f"print('OK')"
        )

        try:
            result = subprocess.run(
                ["python", "-c", import_cmd],
                capture_output=True, text=True, timeout=10,
                env={**os.environ, "PYTHONPATH": python_path},
            )
            if result.returncode == 0 and "OK" in result.stdout:
                return {"passed": True, "detail": f"import {module_name} OK"}
            else:
                error_msg = result.stderr.strip() or result.stdout.strip()
                # 只报告前 500 字符的错误
                error_msg = error_msg[:500]
                return {"passed": False, "error": f"import {module_name} failed: {error_msg}"}
        except subprocess.TimeoutExpired:
            return {"passed": False, "error": f"import {module_name} timed out (10s)"}
        except Exception as e:
            return {"passed": False, "error": f"import check exception: {e}"}

    def _check_critical_symbols(self, content: str, file_path: str) -> Dict:
        """检查关键类/函数是否仍然存在"""
        filename = os.path.basename(file_path)
        critical_symbols = {
            "models.py": ["class SASRec", "class SRModel", "def finetune", "def add_position_embedding"],
            "modules.py": ["class SelfAttention", "class Intermediate", "class EncoderLayer",
                           "class Encoder", "class LayerNorm"],
            "trainers.py": ["def _get_loss", "def acc_metric"],
        }
        required = critical_symbols.get(filename, [])
        missing = [sym for sym in required if sym not in content]
        if missing:
            return {"passed": False, "missing": missing}
        return {"passed": True, "found": required}

    # ════════════════════════════════════════
    # 文件路径解析
    # ════════════════════════════════════════

    def _resolve_file_path(self, target_file: str) -> Optional[str]:
        """将逻辑文件名解析为实际文件路径"""
        normalized_root = os.path.normpath(self.project_root)
        candidates = [
            os.path.join(normalized_root, target_file),
            os.path.join(normalized_root, "Recmodel", target_file),
        ]
        if self.adapter:
            source_map = self.adapter.SOURCE_FILE_MAP
            if target_file in source_map:
                rel_path = source_map[target_file]
                candidates.extend([
                    os.path.join(normalized_root, rel_path),
                    os.path.join(normalized_root, "Recmodel", rel_path),
                ])
        for path in candidates:
            if os.path.exists(path):
                return os.path.normpath(path)
        logger.warning(f"Cannot resolve file path for {target_file}, tried: {candidates}")
        return None

    # ════════════════════════════════════════
    # 本地快照系统 (替代 Git)
    # ════════════════════════════════════════

    def _create_local_snapshot(self) -> Dict:
        """创建本地文件快照"""
        snapshot_id = f"snap_{int(time.time())}"
        snapshot_subdir = self._snapshot_dir / snapshot_id
        snapshot_subdir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        hashes = {}

        for file_key in self._source_files:
            src_path = self._resolve_file_path(file_key)
            if src_path and os.path.exists(src_path):
                dst_path = snapshot_subdir / file_key
                shutil.copy2(src_path, dst_path)
                saved_files.append(file_key)
                with open(src_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                hashes[file_key] = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]

        # 保存快照元数据
        meta = {
            "snapshot_id": snapshot_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_files": saved_files,
            "file_hashes": hashes,
        }
        with open(snapshot_subdir / "_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        self._pre_snapshot_hashes = hashes
        logger.info(f"Local snapshot created: {snapshot_id}, {len(saved_files)} files")
        return {"ok": True, "snapshot_id": snapshot_id, "saved_files": saved_files}

    def _local_rollback(self, snapshot_id: str) -> Dict:
        """从本地快照恢复源码文件"""
        snapshot_subdir = self._snapshot_dir / snapshot_id
        if not snapshot_subdir.exists():
            return {"ok": False, "error": f"Snapshot {snapshot_id} not found"}

        with open(snapshot_subdir / "_meta.json", 'r', encoding='utf-8') as f:
            meta = json.load(f)

        restored_files = []
        verification_ok = True

        for file_key in meta.get("saved_files", []):
            snapshot_path = snapshot_subdir / file_key
            if not snapshot_path.exists():
                continue
            target_path = self._resolve_file_path(file_key)
            if target_path:
                shutil.copy2(snapshot_path, target_path)
                restored_files.append(file_key)
                # 校验 hash
                expected_hash = meta.get("file_hashes", {}).get(file_key)
                if expected_hash:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        current_hash = hashlib.md5(f.read().encode('utf-8')).hexdigest()[:12]
                    if current_hash != expected_hash:
                        verification_ok = False

        logger.info(f"Local rollback completed: {len(restored_files)} files restored")
        return {
            "ok": True,
            "restored_files": restored_files,
            "verification_ok": verification_ok,
            "snapshot_id": snapshot_id,
        }

    def rollback_last_changes(self) -> Dict:
        """回滚最近一次的结构修改"""
        if self._current_snapshot_id:
            result = self._local_rollback(self._current_snapshot_id)
            self._current_snapshot_id = None
            self._backup_branch = None
            self._applied_changes = []
            return result
        return {"ok": False, "error": "No snapshot available for rollback"}

    # ════════════════════════════════════════
    # 工具方法
    # ════════════════════════════════════════

    def get_applied_changes_summary(self) -> str:
        """获取已应用的修改摘要"""
        if not self._applied_changes:
            return "No structural changes applied yet"
        parts = []
        for entry in self._applied_changes:
            change = entry.get("change", {})
            desc = change.get("description", "?")[:80]
            target = change.get("target_class_or_function", "?")
            file = change.get("target_file", "?")
            parts.append(f"  [{file}] {target}: {desc}")
        return "\n".join(parts)

    @staticmethod
    def clean_new_code(raw_code: str) -> str:
        """
        清理 LLM 输出的代码 — 移除 markdown 包裹和解释性文字

        注意: 此方法仅用于旧格式兼容. 新格式 (SEARCH/REPLACE) 不需要此处理.
        """
        code = raw_code
        # 移除 markdown 代码块标记
        code = re.sub(r'^```(?:python)?\s*\n?', '', code)
        code = re.sub(r'\n?```\s*$', '', code)

        # 移除解释性前缀
        lines = code.split('\n')
        clean_lines = []
        code_started = False
        for line in lines:
            stripped = line.strip()
            if not code_started:
                if stripped.startswith('def ') or stripped.startswith('class ') or \
                   stripped.startswith('@') or stripped.startswith('import ') or \
                   stripped.startswith('from ') or stripped.startswith('#') or \
                   stripped.startswith('"""') or stripped.startswith("'''") or \
                   stripped == '':
                    code_started = True
                else:
                    continue
            if code_started:
                clean_lines.append(line)
        return '\n'.join(clean_lines)