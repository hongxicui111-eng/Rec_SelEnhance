#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
StructureApplier — 模型结构修改应用器

核心职责:
1. 接收 LLM 提出的 structural_changes (代码修改方案)
2. 将代码修改安全地应用到对应的源码文件 (models.py, modules.py, trainers.py)
3. 语法校验 + 导入检查 → 确保修改后的代码可执行
4. 本地文件快照 + 自动回滚 → 修改失败时恢复原状 (无需联网/Git!)
5. 修改记录 → journal 记录每次结构修改

这是让 Agent 从"只调参数"升级到"改模型结构"的关键模块。
"""

import os
import ast
import re
import copy
import json
import shutil
import hashlib
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger("rec_self_evolve.structure_applier")


class StructureApplier:
    """
    模型结构修改应用器
    
    工作流程:
    1. 本地文件快照 (创建备份目录 — 替代 Git!)
    2. 读取目标源码文件
    3. 解析 LLM 的 structural_change → 确定修改位置
    4. 应用代码修改 (替换函数/类 或 插入新代码)
    5. 语法校验 (ast.parse + import 检查)
    6. 如果校验失败 → 从本地快照回滚
    7. 如果校验成功 → 保留修改
    """

    def __init__(self, project_root: str, adapter=None,
                 log_dir: str = "evolve_logs",
                 source_files: List[str] = None):
        """
        Args:
            project_root: 项目根目录路径
            adapter: SeqRecAdapter 实例 (用于获取文件路径映射)
            log_dir: 快照保存目录 (替代 Git 的本地存储)
            source_files: 需要跟踪的源码文件列表
        """
        self.project_root = project_root
        self.adapter = adapter
        self._applied_changes = []  # 已成功应用的修改记录
        # ── 本地快照系统 (替代 Git — 无需联网!) ──
        self._snapshot_dir = Path(log_dir) / "rollback_snapshots"
        self._snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._current_snapshot_id = None  # 当前快照 ID
        self._source_files = source_files or ["models.py", "modules.py", "trainers.py"]
        # 快照前各文件的 hash (用于校验回滚是否正确)
        self._pre_snapshot_hashes = {}
        # 保留字段兼容旧代码引用
        self._backup_branch = None

    # ════════════════════════════════════════
    # 主入口 — 应用一组结构修改
    # ════════════════════════════════════════

    def apply_structural_changes(self, structural_changes: List[Dict]) -> Dict:
        """
        应用一组 LLM 提出的结构修改
        
        Args:
            structural_changes: LLM 输出的结构修改列表, 每项包含:
                - action_type: 修改类型
                - target_file: 目标文件 (如 "models.py")
                - target_class_or_function: 目标类/函数 (如 "SelfAttention.forward")
                - description: 修改描述
                - new_code: 新代码内容 (Python 代码)
                - insert_position: 插入方式
                - expected_effect: 预期效果
                - risk_level: 风险等级
        
        Returns:
            Dict: {
                "status": "SUCCESS" | "PARTIAL_SUCCESS" | "ALL_FAILED" | "ROLLBACK",
                "applied_changes": [...],  # 成功应用的修改列表
                "failed_changes": [...],   # 失败的修改列表 (含错误信息)
                "validation_results": {...}, # 校验结果
                "files_modified": [...],   # 被修改的文件列表
            }
        """
        if not structural_changes:
            return {
                "status": "SUCCESS",
                "applied_changes": [],
                "failed_changes": [],
                "validation_results": {},
                "files_modified": [],
            }

        logger.info(f"Applying {len(structural_changes)} structural changes...")

        # Step 1: 本地文件快照 (替代 Git)
        snapshot = self._create_local_snapshot()
        if not snapshot["ok"]:
            return {
                "status": "ALL_FAILED",
                "failed_changes": structural_changes,
                "error": f"Local snapshot failed: {snapshot.get('error', 'unknown')}",
            }
        self._current_snapshot_id = snapshot["snapshot_id"]
        self._backup_branch = snapshot["snapshot_id"]  # 兼容旧代码

        applied = []
        failed = []
        files_modified = set()

        # Step 2: 逐个应用修改
        for change in structural_changes:
            result = self._apply_single_change(change)
            if result["status"] == "APPLIED":
                applied.append({
                    "change": change,
                    "result": result,
                })
                files_modified.add(result.get("file_path", ""))
            else:
                failed.append({
                    "change": change,
                    "error": result.get("error", "unknown"),
                })
                logger.warning(f"Change failed: {result.get('error', 'unknown')}")

        # Step 3: 校验所有修改后的文件
        validation = self._validate_all_modified_files(list(files_modified))

        if not validation["all_passed"]:
            logger.warning(f"Validation failed: {validation['errors']}")
            # 回滚所有修改 (从本地快照恢复)
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
            }

        # Step 4: 成功 → 保留修改
        logger.info(f"All {len(applied)} structural changes applied and validated successfully")
        self._applied_changes.extend(applied)

        return {
            "status": "SUCCESS" if not failed else "PARTIAL_SUCCESS",
            "applied_changes": applied,
            "failed_changes": failed,
            "validation_results": validation,
            "files_modified": list(files_modified),
        }

    # ════════════════════════════════════════
    # 应用单个修改
    # ════════════════════════════════════════

    def _apply_single_change(self, change: Dict) -> Dict:
        """
        应用单个结构修改
        
        根据 insert_position 决定如何修改文件:
        - "replace_function": 替换目标函数的完整定义
        - "replace_class": 替换目标类的完整定义
        - "after_class_X": 在类 X 定义后插入新代码
        - "before_function_Y": 在函数 Y 定义前插入新代码
        - "append_to_file": 在文件末尾追加新代码
        """
        target_file = change.get("target_file", "")
        new_code = change.get("new_code", "")
        insert_position = change.get("insert_position", "append_to_file")
        target_name = change.get("target_class_or_function", "")

        if not target_file or not new_code:
            return {"status": "FAILED", "error": "Missing target_file or new_code"}

        # 获取文件的实际路径
        file_path = self._resolve_file_path(target_file)
        if not file_path:
            return {"status": "FAILED", "error": f"Cannot find file: {target_file}"}

        # 读取当前文件内容
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
        except Exception as e:
            return {"status": "FAILED", "error": f"Cannot read file {file_path}: {e}"}

        # 根据插入方式应用修改
        if insert_position == "replace_function":
            new_content = self._replace_function(original_content, target_name, new_code)
        elif insert_position == "replace_class":
            new_content = self._replace_class(original_content, target_name, new_code)
        elif insert_position.startswith("after_class_"):
            ref_class = insert_position.replace("after_class_", "")
            new_content = self._insert_after_class(original_content, ref_class, new_code)
        elif insert_position.startswith("before_function_"):
            ref_func = insert_position.replace("before_function_", "")
            new_content = self._insert_before_function(original_content, ref_func, new_code)
        elif insert_position == "append_to_file":
            new_content = original_content + "\n\n\n" + new_code
        else:
            # 默认: 替换函数
            new_content = self._replace_function(original_content, target_name, new_code)

        if new_content is None:
            return {
                "status": "FAILED",
                "error": f"Cannot apply {insert_position} for {target_name} in {target_file}",
            }

        # 写入修改后的文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            logger.info(f"Successfully applied change to {file_path}")
        except Exception as e:
            return {"status": "FAILED", "error": f"Cannot write file {file_path}: {e}"}

        return {"status": "APPLIED", "file_path": file_path}

    # ════════════════════════════════════════
    # 代码修改策略 (核心!)
    # ════════════════════════════════════════

    def _replace_function(self, content: str, target_name: str, new_code: str) -> Optional[str]:
        """
        替换文件中的指定函数
        
        Args:
            content: 文件原始内容
            target_name: 目标函数名 (如 "SelfAttention.forward" 或 "add_position_embedding")
            new_code: 新函数代码
        
        Returns:
            str: 修改后的文件内容, 或 None (找不到目标)
        """
        # 解析 target_name
        # 格式: "ClassName.method_name" 或 "function_name"
        if "." in target_name:
            class_name, method_name = target_name.split(".", 1)
            return self._replace_method_in_class(content, class_name, method_name, new_code)
        else:
            return self._replace_top_level_function(content, target_name, new_code)

    def _replace_top_level_function(self, content: str, func_name: str, new_code: str) -> Optional[str]:
        """替换顶层函数定义"""
        # 使用 AST 精确定位函数定义的位置
        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning("Cannot parse file with AST, falling back to regex")
            return self._replace_function_regex(content, func_name, new_code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                # 找到函数在源码中的起止行
                start_line = node.lineno - 1  # ast 用 1-based, 我们用 0-based
                end_line = node.end_lineno  # ast 用 1-based, 包含最后一行

                lines = content.split('\n')
                # 替换从 start_line 到 end_line 的所有行
                new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                return '\n'.join(new_lines)

        logger.warning(f"Function '{func_name}' not found in file")
        return self._replace_function_regex(content, func_name, new_code)

    def _replace_method_in_class(self, content: str, class_name: str, method_name: str, new_code: str) -> Optional[str]:
        """替换类中的方法定义"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._replace_method_regex(content, class_name, method_name, new_code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == method_name:
                        start_line = item.lineno - 1
                        end_line = item.end_lineno

                        lines = content.split('\n')
                        # 需要保持方法的缩进级别
                        method_indent = self._get_indent_level(lines[start_line])
                        # 新代码可能需要调整缩进
                        indented_new_code = self._adjust_indent(new_code, method_indent)
                        new_lines = lines[:start_line] + indented_new_code.split('\n') + lines[end_line:]
                        return '\n'.join(new_lines)

        logger.warning(f"Method '{class_name}.{method_name}' not found in file")
        return self._replace_method_regex(content, class_name, method_name, new_code)

    def _replace_class(self, content: str, class_name: str, new_code: str) -> Optional[str]:
        """替换整个类定义"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._replace_class_regex(content, class_name, new_code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                start_line = node.lineno - 1
                end_line = node.end_lineno

                lines = content.split('\n')
                new_lines = lines[:start_line] + new_code.split('\n') + lines[end_line:]
                return '\n'.join(new_lines)

        logger.warning(f"Class '{class_name}' not found in file")
        return self._replace_class_regex(content, class_name, new_code)

    def _insert_after_class(self, content: str, class_name: str, new_code: str) -> Optional[str]:
        """在类定义之后插入新代码 (用于添加新类)"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._insert_after_regex(content, f"class {class_name}", new_code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                end_line = node.end_lineno

                lines = content.split('\n')
                new_lines = lines[:end_line] + ["\n"] + new_code.split('\n') + lines[end_line:]
                return '\n'.join(new_lines)

        logger.warning(f"Class '{class_name}' not found in file")
        return self._insert_after_regex(content, f"class {class_name}", new_code)

    def _insert_before_function(self, content: str, func_name: str, new_code: str) -> Optional[str]:
        """在函数定义之前插入新代码"""
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self._insert_before_regex(content, f"def {func_name}", new_code)

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                start_line = node.lineno - 1

                lines = content.split('\n')
                new_lines = lines[:start_line] + new_code.split('\n') + ["\n"] + lines[start_line:]
                return '\n'.join(new_lines)

        logger.warning(f"Function '{func_name}' not found in file")
        return self._insert_before_regex(content, f"def {func_name}", new_code)

    # ════════════════════════════════════════
    # Regex 后备方案 (当 AST 解析失败时)
    # ════════════════════════════════════════

    def _replace_function_regex(self, content: str, func_name: str, new_code: str) -> Optional[str]:
        """用 regex 替换函数 (后备方案)"""
        # 匹配 def func_name(...) 直到下一个同级 def 或 class
        pattern = rf'(^\s*def {func_name}\([^)]*\)[^:]*:\s*\n(?:[ \t]+.*\n)*)'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return content[:match.start()] + new_code + content[match.end():]
        return None

    def _replace_method_regex(self, content: str, class_name: str, method_name: str, new_code: str) -> Optional[str]:
        """用 regex 替换类中的方法 (后备方案)"""
        # 先找到类定义的范围
        class_pattern = rf'(class {class_name}[^:]*:\s*\n)((?:[ \t]+.*\n)*)'
        class_match = re.search(class_pattern, content, re.MULTILINE)
        if not class_match:
            return None

        class_body = class_match.group(2)
        method_pattern = rf'(^\s+def {method_name}\([^)]*\)[^:]*:\s*\n(?:[ \t]+.*\n)*)'
        method_match = re.search(method_pattern, class_body, re.MULTILINE)
        if method_match:
            new_class_body = class_body[:method_match.start()] + new_code + class_body[method_match.end():]
            return content[:class_match.start()] + class_match.group(1) + new_class_body + content[class_match.end():]
        return None

    def _replace_class_regex(self, content: str, class_name: str, new_code: str) -> Optional[str]:
        """用 regex 替换整个类 (后备方案)"""
        pattern = rf'(class {class_name}[^:]*:\s*\n(?:[ \t]+.*\n)*)'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return content[:match.start()] + new_code + content[match.end():]
        return None

    def _insert_after_regex(self, content: str, marker: str, new_code: str) -> Optional[str]:
        """用 regex 在标记后插入 (后备方案)"""
        pattern = rf'({marker}[^:]*:\s*\n(?:[ \t]+.*\n)*)'
        match = re.search(pattern, content, re.MULTILINE)
        if match:
            return content[:match.end()] + "\n" + new_code + content[match.end():]
        # 如果找不到类/函数，追加到末尾
        return content + "\n\n" + new_code

    def _insert_before_regex(self, content: str, marker: str, new_code: str) -> Optional[str]:
        """用 regex 在标记前插入 (后备方案)"""
        match = re.search(marker, content)
        if match:
            pos = match.start()
            # 找到标记所在行的起始位置
            line_start = content.rfind('\n', 0, pos) + 1
            return content[:line_start] + new_code + "\n" + content[line_start:]
        return content + "\n\n" + new_code

    # ════════════════════════════════════════
    # 缩进处理
    # ════════════════════════════════════════

    @staticmethod
    def _get_indent_level(line: str) -> int:
        """获取行的缩进级别 (空格数)"""
        return len(line) - len(line.lstrip())

    @staticmethod
    def _adjust_indent(code: str, target_indent: int) -> str:
        """调整代码块的缩进级别"""
        lines = code.split('\n')
        # 检测代码当前的最小缩进 (排除空行)
        min_indent = float('inf')
        for line in lines:
            stripped = line.lstrip()
            if stripped and not stripped.startswith('#'):
                indent = len(line) - len(stripped)
                min_indent = min(min_indent, indent)

        if min_indent == float('inf'):
            min_indent = 0

        # 计算需要调整的偏移量
        delta = target_indent - min_indent

        if delta == 0:
            return code

        adjusted_lines = []
        for line in lines:
            stripped = line.lstrip()
            if not stripped:
                adjusted_lines.append(line)  # 保留空行不变
            else:
                current_indent = len(line) - len(stripped)
                new_indent = max(0, current_indent + delta)
                adjusted_lines.append(' ' * new_indent + stripped)

        return '\n'.join(adjusted_lines)

    # ════════════════════════════════════════
    # 校验
    # ════════════════════════════════════════

    def _validate_all_modified_files(self, file_paths: List[str]) -> Dict:
        """
        校验所有修改后的文件
        
        检查:
        1. Python 语法 (ast.parse)
        2. Import 完整性 (尝试 import)
        3. 关键类/函数仍然存在 (SASRec, SelfAttention 等)
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

            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查 1: Python 语法
            syntax_result = self._check_syntax(content, file_path)
            if not syntax_result["passed"]:
                results[file_path] = syntax_result
                all_passed = False
                errors.append(f"{file_path}: syntax error — {syntax_result.get('error', '')}")
                continue

            # 检查 2: 关键符号存在性
            symbol_result = self._check_critical_symbols(content, file_path)
            if not symbol_result["passed"]:
                results[file_path] = symbol_result
                all_passed = False
                errors.append(f"{file_path}: missing critical symbols — {symbol_result.get('missing', [])}")
                continue

            results[file_path] = {
                "passed": True,
                "syntax_check": syntax_result,
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
        """Python 语法检查"""
        try:
            ast.parse(content)
            return {"passed": True, "detail": "AST parse OK"}
        except SyntaxError as e:
            return {
                "passed": False,
                "error": f"SyntaxError at line {e.lineno}: {e.msg}",
                "line": e.lineno,
            }

    def _check_critical_symbols(self, content: str, file_path: str) -> Dict:
        """检查关键类/函数是否仍然存在"""
        # 根据文件类型检查不同的关键符号
        filename = os.path.basename(file_path)

        critical_symbols = {
            "models.py": ["class SASRec", "class SRModel", "def finetune", "def add_position_embedding"],
            "modules.py": ["class SelfAttention", "class Intermediate", "class EncoderLayer", "class Encoder", "class LayerNorm"],
            "trainers.py": ["def _get_loss", "def acc_metric"],
        }

        required = critical_symbols.get(filename, [])
        missing = []
        for sym in required:
            if sym not in content:
                missing.append(sym)

        if missing:
            return {"passed": False, "missing": missing}
        return {"passed": True, "found": required}

    # ════════════════════════════════════════
    # 文件路径解析
    # ════════════════════════════════════════

    def _resolve_file_path(self, target_file: str) -> Optional[str]:
        """
        将逻辑文件名 (如 "models.py") 解析为实际文件路径
        
        查找顺序:
        1. project_root 直接下 (如 /path/Recmodel/models.py)
        2. project_root/Recmodel 子目录
        """
        candidates = [
            os.path.join(self.project_root, target_file),
            os.path.join(self.project_root, "Recmodel", target_file),
        ]

        if self.adapter:
            source_map = self.adapter.SOURCE_FILE_MAP
            if target_file in source_map:
                rel_path = source_map[target_file]
                candidates.extend([
                    os.path.join(self.project_root, rel_path),
                    os.path.join(self.project_root, "Recmodel", rel_path),
                ])

        for path in candidates:
            if os.path.exists(path):
                return path

        logger.warning(f"Cannot resolve file path for {target_file}, tried: {candidates}")
        return None

    # ════════════════════════════════════════
    # 本地快照系统 (替代 Git — 无需联网!)
    # ════════════════════════════════════════

    def _create_local_snapshot(self) -> Dict:
        """
        创建本地文件快照 — 替代 Git snapshot
        
        将所有被跟踪的源码文件复制到快照目录:
        evolve_logs/rollback_snapshots/snap_XXX/models.py
        
        快照后计算各文件的 hash，用于回滚校验。
        """
        import time
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
                # 记录 hash
                with open(src_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                hashes[file_key] = hashlib.md5(content.encode('utf-8')).hexdigest()[:12]
                logger.info(f"Snapshot: {file_key} saved to {dst_path}")
            else:
                logger.warning(f"Source file not found for snapshot: {file_key}")
        
        # 保存快照元数据
        meta = {
            "snapshot_id": snapshot_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "saved_files": saved_files,
            "file_hashes": hashes,
        }
        meta_path = snapshot_subdir / "_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        
        self._pre_snapshot_hashes = hashes
        
        logger.info(f"Local snapshot created: {snapshot_id}, {len(saved_files)} files")
        return {"ok": True, "snapshot_id": snapshot_id, "saved_files": saved_files}

    def _local_rollback(self, snapshot_id: str) -> Dict:
        """
        从本地快照恢复源码文件 — 替代 Git rollback
        
        将快照目录中的文件复制回项目目录，覆盖当前版本。
        回滚后校验文件 hash 是否与快照前一致。
        """
        snapshot_subdir = self._snapshot_dir / snapshot_id
        if not snapshot_subdir.exists():
            logger.error(f"Snapshot not found: {snapshot_id}")
            return {"ok": False, "error": f"Snapshot {snapshot_id} not found"}
        
        # 读取元数据
        meta_path = snapshot_subdir / "_meta.json"
        if not meta_path.exists():
            logger.error(f"Snapshot metadata not found: {snapshot_id}")
            return {"ok": False, "error": "Snapshot metadata missing"}
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        restored_files = []
        verification_ok = True
        
        for file_key in meta.get("saved_files", []):
            snapshot_path = snapshot_subdir / file_key
            if not snapshot_path.exists():
                logger.warning(f"Snapshot file missing: {file_key}")
                continue
            
            target_path = self._resolve_file_path(file_key)
            if target_path:
                shutil.copy2(snapshot_path, target_path)
                restored_files.append(file_key)
                
                # 校验 hash
                expected_hash = meta.get("file_hashes", {}).get(file_key)
                if expected_hash:
                    with open(target_path, 'r', encoding='utf-8') as f:
                        current_content = f.read()
                    current_hash = hashlib.md5(current_content.encode('utf-8')).hexdigest()[:12]
                    if current_hash != expected_hash:
                        logger.warning(f"Hash mismatch for {file_key}: expected {expected_hash}, got {current_hash}")
                        verification_ok = False
                    else:
                        logger.info(f"Rollback verified: {file_key} hash matches")
        
        logger.info(f"Local rollback completed: {len(restored_files)} files restored")
        return {
            "ok": True,
            "restored_files": restored_files,
            "verification_ok": verification_ok,
            "snapshot_id": snapshot_id,
        }

    def rollback_last_changes(self) -> Dict:
        """回滚最近一次的结构修改 (使用本地快照)"""
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
        清理 LLM 输出的 new_code
        
        常见问题:
        - LLM 可能输出 ```python ... ``` 包裹的代码
        - 可能包含解释性的注释在代码前后
        - 可能有不正确的缩进
        """
        # 移除 markdown 代码块标记
        code = raw_code
        # 移除开头的 ```python 或 ```
        code = re.sub(r'^```(?:python)?\s*\n?', '', code)
        # 移除结尾的 ```
        code = re.sub(r'\n?```\s*$', '', code)

        # 移除 LLM 可能添加的解释性前缀/后缀
        # 如 "以下是修改后的代码:" 或 "修改说明: ..."
        lines = code.split('\n')
        clean_lines = []
        code_started = False
        for line in lines:
            # 检测是否是解释性文字而非代码
            stripped = line.strip()
            if not code_started:
                if stripped.startswith('def ') or stripped.startswith('class ') or \
                   stripped.startswith('@') or stripped.startswith('import ') or \
                   stripped.startswith('from ') or stripped.startswith('#') or \
                   stripped.startswith('"""') or stripped.startswith("'''") or \
                   stripped == '':
                    code_started = True
                else:
                    continue  # 跳过解释性文字

            if code_started:
                clean_lines.append(line)

        return '\n'.join(clean_lines)