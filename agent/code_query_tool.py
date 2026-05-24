"""
CodeQueryTool — LLM 代码查询工具

核心设计理念:
  旧方案: 把所有源码塞进 prompt → 超长就硬截断 → LLM 看到残缺代码 → SEARCH/REPLACE 匹配失败
  新方案: 只给 LLM 一个轻量索引 → LLM 按需提出查询 → 系统精确返回所需代码 → 无截断、无残缺

查询能力:
  1. list_files  — 列出可查询的文件 + 大小 + 简要描述
  2. read_file   — 读取某个文件的完整内容
  3. get_outline — 获取某个文件的 AST 结构概览 (类/函数签名 + 行号)
  4. search_function — 搜索某个类/函数的定义位置和签名
  5. search_pattern  — 在所有文件中搜索文本模式 (如变量名、关键字)
  6. get_signature   — 获取某个类/函数的完整定义 (包括方法体)
  7. get_region      — 获取文件中某个行号范围的代码片段

使用方式:
  LLM 在分析阶段可以输出查询请求 (JSON 格式), 系统解析并执行查询,
  将结果返回给 LLM, LLM 可以继续查询或直接输出改进方案。
"""

import ast
import os
import re
import json
import logging
from typing import Optional, List, Dict, Any, Tuple

logger = logging.getLogger("rec_self_evolve.code_query")


class CodeQueryTool:
    """LLM 代码查询工具 — 让 LLM 按需获取代码, 而不是把所有代码塞进 prompt"""

    def __init__(self, project_root: str, source_file_map: Dict[str, str],
                 adapter=None):
        """
        Args:
            project_root: 项目根目录
            source_file_map: 文件映射 {显示名: 相对路径} (来自 SeqRecAdapter.SOURCE_FILE_MAP)
            adapter: SeqRecAdapter 实例 (可选, 用于获取源码)
        """
        self.project_root = project_root
        self.source_file_map = source_file_map
        self.adapter = adapter

        # 缓存: 避免反复读文件
        self._file_cache: Dict[str, str] = {}
        self._ast_cache: Dict[str, ast.AST] = {}

    # ════════════════════════════════════════
    # 核心查询方法
    # ════════════════════════════════════════

    def _resolve_file_path(self, file_key: str) -> Optional[str]:
        """将 file_key 解析为绝对路径"""
        rel_path = self.source_file_map.get(file_key)
        if not rel_path:
            # 尝试直接作为文件名查找
            rel_path = file_key
        
        candidates = [
            os.path.join(self.project_root, rel_path),
            os.path.join(self.project_root, "Recmodel", rel_path),
        ]
        for path in candidates:
            if os.path.exists(path):
                return path
        return None

    def _read_file_content(self, file_key: str) -> Optional[str]:
        """读取文件内容 (带缓存)"""
        if file_key in self._file_cache:
            return self._file_cache[file_key]
        
        abs_path = self._resolve_file_path(file_key)
        if not abs_path:
            return None
        
        try:
            with open(abs_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self._file_cache[file_key] = content
            return content
        except Exception as e:
            logger.warning(f"Failed to read file {file_key}: {e}")
            return None

    def _parse_ast(self, file_key: str) -> Optional[ast.AST]:
        """解析文件的 AST (带缓存)"""
        if file_key in self._ast_cache:
            return self._ast_cache[file_key]
        
        content = self._read_file_content(file_key)
        if not content:
            return None
        
        try:
            tree = ast.parse(content)
            self._ast_cache[file_key] = tree
            return tree
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_key}: {e}")
            return None

    def refresh_cache(self, file_key: str = None):
        """刷新缓存 (文件被修改后需要调用)"""
        if file_key:
            self._file_cache.pop(file_key, None)
            self._ast_cache.pop(file_key, None)
        else:
            self._file_cache.clear()
            self._ast_cache.clear()

    # ── 查询 1: list_files ──

    def list_files(self) -> str:
        """列出所有可查询的源码文件 + 大小 + 类/函数数量"""
        result_lines = []
        for file_key, rel_path in self.source_file_map.items():
            abs_path = self._resolve_file_path(file_key)
            if not abs_path:
                result_lines.append(f"  {file_key} — ❌ 文件不存在")
                continue
            
            content = self._read_file_content(file_key)
            if not content:
                result_lines.append(f"  {file_key} — ❌ 无法读取")
                continue
            
            # 统计类和函数数量
            tree = self._parse_ast(file_key)
            class_count = 0
            func_count = 0
            class_names = []
            func_names = []
            if tree:
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        class_count += 1
                        class_names.append(node.name)
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        func_count += 1
                        func_names.append(node.name)
            
            size_kb = len(content) / 1024
            line_count = content.count('\n') + 1
            
            desc = f"  {file_key} ({rel_path}) — {line_count}行, {size_kb:.1f}KB"
            if class_names:
                desc += f", 类: {', '.join(class_names)}"
            if func_names:
                desc += f", 函数: {', '.join(func_names)}"
            result_lines.append(desc)
        
        header = "## 可查询的源码文件列表\n\n"
        header += "你可以使用以下查询命令获取代码详情:\n"
        header += "- `read_file(\"文件名\")` — 读取完整文件内容\n"
        header += "- `get_outline(\"文件名\")` — 获取文件结构概览 (类/函数签名)\n"
        header += "- `search_function(\"函数名\")` — 搜索函数/类的定义\n"
        header += "- `search_pattern(\"关键词\")` — 在所有文件中搜索文本\n"
        header += "- `get_signature(\"类名.方法名\"或\"函数名\")` — 获取完整定义\n"
        header += "- `get_region(\"文件名\", 起始行, 结束行)` — 获取指定行范围\n\n"
        
        return header + '\n'.join(result_lines)

    # ── 查询 2: read_file ──

    def read_file(self, file_key: str) -> str:
        """读取文件的完整内容"""
        content = self._read_file_content(file_key)
        if not content:
            return f"❌ 文件 '{file_key}' 不存在或无法读取"
        
        abs_path = self._resolve_file_path(file_key)
        line_count = content.count('\n') + 1
        
        result = f"## 文件: {file_key} ({line_count} 行)\n\n```python\n{content}\n```\n"
        return result

    # ── 查询 3: get_outline ──

    def get_outline(self, file_key: str) -> str:
        """获取文件的 AST 结构概览: 类名、方法名、函数名、行号范围"""
        content = self._read_file_content(file_key)
        if not content:
            return f"❌ 文件 '{file_key}' 不存在或无法读取"
        
        tree = self._parse_ast(file_key)
        if not tree:
            # AST 解析失败 → 返回完整文件 (让 LLM 自己看)
            return self.read_file(file_key)
        
        lines = content.split('\n')
        result_lines = []
        
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                # 类的 docstring
                doc = ast.get_docstring(node) or ""
                doc_short = doc.split('\n')[0][:80] if doc else ""
                
                result_lines.append(
                    f"  📦 class {node.name} (L{start_line}-{end_line})"
                )
                if doc_short:
                    result_lines.append(f"     「{doc_short}」")
                
                # 类的方法列表
                methods = []
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(
                            f"    · {item.name}() (L{item.lineno}-{item.end_lineno or item.lineno})"
                        )
                if methods:
                    result_lines.extend(methods)
                
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = node.lineno
                end_line = node.end_lineno or start_line
                doc = ast.get_docstring(node) or ""
                doc_short = doc.split('\n')[0][:80] if doc else ""
                
                # 函数签名
                args = [a.arg for a in node.args.args if a.arg != 'self']
                sig = f"{node.name}({', '.join(args)})"
                
                result_lines.append(
                    f"  🔧 def {sig} (L{start_line}-{end_line})"
                )
                if doc_short:
                    result_lines.append(f"     「{doc_short}」")
        
        header = f"## 文件 {file_key} 的结构概览\n\n"
        return header + '\n'.join(result_lines)

    # ── 查询 4: search_function ──

    def search_function(self, name: str) -> str:
        """搜索某个类/函数的定义位置和签名 (跨所有文件)
        
        支持多种格式:
        - "SASRec" → 搜索类定义
        - "finetune" → 搜索函数定义
        - "SASRec.__init__" → 搜索类中的方法定义
        - "SASRec.finetune" → 搜索类中的方法定义
        
        也自动清洗 LLM 可能添加的格式污染前缀:
        - "SEARCH: class SASRec" → "SASRec"
        - "SEARCH: SASRec.__init__" → "SASRec.__init__"
        """
        # ── 清洗 LLM 格式污染前缀 ──
        # LLM 有时将 SEARCH/REPLACE edit格式 与查询格式混淆,
        # 输出 "SEARCH: class SASRec" 或 "SEARCH: SASRec.__init__" 等格式
        import re
        cleaned_name = name.strip()
        # 移除 "SEARCH:" / "REPLACE:" 前缀
        cleaned_name = re.sub(r'^\s*(SEARCH|REPLACE)\s*:\s*', '', cleaned_name, flags=re.IGNORECASE)
        # 移除 "class " / "def " 等定义关键字前缀
        cleaned_name = re.sub(r'^\s*(class|def)\s+', '', cleaned_name, flags=re.IGNORECASE)
        # 移除包裹的引号或尖括号
        cleaned_name = cleaned_name.strip().strip('"\'`<>')
        name = cleaned_name
        
        results = []
        
        # ── 解析 ClassName.method 格式 ──
        # "SASRec.__init__" → class_name="SASRec", method_name="__init__"
        if '.' in name:
            class_name, method_name = name.split('.', 1)
            return self._search_method_in_class(class_name, method_name)
        
        # ── 普通搜索: 类名或函数名 ──
        for file_key in self.source_file_map:
            content = self._read_file_content(file_key)
            if not content:
                continue
            
            tree = self._parse_ast(file_key)
            if not tree:
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == name:
                    start = node.lineno
                    end = node.end_lineno or start
                    # 类签名: 显示类名 + 继承 + 方法列表
                    bases = [b.id if isinstance(b, ast.Name) else 
                             (b.attr if isinstance(b, ast.Attribute) else str(b)) 
                             for b in node.bases]
                    base_str = ', '.join(bases) if bases else ""
                    inherit_str = f"({base_str})" if base_str else ""
                    methods = [item.name for item in node.body 
                               if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    
                    results.append(
                        f"  📦 class {name}{inherit_str} → {file_key} (L{start}-{end})\n"
                        f"     方法: {', '.join(methods)}"
                    )
                
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    start = node.lineno
                    end = node.end_lineno or start
                    # 函数签名
                    args_list = []
                    for a in node.args.args:
                        arg_str = a.arg
                        if a.annotation:
                            arg_str += f": {self._format_annotation(a.annotation)}"
                        args_list.append(arg_str)
                    sig = f"{name}({', '.join(args_list)})"
                    
                    results.append(
                        f"  🔧 def {sig} → {file_key} (L{start}-{end})"
                    )
        
        if not results:
            return f"❌ 未找到名为 '{name}' 的类或函数"
        
        header = f"## 搜索结果: '{name}'\n\n"
        return header + '\n'.join(results)

    def _search_method_in_class(self, class_name: str, method_name: str) -> str:
        """搜索类中的方法定义 (支持 ClassName.method_name 格式)"""
        results = []
        found_class = False
        
        for file_key in self.source_file_map:
            content = self._read_file_content(file_key)
            if not content:
                continue
            
            tree = self._parse_ast(file_key)
            if not tree:
                continue
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    found_class = True
                    
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            if item.name == method_name or method_name == '*':
                                start = item.lineno
                                end = item.end_lineno or start
                                args_list = []
                                for a in item.args.args:
                                    arg_str = a.arg
                                    if a.annotation:
                                        arg_str += f": {self._format_annotation(a.annotation)}"
                                    args_list.append(arg_str)
                                sig = f"{class_name}.{item.name}({', '.join(args_list)})"
                                
                                # 返回方法的完整代码 (对方法查询, 仅给签名不够, 需要看到完整实现)
                                method_lines = content.split('\n')[start-1:end]
                                method_code = '\n'.join(method_lines)
                                
                                results.append(
                                    f"  🔧 def {sig} → {file_key} (L{start}-{end})\n"
                                    f"```python\n{method_code}\n```"
                                )
        
        if not found_class:
            return f"❌ 未找到名为 '{class_name}' 的类"
        
        if not results:
            # 类找到了但方法没找到 → 列出类的所有方法帮助 LLM 修正
            for file_key in self.source_file_map:
                content = self._read_file_content(file_key)
                if not content:
                    continue
                tree = self._parse_ast(file_key)
                if not tree:
                    continue
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef) and node.name == class_name:
                        methods = [item.name for item in node.body 
                                   if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        return f"❌ 类 '{class_name}' 中未找到方法 '{method_name}'。\n该类的可用方法: {', '.join(methods)}"
            return f"❌ 类 '{class_name}' 中未找到方法 '{method_name}'"
        
        header = f"## 搜索结果: '{class_name}.{method_name}'\n\n"
        return header + '\n'.join(results)

    # ── 查询 5: search_pattern ──

    def search_pattern(self, pattern: str, file_key: str = None) -> str:
        """在所有文件中搜索文本模式 (如变量名、关键字、注释等)"""
        results = []
        search_files = [file_key] if file_key else list(self.source_file_map.keys())
        
        for fk in search_files:
            content = self._read_file_content(fk)
            if not content:
                continue
            
            lines = content.split('\n')
            matches = []
            for i, line in enumerate(lines, 1):
                if pattern in line:
                    # 上下文: 前后各 1 行
                    ctx_start = max(0, i - 2)
                    ctx_end = min(len(lines), i + 1)
                    context_lines = lines[ctx_start:ctx_end]
                    context_str = '\n'.join(context_lines)
                    matches.append(f"  L{i}: {line.strip()[:120]}")
            
            if matches:
                results.append(f"  📄 {fk} — {len(matches)} 处匹配:")
                # 只展示最多 20 个匹配, 防止输出过长
                for m in matches[:20]:
                    results.append(m)
                if len(matches) > 20:
                    results.append(f"  ... 还有 {len(matches) - 20} 处匹配")
        
        if not results:
            return f"❌ 在所有文件中未找到包含 '{pattern}' 的代码"
        
        header = f"## 文本搜索结果: '{pattern}'\n\n"
        return header + '\n'.join(results)

    # ── 查询 6: get_signature ──

    def get_signature(self, name: str) -> str:
        """获取某个类/函数的完整定义 (包括方法体)
        
        支持两种格式:
          - "ClassName" → 获取整个类定义 (所有方法)
          - "ClassName.method_name" → 获取特定方法
          - "function_name" → 获取独立函数
        """
        # 解析是否是类.方法 格式
        parts = name.split('.')
        class_name = parts[0] if len(parts) > 1 else None
        method_name = parts[1] if len(parts) > 1 else None
        
        if class_name and method_name:
            return self._get_class_method(class_name, method_name)
        elif class_name:
            # 可能是 "ClassName" 或 "ClassName" (没有方法部分)
            return self._get_class_or_function(name)
        else:
            return self._get_class_or_function(name)

    def _get_class_or_function(self, name: str) -> str:
        """获取类或函数的完整定义"""
        results = []
        
        for file_key in self.source_file_map:
            content = self._read_file_content(file_key)
            if not content:
                continue
            
            tree = self._parse_ast(file_key)
            if not tree:
                continue
            
            lines = content.split('\n')
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == name:
                    start = node.lineno - 1
                    end = node.end_lineno
                    code = '\n'.join(lines[start:end])
                    results.append(
                        f"## class {name} (完整定义, {file_key} L{node.lineno}-{node.end_lineno})\n\n"
                        f"```python\n{code}\n```\n"
                    )
                
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
                    start = node.lineno - 1
                    end = node.end_lineno
                    code = '\n'.join(lines[start:end])
                    results.append(
                        f"## def {name}() (完整定义, {file_key} L{node.lineno}-{node.end_lineno})\n\n"
                        f"```python\n{code}\n```\n"
                    )
        
        if not results:
            return f"❌ 未找到 '{name}' 的完整定义"
        
        return '\n'.join(results)

    def _get_class_method(self, class_name: str, method_name: str) -> str:
        """获取类中某个方法的完整定义"""
        results = []
        
        for file_key in self.source_file_map:
            content = self._read_file_content(file_key)
            if not content:
                continue
            
            tree = self._parse_ast(file_key)
            if not tree:
                continue
            
            lines = content.split('\n')
            
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                           and item.name == method_name:
                            start = item.lineno - 1
                            end = item.end_lineno
                            code = '\n'.join(lines[start:end])
                            results.append(
                                f"## {class_name}.{method_name}() "
                                f"(完整定义, {file_key} L{item.lineno}-{item.end_lineno})\n\n"
                                f"```python\n{code}\n```\n"
                            )
        
        if not results:
            # 方法名不精确 → 尜找接近的
            similar = self._find_similar_method(class_name, method_name)
            if similar:
                return f"❌ 未找到 '{class_name}.{method_name}' 的定义\n"
                f"💡 相似的方法: {similar}\n"
                f"请使用 `get_signature(\"{class_name}.{similar[0]}\")` 查询"
            return f"❌ 未找到 '{class_name}.{method_name}' 的定义"
        
        return '\n'.join(results)

    def _find_similar_method(self, class_name: str, method_name: str) -> List[str]:
        """找到类中与目标方法名相似的方法"""
        similar = []
        for file_key in self.source_file_map:
            tree = self._parse_ast(file_key)
            if not tree:
                continue
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            # 简单的前缀/子串匹配
                            if method_name in item.name or item.name in method_name:
                                similar.append(item.name)
        return similar

    # ── 查询 7: get_region ──

    def get_region(self, file_key: str, start_line: int, end_line: int) -> str:
        """获取文件中指定行号范围的代码片段"""
        content = self._read_file_content(file_key)
        if not content:
            return f"❌ 文件 '{file_key}' 不存在或无法读取"
        
        lines = content.split('\n')
        total_lines = len(lines)
        
        # 边界检查
        start_line = max(1, start_line)
        end_line = min(total_lines, end_line)
        
        if start_line > end_line:
            return f"❌ 无效的行号范围: {start_line}-{end_line} (文件共 {total_lines} 行)"
        
        code = '\n'.join(lines[start_line - 1:end_line])
        
        result = f"## {file_key} L{start_line}-{end_line} (共 {end_line - start_line + 1} 行)\n\n"
        # 带行号的输出 — 让 LLM 知道精确位置
        numbered_lines = []
        for i, line in enumerate(lines[start_line - 1:end_line], start_line):
            numbered_lines.append(f"{i:4d} | {line}")
        result += "```python\n" + '\n'.join(numbered_lines) + "\n```\n"
        
        return result

    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════

    def _format_annotation(self, annotation: ast.expr) -> str:
        """格式化类型注解"""
        if isinstance(annotation, ast.Name):
            return annotation.id
        elif isinstance(annotation, ast.Constant):
            return str(annotation.value)
        elif isinstance(annotation, ast.Subscript):
            return f"{self._format_annotation(annotation.value)}[{self._format_annotation(annotation.slice)}]"
        elif isinstance(annotation, ast.Attribute):
            return f"{self._format_annotation(annotation.value)}.{annotation.attr}"
        return "Any"

    # ════════════════════════════════════════
    # 执行查询 (从 LLM 的 JSON 输出中解析)
    # ════════════════════════════════════════

    def execute_query(self, query: Dict[str, Any]) -> str:
        """执行单个查询请求
        
        Args:
            query: {"action": "read_file", "args": {"file_key": "models.py"}}
                   或 {"action": "search_function", "args": {"name": "SASRec"}}
                   也兼容纯字符串输入 (自动转为 search_function)
        
        Returns:
            str: 查询结果
        """
        # ── 兼容字符串输入 ──
        # LLM 有时输出 queries 为纯字符串列表, 而非结构化字典
        if isinstance(query, str):
            query = {"action": "search_function", "args": {"name": query}}
        
        action = query.get("action", "")
        args = query.get("args", {})
        
        action_map = {
            "list_files": lambda: self.list_files(),
            "read_file": lambda: self.read_file(args.get("file_key", "")),
            "get_outline": lambda: self.get_outline(args.get("file_key", "")),
            "search_function": lambda: self.search_function(args.get("name", "")),
            "search_pattern": lambda: self.search_pattern(
                args.get("pattern", ""), args.get("file_key")
            ),
            "get_signature": lambda: self.get_signature(args.get("name", "")),
            "get_region": lambda: self.get_region(
                args.get("file_key", ""),
                args.get("start_line", 1),
                args.get("end_line", 50),
            ),
        }
        
        handler = action_map.get(action)
        if not handler:
            return f"❌ 未知的查询动作: '{action}'。可用动作: {list(action_map.keys())}"
        
        try:
            result = handler()
            logger.info(f"Code query executed: action={action}, args={args}, result_len={len(result)}")
            return result
        except Exception as e:
            logger.warning(f"Code query failed: action={action}, args={args}, error={e}")
            return f"❌ 查询执行失败: {e}"

    def execute_queries(self, queries: List[Dict[str, Any]]) -> str:
        """执行多个查询请求
        
        Args:
            queries: [{"action": "...", "args": {...}}, ...]
        
        Returns:
            str: 所有查询结果拼接
        """
        parts = []
        for q in queries:
            result = self.execute_query(q)
            parts.append(result)
        return '\n\n'.join(parts)

    # ════════════════════════════════════════
    # 构建代码索引 (替代塞全部源码)
    # ════════════════════════════════════════

    def build_code_index(self) -> str:
        """构建轻量级代码索引 — 替代把全部源码塞进 prompt
        
        索引包含:
        1. 文件列表 + 大小
        2. 每个文件的类/函数签名 (不包含方法体)
        3. 关键全局变量/import
        
        总大小通常在 2000-4000 字符, 远小于塞全部源码的 15000+ 字符
        """
        index_parts = []
        index_parts.append("## 源码文件索引\n")
        index_parts.append("以下是你可查询的源码文件的**索引** (仅包含签名, 不包含方法体)。\n")
        index_parts.append("你需要使用查询命令来获取具体的代码详情。\n\n")
        
        for file_key, rel_path in self.source_file_map.items():
            content = self._read_file_content(file_key)
            if not content:
                index_parts.append(f"### {file_key} — ❌ 无法读取\n\n")
                continue
            
            abs_path = self._resolve_file_path(file_key)
            line_count = content.count('\n') + 1
            size_kb = len(content) / 1024
            
            index_parts.append(f"### {file_key} ({rel_path}, {line_count}行, {size_kb:.1f}KB)\n")
            
            # Import 语句 (简要展示)
            tree = self._parse_ast(file_key)
            if tree:
                imports = []
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.append(f"import {alias.name}")
                    elif isinstance(node, ast.ImportFrom):
                        names = ', '.join(a.name for a in node.names)
                        imports.append(f"from {node.module} import {names}")
                if imports:
                    index_parts.append(f"  导入: {', '.join(imports[:5])}")
                    if len(imports) > 5:
                        index_parts.append(f"  ... (+{len(imports)-5} 更多)")
            
            # 类和函数签名
            if tree:
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.ClassDef):
                        # 类签名: 类名 + 继承 + 方法列表
                        bases = [b.id if isinstance(b, ast.Name) else
                                 (b.attr if isinstance(b, ast.Attribute) else str(b))
                                 for b in node.bases]
                        base_str = ', '.join(bases) if bases else ""
                        inherit_str = f"({base_str})" if base_str else ""
                        
                        methods = []
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                m_args = [a.arg for a in item.args.args if a.arg != 'self']
                                methods.append(f"{item.name}({', '.join(m_args)})")
                        
                        doc = ast.get_docstring(node) or ""
                        doc_short = doc.split('\n')[0][:60] if doc else ""
                        
                        index_parts.append(
                            f"  📦 class {node.name}{inherit_str} "
                            f"(L{node.lineno}-{node.end_lineno})"
                        )
                        if doc_short:
                            index_parts.append(f"     「{doc_short}」")
                        if methods:
                            index_parts.append(f"     方法: {', '.join(methods)}")
                    
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [a.arg for a in node.args.args]
                        sig = f"{node.name}({', '.join(args)})"
                        doc = ast.get_docstring(node) or ""
                        doc_short = doc.split('\n')[0][:60] if doc else ""
                        
                        index_parts.append(
                            f"  🔧 def {sig} (L{node.lineno}-{node.end_lineno})"
                        )
                        if doc_short:
                            index_parts.append(f"     「{doc_short}」")
                
                # 关键全局变量
                global_vars = []
                for node in ast.iter_child_nodes(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                global_vars.append(target.id)
                if global_vars:
                    index_parts.append(f"  变量: {', '.join(global_vars[:10])}")
            
            index_parts.append("")  # 文件间空行
        
        # 查询提示
        index_parts.append("\n### 如何查询代码详情\n")
        index_parts.append("你可以使用以下查询命令来获取具体代码 (在回复中使用 JSON 格式):\n")
        index_parts.append("```json\n")
        index_parts.append("[\n")
        index_parts.append("  {\"action\": \"read_file\", \"args\": {\"file_key\": \"文件名\"}},\n")
        index_parts.append("  {\"action\": \"get_outline\", \"args\": {\"file_key\": \"文件名\"}},\n")
        index_parts.append("  {\"action\": \"search_function\", \"args\": {\"name\": \"函数名或类名\"}},\n")
        index_parts.append("  {\"action\": \"search_pattern\", \"args\": {\"pattern\": \"关键词\"}},\n")
        index_parts.append("  {\"action\": \"get_signature\", \"args\": {\"name\": \"类名.方法名\"}},\n")
        index_parts.append("  {\"action\": \"get_region\", \"args\": {\"file_key\": \"文件名\", \"start_line\": 10, \"end_line\": 50}}\n")
        index_parts.append("]\n")
        index_parts.append("```\n")
        index_parts.append("⚠ 修改代码时, SEARCH/REPLACE 的 search 文本必须与源码**完全匹配**! ")
        index_parts.append("请先 `read_file` 或 `get_region` 获取精确的源码文本, 再编写修改。\n")
        
        return '\n'.join(index_parts)