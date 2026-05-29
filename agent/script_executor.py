#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
脚本执行共享工具 — 供 core.py 和 hypothesis_verification_agent.py 共用的脚本执行基础设施

设计原则:
  - 数据加载完全由 LLM 自主编写 (只提供数据文件路径, 不预注入加载代码)
  - 仅注入最小运行时基础设施: OUTPUT_FILE + save_result() (脚本必须有输出目标)
  - 所有代码执行、结果提取、修正循环均为独立功能, 不依赖 self

核心变化 (v2):
  - 移除预注入头部 (不再替 LLM 写 import / 数据加载 / 变量解析)
  - DataInjector 职责缩小为: 序列化数据文件 + 生成最小基础设施
  - LLM 通过 prompt 中的 DATA_FILE 路径自主编写数据加载逻辑
"""

import os
import sys
import json
import subprocess
import logging
import re
from typing import Dict, List, Optional, Tuple
from pathlib import Path
logger = logging.getLogger("rec_self_evolve.script_executor")


# ════════════════════════════════════════
# 输出路径检测
# ════════════════════════════════════════

def extract_output_path(script_content: str) -> Optional[str]:
    """
    从脚本代码中提取输出文件路径 — 多模式匹配
    
    支持的模式:
    - OUTPUT_FILE = "path"
    - output_file = "path"
    - result_path = "path"
    - open("path/write", "w")
    - json.dump(..., open("path", "w"))
    """
    patterns = [
        r'OUTPUT_FILE\s*=\s*["\']([^"\']+)["\']',
        r'output_file\s*=\s*["\']([^"\']+)["\']',
        r'result_path\s*=\s*["\']([^"\']+)["\']',
        r'open\(["\']([^"\']*result[^"\']*\.json)["\']',
        r'json\.dump\([^,]+,\s*open\(["\']([^"\']+\.json)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, script_content)
        if match:
            return match.group(1)
    return None


# ════════════════════════════════════════
# 数据序列化 (最小职责 — 只保存数据文件, 不生成注入头部)
# ════════════════════════════════════════

class DataInjector:
    """
    数据序列化工具 — 将预加载数据序列化为 JSON 文件
    
    职责 (v2 — 缩小):
    - 序列化数据到 JSON 文件 (供 LLM 在脚本中自主加载)
    - 生成最小运行时基础设施 (OUTPUT_FILE + save_result)
    - 不再替 LLM 写 import、数据加载、变量解析等代码
    
    LLM 通过 prompt 中的 DATA_FILE 路径知道数据文件位置,
    自主编写 json.load() 等加载逻辑。
    """
    
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
    
    def serialize_data(self, verification_data: Dict,
                       data_subdir: str = "verification_scripts/data",
                       filename: str = "preloaded_data.json") -> str:
        """
        将验证数据序列化为 JSON 文件
        
        处理策略:
        - 大列表 (>100 条) 只保存前 50 条作为样本, 同时保存 _count
        - 大字典 (>2000 条) 截断到 2000 条
        - 无法序列化的数据记录类型信息
        
        Args:
            verification_data: 待序列化的数据字典
            data_subdir: 数据子目录路径
            filename: 数据文件名
            
        Returns:
            数据文件的绝对路径 (LLM 在 prompt 中会收到这个路径)
        """
        data_dir = os.path.join(self.log_dir, data_subdir)
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, filename)
        
        serializable = {}
        for key, value in verification_data.items():
            try:
                if isinstance(value, list) and len(value) > 100:
                    serializable[key + "_sample"] = value[:50]
                    serializable[key + "_count"] = len(value)
                elif isinstance(value, dict) and len(value) > 2000:
                    items = list(value.items())[:2000]
                    serializable[key + "_total"] = len(value)
                    serializable[key] = dict(items)
                else:
                    serializable[key] = value
            except (TypeError, ValueError):
                serializable[key + "_info"] = f"无法序列化: {type(value).__name__}"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        
        return data_file
    
    def generate_minimal_runtime(self, data_file: str, output_file: str,
                                  model_info: Dict = None,
                                  prev_acquired_files: Dict[str, str] = None) -> str:
        """
        生成最小运行时基础设施
        
        包含:
        1. sys.path.insert — 将模型目录添加到 Python 搜索路径 (解决 import 依赖)
        2. OUTPUT_FILE 常量 (脚本必须有输出目标)
        3. DATA_FILE 常量 (数据文件路径, LLM 可用 json.load(DATA_FILE) 加载)
        4. PREV_ACQUIRED_FILES 常量 (前一步已获取数据的 JSON 文件路径, 供依赖数据加载)
        5. save_result() 辅助函数 (安全写入结果)
        
        不包含:
        - import 语句 (LLM 自主决定需要什么)
        - 数据加载逻辑 (LLM 自主编写)
        - 变量解析 (LLM 自主决定如何使用数据)
        """
        # ── 构建 sys.path 注入 ──
        # 动态扫描 project_root 下包含 models.py 的子目录，自动添加到 sys.path
        # 这样支持任意目录命名（Recmodel、Recmodel2、models 等）
        sys_path_lines = []
        
        # 排除的目录
        exclude_dirs = {".git", "__pycache__", "logs", "data", "output", ".pytest_cache", ".model_probe_cache"}
        
        if model_info:
            project_root = model_info.get("project_root", "")
            
            if project_root:
                root = Path(project_root)
                
                # 1. 如果 project_root 直接包含 models.py，不需要额外 sys.path
                # 2. 搜索所有包含 models.py 的子目录
                if (root / "models.py").exists():
                    pass  # project_root 本身已包含 models.py，不需要添加子目录
                else:
                    for subdir in root.iterdir():
                        if subdir.is_dir() and subdir.name not in exclude_dirs and not subdir.name.startswith("."):
                            if (subdir / "models.py").exists():
                                sys_path_lines.append(
                                    f'sys.path.insert(0, "{subdir}")  # {subdir.name} 目录'
                                )
                                break  # 找到一个就够用了
        
        sys_path_block = ""
        if sys_path_lines:
            sys_path_block = "import sys\n" + "\n".join(sys_path_lines) + "\n\n"
        
        # ── 构建 PREV_ACQUIRED_FILES 注入 ──
        prev_files_block = ""
        if prev_acquired_files:
            prev_lines = []
            for data_name, file_path in prev_acquired_files.items():
                prev_lines.append(f'    "{data_name}": "{file_path}",')
            prev_files_block = "PREV_ACQUIRED_FILES = {\n" + "\n".join(prev_lines) + "\n}\n\n"
        else:
            prev_files_block = "PREV_ACQUIRED_FILES = {}\n\n"
        
        return f"""# ── 运行时基础设施 (最小注入) ──
{sys_path_block}DATA_FILE = "{data_file}"
OUTPUT_FILE = "{output_file}"

{prev_files_block}def save_result(result_dict):
    # 保存验证结果到 JSON 文件
    import json, os, sys
    os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(result_dict, f, ensure_ascii=False, indent=2)

"""
    
    def wrap_script(self, code: str, verification_data: Dict,
                     output_file: str, model_info: Dict = None,
                     prev_acquired_files: Dict[str, str] = None) -> str:
        """
        完整流程: 序列化数据 + 生成最小运行时 + 组合脚本
        
        注入内容已在 prompt 中明确告知 LLM，LLM 知道:
        - DATA_FILE / OUTPUT_FILE 常量已定义，不需要重复定义
        - PREV_ACQUIRED_FILES 常量已定义，包含前一步已获取数据的 JSON 文件路径
        - save_result() 函数已定义，不需要重复定义
        - sys.path 已自动设置，可以直接 from models import SASRec
        
        LLM 的代码自主包含:
        - import 语句 (LLM 自己决定需要什么)
        - 数据加载逻辑 (LLM 通过 DATA_FILE 或 PREV_ACQUIRED_FILES 自主加载)
        - 验证逻辑 (LLM 自主编写)
        - save_result() 调用 (LLM 自主调用)
        """
        data_file = self.serialize_data(verification_data)
        runtime = self.generate_minimal_runtime(
            data_file, output_file, 
            model_info=model_info,
            prev_acquired_files=prev_acquired_files
        )
        return runtime + "\n" + code
    
    def extract_core_code(self, full_code: str) -> str:
        """
        从完整脚本中提取核心验证逻辑 (去掉运行时基础设施)
        """
        lines = full_code.split("\n")
        core_start = 0
        
        # 搜索 save_result 函数定义结束后的标记
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                prev_lines = [l.strip() for l in lines[max(0,i-3):i]]
                if any("sys.stderr" in l for l in prev_lines):
                    core_start = i + 1
                    break
        
        # 回退: 找第一个不属于运行时基础设施的代码行
        if core_start == 0:
            runtime_markers = ("DATA_FILE", "OUTPUT_FILE", "PREV_ACQUIRED_FILES", 
                               "def save_result", "# ── 运行时基础设施")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and \
                   not any(stripped.startswith(m) for m in runtime_markers):
                    core_start = i
                    break
        
        if core_start > 0:
            return "\n".join(lines[core_start:])
        return full_code

    def format_data_keys_for_prompt(self, verification_data: Dict,
                                     prev_acquired_files: Dict[str, str] = None,
                                     model_info: Dict = None,
                                     max_model_info_chars: int = 4000) -> str:
        """
        格式化完整数据上下文 — 专为脚本修正 prompt 使用
        
        包含三部分:
        1. 数据键名映射 (DATA_FILE / PREV_ACQUIRED_FILES 中的 JSON 键名 + 加载方式)
        2. 已获取数据详情 (PREV_ACQUIRED_FILES 的数据结构和加载代码示例)
        3. 模型信息 (checkpoint、model_args、源码等, 供修复模型相关错误)
        
        处理规则 (与 serialize_data 一致):
        - 列表 > 100 → key_sample + key_count
        - 字典 > 2000 → key + key_total
        - 其他 → key
        """
        lines = []
        
        # ── 1. PREV_ACQUIRED_FILES 中的已获取数据 ──
        if prev_acquired_files:
            lines.append("## ⚠️ 前一步已获取的数据 (不要重复获取! 优先通过 PREV_ACQUIRED_FILES 加载)")
            lines.append("")
            for data_name, file_path in prev_acquired_files.items():
                value = verification_data.get(data_name)
                type_desc = ""
                if value is not None:
                    if isinstance(value, dict):
                        top_keys = list(value.keys())[:8]
                        type_desc = f"Dict, {len(value)} 条, 顶层 keys: {top_keys}"
                        inner = value.get("data")
                        if isinstance(inner, dict):
                            inner_keys = list(inner.keys())[:8]
                            type_desc += f"; data 子字段 keys: {inner_keys}"
                    elif isinstance(value, list):
                        type_desc = f"List, {len(value)} 项"
                lines.append(f"- **{data_name}**: {type_desc}")
                lines.append(f"  加载: `prev_data = json.load(open(PREV_ACQUIRED_FILES['{data_name}']))`")
            lines.append("")
            lines.append("**关键提醒**: 修正时如果脚本需要引用以上数据, 直接从 PREV_ACQUIRED_FILES 加载即可, 不要重新获取。")
            lines.append("")
        
        # ── 2. DATA_FILE 中的数据键名映射 ──
        lines.append("## 可用数据 (DATA_FILE 中的 JSON 键名和数据结构)")
        lines.append("加载方式: `data = json.load(open(DATA_FILE))`, 然后用 `data['key']` 访问具体数据。")
        lines.append("")
        
        for key, value in verification_data.items():
            if key.startswith("_"):
                continue  # 内部元数据
            if prev_acquired_files and key in prev_acquired_files:
                continue  # 已在上面展示
            
            if value is None:
                lines.append(f"- `{key}`: None (不可用)")
            elif isinstance(value, list):
                if len(value) > 100:
                    lines.append(f"- `{key}`: List, 共 {len(value)} 项 → JSON 键: `{key}_sample` (前50项) + `{key}_count` (总数)")
                    lines.append(f"  加载: `data['{key}_sample']` 和 `data['{key}_count']`")
                    if value and isinstance(value[0], dict):
                        sample_keys = list(value[0].keys())[:8]
                        lines.append(f"  每个元素 keys: {sample_keys}")
                else:
                    lines.append(f"- `{key}`: List, {len(value)} 项 → JSON 键: `{key}`")
                    lines.append(f"  加载: `data['{key}']`")
                    if value and isinstance(value[0], dict):
                        sample_keys = list(value[0].keys())[:8]
                        lines.append(f"  每个元素 keys: {sample_keys}")
            elif isinstance(value, dict):
                if len(value) > 2000:
                    top_keys = list(value.keys())[:8]
                    lines.append(f"- `{key}`: Dict, 共 {len(value)} 条 → JSON 键: `{key}` (截断至2000) + `{key}_total` (总条数)")
                    lines.append(f"  加载: `data['{key}']` 和 `data['{key}_total']`")
                    if top_keys:
                        lines.append(f"  顶层 keys: {top_keys}")
                else:
                    top_keys = list(value.keys())[:8]
                    lines.append(f"- `{key}`: Dict, {len(value)} 个键 → JSON 键: `{key}`")
                    lines.append(f"  加载: `data['{key}']`")
                    if top_keys:
                        lines.append(f"  顶层 keys: {top_keys}")
            elif isinstance(value, str):
                lines.append(f"- `{key}`: 文件路径 → JSON 键: `{key}`")
            else:
                lines.append(f"- `{key}`: {type(value).__name__} → JSON 键: `{key}`")
        
        # ── 3. 模型信息 ──
        if model_info:
            model_lines = []
            model_lines.append("## 模型信息")
            model_lines.append("")
            
            # 运行参数 (最关键)
            model_args = model_info.get("model_args", {})
            if model_args:
                model_lines.append("### 模型运行参数")
                for k, v in model_args.items():
                    model_lines.append(f"- {k}: {v}")
                model_lines.append("")
            
            # Checkpoint 路径
            checkpoint = model_info.get("checkpoint", "")
            if checkpoint:
                model_lines.append(f"### Checkpoint 路径\n- `{checkpoint}`\n")
            
            # Checkpoint tensor 形状
            checkpoint_shapes = model_info.get("checkpoint_shapes", {})
            if checkpoint_shapes:
                model_lines.append("### Checkpoint 维度")
                for tensor_name, shape in checkpoint_shapes.items():
                    model_lines.append(f"- {tensor_name}: {shape}")
                model_lines.append("")
            
            # 模型源码文件路径
            model_file = model_info.get("model_file", "")
            modules_file = model_info.get("modules_file", "")
            if model_file:
                model_lines.append(f"### 模型源码\n- models.py: `{model_file}`")
            if modules_file:
                model_lines.append(f"- modules.py: `{modules_file}`")
            
            model_text = "\n".join(model_lines)
            if len(model_text) > max_model_info_chars:
                # 保留 model_args 和 checkpoint_shapes (最关键), 截断源码
                model_text = model_text[:max_model_info_chars] + "\n\n⚠ 模型信息截断..."
            
            lines.append("")
            lines.append(model_text)
        
        return "\n".join(lines)


# ════════════════════════════════════════
# 脚本执行器
# ════════════════════════════════════════

class ScriptExecutor:
    """
    Python 脚本执行器 — 执行脚本并提取结果
    
    功能:
    - 执行 Python 脚本 (带超时)
    - 多模式结果提取 (输出文件 → stdout → stderr → 单行 JSON)
    - error-result 检测 (退出码=0 但结果含 "error" 字段)
    """
    
    def __init__(self, project_root: str, timeout: int = 60):
        self.project_root = project_root
        self.timeout = timeout
    
    def execute_script(self, script_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        执行验证脚本
        
        Returns:
            (success, result_dict, error_message)
        """
        from .llm_utils import robust_json_parse, extract_json_block
        
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=self.project_root,
            )
            
            if result.returncode != 0:
                error = result.stderr or result.stdout or "Unknown execution error"
                return False, None, error
            
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            output_file_path = extract_output_path(script_content)
            
            if output_file_path and os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                if isinstance(result_data, dict) and "error" in result_data:
                    error_msg = result_data["error"]
                    logger.warning(f"Script wrote error-result (exitcode=0): {error_msg[:200]}")
                    return False, None, f"Script error-result: {error_msg}"
                return True, result_data, None
            
            # 输出文件不存在 → 从 stdout/stderr 中提取 JSON
            for source_name, source_text in [
                ("stderr", result.stderr),
                ("stdout", result.stdout),
            ]:
                if not source_text or not source_text.strip():
                    continue
                
                json_str = extract_json_block(source_text)
                if json_str:
                    parsed = robust_json_parse(json_str)
                    if parsed is not None:
                        if isinstance(parsed, dict) and "error" in parsed:
                            error_msg = parsed["error"]
                            return False, None, f"Script error-result ({source_name}): {error_msg}"
                        logger.info(f"Result extracted from {source_name}")
                        return True, parsed, None
                
                for line in source_text.split("\n"):
                    stripped = line.strip()
                    if stripped and stripped.startswith("{"):
                        parsed = robust_json_parse(stripped)
                        if parsed is not None:
                            if isinstance(parsed, dict) and "error" in parsed:
                                return False, None, f"Script error-result ({source_name}): {parsed['error']}"
                            return True, parsed, None
            
            error_msg = "No output file found"
            if output_file_path:
                error_msg += f": {output_file_path}"
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            return False, None, error_msg
        
        except subprocess.TimeoutExpired:
            return False, None, f"Script execution timed out ({self.timeout}s)"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
    
    def execute_with_correction_loop(
        self,
        initial_code: str,
        hypothesis_id: str,
        llm_retry_helper,
        fix_prompt_template: str,
        injector: DataInjector,
        verification_data: Dict,
        output_file: str,
        max_rounds: int = 3,
        script_dir: str = None,
        code_query_tool = None,
        fix_context: Dict = None,
        data_samples: str = "",
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        执行验证代码, 如果失败则修正重试
        
        流程: 保存代码 → 执行 → 检查错误 → 修正 → 再执行
        
        修正方式 (两种):
        1. 如果提供了 code_query_tool → 使用 Query-Based 修正 (LLM 先查源码再修正)
        2. 否则 → 使用传统的盲修模式 (只给错误信息让 LLM 修正)
        
        修正后重新 wrap_script (注入最小运行时基础设施),
        LLM 的核心验证代码保持不变。
        
        Args:
            code_query_tool: CodeQueryTool 实例 (可选, 提供时使用 Query-Based 修正)
            fix_context: 修正上下文配置 (可选, 如 VERIF_FIX_CONTEXT, 仅 Query-Based 模式使用)
        """
        from .llm_utils import clean_code_response
        
        code = initial_code
        last_error = None
        
        # ── 预格式化数据信息 (各修正轮次共用) ──
        data_info = injector.format_data_keys_for_prompt(
            verification_data,
            model_info=verification_data.get("_model_info"),
        )
        
        # ── Query-Based 修正所需导入 ──
        if code_query_tool is not None:
            from .prompts import (
                QUERY_BASED_SCRIPT_FIX_PHASE1_PROMPT,
                QUERY_BASED_SCRIPT_FIX_PHASE2_PROMPT,
                QUERY_BASED_SCRIPT_FIX_FINAL_PROMPT,
                VERIF_FIX_CONTEXT,
            )
        
        if script_dir is None:
            script_dir = os.path.join(self.project_root, "logs", "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_file = os.path.join(script_dir, f"verify_{hypothesis_id}.py")
        
        for round_num in range(max_rounds):
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing verification script for {hypothesis_id} (round {round_num + 1})")
            
            success, result, error = self.execute_script(script_file)
            
            if success and result is not None:
                logger.info(f"Script executed successfully for {hypothesis_id}")
                return result, None
            
            if error:
                last_error = error
                logger.warning(f"Script failed (round {round_num + 1}): {error[:200]}")
                
                if round_num < max_rounds - 1:
                    # 提取核心代码 (去掉运行时基础设施)
                    core_code = injector.extract_core_code(code)
                    
                    # 从代码中提取输出路径
                    output_match = re.search(r'OUTPUT_FILE\s*=\s*"([^"]+)"', code)
                    output_path = output_match.group(1) if output_match else output_file
                    
                    # ── 选择修正方式 ──
                    if code_query_tool is not None and fix_context is not None:
                        # ── Query-Based 修正: LLM 先查源码再修正 ──
                        logger.info(f"Using query-based fix mode for {hypothesis_id}")
                        # 构建含数据样本的 extra_context
                        query_fix_context_parts = [data_info]
                        if data_samples and data_samples != "无可用数据样本 (项目数据目录中未找到数据文件)":
                            query_fix_context_parts.append(f"## ⚠️ 数据样本 (请严格按此格式解析数据!)\n{data_samples}")
                        query_fix_extra_context = "\n\n".join(query_fix_context_parts)
                        fixed_core = self._query_based_fix_script(
                            code_query_tool=code_query_tool,
                            llm_retry_helper=llm_retry_helper,
                            error_output=error,
                            original_code=core_code,
                            fix_context=fix_context,
                            extra_context=query_fix_extra_context,
                        )
                    else:
                        # ── 传统盲修模式 ──
                        # 从代码中提取数据路径
                        data_match = re.search(r'DATA_FILE\s*=\s*"([^"]+)"', code)
                        data_path = data_match.group(1) if data_match else None
                        
                        fix_prompt = fix_prompt_template.format(
                            original_code=core_code,
                            error_output=error[:1500],
                            output_file_path=output_path,
                            data_info=data_info,
                            data_samples=data_samples,
                        )
                        
                        fixed_response = llm_retry_helper.call_llm(
                            prompt=fix_prompt,
                            system_content=(
                                "你是一位 Python 专家，擅长根据错误信息修正代码。"
                                "只修正导致错误的部分，保持其他逻辑不变。"
                                "确保代码仍然将结果写入指定 JSON 文件。"
                                "数据文件路径: " + (data_path or "见 DATA_FILE 常量") + "，"
                                "使用 json.load(open(DATA_FILE)) 加载可用数据。"
                            ),
                            temperature=0.1,
                            max_tokens=4096,
                        )
                        
                        if fixed_response is None:
                            logger.error(f"LLM code fix call failed for {hypothesis_id}")
                            break
                        
                        fixed_core = clean_code_response(fixed_response)
                    
                    if not fixed_core or len(fixed_core) < 20:
                        logger.error(f"Fix produced invalid code for {hypothesis_id}")
                        break
                    
                    # 重新 wrap: 注入最小运行时基础设施
                    data_file = injector.serialize_data(verification_data)
                    model_info = verification_data.get("_model_info")
                    runtime = injector.generate_minimal_runtime(data_file, output_path, model_info=model_info)
                    code = runtime + "\n" + fixed_core
                else:
                    logger.error(f"Max code fix rounds reached for {hypothesis_id}")
        
        return None, last_error
    
    def _query_based_fix_script(
        self,
        code_query_tool,
        llm_retry_helper,
        error_output: str,
        original_code: str,
        fix_context: Dict,
        extra_context: str = "",
        max_query_rounds: int = 3,
        max_chars_per_query_result: int = 5000,
    ) -> Optional[str]:
        """
        在 ScriptExecutor 内使用 CodeQueryTool 进行 Query-Based 修正
        
        复用与 HypothesisVerificationAgent._fix_script_with_query_mode 相同的设计,
        但不依赖 HypothesisVerificationAgent 实例 (独立可复用)。
        
        Args:
            code_query_tool: CodeQueryTool 实例
            llm_retry_helper: LLMRetryHelper 实例
            error_output: 执行错误信息
            original_code: 原始脚本代码 (核心部分)
            fix_context: 修正上下文配置 (如 VERIF_FIX_CONTEXT)
            extra_context: 额外上下文 (如可用数据描述、DATA_FILE 键名映射等)
            max_query_rounds: 最大查询轮数
            max_chars_per_query_result: 每次查询结果最大字符数
        
        Returns:
            修正后的完整 Python 脑本代码 (纯代码), 或 None
        """
        from .llm_utils import clean_code_response
        from .prompts import (
            QUERY_BASED_SCRIPT_FIX_PHASE1_PROMPT,
            QUERY_BASED_SCRIPT_FIX_PHASE2_PROMPT,
            QUERY_BASED_SCRIPT_FIX_FINAL_PROMPT,
        )
        
        # ── 构建代码索引 ──
        code_index = code_query_tool.build_code_index()
        
        # ── 从 fix_context 提取配置 ──
        context_title = fix_context.get("context_title", "验证脚本")
        runtime_instructions = fix_context.get("runtime_infrastructure_instructions", "")
        script_instructions = fix_context.get("script_output_instructions", "")
        
        # ── 累积查询结果 ──
        all_queried_code = ""
        observation_summary = ""
        reasoning_summary = ""
        analysis_direction = ""
        query_results_this_round = ""
        
        # ── 构建通用 kwargs ──
        common_kwargs = {
            "context_title": context_title,
            "error_output": error_output[:2000],
            "original_code": original_code,
            "extra_context": extra_context,
            "runtime_infrastructure_instructions": runtime_instructions,
            "script_output_instructions": script_instructions,
        }
        
        system_prompt = (
            "你是一位 Python 专家，擅长修正验证脚本。"
            "你可以通过代码查询工具按需获取模型源码详情，"
            "然后基于精确的代码知识提出修正后的脚本。"
            "⚠ 重要: 如果脚本使用了模型 API, 请先查询源码确认正确的用法!"
        )
        
        for round_idx in range(max_query_rounds):
            logger.info(f"  🔍 [Verif Fix Query Round {round_idx + 1}/{max_query_rounds}]")
            
            # ── 构建当前轮的 prompt ──
            if round_idx == 0:
                kwargs = dict(common_kwargs)
                kwargs["code_index"] = code_index
                kwargs["queried_code"] = "(暂无 — 这是第一轮, 请提出你想查询的代码)"
                prompt = QUERY_BASED_SCRIPT_FIX_PHASE1_PROMPT.format(**kwargs)
            else:
                kwargs = dict(common_kwargs)
                kwargs["query_results"] = query_results_this_round
                kwargs["previous_observation"] = observation_summary
                kwargs["previous_analysis_direction"] = analysis_direction or "待确认"
                prompt = QUERY_BASED_SCRIPT_FIX_PHASE2_PROMPT.format(**kwargs)
            
            # ── 调用 LLM ──
            response = llm_retry_helper.call_llm(
                prompt=prompt,
                system_content=system_prompt,
                temperature=0.1,
                max_tokens=4096,
            )
            
            if response is None:
                logger.warning(f"  ✗ LLM call failed at query round {round_idx + 1}")
                if round_idx == 0:
                    return None
                break
            
            # ── 判断响应类型 ──
            parsed = self._parse_query_response(response)
            
            if parsed and parsed.get("phase") == "query":
                queries = parsed.get("queries", [])
                if not queries:
                    continue
                
                logger.info(f"  🔎 LLM queries: {len(queries)} requests")
                
                # 兼容简化格式
                normalized_queries = []
                for q in queries:
                    if isinstance(q, str):
                        cleaned_q = re.sub(
                            r'^\s*(SEARCH|REPLACE)\s*:\s*(class\s+|def\s+)?', '',
                            q.strip(), flags=re.IGNORECASE
                        ).strip().strip('"\'`<>')
                        normalized_queries.append({"action": "search_function", "args": {"name": cleaned_q}})
                    elif isinstance(q, dict):
                        normalized_queries.append(q)
                
                if not normalized_queries:
                    continue
                
                query_results_this_round = code_query_tool.execute_queries(normalized_queries)
                
                if len(query_results_this_round) > max_chars_per_query_result:
                    query_results_this_round = query_results_this_round[:max_chars_per_query_result] + \
                        f"\n\n⚠ 截断到 {max_chars_per_query_result} 字符"
                
                all_queried_code += "\n\n" + query_results_this_round
                code_query_tool.refresh_cache()
                
                observation_summary = parsed.get("observation", observation_summary)
                reasoning_summary = parsed.get("reasoning", reasoning_summary)
                analysis_direction = parsed.get("analysis_direction", "")
            
            else:
                # 不是 query → 尝试作为修正后的脚本
                fixed_code = clean_code_response(response)
                if fixed_code and len(fixed_code) > 20:
                    logger.info(f"  ✓ LLM produced fixed script after {round_idx + 1} round(s)")
                    return fixed_code
                
                if round_idx == 0:
                    return None
                break
        
        # ── 查询轮数用尽 → 强制输出 ──
        kwargs = dict(common_kwargs)
        kwargs["all_queried_code"] = all_queried_code
        kwargs["observation_summary"] = observation_summary
        kwargs["reasoning_summary"] = reasoning_summary
        final_prompt = QUERY_BASED_SCRIPT_FIX_FINAL_PROMPT.format(**kwargs)
        
        response = llm_retry_helper.call_llm(
            prompt=final_prompt,
            system_content=system_prompt,
            temperature=0.1,
            max_tokens=4096,
        )
        
        if response is None:
            return None
        
        fixed_code = clean_code_response(response)
        if fixed_code and len(fixed_code) > 20:
            return fixed_code
        
        return None
    
    def _parse_query_response(self, response: str) -> Optional[dict]:
        """解析 LLM 在查询模式中的回复 — 只识别 query 阶段的 JSON"""
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
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
                import ast as ast_module
                data = ast_module.literal_eval(json_str)
                if not isinstance(data, dict):
                    return None
            except Exception:
                return None
        
        phase = data.get("phase", "")
        if phase == "query" and "queries" in data:
            return data
        
        return None