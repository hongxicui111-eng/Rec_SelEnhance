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
    
    def generate_minimal_runtime(self, data_file: str, output_file: str) -> str:
        """
        生成最小运行时基础设施
        
        只包含:
        1. OUTPUT_FILE 常量 (脚本必须有输出目标)
        2. DATA_FILE 常量 (数据文件路径, LLM 可用 json.load(DATA_FILE) 加载)
        3. save_result() 辅助函数 (安全写入结果)
        
        不包含:
        - import 语句 (LLM 自主决定需要什么)
        - 数据加载逻辑 (LLM 自主编写)
        - 变量解析 (LLM 自主决定如何使用数据)
        """
        return f"""# ── 运行时基础设施 (最小注入) ──
DATA_FILE = "{data_file}"
OUTPUT_FILE = "{output_file}"

def save_result(result_dict):
    # 保存验证结果到 JSON 文件
    import json, os, sys
    try:
        os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(result_dict, f, ensure_ascii=False, indent=2)
    except Exception as e:
        sys.stderr.write(f'Failed to save result: {{e}}\\n')
        sys.stderr.write(json.dumps(result_dict, ensure_ascii=False) + '\\n')

"""
    
    def wrap_script(self, code: str, verification_data: Dict,
                     output_file: str) -> str:
        """
        完整流程: 序列化数据 + 生成最小运行时 + 组合脚本
        
        LLM 的代码自主包含:
        - import 语句 (LLM 自己决定需要什么)
        - 数据加载逻辑 (LLM 通过 DATA_FILE 自主加载)
        - 验证逻辑 (LLM 自主编写)
        - save_result() 调用 (LLM 自主调用)
        
        我们只提供:
        - DATA_FILE / OUTPUT_FILE 常量
        - save_result() 函数定义
        """
        data_file = self.serialize_data(verification_data)
        runtime = self.generate_minimal_runtime(data_file, output_file)
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
            runtime_markers = ("DATA_FILE", "OUTPUT_FILE", "def save_result",
                               "# ── 运行时基础设施")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and \
                   not any(stripped.startswith(m) for m in runtime_markers):
                    core_start = i
                    break
        
        if core_start > 0:
            return "\n".join(lines[core_start:])
        return full_code


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
    ) -> Tuple[Optional[Dict], Optional[str]]:
        """
        执行验证代码, 如果失败则修正重试
        
        流程: 保存代码 → 执行 → 检查错误 → 带错误信息让 LLM 修正 → 再执行
        
        修正后重新 wrap_script (注入最小运行时基础设施),
        LLM 的核心验证代码保持不变。
        """
        from .llm_utils import clean_code_response
        
        code = initial_code
        last_error = None
        
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
                    
                    # 从代码中提取数据路径
                    data_match = re.search(r'DATA_FILE\s*=\s*"([^"]+)"', code)
                    data_path = data_match.group(1) if data_match else None
                    
                    fix_prompt = fix_prompt_template.format(
                        original_code=core_code,
                        error_output=error[:1500],
                        output_file_path=output_path,
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
                    
                    # 重新 wrap: 注入最小运行时基础设施
                    data_file = injector.serialize_data(verification_data)
                    runtime = injector.generate_minimal_runtime(data_file, output_path)
                    code = runtime + "\n" + fixed_core
                else:
                    logger.error(f"Max code fix rounds reached for {hypothesis_id}")
        
        return None, last_error