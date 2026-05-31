#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM 交互共享工具 — 供 core.py 和 hypothesis_verification_agent.py 共用的 LLM 交互基础设施

提取的共享功能:
  1. 健壮 JSON 解析 (多策略: 标准→literal_eval→修复→模糊匹配)
  2. JSON block 提取 (markdown code block / 裸 JSON)
  3. JSON 错误诊断 (定位错误位置 + 人类可读描述)
  4. LLM 调用 + 解析 + 自动重试 (带错误反馈的闭环)
  5. 代码响应清理 (去掉 markdown 标记、行号前缀、解释文字)
  6. Markdown wrapper 清理 (只去包裹, 保留内容)

设计原则:
  - 所有函数为纯函数或独立类方法, 不依赖 self (可被任何模块直接导入使用)
  - LLMRetryHelper 提供 call_and_parse_with_retry 方法, 需要 llm_client 实例
  - 不包含任何预定义 prompt 内容 (prompt 由各模块自行管理)
"""

import json
import re
import logging
import ast
from typing import Dict, Optional, Callable

logger = logging.getLogger("rec_self_evolve.llm_utils")


# ════════════════════════════════════════
# 思考模式前缀清理
# ════════════════════════════════════════

def _strip_thinking_prefix(response: str) -> str:
    """
    截断 thinking 模型的思考过程，只保留实际回复部分
    
    Thinking 模型（如 Qwen3/Qwen3.5）在思考模式下会首先输出一段思考内容，
    以 </think> 标记结束思考块，之后才是实际回复。思考过程中可能包含 JSON
    代码块等干扰内容，需要在 JSON 提取之前先截断。
    
    示例:
        输入:
            <think>
            [思考过程...]
            ```json
            { "key": "value" }  ← 思考中的示例 JSON，非实际输出
            ```
            </think>
            { "real_key": "real_value" }  ← 实际输出
            
        输出:
            { "real_key": "real_value" }
    
    Args:
        response: LLM 原始响应文本
        
    Returns:
        如果检测到 </think> 标记，返回标记之后的内容；
        否则返回原始响应（非 thinking 模式或没有 thinking 内容）
    """
    if not response:
        return response
    
    # 匹配思考模型结束标记 </think>
    # 思考模型的输出格式: <think>...思考内容...</think>\n实际回复
    # 我们只需要找到 </think> 标记，截断之前的一切思考内容
    match = re.search(r'</think>\s*', response)
    if match:
        end_pos = match.end()
        result = response[end_pos:].strip()
        if result:
            logger.debug(
                f"Stripped thinking prefix: {len(response)} chars → "
                f"{len(result)} chars (</think> at pos {match.start()})"
            )
            return result
    
    return response


# ════════════════════════════════════════
# JSON 解析工具 (多策略健壮解析)
# ════════════════════════════════════════

def extract_json_block(response: str) -> Optional[str]:
    """
    从 LLM 回复中提取 JSON block
    
    策略:
    0. 先截断 thinking 前缀 (思考模型的 `response` 标记之前的内容)
    1. 优先提取 markdown code block (```json ... ```)
    2. 回退: 找第一个 { 到最后 }
    
    Args:
        response: LLM 响应文本
        
    Returns:
        JSON 字符串, 或 None
    """
    # 截断思考模型前缀
    response = _strip_thinking_prefix(response)
    
    # 优先提取 markdown code block
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
    if json_match:
        return json_match.group(1)
    
    # 回退: 找第一个 { 到最后 }
    start = response.find('{')
    end = response.rfind('}')
    if start >= 0 and end > start:
        return response[start:end + 1]
    
    return None


def _normalize_ast_result(obj):
    """
    将 ast.literal_eval 的结果中的非 JSON 类型规范化:
    - set → list (排序以保持稳定顺序)
    - tuple → list
    - 递归处理嵌套的 dict / list
    
    如果规范化后发现顶层是 set/tuple 而非 dict/list, 返回 None
    """
    if isinstance(obj, set):
        return sorted(obj, key=str)
    if isinstance(obj, tuple):
        return list(obj)
    if isinstance(obj, dict):
        return {k: _normalize_ast_result(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_ast_result(item) for item in obj]
    # 基本类型 (str, int, float, bool, None) — 直接返回
    return obj


def robust_json_parse(json_str: str) -> Optional[Dict]:
    """
    多策略 JSON 解析, 带模糊修复
    
    Strategy 1: 标准 json.loads
    Strategy 2: ast.literal_eval (Python 字面量)
    Strategy 3: 修复常见格式问题后重试 (注释、尾随逗号、Python bool)
    Strategy 4: 修复缺失引号的键
    Strategy 5: 单引号→双引号
    Strategy 6: 修复冒号后缺失引号的字符串值
    
    Args:
        json_str: 待解析的 JSON 字符串
        
    Returns:
        解析后的 dict/list, 或 None
    """
    # --- Strategy 1: 标准解析 ---
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # --- Strategy 2: Python literal ---
    try:
        parsed = ast.literal_eval(json_str)
        if isinstance(parsed, (dict, list)):
            parsed = _normalize_ast_result(parsed)
            if parsed is not None:
                return parsed
    except Exception:
        pass
    
    # --- Strategy 3: 修复常见 JSON 格式问题 ---
    fixed = json_str
    
    # 3a: 移除注释 (// 和 /* */)
    fixed = re.sub(r'//[^\n]*', '', fixed)
    fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
    
    # 3b: 移除尾随逗号 (在 } 和 ] 之前)
    fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
    
    # 3c: 将 Python None/True/False 转为 JSON null/true/false
    fixed = fixed.replace(': None', ': null')
    fixed = fixed.replace(': True', ': true')
    fixed = fixed.replace(': False', ': false')
    
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass
    
    # --- Strategy 4: 修复缺失引号的键 ---
    fixed2 = re.sub(
        r'(?<![:"\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
        r'"\1":',
        fixed
    )
    try:
        return json.loads(fixed2)
    except json.JSONDecodeError:
        pass
    
    # --- Strategy 5: 单引号→双引号 ---
    fixed3 = fixed2.replace("'", '"')
    try:
        return json.loads(fixed3)
    except json.JSONDecodeError:
        pass
    
    # --- Strategy 6: 修复冒号后缺失引号的字符串值 ---
    # 常见 LLM 输出错误: "description":修改损失函数... (值缺少开头双引号)
    fixed4 = _fix_unquoted_string_values(fixed3)
    try:
        return json.loads(fixed4)
    except json.JSONDecodeError:
        pass
    
    return None


def _fix_unquoted_string_values(text: str) -> str:
    """
    修复 JSON 字符串值缺少双引号的问题
    
    常见 LLM 输出错误模式:
        "description":修改损失函数结构，从纯 InfoNCE 改为...
    期望修复为:
        "description": "修改损失函数结构，从纯 InfoNCE 改为..."
    
    算法:逐行扫描，当发现 "key": <裸值> 模式时，将裸值用双引号包裹。
    裸值判定:冒号后的值不以合法 JSON 值起始字符开始。
    合法 JSON 值起始字符: " { [ 数字 - t/f/n
    
    Args:
        text: 待修复的 JSON 文本
        
    Returns:
        修复后的文本
    """
    valid_value_starts = '"{[-tfnN'
    
    lines = text.split('\n')
    fixed_lines = []
    
    for line in lines:
        m = re.match(
            r'^(\s*"[^"]+"\s*:\s*)'
            r'([^\s' + valid_value_starts + r'][^,\n]*?)'
            r'(\s*,?\s*)$',
            line
        )
        if m:
            prefix = m.group(1)
            value = m.group(2).rstrip()
            suffix = m.group(3)
            has_comma = value.endswith(',') or suffix.strip() == ','
            value = value.rstrip(',').rstrip()
            value = value.replace('"', '\\"')
            if has_comma:
                fixed_lines.append(f'{prefix}"{value}",')
            else:
                fixed_lines.append(f'{prefix}"{value}"')
        else:
            fixed_lines.append(line)
    
    return '\n'.join(fixed_lines)


def diagnose_json_error(response: str) -> str:
    """
    诊断 JSON 解析错误的具体原因
    
    Args:
        response: LLM 响应文本
        
    Returns:
        人类可读的错误描述
    """
    if not response:
        return "空响应"
    
    # 提取 JSON block
    json_str = extract_json_block(response)
    if json_str is None:
        return "无法从响应中提取 JSON block (缺少 { 或 ```json 标记)"
    
    # 尝试加载并报告具体错误
    try:
        json.loads(json_str)
        return "标准解析看似正常 (可能 validation 阶段失败)"
    except json.JSONDecodeError as e:
        pos = e.pos
        context_start = max(0, pos - 40)
        context_end = min(len(json_str), pos + 40)
        context = json_str[context_start:context_end]
        
        diagnosis_parts = [f"JSON 解析错误 (位置 {pos}): {e.msg}"]
        diagnosis_parts.append(f"附近上下文: ...{context}...")
        
        # 常见问题诊断
        snippet = json_str[max(0, pos - 5):min(len(json_str), pos + 5)]
        
        if "Expecting ',' delimiter" in str(e):
            diagnosis_parts.append("诊断: 可能在对象/数组内缺少逗号分隔符")
        elif "Expecting property name" in str(e) or "Expecting ':' delimiter" in str(e):
            if pos > 0 and json_str[pos-1] == "'":
                diagnosis_parts.append("诊断: 有单引号未转为双引号")
            else:
                diagnosis_parts.append("诊断: 键名缺少双引号或冒号")
        elif "Extra data" in str(e):
            diagnosis_parts.append("诊断: JSON 后有额外内容 (多个 JSON 对象)")
        elif "Unterminated string" in str(e):
            diagnosis_parts.append("诊断: 字符串未正确结束 (包含未转义的控制字符)")
        elif "Expecting value" in str(e):
            diagnosis_parts.append("诊断: 预期值位置出现意外字符")
        
        diagnosis_parts.append(f"错误附近字符: ...{snippet}...")
        
        return "\n".join(diagnosis_parts)


def parse_json_from_response(response: str) -> Optional[Dict]:
    """
    从 LLM 回复中解析 JSON (多策略健壮解析)
    
    组合: _strip_thinking_prefix → extract_json_block → robust_json_parse
    
    Args:
        response: LLM 响应文本
        
    Returns:
        解析后的 dict, 或 None
    """
    json_str = extract_json_block(response)
    if json_str is None:
        logger.warning("Cannot extract JSON from response")
        return None
    
    return robust_json_parse(json_str)


# ════════════════════════════════════════
# 代码响应清理
# ════════════════════════════════════════

def clean_code_response(response: str) -> str:
    """
    清理 LLM 生成的代码回复
    
    去掉:
    - markdown code block 标记 (```python ... ```)
    - 开头/结尾的解释文字
    - 行号前缀
    
    Args:
        response: LLM 生成的代码回复
        
    Returns:
        清理后的纯代码字符串
    """
    # 提取 markdown code block
    code_match = re.search(
        r'```(?:python[3]?)?\s*\n(.*?)\n?```', response, re.DOTALL
    )
    if code_match:
        code = code_match.group(1)
    else:
        code = response
    
    # 去掉行号前缀 (如 "  123→" 或 "123|")
    lines = code.split("\n")
    cleaned = []
    for line in lines:
        line = re.sub(r'^\s*\d+[→|]\s*', '', line)
        cleaned.append(line)
    code = "\n".join(cleaned)
    
    # 去掉开头的解释文字 — 找到第一个代码行
    code_lines = []
    found_start = False
    for line in code.split("\n"):
        stripped = line.strip()
        if not found_start:
            if stripped.startswith("import ") or \
               stripped.startswith("from ") or \
               stripped.startswith("def ") or \
               stripped.startswith("class ") or \
               stripped.startswith("#") or \
               stripped == "" or \
               "=" in stripped or \
               stripped.startswith("if ") or \
               stripped.startswith("for ") or \
               stripped.startswith("while ") or \
               stripped.startswith("try:") or \
               stripped.startswith("with ") or \
               stripped.startswith("@"):
                found_start = True
                code_lines.append(line)
        else:
            code_lines.append(line)
    
    return "\n".join(code_lines) if code_lines else code


def clean_markdown_wrapper(text: str) -> str:
    """
    只清理 markdown 代码块标记和多余空白, 不做"移除解释性前缀"处理
    
    这是 SEARCH/REPLACE 格式的安全清理 — 保留所有代码内容
    
    Args:
        text: 可能包含 markdown 包裹的文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    # 移除 markdown 代码块标记
    text = re.sub(r'^```(?:python)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    # 去除首尾多余空行 (但保留代码内部的空行)
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)


# ════════════════════════════════════════
# LLM 调用 + 解析 + 自动重试
# ════════════════════════════════════════

# 通用 JSON 修正 Prompt (不包含任何领域特定内容)
JSON_FIX_PROMPT_TEMPLATE = """你之前输出的 JSON 格式有误, 请根据以下**原始输出**和**解析错误**信息, 重新输出**正确的 JSON**。

## 你之前的原始输出 (完整内容, 未截断)
```
{raw_response_full}
```

## 解析错误
{parse_error}

## 修正要求
1. 保持内容不变, 只修复 JSON 格式
2. 确保是**严格合法的 JSON** (双引号, 无结尾逗号, 无注释)
3. **只输出 JSON**, 不要解释文字, 不要 markdown 标记
{additional_instructions}"""


class LLMRetryHelper:
    """
    LLM 调用 + 解析 + 自动重试工具
    
    模式: LLM 调用 → 健壮 JSON 解析 → 结构验证
          ↓ 解析/验证失败
          错误诊断 + 重试 (带 JSON_FIX_PROMPT_TEMPLATE)
          ↓ 重试耗尽
          None
    
    使用方式:
    ```python
    helper = LLMRetryHelper(llm_client)
    result = helper.call_and_parse_with_retry(
        prompt="...",
        system_content="...",
        temperature=0.3,
        max_tokens=None,   # None = 使用 llm_client.default_max_tokens (统一配置)
        max_retries=2,
        additional_instructions="...",
        validate_func=my_validate_func,
    )
    ```
    """
    
    def __init__(self, llm_client):
        """
        Args:
            llm_client: LLMClient 实例 (必须有 .chat() 方法)
        """
        self.llm = llm_client
    
    def call_and_parse_with_retry(
        self,
        prompt: str,
        system_content: str,
        temperature: float = 0.3,
        max_tokens: int = None,
        max_retries: int = 2,
        additional_instructions: str = "",
        validate_func: Optional[Callable[[Dict], bool]] = None,
        suppress_response_log: bool = False,
    ) -> Optional[Dict]:
        """
        通用 JSON 调用 + 解析 + 自动重试
        
        Args:
            prompt: 调用的 prompt
            system_content: system message 内容
            temperature: LLM temperature
            max_tokens: 最大 token 数 (None=使用 llm_client 的 default_max_tokens)
            max_retries: 最大重试次数
            additional_instructions: JSON_FIX_PROMPT 额外说明
            validate_func: 可选验证函数, 接收 parsed dict, 返回 bool
            suppress_response_log: 如果为 True, 不输出响应日志 (用于代码生成等场景)
            
        Returns:
            Parsed JSON dict, 或 None
        """
        if max_tokens is None:
            max_tokens = getattr(self.llm, 'default_max_tokens', 4096)
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            suppress_response_log=suppress_response_log,
        )
        
        if response is None:
            logger.error("LLM call returned None")
            return None
        
        for attempt in range(max_retries + 1):  # 首次 + retries
            parsed = parse_json_from_response(response)
            
            if parsed is not None:
                # 验证 (如果提供了验证函数)
                if validate_func is None or validate_func(parsed):
                    if attempt > 0:
                        logger.info(f"JSON parsed successfully on retry #{attempt}")
                    return parsed
                else:
                    logger.warning("JSON parsed but validation failed")
                    if attempt >= max_retries:
                        return None
            
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries + 1} attempts exhausted. "
                    f"Response preview: {(response if response else 'None')}..."
                )
                return None
            
            # 准备重试 — 传递完整原始响应, 不截断 (截断会导致 LLM 无法重建完整 JSON)
            # 先剔除 thinking 模型的思考前缀, 避免干扰 LLM 修正 JSON
            raw_response_full = _strip_thinking_prefix(response) or "None"
            parse_error = diagnose_json_error(response)
            
            retry_prompt = JSON_FIX_PROMPT_TEMPLATE.format(
                raw_response_full=raw_response_full,
                parse_error=parse_error,
                additional_instructions=additional_instructions,
            )
            
            logger.info(
                f"JSON retry #{attempt + 1}/{max_retries} "
                f"(parse error: {parse_error}...)"
            )
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的数据专家。你之前输出的 JSON 格式有误，"
                        "请重新输出，确保是严格合法的 JSON 格式且内容正确。"
                    )},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            
            if response is None:
                logger.error("No response from LLM during JSON retry")
                return None
        
        return None
    
    def call_llm(self, prompt: str, system_content: str,
                 temperature: float = 0.3, max_tokens: int = None,
                 suppress_response_log: bool = False) -> Optional[str]:
        """
        简单 LLM 调用 (不带解析/重试)
        
        Args:
            prompt: 用户 prompt
            system_content: system message
            temperature: 温度
            max_tokens: 最大 token (None=使用 llm_client 的 default_max_tokens)
            suppress_response_log: 如果为 True, 不输出响应日志 (用于代码生成等场景)
            
        Returns:
            LLM 响应字符串, 或 None
        """
        if max_tokens is None:
            max_tokens = getattr(self.llm, 'default_max_tokens', 4096)
        return self.llm.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            suppress_response_log=suppress_response_log,
        )