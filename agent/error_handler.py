"""
LLM 输出解析与错误处理模块
- 处理 LLM 输出格式异常
- 从自由文本中提取 diff / JSON / 配置变更
- 自动修复常见语法错误
"""
import re
import json
import logging
from typing import Optional

logger = logging.getLogger("rec_self_evolve.error_handler")


class ProposalParser:
    """
    解析 LLM 的输出提案, 处理多种格式异常
    对应 Google 论文 L1: Delta-based 配置生成减少幻觉
    对应 Self-EvolveRec: "linter" persona 审查代码
    """

    @staticmethod
    def parse(raw_text: Optional[str]) -> dict:
        """
        解析 LLM 输出, 返回结构化的提案
        返回格式:
        {
            "valid": bool,
            "explanation": str,
            "diff": str (代码变更),
            "diff_type": "python" | "config" | "yaml" | "unknown",
            "confidence": float (0-1),
            "error": str (解析失败原因, 仅 valid=False),
            "action": str (恢复策略),
        }
        """
        if not raw_text:
            return {
                "valid": False,
                "error": "LLM output is empty",
                "action": "skip_iteration"
            }

        # 异常: 输出过短
        if len(raw_text) < 50:
            return {
                "valid": False,
                "error": f"Output too short ({len(raw_text)} chars)",
                "action": "regenerate",
                "raw_output": raw_text
            }

        # 异常: LLM 拒绝生成 (安全对齐拒绝)
        refusal_patterns = [
            r"(?i)i.?m (?:sorry|unable|cannot|not able)",
            r"(?i)i can.?t (?:help|assist|generate)",
            r"(?i)as an? (?:ai|language model)",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, raw_text[:200]):
                return {
                    "valid": False,
                    "error": "LLM refused to generate (safety alignment triggered)",
                    "action": "regenerate_with_steering",
                    "raw_output": raw_text
                }

        # --- 尝试提取 diff ---
        result = ProposalParser._extract_diff(raw_text)
        if result["valid"]:
            return result

        # --- 尝试提取 JSON 配置变更 ---
        result = ProposalParser._extract_json(raw_text)
        if result["valid"]:
            return result

        # --- 尝试提取 YAML 变更 ---
        result = ProposalParser._extract_yaml(raw_text)
        if result["valid"]:
            return result

        # --- 无法解析: 返回修复建议 ---
        return {
            "valid": False,
            "error": "无法从 LLM 输出中提取有效代码变更",
            "action": "regenerate_with_format_reminder",
            "raw_output": raw_text[:1000],
        }

    # ════════════════════════════════════════
    # Diff 提取
    # ════════════════════════════════════════

    @staticmethod
    def _extract_diff(text: str) -> dict:
        """从文本中提取代码 diff, 支持多种格式"""

        # 格式 1: ```python ... ``` 或 ```diff ... ```
        code_block_pattern = r'```(?:python|diff|py|code)?\s*\n(.*?)```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)

        if matches:
            code = matches[0].strip()
            if len(code) > 10:
                explanation = ProposalParser._extract_explanation(text, code)

                # 子检查: 语法正确性
                syntax_issues = ProposalParser._check_syntax(code)

                return {
                    "valid": True,
                    "explanation": explanation,
                    "diff": code,
                    "diff_type": "python",
                    "confidence": 0.8 if not syntax_issues else 0.6,
                    "syntax_issues": syntax_issues,
                    "action": "proceed" if not syntax_issues else "needs_fix",
                }

        # 格式 2: 统一 diff 格式 (--- a/... +++ b/...)
        unified_pattern = r'(?:^|\n)(--- [^\n]+\n\+\+\+ [^\n]+\n(?:@@[^\n]*\n(?:[+- ].*\n?)*))'
        unified_match = re.search(unified_pattern, text, re.MULTILINE)
        if unified_match:
            return {
                "valid": True,
                "diff": unified_match.group(0),
                "diff_type": "unified_diff",
                "confidence": 0.9,
                "action": "proceed",
            }

        # 格式 3: 行内建议 (如 "将第10行的 X 改为 Y")
        inline_pattern = r'(?:将|把|修改|替换|change|replace|modify).*?第?\d+\s*[行:].*?(?:改为|为|to).+'
        if re.search(inline_pattern, text, re.IGNORECASE):
            explanation = ProposalParser._extract_explanation(text, "")
            return {
                "valid": True,
                "explanation": explanation,
                "diff": text,  # 需要人类辅助确认
                "diff_type": "natural_language",
                "confidence": 0.3,
                "action": "needs_human_review",
            }

        return {"valid": False}

    @staticmethod
    def _extract_explanation(text: str, code: str) -> str:
        """从文本中提取人类可读的解释"""
        # 优先提取 explanation 字段
        exp_pattern = r'"explanation"\s*:\s*"([^"]+)"'
        exp_match = re.search(exp_pattern, text)
        if exp_match:
            return exp_match.group(1)

        # 去掉代码块后取前面的文字
        text_without_code = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        # 取前 300 字符作为解释
        clean = text_without_code.strip().replace('\n', ' ')[:300]
        return clean

    # ════════════════════════════════════════
    # JSON / YAML 提取
    # ════════════════════════════════════════

    @staticmethod
    def _extract_json(text: str) -> dict:
        """尝试提取 JSON 格式的配置变更"""
        # 匹配 {...} 或 [{...}]
        json_pattern = r'(\{.*"diff".*"explanation".*\})'
        match = re.search(json_pattern, text, re.DOTALL)
        if not match:
            json_pattern = r'(\[[\s\S]*?"diff"[\s\S]*?\])'
            match = re.search(json_pattern, text, re.DOTALL)

        if match:
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, list):
                    parsed = parsed[0]
                return {
                    "valid": True,
                    "explanation": parsed.get("explanation", ""),
                    "diff": json.dumps(parsed.get("diff", parsed), indent=2),
                    "diff_type": "json",
                    "confidence": 0.85,
                    "action": "proceed",
                }
            except json.JSONDecodeError:
                pass

        return {"valid": False}

    @staticmethod
    def _extract_yaml(text: str) -> dict:
        """尝试提取 YAML 格式的配置变更"""
        yaml_block = r'```yaml\s*\n(.*?)```'
        match = re.search(yaml_block, text, re.DOTALL)
        if match:
            return {
                "valid": True,
                "diff": match.group(1).strip(),
                "diff_type": "yaml",
                "confidence": 0.75,
                "action": "proceed",
            }
        return {"valid": False}

    # ════════════════════════════════════════
    # 语法检查
    # ════════════════════════════════════════

    @staticmethod
    def _check_syntax(code: str) -> list:
        """
        检查 Python 语法, 返回问题列表
        对应 Google 论文 Phase II: Compilation Check
        """
        issues = []

        # 检查: import 缺失
        import_pattern = r'(?:^|\n)(?:from|import)\s+(\S+)'
        imports = re.findall(import_pattern, code)
        for imp in imports:
            imp_name = imp.split('.')[0].split(' import ')[-1].strip()
            if imp_name in {"tensorflow", "torch", "numpy", "pandas", "sklearn",
                             "transformers", "scipy"}:
                continue  # 主流库, 大概率已安装
            issues.append(f"verify_import: {imp_name}")

        # 检查: 括号匹配
        stack = []
        pairs = {'(': ')', '[': ']', '{': '}'}
        in_string = False
        string_char = None
        for ch in code:
            if ch in '"\'' and not in_string:
                in_string = True
                string_char = ch
            elif ch == string_char and in_string:
                in_string = False
                string_char = None
            if not in_string and ch in pairs:
                stack.append(ch)
            elif not in_string and ch in pairs.values():
                if not stack or pairs[stack.pop()] != ch:
                    issues.append("unmatched_bracket")
                    break
        if stack:
            issues.append("unclosed_bracket")

        return issues

    # ════════════════════════════════════════
    # 自动修复
    # ════════════════════════════════════════

    @staticmethod
    def auto_fix_diff(diff: str, issues: list) -> str:
        """
        自动修复常见语法问题
        对应 Google 论文: "linter" persona 审查并修复代码
        """
        if not issues:
            return diff

        fixed = diff

        for issue in issues:
            if issue.startswith("verify_import:"):
                pkg = issue.split(":")[1]
                # 添加 try/except import
                if pkg not in fixed:
                    fixed = f"try:\n    import {pkg}\nexcept ImportError:\n    pass\n\n{fixed}"

        return fixed


class LLMFixer:
    """
    当解析失败时, 让 LLM 自己修正输出格式
    对应 Google 论文: "Think-Code-Verify" 中的 Refinement by LLM
    """

    def __init__(self, llm_client):
        self.llm = llm_client

    def fix_format(self, raw_output: str, error_reason: str) -> str:
        """
        让 LLM 重新输出, 强制要求正确的格式
        """
        from .prompts import FORMAT_FIX_PROMPT
        prompt = FORMAT_FIX_PROMPT.format(
            raw_output=raw_output,
            error_reason=error_reason,
        )
        result = self.llm.chat([
            {"role": "user", "content": prompt}
        ], temperature=0.1, max_tokens=4096)
        return result or raw_output