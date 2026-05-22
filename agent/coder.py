"""
Coder Agent 模块 - 负责代码生成和自动调试

功能：
1. 根据研究想法生成代码
2. 自动调试和修复错误
3. 支持多轮反思优化
4. 代码版本管理

设计原则 (模仿 Self-EvolveRec):
  - 使用 CODER_INSTRUCTIONS 模板生成代码修改 (SEARCH/REPLACE 格式)
  - 使用 DEBUGGER_INSTRUCTIONS 模板进行代码验证和调试
  - Self_EvolveRec-BLOCK 标记追踪代码修改历史
"""

import asyncio
import json
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from agent.llm_client import LLMClient
from agent.prompts import CODER_INSTRUCTIONS, DEBUGGER_INSTRUCTIONS

logger = logging.getLogger("rec_self_evolve.coder")


@dataclass
class CodeChange:
    """代码变更"""
    target_file: str
    target_class_or_function: str
    description: str
    new_code: str
    insert_position: str  # replace_function / replace_class / append_to_file
    expected_effect: str
    confidence: str = "中"


@dataclass
class CodeResult:
    """代码生成结果"""
    code: str
    changes: List[CodeChange] = field(default_factory=list)
    diff_text: str = ""
    success: bool = True
    error: Optional[str] = None


class CoderAgent:
    """
    编码 Agent - 负责代码生成和调试
    
    对比 DeepEvolve 的设计：
    - 集成代码生成和调试
    - 支持多轮反思
    - 使用结构化变更格式
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "gpt-4o",
        temperature: float = 0.4,
        max_reflection_times: int = 3,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_reflection_times = max_reflection_times
        
        self.llm_client = LLMClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # 主题上下文
        self.query = ""
        self.problem_name = ""
        self.problem_description = ""
        
    def update_topic(
        self,
        query: str,
        problem_name: str,
        problem_description: str,
    ):
        """更新编码主题"""
        self.query = query
        self.problem_name = problem_name
        self.problem_description = problem_description
        logger.info(f"Coder updated topic: {problem_name}")
        
    async def run(
        self,
        new_idea: Any,
        parent: Any,
        inspirations: List[Any],
        trace_id: str = "",
        max_reflection_times: Optional[int] = None,
    ) -> Tuple[List[str], List[str]]:
        """
        运行编码流程
        
        Args:
            new_idea: 研究想法
            parent: 父程序
            inspirations: 灵感列表
            trace_id: 追踪ID
            max_reflection_times: 最大反思次数
            
        Returns:
            (all_diff_text, all_program_code)
        """
        max_reflection = max_reflection_times or self.max_reflection_times
        
        all_diff_text = []
        all_program_code = []
        
        # 初始代码生成
        diff_text, program_code = await self._generate_code(
            new_idea, parent, inspirations
        )
        all_diff_text.append(diff_text)
        all_program_code.append(program_code)
        
        # 反思和调试循环
        for reflection_idx in range(max_reflection):
            # 检查代码是否有效
            is_valid = await self._validate_code(program_code)
            
            if is_valid:
                logger.info(f"Code validation passed at reflection {reflection_idx}")
                break
                
            # 调试并生成修复后的代码
            logger.info(f"Code validation failed at reflection {reflection_idx}, attempting debug")
            
            debug_result = await self._debug_code(
                program_code, parent, new_idea, reflection_idx
            )
            
            if debug_result.success:
                all_diff_text.append(debug_result.diff_text)
                all_program_code.append(debug_result.code)
                program_code = debug_result.code
            else:
                logger.warning(f"Debug failed: {debug_result.error}")
                break
                
        return all_diff_text, all_program_code
        
    async def _generate_code(
        self,
        new_idea: Any,
        parent: Any,
        inspirations: List[Any],
    ) -> Tuple[str, str]:
        """生成代码 — 使用 CODER_INSTRUCTIONS 模板"""
        
        # ── 优先使用 CODER_INSTRUCTIONS 模板 ──
        # 构建研究思路描述
        research_idea = ""
        if new_idea:
            research_idea = (
                f"{new_idea.title}: {new_idea.description}\n"
                f"实现思路: {new_idea.content}"
            )
            if hasattr(new_idea, 'supplement') and new_idea.supplement:
                research_idea += f"\n补充: {new_idea.supplement}"
        
        # 构建目标指标
        target_metrics = json.dumps(parent.metrics, ensure_ascii=False) if parent and hasattr(parent, 'metrics') else '{}'
        
        # 构建源码上下文
        source_code_context = ""
        if parent and hasattr(parent, 'code'):
            source_code_context = parent.code[:8000]  # 限制长度
        
        # 添加灵感信息
        if inspirations:
            source_code_context += "\n\n## 历史成功案例\n"
            for i, insp in enumerate(inspirations[:2]):
                if hasattr(insp, 'idea') and hasattr(insp.idea, 'title'):
                    source_code_context += (
                        f"- {insp.idea.title}: 指标 {json.dumps(insp.metrics, ensure_ascii=False)}\n"
                    )
        
        # 使用 CODER_INSTRUCTIONS 模板
        prompt = CODER_INSTRUCTIONS.format(
            research_idea=research_idea or f"改进推荐系统 {self.problem_name}",
            target_metrics=target_metrics,
            source_code_context=source_code_context or f"推荐系统模型: {self.problem_name}\n问题描述: {self.problem_description}",
        )
        
        response = await self.llm_client.chat(prompt)
        
        # 解析响应 — 优先解析 SEARCH/REPLACE 格式
        try:
            result = self._parse_code_response_with_diff(response, research_idea)
            if result and result.success:
                return result.diff_text, result.code
        except Exception as e:
            logger.warning(f"Failed to parse CODER_INSTRUCTIONS response with diff: {e}")
        
        # 降级: 使用传统 JSON 格式解析
        try:
            result = self._parse_code_response(response)
            return result.diff_text, result.code
        except Exception as e:
            logger.warning(f"Failed to parse code response: {e}")
            # 返回父代码作为后备
            return "", getattr(parent, 'code', '')
    
    def _parse_code_response_with_diff(
        self,
        response: str,
        research_idea: str,
    ) -> Optional[CodeResult]:
        """
        解析包含 SEARCH/REPLACE diff 格式的 LLM 输出
        
        SEARCH/REPLACE 格式:
        <<<<<<< SEARCH
        # original code
        =======
        ### >>> Self_EvolveRec-BLOCK-START: <idea>
        # new code
        ### <<< Self_EvolveRec-BLOCK-END
        >>>>>>> REPLACE
        """
        # 提取所有 SEARCH/REPLACE 块
        diff_pattern = r'<<<<<<< SEARCH\s*\n(.*?)=======\s*\n(.*?)>>>>>>> REPLACE'
        matches = re.findall(diff_pattern, response, re.DOTALL)
        
        if not matches:
            return None
        
        changes = []
        all_new_code = []
        
        for search_code, replace_code in matches:
            search_code = search_code.strip()
            replace_code = replace_code.strip()
            
            # 提取 Self_EvolveRec 块描述
            block_idea = research_idea[:50]
            idea_match = re.search(
                r'Self_EvolveRec-BLOCK-START:\s*(.*?)(?:\n|$)',
                replace_code,
            )
            if idea_match:
                block_idea = idea_match.group(1).strip()
            
            # 推断目标文件和函数
            target_file = "Recmodel/model.py"  # 默认
            target_func = "unknown"
            
            class_match = re.search(r'class\s+(\w+)', search_code)
            func_match = re.search(r'def\s+(\w+)', search_code)
            if class_match:
                target_func = class_match.group(1)
            elif func_match:
                target_func = func_match.group(1)
            
            # 清理新代码 (去掉标记行)
            clean_code = self._strip_evolve_markers(replace_code)
            
            change = CodeChange(
                target_file=target_file,
                target_class_or_function=target_func,
                description=f"[Self_EvolveRec] {block_idea}",
                new_code=clean_code,
                insert_position="replace_function",
                expected_effect=f"Implement: {block_idea}",
                confidence="中",
            )
            changes.append(change)
            all_new_code.append(clean_code)
        
        # 生成 diff 文本
        diff_text = self._generate_diff_text(changes)
        
        # 合并代码
        combined_code = "\n\n".join(all_new_code)
        
        return CodeResult(
            code=combined_code,
            changes=changes,
            diff_text=diff_text,
            success=True,
        )
    
    def _strip_evolve_markers(self, code: str) -> str:
        """从代码中移除 Self_EvolveRec-BLOCK-START/END 标记行"""
        lines = code.split('\n')
        clean_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('### >>> Self_EvolveRec-BLOCK-START:'):
                # 保留注释内容但去掉标记格式
                idea = stripped.replace('### >>> Self_EvolveRec-BLOCK-START:', '').strip()
                clean_lines.append(f"# [Self_EvolveRec] {idea}")
            elif stripped == '### <<< Self_EvolveRec-BLOCK-END':
                continue  # 移除结束标记
            else:
                clean_lines.append(line)
        return '\n'.join(clean_lines)
            
    def _parse_code_response(self, response: str) -> CodeResult:
        """解析代码响应"""
        
        # 提取 JSON 部分
        json_match = re.search(r'\{{.*\}}', response, re.DOTALL)
        
        changes = []
        code = ""
        
        if json_match:
            try:
                data = json.loads(json_match.group())
                if 'changes' in data:
                    changes = [CodeChange(**c) for c in data['changes']]
            except Exception as e:
                logger.warning(f"JSON parse error: {e}")
                
        # 提取代码部分（查找 ```python ... ```）
        code_blocks = re.findall(r'```python\n(.*?)```', response, re.DOTALL)
        if code_blocks:
            code = code_blocks[0]
        else:
            # 如果没有代码块，尝试提取整个响应作为代码
            code = response
            
        # 生成 diff 文本
        diff_text = self._generate_diff_text(changes)
        
        return CodeResult(
            code=code,
            changes=changes,
            diff_text=diff_text,
            success=True
        )
        
    def _generate_diff_text(self, changes: List[CodeChange]) -> str:
        """生成 diff 文本"""
        diff_parts = []
        for change in changes:
            diff_parts.append(f"## {change.target_file}")
            diff_parts.append(f"### {change.target_class_or_function}")
            diff_parts.append(change.description)
            diff_parts.append("```python")
            diff_parts.append(change.new_code)
            diff_parts.append("```")
            diff_parts.append("")
        return "\n".join(diff_parts)
        
    async def _validate_code(self, code: str) -> bool:
        """验证代码语法"""
        try:
            import ast
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.warning(f"Syntax error: {e}")
            return False
            
    async def _debug_code(
        self,
        code: str,
        parent: Any,
        new_idea: Any,
        reflection_idx: int,
    ) -> CodeResult:
        """调试代码 — 使用 DEBUGGER_INSTRUCTIONS 模板"""
        
        # 首先验证代码获取详细错误
        try:
            import ast
            ast.parse(code)
        except SyntaxError as e:
            error_info = f"SyntaxError: {e}"
        except Exception as e:
            error_info = f"Error: {e}"
        else:
            error_info = "代码语法正确，但可能存在运行时错误"
        
        # ── 优先使用 DEBUGGER_INSTRUCTIONS 模板 ──
        # 构建当前代码上下文
        current_code = code[:6000]  # 限制长度
        parent_code = getattr(parent, 'code', '')[:3000]
        
        # 合并代码上下文
        full_current_code = f"## 有问题的代码\n```python\n{current_code}\n```\n\n## 父程序代码\n```python\n{parent_code}\n```"
        
        debugger_prompt = DEBUGGER_INSTRUCTIONS.format(
            current_code=full_current_code,
            error_info=error_info,
        )
        
        # 添加研究想法上下文
        if new_idea:
            debugger_prompt += f"\n\n## 研究想法\n标题: {new_idea.title}\n描述: {new_idea.description}"
        
        debugger_response = await self.llm_client.chat(debugger_prompt)
        
        # 解析 DEBUGGER_INSTRUCTIONS 输出 — 支持 SEARCH/REPLACE 和 JSON 格式
        # 尝试解析 SEARCH/REPLACE 格式
        debug_diff_result = self._parse_code_response_with_diff(
            debugger_response,
            f"Debug: {error_info[:50]}",
        )
        if debug_diff_result and debug_diff_result.success:
            debug_diff_result.diff_text = f"[DEBUG Attempt {reflection_idx + 1}]\n" + debug_diff_result.diff_text
            return debug_diff_result
        
        # 降级: 使用传统 JSON 格式解析
        # 构建传统格式 fallback prompt
        fallback_prompt = f"""你之前生成的代码存在问题，需要修复。

## 错误信息
{error_info}

## 父程序代码
```python
{getattr(parent, 'code', '')}
```

## 研究想法
标题: {new_idea.title}
描述: {new_idea.description}

请修复代码问题，确保修改后代码可以正常运行。

### 输出格式
```json
{{
  "changes": [
    {{
      "target_file": "Recmodel/model.py",
      "target_class_or_function": "SASRec",
      "description": "修复描述",
      "new_code": "修复后的完整代码",
      "insert_position": "replace_class",
      "expected_effect": "预期效果",
      "confidence": "高"
    }}
  ]
}}
```

**关键要求：**
- new_code 必须是完整的、可执行的 Python 代码
- 修复所有语法错误
- 确保维度对齐正确
"""
        
        fallback_response = await self.llm_client.chat(fallback_prompt)
        
        try:
            result = self._parse_code_response(fallback_response)
            result.diff_text = f"[DEBUG Attempt {reflection_idx + 1}]\n" + result.diff_text
            return result
        except Exception as e:
            return CodeResult(
                code=code,
                success=False,
                error=str(e),
            )


# 导出
__all__ = [
    'CoderAgent',
    'CodeChange',
    'CodeResult',
]
