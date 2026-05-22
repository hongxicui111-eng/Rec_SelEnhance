"""
Deep Research 模块 - 为推荐系统进化提供深度研究支持

功能：
1. 搜索网络获取推荐系统最新研究
2. 阅读论文并提取关键信息
3. 生成研究计划和报告
4. 结合历史灵感生成新想法

设计原则 (模仿 Self-EvolveRec):
  - 使用 RESEARCHER_INSTRUCTIONS 模板生成研究方案
  - 使用 SEARCH_INSTRUCTIONS 模板搜索文献
  - 使用 REFLECTION_INSTRUCTIONS 模板进行反思
  - 多轮迭代反思机制，持续优化研究思路
"""

import asyncio
import json
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime

from agent.llm_client import LLMClient
from agent.prompts import (
    RESEARCHER_INSTRUCTIONS, SEARCH_INSTRUCTIONS, REFLECTION_INSTRUCTIONS,
)

logger = logging.getLogger("rec_self_evolve.researcher")


@dataclass
class SearchResult:
    """搜索结果"""
    title: str
    url: str
    snippet: str
    source: str = "web"


@dataclass
class ResearchPlan:
    """研究计划"""
    title: str
    description: str
    expected_improvement: str
    implementation_hints: List[str] = field(default_factory=list)
    confidence: str = "中"


@dataclass
class ResearchReport:
    """研究报告"""
    idea: 'IdeaData'
    markdown_report: str
    search_results: List[SearchResult] = field(default_factory=list)
    plans: List[ResearchPlan] = field(default_factory=list)


@dataclass
class IdeaData:
    """想法数据"""
    title: str
    description: str
    content: str
    supplement: str = ""
    source: str = "research"
    confidence: str = "中"


class ResearcherAgent:
    """
    研究 Agent - 负责深度研究和想法生成
    
    对比 DeepEvolve 的设计：
    - 结合网络搜索和论文阅读
    - 生成多个研究计划供选择
    - 支持 reflection 机制
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_reflection_times: int = 3,
        search_time_bias: float = 0.5,
        timeout: int = 120,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_reflection_times = max_reflection_times
        self.search_time_bias = search_time_bias
        
        self.llm_client = LLMClient(
            api_url=api_url,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        # 研究主题上下文
        self.query = ""
        self.problem_name = ""
        self.problem_description = ""
        
        # 搜索结果缓存
        self._search_cache: List[SearchResult] = []
        
    def update_topic(
        self,
        query: str,
        problem_name: str,
        problem_description: str,
        search_time_bias: float = 0.5,
    ):
        """更新研究主题"""
        self.query = query
        self.problem_name = problem_name
        self.problem_description = problem_description
        self.search_time_bias = search_time_bias
        self._search_cache = []
        logger.info(f"Researcher updated topic: {problem_name}")
        
    async def run(
        self,
        parent: Any,
        inspirations: List[Any],
        trace_id: str = "",
        max_reflection_times: Optional[int] = None,
    ) -> tuple:
        """
        运行研究流程
        
        Args:
            parent: 父程序（包含指标和代码）
            inspirations: 灵感列表（历史成功的程序）
            trace_id: 追踪ID
            max_reflection_times: 最大反思次数
            
        Returns:
            (plans, search_results, reports)
        """
        max_reflection = max_reflection_times or self.max_reflection_times
        
        # Step 1: 搜索相关研究
        search_results = await self._search_research()
        
        # Step 2: 生成研究计划
        plans = await self._generate_plans(parent, inspirations, search_results)
        
        # Step 3: 反思和迭代优化
        reports = []
        for reflection_idx in range(max_reflection):
            report = await self._generate_report(
                plans, search_results, parent, inspirations, reflection_idx
            )
            reports.append(report)
            
            # 如果是最后一次迭代，不再反思
            if reflection_idx == max_reflection - 1:
                break
                
            # 基于之前的报告改进计划
            plans = await self._reflect_on_report(
                report, parent, inspirations, search_results
            )
            
        return plans, search_results, reports
        
    async def _search_research(self) -> List[SearchResult]:
        """搜索相关研究 — 使用 SEARCH_INSTRUCTIONS 模板作为辅助"""
        # 构建搜索查询 — SEARCH_INSTRUCTIONS 提供了结构化的搜索方向
        search_queries = [
            f"sequential recommendation {self.problem_name} SASRec latest research 2024 2025",
            f"sequence modeling recommendation system transformer attention mechanism",
            f"self-supervised learning recommendation system recent advances",
        ]
        
        all_results = []
        
        for query in search_queries[:2]:  # 限制搜索次数
            try:
                results = await self._do_search(query)
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Search failed for query '{query}': {e}")
                
        # 去重
        seen = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen:
                seen.add(r.url)
                unique_results.append(r)
                
        self._search_cache = unique_results
        logger.info(f"Found {len(unique_results)} unique search results")
        
        # ── 使用 SEARCH_INSTRUCTIONS 进行结构化搜索分析 (可选) ──
        if unique_results and self.problem_name:
            search_prompt = SEARCH_INSTRUCTIONS.format(
                research_question=self.query or self.problem_name,
                current_model=self.problem_name,
                current_metrics="N/A (搜索阶段暂无)",
            )
            
            # 添加搜索结果上下文
            search_results_str = "\n## 已获取的搜索结果\n"
            for r in unique_results[:5]:
                search_results_str += f"- {r.title}: {r.snippet[:100]}...\n"
            search_prompt += search_results_str
            
            try:
                search_response = await self.llm_client.chat(search_prompt)
                # 解析搜索分析结果以提取更多关键词
                search_analysis = self._parse_researcher_response(search_response)
                if search_analysis and search_analysis.get("synthesis"):
                    # 从合成分析中提取额外关键词进行补充搜索
                    key_insights = search_analysis.get("synthesis", {}).get("key_insights", [])
                    for insight in key_insights[:2]:
                        try:
                            extra_results = await self._do_search(insight)
                            unique_results.extend(extra_results)
                        except Exception as e:
                            logger.warning(f"Extra search failed: {e}")
            except Exception as e:
                logger.warning(f"SEARCH_INSTRUCTIONS analysis failed: {e}")
        
        return unique_results
        
    async def _do_search(self, query: str) -> List[SearchResult]:
        """执行搜索（使用搜索工具）"""
        # 这里使用 search_web 工具
        from tools import search_web
        
        try:
            results = await search_web(query=query, gl="en", hl="en")
            search_results = []
            
            for item in results.get("results", [])[:5]:  # 取前5个
                search_results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("snippet", ""),
                    source="web"
                ))
                
            return search_results
        except Exception as e:
            logger.warning(f"Search error: {e}")
            return []
            
    async def _generate_plans(
        self,
        parent: Any,
        inspirations: List[Any],
        search_results: List[SearchResult],
    ) -> List[ResearchPlan]:
        """生成研究计划 — 使用 RESEARCHER_INSTRUCTIONS 模板"""
        
        # 构建上下文信息
        metrics_str = json.dumps(parent.metrics, ensure_ascii=False) if parent and hasattr(parent, 'metrics') else '{}'
        
        # 构建灵感描述
        inspirations_str = ""
        if inspirations:
            inspirations_str = "## 历史成功案例\n"
            for i, insp in enumerate(inspirations[:3]):
                inspirations_str += (
                    f"- {insp.idea.title if hasattr(insp, 'idea') else 'Unknown'}: "
                    f"指标 {json.dumps(insp.metrics, ensure_ascii=False)}\n"
                )
        
        # 构建搜索结果描述
        search_str = ""
        if search_results:
            search_str = "## 最新研究发现\n"
            for r in search_results[:5]:
                search_str += f"- {r.title}: {r.snippet[:100]}...\n"
        
        # 构建源码上下文 (如果有)
        source_code_str = ""
        if parent and hasattr(parent, 'code'):
            source_code_str = parent.code[:3000]  # 限制长度
        
        # 使用 RESEARCHER_INSTRUCTIONS 模板
        prompt = RESEARCHER_INSTRUCTIONS.format(
            research_direction=self.query or self.problem_name,
            current_metrics=metrics_str,
            experiment_journal=inspirations_str,
            source_code_context=source_code_str,
        )
        
        # 添加搜索结果作为补充
        if search_str:
            prompt += f"\n\n{search_str}"
        
        response = await self.llm_client.chat(prompt)
        
        # 解析响应 — RESEARCHER_INSTRUCTIONS 使用 JSON 输出格式
        try:
            result = self._parse_researcher_response(response)
            if result and result.get("proposed_solutions"):
                plans = []
                for sol in result["proposed_solutions"]:
                    plan = ResearchPlan(
                        title=sol.get("solution_name", "未命名方案"),
                        description=sol.get("theoretical_basis", ""),
                        expected_improvement=sol.get("expected_benefits", ""),
                        implementation_hints=sol.get("implementation_approach", "").split("\n") if sol.get("implementation_approach") else [],
                        confidence=sol.get("potential_risks", "中").replace("高风险", "低").replace("中风险", "中").replace("低风险", "高") if "风险" in sol.get("potential_risks", "") else "中",
                    )
                    plans.append(plan)
                return plans
        except Exception as e:
            logger.warning(f"Failed to parse RESEARCHER_INSTRUCTIONS response: {e}")
        
        # 降级: 使用旧格式的硬编码 prompt
        context_parts = [
            f"## 当前任务",
            f"优化推荐系统: {self.problem_name}",
            f"问题描述: {self.problem_description}",
            "",
        ]
        
        if parent:
            context_parts.extend([
                f"## 父程序表现",
                f"指标: {json.dumps(parent.metrics, ensure_ascii=False)}",
                "",
            ])
            
        if inspirations:
            context_parts.append("## 历史成功案例")
            for i, insp in enumerate(inspirations[:3]):
                context_parts.append(
                    f"- {insp.idea.title if hasattr(insp, 'idea') else 'Unknown'}: "
                    f"指标 {json.dumps(insp.metrics, ensure_ascii=False)}"
                )
            context_parts.append("")
            
        if search_results:
            context_parts.append("## 最新研究发现")
            for r in search_results[:5]:
                context_parts.append(f"- {r.title}: {r.snippet[:100]}...")
            context_parts.append("")
            
        context = "\n".join(context_parts)
        
        fallback_prompt = f"""你是一位推荐系统研究员，正在为算法进化寻找新的研究方向。

{context}

请基于以上信息，提出 2-3 个可能提升效果的研究方向。

### 输出格式
```json
[
  {{
    "title": "研究方向标题",
    "description": "详细描述这个方向为什么可能有效",
    "expected_improvement": "预期的改进效果",
    "implementation_hints": ["实现提示1", "实现提示2"],
    "confidence": "高/中/低"
  }}
]
```"""
        
        response = await self.llm_client.chat(fallback_prompt)
        
        try:
            plans = self._parse_plans(response)
        except Exception as e:
            logger.warning(f"Failed to parse fallback plans: {e}")
            plans = []
            
        return plans
    
    def _parse_researcher_response(self, response: str) -> Optional[dict]:
        """
        解析 RESEARCHER_INSTRUCTIONS 的 JSON 输出格式
        
        输出格式:
        {
          "problem_analysis": {...},
          "proposed_solutions": [...],
          "recommended_solution": {...}
        }
        """
        import re
        
        # 提取 JSON
        json_patterns = [
            r'```json\s*\n(.*?)```',
            r'```(.*?)```',
            r'\{.*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(1) if '```' in pattern else match.group(0))
                except json.JSONDecodeError:
                    continue
        
        return None
        
    def _parse_plans(self, response: str) -> List[ResearchPlan]:
        """解析研究计划"""
        import re
        
        # 尝试提取 JSON
        json_match = re.search(r'\[.*\]', response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            return [ResearchPlan(**item) for item in data]
        return []
        
    async def _generate_report(
        self,
        plans: List[ResearchPlan],
        search_results: List[SearchResult],
        parent: Any,
        inspirations: List[Any],
        reflection_idx: int,
    ) -> ResearchReport:
        """生成研究报告"""
        
        # 选择最佳计划
        best_plan = plans[0] if plans else None
        
        if not best_plan:
            # 如果没有计划，生成默认想法
            idea = IdeaData(
                title="参数调优",
                description="通过调整超参数来改进模型",
                content="调整学习率、hidden_size等参数",
                confidence="中"
            )
            return ResearchReport(
                idea=idea,
                markdown_report="无有效研究计划",
                search_results=search_results,
                plans=plans
            )
            
        # 构建报告
        context_parts = [
            f"## 选定的研究方向",
            f"标题: {best_plan.title}",
            f"描述: {best_plan.description}",
            f"预期改进: {best_plan.expected_improvement}",
            "",
        ]
        
        if best_plan.implementation_hints:
            context_parts.append("## 实现提示")
            for hint in best_plan.implementation_hints:
                context_parts.append(f"- {hint}")
            context_parts.append("")
            
        context_parts.extend([
            f"## 父程序指标",
            f"{json.dumps(parent.metrics, ensure_ascii=False) if parent else 'N/A'}",
            "",
        ])
        
        context = "\n".join(context_parts)
        
        prompt = f"""基于以下研究计划，生成一个具体的算法改进想法。

{context}

请生成一个包含以下字段的想法：
- title: 想法标题
- description: 详细描述
- content: 具体实现思路
- supplement: 补充说明（可选）

### 输出格式
```json
{{
  "title": "...",
  "description": "...",
  "content": "...",
  "supplement": "..."
}}
```"""
        
        response = await self.llm_client.chat(prompt)
        
        # 解析响应
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                idea_data = json.loads(json_match.group())
                idea = IdeaData(**idea_data)
            else:
                idea = IdeaData(
                    title=best_plan.title,
                    description=best_plan.description,
                    content=best_plan.expected_improvement,
                    confidence=best_plan.confidence
                )
        except Exception as e:
            logger.warning(f"Failed to parse idea: {e}")
            idea = IdeaData(
                title=best_plan.title,
                description=best_plan.description,
                content=best_plan.expected_improvement,
                confidence="中"
            )
            
        # 生成 markdown 报告
        markdown = self._generate_markdown_report(idea, best_plan, search_results)
        
        return ResearchReport(
            idea=idea,
            markdown_report=markdown,
            search_results=search_results,
            plans=plans
        )
        
    def _generate_markdown_report(
        self,
        idea: IdeaData,
        plan: ResearchPlan,
        search_results: List[SearchResult],
    ) -> str:
        """生成 Markdown 格式的研究报告"""
        
        report_parts = [
            f"# 研究报告 - {idea.title}",
            "",
            f"## 想法描述",
            idea.description,
            "",
            f"## 实现思路",
            idea.content,
            "",
        ]
        
        if idea.supplement:
            report_parts.extend([
                f"## 补充说明",
                idea.supplement,
                "",
            ])
            
        if search_results:
            report_parts.append("## 相关研究")
            for r in search_results[:3]:
                report_parts.append(f"- [{r.title}]({r.url})")
            report_parts.append("")
            
        if plan.implementation_hints:
            report_parts.append("## 实现提示")
            for hint in plan.implementation_hints:
                report_parts.append(f"- {hint}")
            report_parts.append("")
            
        return "\n".join(report_parts)
        
    async def _reflect_on_report(
        self,
        report: ResearchReport,
        parent: Any,
        inspirations: List[Any],
        search_results: List[SearchResult],
    ) -> List[ResearchPlan]:
        """基于报告进行反思，生成改进的计划 — 使用 REFLECTION_INSTRUCTIONS 模板"""
        
        # 构建反思上下文
        previous_direction = report.idea.title if report.idea else "未知"
        previous_results = json.dumps(parent.metrics, ensure_ascii=False) if parent and hasattr(parent, 'metrics') else '{}'
        current_metrics = previous_results  # 反思阶段指标可能不变
        
        # 使用 REFLECTION_INSTRUCTIONS 模板
        prompt = REFLECTION_INSTRUCTIONS.format(
            iteration_count=self._current_reflection_idx if hasattr(self, '_current_reflection_idx') else 0,
            previous_direction=previous_direction,
            previous_results=previous_results,
            current_metrics=current_metrics,
        )
        
        # 添加之前的报告内容
        prompt += f"\n\n## 之前的研究报告\n{report.markdown_report}"
        
        response = await self.llm_client.chat(prompt)
        
        # 解析反思结果
        try:
            reflection_result = self._parse_researcher_response(response)
            if reflection_result:
                recommendations = reflection_result.get("recommendations", {})
                # 如果反思建议继续当前方向 — 生成改进的计划
                next_steps = recommendations.get("next_steps", [])
                
                if next_steps:
                    plans = []
                    for step in next_steps[:3]:
                        plan = ResearchPlan(
                            title=step[:50] if len(step) > 50 else step,
                            description=reflection_result.get("rationale", ""),
                            expected_improvement=step,
                            implementation_hints=[],
                            confidence="中",
                        )
                        plans.append(plan)
                    return plans if plans else report.plans
                
                # 如果反思建议转向 — 使用 suggested_pivot 作为新方向
                suggested_pivot = recommendations.get("suggested_pivot", "")
                if suggested_pivot:
                    return [ResearchPlan(
                        title=suggested_pivot[:50],
                        description=f"Reflection pivot: {reflection_result.get('rationale', '')}",
                        expected_improvement=suggested_pivot,
                        implementation_hints=[],
                        confidence="中",
                    )]
        except Exception as e:
            logger.warning(f"Failed to parse REFLECTION_INSTRUCTIONS response: {e}")
        
        # 降级: 使用旧格式的硬编码 prompt
        fallback_prompt = f"""你正在对之前生成的研究报告进行反思和改进。

## 之前的研究报告
{report.markdown_report}

## 父程序指标
{json.dumps(parent.metrics, ensure_ascii=False) if parent else 'N/A'}

请分析之前报告的不足之处，并提出改进的研究计划。

### 输出格式
```json
[
  {{
    "title": "改进后的研究方向",
    "description": "改进描述",
    "expected_improvement": "预期改进",
    "implementation_hints": ["提示1", "提示2"],
    "confidence": "高/中/低"
  }}
]
```"""
        
        response = await self.llm_client.chat(fallback_prompt)
        
        try:
            plans = self._parse_plans(response)
        except Exception as e:
            logger.warning(f"Failed to parse reflected plans: {e}")
            plans = report.plans
            
        return plans
        
    async def read_paper(
        self,
        title: str,
        content: str,
        supplement: str = "",
    ) -> IdeaData:
        """阅读论文并提取想法"""
        
        prompt = f"""请阅读以下论文/想法信息，并提取关键内容。

标题: {title}
内容: {content}
补充: {supplement}

请生成一个结构化的想法数据：

### 输出格式
```json
{{
  "title": "提取的标题",
  "description": "核心描述",
  "content": "关键技术点",
  "supplement": "补充说明"
}}
```"""
        
        response = await self.llm_client.chat(prompt)
        
        try:
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                idea_data = json.loads(json_match.group())
                return IdeaData(**idea_data)
        except Exception as e:
            logger.warning(f"Failed to parse paper: {e}")
            
        return IdeaData(
            title=title,
            description=content,
            content=supplement or content
        )


# 导出
__all__ = [
    'ResearcherAgent',
    'SearchResult',
    'ResearchPlan', 
    'ResearchReport',
    'IdeaData',
]
