"""
Evolution Engine - 整合所有改进的主循环

特点：
1. 分离 Researcher 和 Coder Agent
2. Island-based Evolution
3. Deep Research 支持
4. Reflection 机制
5. 完整的版本管理
"""

import asyncio
import logging
import os
import time
import uuid
import json
from pathlib import Path
from typing import Optional, Dict, Any

from agent.researcher import ResearcherAgent, IdeaData
from agent.coder import CoderAgent
from agent.database import Program, ProgramDatabase
from agent.llm_client import LLMClient

logger = logging.getLogger("rec_self_evolve.evolve_engine")


class EvolutionEngine:
    """
    进化引擎 - 整合所有改进的主循环
    
    流程：
    1. 采样父程序和灵感
    2. Deep Research（搜索网络 + 生成研究计划）
    3. Coder（代码生成 + 自动调试）
    4. 评估并添加到数据库
    5. 定期迁移
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        project_root: str,
        max_iterations: int = 10,
        # 研究配置
        max_research_reflect: int = 3,
        search_time_bias: float = 0.5,
        # 编码配置
        max_coding_reflect: int = 3,
        # Island 配置
        population_size: int = 20,
        num_islands: int = 4,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
        # Checkpoint 配置
        checkpoint_interval: int = 5,
    ):
        self.config = config
        self.project_root = project_root
        self.max_iterations = max_iterations
        
        # 初始化 Agents
        self.researcher = ResearcherAgent(
            api_url=config.get("api_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key", "EMPTY"),
            model=config.get("model", "gpt-4o"),
            temperature=0.7,
            max_reflection_times=max_research_reflect,
            search_time_bias=search_time_bias,
        )
        
        self.coder = CoderAgent(
            api_url=config.get("api_url", "http://localhost:8000/v1"),
            api_key=config.get("api_key", "EMPTY"),
            model=config.get("model", "gpt-4o"),
            temperature=0.4,
            max_reflection_times=max_coding_reflect,
        )
        
        # 初始化数据库
        self.database = ProgramDatabase(
            population_size=population_size,
            num_islands=num_islands,
            migration_interval=migration_interval,
            migration_rate=migration_rate,
        )
        
        self.checkpoint_interval = checkpoint_interval
        
        # 项目信息
        self.problem_name = config.get("problem_name", "recommendation")
        self.problem_description = config.get(
            "problem_description", 
            "序列推荐系统优化"
        )
        
        # 训练相关
        self.train_command = config.get("train_command", "")
        self.eval_command = config.get("eval_command", "")
        
    def update_topic(self, query: str):
        """更新研究主题"""
        self.researcher.update_topic(
            query=query,
            problem_name=self.problem_name,
            problem_description=self.problem_description,
        )
        
        self.coder.update_topic(
            query=query,
            problem_name=self.problem_name,
            problem_description=self.problem_description,
        )
        
    async def run(
        self,
        initial_code: str,
        initial_idea: Optional[IdeaData] = None,
        initial_metrics: Optional[Dict[str, float]] = None,
    ) -> Program:
        """
        运行进化流程
        
        Args:
            initial_code: 初始代码
            initial_idea: 初始想法
            initial_metrics: 初始指标
            
        Returns:
            最佳程序
        """
        # 添加初始程序
        await self._add_initial_program(
            initial_code, initial_idea, initial_metrics
        )
        
        # 主循环
        for i in range(self.max_iterations):
            logger.info(f"\n{'='*50}")
            logger.info(f"迭代 {i+1} / {self.max_iterations}")
            logger.info(f"{'='*50}")
            
            # Step 1: 采样
            parent, inspirations = self.database.sample()
            
            if parent is None:
                logger.warning("No parent found, using best program")
                parent = self.database.get_best_program()
                
            # Step 2: Deep Research
            research_plans, search_results, research_reports = (
                await self.researcher.run(
                    parent=parent,
                    inspirations=inspirations,
                    max_reflection_times=None,  # 使用默认配置
                )
            )
            
            new_idea = research_reports[-1].idea if research_reports else None
            
            logger.info(f"研究想法: {new_idea.title if new_idea else 'N/A'}")
            
            # Step 3: Coder
            all_diff_text, all_program_code = await self.coder.run(
                new_idea=new_idea,
                parent=parent,
                inspirations=inspirations,
                max_reflection_times=None,
            )
            
            child_code = all_program_code[-1]
            
            # Step 4: 评估
            child_metrics = await self._evaluate(child_code)
            
            if child_metrics is None:
                logger.warning(f"评估失败，使用父程序指标")
                child_metrics = parent.metrics.copy() if parent else {}
                
            # Step 5: 添加到数据库
            child_program = Program(
                id=str(uuid.uuid4()),
                code=child_code,
                idea=new_idea or IdeaData(
                    title="参数调整",
                    description="调整参数",
                    content="参数调整"
                ),
                parent_id=parent.id if parent else "root",
                language="python",
                metrics=child_metrics,
                iteration_found=i + 1,
                evolution_history=(
                    parent.evolution_history + [new_idea]
                    if parent and new_idea
                    else []
                ),
                report=(
                    research_reports[-1].markdown_report
                    if research_reports else ""
                ),
                metadata={"parent_metrics": parent.metrics} if parent else {},
            )
            
            self.database.add(child_program, iteration=i + 1)
            
            # 增加 island 代数
            self.database.increment_island_generation()
            
            # 检查迁移
            if self.database.should_migrate():
                logger.info(f"执行迁移...")
                self.database.migrate_programs()
                
            # 记录进度
            improvement = ""
            if parent:
                old_score = parent.metrics.get("ndcg", 0)
                new_score = child_metrics.get("ndcg", 0)
                improvement = f"Δ NDCG: {new_score - old_score:+.4f}"
                
            logger.info(
                f"迭代 {i+1} 完成: {child_program.id} "
                f"指标: {json.dumps(child_metrics, ensure_ascii=False)} "
                f"{improvement}"
            )
            
            # 检查点
            if (i + 1) % self.checkpoint_interval == 0:
                self._save_checkpoint(i + 1)
                
        # 获取最佳程序
        best = self.database.get_best_program()
        
        if best:
            logger.info(
                f"进化完成。最佳程序: {best.id} "
                f"指标: {json.dumps(best.metrics, ensure_ascii=False)}"
            )
            self._save_best_program(best)
        else:
            logger.warning("未找到有效程序")
            
        return best
        
    async def _add_initial_program(
        self,
        initial_code: str,
        initial_idea: Optional[IdeaData],
        initial_metrics: Optional[Dict[str, float]],
    ) -> None:
        """添加初始程序"""
        logger.info("添加初始程序到数据库...")
        
        # 评估初始代码
        if initial_metrics is None:
            initial_metrics = await self._evaluate(initial_code)
            
        # 如果没有想法，从代码生成一个
        if initial_idea is None:
            initial_idea = IdeaData(
                title="初始 SASRec",
                description="基于 SASRec 的序列推荐模型",
                content="标准 SASRec 实现"
            )
            
        initial_program = Program(
            id='root',
            code=initial_code,
            idea=initial_idea,
            parent_id='root',
            language='python',
            metrics=initial_metrics or {},
            iteration_found=0,
            evolution_history=[],
            report="初始程序",
        )
        
        self.database.add(initial_program, iteration=0)
        
    async def _evaluate(self, code: str) -> Optional[Dict[str, float]]:
        """
        评估代码
        
        注意：这个方法需要根据实际项目实现
        """
        # 这里应该调用实际的训练和评估
        # 暂时返回 None 让调用者处理
        logger.info("评估代码...")
        
        # TODO: 实现实际的评估逻辑
        # 可以调用 train_runner 来执行训练
        # 然后解析评估结果
        
        return None
        
    def _save_checkpoint(self, iteration: int) -> None:
        """保存检查点"""
        checkpoint_dir = os.path.join(
            self.project_root,
            "checkpoints",
            f"checkpoint_{iteration}"
        )
        
        self.database.save(checkpoint_dir, iteration)
        
    def _save_best_program(self, program: Program) -> None:
        """保存最佳程序"""
        best_dir = os.path.join(
            self.project_root,
            "checkpoints",
            "best"
        )
        
        os.makedirs(best_dir, exist_ok=True)
        
        # 保存代码
        code_path = os.path.join(best_dir, "best_program.py")
        with open(code_path, 'w', encoding='utf-8') as f:
            f.write(program.code)
            
        # 保存信息
        info = {
            'id': program.id,
            'parent_id': program.parent_id,
            'iteration_found': program.iteration_found,
            'metrics': program.metrics,
            'timestamp': program.timestamp,
        }
        
        info_path = os.path.join(best_dir, "best_program_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
            
        logger.info(f"最佳程序已保存到 {best_dir}")


# 导出
__all__ = [
    'EvolutionEngine',
]
