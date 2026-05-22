"""
Program Database 模块 - 完整的版本管理系统

功能：
1. 存储所有程序版本
2. Island-based Evolution（多岛并行进化）
3. 程序采样和选择
4. 迁移机制
5. Checkpoint 保存
"""

import json
import logging
import os
import time
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("rec_self_evolve.database")


@dataclass
class Program:
    """程序版本"""
    id: str
    code: str
    idea: Any  # IdeaData
    parent_id: str
    language: str = "python"
    metrics: Dict[str, float] = field(default_factory=dict)
    iteration_found: int = 0
    evolution_history: List[Any] = field(default_factory=list)
    report: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


class ProgramDatabase:
    """
    程序数据库 - 支持 Island-based Evolution
    
    核心特性：
    - 多 island（种群）并行进化
    - 定期迁移促进多样性
    - 基于指标的父程序选择
    - Checkpoint 和恢复
    """
    
    def __init__(
        self,
        population_size: int = 20,
        num_islands: int = 4,
        migration_interval: int = 5,
        migration_rate: float = 0.1,
    ):
        self.population_size = population_size
        self.num_islands = num_islands
        self.migration_interval = migration_interval
        self.migration_rate = migration_rate
        
        # 所有程序
        self.programs: Dict[str, Program] = {}
        
        # Island 管理
        self.islands: Dict[int, List[str]] = {
            i: [] for i in range(num_islands)
        }
        self.current_island = 0
        self.island_generation: Dict[int, int] = {
            i: 0 for i in range(num_islands)
        }
        
        # 最佳程序追踪
        self.best_program_id: Optional[str] = None
        self.best_metric = "ndcg"  # 默认追踪 NDCG
        
        # 迭代计数
        self.last_iteration = 0
        
        logger.info(
            f"Initialized ProgramDatabase with {num_islands} islands, "
            f"population size {population_size}"
        )
        
    def add(self, program: Program, iteration: int = 0) -> None:
        """添加程序到数据库"""
        self.programs[program.id] = program
        self.last_iteration = max(self.last_iteration, iteration)
        
        # 分配到当前 island
        island_idx = self.current_island
        self.islands[island_idx].append(program.id)
        
        # 更新最佳程序
        self._update_best(program)
        
        # 如果 island 超过容量，移除最差的
        self._maintain_island_size(island_idx)
        
        logger.debug(
            f"Added program {program.id} to island {island_idx}, "
            f"total programs: {len(self.programs)}"
        )
        
    def _update_best(self, program: Program) -> None:
        """更新最佳程序"""
        if self.best_program_id is None:
            self.best_program_id = program.id
            return
            
        best_program = self.programs.get(self.best_program_id)
        if best_program is None:
            self.best_program_id = program.id
            return
            
        # 比较 metrics
        current_score = program.metrics.get(self.best_metric, 0)
        best_score = best_program.metrics.get(self.best_metric, 0)
        
        if current_score > best_score:
            self.best_program_id = program.id
            logger.info(f"New best program: {program.id} with {self.best_metric}={current_score}")
            
    def _maintain_island_size(self, island_idx: int) -> None:
        """维护 island 大小"""
        island = self.islands[island_idx]
        
        if len(island) > self.population_size // self.num_islands:
            # 移除最差的程序
            island_programs = [
                (pid, self.programs[pid].metrics.get(self.best_metric, 0))
                for pid in island
            ]
            island_programs.sort(key=lambda x: x[1])  # 按分数升序
            
            # 移除最差的
            num_remove = len(island) - (self.population_size // self.num_islands)
            for i in range(num_remove):
                worst_id = island_programs[i][0]
                island.remove(worst_id)
                # 不从 self.programs 中删除，保留历史
                
    def sample(self) -> Tuple[Program, List[Program]]:
        """
        采样父程序和灵感
        
        Returns:
            (parent, inspirations)
        """
        island = self.islands[self.current_island]
        
        if not island:
            # 如果当前 island 为空，返回最佳程序
            if self.best_program_id:
                parent = self.programs[self.best_program_id]
                return parent, []
            return None, []
            
        # 基于适应度选择父程序
        parent = self._select_parent(island)
        
        # 选择灵感（历史中表现好的程序）
        inspirations = self._select_inspirations(parent, k=3)
        
        return parent, inspirations
        
    def _select_parent(self, island: List[str]) -> Program:
        """基于适应度选择父程序"""
        candidates = []
        
        for pid in island:
            program = self.programs.get(pid)
            if program:
                score = program.metrics.get(self.best_metric, 0)
                candidates.append((pid, score))
                
        if not candidates:
            return None
            
        # 排序
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        # 锦标赛选择：取前几名中随机选
        top_k = min(3, len(candidates))
        selected = candidates[0][0]  # 默认选最好的
        
        # 也可以随机选一个增加多样性
        import random
        if random.random() < 0.3 and top_k > 1:
            selected = random.choice(candidates[:top_k])[0]
            
        return self.programs[selected]
        
    def _select_inspirations(self, parent: Program, k: int = 3) -> List[Program]:
        """选择灵感程序"""
        if not parent:
            return []
            
        inspirations = []
        
        # 从所有程序中选择表现最好的作为灵感
        all_programs = sorted(
            self.programs.values(),
            key=lambda p: p.metrics.get(self.best_metric, 0),
            reverse=True
        )
        
        for p in all_programs:
            if p.id != parent.id and len(inspirations) < k:
                inspirations.append(p)
                
        return inspirations
        
    def next_island(self) -> None:
        """切换到下一个 island"""
        self.current_island = (self.current_island + 1) % self.num_islands
        logger.debug(f"Switched to island {self.current_island}")
        
    def increment_island_generation(self) -> None:
        """增加 island 代数"""
        self.island_generation[self.current_island] += 1
        
    def should_migrate(self) -> bool:
        """检查是否应该迁移"""
        gen = self.island_generation[self.current_island]
        return gen > 0 and gen % self.migration_interval == 0
        
    def migrate_programs(self) -> None:
        """迁移程序到其他 island"""
        if self.num_islands <= 1:
            return
            
        # 计算迁移数量
        num_migrate = max(1, int(self.population_size * self.migration_rate))
        
        # 从当前 island 选择最好的程序迁移到其他 island
        island = self.islands[self.current_island]
        
        if len(island) <= 1:
            return
            
        # 选择要迁移的程序
        island_programs = [
            (pid, self.programs[pid].metrics.get(self.best_metric, 0))
            for pid in island
        ]
        island_programs.sort(key=lambda x: x[1], reverse=True)
        
        migrate_count = min(num_migrate, len(island_programs) - 1)
        
        for i in range(migrate_count):
            migrate_id = island_programs[i][0]
            
            # 选择目标 island（随机）
            import random
            target_island = random.choice(
                [i for i in range(self.num_islands) if i != self.current_island]
            )
            
            # 迁移
            island.remove(migrate_id)
            self.islands[target_island].append(migrate_id)
            
            logger.debug(f"Migrated program {migrate_id} from island {self.current_island} to island {target_island}")
            
    def get(self, program_id: str) -> Optional[Program]:
        """获取程序"""
        return self.programs.get(program_id)
        
    def get_best_program(self, metric: Optional[str] = None) -> Optional[Program]:
        """获取最佳程序"""
        metric = metric or self.best_metric
        
        if not self.programs:
            return None
            
        best = max(
            self.programs.values(),
            key=lambda p: p.metrics.get(metric, 0)
        )
        
        return best
        
    def log_island_status(self) -> None:
        """记录 island 状态"""
        logger.info("=== Island Status ===")
        for island_idx in range(self.num_islands):
            island = self.islands[island_idx]
            gen = self.island_generation[island_idx]
            
            # 计算 island 内最佳分数
            best_score = 0
            if island:
                scores = [
                    self.programs[pid].metrics.get(self.best_metric, 0)
                    for pid in island if pid in self.programs
                ]
                if scores:
                    best_score = max(scores)
                    
            logger.info(
                f"Island {island_idx}: {len(island)} programs, "
                f"generation {gen}, best {self.best_metric}={best_score:.4f}"
            )
            
    def save(self, checkpoint_dir: str, iteration: int) -> None:
        """保存 checkpoint"""
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # 保存程序
        programs_file = os.path.join(checkpoint_dir, "programs.json")
        programs_data = []
        
        for program in self.programs.values():
            # 序列化 idea
            idea_dict = {}
            if hasattr(program.idea, 'model_dump'):
                idea_dict = program.idea.model_dump()
            elif hasattr(program.idea, '__dict__'):
                idea_dict = program.idea.__dict__
                
            programs_data.append({
                'id': program.id,
                'code': program.code,
                'idea': idea_dict,
                'parent_id': program.parent_id,
                'language': program.language,
                'metrics': program.metrics,
                'iteration_found': program.iteration_found,
                'report': program.report,
                'metadata': program.metadata,
                'timestamp': program.timestamp,
            })
            
        with open(programs_file, 'w', encoding='utf-8') as f:
            json.dump(programs_data, f, indent=2, ensure_ascii=False)
            
        # 保存 island 状态
        island_file = os.path.join(checkpoint_dir, "islands.json")
        island_data = {
            'islands': self.islands,
            'current_island': self.current_island,
            'island_generation': self.island_generation,
            'best_program_id': self.best_program_id,
            'last_iteration': self.last_iteration,
        }
        
        with open(island_file, 'w', encoding='utf-8') as f:
            json.dump(island_data, f, indent=2)
            
        logger.info(f"Saved checkpoint to {checkpoint_dir}")
        
    def load(self, checkpoint_dir: str) -> int:
        """加载 checkpoint"""
        island_file = os.path.join(checkpoint_dir, "islands.json")
        
        if not os.path.exists(island_file):
            logger.warning(f"No checkpoint found at {checkpoint_dir}")
            return 0
            
        with open(island_file, 'r', encoding='utf-8') as f:
            island_data = json.load(f)
            
        self.islands = island_data.get('islands', {i: [] for i in range(self.num_islands)})
        self.current_island = island_data.get('current_island', 0)
        self.island_generation = island_data.get('island_generation', {i: 0 for i in range(self.num_islands)})
        self.best_program_id = island_data.get('best_program_id')
        self.last_iteration = island_data.get('last_iteration', 0)
        
        # 加载程序
        programs_file = os.path.join(checkpoint_dir, "programs.json")
        if os.path.exists(programs_file):
            with open(programs_file, 'r', encoding='utf-8') as f:
                programs_data = json.load(f)
                
            from agent.researcher import IdeaData
            
            for data in programs_data:
                idea = None
                if 'idea' in data:
                    try:
                        idea = IdeaData(**data['idea'])
                    except:
                        idea = data.get('idea', {})
                        
                program = Program(
                    id=data['id'],
                    code=data['code'],
                    idea=idea,
                    parent_id=data['parent_id'],
                    language=data.get('language', 'python'),
                    metrics=data.get('metrics', {}),
                    iteration_found=data.get('iteration_found', 0),
                    report=data.get('report', ''),
                    metadata=data.get('metadata', {}),
                    timestamp=data.get('timestamp', time.time()),
                )
                self.programs[program.id] = program
                
        logger.info(
            f"Loaded checkpoint from {checkpoint_dir}, "
            f"iteration {self.last_iteration}, {len(self.programs)} programs"
        )
        
        return self.last_iteration


# 导出
__all__ = [
    'Program',
    'ProgramDatabase',
]
