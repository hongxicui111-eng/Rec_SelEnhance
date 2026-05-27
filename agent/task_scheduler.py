# -*- coding: utf-8 -*-
"""
任务调度器模块 - 管理复杂多步骤假设验证任务的执行

功能：
1. 多步骤任务编排与执行
2. 任务状态与进度持久化（记忆机制）
3. 步骤执行结果验证与自动重试
4. 支持任务恢复与断点续做
"""

import os
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import copy

logger = logging.getLogger("rec_self_evolve.task_scheduler")


class TaskStatus(Enum):
    """任务执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    CANCELLED = "cancelled"


class StepStatus(Enum):
    """步骤执行状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    VALIDATING = "validating"  # 正在验证结果


@dataclass
class TaskStep:
    """任务步骤定义"""
    step_id: str
    name: str
    description: str
    execute_fn: Optional[Callable] = None  # 执行函数
    validate_fn: Optional[Callable] = None  # 验证函数
    required_data: List[str] = field(default_factory=list)  # 所需输入数据
    output_data: List[str] = field(default_factory=list)  # 产出输出数据
    max_retries: int = 3  # 最大重试次数
    retry_delay: float = 1.0  # 重试延迟（秒）
    timeout: float = 600.0  # 超时时间（秒）
    dependencies: List[str] = field(default_factory=list)  # 依赖步骤ID
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据


@dataclass
class StepResult:
    """步骤执行结果"""
    step_id: str
    status: StepStatus
    output_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    retry_count: int = 0
    validation_result: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskState:
    """任务状态"""
    task_id: str
    hypothesis_id: str
    status: TaskStatus
    current_step: Optional[str] = None
    completed_steps: List[str] = field(default_factory=list)
    failed_steps: List[str] = field(default_factory=list)
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryPersistence:
    """任务记忆持久化 - 保存和恢复任务执行状态"""

    def __init__(self, storage_dir: str):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._task_index_file = self.storage_dir / "task_index.json"
        self._load_index()

    def _load_index(self):
        """加载任务索引"""
        if self._task_index_file.exists():
            with open(self._task_index_file, 'r') as f:
                self.task_index = json.load(f)
        else:
            self.task_index = {}

    def _save_index(self):
        """保存任务索引"""
        with open(self._task_index_file, 'w') as f:
            json.dump(self.task_index, f, indent=2)

    def save_task_state(self, state: TaskState):
        """保存任务状态"""
        task_file = self.storage_dir / f"{state.task_id}.json"
        state_dict = {
            "task_id": state.task_id,
            "hypothesis_id": state.hypothesis_id,
            "status": state.status.value,
            "current_step": state.current_step,
            "completed_steps": state.completed_steps,
            "failed_steps": state.failed_steps,
            "step_results": {
                k: {
                    "step_id": v.step_id,
                    "status": v.status.value,
                    "output_data": v.output_data,
                    "error": v.error,
                    "execution_time": v.execution_time,
                    "retry_count": v.retry_count,
                    "validation_result": v.validation_result,
                    "metadata": v.metadata
                }
                for k, v in state.step_results.items()
            },
            "created_at": state.created_at,
            "updated_at": state.updated_at,
            "started_at": state.started_at,
            "completed_at": state.completed_at,
            "error_message": state.error_message,
            "metadata": state.metadata
        }
        with open(task_file, 'w') as f:
            json.dump(state_dict, f, indent=2)

        # 更新索引
        self.task_index[state.task_id] = {
            "hypothesis_id": state.hypothesis_id,
            "status": state.status.value,
            "updated_at": state.updated_at
        }
        self._save_index()

    def load_task_state(self, task_id: str) -> Optional[TaskState]:
        """加载任务状态"""
        task_file = self.storage_dir / f"{task_id}.json"
        if not task_file.exists():
            return None

        with open(task_file, 'r') as f:
            state_dict = json.load(f)

        # 重建 step_results
        step_results = {}
        for k, v in state_dict.get("step_results", {}).items():
            step_results[k] = StepResult(
                step_id=v["step_id"],
                status=StepStatus(v["status"]),
                output_data=v.get("output_data", {}),
                error=v.get("error"),
                execution_time=v.get("execution_time", 0.0),
                retry_count=v.get("retry_count", 0),
                validation_result=v.get("validation_result"),
                metadata=v.get("metadata", {})
            )

        return TaskState(
            task_id=state_dict["task_id"],
            hypothesis_id=state_dict["hypothesis_id"],
            status=TaskStatus(state_dict["status"]),
            current_step=state_dict.get("current_step"),
            completed_steps=state_dict.get("completed_steps", []),
            failed_steps=state_dict.get("failed_steps", []),
            step_results=step_results,
            created_at=state_dict.get("created_at", time.time()),
            updated_at=state_dict.get("updated_at", time.time()),
            started_at=state_dict.get("started_at"),
            completed_at=state_dict.get("completed_at"),
            error_message=state_dict.get("error_message"),
            metadata=state_dict.get("metadata", {})
        )

    def list_tasks(self, hypothesis_id: Optional[str] = None, status: Optional[TaskStatus] = None) -> List[TaskState]:
        """列出任务"""
        tasks = []
        for task_id in self.task_index:
            state = self.load_task_state(task_id)
            if state:
                if hypothesis_id and state.hypothesis_id != hypothesis_id:
                    continue
                if status and state.status != status:
                    continue
                tasks.append(state)
        return tasks

    def delete_task_state(self, task_id: str):
        """删除任务状态"""
        task_file = self.storage_dir / f"{task_id}.json"
        if task_file.exists():
            task_file.unlink()
        if task_id in self.task_index:
            del self.task_index[task_id]
            self._save_index()


class TaskScheduler:
    """
    任务调度器 - 管理复杂多步骤假设验证任务的执行

    核心功能：
    1. 多步骤任务编排：支持定义依赖关系的步骤链
    2. 步骤执行：按依赖顺序执行步骤，支持同步/异步执行
    3. 步骤验证：每个步骤执行后可选择验证结果
    4. 自动重试：验证失败自动重试，支持最大重试次数
    5. 记忆持久化：任务状态自动保存，支持断点续做
    6. 错误恢复：分析错误原因，决定是重试还是跳过
    """

    def __init__(
        self,
        project_root: str,
        storage_dir: Optional[str] = None,
        llm_client=None,
        max_concurrent_steps: int = 1
    ):
        self.project_root = Path(project_root)
        self.storage_dir = Path(storage_dir) if storage_dir else self.project_root / ".task_scheduler"
        self.llm_client = llm_client
        self.max_concurrent_steps = max_concurrent_steps

        # 初始化持久化
        self.memory = MemoryPersistence(str(self.storage_dir))

        # 任务定义和状态
        self.current_task: Optional[TaskState] = None
        self.task_steps: Dict[str, TaskStep] = {}

        # 回调函数
        self.step_callbacks: Dict[str, List[Callable]] = {
            "before_execute": [],
            "after_execute": [],
            "before_validate": [],
            "after_validate": [],
            "on_retry": [],
            "on_error": []
        }

    def register_step_callback(self, event: str, callback: Callable):
        """注册步骤回调"""
        if event in self.step_callbacks:
            self.step_callbacks[event].append(callback)

    def define_task(
        self,
        task_id: str,
        hypothesis_id: str,
        steps: List[TaskStep],
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskState:
        """定义任务"""
        # 验证步骤依赖
        step_ids = {s.step_id for s in steps}
        for step in steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    raise ValueError(f"Step {step.step_id} depends on unknown step {dep}")

        # 保存步骤定义
        self.task_steps = {s.step_id: s for s in steps}

        # 创建任务状态
        state = TaskState(
            task_id=task_id,
            hypothesis_id=hypothesis_id,
            status=TaskStatus.PENDING,
            metadata=metadata or {}
        )

        # 保存初始状态
        self.memory.save_task_state(state)
        self.current_task = state

        logger.info(f"Defined task {task_id} with {len(steps)} steps")
        return state

    def execute_task(
        self,
        task_id: str,
        resume: bool = True,
        start_from_step: Optional[str] = None
    ) -> TaskState:
        """
        执行任务

        Args:
            task_id: 任务ID
            resume: 是否从上次中断的地方继续
            start_from_step: 从指定步骤开始（忽略之前完成的部分）

        Returns:
            最终任务状态
        """
        # 加载任务状态
        state = self.memory.load_task_state(task_id)
        if not state:
            raise ValueError(f"Task {task_id} not found")

        self.current_task = state
        self.task_steps = {s.step_id: s for s in self.task_steps.values()} if self.task_steps else {}

        # 处理恢复或重新开始
        if resume and start_from_step is None:
            if state.status == TaskStatus.RUNNING:
                logger.info(f"Resuming task {task_id} from step {state.current_step}")
            elif state.status == TaskStatus.FAILED:
                logger.info(f"Retrying failed task {task_id}")
        elif start_from_step:
            # 重置指定步骤之后的状态
            state = self._reset_from_step(state, start_from_step)

        # 开始执行
        state.status = TaskStatus.RUNNING
        state.started_at = state.started_at or time.time()
        state.updated_at = time.time()
        self.memory.save_task_state(state)

        try:
            # 执行任务步骤
            self._execute_steps(state, start_from_step)

            # 判断任务结果
            if state.failed_steps:
                state.status = TaskStatus.FAILED
                state.error_message = f"Failed steps: {', '.join(state.failed_steps)}"
            else:
                state.status = TaskStatus.COMPLETED

        except Exception as e:
            state.status = TaskStatus.FAILED
            state.error_message = str(e)
            logger.exception(f"Task {task_id} failed with error")

        state.completed_at = time.time()
        state.updated_at = time.time()
        self.memory.save_task_state(state)

        return state

    def _reset_from_step(self, state: TaskState, start_from_step: str) -> TaskState:
        """重置从指定步骤开始的状态"""
        # 找到要重置的步骤索引
        step_order = self._get_step_execution_order()
        try:
            start_idx = step_order.index(start_from_step)
        except ValueError:
            raise ValueError(f"Step {start_from_step} not found")

        # 重置该步骤之后的所有步骤
        steps_to_reset = set(step_order[start_idx:])
        state.completed_steps = [s for s in state.completed_steps if s not in steps_to_reset]
        state.failed_steps = [s for s in state.failed_steps if s not in steps_to_reset]

        # 删除相关步骤的结果
        for step_id in steps_to_reset:
            if step_id in state.step_results:
                del state.step_results[step_id]

        state.current_step = start_from_step
        state.status = TaskStatus.PENDING
        state.error_message = None

        self.memory.save_task_state(state)
        return state

    def _get_step_execution_order(self) -> List[str]:
        """获取步骤执行顺序（拓扑排序）"""
        # 简单的拓扑排序
        in_degree = {sid: 0 for sid in self.task_steps}
        for sid, step in self.task_steps.items():
            for dep in step.dependencies:
                in_degree[sid] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        result = []

        while queue:
            sid = queue.pop(0)
            result.append(sid)

            # 更新依赖该步骤的步骤的入度
            for other_sid, step in self.task_steps.items():
                if sid in step.dependencies:
                    in_degree[other_sid] -= 1
                    if in_degree[other_sid] == 0:
                        queue.append(other_sid)

        return result

    def _execute_steps(self, state: TaskState, start_from_step: Optional[str] = None):
        """执行所有任务步骤"""
        step_order = self._get_step_execution_order()

        # 确定起始位置
        start_idx = 0
        if start_from_step:
            try:
                start_idx = step_order.index(start_from_step)
            except ValueError:
                pass

        for idx in range(start_idx, len(step_order)):
            step_id = step_order[idx]
            step = self.task_steps[step_id]

            # 检查依赖是否满足
            if not all(dep in state.completed_steps for dep in step.dependencies):
                logger.warning(f"Skipping step {step_id} due to unmet dependencies")
                state.step_results[step_id] = StepResult(
                    step_id=step_id,
                    status=StepStatus.SKIPPED,
                    error="Unmet dependencies"
                )
                continue

            # 执行步骤
            state.current_step = step_id
            state.updated_at = time.time()
            self.memory.save_task_state(state)

            result = self._execute_single_step(state, step)
            state.step_results[step_id] = result

            if result.status == StepStatus.COMPLETED:
                state.completed_steps.append(step_id)
            elif result.status == StepStatus.FAILED:
                state.failed_steps.append(step_id)
                # 决定是否继续
                if not self._should_continue_on_failure(step, result):
                    logger.info(f"Stopping execution due to step {step_id} failure")
                    break

            state.updated_at = time.time()
            self.memory.save_task_state(state)

    def _execute_single_step(self, state: TaskState, step: TaskStep) -> StepResult:
        """执行单个步骤"""
        result = StepResult(step_id=step.step_id, status=StepStatus.RUNNING)

        # 触发前置回调
        for callback in self.step_callbacks["before_execute"]:
            try:
                callback(state, step)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

        start_time = time.time()

        # 执行步骤（带重试）
        for retry in range(step.max_retries + 1):
            result.retry_count = retry

            try:
                # 执行
                if step.execute_fn:
                    output_data = step.execute_fn(
                        task_state=state,
                        step=step,
                        previous_results={k: v.output_data for k, v in state.step_results.items()}
                    )
                    result.output_data = output_data or {}

                result.status = StepStatus.COMPLETED
                result.execution_time = time.time() - start_time

                # 触发后置回调
                for callback in self.step_callbacks["after_execute"]:
                    try:
                        callback(state, step, result)
                    except Exception as e:
                        logger.warning(f"Callback error: {e}")

                # 验证结果
                if step.validate_fn:
                    validation_result = self._validate_step_result(state, step, result)
                    result.validation_result = validation_result

                    if not validation_result.get("valid", False):
                        result.status = StepStatus.FAILED
                        result.error = validation_result.get("reason", "Validation failed")

                        # 重试
                        if retry < step.max_retries:
                            logger.info(f"Step {step.step_id} validation failed, retry {retry + 1}/{step.max_retries}")
                            for callback in self.step_callbacks["on_retry"]:
                                try:
                                    callback(state, step, result, validation_result)
                                except Exception as e:
                                    logger.warning(f"Callback error: {e}")

                            time.sleep(step.retry_delay)
                            continue
                        else:
                            break
                    else:
                        # 验证通过
                        break

                break  # 执行成功或不需要验证

            except Exception as e:
                result.error = str(e)
                result.execution_time = time.time() - start_time

                if retry < step.max_retries:
                    logger.warning(f"Step {step.step_id} error: {e}, retry {retry + 1}/{step.max_retries}")
                    time.sleep(step.retry_delay)
                else:
                    result.status = StepStatus.FAILED

        if result.status == StepStatus.FAILED and not result.error:
            result.error = f"Failed after {step.max_retries} retries"

        return result

    def _validate_step_result(
        self,
        state: TaskState,
        step: TaskStep,
        result: StepResult
    ) -> Dict[str, Any]:
        """验证步骤结果"""
        # 触发验证前回调
        for callback in self.step_callbacks["before_validate"]:
            try:
                callback(state, step, result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

        try:
            validation_result = step.validate_fn(
                task_state=state,
                step=step,
                result=result,
                llm_client=self.llm_client
            )
        except Exception as e:
            validation_result = {"valid": False, "reason": str(e)}

        # 触发验证后回调
        for callback in self.step_callbacks["after_validate"]:
            try:
                callback(state, step, result, validation_result)
            except Exception as e:
                logger.warning(f"Callback error: {e}")

        return validation_result

    def _should_continue_on_failure(self, step: TaskStep, result: StepResult) -> bool:
        """判断是否继续执行后续步骤"""
        # 可以根据步骤的元数据或错误类型决定
        continue_on_failure = step.metadata.get("continue_on_failure", True)
        return continue_on_failure

    def get_task_state(self, task_id: str) -> Optional[TaskState]:
        """获取任务状态"""
        return self.memory.load_task_state(task_id)

    def list_tasks(self, hypothesis_id: Optional[str] = None) -> List[TaskState]:
        """列出所有任务"""
        return self.memory.list_tasks(hypothesis_id=hypothesis_id)

    def cancel_task(self, task_id: str):
        """取消任务"""
        state = self.memory.load_task_state(task_id)
        if state:
            state.status = TaskStatus.CANCELLED
            state.updated_at = time.time()
            self.memory.save_task_state(state)

    def pause_task(self, task_id: str):
        """暂停任务"""
        state = self.memory.load_task_state(task_id)
        if state and state.status == TaskStatus.RUNNING:
            state.status = TaskStatus.PAUSED
            state.updated_at = time.time()
            self.memory.save_task_state(state)

    def resume_task(self, task_id: str) -> TaskState:
        """恢复任务"""
        return self.execute_task(task_id, resume=True)


class LLMBasedStepValidator:
    """基于LLM的步骤验证器 - 用于复杂步骤结果的智能验证"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def validate_with_llm(
        self,
        step: TaskStep,
        result: StepResult,
        validation_criteria: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        使用LLM验证步骤结果

        Args:
            step: 步骤定义
            result: 步骤执行结果
            validation_criteria: 验证标准，包含：
                - criteria: 具体验证指标
                - expected_format: 期望的输出格式
                - threshold: 阈值（如准确率 > 0.8）

        Returns:
            验证结果字典
        """
        prompt = self._build_validation_prompt(step, result, validation_criteria)

        try:
            response = self.llm_client.chat(prompt)
            validation_result = self._parse_validation_response(response)

            # 添加LLM分析
            validation_result["llm_analysis"] = response

            return validation_result
        except Exception as e:
            return {"valid": False, "reason": f"LLM validation failed: {e}"}

    def _build_validation_prompt(
        self,
        step: TaskStep,
        result: StepResult,
        validation_criteria: Dict[str, Any]
    ) -> str:
        """构建验证提示"""
        return f"""你是一个严谨的验证器，正在验证步骤 "{step.name}" 的执行结果。

## 步骤描述
{step.description}

## 验证标准
{json.dumps(validation_criteria, indent=2, ensure_ascii=False)}

## 执行结果
{json.dumps(result.output_data, indent=2, ensure_ascii=False)}

## 你的任务
1. 分析执行结果是否满足验证标准
2. 检查输出数据格式是否正确
3. 判断是否有错误或异常
4. 输出JSON格式的验证结果：
{{
    "valid": true/false,
    "reason": "具体原因",
    "details": {{"具体发现"}},
    "suggestions": ["如果验证失败，建议如何修复"]
}}
"""

    def _parse_validation_response(self, response: str) -> Dict[str, Any]:
        """解析LLM验证响应"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass

        return {"valid": False, "reason": "Failed to parse LLM response"}


def create_training_task(
    task_id: str,
    hypothesis_id: str,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    snapshot_config: Dict[str, Any]
) -> List[TaskStep]:
    """
    创建模型训练任务（用于H6等需要运行训练的假设验证）

    返回步骤列表：
    1. prepare_training_data - 准备训练数据
    2. setup_training_environment - 设置训练环境（复制代码、修改配置）
    3. run_training - 运行训练并保存嵌入/梯度快照
    4. collect_snapshots - 收集嵌入快照
    5. analyze_results - 分析结果

    Args:
        task_id: 任务ID
        hypothesis_id: 假设ID
        model_config: 模型配置
        training_config: 训练配置
        snapshot_config: 快照配置

    Returns:
        步骤列表
    """
    steps = [
        TaskStep(
            step_id="prepare_training_data",
            name="准备训练数据",
            description="准备训练所需的数据集，包括训练/验证/测试集",
            required_data=["dataset_path", "item_metadata"],
            output_data=["train_data", "valid_data", "test_data"],
            max_retries=2,
            timeout=300.0
        ),
        TaskStep(
            step_id="setup_training_environment",
            name="设置训练环境",
            description="复制模型代码到工作目录，修改配置参数以支持嵌入/梯度保存",
            required_data=["model_code_path", "training_config"],
            output_data=["modified_code_path", "config_file"],
            max_retries=3,
            timeout=300.0
        ),
        TaskStep(
            step_id="run_training",
            name="运行训练",
            description="执行模型训练，每轮保存嵌入向量和梯度日志",
            required_data=["modified_code_path", "config_file", "train_data"],
            output_data=["embeddings_snapshot_dir", "gradient_log_dir"],
            max_retries=2,
            timeout=training_config.get("timeout", 3600),
            metadata={"num_epochs": training_config.get("num_epochs", 10)}
        ),
        TaskStep(
            step_id="collect_snapshots",
            name="收集嵌入快照",
            description="从训练输出中收集所有轮次的嵌入向量",
            required_data=["embeddings_snapshot_dir"],
            output_data=["collected_embeddings"],
            max_retries=2,
            timeout=120.0
        ),
        TaskStep(
            step_id="analyze_embeddings",
            name="分析嵌入相似性与梯度",
            description="计算冷门/热门物品的嵌入相似度和梯度更新频率",
            required_data=["collected_embeddings", "gradient_log_dir", "item_popularity"],
            output_data=["similarity_stats", "gradient_stats", "verification_result"],
            max_retries=3,
            timeout=300.0
        )
    ]

    return steps
