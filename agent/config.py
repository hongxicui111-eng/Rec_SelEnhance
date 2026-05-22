"""
Agent 配置管理 — 增加项目特定参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """自增强 Agent 的全部配置"""

    # ---- LLM 配置 ----
    llm_api_url: str = "http://localhost:8000/v1"
    llm_api_key: str = "EMPTY"
    llm_model: str = "Qwen2.5-72B-Instruct"
    llm_timeout: int = 120
    llm_max_retries: int = 3
    llm_temperature: float = 0.7
    llm_max_tokens: int = 8192
    # LLM 上下文控制：防止 prompt 过长导致 context overflow
    llm_max_context_tokens: int = 32768
    llm_prompt_safety_ratio: float = 0.75
    # LLM 语义压缩配置
    llm_enable_semantic_compression: bool = True
    llm_compression_chunk_chars: int = 5000
    llm_compression_target_chars: int = 3500
    llm_compression_enable_cache: bool = True
    llm_compression_cache_ttl_seconds: int = 86400
    llm_compression_cache_path: str = "logs/context_compression_cache.json"

    # ---- 项目路径 ----
    project_root: str = "/path/to/your/rec_project"

    # ---- 项目特定配置 ----
    # 这些是 run_finetune_full.py 特有的参数
    data_name: str = "Beauty"           # 数据集
    backbone: str = "SASRec"            # 默认模型
    gpu_id: str = "0"                   # GPU ID
    script_name: str = "run_finetune_full.py"  # 训练脚本
    output_dir: str = "output"          # 输出目录
    extra_args: dict = field(default_factory=dict)  # 其他固定参数

    # ---- 训练容错 ----
    train_timeout: int = 7200  # 秒
    oom_reduce_factor: float = 0.5  # OOM 时 batch_size 缩小比例
    nan_reduce_factor: float = 0.5  # NaN 时 lr 缩小比例

    # ---- 进化控制 ----
    max_iterations: int = 20
    quality_window: int = 5  # 质量监控窗口
    degrade_threshold: float = 0.95  # 连续退化触发回滚的阈值
    plateau_threshold: float = 0.001  # 收敛停滞检测阈值
    exploration_modes: list = field(default_factory=lambda: [
        "balanced", "aggressive", "conservative"
    ])

    # ---- 安全护栏 ----
    metric_guardrails: dict = field(default_factory=lambda: {
        "NDCG@10": {"min": 0.0, "max": 1.0, "regression_limit": 0.05},
        "R@10": {"min": 0.0, "max": 1.0, "regression_limit": 0.05},
    })

    # ---- 日志 ----
    log_dir: str = "logs"
    journal_file: str = "experiment_journal.jsonl"

    # ---- 惊喜评估配置 ----
    item_text_map_path: str = ""  # 物品 ID → 文本描述映射文件路径
    surprise_eval_topk: int = 20  # 惊喜评估的 Top-K 阈值
    num_wrong_case_samples: int = 500  # 提取的错误案例数量
    num_train_subset: int = 500  # 训练子集评估的用户数量
    surprise_threshold: float = 0.5  # 惊喜度阈值 (≥ 此值为"惊喜"交互)

    # ---- 多角色工作流配置 ----
    enable_multi_role_workflow: bool = False  # 是否启用 Planner→Researcher→Coder→Debugger 多角色工作流
    planner_model: str = ""       # Planner 使用的模型 (空则使用 llm_model)
    researcher_model: str = ""    # Researcher 使用的模型
    coder_model: str = ""         # Coder 使用的模型
    debugger_model: str = ""      # Debugger 使用的模型
    planner_temperature: float = 0.7   # Planner 温度
    researcher_temperature: float = 0.7  # Researcher 温度
    coder_temperature: float = 0.4      # Coder 温度
    debugger_temperature: float = 0.2   # Debugger 温度
    max_reflection_rounds: int = 3      # 最大反思轮次