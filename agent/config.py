"""
Agent 配置管理
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
    llm_output_tokens: int = 32768           # LLM 单次输出的最大 token 数 (统一超参)
                                             # 对于 thinking/推理模型，需要同时容纳 think chain + 最终回答，
                                             # 推荐 ≥ 16384；纯指令模型可设为 4096-8192
    llm_max_context_tokens: int = 32000       # LLM 最大上下文 token 数
    llm_prompt_safety_ratio: float = 0.85      # prompt 占上下文比例的安全阈值

    # ---- LLM 语义压缩 ----
    llm_enable_semantic_compression: bool = True   # 启用语义压缩 (减少重复信息)
    llm_compression_chunk_chars: int = 3000        # 压缩时的分块字符数
    llm_compression_target_chars: int = 1500       # 压缩后的目标字符数
    llm_compression_enable_cache: bool = True       # 启用压缩缓存
    llm_compression_cache_ttl_seconds: int = 3600   # 缓存过期时间 (秒)
    llm_compression_cache_path: str = "compression_cache.json"  # 缓存文件路径

    # ---- 项目路径 ----
    project_root: str = "/path/to/your/rec_project"
    train_script: str = "train.py"
    eval_script: str = "eval.py"

    # ---- 项目适配器参数 (传给 SeqRecAdapter) ----
    script_name: str = "run_finetune_full.py"  # 训练脚本名
    data_name: str = "Beauty"                  # 数据集名
    backbone: str = "SASRec"                   # 模型 backbone
    gpu_id: str = "0"                          # GPU ID
    output_dir: str = "output"                 # 训练输出目录
    extra_args: dict = field(default_factory=dict)  # 额外训练参数

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
        "ndcg@5": {"min": 0.0, "max": 1.0, "regression_limit": 0.05},
    })

    # ---- 日志 ----
    log_dir: str = "logs"
    journal_file: str = "experiment_journal.jsonl"

    # ---- 代码查询模式 ----
    enable_code_query: bool = True                # 启用代码查询模式 (LLM 按需获取代码)
    max_query_rounds: int = 5                     # 查询模式最大轮数
    code_query_max_chars_per_result: int = 5000   # 每次查询结果的最大字符数

    # ---- 多角色工作流 ----
    enable_multi_role_workflow: bool = False  # 启用 Planner→Researcher→Coder→Debugger 多角色工作流
    researcher_model: str = ""                # 研究者模型 (空=使用 llm_model)
    researcher_temperature: float = 0.7       # 研究者温度
    coder_model: str = ""                     # 编码者模型 (空=使用 llm_model)
    coder_temperature: float = 0.5            # 编码者温度
    debugger_model: str = ""                  # 调试者模型 (空=使用 llm_model)
    debugger_temperature: float = 0.15        # 调试者温度 (低温度更精确)
    planner_temperature: float = 0.7          # 规划者温度
    max_reflection_rounds: int = 3            # 反思最大轮数

    # ---- 惊喜评估配置 ----
    item_text_map_path: str = ""  # 物品 ID → 文本描述映射文件路径
    surprise_eval_topk: int = 20  # 惊喜评估的 Top-K 阈值
    num_wrong_case_samples: int = 500  # 提取的错误案例数量
    num_train_subset: int = 500  # 训练子集评估的用户数量
    surprise_threshold: float = 0.5  # 惊喜度阈值 (≥ 此值为"惊喜"交互)