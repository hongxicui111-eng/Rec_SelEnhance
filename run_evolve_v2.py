#!/usr/bin/env python3
"""
RecSelfEvolve V2 — DeepEvolve-inspired 进化引擎启动入口

基于 DeepEvolve 的架构优化：
  - Researcher + Coder Agent 分离
  - Island-based Evolution
  - Deep Research 网络搜索
  - Reflection 反思机制
  - Program Database 版本管理

使用方式:
    python run_evolve_v2.py
    python run_evolve_v2.py --data Beauty --backbone SASRec --iterations 30 --num-islands 4

环境变量:
    LLM_API_URL      LLM 服务地址
    LLM_API_KEY      API Key
    LLM_MODEL        模型名
    PROJECT_ROOT     推荐模型项目根目录
"""
import argparse
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import AgentConfig
from agent.researcher import ResearcherAgent, IdeaData
from agent.coder import CoderAgent
from agent.database import Program, ProgramDatabase
from agent.evolve_engine import EvolutionEngine
from agent.llm_client import LLMClient


def parse_args():
    parser = argparse.ArgumentParser(
        description="RecSelfEvolve V2 — DeepEvolve-inspired 进化引擎",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- LLM 配置 ----
    llm_group = parser.add_argument_group("LLM 配置")
    llm_group.add_argument("--llm-url", default=None,
                           help="LLM API 地址 (默认: http://localhost:8000/v1)")
    llm_group.add_argument("--llm-key", default=None,
                           help="API Key")
    llm_group.add_argument("--llm-model", default=None,
                           help="模型名 (默认: Qwen2.5-72B-Instruct)")
    llm_group.add_argument("--researcher-temp", type=float, default=0.7,
                           help="Researcher LLM 温度 (默认: 0.7)")
    llm_group.add_argument("--coder-temp", type=float, default=0.4,
                           help="Coder LLM 温度 (默认: 0.4)")

    # ---- 项目配置 ----
    proj_group = parser.add_argument_group("推荐模型配置")
    proj_group.add_argument("--project", "--root", default=None,
                            help="推荐模型项目根目录")
    proj_group.add_argument("--data", "--data-name", default="Beauty",
                            choices=["Beauty", "Toys_and_Games", "Yelp",
                                     "Video_Games", "Sports_and_Outdoors"],
                            help="数据集 (默认: Beauty)")
    proj_group.add_argument("--backbone", default="SASRec",
                            choices=["SASRec"],
                            help="模型架构 (默认: SASRec)")
    proj_group.add_argument("--script", default="run_finetune_full.py",
                            help="训练脚本路径")
    proj_group.add_argument("--gpu", default="0",
                            help="GPU ID (默认: 0)")

    # ---- 初始训练参数 ----
    hyper_group = parser.add_argument_group("初始训练参数")
    hyper_group.add_argument("--lr", type=float, default=0.001,
                             help="学习率")
    hyper_group.add_argument("--batch-size", type=int, default=1024,
                             help="Batch size")
    hyper_group.add_argument("--hidden-size", type=int, default=64,
                             help="隐藏层维度")
    hyper_group.add_argument("--neg-sampler", default="Uniform",
                             choices=["Uniform", "DNS"],
                             help="负采样策略")
    hyper_group.add_argument("--loss-type", default="BCE",
                             choices=["BCE", "BPR"],
                             help="损失函数")
    hyper_group.add_argument("--cl-type", default="Radical",
                             choices=["Radical", "Gentle"],
                             help="对比学习类型")
    hyper_group.add_argument("--N", type=int, default=200,
                             help="负采样候选数")
    hyper_group.add_argument("--M", type=int, default=10,
                             help="DNS pool 大小")
    hyper_group.add_argument("--epochs", type=int, default=500,
                             help="最大训练轮次")

    # ---- 进化控制 ----
    evolve_group = parser.add_argument_group("进化控制 (V2 新增)")
    evolve_group.add_argument("-n", "--iterations", type=int, default=20,
                              help="最大迭代轮数 (默认: 20)")
    evolve_group.add_argument("--timeout", type=int, default=7200,
                              help="训练超时秒数 (默认: 7200)")
    evolve_group.add_argument("--strategy", default="balanced",
                              choices=["balanced", "aggressive", "conservative",
                                       "explorative", "focused"],
                              help="初始探索策略")

    # ---- Deep Research 配置 ----
    research_group = parser.add_argument_group("Deep Research 配置 (V2 新增)")
    research_group.add_argument("--max-research-reflect", type=int, default=3,
                                help="Researcher 最大反思轮次 (默认: 3)")
    research_group.add_argument("--search-time-bias", type=float, default=0.5,
                                help="搜索时间偏好 (0=随机, 1=最新) (默认: 0.5)")
    research_group.add_argument("--disable-search", action="store_true",
                                help="禁用网络搜索 (仅用 LLM 内部知识)")

    # ---- Coder 配置 ----
    coder_group = parser.add_argument_group("Coder 配置 (V2 新增)")
    coder_group.add_argument("--max-coding-reflect", type=int, default=3,
                             help="Coder 最大反思/调试轮次 (默认: 3)")

    # ---- Island Evolution 配置 ----
    island_group = parser.add_argument_group("Island Evolution 配置 (V2 新增)")
    island_group.add_argument("--num-islands", type=int, default=4,
                              help="进化岛屿数量 (默认: 4)")
    island_group.add_argument("--population-size", type=int, default=20,
                              help="总种群大小 (默认: 20)")
    island_group.add_argument("--migration-interval", type=int, default=5,
                              help="迁移间隔代数 (默认: 5)")
    island_group.add_argument("--migration-rate", type=float, default=0.1,
                              help="迁移比例 (默认: 0.1)")

    # ---- Checkpoint 配置 ----
    ckpt_group = parser.add_argument_group("Checkpoint 配置")
    ckpt_group.add_argument("--checkpoint-interval", type=int, default=5,
                            help="Checkpoint 保存间隔 (默认: 5)")
    ckpt_group.add_argument("--resume-from", default=None,
                            help="从 checkpoint 目录恢复运行")

    # ---- 惊喜评估配置 ----
    surprise_group = parser.add_argument_group("惊喜评估配置")
    surprise_group.add_argument("--item-text-map", default=None,
                                help="物品 ID → 元数据映射文件路径")
    surprise_group.add_argument("--surprise-topk", type=int, default=20,
                                help="惊喜评估 Top-K 阈值")
    surprise_group.add_argument("--num-wrong-cases", type=int, default=500,
                                help="提取的错误案例数量")
    surprise_group.add_argument("--num-train-subset", type=int, default=500,
                                help="训练子集评估用户数量")
    surprise_group.add_argument("--surprise-threshold", type=float, default=0.5,
                                help="惊喜度阈值")

    # ---- 日志与其他 ----
    parser.add_argument("--log-dir", default="logs_v2",
                        help="日志目录 (默认: logs_v2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查配置, 不运行")
    parser.add_argument("--check-llm", action="store_true",
                        help="仅检查 LLM 服务")

    return parser.parse_args()


def build_config(args) -> AgentConfig:
    """从命令行参数构建基础配置（兼容旧 AgentConfig）"""
    config = AgentConfig()

    # LLM
    config.llm_api_url = args.llm_url or os.environ.get("LLM_API_URL") or config.llm_api_url
    config.llm_api_key = args.llm_key or os.environ.get("LLM_API_KEY") or config.llm_api_key
    config.llm_model = args.llm_model or os.environ.get("LLM_MODEL") or config.llm_model

    # 项目
    config.project_root = args.project or os.environ.get("PROJECT_ROOT") or config.project_root
    config.data_name = args.data
    config.backbone = args.backbone
    config.script_name = args.script
    config.gpu_id = args.gpu

    # 初始训练参数
    config.extra_args = {
        "lr": args.lr,
        "batch_size": args.batch_size,
        "hidden_size": args.hidden_size,
        "neg_sampler": args.neg_sampler,
        "loss_type": args.loss_type,
        "CL_type": args.cl_type,
        "N": args.N,
        "M": args.M,
        "epochs": args.epochs,
    }

    # 进化控制
    config.max_iterations = args.iterations
    config.llm_temperature = args.researcher_temp
    config.train_timeout = args.timeout
    config.log_dir = args.log_dir

    # 惊喜评估配置
    config.item_text_map_path = args.item_text_map or ""
    config.surprise_eval_topk = args.surprise_topk
    config.num_wrong_case_samples = args.num_wrong_cases
    config.num_train_subset = args.num_train_subset
    config.surprise_threshold = args.surprise_threshold

    return config


def build_engine_config(args) -> dict:
    """构建 EvolutionEngine 专用配置"""
    return {
        "model": args.llm_model or os.environ.get("LLM_MODEL") or "Qwen2.5-72B-Instruct",
        "llm_url": args.llm_url or os.environ.get("LLM_API_URL") or "http://localhost:8000/v1",
        "llm_key": args.llm_key or os.environ.get("LLM_API_KEY") or "EMPTY",
        "problem_name": f"{args.backbone}_{args.data}",
        "problem_description": (
            f"优化 {args.backbone} 序列推荐模型在 {args.data} 数据集上的表现。"
            f"当前使用 hidden_size={args.hidden_size}, lr={args.lr}, "
            f"batch_size={args.batch_size}, neg_sampler={args.neg_sampler}, "
            f"loss_type={args.loss_type}, CL_type={args.cl_type}"
        ),
        "researcher_temperature": args.researcher_temp,
        "coder_temperature": args.coder_temp,
        "disable_search": args.disable_search,
    }


def print_config(args, config: AgentConfig):
    print("=" * 60)
    print("  RecSelfEvolve V2 — DeepEvolve-inspired 配置")
    print("=" * 60)
    print(f"  LLM API:      {config.llm_api_url}")
    print(f"  LLM Model:    {config.llm_model}")
    print(f"  Project:      {config.project_root}")
    print(f"  Backbone:     {config.backbone}")
    print(f"  Data:         {config.data_name}")
    print(f"  GPU:          {config.gpu_id}")
    print(f"  Iterations:   {args.iterations}")
    print(f"  Strategy:     {args.strategy}")
    print("-" * 60)
    print(f"  Num Islands:  {args.num_islands}")
    print(f"  Population:   {args.population_size}")
    print(f"  Migration:    every {args.migration_interval} gen, rate {args.migration_rate}")
    print("-" * 60)
    print(f"  Researcher:   temp={args.researcher_temp}, reflect={args.max_research_reflect}")
    print(f"  Coder:        temp={args.coder_temp}, reflect={args.max_coding_reflect}")
    print(f"  Deep Search:  {'DISABLED' if args.disable_search else 'ENABLED'}")
    print("-" * 60)
    print(f"  Checkpoint:   every {args.checkpoint_interval} iterations")
    print(f"  Resume:       {args.resume_from or 'None'}")
    print(f"  Log dir:      {args.log_dir}")
    print("=" * 60)


async def run_evolution(args, config: AgentConfig):
    """运行 V2 进化流程"""

    # 构建 engine 配置
    engine_config = build_engine_config(args)

    # 初始化 EvolutionEngine
    engine = EvolutionEngine(
        config=engine_config,
        project_root=config.project_root,
        max_iterations=args.iterations,
        max_research_reflect=args.max_research_reflect,
        search_time_bias=args.search_time_bias,
        max_coding_reflect=args.max_coding_reflect,
        population_size=args.population_size,
        num_islands=args.num_islands,
        migration_interval=args.migration_interval,
        migration_rate=args.migration_rate,
        checkpoint_interval=args.checkpoint_interval,
    )

    # 恢复 checkpoint
    if args.resume_from:
        engine.database.load(args.resume_from)
        print(f"已从 {args.resume_from} 恢复, 继续迭代 {engine.database.last_iteration}")

    # 设置主题
    query = (
        f"优化 {args.backbone} 在 {args.data} 数据集上的序列推荐性能。"
        f"目标是提升 NDCG@{args.surprise_topk} 和 Recall@{args.surprise_topk}。"
    )
    engine.update_topic(query)

    # 读取初始代码
    from agent.project_adapter import SeqRecAdapter

    adapter = SeqRecAdapter(config)
    initial_code = adapter.get_source_code_concat()

    # 运行基线评估获取初始指标
    print("\n[Step 0] 运行基线评估...")
    from agent.train_runner import TrainRunner
    runner = TrainRunner(config)
    baseline_result = runner.run_train()

    initial_metrics = {}
    if baseline_result and baseline_result.get("metrics"):
        initial_metrics = baseline_result["metrics"]
        print(f"基线指标: {json.dumps(initial_metrics, ensure_ascii=False)}")

    # 创建初始想法
    initial_idea = IdeaData(
        title=f"初始 {args.backbone}",
        description=f"基于 {args.backbone} 的序列推荐模型",
        content=f"标准 {args.backbone} 实现, hidden_size={args.hidden_size}",
    )

    # 运行进化
    print("\n[进化开始]")
    best = await engine.run(
        initial_code=initial_code,
        initial_idea=initial_idea,
        initial_metrics=initial_metrics,
    )

    # 输出结果
    if best:
        print("\n" + "=" * 60)
        print("  进化完成 — 最佳结果")
        print("=" * 60)
        print(f"  Program ID:  {best.id}")
        print(f"  Iteration:   {best.iteration_found}")
        print(f"  Metrics:     {json.dumps(best.metrics, ensure_ascii=False)}")
        print(f"  Idea:        {best.idea.title}")
        print("=" * 60)

        # 保存结果摘要
        os.makedirs(config.log_dir, exist_ok=True)
        summary = {
            "version": "v2",
            "best_metrics": best.metrics,
            "best_iteration": best.iteration_found,
            "total_iterations": args.iterations,
            "num_islands": args.num_islands,
            "population_size": args.population_size,
            "best_program_id": best.id,
        }
        summary_path = os.path.join(config.log_dir, "evolution_summary_v2.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"结果摘要: {summary_path}")
    else:
        print("\n⚠ 进化未产出有效程序")

    return best


def main():
    args = parse_args()
    config = build_config(args)
    print_config(args, config)

    # Dry run
    if args.dry_run:
        print("\n配置正确。运行 --check-llm 检查 LLM 服务, 或直接运行启动进化。")
        return

    # 检查 LLM
    if args.check_llm:
        llm_url = args.llm_url or os.environ.get("LLM_API_URL") or "http://localhost:8000/v1"
        llm_key = args.llm_key or os.environ.get("LLM_API_KEY") or "EMPTY"
        llm_model = args.llm_model or os.environ.get("LLM_MODEL") or "Qwen2.5-72B-Instruct"
        llm = LLMClient(llm_url, llm_key, llm_model)
        ok = llm.check_health()
        print(f"\n{'✓' if ok else '✗'} LLM: {llm_url} [{llm_model}]")
        return

    # 验证项目路径
    if not os.path.isdir(config.project_root):
        print(f"\n⚠ 项目路径不存在: {config.project_root}")
        print("   请设置 PROJECT_ROOT 环境变量或使用 --project")
        sys.exit(1)

    # 验证训练脚本
    script_path = os.path.join(config.project_root, config.script_name)
    if not os.path.isfile(script_path):
        print(f"\n⚠ 训练脚本不存在: {script_path}")
        print(f"   使用 --script 指定正确的路径")
        sys.exit(1)

    # 启动异步进化
    try:
        asyncio.run(run_evolution(args, config))
    except KeyboardInterrupt:
        print("\n\n用户中断。")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()