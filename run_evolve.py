#!/usr/bin/env python3
"""
RecSelfEvolve — 推荐系统自增强 Agent 启动入口

适配你的项目 (Neg_samples_DNS_hx):
    python run_evolve.py
    python run_evolve.py --data Beauty --backbone SASRec --iterations 30

环境变量:
    LLM_API_URL      LLM 服务地址
    LLM_API_KEY      API Key
    LLM_MODEL        模型名
    PROJECT_ROOT     推荐模型项目根目录
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.config import AgentConfig
from agent.core import RecSelfEvolveAgent


def parse_args():
    parser = argparse.ArgumentParser(
        description="RecSelfEvolve — 推荐系统自增强 Agent",
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
    evolve_group = parser.add_argument_group("进化控制")
    evolve_group.add_argument("-n", "--iterations", type=int, default=20,
                              help="最大迭代轮数 (默认: 20)")
    evolve_group.add_argument("--strategy", default="balanced",
                              choices=["balanced", "aggressive", "conservative",
                                       "explorative", "focused"],
                              help="初始探索策略")
    evolve_group.add_argument("--temperature", type=float, default=0.7,
                              help="LLM 温度 (默认: 0.7)")
    evolve_group.add_argument("--timeout", type=int, default=7200,
                              help="训练超时秒数 (默认: 7200)")

    # ---- 其他 ----
    parser.add_argument("--log-dir", default="logs",
                        help="日志目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅检查配置, 不运行")
    parser.add_argument("--check-llm", action="store_true",
                        help="仅检查 LLM 服务")

    # ---- 惊喜评估配置 ----
    surprise_group = parser.add_argument_group("惊喜评估配置")
    surprise_group.add_argument("--item-text-map", default=None,
                                help="物品 ID → 元数据映射文件 (id_meta_data.json) 路径")
    surprise_group.add_argument("--surprise-topk", type=int, default=20,
                                help="惊喜评估 Top-K 阈值")
    surprise_group.add_argument("--num-wrong-cases", type=int, default=500,
                                help="提取的错误案例数量")
    surprise_group.add_argument("--num-train-subset", type=int, default=500,
                                help="训练子集评估用户数量")
    surprise_group.add_argument("--surprise-threshold", type=float, default=0.5,
                                help="惊喜度阈值")

    return parser.parse_args()


def build_config(args) -> AgentConfig:
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

    # 初始训练参数作为 extra_args
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
    config.llm_temperature = args.temperature
    config.train_timeout = args.timeout
    config.log_dir = args.log_dir

    # 惊喜评估配置
    config.item_text_map_path = args.item_text_map or ""
    config.surprise_eval_topk = args.surprise_topk
    config.num_wrong_case_samples = args.num_wrong_cases
    config.num_train_subset = args.num_train_subset
    config.surprise_threshold = args.surprise_threshold

    return config


def print_config(config: AgentConfig):
    print("=" * 60)
    print("  RecSelfEvolve 配置")
    print("=" * 60)
    print(f"  LLM API:    {config.llm_api_url}")
    print(f"  LLM Model:  {config.llm_model}")
    print(f"  Project:    {config.project_root}")
    print(f"  Backbone:   {config.backbone}")
    print(f"  Data:       {config.data_name}")
    print(f"  GPU:        {config.gpu_id}")
    print(f"  Iterations: {config.max_iterations}")
    print(f"  Strategy:   {args.strategy if 'args' in dir() else 'balanced'}")
    print(f"  Log dir:    {config.log_dir}")
    print("=" * 60)


def main():
    global args
    args = parse_args()
    config = build_config(args)
    print_config(config)

    # Dry run
    if args.dry_run:
        print("\n配置正确。运行 --check-llm 检查 LLM 服务, 或直接运行启动进化。")
        return

    # 检查 LLM
    if args.check_llm:
        from agent.llm_client import LLMClient
        llm = LLMClient(config.llm_api_url, config.llm_api_key, config.llm_model)
        ok = llm.check_health()
        print(f"\n{'✓' if ok else '✗'} LLM: {config.llm_api_url} [{config.llm_model}]")
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

    # 启动进化
    agent = RecSelfEvolveAgent(config=config)
    try:
        result = agent.evolve()

        # 保存结果
        os.makedirs(config.log_dir, exist_ok=True)
        summary = {
            "best_metrics": result["best_metrics"],
            "best_iteration": result["best_iteration"],
            "total_iterations": result["total_iterations"],
        }
        summary_path = os.path.join(config.log_dir, "evolution_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n结果摘要: {summary_path}")

    except KeyboardInterrupt:
        print("\n\n用户中断。实验日志: {}".format(
            os.path.join(config.log_dir, config.journal_file)
        ))
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()