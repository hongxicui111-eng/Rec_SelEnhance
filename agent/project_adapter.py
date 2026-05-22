"""
ProjectAdapter — 推荐项目适配器

这个适配器是 Agent 和你的具体推荐项目之间的桥梁。
它知道:
1. 你的项目怎么运行 (命令行格式)
2. 你的评估指标怎么输出 (文本日志 → 结构化 metrics)
3. 哪些参数可以被 LLM 修改 (搜索空间)
"""

import re
import json
import os
import shlex
import logging
from typing import Optional, List

logger = logging.getLogger("rec_self_evolve.adapter")


class SeqRecAdapter:
    """
    序列推荐 (SeqRec) 项目适配器
    适配 run_finetune_full.py 风格的训练脚本
    
    这是**你需要关注和修改的核心文件**。
    如果你换了项目（比如换个推荐框架），只需要修改这个适配器。
    """

    # ════════════════════════════════════════
    # 1. 项目描述 (给 LLM 知道它在操作什么)
    # ════════════════════════════════════════

    PROJECT_DESCRIPTION = """
### 项目信息
这是一个**序列推荐 (Sequential Recommendation) 模型**项目。
- 训练脚本: run_finetune_full.py (接受命令行参数)
- 这是一个**完全开放**的系统，LLM 可以自由探索任何可能的模型架构、损失函数、采样策略

### 核心理念: 自主探索与创新
**不要被现有实现限制想象力!** LLM 应该：
1. 质疑现有设计: 当前 SASRec + InfoNCE 是最优的吗?
2. 大胆提出新架构:  替代 Transformer?
3. 探索新损失函数: 对比学习、度量学习、生成式损失...任何可能的方案
4. 发明新采样策略: 基于图、基于知识、基于强化学习的采样...
5. 添加新模块: 注意力机制、记忆网络、多兴趣提取器、时序建模...

### 模型源码结构 (LLM 可以修改的代码文件)
项目中的所有 .py 文件都可以被修改，LLM 可以自由探索:

1. **models.py** — 模型顶层定义
   - `SRModel`: 基础推荐模型类 (item_embeddings + init_weights)
   - `SASRec`: SASRec 模型类 (继承 SRModel)
     - `add_position_embedding()`: 物品嵌入 + 位置嵌入 + LayerNorm + Dropout
     - `finetune()`: 构建因果 mask → 编码 → 输出 sequence_output
     - 使用 `Encoder` (来自 modules.py) 作为 item_encoder

2. **modules.py** — Transformer 编码器组件
   - `LayerNorm`: 标准 LayerNorm
   - `SelfAttention`: 多头自注意力 (Q/K/V Linear → scaled dot-product → dropout → dense → LayerNorm)
   - `Intermediate`: 前馈网络 (dense_1: hidden→4*hidden → act → dense_2: 4*hidden→hidden → LayerNorm)
   - `EncoderLayer`: SelfAttention + Intermediate
   - `Encoder`: N 个 EncoderLayer 堆叠

3. **trainers.py** — 训练器 (LLM 也可修改训练逻辑)
   - 包含训练循环、损失计算、评估逻辑
   - `_get_neg_sample()`: 负采样逻辑
   - `acc_metric()`: 评估指标计算

4. **datasets.py** — 数据集处理

### 可执行的结构修改类型 (完全开放!)
**没有任何预定义的修改类型限制!** LLM 可以提出任何类型的修改:
- 修改、添加、替换任何模型组件
- 修改训练逻辑、损失计算、采样策略
- 添加任何新模块或完全重写某个组件
- 任何你认为能提升模型性能的修改

每项结构修改必须输出: (1) 修改哪个文件 (2) 修改哪个类/函数 (3) 具体的代码变更 (Python diff 格式)


**关键原则**: 如果现有实现不支持你想要的方法，直接修改代码来实现它!
"""

    # ════════════════════════════════════════
    # 2. LLM 可以修改的参数空间 + 结构修改空间
    # ════════════════════════════════════════

    # --- 源码文件: LLM 可以修改项目中任何 .py 文件 ---
    # 不预设范围，LLM 可以直接指定任意 .py 文件路径
    SOURCE_FILE_MAP = {
        # ── 核心模型文件 ──
        "models.py": "models.py",
        "modules.py": "modules.py",
        "trainers.py": "trainers.py",
        "datasets.py": "datasets.py",
        # ── 工具与评估文件 ──
        "utils.py": "utils.py",
        "error_case_extractor.py": "error_case_extractor.py",
        "surprise_eval.py": "surprise_eval.py",
        # ── 训练脚本 ──
        "run_finetune_full.py": "run_finetune_full.py",
        # ── 数据处理 ──
        "data_process.py": "data/data_process.py",
        # LLM 也可以通过完整路径指定其他文件
    }

    # --- 结构修改类型: 完全开放，没有预定义限制 ---
    # LLM 可以自由描述任何类型的结构修改，不需要使用预定义的 action_type
    # 系统会接受任何合理的结构修改请求
    STRUCTURAL_ACTIONS = {
        # 这是一个完全开放的字典
        # LLM 可以提出任何类型的修改，不需要局限于以下类型
        # 保留这些只是为了兼容性，LLM 可以完全忽略它们
        "modify": {
            "desc": "修改已有的类/函数/模块",
            "note": "可以是任何修改: 注意力机制、FFN、位置编码、嵌入、前向传播、训练逻辑等",
        },
        "add_module": {
            "desc": "添加新的模块/类/组件",
            "note": "可以是任何新组件: 新架构、新注意力、新损失、新采样器等",
        },
        "add_loss": {
            "desc": "添加新的损失组件或训练目标",
            "note": "可以是任何额外的损失项: 对比学习、度量学习、生成式、对抗性等",
        },
        "replace_backbone": {
            "desc": "替换整个模型 backbone",
            "note": "可以用任何模型替代 SASRec"
        },
        "custom": {
            "desc": "任何其他自定义修改",
            "note": "LLM 可以自由定义，不受任何限制",
        },
    }

    # --- 可调参数: 完全开放，LLM 可以自由探索任何参数 ---
    # 不再预设固定参数列表，LLM 可以提出任何新的参数
    # range 仅作为宽松参考; LLM 可以提出超出范围的值，系统会做温和提醒而非拒绝
    # 关键: LLM 也可以提出全新的参数名，系统会尝试将其添加到命令行
    TUNABLE_PARAMS = {
        # === 超参数 (宽松范围，LLM 可自由选择) ===
        "lr": {"type": "float", "range": [1e-6, 1e-1], "default": 0.001, "soft_limit": True,
               "desc": "学习率 (宽松范围，LLM 可自由选择任何正值)"},
        "batch_size": {"type": "int", "range": [16, 8192], "default": 1024, "soft_limit": True,
                       "desc": "批量大小 (宽松范围，LLM 可自由选择)"},
        "hidden_size": {"type": "int", "range": [16, 1024], "default": 64, "soft_limit": True,
                        "desc": "隐藏层维度 (宽松范围，注意与模型结构对齐)"},
        "hidden_dropout_prob": {"type": "float", "range": [0.0, 0.99], "default": 0.5, "soft_limit": True,
                                "desc": "Dropout 概率 (0.0-0.99，LLM 可自由探索)"},
        "attention_probs_dropout_prob": {"type": "float", "range": [0.0, 0.99], "default": 0.5, "soft_limit": True,
                                         "desc": "Attention Dropout (0.0-0.99)"},
        "weight_decay": {"type": "float", "range": [0.0, 0.5], "default": 0.0, "soft_limit": True,
                         "desc": "权重衰减 (宽松范围，LLM 可自由选择)"},
        "max_seq_length": {"type": "int", "range": [10, 500], "default": 50, "soft_limit": True,
                           "desc": "最大序列长度 (宽松范围)"},

        # === 架构参数 (LLM 可自由探索) ===
        "num_hidden_layers": {"type": "int", "range": [1, 16], "default": 2, "soft_limit": True,
                              "desc": "Transformer 层数 (1-16，LLM 可自由探索)"},
        "num_attention_heads": {"type": "int", "range": [1, 32], "default": 2, "soft_limit": True,
                                "desc": "Attention 头数 (1-32)"},
        "hidden_act": {"type": "str", "choices": None, "default": "gelu", "soft_limit": True,
                       "desc": "激活函数 (可以是任何合法的 PyTorch 激活函数名)"},

        # === 损失函数 (完全开放，LLM 可以探索任何损失) ===
        "loss_type": {"type": "str", "choices": None, "default": "InfoNCE", "soft_limit": True,
                      "desc": "损失函数类型 (完全开放，自主选择)"},
        "temperature": {"type": "float", "range": [0.01, 2.0], "default": 0.1, "soft_limit": True,
                        "desc": "温度系数 (用于 InfoNCE 等对比学习损失，LLM 可探索 0.001~10.0)"},
        "margin": {"type": "float", "range": [0.0, 5.0], "default": 1.0, "soft_limit": True,
                   "desc": "Triplet/Margin loss margin (LLM 可探索 0.0~10.0)"},
        "tau": {"type": "float", "range": [0.01, 2.0], "default": 0.1, "soft_limit": True,
                "desc": "对比学习温度 tau (类似 temperature，LLM 可探索 0.001~10.0)"},

        # === 负采样策略 (完全开放，LLM 可以发明新采样策略) ===
        "neg_sampler": {"type": "str", "choices": None, "default": "Uniform", "soft_limit": True,
                        "desc": "负采样策略 (完全开放，自主选择)"},
        "N": {"type": "int", "range": [1, 10000], "default": 200, "soft_limit": True,
              "desc": "负采样候选数 (LLM 可探索 1~100000)"},
        "M": {"type": "int", "range": [1, 1000], "default": 10, "soft_limit": True,
              "desc": "负采样池大小 (LLM 可探索 1~10000)"},

        # === 对比学习 (完全开放) ===
        "CL_type": {"type": "str", "choices": None, "default": "Radical", "soft_limit": True,
                    "desc": "对比学习类型 (完全开放，自主选择)"},
        "start_epoch": {"type": "int", "range": [0, 1000], "default": 30, "soft_limit": True,
                        "desc": "开始困难负采样的轮次 (LLM 可探索 0~10000)"},
        "K": {"type": "float", "range": [0.0, 1.0], "default": 0.05, "soft_limit": True,
              "desc": "对比学习超参数 (LLM 可探索 0.0~10.0)"},

        # === 训练参数 (宽松范围) ===
        "epochs": {"type": "int", "range": [10, 2000], "default": 500, "soft_limit": True,
                   "desc": "最大训练轮次 (宽松范围)"},
        "seed": {"type": "int", "range": [0, 99999], "default": 42, "soft_limit": True,
                 "desc": "随机种子"},

        # === 模型结构参数 (新增，完全开放) ===
        "backbone": {"type": "str", "choices": None, "default": "SASRec", "soft_limit": True,
                     "desc": "模型 backbone (完全开放，自主选择)"},
        "embedding_dim": {"type": "int", "range": [16, 2048], "default": 64, "soft_limit": True,
                         "desc": "嵌入维度 (与 hidden_size 类似但独立，LLM 可探索 16~4096)"},
        "num_layers": {"type": "int", "range": [1, 32], "default": 2, "soft_limit": True,
                       "desc": "模型层数 (与 num_hidden_layers 类似，LLM 可探索 1~64)"},

        # === 优化器参数 (新增) ===
        "optimizer": {"type": "str", "choices": None, "default": "Adam", "soft_limit": True,
                     "desc": "优化器类型 (完全开放，自主选择)"},
        "adam_beta1": {"type": "float", "range": [0.0, 1.0], "default": 0.9, "soft_limit": True,
                       "desc": "Adam beta1 (0.0-1.0)"},
        "adam_beta2": {"type": "float", "range": [0.0, 1.0], "default": 0.999, "soft_limit": True,
                       "desc": "Adam beta2 (0.0-1.0)"},
        "adam_epsilon": {"type": "float", "range": [1e-10, 1e-1], "default": 1e-8, "soft_limit": True,
                         "desc": "Adam epsilon (1e-10~1e-1)"},

        # === 学习率调度 (新增) ===
        "lr_scheduler": {"type": "str", "choices": None, "default": "none", "soft_limit": True,
                        "desc": "学习率调度器 (完全开放，自主选择)"},
        "lr_decay_step": {"type": "int", "range": [1, 10000], "default": 1000, "soft_limit": True,
                          "desc": "学习率衰减步数"},
        "lr_decay_rate": {"type": "float", "range": [0.01, 1.0], "default": 0.5, "soft_limit": True,
                          "desc": "学习率衰减率"},
        "warmup_steps": {"type": "int", "range": [0, 10000], "default": 0, "soft_limit": True,
                         "desc": "warmup 步数"},

        # === 正则化 (新增) ===
        "layer_norm_eps": {"type": "float", "range": [1e-10, 1e-3], "default": 1e-8, "soft_limit": True,
                          "desc": "LayerNorm epsilon"},
        "initializer_range": {"type": "float", "range": [0.001, 1.0], "default": 0.02, "soft_limit": True,
                             "desc": "初始化范围"},

        # === 评估参数 (新增) ===
        "eval_at": {"type": "str", "choices": None, "default": "5,10,20", "soft_limit": True,
                    "desc": "评估 K 值 (如 5,10,20，LLM 可以设置任意组合)"},
        "do_eval": {"type": "bool", "choices": None, "default": True, "soft_limit": True,
                    "desc": "是否在训练过程中评估"},
        "eval_batch_size": {"type": "int", "range": [1, 4096], "default": 256, "soft_limit": True,
                            "desc": "评估批量大小"},

        # === 早停与保存 (新增) ===
        "patience": {"type": "int", "range": [1, 100], "default": 10, "soft_limit": True,
                     "desc": "早停耐心值"},
        "save_freq": {"type": "int", "range": [1, 100], "default": 100, "soft_limit": True,
                      "desc": "保存频率 (轮次)"},
        "log_freq": {"type": "int", "range": [1, 1000], "default": 1, "soft_limit": True,
                     "desc": "日志输出频率 (轮次)"},

        # === 梯度相关 (新增) ===
        "max_grad_norm": {"type": "float", "range": [0.1, 10.0], "default": 5.0, "soft_limit": True,
                         "desc": "梯度裁剪范数"},
        "gradient_accumulation_steps": {"type": "int", "range": [1, 64], "default": 1, "soft_limit": True,
                                        "desc": "梯度累积步数"},
    }

    # ════════════════════════════════════════
    # 3. 默认命令模板
    # ════════════════════════════════════════

    def __init__(self, project_root: str, script_name: str = "run_finetune_full.py",
                 data_name: str = "Beauty", backbone: str = "SASRec",
                 gpu_id: str = "0", output_dir: str = "output",
                 extra_args: dict = None):
        self.project_root = project_root
        self.script_name = script_name
        self.data_name = data_name
        self.backbone = backbone
        self.gpu_id = gpu_id
        self.output_dir = output_dir
        # 默认参数 (除 project_root/data/backbone 以外的固定参数)
        self.base_args = {
            "data_name": data_name,
            "backbone": backbone,
            "ckp": 0,
            "num_split": 6,
            "hidden_size": 64,
            "lr": 0.001,
            "batch_size": 1024,
            "epochs": 500,
            "no_cuda": False,
            "log_freq": 1,
            "seed": 42,
            "weight_decay": 0.0,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "gpu_id": gpu_id,
            "N": 200,
            "M": 10,
            "neg_sampler": "Uniform",
            "loss_type": "InfoNCE",
            "temperature": 0.1,
            "tau": 0.1,
            "margin": 1.0,
            "CL_type": "Radical",
            "start_epoch": 30,
            "K": 0.05,
            "hidden_dropout_prob": 0.5,
            "attention_probs_dropout_prob": 0.5,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_seq_length": 50,
            "hidden_act": "gelu",
            "initializer_range": 0.02,
        }
        if extra_args:
            self.base_args.update(extra_args)

    # ════════════════════════════════════════
    # 4. 构建训练命令 (核心!)
    # ════════════════════════════════════════

    def build_train_command(self, param_overrides: dict = None,
                            eval_only: bool = False) -> str:
        """
        根据参数变更，构建完整的训练命令
        
        Args:
            param_overrides: 要修改的参数 {"lr": 0.0005, "batch_size": 512, ...}
            eval_only: 是否只做评估 (do_eval=True)
            
        Returns:
            str: 完整的 shell 命令
        """
        # 合并参数: base + overrides
        args = dict(self.base_args)
        if param_overrides:
            args.update(param_overrides)
            logger.info(f"Param overrides applied: {param_overrides}")

        # 构建命令行
        # ⚠ 关键修复: cd 后必须加 &&, 否则 \ 续行符会把后续所有参数变成 cd 的额外参数
        # 导致 cd 成功 (exit 0) 但训练脚本根本不执行
        cmd_parts = [
            f"cd {self.project_root} &&",
            f"CUDA_VISIBLE_DEVICES={self.gpu_id}",
            "python3 -u",
            self.script_name,
        ]

        # 添加每个参数
        for key, value in args.items():
            if isinstance(value, bool) and value:
                cmd_parts.append(f"--{key}")
            elif isinstance(value, bool) and not value:
                pass  # 不传 False 的 bool 参数
            else:
                cmd_parts.append(f'--{key}={value}')

        if eval_only:
            cmd_parts.append("--do_eval")

        return " \\\n    ".join(cmd_parts)

    def validate_train_command(self, cmd: str) -> dict:
        """
        对即将执行的训练命令做前置校验（开源 Agent 常用 guard）。

        Returns:
            {
                "ok": bool,
                "issues": [str],
                "warnings": [str],
            }
        """
        issues: List[str] = []
        warnings: List[str] = []

        cmd = (cmd or "").strip()
        if not cmd:
            return {"ok": False, "issues": ["Empty command"], "warnings": []}

        # 必须包含 cd && python3 -u script
        if "cd " not in cmd:
            issues.append("Missing 'cd <project_root>' prefix")
        if "&&" not in cmd:
            issues.append("Missing '&&' after cd, command chaining is unsafe")
        if "python3 -u" not in cmd:
            warnings.append("Command does not contain 'python3 -u' (buffered output may hide logs)")

        # 目录存在性
        project_path = os.path.normpath(self.project_root)
        if not os.path.exists(project_path):
            issues.append(f"Project root not found: {project_path}")

        # 脚本存在性（优先 project_root，其次 Recmodel）
        script_candidates = [
            os.path.join(project_path, self.script_name),
            os.path.join(project_path, "Recmodel", self.script_name),
        ]
        if not any(os.path.exists(p) for p in script_candidates):
            issues.append(
                f"Training script not found: {self.script_name} (checked: {script_candidates})"
            )

        # 参数结构快速校验（检查是否有明显坏 token）
        # 例如 --k v（中间被拆）不是本项目推荐格式，建议 --k=v
        try:
            tokens = shlex.split(cmd)
            for i, tk in enumerate(tokens):
                if tk.startswith("--") and "=" not in tk:
                    # 允许 bool flag（无值）但这里的训练参数基本都应为 --k=v
                    if i + 1 < len(tokens) and not tokens[i + 1].startswith("--"):
                        warnings.append(
                            f"Prefer '--key=value' style, found split arg: {tk} {tokens[i+1]}"
                        )
        except Exception:
            warnings.append("Unable to parse command with shlex; shell escaping may be risky")

        return {
            "ok": len(issues) == 0,
            "issues": issues,
            "warnings": warnings,
        }


    def parse_metrics_from_log(self, log_text: str) -> dict:
        """
        从训练日志中解析评估指标
        
        训练脚本输出格式 (run_finetune_full.py):
        
        full_sort 模式 (最终测试结果):
            {'Epoch': 100, 'HIT@5': '0.2790', 'NDCG@5': '0.2241',
             'HIT@10': '0.5357', 'NDCG@10': '0.3382', ...}
        
        sample 模式 (每 epoch 训练输出):
            {'epoch': 5, 'loss': '0.1234'}
            {'Epoch': 5, 'HR_5': '0.1234', 'NDCG_5': '0.5678', ...}
        
        通用格式:
            R_5=0.1234 NDCG_5=0.5678 MRR_5=0.9012
        """
        metrics = {}

        # ── 模式 1: HIT@K='value' 格式 (get_full_sort_score 输出) ──
        hit_pattern = r"'HIT@(\d+)':\s*'([\d.]+)'"
        for match in re.finditer(hit_pattern, log_text):
            k, v = match.group(1), match.group(2)
            metrics[f"HIT@{k}"] = float(v)

        # ── 模式 2: NDCG@K='value' 格式 (get_full_sort_score 输出) ──
        ndcg_at_pattern = r"'NDCG@(\d+)':\s*'([\d.]+)'"
        for match in re.finditer(ndcg_at_pattern, log_text):
            k, v = match.group(1), match.group(2)
            metrics[f"NDCG@{k}"] = float(v)

        # ── 模式 3: HR_K='value' 格式 (get_sample_scores 输出) ──
        hr_pattern = r"'HR_(\d+)':\s*'([\d.]+)'"
        for match in re.finditer(hr_pattern, log_text):
            k, v = match.group(1), match.group(2)
            metrics[f"HR@{k}"] = float(v)

        # ── 模式 4: NDCG_K='value' 格式 (get_sample_scores 输出) ──
        ndcg_underscore_pattern = r"'NDCG_(\d+)':\s*'([\d.]+)'"
        for match in re.finditer(ndcg_underscore_pattern, log_text):
            k, v = match.group(1), match.group(2)
            # 不覆盖已有的 NDCG@K
            if f"NDCG@{k}" not in metrics:
                metrics[f"NDCG@{k}"] = float(v)

        # ── 模式 5: MRR@K='value' 或 MRR_K='value' ──
        mrr_pattern = r"'MRR@(\d+)':\s*'([\d.]+)'|'MRR_(\d+)':\s*'([\d.]+)'"
        for match in re.finditer(mrr_pattern, log_text):
            k = match.group(1) or match.group(3)
            v = match.group(2) or match.group(4)
            metrics[f"MRR@{k}"] = float(v)

        # ── 模式 6: 通用 R_K=value NDCG_K=value MRR_K=value 格式 ──
        general_patterns = [
            (r"R_(\d+)\s*=\s*([\d.]+)", "R@"),
            (r"NDCG_(\d+)\s*=\s*([\d.]+)", "NDCG@"),
            (r"MRR_(\d+)\s*=\s*([\d.]+)", "MRR@"),
        ]
        for pattern, prefix in general_patterns:
            for match in re.finditer(pattern, log_text):
                k, v = match.group(1), match.group(2)
                key = f"{prefix}{k}"
                if key not in metrics:  # 不覆盖已有值
                    metrics[key] = float(v)

        # ── 解析 Loss ──
        loss_match = re.search(r"'loss':\s*'([\d.e+\-]+)'", log_text)
        if loss_match:
            metrics["loss"] = float(loss_match.group(1))

        # ── 解析 Early Stopping ──
        # "Early stopping" 表示训练收敛
        if "Early stopping" in log_text:
            # 尝试找到早停的 epoch
            early_stop_match = re.search(r"EP_\w+:(\d+)", log_text[-200:])
            if early_stop_match:
                metrics["early_stop_epoch"] = int(early_stop_match.group(1))

        # ── 解析最终 epoch ──
        # 训练脚本最后输出的 Epoch 字段
        final_epoch_match = re.findall(r"'Epoch':\s*(\d+)", log_text)
        if final_epoch_match:
            metrics["epoch"] = int(final_epoch_match[-1])  # 取最后一个 Epoch

        return metrics

    def format_metrics_for_llm(self, metrics: dict) -> str:
        """格式化指标为 LLM 友好的字符串"""
        if not metrics:
            return "暂无评估指标"
        parts = []
        # 优先显示重要指标
        for key in ["NDCG@10", "NDCG@5", "NDCG@20", "R@10", "R@5", "R@20",
                     "MRR@10", "loss"]:
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        # 其他指标
        for k, v in metrics.items():
            if k not in ["NDCG@10", "NDCG@5", "NDCG@20", "R@10", "R@5",
                          "R@20", "MRR@10", "loss"]:
                if isinstance(v, (int, float)):
                    parts.append(f"{k}={v:.4f}")
                else:
                    parts.append(f"{k}={v}")
        return " | ".join(parts)

    # ════════════════════════════════════════
    # 6. 构建 LLM 上下文 — 让 LLM 理解项目
    # ════════════════════════════════════════

    def build_llm_context(self, current_args: dict = None) -> str:
        """
        构建项目的上下文描述，作为 LLM Prompt 的一部分
        让 LLM 理解它可以修改什么 (参数 + 模型结构)

        核心理念: 完全开放，让 LLM 自主探索
        """
        args = current_args or self.base_args

        context = f"""
--- 项目上下文 (Project Context) ---

{self.PROJECT_DESCRIPTION}

### 当前训练配置
```
数据: {args.get('data_name', '?')}
模型: {args.get('backbone', '?')}
损失: {args.get('loss_type', '?')}
负采样: {args.get('neg_sampler', '?')} (N={args.get('N', '?')}, M={args.get('M', '?')})
对比学习: {args.get('CL_type', '?')}
学习率: {args.get('lr', '?')}
Batch Size: {args.get('batch_size', '?')}
隐藏层: {args.get('hidden_size', '?')} (层数={args.get('num_hidden_layers', '?')})
序列长度: {args.get('max_seq_length', '?')}
```

### 可调优参数列表 (超参数修改 — 完全开放!)
```
{json.dumps({k:
    {"type": v["type"],
     "current": args.get(k, v["default"]),
     "desc": v["desc"]}
    for k, v in self.TUNABLE_PARAMS.items()
}, indent=2, ensure_ascii=False)}
```

**重要**: 上述参数列表不是限制! LLM 可以:
1. 自由探索列表中的任何参数 (范围仅供参考，不是限制)
2. 提出全新的参数名，系统会尝试将其添加到命令行
3. 质疑任何参数的合理性，提出替代方案

### 可执行的结构修改类型 (模型结构修改 — 完全开放!)
```
{json.dumps({k:
    {"desc": v["desc"],
     "note": v.get("note", ""),
     "target_file": v.get("target_file", "see SOURCE_FILE_MAP"),
     "risk": v.get("risk", "low")}
    for k, v in self.STRUCTURAL_ACTIONS.items()
}, indent=2, ensure_ascii=False)}
```

### 可修改的源码文件 (Recmodel 目录下全部 .py 文件)
```
{json.dumps(self.SOURCE_FILE_MAP, indent=2, ensure_ascii=False)}
```

**重要**: 上述修改类型和文件列表不是限制! LLM 可以:
1. 提出任何类型的结构修改，不需要局限于预定义类型
2. 完全替换或重写任何组件
3. 修改任何 Recmodel 目录下的源码文件
4. 从对推荐系统的深度理解出发提出创新方案

### 注意事项
1. 训练脚本是 `run_finetune_full.py`，所有参数通过命令行传入
2. 如果有 `checkpoint_path`，训练会从已有 checkpoint 继续
3. **结构修改时务必确保**: 新增模块的输入/输出维度与 hidden_size 对齐，新增参数需要 args 支持
4. **结构修改是关键**: 不要只调参数! 如果模型瓶颈是架构性的，请提出结构性修改方案
5. **完全开放的理念**: 不要被现有实现限制想象力! 任何创新都是可能的
"""
        return context

    def get_source_code(self, file_key: str) -> Optional[str]:
        """
        读取指定源码文件的完整内容，供 LLM 分析和提出修改
        
        Args:
            file_key: SOURCE_FILE_MAP 中的 key (如 "models.py", "modules.py")
            
        Returns:
            str: 文件内容，或 None (文件不存在)
        """
        rel_path = self.SOURCE_FILE_MAP.get(file_key)
        if not rel_path:
            return None
        
        # 查找文件: 先在 project_root 下找，再在 Recmodel 子目录下找
        candidates = [
            os.path.join(self.project_root, rel_path),
            os.path.join(self.project_root, "Recmodel", rel_path),
        ]
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        logger.warning(f"Source file {file_key} not found at {candidates}")
        return None

    def build_source_code_context(self, include_files: list = None,
                                  max_total_chars: int = 8000,
                                  iterative_memory=None) -> str:
        """
        构建模型源码的上下文，让 LLM 能看到当前代码并提出修改
        
        ⚠ 不再使用简单的截断策略! 改为两种模式:
        - 如果提供了 iterative_memory (IterativeMemory实例) → 使用智能截断:
          优先展示被修改过的区域 + 核心类定义签名
        - 如果没有 iterative_memory → 展示完整代码 (但仍保留长度上限)
        
        Args:
            include_files: 要包含的源码文件列表 (默认包含核心三个文件)
            max_total_chars: 最大总字符数 (默认 8000，远大于之前的 4000)
            iterative_memory: IterativeMemory 实例 (用于智能截断)
        
        Returns:
            str: 格式化的源码上下文
        """
        include_files = include_files
        
        # ── 如果有 IterativeMemory，使用智能截断 ──
        if iterative_memory is not None:
            return iterative_memory.build_smart_source_context(
                include_files=include_files,
                max_total_chars=max_total_chars,
            )
        
        # ── 降级模式: 无 IterativeMemory，展示完整代码 ──
        # (不再像以前那样截断到 4000 字符，而是给更多空间)
        parts = []
        remaining_chars = max_total_chars
        
        for file_key in include_files:
            code = self.get_source_code(file_key)
            if code:
                rel_path = self.SOURCE_FILE_MAP.get(file_key, file_key)
                
                # 不再盲目截断前 4000 字! 改为在总限制下合理分配
                if len(code) <= remaining_chars - 200:
                    # 文件不长，完整展示
                    parts.append(f"### 文件: {file_key} ({rel_path})\n```python\n{code}\n```")
                    remaining_chars -= len(code) + 200
                else:
                    # 文件太长 → 只展示核心类定义和关键方法
                    import ast
                    try:
                        tree = ast.parse(code)
                        # 找到所有类和函数定义的行号范围
                        key_sections = []
                        lines = code.split('\n')
                        for node in ast.iter_child_nodes(tree):
                            if isinstance(node, ast.ClassDef):
                                # 类的 __init__ 和关键方法
                                start = node.lineno - 1
                                end = min(node.end_lineno, len(lines))
                                # 只展示类的开头 (定义 + __init__ + forward/finetune)
                                section_lines = lines[start:start + min(30, end - start)]
                                key_sections.append(f"# === {node.name} ===\n" + 
                                                    '\n'.join(section_lines) + 
                                                    "\n# ... [部分省略]")
                            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                start = node.lineno - 1
                                end = min(node.end_lineno, len(lines))
                                key_sections.append('\n'.join(lines[start:end]))
                        
                        truncated_code = '\n\n'.join(key_sections)
                        if len(truncated_code) > remaining_chars - 200:
                            truncated_code = truncated_code[:remaining_chars - 200]
                        parts.append(
                            f"### 文件: {file_key} ({rel_path})\n"
                            f"⚠ 文件较长 ({len(code)} chars)，展示关键定义\n"
                            f"```python\n{truncated_code}\n```"
                        )
                        remaining_chars -= len(truncated_code) + 200
                    except SyntaxError:
                        # AST 解析失败，展示前面部分
                        parts.append(
                            f"### 文件: {file_key} ({rel_path})\n"
                            f"⚠ 文件较长，展示前 {remaining_chars} chars\n"
                            f"```python\n{code[:remaining_chars - 200]}\n```"
                        )
                        remaining_chars = 0
        
        if parts:
            return "\n\n".join(parts)
        return ""


# ════════════════════════════════════════
# 工厂函数 — 创建适配器
# ════════════════════════════════════════

def create_adapter(project_root: str, **kwargs) -> SeqRecAdapter:
    """
    创建项目适配器
    如果将来你有多个项目，可以在这里做分发
    
    用法:
        adapter = create_adapter("/path/to/project", backbone="SASRec")
    """
    return SeqRecAdapter(project_root=project_root, **kwargs)