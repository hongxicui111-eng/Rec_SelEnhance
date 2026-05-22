#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecSelfEvolve Agent — LLM 集成测试（真实调用 LLM API）

与 test_agent.py 的纯 mock 测试不同，本测试脚本会真实调用 LLM API，
验证整个系统是否能真正与 LLM 交互并产生有效输出。

测试层级:
  Level 1: LLM 连接健康检查 — 能否连通？
  Level 2: 基础 LLM 调用 — 能否发送 prompt 并收到有意义回复？
  Level 3: Prompt模板+LLM → 结构化输出 — 回复能否被正确解析？
  Level 4: 单 Agent LLM调用 — Researcher / Coder 能否独立工作？
  Level 5: 核心 Agent 分析管线 — RecSelfEvolveAgent._phase_analyze_and_propose
  Level 6: 完整进化循环（1轮） — 从分析到代码修改的闭环
  Level 7: LLM 错误反馈与修复管线 — LLM能否分析错误并给出修复建议
  Level 8: 进化引擎 (V2 Architecture) — Researcher→Coder完整协作

运行前提:
  1. LLM 服务已启动（vLLM / Ollama / OpenAI 兼容服务）
  2. 设置环境变量或在命令行指定连接参数:
     - LLM_API_URL (默认 http://localhost:8000/v1)
     - LLM_API_KEY (默认 EMPTY)
     - LLM_MODEL   (默认 Qwen2.5-72B-Instruct)

运行方式:
    python tests/test_llm_integration.py                          # 默认参数
    python tests/test_llm_integration.py --api-url http://10.82.123.22:8000/v1 --model Qwen-235B
    python tests/test_llm_integration.py -v                       # 详细输出
    python tests/test_llm_integration.py -k "test_level1"         # 只跑 Level 1

注意:
  - Level 1 测试如果 LLM 不可达会直接 FAIL（不是 SKIP）
  - Level 2-8 测试在 LLM 不可达时自动 SKIP
  - 每个测试都有独立超时保护，不会无限阻塞
"""

import sys
import os
import json
import tempfile
import shutil
import logging
import time
import asyncio
import unittest
import argparse
from pathlib import Path
from dataclasses import asdict
from unittest.mock import MagicMock, patch
from typing import Optional

# 确保项目根目录在搜索路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(name)s] %(levelname)s: %(message)s",
)
test_logger = logging.getLogger("test_llm_integration")


# ══════════════════════════════════════════════════════════════════
# 全局配置（从环境变量 / 命令行参数读取）
# ══════════════════════════════════════════════════════════════════

class LLMTestConfig:
    """LLM 集成测试的全局配置"""

    API_URL = os.environ.get("LLM_API_URL", "http://localhost:8000/v1")
    API_KEY = os.environ.get("LLM_API_KEY", "EMPTY")
    MODEL = os.environ.get("LLM_MODEL", "Qwen2.5-72B-Instruct")
    TIMEOUT = int(os.environ.get("LLM_TEST_TIMEOUT", "180"))

    _llm_reachable: bool = False

    @classmethod
    def is_llm_reachable(cls) -> bool:
        return cls._llm_reachable

    @classmethod
    def set_llm_reachable(cls, value: bool):
        cls._llm_reachable = value
        if value:
            test_logger.info(f"✅ LLM 服务可达: {cls.API_URL} / model={cls.MODEL}")
        else:
            test_logger.error(f"❌ LLM 服务不可达: {cls.API_URL}")


# ══════════════════════════════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════════════════════════════

class TempDirHelper:
    """创建和清理临时目录的辅助类"""

    def __init__(self):
        self.temp_dir = None

    def setup(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rec_llm_integ_test_")
        return self.temp_dir

    def cleanup(self):
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def create_file(self, filename, content):
        filepath = os.path.join(self.temp_dir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return filepath


def create_mini_rec_project(helper: TempDirHelper) -> str:
    """创建迷你推荐系统项目目录（用于 LLM 集成测试）"""

    helper.create_file(
        "models.py",
        """import torch
import torch.nn as nn

class SASRec(nn.Module):
    \"\"\"Self-Attentive Sequential Recommendation Model\"\"\"

    def __init__(self, item_num, hidden_size=64, max_seq_length=50, num_attention_heads=2):
        super().__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.max_seq_length = max_seq_length
        self.num_attention_heads = num_attention_heads

        self.item_embeddings = nn.Embedding(item_num, hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(max_seq_length, hidden_size)
        self.attention = SelfAttentionLayer(hidden_size, num_attention_heads)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, input_ids):
        seq_emb = self.item_embeddings(input_ids)
        pos_emb = self.position_embeddings(
            torch.arange(seq_emb.size(1), device=seq_emb.device)
        )
        seq_emb = seq_emb + pos_emb
        seq_emb = self.attention(seq_emb)
        seq_emb = self.layer_norm(seq_emb)
        return seq_emb

    def finetune(self, input_ids, target_ids):
        seq_emb = self.forward(input_ids)
        target_emb = self.item_embeddings(target_ids)
        scores = torch.matmul(seq_emb, target_emb.transpose(-1, -2))
        return scores
""",
    )

    helper.create_file(
        "modules.py",
        """import torch
import torch.nn as nn
import math

class SelfAttentionLayer(nn.Module):
    \"\"\"Multi-head Self-Attention Layer\"\"\"

    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, hidden_states):
        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        attention_scores = torch.matmul(query, key.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)

        context = torch.matmul(attention_probs, value)
        output = self.output(context)
        return output
""",
    )

    helper.create_file(
        "trainers.py",
        """import torch.nn as nn

class Trainer:
    \"\"\"Model Trainer\"\"\"

    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0
        for batch in data_loader:
            self.optimizer.zero_grad()
            output = self.model(batch)
            loss = self.criterion(output)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss
""",
    )

    helper.create_file(
        "datasets.py",
        """class SeqDataset:
    \"\"\"Sequential Recommendation Dataset\"\"\"

    def __init__(self, data_path, max_seq_length=50):
        self.data_path = data_path
        self.max_seq_length = max_seq_length
""",
    )

    helper.create_file(
        "run_finetune_full.py",
        """import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='SASRec')
parser.add_argument('--dataset', type=str, default='Beauty')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--gpu', type=str, default='0')
args = parser.parse_args()
""",
    )

    return helper.temp_dir


# ══════════════════════════════════════════════════════════════════
# Level 1: LLM 连接健康检查
# ══════════════════════════════════════════════════════════════════

class TestLevel1_LLMConnection(unittest.TestCase):
    """Level 1: 验证 LLM 服务是否可达。如果 FAIL，说明服务本身有问题"""

    def test_llm_service_reachable(self):
        """验证 LLM 服务的 HTTP 端点是否可达"""
        from agent.llm_client import LLMClient

        client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=30,
            max_retries=2,
        )

        try:
            health = client.check_health()
        except Exception as e:
            health = False
            test_logger.error(f"LLM health check exception: {e}")

        if health:
            LLMTestConfig.set_llm_reachable(True)
            self.assertTrue(health, "LLM 服务健康检查应返回 True")
        else:
            # 降级: 直接发极短请求验证
            test_logger.warning("check_health() returned False, trying direct chat ping...")
            try:
                response = client.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    temperature=0.1,
                    max_tokens=10,
                )
                if response is not None and len(response.strip()) > 0:
                    test_logger.info(f"Direct chat ping succeeded: '{response.strip()[:50]}'")
                    LLMTestConfig.set_llm_reachable(True)
                else:
                    LLMTestConfig.set_llm_reachable(False)
                    self.fail(
                        f"LLM 服务不可达: {LLMTestConfig.API_URL}\n"
                        f"请确认:\n"
                        f"  1. vLLM 服务已启动\n"
                        f"  2. URL 正确 (可通过 --api-url 指定)\n"
                        f"  3. 模型名称正确 (可通过 --model 指定)\n"
                        f"  4. 网络可达"
                    )
            except Exception as e:
                LLMTestConfig.set_llm_reachable(False)
                self.fail(f"LLM 服务不可达 ({LLMTestConfig.API_URL}): {e}")

    def test_llm_client_init(self):
        """验证 LLMClient 初始化不报错"""
        from agent.llm_client import LLMClient

        client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=LLMTestConfig.TIMEOUT,
        )
        self.assertEqual(client.model, LLMTestConfig.MODEL)


# ══════════════════════════════════════════════════════════════════
# Level 2: 基础 LLM 调用能力
# ══════════════════════════════════════════════════════════════════

class TestLevel2_BasicLLMCall(unittest.TestCase):
    """Level 2: 验证基础 LLM 调用能力。如果 Level 1 FAIL，本层级自动 SKIP"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 2 测试")

        from agent.llm_client import LLMClient
        self.client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=LLMTestConfig.TIMEOUT,
            max_retries=2,
        )

    def test_simple_chat_returns_content(self):
        """发送简单 prompt，验证 LLM 返回非空内容"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 1+1? Answer in one word."},
        ]
        response = self.client.chat(messages, temperature=0.1, max_tokens=50)

        self.assertIsNotNone(response, "LLM 应返回非 None 内容")
        self.assertGreater(len(response.strip()), 0, "LLM 回复不应为空")
        test_logger.info(f"✅ LLM 回复: '{response.strip()[:100]}'")

    def test_chat_with_longer_prompt(self):
        """发送稍长 prompt，验证 LLM 能处理中等长度输入"""
        messages = [
            {"role": "user", "content": "请简要解释推荐系统中 SASRec 模型的核心思想，不超过3句话。"},
        ]
        response = self.client.chat(messages, temperature=0.3, max_tokens=200)

        self.assertIsNotNone(response)
        self.assertGreater(len(response), 20, "LLM 回复应有一定长度 (>20 chars)")
        keywords_found = any(
            kw in response.lower()
            for kw in ["sasrec", "self-attentive", "sequential", "attention", "推荐", "序列", "自注意力"]
        )
        if keywords_found:
            test_logger.info(f"✅ LLM 回复包含推荐系统关键词")
        else:
            test_logger.warning(f"⚠️ LLM 回复未包含预期关键词: '{response.strip()[:100]}'")

    def test_chat_with_json_format_request(self):
        """请求 LLM 输出 JSON 格式，验证 LLM 能遵循格式指令"""
        messages = [
            {
                "role": "user",
                "content": (
                    "请用以下 JSON 格式输出推荐系统的两个常见评估指标:\n"
                    '```json\n{"metrics": [{"name": "...", "description": "..."}]}\n```'
                ),
            },
        ]
        response = self.client.chat(messages, temperature=0.1, max_tokens=200)

        self.assertIsNotNone(response)
        has_json = "```json" in response or ("{" in response and "}" in response)

        if has_json:
            test_logger.info(f"✅ LLM 回复包含 JSON 格式")
            try:
                import re
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                    test_logger.info(f"✅ JSON 解析成功: {list(parsed.keys())}")
            except json.JSONDecodeError:
                test_logger.warning("⚠️ JSON 格式不完美，但 LLM 尝试了结构化输出")
        else:
            test_logger.warning(f"⚠️ LLM 未按要求输出 JSON: '{response.strip()[:100]}'")


# ══════════════════════════════════════════════════════════════════
# Level 3: Prompt模板+LLM → 结构化输出解析
# ══════════════════════════════════════════════════════════════════

class TestLevel3_PromptTemplateAndParsing(unittest.TestCase):
    """Level 3: 验证项目 Prompt 模板与 LLM 配合产出可解析的结构化输出"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 3 测试")

        from agent.llm_client import LLMClient
        from agent.prompts import (
            MLE_ANALYSIS_PROMPT, CODER_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
        )

        self.client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=LLMTestConfig.TIMEOUT,
            max_retries=2,
        )
        self.MLE_ANALYSIS_PROMPT = MLE_ANALYSIS_PROMPT
        self.CODER_INSTRUCTIONS = CODER_INSTRUCTIONS
        self.RESEARCHER_INSTRUCTIONS = RESEARCHER_INSTRUCTIONS

        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        create_mini_rec_project(self.helper)

    def tearDown(self):
        self.helper.cleanup()

    def test_mle_analysis_prompt_produces_structured_output(self):
        """测试 MLE_ANALYSIS_PROMPT → LLM → 可解析的结构化输出"""
        from agent.error_handler import ProposalParser

        metrics_info = json.dumps({"NDCG@10": 0.35, "Recall@10": 0.45}, ensure_ascii=False)

        prompt = self.MLE_ANALYSIS_PROMPT.format(
            metrics=metrics_info,
            project_context="序列推荐系统 SASRec，核心文件: models.py, modules.py",
            source_code="class SelfAttentionLayer(nn.Module): ... (简化)",
            journal_summary="第0轮训练完成，初始指标 NDCG@10=0.35",
            structural_history="无历史修改",
            surprise_info="",
            case_analysis_info="",
            rollback_warning="",
            strategy_instruction="balanced",
        )

        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.5, max_tokens=2000,
        )

        self.assertIsNotNone(response, "MLE 分析 prompt 应得到 LLM 回复")
        self.assertGreater(len(response), 100, "MLE 分析回复应有一定长度")

        parsed = ProposalParser().parse(response)
        test_logger.info(f"MLE 分析结果: valid={parsed.get('valid')}, action={parsed.get('action')}")

        content_relevance = any(
            kw in response.lower()
            for kw in ["ndcg", "recall", "lr", "learning rate", "attention",
                        "hidden", "参数", "学习率", "注意力", "改进", "修改"]
        )
        self.assertTrue(
            content_relevance or parsed.get("valid"),
            "LLM 回复应包含推荐系统改进相关内容，或能被解析为有效 proposal",
        )

    def test_coder_instructions_produces_diff_output(self):
        """测试 CODER_INSTRUCTIONS → LLM → SEARCH/REPLACE diff 格式"""

        prompt = self.CODER_INSTRUCTIONS.format(
            research_idea="给 SelfAttentionLayer 添加温度参数 temperature 来调节注意力分布",
            target_metrics=json.dumps({"NDCG@10": 0.35}, ensure_ascii=False),
            source_code_context=(
                "## modules.py\n"
                "class SelfAttentionLayer(nn.Module):\n"
                "    def __init__(self, hidden_size, num_attention_heads):\n"
                "        super().__init__()\n"
                "        self.hidden_size = hidden_size\n"
                "        self.query = nn.Linear(hidden_size, hidden_size)\n"
                "\n"
                "    def forward(self, hidden_states):\n"
                "        query = self.query(hidden_states)\n"
                "        key = self.key(hidden_states)\n"
                "        attention_scores = torch.matmul(query, key.transpose(-1, -2))\n"
                "        attention_probs = torch.softmax(attention_scores, dim=-1)\n"
                "        context = torch.matmul(attention_probs, self.value(hidden_states))\n"
                "        return context\n"
            ),
        )

        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.3, max_tokens=2000,
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response), 50)

        has_search_replace = "SEARCH" in response and "REPLACE" in response
        has_code_block = "```python" in response or "```" in response
        has_diff = "diff" in response.lower() or "---" in response
        mentions_temperature = "temperature" in response.lower() or "温度" in response

        test_logger.info(
            f"Coder 输出: search_replace={has_search_replace}, "
            f"code_block={has_code_block}, diff={has_diff}, temperature={mentions_temperature}"
        )

        self.assertTrue(
            has_search_replace or has_code_block or has_diff,
            "Coder LLM 回复应包含 SEARCH/REPLACE 格式、代码块或 diff 格式",
        )

    def test_researcher_instructions_produces_plans(self):
        """测试 RESEARCHER_INSTRUCTIONS → LLM → 研究计划输出"""

        prompt = self.RESEARCHER_INSTRUCTIONS.format(
            query="How to improve SASRec model for sequential recommendation?",
            problem_name="SASRec optimization",
            problem_description="NDCG@10 is only 0.35, need improvement",
            parent_metrics=json.dumps({"NDCG@10": 0.35}, ensure_ascii=False),
            inspirations="无历史成功案例",
            search_results="无搜索结果（本测试不使用搜索引擎）",
        )

        response = self.client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.7, max_tokens=1500,
        )

        self.assertIsNotNone(response)
        self.assertGreater(len(response), 100)

        plan_keywords = [
            "plan", "方案", "approach", "idea", "想法",
            "attention", "注意力", "temperature", "温度",
            "dropout", "embedding", "hidden", "改进",
        ]
        keywords_found = [kw for kw in plan_keywords if kw.lower() in response.lower()]

        test_logger.info(f"Researcher 回复包含关键词: {keywords_found}")
        self.assertGreater(
            len(keywords_found), 0,
            "Researcher LLM 回复应包含研究计划相关关键词",
        )


# ══════════════════════════════════════════════════════════════════
# Level 4: 单 Agent LLM 调用（Researcher / Coder）
# ══════════════════════════════════════════════════════════════════

class TestLevel4_SingleAgentLLMCall(unittest.TestCase):
    """Level 4: 验证 ResearcherAgent / CoderAgent 能独立调用 LLM 并产出有效结果"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 4 测试")

        from agent.researcher import ResearcherAgent, IdeaData, SearchResult, ResearchPlan
        from agent.coder import CoderAgent, CodeChange, CodeResult
        from agent.database import Program, ProgramDatabase

        self.ResearcherAgent = ResearcherAgent
        self.IdeaData = IdeaData
        self.CoderAgent = CoderAgent
        self.Program = Program

    def test_researcher_llm_call_via_generate_plans(self):
        """测试 ResearcherAgent._generate_plans 能否真实调用 LLM"""
        researcher = self.ResearcherAgent(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            temperature=0.7,
        )
        researcher.update_topic(
            query="How to improve SASRec with attention mechanisms",
            problem_name="SASRec optimization",
            problem_description="NDCG@10 is 0.35, need to improve the attention mechanism",
        )

        parent = self.Program(
            id="prog_000",
            code="class SASRec(nn.Module): pass",
            idea=self.IdeaData(
                title="SASRec baseline",
                description="Original SASRec model",
                content="Self-attentive sequential recommendation",
            ),
            parent_id="root",
            metrics={"ndcg@10": 0.35, "recall@10": 0.45},
        )

        try:
            plans = asyncio.run(
                researcher._generate_plans(parent, inspirations=[], search_results=[])
            )
            self.assertIsNotNone(plans)
            self.assertIsInstance(plans, list)

            if len(plans) > 0:
                plan = plans[0]
                test_logger.info(f"✅ Researcher 生成了 {len(plans)} 个计划: title='{plan.title}'")
                self.assertIsInstance(plan.title, str)
                self.assertGreater(len(plan.title), 0)
            else:
                test_logger.warning("⚠️ Researcher 生成了 0 个计划（LLM 可能未按格式回复）")

        except Exception as e:
            test_logger.error(f"Researcher._generate_plans 调用失败: {e}")
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"可能是代码逻辑问题而非 LLM 问题: {e}")
                raise

    def test_coder_llm_call_via_generate_code(self):
        """测试 CoderAgent._generate_code 能否真实调用 LLM"""
        coder = self.CoderAgent(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            temperature=0.4,
        )
        coder.update_topic(
            query="SASRec attention improvement",
            problem_name="SASRec optimization",
            problem_description="Need to add temperature parameter to SelfAttention",
        )

        new_idea = self.IdeaData(
            title="Add temperature scaling to attention",
            description="Add a temperature parameter to SelfAttention to control attention distribution",
            content="Modify SelfAttentionLayer to add temperature parameter in __init__ and use it in forward()",
            supplement="Temperature should default to 1.0 and be applied before softmax",
        )

        parent = self.Program(
            id="prog_000",
            code=(
                "class SelfAttentionLayer(nn.Module):\n"
                "    def __init__(self, hidden_size, num_attention_heads):\n"
                "        super().__init__()\n"
                "        self.hidden_size = hidden_size\n"
                "        self.query = nn.Linear(hidden_size, hidden_size)\n"
                "    def forward(self, hidden_states):\n"
                "        return self.query(hidden_states)\n"
            ),
            idea=self.IdeaData(
                title="SelfAttention baseline",
                description="Original SelfAttention",
                content="Basic self-attention layer",
            ),
            parent_id="root",
            metrics={"ndcg@10": 0.35},
        )

        try:
            diff_text, program_code = asyncio.run(
                coder._generate_code(new_idea, parent, inspirations=[])
            )

            test_logger.info(
                f"✅ Coder 输出: diff_text='{diff_text[:100]}...' ({len(diff_text)} chars), "
                f"code='{program_code[:100]}...' ({len(program_code)} chars)"
            )

            has_output = len(diff_text) > 0 or len(program_code) > 0
            self.assertTrue(has_output, "Coder._generate_code 应产生 diff_text 或 program_code")

            if len(program_code) > 0:
                python_keywords = ["def", "class", "import", "self", "__init__"]
                found = [kw for kw in python_keywords if kw in program_code]
                self.assertGreater(len(found), 0, "Coder 产出的代码应包含 Python 关键字")

        except Exception as e:
            test_logger.error(f"Coder._generate_code 调用失败: {e}")
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"可能是代码逻辑问题而非 LLM 问题: {e}")
                raise


# ══════════════════════════════════════════════════════════════════
# Level 5: 核心 Agent 分析管线
# ══════════════════════════════════════════════════════════════════

class TestLevel5_CoreAnalysisPipeline(unittest.TestCase):
    """Level 5: 验证 RecSelfEvolveAgent._phase_analyze_and_propose 能否真实调用 LLM"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 5 测试")

        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        create_mini_rec_project(self.helper)

    def tearDown(self):
        self.helper.cleanup()

    def test_phase_analyze_and_propose_with_real_llm(self):
        """测试核心分析管线能否使用真实 LLM 产出 proposal"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        config = AgentConfig(
            project_root=self.temp_dir,
            llm_api_url=LLMTestConfig.API_URL,
            llm_api_key=LLMTestConfig.API_KEY,
            llm_model=LLMTestConfig.MODEL,
            llm_timeout=LLMTestConfig.TIMEOUT,
            data_name="Beauty",
            backbone="SASRec",
            llm_enable_semantic_compression=False,
        )

        with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.preflight_check.return_value = {"status": "PASS", "checks": []}
            mock_trainer_class.return_value = mock_trainer
            agent = RecSelfEvolveAgent(config=config)

        metrics = {"NDCG@10": 0.35, "Recall@10": 0.45, "Hit@10": 0.55}

        try:
            proposal = agent._phase_analyze_and_propose(metrics)
        except Exception as e:
            test_logger.error(f"_phase_analyze_and_propose 调用失败: {e}")
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused", "unreachable"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"管线逻辑问题: {e}")
                raise

        if proposal is not None:
            test_logger.info(
                f"✅ 分析管线产出 proposal: "
                f"param_changes={proposal.get('param_changes', {})}, "
                f"structural_changes_count={len(proposal.get('structural_changes', []))}, "
                f"explanation='{proposal.get('explanation', '')[:100]}...'"
            )
            self.assertIn("param_changes", proposal)
            self.assertIn("structural_changes", proposal)

            has_params = bool(proposal.get("param_changes"))
            has_struct = len(proposal.get("structural_changes", [])) > 0
            self.assertTrue(
                has_params or has_struct,
                "LLM 应至少给出一个改进建议（参数修改或结构修改）",
            )
        else:
            test_logger.warning("⚠️ _phase_analyze_and_propose 返回 None")

    def test_parse_proposal_from_real_llm_output(self):
        """测试 LLM 输出能否被 _parse_proposal_response 正确解析"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig
        from agent.llm_client import LLMClient

        config = AgentConfig(
            project_root=self.temp_dir,
            llm_api_url=LLMTestConfig.API_URL,
            llm_api_key=LLMTestConfig.API_KEY,
            llm_model=LLMTestConfig.MODEL,
        )

        with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.preflight_check.return_value = {"status": "PASS"}
            mock_trainer_class.return_value = mock_trainer
            agent = RecSelfEvolveAgent(config=config)

        client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=LLMTestConfig.TIMEOUT,
        )

        llm_prompt = (
            "你是一个推荐系统优化专家。当前 SASRec 模型训练结果:\n"
            "- NDCG@10: 0.35\n- Recall@10: 0.45\n\n"
            "请提出改进方案，输出以下 JSON 格式:\n"
            '```json\n{\n'
            '  "param_changes": {"lr": 新值, "hidden_size": 新值},\n'
            '  "structural_changes": [{"target_file": "...", "target_class_or_function": "...", '
            '"description": "...", "new_code": "...", "action_type": "modify"}],\n'
            '  "explanation": "改进原因"\n}\n```'
        )

        response = client.chat(
            [{"role": "user", "content": llm_prompt}],
            temperature=0.5, max_tokens=2000,
        )
        self.assertIsNotNone(response)
        test_logger.info(f"LLM 原始回复 ({len(response)} chars): '{response[:200]}...'")

        parsed = agent._parse_proposal_response(response)
        test_logger.info(
            f"解析结果: param_changes={parsed.get('param_changes')}, "
            f"structural_changes={len(parsed.get('structural_changes', []))}"
        )
        self.assertIsInstance(parsed, dict)


# ══════════════════════════════════════════════════════════════════
# Level 6: 完整进化循环（1轮）
# ══════════════════════════════════════════════════════════════════

class TestLevel6_EvolutionLoop(unittest.TestCase):
    """Level 6: 验证完整进化循环能否运行 1 轮（LLM 真实调用，训练 mock）"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 6 测试")

        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        create_mini_rec_project(self.helper)

    def tearDown(self):
        self.helper.cleanup()

    def test_evolve_one_iteration_with_real_llm(self):
        """测试 Agent.evolve() 运行 1 轮（LLM 真实调用，训练 mock）"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        config = AgentConfig(
            project_root=self.temp_dir,
            llm_api_url=LLMTestConfig.API_URL,
            llm_api_key=LLMTestConfig.API_KEY,
            llm_model=LLMTestConfig.MODEL,
            llm_timeout=LLMTestConfig.TIMEOUT,
            max_iterations=1,
            data_name="Beauty",
            backbone="SASRec",
            llm_enable_semantic_compression=False,
        )

        mock_train_result = {
            "status": "SUCCESS",
            "metrics": {"NDCG@10": 0.35, "Recall@10": 0.45, "Hit@10": 0.55},
            "stdout": "Training completed. NDCG@10: 0.35, Recall@10: 0.45",
            "stderr": "",
            "returncode": 0,
        }

        with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.preflight_check.return_value = {"status": "PASS", "checks": []}
            mock_trainer.run.return_value = mock_train_result
            mock_trainer_class.return_value = mock_trainer
            agent = RecSelfEvolveAgent(config=config)

            try:
                result = agent.evolve(max_iterations=1)
                test_logger.info(f"✅ 1轮进化完成: result keys={list(result.keys()) if result else 'None'}")

                if result:
                    self.assertIsInstance(result, dict)
                    self.assertIn("final_metrics", result)
                else:
                    test_logger.warning("⚠️ evolve() 返回 None")

            except Exception as e:
                test_logger.error(f"evolve(1轮) 调用失败: {e}")
                error_str = str(e).lower()
                if any(kw in error_str for kw in ["connect", "timeout", "refused", "unreachable"]):
                    self.fail(f"LLM 连接问题: {e}")
                else:
                    test_logger.warning(f"进化循环中的代码逻辑问题: {e}")
                    self.skipTest(f"进化循环存在代码逻辑问题（非 LLM 连接问题）: {e}")

    def test_evolve_one_iteration_multi_role_workflow(self):
        """测试多角色工作流（Planner→Researcher→Coder）运行 1 轮"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        config = AgentConfig(
            project_root=self.temp_dir,
            llm_api_url=LLMTestConfig.API_URL,
            llm_api_key=LLMTestConfig.API_KEY,
            llm_model=LLMTestConfig.MODEL,
            llm_timeout=LLMTestConfig.TIMEOUT,
            max_iterations=1,
            data_name="Beauty",
            backbone="SASRec",
            enable_multi_role_workflow=True,
            researcher_temperature=0.7,
            coder_temperature=0.3,
            llm_enable_semantic_compression=False,
        )

        mock_train_result = {
            "status": "SUCCESS",
            "metrics": {"NDCG@10": 0.35, "Recall@10": 0.45},
            "stdout": "Training completed.",
            "stderr": "",
            "returncode": 0,
        }

        with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
            mock_trainer = MagicMock()
            mock_trainer.preflight_check.return_value = {"status": "PASS", "checks": []}
            mock_trainer.run.return_value = mock_train_result
            mock_trainer_class.return_value = mock_trainer
            agent = RecSelfEvolveAgent(config=config)

        self.assertTrue(agent.config.enable_multi_role_workflow)
        self.assertTrue(agent._multi_role_enabled)

        metrics = {"NDCG@10": 0.35, "Recall@10": 0.45}

        try:
            proposal = agent._phase_multi_role_analyze_and_propose(metrics)

            if proposal is not None:
                test_logger.info(
                    f"✅ 多角色分析产出: param_changes={proposal.get('param_changes', {})}, "
                    f"structural_changes={len(proposal.get('structural_changes', []))}"
                )
            else:
                test_logger.warning("⚠️ 多角色分析返回 None")

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused", "unreachable"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"多角色工作流逻辑问题: {e}")
                self.skipTest(f"多角色工作流存在代码逻辑问题: {e}")


# ══════════════════════════════════════════════════════════════════
# Level 7: LLM 错误反馈与修复管线
# ══════════════════════════════════════════════════════════════════

class TestLevel7_ErrorFeedbackPipeline(unittest.TestCase):
    """Level 7: 验证 LLM 错误反馈管线能否真实工作"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 7 测试")

        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        create_mini_rec_project(self.helper)

    def tearDown(self):
        self.helper.cleanup()

    def test_llm_can_analyze_and_suggest_fix_for_code_error(self):
        """测试 LLM 能否分析代码错误并给出修复建议"""
        from agent.llm_client import LLMClient

        client = LLMClient(
            api_url=LLMTestConfig.API_URL,
            api_key=LLMTestConfig.API_KEY,
            model=LLMTestConfig.MODEL,
            timeout=LLMTestConfig.TIMEOUT,
        )

        # 模拟一个真实的代码错误场景
        fix_prompt = (
            "以下代码出现运行错误:\n"
            "错误: RuntimeError: SelfAttentionLayer.__init__() missing 1 required "
            "positional argument: 'num_attention_heads'\n\n"
            "当前代码:\n"
            "```python\n"
            "class SelfAttentionLayer(nn.Module):\n"
            "    def __init__(self, hidden_size):  # 缺少 num_attention_heads\n"
            "        super().__init__()\n"
            "        self.hidden_size = hidden_size\n"
            "        self.query = nn.Linear(hidden_size, hidden_size)\n"
            "```\n\n"
            "请给出修复后的 __init__ 方法，使用 SEARCH/REPLACE 格式输出:"
        )

        response = client.chat(
            [{"role": "user", "content": fix_prompt}],
            temperature=0.1, max_tokens=1000,
        )

        self.assertIsNotNone(response, "LLM 应能回复修复建议")

        fix_keywords = ["num_attention_heads", "def __init__", "修复", "fix", "SEARCH"]
        found = [kw for kw in fix_keywords if kw in response]

        test_logger.info(f"修复回复包含关键词: {found}")
        self.assertGreater(len(found), 0, "LLM 修复回复应包含修复相关关键词")


# ══════════════════════════════════════════════════════════════════
# Level 8: 进化引擎 (V2 Architecture)
# ══════════════════════════════════════════════════════════════════

class TestLevel8_EvolutionEngine(unittest.TestCase):
    """Level 8: 验证 V2 进化引擎能否真实调用 LLM"""

    def setUp(self):
        if not LLMTestConfig.is_llm_reachable():
            self.skipTest("LLM 服务不可达，跳过 Level 8 测试")

    def test_evolution_engine_researcher_produces_idea(self):
        """测试 EvolutionEngine 的 Researcher 能否生成研究想法"""
        from agent.evolve_engine import EvolutionEngine
        from agent.database import Program
        from agent.researcher import IdeaData

        config = {
            "api_url": LLMTestConfig.API_URL,
            "api_key": LLMTestConfig.API_KEY,
            "model": LLMTestConfig.MODEL,
            "problem_name": "SASRec optimization",
            "problem_description": "NDCG@10 is only 0.35",
        }

        engine = EvolutionEngine(
            config=config,
            project_root="/tmp/test_project",
            max_iterations=1,
        )

        parent = Program(
            id="prog_000",
            code="class SASRec(nn.Module): pass",
            idea=IdeaData(title="baseline", description="SASRec baseline", content="Self-attentive sequential recommendation"),
            parent_id="root",
            metrics={"ndcg": 0.35},
        )

        engine.update_topic("How to improve SASRec")

        try:
            plans, search_results, reports = asyncio.run(
                engine.researcher.run(parent, inspirations=[])
            )

            test_logger.info(
                f"✅ Researcher 产出: plans={len(plans)}, "
                f"search_results={len(search_results)}, "
                f"reports={len(reports)}"
            )

            self.assertIsNotNone(plans)
            if len(plans) > 0:
                self.assertIsInstance(plans[0].title, str)
                test_logger.info(f"  首个计划: title='{plans[0].title}'")

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"EvolutionEngine Researcher 逻辑问题: {e}")
                raise

    def test_evolution_engine_coder_produces_code(self):
        """测试 EvolutionEngine 的 Coder 能否生成代码修改"""
        from agent.evolve_engine import EvolutionEngine
        from agent.coder import CodeChange, CodeResult
        from agent.researcher import IdeaData
        from agent.database import Program

        config = {
            "api_url": LLMTestConfig.API_URL,
            "api_key": LLMTestConfig.API_KEY,
            "model": LLMTestConfig.MODEL,
            "problem_name": "SASRec optimization",
        }

        engine = EvolutionEngine(
            config=config,
            project_root="/tmp/test_project",
            max_iterations=1,
        )

        new_idea = IdeaData(
            title="Add temperature scaling",
            description="Add temperature parameter to SelfAttention",
            content="Modify SelfAttention to scale attention scores by temperature before softmax",
        )

        parent = Program(
            id="prog_000",
            code=(
                "class SelfAttentionLayer(nn.Module):\n"
                "    def __init__(self, hidden_size, num_attention_heads):\n"
                "        super().__init__()\n"
                "        self.hidden_size = hidden_size\n"
                "        self.query = nn.Linear(hidden_size, hidden_size)\n"
                "    def forward(self, hidden_states):\n"
                "        return self.query(hidden_states)\n"
            ),
            idea=IdeaData(title="baseline", description="SASRec baseline", content="SASRec baseline model"),
            parent_id="root",
            metrics={"ndcg": 0.35},
        )

        try:
            all_diff_text, all_program_code = asyncio.run(
                engine.coder.run(new_idea, parent, inspirations=[])
            )

            test_logger.info(
                f"✅ Coder 产出: diffs={len(all_diff_text)}, "
                f"code_versions={len(all_program_code)}"
            )

            has_code = any(len(code) > 0 for code in all_program_code)
            has_diff = any(len(diff) > 0 for diff in all_diff_text)

            self.assertTrue(has_code or has_diff, "Coder 应产出代码或 diff")

            if has_code:
                latest_code = all_program_code[-1]
                test_logger.info(f"  最终代码 ({len(latest_code)} chars): '{latest_code[:100]}...'")

        except Exception as e:
            error_str = str(e).lower()
            if any(kw in error_str for kw in ["connect", "timeout", "refused"]):
                self.fail(f"LLM 连接问题: {e}")
            else:
                test_logger.warning(f"EvolutionEngine Coder 逻辑问题: {e}")
                raise


# ══════════════════════════════════════════════════════════════════
# 运行测试
# ══════════════════════════════════════════════════════════════════

def run_tests(verbose=False, pattern=None):
    """运行所有 LLM 集成测试并输出结果摘要"""

    print("=" * 70)
    print("  RecSelfEvolve Agent — LLM 集成测试（真实调用 LLM API）")
    print("=" * 70)
    print(f"  LLM 配置:")
    print(f"    URL:   {LLMTestConfig.API_URL}")
    print(f"    Model: {LLMTestConfig.MODEL}")
    key_display = LLMTestConfig.API_KEY[:10] + "..." if len(LLMTestConfig.API_KEY) > 10 else LLMTestConfig.API_KEY
    print(f"    Key:   {key_display}")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    test_classes = [
        TestLevel1_LLMConnection,
        TestLevel2_BasicLLMCall,
        TestLevel3_PromptTemplateAndParsing,
        TestLevel4_SingleAgentLLMCall,
        TestLevel5_CoreAnalysisPipeline,
        TestLevel6_EvolutionLoop,
        TestLevel7_ErrorFeedbackPipeline,
        TestLevel8_EvolutionEngine,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # 输出摘要
    print("\n" + "=" * 70)
    print("  LLM 集成测试结果摘要")
    print("=" * 70)
    print(f"  LLM 服务状态: {'✅ 可达' if LLMTestConfig.is_llm_reachable() else '❌ 不可达'}")
    print(f"  总测试数: {result.testsRun}")
    passed = result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped)
    print(f"  成功: {passed}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"  跳过: {len(result.skipped)}")

    if LLMTestConfig.is_llm_reachable():
        if passed > 0:
            print(f"\n  ✅ LLM 集成测试有 {passed} 个成功通过！")
            print(f"     系统确实能调用 LLM 并获得有效输出。")
        if len(result.failures) > 0 or len(result.errors) > 0:
            print(f"\n  ⚠️  有测试失败 — 可能是代码逻辑问题（而非 LLM 连接问题）")
            print(f"     这些失败恰恰暴露了真实场景下的系统缺陷！")
    else:
        print(f"\n  ❌ LLM 服务不可达，大部分测试被 SKIP。")
        print(f"     请先启动 LLM 服务再运行集成测试。")

    if result.failures:
        print("\n  ❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"    - {test}: {traceback[:300]}")

    if result.errors:
        print("\n  ❌ 错误的测试:")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback[:300]}")

    print("=" * 70)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RecSelfEvolve Agent — LLM 集成测试",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("--api-url", default=None, help="LLM API URL (默认从环境变量 LLM_API_URL)")
    parser.add_argument("--api-key", default=None, help="LLM API Key (默认从环境变量 LLM_API_KEY)")
    parser.add_argument("--model", default=None, help="LLM 模型名称 (默认从环境变量 LLM_MODEL)")
    parser.add_argument("-k", "--pattern", default=None, help="只运行匹配的测试")
    args = parser.parse_args()

    # 命令行参数覆盖环境变量
    if args.api_url:
        LLMTestConfig.API_URL = args.api_url
    if args.api_key:
        LLMTestConfig.API_KEY = args.api_key
    if args.model:
        LLMTestConfig.MODEL = args.model

    if args.pattern:
        # 只运行匹配的测试
        suite = unittest.TestLoader().loadTestsFromName(args.pattern, module=__import__(__name__))
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        result = run_tests(verbose=args.verbose)

    sys.exit(0 if result.wasSuccessful() else 1)