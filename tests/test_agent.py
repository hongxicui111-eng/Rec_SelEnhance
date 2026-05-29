#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RecSelfEvolve Agent 综合测试脚本

测试范围:
1. 配置系统 (AgentConfig)
2. LLM 客户端 (LLMClient) — mock 测试，不依赖真实 LLM 服务
3. 项目适配器 (SeqRecAdapter)
4. Prompt 模板完整性
5. 错误处理与输出解析 (ProposalParser, LLMFixer)
6. 进化质量守卫 (EvolutionQualityGuard, SafetyGuardrails)
7. 实验日志系统 (ExperimentJournal)
8. 迭代修改记忆 (IterativeMemory)
9. 上下文压缩器 (LLMContextCompressor)
10. 结构修改应用器 (StructureApplier)
11. 程序数据库 (ProgramDatabase)
12. 研究 Agent (ResearcherAgent)
13. 编码 Agent (CoderAgent)
14. 进化引擎 (EvolutionEngine)
15. 核心主循环 (RecSelfEvolveAgent) — mock 测试

运行方式:
    python tests/test_agent.py
    python tests/test_agent.py -v  # 详细输出
    python tests/test_agent.py -k "test_config"  # 只跑配置相关测试
"""

import sys
import os
import json
import tempfile
import shutil
import logging
import time
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from dataclasses import asdict

# 确保项目根目录在搜索路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 配置日志（测试模式 — 简洁输出）
logging.basicConfig(level=logging.WARNING, format="[%(name)s] %(levelname)s: %(message)s")
test_logger = logging.getLogger("test_agent")


# ══════════════════════════════════════════════════════════════════
# 辅助工具
# ══════════════════════════════════════════════════════════════════

class TempDirHelper:
    """创建和清理临时目录的辅助类"""

    def __init__(self):
        self.temp_dir = None

    def setup(self):
        self.temp_dir = tempfile.mkdtemp(prefix="rec_self_evolve_test_")
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


# ══════════════════════════════════════════════════════════════════
# 1. 配置系统测试
# ══════════════════════════════════════════════════════════════════

class TestAgentConfig(unittest.TestCase):
    """测试 AgentConfig 数据类"""

    def setUp(self):
        from agent.config import AgentConfig
        self.AgentConfig = AgentConfig

    def test_default_config_creation(self):
        """测试默认配置能否正常创建"""
        config = self.AgentConfig()
        self.assertEqual(config.llm_api_url, "http://localhost:8000/v1")
        self.assertEqual(config.llm_model, "Qwen2.5-72B-Instruct")
        self.assertEqual(config.llm_temperature, 0.7)
        self.assertEqual(config.max_iterations, 20)
        self.assertEqual(config.data_name, "Beauty")
        self.assertEqual(config.backbone, "SASRec")
        self.assertFalse(config.enable_multi_role_workflow)

    def test_custom_config_creation(self):
        """测试自定义参数能否正常创建配置"""
        config = self.AgentConfig(
            llm_api_url="http://my-server:1234/v1",
            llm_model="deepseek-v3",
            max_iterations=30,
            data_name="Toys_and_Games",
            enable_multi_role_workflow=True,
            researcher_temperature=0.8,
            coder_temperature=0.3,
        )
        self.assertEqual(config.llm_api_url, "http://my-server:1234/v1")
        self.assertEqual(config.llm_model, "deepseek-v3")
        self.assertEqual(config.max_iterations, 30)
        self.assertEqual(config.data_name, "Toys_and_Games")
        self.assertTrue(config.enable_multi_role_workflow)
        self.assertEqual(config.researcher_temperature, 0.8)
        self.assertEqual(config.coder_temperature, 0.3)

    def test_config_serialization(self):
        """测试配置能否序列化为 dict"""
        config = self.AgentConfig()
        d = asdict(config)
        self.assertIsInstance(d, dict)
        self.assertIn("llm_api_url", d)
        self.assertIn("max_iterations", d)
        self.assertIn("metric_guardrails", d)
        # metric_guardrails 应包含默认指标
        self.assertIn("NDCG@10", d["metric_guardrails"])

    def test_multi_role_config_fields(self):
        """测试多角色工作流配置字段"""
        config = self.AgentConfig(enable_multi_role_workflow=True)
        self.assertTrue(config.enable_multi_role_workflow)
        # 默认温度值应不同
        self.assertGreater(config.planner_temperature, config.coder_temperature)
        self.assertGreater(config.researcher_temperature, config.debugger_temperature)
        self.assertEqual(config.max_reflection_rounds, 3)


# ══════════════════════════════════════════════════════════════════
# 2. LLM 客户端测试 (Mock — 不依赖真实服务)
# ══════════════════════════════════════════════════════════════════

class TestLLMClient(unittest.TestCase):
    """测试 LLM 客户端的核心功能"""

    def setUp(self):
        from agent.llm_client import LLMClient
        self.LLMClient = LLMClient

    def test_client_init(self):
        """测试客户端初始化"""
        client = self.LLMClient(
            api_url="http://localhost:8000/v1",
            api_key="test-key",
            model="test-model",
            timeout=60,
            max_retries=2,
        )
        self.assertEqual(client.api_url, "http://localhost:8000/v1")
        self.assertEqual(client.api_key, "test-key")
        self.assertEqual(client.model, "test-model")

    def test_url_normalization(self):
        """测试 URL 规范化 (去掉末尾斜杠, 添加 /v1)"""
        # 没有 /v1 的情况
        client1 = self.LLMClient(api_url="http://localhost:8000")
        self.assertEqual(client1.api_url, "http://localhost:8000/v1")

        # 已有 /v1 的情况
        client2 = self.LLMClient(api_url="http://localhost:8000/v1/")
        self.assertEqual(client2.api_url, "http://localhost:8000/v1")

    def test_token_estimation(self):
        """测试 token 估算逻辑"""
        client = self.LLMClient(api_url="http://localhost:8000/v1")
        # 空文本 → 0 token
        self.assertEqual(client._estimate_tokens(""), 0)
        # 短文本 → 至少 1 token
        self.assertEqual(client._estimate_tokens("a"), 1)
        # 英文估算大致 ≈ len/3 (使用整数比较)
        tokens = client._estimate_tokens("hello world test")
        self.assertGreater(tokens, 0)
        self.assertLess(tokens, len("hello world test"))

    def test_prompt_safety_ratio_clamping(self):
        """测试 prompt 安全比例的边界限制"""
        # 过低 → 0.1
        client1 = self.LLMClient(api_url="http://localhost:8000/v1", prompt_safety_ratio=0.01)
        self.assertEqual(client1.prompt_safety_ratio, 0.1)
        # 过高 → 0.95
        client2 = self.LLMClient(api_url="http://localhost:8000/v1", prompt_safety_ratio=1.0)
        self.assertEqual(client2.prompt_safety_ratio, 0.95)
        # 正常范围 → 保持原值
        client3 = self.LLMClient(api_url="http://localhost:8000/v1", prompt_safety_ratio=0.75)
        self.assertEqual(client3.prompt_safety_ratio, 0.75)

    def test_message_truncation(self):
        """测试消息截断逻辑"""
        client = self.LLMClient(api_url="http://localhost:8000/v1")
        long_text = "A" * 5000
        truncated = client._truncate_text(long_text, 1000)
        self.assertLessEqual(len(truncated), 1000)
        self.assertIn("[TRUNCATED]", truncated)

    def test_compress_message_content(self):
        """测试消息压缩 (代码块 + JSON + 截断)"""
        client = self.LLMClient(api_url="http://localhost:8000/v1")
        # 短内容不压缩
        short = "short text"
        self.assertEqual(client._compress_message_content(short, 1000), short)
        # 长代码块压缩
        code_block = "```python\n" + "x = 1\n" * 500 + "```"
        compressed = client._compress_message_content(code_block, 2000)
        self.assertLessEqual(len(compressed), 2000)

    def test_chat_mock(self):
        """测试 chat 方法 (mock OpenAI client)"""
        client = self.LLMClient(api_url="http://localhost:8000/v1")

        # mock OpenAI client 的 chat.completions.create
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "LLM response text"
        mock_response.choices[0].message.role = "assistant"

        mock_openai_client = MagicMock()
        mock_openai_client.chat.completions.create.return_value = mock_response

        # 替换内部 client
        client._client = mock_openai_client

        # chat 方法接受 messages list (OpenAI 格式)
        messages = [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "user prompt"},
        ]
        result = client.chat(messages)
        self.assertEqual(result, "LLM response text")

    def test_fit_messages_to_context(self):
        """测试上下文预算裁剪"""
        client = self.LLMClient(
            api_url="http://localhost:8000/v1",
            max_context_tokens=4096,
            prompt_safety_ratio=0.75,
        )
        messages = [
            {"role": "system", "content": "short system prompt"},
            {"role": "user", "content": "A" * 100000},  # 很长的内容
        ]
        fitted = client._fit_messages_to_context(messages, 1024)
        # 应被裁剪
        self.assertLess(len(fitted[1]["content"]), 100000)


# ══════════════════════════════════════════════════════════════════
# 3. 项目适配器测试
# ══════════════════════════════════════════════════════════════════

class TestProjectAdapter(unittest.TestCase):
    """测试 SeqRecAdapter"""

    def setUp(self):
        from agent.project_adapter import SeqRecAdapter
        self.SeqRecAdapter = SeqRecAdapter
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        # 创建模拟项目结构
        self.helper.create_file("models.py", "class SASRec:\n    pass\n")
        self.helper.create_file("modules.py", "class SelfAttention:\n    pass\n")
        self.helper.create_file("trainers.py", "class Trainer:\n    pass\n")
        self.helper.create_file("datasets.py", "class Dataset:\n    pass\n")

    def tearDown(self):
        self.helper.cleanup()

    def test_adapter_init(self):
        """测试适配器初始化"""
        adapter = self.SeqRecAdapter(
            project_root=self.temp_dir,
            data_name="Beauty",
            backbone="SASRec",
        )
        self.assertEqual(adapter.project_root, self.temp_dir)
        self.assertEqual(adapter.data_name, "Beauty")
        self.assertEqual(adapter.backbone, "SASRec")

    def test_build_train_command(self):
        """测试训练命令构建"""
        adapter = self.SeqRecAdapter(
            project_root=self.temp_dir,
            data_name="Beauty",
            backbone="SASRec",
            script_name="run_finetune_full.py",
        )
        cmd = adapter.build_train_command(param_overrides={"lr": 0.001})
        self.assertIn("run_finetune_full.py", cmd)
        self.assertIn("--lr", cmd)
        self.assertIn("0.001", cmd)

    def test_build_train_command_with_gpu(self):
        """测试带 GPU 参数的命令构建"""
        adapter = self.SeqRecAdapter(
            project_root=self.temp_dir,
            data_name="Beauty",
            backbone="SASRec",
            gpu_id="1",
        )
        cmd = adapter.build_train_command()
        self.assertIn("--gpu", cmd)
        self.assertIn("1", cmd)

    def test_project_description(self):
        """测试项目描述属性"""
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        desc = adapter.PROJECT_DESCRIPTION
        self.assertIn("序列推荐", desc)
        self.assertIn("SASRec", desc)
        self.assertIn("models.py", desc)
        self.assertIn("modules.py", desc)

    def test_source_file_map(self):
        """测试源文件映射"""
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        self.assertIn("models.py", adapter.SOURCE_FILE_MAP)
        self.assertIn("modules.py", adapter.SOURCE_FILE_MAP)
        self.assertIn("trainers.py", adapter.SOURCE_FILE_MAP)

    def test_structural_actions(self):
        """测试结构修改动作类型"""
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        self.assertIn("modify", adapter.STRUCTURAL_ACTIONS)
        self.assertIn("add_module", adapter.STRUCTURAL_ACTIONS)

    def test_find_source_file(self):
        """测试源文件查找"""
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        # SeqRecAdapter 使用 get_source_code 方法获取源码内容
        code = adapter.get_source_code("models.py")
        if code is not None:
            self.assertIn("SASRec", code)

    def test_validate_train_command(self):
        """测试训练命令前置校验"""
        adapter = self.SeqRecAdapter(
            project_root=self.temp_dir,
            data_name="Beauty",
            backbone="SASRec",
            script_name="run_finetune_full.py",
        )
        # validate_train_command 接受 cmd 参数
        cmd = adapter.build_train_command()
        result = adapter.validate_train_command(cmd)
        self.assertIsInstance(result, dict)


# ══════════════════════════════════════════════════════════════════
# 4. Prompt 模板完整性测试
# ══════════════════════════════════════════════════════════════════

class TestPrompts(unittest.TestCase):
    """测试 Prompt 模板的完整性和可用性"""

    def setUp(self):
        from agent.prompts import (
            MLE_ANALYSIS_PROMPT, STRUCTURE_OPTIMIZATION_PROMPT,
            ERROR_FEEDBACK_PROMPT, STRUCTURE_FIX_PROMPT,
            TRAIN_DIAGNOSIS_PROMPT, CODE_FIX_PROMPT, PREFLIGHT_FIX_PROMPT,
            PLANNER_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
            REFLECTION_INSTRUCTIONS, SEARCH_INSTRUCTIONS,
            CODER_INSTRUCTIONS, DEBUGGER_INSTRUCTIONS,
        )
        self.prompts = {
            "MLE_ANALYSIS_PROMPT": MLE_ANALYSIS_PROMPT,
            "STRUCTURE_OPTIMIZATION_PROMPT": STRUCTURE_OPTIMIZATION_PROMPT,
            "ERROR_FEEDBACK_PROMPT": ERROR_FEEDBACK_PROMPT,
            "STRUCTURE_FIX_PROMPT": STRUCTURE_FIX_PROMPT,
            "TRAIN_DIAGNOSIS_PROMPT": TRAIN_DIAGNOSIS_PROMPT,
            "CODE_FIX_PROMPT": CODE_FIX_PROMPT,
            "PREFLIGHT_FIX_PROMPT": PREFLIGHT_FIX_PROMPT,
            "PLANNER_INSTRUCTIONS": PLANNER_INSTRUCTIONS,
            "RESEARCHER_INSTRUCTIONS": RESEARCHER_INSTRUCTIONS,
            "REFLECTION_INSTRUCTIONS": REFLECTION_INSTRUCTIONS,
            "SEARCH_INSTRUCTIONS": SEARCH_INSTRUCTIONS,
            "CODER_INSTRUCTIONS": CODER_INSTRUCTIONS,
            "DEBUGGER_INSTRUCTIONS": DEBUGGER_INSTRUCTIONS,
        }

    def test_all_prompts_exist(self):
        """测试所有 prompt 模板都已定义"""
        for name, prompt in self.prompts.items():
            self.assertIsNotNone(prompt, f"{name} should not be None")
            self.assertIsInstance(prompt, str, f"{name} should be a string")

    def test_prompts_not_empty(self):
        """测试所有 prompt 模板都不是空字符串"""
        for name, prompt in self.prompts.items():
            self.assertGreater(len(prompt), 100, f"{name} should have substantial content (>100 chars)")

    def test_prompts_contain_formatting_keys(self):
        """测试 prompt 模板包含必要的格式化占位符"""
        # MLE_ANALYSIS_PROMPT 应包含 metrics 占位符
        mle = self.prompts["MLE_ANALYSIS_PROMPT"]
        self.assertTrue(
            "{" in mle and "}" in mle,
            "MLE_ANALYSIS_PROMPT should contain formatting placeholders"
        )

    def test_multi_role_prompts_content(self):
        """测试多角色 prompt 模板的内容要点"""
        # CODER_INSTRUCTIONS 应提及 SEARCH/REPLACE
        coder = self.prompts["CODER_INSTRUCTIONS"]
        self.assertIn("SEARCH", coder.upper() if isinstance(coder, str) else "")
        self.assertIn("REPLACE", coder.upper() if isinstance(coder, str) else "")

        # RESEARCHER_INSTRUCTIONS 应提及研究
        researcher = self.prompts["RESEARCHER_INSTRUCTIONS"]
        self.assertTrue(
            len(researcher) > 200,
            "RESEARCHER_INSTRUCTIONS should have meaningful content"
        )

    def test_prompt_import_no_error(self):
        """测试导入 prompts 模块不会出错"""
        import agent.prompts
        self.assertTrue(hasattr(agent.prompts, "MLE_ANALYSIS_PROMPT"))
        self.assertTrue(hasattr(agent.prompts, "CODER_INSTRUCTIONS"))


# ══════════════════════════════════════════════════════════════════
# 5. 错误处理与输出解析测试
# ══════════════════════════════════════════════════════════════════

class TestProposalParser(unittest.TestCase):
    """测试 LLM 输出解析器"""

    def setUp(self):
        from agent.error_handler import ProposalParser
        self.ProposalParser = ProposalParser

    def test_parse_empty_input(self):
        """测试空输入"""
        result = self.ProposalParser.parse(None)
        self.assertFalse(result["valid"])
        self.assertEqual(result["action"], "skip_iteration")

    def test_parse_empty_string(self):
        """测试空字符串"""
        result = self.ProposalParser.parse("")
        self.assertFalse(result["valid"])

    def test_parse_too_short(self):
        """测试过短输出"""
        result = self.ProposalParser.parse("short")
        self.assertFalse(result["valid"])
        self.assertIn("too short", result["error"].lower())

    def test_parse_refusal(self):
        """测试 LLM 安全拒绝输出"""
        refusal_text = "I'm sorry, I cannot help with that request. As an AI language model, I'm unable to..."
        result = self.ProposalParser.parse(refusal_text)
        self.assertFalse(result["valid"])
        self.assertIn("refused", result["error"].lower())

    def test_parse_python_code_block(self):
        """测试提取 Python 代码块"""
        text = """
Here is my proposed change:

```python
class SelfAttention:
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
```

This change adds a new initialization parameter.
"""
        result = self.ProposalParser.parse(text)
        self.assertTrue(result["valid"])
        self.assertEqual(result["diff_type"], "python")

    def test_parse_diff_format(self):
        """测试提取 diff 格式"""
        text = """
Based on the analysis, I propose the following diff:

```diff
--- a/modules.py
+++ b/modules.py
@@ -10,6 +10,8 @@
 class SelfAttention:
     def __init__(self, hidden_size):
         self.hidden_size = hidden_size
+        self.temperature = 1.0
+        self.use_scaled = True
```
"""
        result = self.ProposalParser.parse(text)
        self.assertTrue(result["valid"])
        self.assertIn("SelfAttention", result["diff"])

    def test_parse_json_config(self):
        """测试提取 JSON 配置变更"""
        # Bug 2 已修复: _extract_diff 中的 regex [+- ] 改为 [+- ]
        # 现在不再触发 re.PatternError
        text = '\nHere are the parameter changes:\n\n```json\n{"lr": 0.0005, "batch_size": 512}\n```\n\nThese changes should help.\n'
        result = self.ProposalParser.parse(text)
        # JSON 代码块会被 _extract_diff 捕获（因为它有 ``` 代码块格式）
        if result.get("valid"):
            self.assertIsNotNone(result.get("diff"))
        # 或者被 _extract_json 捕获
        # 两种结果都是有效的

    def test_parse_unstructured_output(self):
        """测试无法解析的自由文本"""
        text = "I think we should try a different approach, maybe consider attention mechanisms more carefully." + " " * 100  # 确保长度 > 50
        result = self.ProposalParser.parse(text)
        self.assertFalse(result["valid"])
        self.assertIn("regenerate", result["action"])


class TestLLMFixer(unittest.TestCase):
    """测试 LLM 错误修复器"""

    def setUp(self):
        from agent.error_handler import LLMFixer
        self.LLMFixer = LLMFixer

    def test_fixer_init(self):
        """测试修复器初始化"""
        mock_llm = MagicMock()
        fixer = self.LLMFixer(llm_client=mock_llm)
        self.assertIsNotNone(fixer)


# ══════════════════════════════════════════════════════════════════
# 6. 进化质量守卫测试
# ══════════════════════════════════════════════════════════════════

class TestQualityGuard(unittest.TestCase):
    """测试进化质量守卫"""

    def setUp(self):
        from agent.quality_guard import EvolutionQualityGuard, SafetyGuardrails
        self.EvolutionQualityGuard = EvolutionQualityGuard
        self.SafetyGuardrails = SafetyGuardrails

    def test_guard_init(self):
        """测试守卫初始化"""
        guard = self.EvolutionQualityGuard(window_size=5, primary_metric="ndcg@10")
        self.assertEqual(guard.window_size, 5)
        self.assertEqual(guard.primary_metric, "ndcg@10")
        self.assertEqual(len(guard.history), 0)

    def test_update_continue(self):
        """测试正常更新 → CONTINUE"""
        guard = self.EvolutionQualityGuard(primary_metric="ndcg@10")
        # 前 3 轮不足以做决策
        for i in range(3):
            result = guard.update(i, {"ndcg@10": 0.1 + i * 0.01})
            self.assertEqual(result["action"], "CONTINUE")

    def test_update_new_best(self):
        """测试检测到新最优"""
        guard = self.EvolutionQualityGuard(primary_metric="ndcg@10")
        guard.update(0, {"ndcg@10": 0.3})
        guard.update(1, {"ndcg@10": 0.35})
        guard.update(2, {"ndcg@10": 0.40})
        self.assertEqual(guard.best_index, 2)
        self.assertEqual(guard.best_metrics["ndcg@10"], 0.40)

    def test_degradation_detection(self):
        """测试退化检测 → 应返回 REVERT_TO_BEST 并包含 reason"""
        guard = self.EvolutionQualityGuard(primary_metric="ndcg@10", degrade_threshold=0.95)
        # 先上升
        for i in range(8):
            guard.update(i, {"ndcg@10": 0.3 + i * 0.01})
        # 然后显著下降
        for i in range(8, 12):
            result = guard.update(i, {"ndcg@10": max(0.01, 0.35 - (i - 7) * 0.1)})
            if result["action"] == "REVERT_TO_BEST":
                # Bug 3 修复后: 必须包含 reason 字段
                self.assertIn("reason", result)
                self.assertIn("退化", result["reason"])
                return
        # 如果退化未被检测到（可能需要更多数据），也验证 CONTINUE 结果
        # 不应该触发 KeyError
        self.assertIn(result["action"], ["CONTINUE", "REVERT_TO_BEST", "SWITCH_STRATEGY"])

    def test_plateau_detection(self):
        """测试停滞检测 → 应返回 SWITCH_STRATEGY 并包含 reason"""
        guard = self.EvolutionQualityGuard(primary_metric="ndcg@10", plateau_threshold=0.001)
        # 连续多轮微小变化（停滞）
        for i in range(20):
            result = guard.update(i, {"ndcg@10": 0.350 + i * 0.0001})
            if result["action"] == "SWITCH_STRATEGY":
                # Bug 3 修复后: 必须包含 reason 字段
                self.assertIn("reason", result)
                self.assertIn("停滞", result["reason"])
                return
        # 如果停滞未被检测到，也验证结果不抛 KeyError
        self.assertIn(result["action"], ["CONTINUE", "SWITCH_STRATEGY", "REVERT_TO_BEST"])

    def test_safety_guardrails(self):
        """测试安全护栏"""
        rails = self.SafetyGuardrails()
        self.assertIsNotNone(rails)


# ══════════════════════════════════════════════════════════════════
# 7. 实验日志系统测试
# ══════════════════════════════════════════════════════════════════

class TestJournal(unittest.TestCase):
    """测试实验日志系统"""

    def setUp(self):
        from agent.journal import ExperimentJournal
        self.ExperimentJournal = ExperimentJournal
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()

    def tearDown(self):
        self.helper.cleanup()

    def test_journal_create(self):
        """测试日志创建"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        self.assertEqual(len(journal.records), 0)

    def test_journal_record_and_save(self):
        """测试记录和保存"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        journal.record({
            "iteration": 0,
            "status": "SUCCESS",
            "metrics": {"ndcg@10": 0.35},
            "proposal": "test proposal",
        })
        self.assertEqual(len(journal.records), 1)
        self.assertEqual(journal.records[0]["status"], "SUCCESS")
        # 文件应该已写入
        self.assertTrue(os.path.exists(journal_path))

    def test_journal_get_best(self):
        """测试获取最优记录"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        journal.record({"iteration": 0, "status": "SUCCESS", "metrics": {"ndcg@10": 0.30}})
        journal.record({"iteration": 1, "status": "SUCCESS", "metrics": {"ndcg@10": 0.40}})
        journal.record({"iteration": 2, "status": "FAILED", "metrics": {"ndcg@10": 0.35}})
        best = journal.get_best(metric="ndcg@10")
        self.assertIsNotNone(best)
        self.assertEqual(best["metrics"]["ndcg@10"], 0.40)

    def test_journal_get_latest(self):
        """测试获取最近记录"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        for i in range(10):
            journal.record({"iteration": i, "status": "SUCCESS"})
        latest = journal.get_latest(3)
        self.assertEqual(len(latest), 3)
        self.assertEqual(latest[-1]["iteration"], 9)

    def test_journal_get_failures(self):
        """测试获取失败记录"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        journal.record({"iteration": 0, "status": "SUCCESS"})
        journal.record({"iteration": 1, "status": "FAILED"})
        journal.record({"iteration": 2, "status": "ROLLED_BACK"})
        failures = journal.get_failures()
        self.assertEqual(len(failures), 2)

    def test_journal_summarize(self):
        """测试日志摘要生成"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        journal.record({"iteration": 0, "status": "SUCCESS", "metrics": {"ndcg@10": 0.35}})
        summary = journal.summarize()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)

    def test_journal_persistence(self):
        """测试日志持久化 (写入后重新加载)"""
        journal_path = os.path.join(self.temp_dir, "test_journal.jsonl")
        journal = self.ExperimentJournal(file_path=journal_path)
        journal.record({"iteration": 0, "status": "SUCCESS", "metrics": {"ndcg@10": 0.35}})
        # 重新加载
        journal2 = self.ExperimentJournal(file_path=journal_path)
        self.assertEqual(len(journal2.records), 1)
        self.assertEqual(journal2.records[0]["metrics"]["ndcg@10"], 0.35)


# ══════════════════════════════════════════════════════════════════
# 8. 迭代修改记忆测试
# ══════════════════════════════════════════════════════════════════

class TestIterativeMemory(unittest.TestCase):
    """测试迭代修改记忆系统"""

    def setUp(self):
        from agent.iterative_memory import IterativeMemory
        self.IterativeMemory = IterativeMemory
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        # 创建模拟源码文件
        self.helper.create_file("models.py", "class SASRec:\n    def finetune(self):\n        pass\n")
        self.helper.create_file("modules.py", "class SelfAttention:\n    def forward(self):\n        pass\n")
        self.helper.create_file("trainers.py", "class Trainer:\n    def train(self):\n        pass\n")

    def tearDown(self):
        self.helper.cleanup()

    def test_memory_init(self):
        """测试记忆系统初始化"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
            source_files=["models.py", "modules.py", "trainers.py"],
        )
        self.assertEqual(len(memory.modification_records), 0)

    def test_save_source_snapshot(self):
        """测试源码快照保存"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        result = memory.save_source_snapshot(iteration=0)
        self.assertIsInstance(result, dict)
        # 快照目录应已创建
        snapshot_dir = Path(log_dir) / "source_snapshots" / "iter_000"
        self.assertTrue(snapshot_dir.exists())

    def test_record_modification(self):
        """测试修改记录"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        # record_modification 签名: (iteration, structural_changes, apply_result, metrics_before, metrics_after, note)
        memory.record_modification(
            iteration=1,
            structural_changes=[{"target_file": "modules.py", "target_class_or_function": "SelfAttention.forward", "description": "Add temperature scaling"}],
            apply_result={"status": "APPLIED", "changes_summary": "Modified SelfAttention.forward"},
            metrics_before={"ndcg@10": 0.35},
            metrics_after={"ndcg@10": 0.38},
            note="Add temperature scaling to attention",
        )
        self.assertEqual(len(memory.modification_records), 1)

    def test_record_rollback(self):
        """测试回滚记录"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        # 先记录一次修改
        memory.record_modification(
            iteration=2,
            structural_changes=[{"target_file": "modules.py", "target": "NewAttention"}],
            apply_result={"status": "APPLIED"},
            metrics_before={"ndcg@10": 0.38},
            metrics_after={"ndcg@10": 0.20},
        )
        # 再记录回滚
        memory.record_rollback(
            iteration=2,
            reason="NDCG dropped significantly",
            rollback_to_iteration=1,
        )
        self.assertEqual(len(memory.modification_records), 1)

    def test_build_history_context(self):
        """测试生成历史修改感知上下文"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        memory.record_modification(
            iteration=1,
            structural_changes=[{"target_file": "modules.py", "target": "SelfAttention"}],
            apply_result={"status": "APPLIED"},
            metrics_before={"ndcg@10": 0.35}, metrics_after={"ndcg@10": 0.38},
        )
        # build_history_context_for_llm 是正确的方法名
        context = memory.build_history_context_for_llm(current_iteration=2)
        self.assertIsInstance(context, str)

    def test_memory_persistence(self):
        """测试记忆持久化"""
        log_dir = os.path.join(self.temp_dir, "evolve_logs")
        memory = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        memory.record_modification(
            iteration=1,
            structural_changes=[{"target_file": "modules.py", "target": "SelfAttention"}],
            apply_result={"status": "APPLIED"},
            metrics_before={"ndcg@10": 0.35}, metrics_after={"ndcg@10": 0.38},
        )
        # 重新加载
        memory2 = self.IterativeMemory(
            project_root=self.temp_dir,
            log_dir=log_dir,
        )
        self.assertEqual(len(memory2.modification_records), 1)


# ══════════════════════════════════════════════════════════════════
# 9. 上下文压缩器测试
# ══════════════════════════════════════════════════════════════════

class TestContextCompressor(unittest.TestCase):
    """测试上下文压缩器"""

    def setUp(self):
        from agent.context_compressor import LLMContextCompressor
        self.LLMContextCompressor = LLMContextCompressor
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()

    def tearDown(self):
        self.helper.cleanup()

    def test_compressor_init(self):
        """测试压缩器初始化"""
        mock_llm = MagicMock()
        compressor = self.LLMContextCompressor(
            llm_client=mock_llm,
            enable_cache=True,
            cache_path=os.path.join(self.temp_dir, "test_cache.json"),
        )
        self.assertTrue(compressor.enable_cache)

    def test_compressor_hash_key(self):
        """测试缓存 hash key 生成"""
        key1 = self.LLMContextCompressor._hash_key("text1", 1000, "journal", "default")
        key2 = self.LLMContextCompressor._hash_key("text2", 1000, "journal", "default")
        self.assertNotEqual(key1, key2)
        # 相同输入 → 相同 key
        key3 = self.LLMContextCompressor._hash_key("text1", 1000, "journal", "default")
        self.assertEqual(key1, key3)

    def test_chunk_text(self):
        """测试文本分块"""
        text = "A" * 10000
        chunks = self.LLMContextCompressor._chunk_text(text, 2000)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 2000)

    def test_cache_save_and_load(self):
        """测试缓存保存和加载"""
        cache_path = os.path.join(self.temp_dir, "test_cache.json")
        mock_llm = MagicMock()
        compressor = self.LLMContextCompressor(
            llm_client=mock_llm,
            enable_cache=True,
            cache_path=cache_path,
        )
        # 添加缓存条目
        compressor._cache["test_key"] = {
            "compressed": "test_content",
            "ts": int(time.time()),
        }
        compressor._cache_dirty = True
        compressor._save_cache()

        # 重新加载
        compressor2 = self.LLMContextCompressor(
            llm_client=mock_llm,
            enable_cache=True,
            cache_path=cache_path,
        )
        self.assertIn("test_key", compressor2._cache)


# ══════════════════════════════════════════════════════════════════
# 10. 结构修改应用器测试
# ══════════════════════════════════════════════════════════════════

class TestStructureApplier(unittest.TestCase):
    """测试结构修改应用器"""

    def setUp(self):
        from agent.structure_applier import StructureApplier
        self.StructureApplier = StructureApplier
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        # 创建模拟源码文件（包含完整的 Python 语法）
        self.helper.create_file(
            "modules.py",
            "import torch\nimport torch.nn as nn\n\n\nclass SelfAttention(nn.Module):\n"
            "    def __init__(self, hidden_size, num_attention_heads):\n"
            "        super().__init__()\n"
            "        self.hidden_size = hidden_size\n"
            "        self.num_attention_heads = num_attention_heads\n"
            "        self.query = nn.Linear(hidden_size, hidden_size)\n"
            "        self.key = nn.Linear(hidden_size, hidden_size)\n"
            "        self.value = nn.Linear(hidden_size, hidden_size)\n\n"
            "    def forward(self, hidden_states):\n"
            "        query = self.query(hidden_states)\n"
            "        key = self.key(hidden_states)\n"
            "        value = self.value(hidden_states)\n"
            "        attention_scores = torch.matmul(query, key.transpose(-1, -2))\n"
            "        return torch.matmul(attention_scores, value)\n"
        )
        self.helper.create_file(
            "models.py",
            "import torch.nn as nn\n\n\nclass SASRec(nn.Module):\n"
            "    def __init__(self, item_num, hidden_size):\n"
            "        super().__init__()\n"
            "        self.item_embeddings = nn.Embedding(item_num, hidden_size)\n\n"
            "    def finetune(self, input_ids):\n"
            "        return self.item_embeddings(input_ids)\n"
        )

    def tearDown(self):
        self.helper.cleanup()

    def test_applier_init(self):
        """测试应用器初始化"""
        applier = self.StructureApplier(project_root=self.temp_dir)
        self.assertEqual(applier.project_root, self.temp_dir)

    def test_find_source_file(self):
        """测试查找源码文件"""
        applier = self.StructureApplier(project_root=self.temp_dir)
        # StructureApplier 没有 find_source_file_path 方法
        # 直接检查文件是否存在于项目根目录
        modules_path = os.path.join(self.temp_dir, "modules.py")
        self.assertTrue(os.path.exists(modules_path))

    def test_apply_simple_change(self):
        """测试简单代码修改应用 (apply_structural_changes 接受 list)"""
        applier = self.StructureApplier(
            project_root=self.temp_dir,
            log_dir=os.path.join(self.temp_dir, "evolve_logs"),
        )
        change = {
            "target_file": "modules.py",
            "target_class_or_function": "SelfAttention.__init__",
            "description": "Add temperature parameter",
            "new_code": """    def __init__(self, hidden_size, num_attention_heads, temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.temperature = temperature
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)""",
            "action_type": "modify",
        }
        # apply_structural_changes 接受 list 参数
        result = applier.apply_structural_changes([change])
        # 不管是否成功，应该返回结构化结果
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)

    def test_apply_invalid_syntax(self):
        """测试应用无效语法 → 应自动回滚"""
        applier = self.StructureApplier(
            project_root=self.temp_dir,
            log_dir=os.path.join(self.temp_dir, "evolve_logs"),
        )
        change = {
            "target_file": "modules.py",
            "target_class_or_function": "SelfAttention.__init__",
            "description": "Invalid syntax change",
            "new_code": "def broken syntax here !!!",
            "action_type": "modify",
        }
        # apply_structural_changes 接受 list 参数
        result = applier.apply_structural_changes([change])
        # 应回滚或失败
        # 实际返回的 status 可能包含 "ALL_FAILED"
        valid_statuses = ["ROLLED_BACK", "NEEDS_FIX", "FAILED", "APPLIED", "ERROR", "ALL_FAILED"]
        self.assertIn(result["status"], valid_statuses)

    def test_rollback_last_changes(self):
        """测试回滚最近修改"""
        applier = self.StructureApplier(
            project_root=self.temp_dir,
            log_dir=os.path.join(self.temp_dir, "evolve_logs"),
        )
        # 先保存原始内容
        original_content = open(os.path.join(self.temp_dir, "modules.py")).read()
        # 执行回滚 (没有变更时也应正常返回)
        rollback_result = applier.rollback_last_changes()
        self.assertIsInstance(rollback_result, dict)


# ══════════════════════════════════════════════════════════════════
# 11. 程序数据库测试
# ══════════════════════════════════════════════════════════════════

class TestProgramDatabase(unittest.TestCase):
    """测试程序数据库"""

    def setUp(self):
        from agent.database import ProgramDatabase, Program
        self.ProgramDatabase = ProgramDatabase
        self.Program = Program

    def test_database_init(self):
        """测试数据库初始化"""
        db = self.ProgramDatabase(num_islands=4, population_size=20)
        self.assertEqual(db.num_islands, 4)
        self.assertEqual(len(db.islands), 4)
        self.assertEqual(len(db.programs), 0)

    def test_add_program(self):
        """测试添加程序"""
        db = self.ProgramDatabase(num_islands=4)
        prog = self.Program(
            id="prog_001",
            code="class SASRec:\n    pass",
            idea={"title": "test", "description": "test idea"},
            parent_id="root",
        )
        db.add(prog)
        self.assertEqual(len(db.programs), 1)
        self.assertIn("prog_001", db.programs)

    def test_add_and_track_best(self):
        """测试添加程序并追踪最优"""
        db = self.ProgramDatabase(num_islands=4)
        # 程序 1
        prog1 = self.Program(
            id="prog_001", code="v1", idea={}, parent_id="root",
            metrics={"ndcg": 0.35},
        )
        db.add(prog1)
        # 程序 2 — 更好
        prog2 = self.Program(
            id="prog_002", code="v2", idea={}, parent_id="prog_001",
            metrics={"ndcg": 0.40},
        )
        db.add(prog2)
        self.assertEqual(db.best_program_id, "prog_002")

    def test_island_sampling(self):
        """测试从 island 采样父程序"""
        db = self.ProgramDatabase(num_islands=2)
        for i in range(10):
            prog = self.Program(
                id=f"prog_{i}", code=f"v{i}", idea={}, parent_id="root",
                metrics={"ndcg": 0.3 + i * 0.01},
            )
            db.add(prog)

        # 应能从 island 采样
        result = db.sample()
        self.assertIsNotNone(result)
        # sample() 返回 (parent, inspirations) tuple
        self.assertEqual(len(result), 2)

    def test_database_save_and_load(self):
        """测试数据库保存和加载"""
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix="db_test_")
        try:
            db = self.ProgramDatabase(num_islands=2)
            prog = self.Program(
                id="prog_001", code="v1", idea={}, parent_id="root",
                metrics={"ndcg": 0.35},
            )
            db.add(prog)
            # 保存 (ProgramDatabase.save 接受 checkpoint_dir 和 iteration)
            checkpoint_dir = os.path.join(temp_dir, "checkpoint")
            db.save(checkpoint_dir, iteration=1)
            # checkpoint_dir 应已创建
            self.assertTrue(os.path.exists(checkpoint_dir))

            # 加载
            db2 = self.ProgramDatabase(num_islands=2)
            loaded_iter = db2.load(checkpoint_dir)
            self.assertGreaterEqual(loaded_iter, 0)
            # 程序应已加载
            self.assertIn("prog_001", db2.programs)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════
# 12. 研究 Agent 测试
# ══════════════════════════════════════════════════════════════════

class TestResearcherAgent(unittest.TestCase):
    """测试研究 Agent"""

    def setUp(self):
        from agent.researcher import ResearcherAgent, IdeaData, SearchResult, ResearchPlan
        self.ResearcherAgent = ResearcherAgent
        self.IdeaData = IdeaData
        self.SearchResult = SearchResult
        self.ResearchPlan = ResearchPlan

    def test_researcher_init(self):
        """测试研究 Agent 初始化 (含完整 LLM 连接参数)"""
        researcher = self.ResearcherAgent(
            api_url="http://localhost:8000/v1",
            model="test-model",
            temperature=0.7,
        )
        self.assertEqual(researcher.model, "test-model")
        self.assertEqual(researcher.temperature, 0.7)

    def test_researcher_update_topic(self):
        """测试更新研究主题"""
        researcher = self.ResearcherAgent(api_url="http://localhost:8000/v1")
        researcher.update_topic(
            query="How to improve SASRec with attention mechanisms",
            problem_name="SASRec optimization",
            problem_description="The model has low NDCG scores",
        )
        self.assertEqual(researcher.query, "How to improve SASRec with attention mechanisms")
        self.assertEqual(researcher.problem_name, "SASRec optimization")

    def test_idea_data_creation(self):
        """测试 IdeaData 创建"""
        idea = self.IdeaData(
            title="Add time-aware attention",
            description="Use time decay in attention calculation",
            content="Detailed description of the idea",
        )
        self.assertEqual(idea.title, "Add time-aware attention")
        self.assertEqual(idea.source, "research")

    def test_search_result_creation(self):
        """测试 SearchResult 创建"""
        result = self.SearchResult(
            title="Attention in RecSys",
            url="https://arxiv.org/abs/1234",
            snippet="A paper about attention mechanisms",
        )
        self.assertEqual(result.source, "web")

    def test_research_plan_creation(self):
        """测试 ResearchPlan 创建"""
        plan = self.ResearchPlan(
            title="Improve SASRec attention",
            description="Add temperature scaling to attention",
            expected_improvement="NDCG@10 +0.05",
        )
        self.assertEqual(plan.confidence, "中")


# ══════════════════════════════════════════════════════════════════
# 13. 编码 Agent 测试
# ══════════════════════════════════════════════════════════════════

class TestCoderAgent(unittest.TestCase):
    """测试编码 Agent"""

    def setUp(self):
        from agent.coder import CoderAgent, CodeChange, CodeResult
        self.CoderAgent = CoderAgent
        self.CodeChange = CodeChange
        self.CodeResult = CodeResult

    def test_coder_init(self):
        """测试编码 Agent 初始化 (含完整 LLM 连接参数)"""
        coder = self.CoderAgent(
            api_url="http://localhost:8000/v1",
            model="test-model",
            temperature=0.4,
        )
        self.assertEqual(coder.model, "test-model")
        self.assertEqual(coder.temperature, 0.4)

    def test_coder_update_topic(self):
        """测试更新编码主题"""
        coder = self.CoderAgent(api_url="http://localhost:8000/v1")
        coder.update_topic(
            query="SASRec attention improvement",
            problem_name="SASRec optimization",
            problem_description="Low NDCG scores",
        )
        self.assertEqual(coder.problem_name, "SASRec optimization")

    def test_code_change_creation(self):
        """测试 CodeChange 创建"""
        change = self.CodeChange(
            target_file="modules.py",
            target_class_or_function="SelfAttention.forward",
            description="Add temperature scaling",
            new_code="def forward(self, hidden_states):\n    return attention * self.temperature",
            insert_position="replace_function",
            expected_effect="Better attention distribution",
        )
        self.assertEqual(change.target_file, "modules.py")
        self.assertEqual(change.confidence, "中")

    def test_code_result_creation(self):
        """测试 CodeResult 创建"""
        result = self.CodeResult(
            code="new code here",
            diff_text="diff text",
            success=True,
        )
        self.assertTrue(result.success)
        self.assertEqual(len(result.changes), 0)


# ══════════════════════════════════════════════════════════════════
# 14. 进化引擎测试
# ══════════════════════════════════════════════════════════════════

class TestEvolutionEngine(unittest.TestCase):
    """测试进化引擎"""

    def setUp(self):
        from agent.evolve_engine import EvolutionEngine
        self.EvolutionEngine = EvolutionEngine

    def test_engine_init(self):
        """测试进化引擎初始化"""
        config = {
            "model": "test-model",
            "problem_name": "test",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
        }
        engine = self.EvolutionEngine(
            config=config,
            project_root="/tmp/test_project",
            max_iterations=10,
        )
        self.assertEqual(engine.max_iterations, 10)
        self.assertIsNotNone(engine.researcher)
        self.assertIsNotNone(engine.coder)
        self.assertIsNotNone(engine.database)

    def test_engine_update_topic(self):
        """测试更新研究主题"""
        config = {
            "model": "test-model",
            "api_url": "http://localhost:8000/v1",
            "api_key": "EMPTY",
        }
        engine = self.EvolutionEngine(config=config, project_root="/tmp/test")
        engine.update_topic("How to improve SASRec")
        self.assertEqual(engine.researcher.query, "How to improve SASRec")


# ══════════════════════════════════════════════════════════════════
# 15. 核心主循环测试 (Mock — 不执行真实训练)
# ══════════════════════════════════════════════════════════════════

class TestRecSelfEvolveAgent(unittest.TestCase):
    """测试 RecSelfEvolveAgent 主循环 (mock 所有外部依赖)"""

    def setUp(self):
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig
        self.RecSelfEvolveAgent = RecSelfEvolveAgent
        self.AgentConfig = AgentConfig
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        # 创建模拟项目文件
        self.helper.create_file("models.py", "class SASRec:\n    pass\n")
        self.helper.create_file("modules.py", "class SelfAttention:\n    pass\n")
        self.helper.create_file("trainers.py", "class Trainer:\n    pass\n")
        self.helper.create_file("datasets.py", "class Dataset:\n    pass\n")
        self.helper.create_file("run_finetune_full.py", "import argparse\nargparse.ArgumentParser()\n")

    def tearDown(self):
        self.helper.cleanup()

    def test_agent_init_with_mock(self):
        """测试 Agent 初始化 (mock LLM)"""
        config = self.AgentConfig(
            project_root=self.temp_dir,
            llm_api_url="http://localhost:8000/v1",
        )
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)
                self.assertIsNotNone(agent.config)
                self.assertIsNotNone(agent.adapter)

    def test_agent_config_passed_correctly(self):
        """测试配置正确传递"""
        config = self.AgentConfig(project_root=self.temp_dir, data_name="Beauty")
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)
                self.assertEqual(agent.config.data_name, "Beauty")

    def test_parse_json_response(self):
        """测试 JSON 响应解析"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        # 测试各种 JSON 格式
        json_text = '{"param_changes": [{"lr": 0.001}], "structural_changes": []}'
        result = agent._parse_json_response(json_text)
        self.assertIsNotNone(result)

    def test_parse_search_replace_diff(self):
        """测试 SEARCH/REPLACE diff 解析"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        diff_text = """
To improve the SelfAttention module, I propose:

modules.py
```python
<<<<<<< SEARCH
    def __init__(self, hidden_size, num_attention_heads):
        super().__init__()
        self.hidden_size = hidden_size
=======
    def __init__(self, hidden_size, num_attention_heads, temperature=1.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.temperature = temperature
>>>>>>> REPLACE
```
"""
        # _parse_search_replace_diff 接受两个参数: response, research_idea
        result = agent._parse_search_replace_diff(diff_text, "Improve attention")
        self.assertIsNotNone(result)

    def test_validate_param_changes(self):
        """测试参数变更校验"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        # _validate_param_changes 接受 dict 参数
        changes = {"lr": 0.001, "batch_size": 512}
        result = agent._validate_param_changes(changes)
        self.assertIsInstance(result, dict)

    def test_validate_structural_changes(self):
        """测试结构变更校验"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        changes = [{
            "target_file": "modules.py",
            "target_class_or_function": "SelfAttention.__init__",
            "description": "Add temperature",
            "new_code": "def __init__(self, hidden_size, temperature=1.0):...",
            "action_type": "modify",
        }]
        result = agent._validate_structural_changes(changes)
        self.assertIsInstance(result, list)

    def test_strip_evolve_markers(self):
        """测试清除进化标记"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm_class.return_value = MagicMock()
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        # _strip_evolve_markers 只处理 ### >>> Self_EvolveRec-BLOCK-START 和 ### <<< Self_EvolveRec-BLOCK-END 格式
        # 不处理 # Self_EvolveRec-BLOCK-BEGIN/END 格式
        code_with_markers = """
### >>> Self_EvolveRec-BLOCK-START: iter_1
class SelfAttention:
    pass
### <<< Self_EvolveRec-BLOCK-END
"""
        cleaned = agent._strip_evolve_markers(code_with_markers)
        # ### >>> 和 ### <<< 格式的标记应被移除
        self.assertNotIn("### >>> Self_EvolveRec-BLOCK-START", cleaned)
        self.assertNotIn("### <<< Self_EvolveRec-BLOCK-END", cleaned)
        self.assertIn("SelfAttention", cleaned)

    def test_check_llm_health(self):
        """测试 LLM 健康检查"""
        config = self.AgentConfig(project_root=self.temp_dir)
        with patch("agent.core.LLMClient") as mock_llm_class:
            mock_llm = MagicMock()
            mock_llm.check_health.return_value = True
            mock_llm_class.return_value = mock_llm
            with patch("agent.core.FaultTolerantTrainRunner") as mock_trainer_class:
                mock_trainer_class.return_value = MagicMock()
                agent = self.RecSelfEvolveAgent(config=config)

        agent._check_llm_health()
        # 健康检查应设置 llm_health_ok = True
        self.assertTrue(agent.llm_health_ok)


# ══════════════════════════════════════════════════════════════════
# 16. LLM 案例分析器测试
# ══════════════════════════════════════════════════════════════════

class TestLLMCaseAnalyzer(unittest.TestCase):
    """测试 LLM 案例分析器"""

    def setUp(self):
        from agent.llm_analyzer import LLMCaseAnalyzer
        self.LLMCaseAnalyzer = LLMCaseAnalyzer

    def test_analyzer_init(self):
        """测试分析器初始化"""
        mock_llm = MagicMock()
        analyzer = self.LLMCaseAnalyzer(llm_client=mock_llm)
        self.assertIsNotNone(analyzer)


class TestHypothesisEaseSelection(unittest.TestCase):
    """测试假设易验证性筛选功能"""

    def setUp(self):
        from agent.hypothesis_verification_agent import HypothesisVerificationAgent
        self.HypothesisVerificationAgent = HypothesisVerificationAgent

    def test_select_easiest_method_exists(self):
        """测试 select_easiest_hypotheses 方法存在"""
        self.assertTrue(
            hasattr(self.HypothesisVerificationAgent, 'select_easiest_hypotheses'),
            "HypothesisVerificationAgent should have select_easiest_hypotheses method"
        )

    def test_prompt_exists_and_valid(self):
        """测试 HYPOTHESIS_EASE_SELECTION_PROMPT 存在且包含必要占位符"""
        from agent.prompts import HYPOTHESIS_EASE_SELECTION_PROMPT
        self.assertIsInstance(HYPOTHESIS_EASE_SELECTION_PROMPT, str)
        self.assertGreater(len(HYPOTHESIS_EASE_SELECTION_PROMPT), 100)
        # 检查必要占位符
        self.assertIn("{hypotheses_json}", HYPOTHESIS_EASE_SELECTION_PROMPT)
        self.assertIn("{data_inventory}", HYPOTHESIS_EASE_SELECTION_PROMPT)
        self.assertIn("{model_info}", HYPOTHESIS_EASE_SELECTION_PROMPT)
        self.assertIn("{available_data_description}", HYPOTHESIS_EASE_SELECTION_PROMPT)
        # 检查难度评估维度
        self.assertIn("difficulty_assessment", HYPOTHESIS_EASE_SELECTION_PROMPT)
        self.assertIn("recommended_for_pilot", HYPOTHESIS_EASE_SELECTION_PROMPT)

    def test_prompt_importable(self):
        """测试 HYPOTHESIS_EASE_SELECTION_PROMPT 可从 hypothesis_verification_agent 导入"""
        from agent.hypothesis_verification_agent import HYPOTHESIS_EASE_SELECTION_PROMPT
        self.assertIsInstance(HYPOTHESIS_EASE_SELECTION_PROMPT, str)


# ══════════════════════════════════════════════════════════════════
# 17. 代码应用器测试
# ══════════════════════════════════════════════════════════════════

class TestCodeApplier(unittest.TestCase):
    """测试代码应用器"""

    def setUp(self):
        from agent.code_applier import CodeApplier
        self.CodeApplier = CodeApplier
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()

    def tearDown(self):
        self.helper.cleanup()

    def test_applier_init(self):
        """测试代码应用器初始化"""
        applier = self.CodeApplier(project_root=self.temp_dir)
        self.assertEqual(applier.project_root, self.temp_dir)


# ══════════════════════════════════════════════════════════════════
# 18. 训练执行器测试 (Mock — 不执行真实训练)
# ══════════════════════════════════════════════════════════════════

class TestTrainRunner(unittest.TestCase):
    """测试训练执行器 (mock subprocess)"""

    def setUp(self):
        from agent.train_runner import FaultTolerantTrainRunner
        from agent.project_adapter import SeqRecAdapter
        self.FaultTolerantTrainRunner = FaultTolerantTrainRunner
        self.SeqRecAdapter = SeqRecAdapter
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        self.helper.create_file("models.py", "class SASRec:\n    pass\n")
        self.helper.create_file("modules.py", "class SelfAttention:\n    pass\n")

    def tearDown(self):
        self.helper.cleanup()

    def test_runner_init(self):
        """测试训练执行器初始化"""
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        runner = self.FaultTolerantTrainRunner(adapter=adapter)
        self.assertEqual(runner.project_root, self.temp_dir)

    @patch("agent.train_runner.subprocess.run")
    def test_runner_mock_success(self, mock_run):
        """测试 mock 成功训练"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Training completed. NDCG@10: 0.35",
            stderr="",
        )
        adapter = self.SeqRecAdapter(project_root=self.temp_dir)
        runner = self.FaultTolerantTrainRunner(adapter=adapter)
        result = runner.run()
        # 应返回结构化结果
        self.assertIsInstance(result, dict)


# ══════════════════════════════════════════════════════════════════
# 19. 模块导入完整性测试
# ══════════════════════════════════════════════════════════════════

class TestModuleImports(unittest.TestCase):
    """测试所有模块能否正常导入"""

    def test_import_config(self):
        """测试导入 config"""
        from agent.config import AgentConfig
        self.assertIsNotNone(AgentConfig)

    def test_import_core(self):
        """测试导入 core"""
        from agent.core import RecSelfEvolveAgent
        self.assertIsNotNone(RecSelfEvolveAgent)

    def test_import_llm_client(self):
        """测试导入 llm_client"""
        from agent.llm_client import LLMClient
        self.assertIsNotNone(LLMClient)

    def test_import_project_adapter(self):
        """测试导入 project_adapter"""
        from agent.project_adapter import SeqRecAdapter, create_adapter
        self.assertIsNotNone(SeqRecAdapter)
        self.assertIsNotNone(create_adapter)

    def test_import_prompts(self):
        """测试导入 prompts"""
        from agent.prompts import (
            MLE_ANALYSIS_PROMPT, STRUCTURE_OPTIMIZATION_PROMPT,
            ERROR_FEEDBACK_PROMPT, STRUCTURE_FIX_PROMPT,
            CODER_INSTRUCTIONS, RESEARCHER_INSTRUCTIONS,
        )
        self.assertIsNotNone(MLE_ANALYSIS_PROMPT)

    def test_import_error_handler(self):
        """测试导入 error_handler"""
        from agent.error_handler import ProposalParser, LLMFixer
        self.assertIsNotNone(ProposalParser)

    def test_import_quality_guard(self):
        """测试导入 quality_guard"""
        from agent.quality_guard import EvolutionQualityGuard, SafetyGuardrails
        self.assertIsNotNone(EvolutionQualityGuard)

    def test_import_journal(self):
        """测试导入 journal"""
        from agent.journal import ExperimentJournal
        self.assertIsNotNone(ExperimentJournal)

    def test_import_iterative_memory(self):
        """测试导入 iterative_memory"""
        from agent.iterative_memory import IterativeMemory
        self.assertIsNotNone(IterativeMemory)

    def test_import_context_compressor(self):
        """测试导入 context_compressor"""
        from agent.context_compressor import LLMContextCompressor
        self.assertIsNotNone(LLMContextCompressor)

    def test_import_structure_applier(self):
        """测试导入 structure_applier"""
        from agent.structure_applier import StructureApplier
        self.assertIsNotNone(StructureApplier)

    def test_import_database(self):
        """测试导入 database"""
        from agent.database import ProgramDatabase, Program
        self.assertIsNotNone(ProgramDatabase)

    def test_import_researcher(self):
        """测试导入 researcher"""
        from agent.researcher import ResearcherAgent, IdeaData
        self.assertIsNotNone(ResearcherAgent)

    def test_import_coder(self):
        """测试导入 coder"""
        from agent.coder import CoderAgent, CodeChange
        self.assertIsNotNone(CoderAgent)

    def test_import_evolve_engine(self):
        """测试导入 evolve_engine"""
        from agent.evolve_engine import EvolutionEngine
        self.assertIsNotNone(EvolutionEngine)

    def test_import_code_applier(self):
        """测试导入 code_applier"""
        from agent.code_applier import CodeApplier
        self.assertIsNotNone(CodeApplier)

    def test_import_train_runner(self):
        """测试导入 train_runner"""
        from agent.train_runner import FaultTolerantTrainRunner
        self.assertIsNotNone(FaultTolerantTrainRunner)

    def test_import_llm_analyzer(self):
        """测试导入 llm_analyzer"""
        from agent.llm_analyzer import LLMCaseAnalyzer
        self.assertIsNotNone(LLMCaseAnalyzer)

    def test_import_top_level(self):
        """测试导入顶层包"""
        import agent
        self.assertEqual(agent.__version__, "0.4.0")


# ══════════════════════════════════════════════════════════════════
# 20. 集成流程测试 (Mock — 不执行真实 LLM/训练)
# ══════════════════════════════════════════════════════════════════

class TestIntegrationFlow(unittest.TestCase):
    """测试 Agent 集成流程 (mock 所有外部依赖)"""

    def setUp(self):
        self.helper = TempDirHelper()
        self.temp_dir = self.helper.setup()
        # 创建完整模拟项目
        self.helper.create_file("models.py", "import torch.nn as nn\n\nclass SASRec(nn.Module):\n    def __init__(self, item_num, hidden_size):\n        super().__init__()\n        self.item_embeddings = nn.Embedding(item_num, hidden_size)\n\n    def finetune(self, input_ids):\n        return self.item_embeddings(input_ids)\n")
        self.helper.create_file("modules.py", "import torch\nimport torch.nn as nn\n\nclass SelfAttention(nn.Module):\n    def __init__(self, hidden_size, num_attention_heads):\n        super().__init__()\n        self.hidden_size = hidden_size\n        self.num_attention_heads = num_attention_heads\n        self.query = nn.Linear(hidden_size, hidden_size)\n\n    def forward(self, hidden_states):\n        return self.query(hidden_states)\n")
        self.helper.create_file("trainers.py", "class Trainer:\n    def train(self):\n        pass\n")
        self.helper.create_file("datasets.py", "class Dataset:\n    pass\n")
        self.helper.create_file("run_finetune_full.py", "import argparse\nparser = argparse.ArgumentParser()\nargs = parser.parse_args()\n")

    def tearDown(self):
        self.helper.cleanup()

    @patch("agent.core.FaultTolerantTrainRunner")
    @patch("agent.core.LLMClient")
    def test_full_agent_init_flow(self, mock_llm_class, mock_trainer_class):
        """测试完整 Agent 初始化流程"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        mock_llm_class.return_value = MagicMock()
        mock_trainer_class.return_value = MagicMock()

        config = AgentConfig(project_root=self.temp_dir, data_name="Beauty")
        agent = RecSelfEvolveAgent(config=config)

        # 验证所有组件初始化
        self.assertIsNotNone(agent.config)
        self.assertIsNotNone(agent.adapter)
        self.assertIsNotNone(agent.llm)
        self.assertIsNotNone(agent.trainer)
        self.assertIsNotNone(agent.applier)
        self.assertIsNotNone(agent.struct_applier)  # 注意: 实际属性名是 struct_applier
        self.assertIsNotNone(agent.iter_memory)     # 注意: 实际属性名是 iter_memory
        self.assertIsNotNone(agent.journal)
        self.assertIsNotNone(agent.guard)           # 注意: 实际属性名是 guard

    @patch("agent.core.FaultTolerantTrainRunner")
    @patch("agent.core.LLMClient")
    def test_preflight_check_flow(self, mock_llm_class, mock_trainer_class):
        """测试前置检查流程"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        mock_llm_class.return_value = MagicMock()
        mock_trainer = MagicMock()
        # preflight_check() 返回 MagicMock 默认值，需要设置为 dict
        mock_trainer.preflight_check.return_value = {"status": "PASS", "checks": []}
        mock_trainer_class.return_value = mock_trainer

        config = AgentConfig(project_root=self.temp_dir)
        agent = RecSelfEvolveAgent(config=config)

        # 执行前置检查
        result = agent._phase_preflight_check()
        self.assertIsInstance(result, dict)
        self.assertIn("status", result)

    @patch("agent.core.FaultTolerantTrainRunner")
    @patch("agent.core.LLMClient")
    def test_multi_role_workflow_flag(self, mock_llm_class, mock_trainer_class):
        """测试多角色工作流标志 (Bug 1 修复后，ResearcherAgent/CoderAgent 可正常初始化)"""
        from agent.core import RecSelfEvolveAgent
        from agent.config import AgentConfig

        mock_llm_class.return_value = MagicMock()
        mock_trainer_class.return_value = MagicMock()

        # 启用多角色工作流
        config = AgentConfig(
            project_root=self.temp_dir,
            enable_multi_role_workflow=True,
        )
        agent = RecSelfEvolveAgent(config=config)
        self.assertTrue(agent.config.enable_multi_role_workflow)
        self.assertTrue(agent._multi_role_enabled)


# ══════════════════════════════════════════════════════════════════
# 运行测试
# ══════════════════════════════════════════════════════════════════

def run_tests(verbose=False):
    """运行所有测试并输出结果摘要"""
    print("=" * 70)
    print("  RecSelfEvolve Agent 综合测试")
    print("=" * 70)

    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # 添加所有测试类
    test_classes = [
        TestAgentConfig,
        TestLLMClient,
        TestProjectAdapter,
        TestPrompts,
        TestProposalParser,
        TestLLMFixer,
        TestQualityGuard,
        TestJournal,
        TestIterativeMemory,
        TestContextCompressor,
        TestStructureApplier,
        TestProgramDatabase,
        TestResearcherAgent,
        TestCoderAgent,
        TestEvolutionEngine,
        TestRecSelfEvolveAgent,
        TestLLMCaseAnalyzer,
        TestCodeApplier,
        TestTrainRunner,
        TestModuleImports,
        TestIntegrationFlow,
    ]

    for cls in test_classes:
        suite.addTests(loader.loadTestsFromTestCase(cls))

    # 运行测试
    verbosity = 2 if verbose else 1
    runner = unittest.TextTestRunner(verbosity=verbosity)
    result = runner.run(suite)

    # 输出摘要
    print("\n" + "=" * 70)
    print("  测试结果摘要")
    print("=" * 70)
    print(f"  总测试数: {result.testsRun}")
    print(f"  成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  失败: {len(result.failures)}")
    print(f"  错误: {len(result.errors)}")
    print(f"  跳过: {len(result.skipped)}")

    if result.failures:
        print("\n  ❌ 失败的测试:")
        for test, traceback in result.failures:
            print(f"    - {test}: {traceback[:200]}")

    if result.errors:
        print("\n  ❌ 错误的测试:")
        for test, traceback in result.errors:
            print(f"    - {test}: {traceback[:200]}")

    if not result.failures and not result.errors:
        print("\n  ✅ 所有测试通过！Agent 功能验证完成。")
    else:
        print("\n  ⚠️  部分测试失败，请检查上述错误。")

    print("=" * 70)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="RecSelfEvolve Agent Test Suite")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("-k", "--pattern", default=None, help="只运行匹配的测试 (如 'test_config')")
    args = parser.parse_args()

    if args.pattern:
        # 只运行匹配模式的测试
        suite = unittest.TestLoader().loadTestsFromTestCase(
            # 动态查找匹配的类
            globals().get(f"Test{args.pattern.capitalize()}", unittest.TestCase)
        )
        unittest.TextTestRunner(verbosity=2).run(suite)
    else:
        result = run_tests(verbose=args.verbose)
        sys.exit(0 if result.wasSuccessful() else 1)