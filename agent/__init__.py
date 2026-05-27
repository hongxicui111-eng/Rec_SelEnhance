"""
RecSelfEvolve — 推荐系统自增强 Agent 框架

基于两篇顶会论文的核心方法论：
1. Self-Evolving Recommendation System: End-To-End Autonomous Model Optimization With LLM Agents (Google/YouTube, 2026)
2. Self-EvolveRec: Self-Evolving Recommender Systems with LLM-based Directional Feedback (KAIST, ICLR 2026)

增强功能:
- 惊喜子集评估 (Surprise Subset Evaluation): 检测模型对"惊喜"交互的捕获能力
- 错误案例提取与文本转换: 从推理错误中提取 500 个案例供 LLM 分析
- LLM 案例分析: 让 LLM 从错误案例推理出模型瓶颈和改进方向
- **模型结构修改**: LLM 不仅调参数，还能提出模型结构的代码修改方案并自动执行
- **迭代修改记忆**: LLM 每次修改都能感知完整的历史修改因果链 (避免重复踩坑)
"""

from .core import RecSelfEvolveAgent
from .config import AgentConfig
from .llm_analyzer import LLMCaseAnalyzer
from .structure_applier import StructureApplier
from .iterative_memory import IterativeMemory
from .context_compressor import LLMContextCompressor
from .code_query_tool import CodeQueryTool

from .hypothesis_verification_agent import HypothesisVerificationAgent

from .llm_utils import (
    extract_json_block, robust_json_parse, diagnose_json_error,
    parse_json_from_response, clean_code_response, clean_markdown_wrapper,
    LLMRetryHelper,
)
from .script_executor import (
    extract_output_path, DataInjector, ScriptExecutor,
)

__version__ = "0.8.0"