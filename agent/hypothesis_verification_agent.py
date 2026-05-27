#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
假设验证 Agent — 自主式假设验证框架

核心设计理念:
  旧版 HypothesisVerifier 的局限:
    - 固定 6 种验证方法 (item_popularity, category_bias, ...)
    - 每种方法用硬编码阈值判断 (如 ratio > 1.5 → CONFIRMED)
    - 自定义方法只能返回 UNVERIFIABLE
    - LLM 只做假设提取 + 模板匹配, 不参与真正的验证推理

  新版 HypothesisVerificationAgent 的突破:
    - LLM 为每个假设**自动生成**验证方案 (不受限于固定类型)
    - Agent 根据验证方案**自主写代码**执行数据验证
    - 代码执行失败时, Agent 带着错误信息**自动修正代码** (类似 self-correction)
    - LLM **解读执行结果**, 判断假设是否被数据支持 (而非硬编码阈值)
    - 整个流程是一个完整的 Agent loop: Plan → Code → Execute → Analyze

  工作流程:
    1. extract_hypotheses(): 从 LLM 分析中提取可验证假设
    2. generate_verification_plan(): 为每个假设生成验证方案
    3. discover_data(): 发现可用的数据源
    4. generate_verification_code(): 根据方案写验证脚本
    5. execute_verification_code(): 执行脚本 (失败则修正重试)
    6. analyze_results(): LLM 解读统计结果, 判断假设成立与否
    7. generate_verification_report(): 汇总报告

  与旧版兼容:
    - 保持与 HypothesisVerifier 相同的外部接口
    - core.py 只需替换 import, 无需修改调用方式
    - 旧版作为 fallback 在新版完全失败时使用
"""

import os
import sys
import json
import logging
import subprocess
import tempfile
import shutil
import time
import traceback
import re
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("rec_self_evolve.hypothesis_verification_agent")


# ════════════════════════════════════════
# 假设提取 Prompt (与旧版一致, 但去掉固定方法限制)
# ════════════════════════════════════════

HYPOTHESIS_EXTRACTION_PROMPT_V2 = """你是一位严谨的数据科学家，正在从推荐系统分析报告中提取**可验证的假设**。

## 背景
LLM 分析了推荐模型 (SASRec) 的错误案例，给出了错误模式、模型瓶颈和改进建议。
但 LLM 的分析可能包含幻觉或主观臆断。我们需要提取其中的**可验证假设**，
用数据来确认或反驳这些结论。

## LLM 分析结论
```json
{llm_analysis_json}
```

## 可用数据资源
以下是我们项目中可用的数据资源:
{data_inventory}

## 任务
从上面的分析结论中，提取所有**可以用数据验证的假设**。

一个假设是可验证的，意味着:
- 我们有 (或能计算得到) 验证所需的数据
- 可以通过统计分析、对比实验、分布检验等方式量化验证
- 验证结果能给出明确的"成立/不成立/部分成立"判断

**不要局限于固定验证类型!** 任何可以用数据回答的问题都是可验证假设。
例如:
- "冷门物品误推率更高" → 对比误推案例中冷门物品占比与全量占比
- "模型对长序列用户的注意力衰减" → 分析不同序列长度下的注意力分布
- "训练数据中类别分布不均衡导致偏差" → 检查类别分布与误推类别分布的相关性
- "模型倾向于推荐近期交互的物品" → 计算预测物品与最近交互物品的时间距离

对于每个假设:
- 用精确的描述说明假设内容 (不要模糊)
- 说明验证思路 (如何用数据验证这个假设)
- 说明需要什么数据 (具体到数据文件或计算方式)
- 期望的数据现象 (如果假设成立, 应观察到什么?)
- 反驳的数据现象 (如果假设不成立, 应观察到什么?)

### 输出格式 (严格遵守)

```json
{{
  "hypotheses": [
    {{
      "id": "H1",
      "claim": "精确描述假设内容",
      "source_field": "error_patterns | model_bottleneck | surprise_failure_reasons | improvement_suggestions",
      "verification_thought": "验证思路: 如何用数据来验证这个假设 (自由描述, 不受限于固定类型)",
      "data_needed": ["需要的数据列表, 如: 训练数据交互序列, 物品元数据, 误推案例物品ID, ..."],
      "expected_if_true": "如果假设成立, 应观察到的数据现象",
      "expected_if_false": "如果假设不成立, 应观察到的数据现象",
      "confidence_in_llm": "high | medium | low (LLM 对此结论的置信度估计)",
      "priority": 1-5 (验证优先级, 5=最高)
    }}
  ],
  "summary": "哪些 LLM 结论最可能是幻觉/臆断, 哪些最可能有数据支撑"
}}
```"""


# ════════════════════════════════════════
# 验证方案生成 Prompt
# ════════════════════════════════════════

VERIFICATION_PLAN_PROMPT = """你是一位数据科学家，正在为以下假设设计详细的验证方案。

## 假设
- ID: {hypothesis_id}
- Claim: {hypothesis_claim}
- 验证思路: {verification_thought}
- 需要的数据: {data_needed}
- 如果成立应观察到的现象: {expected_if_true}
- 如果不成立应观察到的现象: {expected_if_false}

## 可用数据资源
{data_inventory}

## 已加载的数据摘要
{loaded_data_summary}

## 任务
设计一个**具体的、可执行的验证方案**。方案必须:
1. 明确指定使用哪些数据
2. 明确具体的统计分析方法 (如: 比率对比, t-test, 相关性分析, 分布对比, ...)
3. 给出具体的验证代码思路 (不需要完整代码, 但需要清晰的算法描述)
4. 定义确认/反驳的判断标准 (什么样的数据结果算是确认? 什么样的算是反驳?)

### 输出格式

```json
{{
  "hypothesis_id": "{hypothesis_id}",
  "verification_plan": {{
    "method_name": "验证方法名称 (自由命名)",
    "method_description": "验证方法的详细描述",
    "data_sources": ["使用的具体数据源"],
    "analysis_steps": [
      "Step 1: ...",
      "Step 2: ...",
      "Step 3: ..."
    ],
    "statistical_method": "使用的统计方法 (如: proportion_comparison, chi_square, t_test, correlation, etc.)",
    "code_outline": "验证代码的核心算法描述 (伪代码或步骤)",
    "confirm_criteria": "什么统计结果算确认假设",
    "refute_criteria": "什么统计结果算反驳假设",
    "partial_criteria": "什么统计结果算部分确认",
    "expected_output_format": "验证代码应输出的 JSON 结果格式"
  }}
}}
```"""


# ════════════════════════════════════════
# 验证代码生成 Prompt
# ════════════════════════════════════════

VERIFICATION_CODE_PROMPT = """你是一位 Python 数据科学家，正在编写一个验证脚本来检验假设。

## 假设
- Claim: {hypothesis_claim}

## 验证方案
```json
{verification_plan_json}
```

## 可用数据文件与变量
以下数据已准备好, 你的代码可以直接使用:
{available_data_description}

## 重要约束
1. 代码必须是**独立的 Python 脚本**, 不依赖任何外部 API 或特殊库 (只用 numpy, json, collections, os, math, scipy.stats)
2. 代码必须将结果输出为 JSON 并写入 `{output_file_path}`
3. JSON 输出格式必须包含以下字段:
```json
{{
  "hypothesis_id": "{hypothesis_id}",
  "statistics": {{
    // 具体的统计量 (自由定义, 但要包含验证方案中提到的所有指标)
  }},
  "interpretation": "对统计结果的简要文字解读",
  "raw_data_sample": "一小段原始数据样本 (用于辅助分析)"
}}
```
4. 如果某些数据不可用, 在 statistics 中用 null 表示, 不要让脚本崩溃
5. 代码中不要使用 print() 输出中间结果 (只写最终 JSON)
6. 代码中必须处理异常 — 如果数据加载失败, 写一个带 error 字段的 JSON 而不是崩溃

## 输出
只输出完整的 Python 脚本代码 (不要任何解释文字, 不要 markdown code block 标记, 直接写代码)"""


# ════════════════════════════════════════
# 代码修正 Prompt
# ════════════════════════════════════════

VERIFICATION_CODE_FIX_PROMPT = """验证脚本执行失败, 请修正代码。

## 原始脚本
```python
{original_code}
```

## 执行错误
```
{error_output}
```

## 修正要求
1. 分析错误原因, 修正代码
2. 保持相同的数据加载和统计逻辑
3. 仍然将结果写入 `{output_file_path}`
4. 如果数据文件不存在, 用 try/except 处理并写入 error 信息到 JSON
5. 不要使用 print() 输出中间结果

只输出修正后的完整 Python 脚本代码 (不要解释文字, 不要 markdown 标记)"""


# ════════════════════════════════════════
# 结果分析 Prompt
# ════════════════════════════════════════

RESULT_ANALYSIS_PROMPT = """你是一位严谨的数据科学家, 正在分析验证结果以判断假设是否成立。

## 假设
- ID: {hypothesis_id}
- Claim: {hypothesis_claim}
- 如果成立应观察到的现象: {expected_if_true}
- 如果不成立应观察到的现象: {expected_if_false}

## 验证方案
```json
{verification_plan_json}
```

## 验证代码执行结果
```json
{execution_result_json}
```

## 判断标准
- **CONFIRMED**: 数据结果与"假设成立应观察到的现象"高度吻合
- **PARTIALLY_CONFIRMED**: 数据结果部分吻合, 但差异不够显著或存在混杂因素
- **REFUTED**: 数据结果与"假设不成立应观察到的现象"吻合, 或与假设方向相反
- **UNVERIFIABLE**: 数据缺失或统计方法无法得出有效结论

## 任务
根据验证结果, 给出你的判断。必须:
1. 解释数据结果的具体含义 (用通俗语言)
2. 对比数据结果与期望现象
3. 给出明确的判断 (不能模棱两可)
4. 如果判断为 PARTIALLY_CONFIRMED, 说明哪些方面吻合、哪些不吻合

### 输出格式

```json
{{
  "hypothesis_id": "{hypothesis_id}",
  "status": "CONFIRMED | PARTIALLY_CONFIRMED | REFUTED | UNVERIFIABLE",
  "brief": "简洁的判断摘要 (一句话)",
  "detailed_reasoning": "详细的推理过程 (为什么做出这个判断)",
  "evidence_summary": {{
    "key_statistic": "最关键的统计量及其数值",
    "comparison": "数据结果与期望现象的对比",
    "confidence": "判断的置信度 (0.0-1.0)"
  }},
  "limitations": "验证的局限性或混杂因素 (如有)"
}}
```"""


# ════════════════════════════════════════
# JSON 修正 Prompt (假设提取阶段重试用)
# ════════════════════════════════════════

HYPOTHESIS_JSON_FIX_PROMPT = """你之前提取的假设 JSON 格式有误, 请根据以下**原始输出**和**解析错误**信息, 重新输出**正确的 JSON**。

## 你之前的原始输出 (RAW)
```
{raw_response_truncated}
```

## 解析错误
{parse_error}

## 修正要求
1. 保持假设内容不变, 只修复 JSON 格式
2. 确保是**严格合法的 JSON** (双引号, 无结尾逗号, 无注释)
3. 保持 {{
  "hypotheses": [...],
  "summary": "..."
}} 格式
4. **只输出 JSON**, 不要解释文字, 不要 markdown 标记

## 部分有效假设 (如果 JSON 中有部分解析成功的假设, 保留它们)
{partial_hypotheses_text}"""


# ════════════════════════════════════════
# 通用 JSON 修正 Prompt (验证流程各阶段重试用)
# ════════════════════════════════════════

JSON_FIX_PROMPT_TEMPLATE = """你之前输出的 JSON 格式有误, 请根据以下**原始输出**和**解析错误**信息, 重新输出**正确的 JSON**。

## 你之前的原始输出 (RAW)
```
{raw_response_truncated}
```

## 解析错误
{parse_error}

## 修正要求
1. 保持内容不变, 只修复 JSON 格式
2. 确保是**严格合法的 JSON** (双引号, 无结尾逗号, 无注释)
3. **只输出 JSON**, 不要解释文字, 不要 markdown 标记
{additional_instructions}"""


class DataInventory:
    """
    数据盘点器 — 发现项目中可用的数据资源
    
    探索:
    - 训练/测试数据文件
    - 元数据文件 (id_meta_data.json 等)
    - 已计算好的统计量 (item_popularity 等)
    - 训练日志中的指标
    - 模型 checkpoint 信息
    
    新增能力 (v2):
    - 识别假设所需但尚不存在的数据
    - 通过 DataComputationEngine 自动计算所需的派生数据
    """
    
    def __init__(self, project_root: str, data_dir: str = None, log_dir: str = None):
        self.project_root = project_root
        self.data_dir = data_dir or os.path.join(project_root, "Recmodel", "data")
        self.log_dir = log_dir or os.path.join(project_root, "logs")
        self._inventory = None
        self._computation_engine = None  # lazy init
    
    def discover(self) -> Dict:
        """
        发现所有可用数据资源
        
        Returns:
            Dict containing:
            - data_files: 可用数据文件列表及描述
            - computed_stats: 已计算好的统计量
            - metadata_files: 元数据文件
        """
        if self._inventory is not None:
            return self._inventory
        
        inventory = {
            "data_files": [],
            "computed_stats": [],
            "metadata_files": [],
            "available_variables": [],
        }
        
        # --- 扫描数据目录 ---
        if os.path.exists(self.data_dir):
            for fname in os.listdir(self.data_dir):
                fpath = os.path.join(self.data_dir, fname)
                if not os.path.isfile(fpath):
                    continue
                desc = self._describe_data_file(fname, fpath)
                inventory["data_files"].append(desc)
        
        # --- 扫描日志目录 (已计算的统计量) ---
        if os.path.exists(self.log_dir):
            for fname in os.listdir(self.log_dir):
                if fname.startswith("item_popularity") and fname.endswith(".json"):
                    inventory["computed_stats"].append({
                        "name": "item_popularity",
                        "path": os.path.join(self.log_dir, fname),
                        "description": "物品热度分布 (item_id → interaction_count)",
                        "format": "Dict[str, int]",
                    })
                if fname.startswith("verification") and fname.endswith(".json"):
                    # 之前迭代已有的验证报告
                    inventory["computed_stats"].append({
                        "name": "previous_verification",
                        "path": os.path.join(self.log_dir, fname),
                        "description": "之前迭代的假设验证报告",
                    })
        
        # --- 扫描项目根目录中的关键文件 ---
        for fname in os.listdir(self.project_root):
            fpath = os.path.join(self.project_root, fname)
            if fname.endswith(".py") and fname.startswith("run_"):
                inventory["available_variables"].append({
                    "name": fname,
                    "description": f"训练脚本: {fname}",
                    "type": "script",
                })
        
        self._inventory = inventory
        return inventory
    
    def _describe_data_file(self, fname: str, fpath: str) -> Dict:
        """描述一个数据文件"""
        size_kb = os.path.getsize(fpath) / 1024
        
        desc = {
            "name": fname,
            "path": fpath,
            "size_kb": round(size_kb, 1),
        }
        
        # 根据文件名推断类型
        if "train" in fname.lower():
            desc["description"] = "训练数据 (用户-物品交互序列)"
            desc["format"] = "每行一个用户序列: item_id item_id ..."
            desc["can_load"] = True
        elif "test" in fname.lower():
            desc["description"] = "测试数据"
            desc["format"] = "每行一个用户序列"
            desc["can_load"] = True
        elif "val" in fname.lower():
            desc["description"] = "验证数据"
            desc["format"] = "每行一个用户序列"
            desc["can_load"] = True
        elif fname.endswith(".json") and "meta" in fname.lower():
            desc["description"] = "物品元数据 (ID → 类别/标题/描述)"
            desc["format"] = "JSON: item_id → {title, categories, ...}"
            desc["can_load"] = True
        elif fname.endswith(".json"):
            desc["description"] = "JSON 数据文件"
            desc["format"] = "JSON"
            desc["can_load"] = True
        elif fname.endswith(".py"):
            desc["description"] = f"Python 脚本: {fname}"
            desc["format"] = "Python source"
            desc["can_load"] = False
        else:
            desc["description"] = f"数据文件: {fname}"
            desc["can_load"] = True
        
        return desc
    
    def format_inventory_for_prompt(self) -> str:
        """格式化数据盘点结果为 LLM prompt 文本"""
        inventory = self.discover()
        
        lines = []
        if inventory["data_files"]:
            lines.append("### 数据文件")
            for df in inventory["data_files"]:
                lines.append(f"- **{df['name']}** ({df['size_kb']}KB): {df['description']}")
                if df.get('format'):
                    lines.append(f"  格式: {df['format']}")
                lines.append(f"  路径: `{df['path']}`")
        
        if inventory["computed_stats"]:
            lines.append("\n### 已计算统计量")
            for cs in inventory["computed_stats"]:
                lines.append(f"- **{cs['name']}**: {cs['description']}")
                lines.append(f"  路径: `{cs['path']}`")
        
        return "\n".join(lines) if lines else "暂无可用数据资源信息"
    
    def load_data_for_verification(self, data_needed: List[str]) -> Dict:
        """
        根据验证需要的数据列表, 尝试加载可用数据
        
        Args:
            data_needed: 验证方案中声明需要的数据列表
            
        Returns:
            Dict of loaded data, keyed by data type
        """
        loaded = {}
        inventory = self.discover()
        
        # 加载物品热度
        if any(kw in " ".join(data_needed).lower() for kw in 
               ["热度", "popularity", "频次", "交互次数", "频次分布"]):
            for cs in inventory["computed_stats"]:
                if cs["name"] == "item_popularity":
                    try:
                        with open(cs["path"], 'r', encoding='utf-8') as f:
                            loaded["item_popularity"] = json.load(f)
                        logger.info(f"Loaded item_popularity: {len(loaded['item_popularity'])} items")
                    except Exception as e:
                        logger.warning(f"Failed to load item_popularity: {e}")
        
        # 加载元数据
        if any(kw in " ".join(data_needed).lower() for kw in
               ["元数据", "metadata", "类别", "category", "标题", "title", "描述", "description"]):
            for df in inventory["data_files"]:
                if df["name"].endswith(".json") and "meta" in df["name"].lower():
                    try:
                        with open(df["path"], 'r', encoding='utf-8') as f:
                            loaded["item_metadata"] = json.load(f)
                        logger.info(f"Loaded item_metadata: {len(loaded['item_metadata'])} items")
                    except Exception as e:
                        logger.warning(f"Failed to load metadata: {e}")
        
        # 加载训练数据 (用于计算统计量)
        if any(kw in " ".join(data_needed).lower() for kw in
               ["训练数据", "train data", "交互序列", "用户序列", "序列长度", "sequence length"]):
            for df in inventory["data_files"]:
                if "train" in df["name"].lower() and df.get("can_load"):
                    # 训练数据可能太大, 只加载统计摘要
                    loaded["train_data_path"] = df["path"]
                    loaded["train_data_info"] = {
                        "path": df["path"],
                        "size_kb": df["size_kb"],
                        "name": df["name"],
                    }
                    break
        
        # 加载测试数据
        if any(kw in " ".join(data_needed).lower() for kw in
               ["测试数据", "test data", "误推", "wrong prediction", "错误案例"]):
            for df in inventory["data_files"]:
                if "test" in df["name"].lower() and df.get("can_load"):
                    loaded["test_data_path"] = df["path"]
                    break
        
        return loaded
    
    def format_loaded_data_summary(self, loaded: Dict) -> str:
        """格式化已加载的数据摘要为 LLM prompt 文本"""
        lines = []
        for key, value in loaded.items():
            if isinstance(value, dict) and "path" in value:
                lines.append(f"- **{key}**: 文件路径 `{value['path']}`, 大小 {value.get('size_kb', '?')}KB")
            elif isinstance(value, dict):
                lines.append(f"- **{key}**: {len(value)} 条记录 (已加载到内存)")
                # 展示一条样本
                if value:
                    sample_key = list(value.keys())[0]
                    sample_val = value[sample_key]
                    if isinstance(sample_val, dict):
                        lines.append(f"  样本: key={sample_key}, value={json.dumps(sample_val, ensure_ascii=False)[:100]}")
                    else:
                        lines.append(f"  样本: key={sample_key}, value={sample_val}")
            elif isinstance(value, str):
                lines.append(f"- **{key}**: `{value}`")
            else:
                lines.append(f"- **{key}**: {type(value).__name__}")
        
        return "\n".join(lines) if lines else "暂无已加载的数据"
    
    def get_computation_engine(self, llm_client=None):
        """获取数据计算引擎 (lazy init)"""
        if self._computation_engine is None:
            self._computation_engine = DataComputationEngine(
                project_root=self.project_root,
                data_dir=self.data_dir,
                log_dir=self.log_dir,
                llm_client=llm_client,
            )
        elif llm_client is not None and self._computation_engine.llm is None:
            self._computation_engine.llm = llm_client
        return self._computation_engine
    
    _model_probing_engine = None  # lazy init
    
    def get_model_probing_engine(self, llm_client=None):
        """获取模型探测引擎 (lazy init)"""
        if self._model_probing_engine is None:
            self._model_probing_engine = ModelProbingEngine(
                project_root=self.project_root,
                data_dir=self.data_dir,
                log_dir=self.log_dir,
                llm_client=llm_client,
            )
        elif llm_client is not None and self._model_probing_engine.llm is None:
            self._model_probing_engine.llm = llm_client
        return self._model_probing_engine
    
    def identify_missing_data(self, data_needed: List[str], loaded: Dict) -> List[str]:
        """
        识别假设需要但尚未加载/可用的数据
        
        Args:
            data_needed: 假设声明需要的数据列表
            loaded: 已加载的数据
            
        Returns:
            List of data descriptions that are still missing
        """
        missing = []
        loaded_keys = set(loaded.keys())
        data_text = " ".join(data_needed).lower()
        
        # 检查类别重叠数据
        if any(kw in data_text for kw in 
               ["类别重叠", "category overlap", "跨类别", "类别交集", "类别相关性", "类别匹配"]):
            if "category_overlap_stats" not in loaded_keys:
                missing.append("category_overlap_stats: 目标物品与用户历史序列的类别重叠统计")
        
        # 检查类别分布数据
        if any(kw in data_text for kw in
               ["类别分布", "category distribution", "类别偏差", "类别集中", "类别占比"]):
            if "category_distribution" not in loaded_keys and "item_metadata" not in loaded_keys:
                missing.append("category_distribution: 训练数据中各类别的物品分布")
        
        # 检查误推案例类别数据
        if any(kw in data_text for kw in
               ["误推类别", "error category", "错误案例类别", "误推目标类别"]):
            if "wrong_case_category_stats" not in loaded_keys:
                missing.append("wrong_case_category_stats: 误推案例中目标物品的类别分布")
        
        # 检查推荐频率数据
        if any(kw in data_text for kw in
               ["推荐频率", "recommendation frequency", "推荐频次", "推荐列表", "top-k", "topk",
                "模型推荐结果", "prediction frequency", "推荐占比"]):
            if "recommendation_frequency" not in loaded_keys:
                missing.append("recommendation_frequency: 模型推荐结果中各物品的推荐频次")
        
        # 检查序列-目标关联数据
        if any(kw in data_text for kw in
               ["历史序列", "序列与目标", "序列关联", "历史行为", "用户历史", "序列模式"]):
            if "sequence_target_mapping" not in loaded_keys and "wrong_text_cases" not in loaded_keys:
                missing.append("sequence_target_mapping: 用户历史序列与目标物品的关联数据")
        
        # 检查物品交互频率 (训练数据)
        if any(kw in data_text for kw in
               ["交互频率", "interaction frequency", "训练交互", "物品频率", "真实交互频率"]):
            if "item_interaction_freq" not in loaded_keys and "item_popularity" not in loaded_keys:
                missing.append("item_interaction_freq: 训练数据中各物品的交互频率统计")
        
        # ─── 模型内部数据检测 (新增) ───
        # 注意力权重 / 注意力分布
        if any(kw in data_text for kw in
               ["注意力权重", "attention weights", "attention_probs", "注意力分布",
                "注意力坍缩", "attention collapse", "注意力集中",
                "注意力熵", "attention entropy",
                "注意力重心", "attention centroid",
                "注意力矩阵", "attention matrix"]):
            if "attention_weights" not in loaded_keys:
                missing.append("attention_weights: 模型自注意力权重矩阵 (需通过模型探测提取)")
        
        # 模型隐藏状态 / 编码器输出
        if any(kw in data_text for kw in
               ["隐藏状态", "hidden states", "编码器输出", "encoder output",
                "中间表示", "intermediate representation",
                "序列输出", "sequence output", "item encoded layers"]):
            if "hidden_states" not in loaded_keys:
                missing.append("hidden_states: Transformer编码器各层的隐藏状态输出 (需通过模型探测提取)")
        
        # 物品嵌入向量
        if any(kw in data_text for kw in
               ["物品嵌入", "item embeddings", "嵌入向量", "embedding weights",
                "嵌入表示", "embedding representation", "物品嵌入矩阵"]):
            if "item_embeddings" not in loaded_keys:
                missing.append("item_embeddings: 物品嵌入向量矩阵 (需通过模型探测提取)")
        
        # 模型预测 / 推荐分数
        if any(kw in data_text for kw in
               ["模型预测分数", "prediction scores", "推荐分数", "模型打分",
                "预测概率", "prediction probability", "模型推理结果",
                "模型输出分数", "model scores"]):
            if "model_predictions" not in loaded_keys:
                missing.append("model_predictions: 模型对物品的预测分数 (需通过模型探测提取)")
        
        # 梯度信息
        if any(kw in data_text for kw in
               ["梯度", "gradient", "梯度分布", "梯度范数", "gradient norm"]):
            if "gradient_info" not in loaded_keys:
                missing.append("gradient_info: 模型梯度信息 (需通过模型探测提取)")
        
        return missing
    
    def identify_model_internal_data(self, missing_data: List[str]) -> List[str]:
        """
        从缺失数据列表中筛选出需要模型探测的数据
        
        Args:
            missing_data: identify_missing_data() 返回的缺失数据列表
            
        Returns:
            List of data descriptions that require model probing
        """
        model_internal = []
        for desc in missing_data:
            if "需通过模型探测提取" in desc or "模型自注意力" in desc or \
               "Transformer编码器" in desc or "嵌入向量矩阵" in desc or \
               "模型对物品的预测分数" in desc or "模型梯度信息" in desc:
                model_internal.append(desc)
        return model_internal
    
    def identify_computable_data(self, missing_data: List[str]) -> List[str]:
        """
        从缺失数据列表中筛选出可以通过数据计算获取的数据
        
        Args:
            missing_data: identify_missing_data() 返回的缺失数据列表
            
        Returns:
            List of data descriptions that can be computed from existing data
        """
        return [desc for desc in missing_data if desc not in self.identify_model_internal_data(missing_data)]


# ════════════════════════════════════════
# 模型探测 Prompt (用于生成模型内部数据提取脚本)
# ════════════════════════════════════════

MODEL_PROBING_ANALYSIS_PROMPT = """你是一位深度学习工程师，正在分析一个推荐系统模型的内部结构，以确定如何提取假设验证所需的模型内部数据。

## 需要提取的数据
{missing_data_description}

## 假设背景
- 假设 ID: {hypothesis_id}
- 假设内容: {hypothesis_claim}
- 验证思路: {verification_thought}

## 模型源码

### models.py
```python
{models_source}
```

### modules.py (包含 SelfAttention、Encoder 等核心组件)
```python
{modules_source}
```

## 分析任务
请分析模型源码，确定:
1. **需要 hook 的模块和类** — 哪个类/方法包含了我们需要的数据 (如 SelfAttention 中的 attention_probs)
2. **hook 类型** — forward hook (捕获输出) 还是直接读取模型参数 (如 item_embeddings.weight)
3. **数据提取方法** — 如何从模型中提取指定的数据:
   - 对于注意力权重: 在 SelfAttention 上注册 forward hook，在 hook 函数中捕获 attention_probs
   - 对于嵌入向量: 直接读取 model.item_embeddings.weight
   - 对于隐藏状态: 在 EncoderLayer 上注册 forward hook
   - 对于模型预测: 执行完整的 forward pass
4. **需要加载的模块路径** — 模型类所在的文件路径 (如 "models.py" → "SASRec")

请输出 JSON 格式的分析结果:
```json
{{
  "target_modules": [
    {{
      "class_name": "SelfAttention",
      "file_path": "modules.py",
      "hook_type": "forward_hook",
      "description": "捕获注意力概率矩阵 attention_probs",
      "data_key": "attention_weights"
    }}
  ],
  "model_class": {{
    "name": "SASRec",
    "file_path": "models.py",
    "init_args_needed": ["item_size", "hidden_size", "num_attention_heads", ...],
    "finetune_method": "finetune(input_ids) → sequence_output"
  }},
  "data_extraction_strategy": "使用 PyTorch register_forward_hook 在 SelfAttention 的 forward 方法上注册钩子，捕获经过 softmax 后的 attention_probs 张量。将 attention_probs 转为 numpy 并保存。",
  "additional_notes": "attention_probs 的形状为 [batch_size, num_heads, seq_len, seq_len]，需要用 .detach().cpu().numpy() 转换"
}}
```

只输出 JSON，不要解释文字。"""

MODEL_PROBING_SCRIPT_PROMPT = """你是一位深度学习工程师，正在编写一个**模型探测脚本**来提取假设验证所需的模型内部数据。

## 需要提取的数据
{missing_data_description}

## 假设背景
- 假设 ID: {hypothesis_id}
- 假设内容: {hypothesis_claim}
- 验证思路: {verification_thought}

## 模型分析结果 (已由上一步分析完成)
{model_analysis_json}

## 项目环境信息
- 项目根目录: `{project_root}`
- 模型 checkpoint 路径: `{checkpoint_path}`
- 数据目录: `{data_dir}`
- 模型参数: {model_args}

## 模型源码

### modules.py
```python
{modules_source}
```

## 重要要求
1. 代码必须是**独立的 Python 脚本**，可以直接运行
2. 使用 `sys.path.insert(0, '{recmodel_dir}')` 将 Recmodel 目录加入搜索路径
3. 使用 PyTorch 的 `register_forward_hook` 来捕获模型内部数据，**不要修改原始模型源码**
4. 对于注意力权重等中间计算结果，在 hook 函数中用 `output[1]` (hook 返回的第二个参数是模块的输出，第三个参数是模块forward的输出) 来捕获
5. 注意: PyTorch 的 forward_hook 签名是 `hook(module, input, output)`, 其中 output 是模块 forward() 的返回值
6. **对于 SelfAttention 类**: 它的 forward() 方法只返回 `hidden_states`，不返回 `attention_probs`。因此需要用**替代方案**:
   - 方案A: 在 SelfAttention.forward 中，attention_probs 在 Softmax 之后计算但作为中间变量使用。由于 forward 只返回 hidden_states，forward_hook 无法直接获取 attention_probs。
   - 方案B (推荐): **创建一个临时修改版 SelfAttention**，让 forward() 同时返回 attention_probs 和 hidden_states。具体做法:
     1. 将 modules.py 复制到临时目录
     2. 修改 SelfAttention.forward() 使其返回 `(hidden_states, attention_probs)` 而非仅 `hidden_states`
     3. 修改 EncoderLayer.forward() 使其传递 attention_probs
     4. 修改 Encoder.forward() 使其收集并返回所有层的 attention_probs
     5. 修改 SASRec.finetune() 使其返回 `(sequence_output, all_attention_probs)`
     6. 从临时目录导入修改后的模型
   - 方案C: 用 `torch.nn.Module.register_forward_pre_hook` 结合手动重写来获取中间变量
7. 加载 checkpoint 时使用: `model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))`
8. 使用测试数据 (`{data_dir}/Beauty_test.txt`) 创建输入序列
9. 只在少量样本上运行推理 (最多 50 个用户)，避免内存溢出
10. 将提取的数据以 JSON 格式保存到 `{output_file_path}`
11. 结果 JSON 格式:
```json
{{
  "computed_data_name": "数据名称",
  "description": "提取数据的描述",
  "data": {{
    // 具体的提取结果 — 必须是可序列化的 JSON
    // 对于注意力权重，提供统计摘要 (熵、重心位置、头部权重等)
    // 不要直接存储原始注意力矩阵 (太大)，只存储分析结果
  }},
  "statistics": {{
    // 关键统计摘要 (用于 LLM 快速理解数据)
    // 如: mean_entropy, mean_centroid_position, tail_weight_ratio 等
  }},
  "sample": {{
    // 1-3 条数据样本
  }},
  "computation_method": "使用 PyTorch forward_hook + 临时模型修改提取注意力权重",
  "data_record_count": 采样数量
}}
```

## 特别提示 — 如何临时修改模型以提取注意力权重

这是一个关键步骤。示例代码框架:

```python
import sys, os, shutil, importlib

# Step 1: 复制 modules.py 到临时目录并修改
recmodel_dir = "{recmodel_dir}"
temp_dir = os.path.join(os.path.dirname(output_file), "_temp_model_probe")
os.makedirs(temp_dir, exist_ok=True)

# 复制 modules.py
shutil.copy2(os.path.join(recmodel_dir, "modules.py"), temp_dir)

# 读取并修改 modules.py
with open(os.path.join(temp_dir, "modules.py"), 'r') as f:
    modules_code = f.read()

# 修改 SelfAttention.forward 使其返回 attention_probs
# 在 "return hidden_states" 之前，将 attention_probs 保存
# 修改为: return (hidden_states, attention_probs)

# ... 对 EncoderLayer.forward, Encoder.forward, SASRec.finetune 的类似修改 ...

sys.path.insert(0, temp_dir)
# 先导入修改后的 modules
import modules as mod_modules
sys.modules['modules'] = mod_modules
# 然后导入 models
# 需要先复制 models.py 到 temp_dir 并修改导入路径
shutil.copy2(os.path.join(recmodel_dir, "models.py"), temp_dir)
import models as mod_models
```

只输出完整的 Python 脚本代码 (不要解释文字, 不要 markdown 标记)"""

MODEL_PROBING_FIX_PROMPT = """模型探测脚本执行失败, 请修正代码。

## 原始代码
```python
{original_code}
```

## 错误信息
{error_message}

## 修正要求
1. 仔细分析错误原因
2. 修正代码中的 bug
3. 特别注意:
   - 模块导入路径是否正确
   - 临时修改的 modules.py 是否正确保存
   - checkpoint 是否能正确加载
   - 数据路径是否正确
   - torch hook 是否正确注册
   - 内存是否溢出 (减少样本数量)
4. 输出修正后的完整 Python 脚本 (不要只输出 diff)
"""

# ════════════════════════════════════════
# 数据计算 Prompt (用于生成数据计算脚本)
# ════════════════════════════════════════

DATA_COMPUTATION_PROMPT = """你是一位 Python 数据工程师，正在编写一个数据计算脚本来生成假设验证所需的派生数据。

## 需要计算的数据
{missing_data_description}

## 原始数据源 (可以直接读取)
以下数据文件在项目中可用，你的脚本可以直接打开并读取:

{raw_data_sources}

## 假设背景
- 假设 ID: {hypothesis_id}
- 假设内容: {hypothesis_claim}
- 验证思路: {verification_thought}
- 需要的数据: {data_needed}

## 输出要求
1. 代码必须是**独立的 Python 脚本**
2. 只依赖标准库和 numpy、json (不要使用 scipy 或其他特殊库)
3. 结果必须以 JSON 格式写入 `{output_file_path}`
4. JSON 输出格式:
```json
{{
  "computed_data_name": "数据名称",
  "description": "计算结果的描述",
  "data": {{
    // 具体的计算结果 — 必须是可序列化的 JSON (Dict, List, 数值)
    // 对于大字典，最多保留 5000 条记录 (超过的截断)
  }},
  "statistics": {{
    // 关键统计摘要 (用于 LLM 快速理解数据)
  }},
  "sample": {{
    // 1-3 条数据样本 (用于 LLM 理解数据格式)
  }},
  "computation_method": "计算方法的简要描述",
  "data_record_count": 1234  // 数据记录总数
}}
```
5. 如果某个原始数据文件不存在或读取失败，用 try/except 处理，在结果中用 error 字段说明
6. 对于大字典数据 (>5000条)，只保留前 5000 条并标注截断
7. 不要使用 print() 输出中间结果

## 重要提示
- 训练数据格式: 每行一个用户交互序列，物品ID以空格分隔 (如: "1 2 3 4 5")
- 测试数据格式: 同上
- 元数据格式: JSON 文件，key为物品ID(str), value为 {{title, categories, description, asin}}
- categories 字段格式: "大类 > 子类1 > 子类2" (用 > 分隔的层级路径)

只输出完整的 Python 脚本代码 (不要解释文字, 不要 markdown 标记)"""


class DataComputationEngine:
    """
    数据计算引擎 — 编写和执行程序来生成验证所需的派生数据
    
    核心设计:
    1. **内置计算**: 提供常见派生数据的快速计算方法 (不需要 LLM)
       - 物品交互频率 (从训练数据统计)
       - 类别重叠统计 (从误推案例+元数据计算)
       - 类别分布统计
       - 推荐频率统计 (从误推案例的预测列表统计)
    
    2. **LLM 计算**: 对于复杂/新颖的数据需求，让 LLM 编写计算脚本并执行
    
    3. **缓存**: 计算结果保存为 JSON 文件，避免重复计算
    
    工作流程:
    a. identify_missing_data() → 发现缺失的数据
    b. try_builtin_computation() → 尝试内置方法快速计算
    c. compute_with_llm() → 如果内置方法无法满足，让 LLM 写计算脚本
    d. execute_computation_script() → 执行计算脚本 (带修正循环)
    """
    
    MAX_COMPUTE_TIMEOUT = 120  # 计算脚本执行超时 (秒)
    MAX_COMPUTE_FIX_ROUNDS = 2  # 计算脚本修正最大轮数
    
    def __init__(self, project_root: str, data_dir: str = None,
                 log_dir: str = None, llm_client=None):
        self.project_root = project_root
        self.data_dir = data_dir or os.path.join(project_root, "Recmodel", "data")
        self.log_dir = log_dir or os.path.join(project_root, "logs")
        self.llm = llm_client  # 可选 — 内置方法不需要 LLM
        
        # 计算结果缓存目录
        self.cache_dir = os.path.join(self.log_dir, "verification_scripts", "computed_data")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 已计算的数据 (内存缓存)
        self._computed_cache = {}
    
    def compute_needed_data(self, missing_data: List[str],
                            hypothesis: Dict,
                            preloaded_data: Dict) -> Dict:
        """
        计算假设验证所需的派生数据
        
        流程:
        1. 尝试内置计算方法
        2. 内置方法无法满足的，使用 LLM 编写计算脚本
        3. 执行脚本获取结果
        
        Args:
            missing_data: DataInventory.identify_missing_data() 返回的缺失数据列表
            hypothesis: 当前假设
            preloaded_data: 已加载的数据 (可作为计算输入)
            
        Returns:
            Dict of computed data, keyed by data name
        """
        computed = {}
        remaining_missing = []
        
        # --- Phase 1: 尝试内置计算 ---
        for missing_desc in missing_data:
            data_name = missing_desc.split(":")[0].strip() if ":" in missing_desc else missing_desc.strip()
            
            # 检查缓存
            if data_name in self._computed_cache:
                computed[data_name] = self._computed_cache[data_name]
                logger.info(f"Using cached computed data: {data_name}")
                continue
            
            # 检查文件缓存
            cached_file = os.path.join(self.cache_dir, f"{data_name}.json")
            if os.path.exists(cached_file):
                try:
                    with open(cached_file, 'r', encoding='utf-8') as f:
                        cached_result = json.load(f)
                    computed[data_name] = cached_result.get("data", cached_result)
                    self._computed_cache[data_name] = computed[data_name]
                    logger.info(f"Using file-cached computed data: {data_name}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached {data_name}: {e}")
            
            # 尝试内置方法
            builtin_result = self._try_builtin_computation(
                data_name, hypothesis, preloaded_data
            )
            if builtin_result is not None:
                computed[data_name] = builtin_result
                self._computed_cache[data_name] = builtin_result
                # 保存到文件缓存
                self._save_computed_to_cache(data_name, builtin_result)
                logger.info(f"Computed {data_name} with builtin method")
            else:
                remaining_missing.append(missing_desc)
        
        # --- Phase 2: LLM 计算 ---
        if remaining_missing and self.llm is not None:
            for missing_desc in remaining_missing:
                data_name = missing_desc.split(":")[0].strip() if ":" in missing_desc else missing_desc.strip()
                
                llm_result = self._compute_with_llm(
                    missing_desc, hypothesis, preloaded_data
                )
                if llm_result is not None:
                    computed[data_name] = llm_result
                    self._computed_cache[data_name] = llm_result
                    self._save_computed_to_cache(data_name, llm_result)
                    logger.info(f"Computed {data_name} with LLM-generated script")
                else:
                    logger.warning(f"Failed to compute {data_name} (both builtin and LLM methods failed)")
        
        if remaining_missing and self.llm is None:
            logger.warning(f"No LLM available to compute: {remaining_missing}")
        
        return computed
    
    # ════════════════════════════════════════
    # 内置计算方法 (不需要 LLM)
    # ════════════════════════════════════════
    
    def _try_builtin_computation(self, data_name: str,
                                  hypothesis: Dict,
                                  preloaded_data: Dict) -> Optional[Dict]:
        """
        尝试用内置方法计算所需的派生数据
        
        Args:
            data_name: 需要的数据名称 (如 "category_overlap_stats")
            hypothesis: 当前假设
            preloaded_data: 已加载的数据
            
        Returns:
            Computed data dict, or None if builtin method can't handle this
        """
        # --- 物品交互频率 (从训练数据统计) ---
        if data_name in ["item_interaction_freq", "item_popularity_from_train"]:
            return self._compute_item_interaction_freq(preloaded_data)
        
        # --- 类别重叠统计 ---
        if data_name == "category_overlap_stats":
            return self._compute_category_overlap_stats(hypothesis, preloaded_data)
        
        # --- 类别分布统计 ---
        if data_name in ["category_distribution", "category_stats"]:
            return self._compute_category_distribution(preloaded_data)
        
        # --- 误推案例类别统计 ---
        if data_name == "wrong_case_category_stats":
            return self._compute_wrong_case_category_stats(hypothesis, preloaded_data)
        
        # --- 推荐频率统计 ---
        if data_name == "recommendation_frequency":
            return self._compute_recommendation_frequency(preloaded_data)
        
        # --- 序列-目标关联数据 ---
        if data_name == "sequence_target_mapping":
            return self._compute_sequence_target_mapping(preloaded_data)
        
        # 未知数据类型 → 返回 None (需要 LLM 计算)
        logger.info(f"No builtin computation for: {data_name}")
        return None
    
    def _compute_item_interaction_freq(self, preloaded_data: Dict) -> Optional[Dict]:
        """
        从训练数据计算物品交互频率
        
        读取训练数据文件，统计每个物品在所有用户序列中出现的次数
        """
        train_path = preloaded_data.get("train_data_path")
        if not train_path:
            # 尝试从数据目录找到训练文件
            if os.path.exists(self.data_dir):
                for fname in os.listdir(self.data_dir):
                    if "train" in fname.lower() and fname.endswith(".txt"):
                        train_path = os.path.join(self.data_dir, fname)
                        break
        
        if not train_path or not os.path.exists(train_path):
            logger.warning("No train data file found for item_interaction_freq computation")
            return None
        
        try:
            item_freq = Counter()
            total_sequences = 0
            
            with open(train_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    items = line.split()
                    for item_id in items:
                        item_freq[item_id] += 1
                    total_sequences += 1
            
            # 转换为普通 dict (保留数值为 int)
            freq_dict = {k: int(v) for k, v in item_freq.items()}
            
            # 统计摘要
            freq_values = list(freq_dict.values())
            stats = {
                "total_items": len(freq_dict),
                "total_sequences": total_sequences,
                "mean_freq": round(sum(freq_values) / len(freq_values), 2) if freq_values else 0,
                "median_freq": round(sorted(freq_values)[len(freq_values)//2], 2) if freq_values else 0,
                "max_freq": max(freq_values) if freq_values else 0,
                "min_freq": min(freq_values) if freq_values else 0,
                "cold_threshold_5": sum(1 for v in freq_values if v < 5),
                "cold_threshold_10": sum(1 for v in freq_values if v < 10),
            }
            
            # 样本
            top_items = item_freq.most_common(3)
            sample = {item_id: count for item_id, count in top_items}
            
            return {
                "freq_dict": freq_dict,
                "statistics": stats,
                "sample": sample,
                "computation_method": "builtin: count item occurrences in training sequences",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute item_interaction_freq: {e}")
            return None
    
    def _compute_category_overlap_stats(self, hypothesis: Dict,
                                         preloaded_data: Dict) -> Optional[Dict]:
        """
        计算目标物品与用户历史序列的类别重叠统计
        
        对于每个误推/正确推荐案例:
        - 获取目标物品的类别
        - 获取历史序列中所有物品的类别集合
        - 计算重叠率 (目标类别与历史类别的交集大小 / 历史类别总数)
        """
        wrong_cases = preloaded_data.get("wrong_text_cases", [])
        item_metadata = preloaded_data.get("item_text_map") or preloaded_data.get("item_metadata")
        
        if not wrong_cases:
            logger.warning("No wrong cases available for category_overlap computation")
            return None
        
        if not item_metadata:
            # 尝试加载元数据
            item_metadata = self._load_metadata()
            if not item_metadata:
                logger.warning("No item metadata available for category_overlap computation")
                return None
        
        try:
            def get_top_category(meta_entry):
                """提取物品的最顶层类别"""
                if isinstance(meta_entry, dict):
                    cat = meta_entry.get("categories", "")
                    if cat and ">" in cat:
                        return cat.split(">")[0].strip()
                    elif cat:
                        return cat.strip()
                elif isinstance(meta_entry, str):
                    # 旧版格式: 纯文本字符串
                    return meta_entry.strip()
                return "unknown"
            
            def get_all_categories(meta_entry):
                """提取物品的所有类别层级"""
                if isinstance(meta_entry, dict):
                    cat = meta_entry.get("categories", "")
                    if cat and ">" in cat:
                        return [c.strip() for c in cat.split(">")]
                    elif cat:
                        return [cat.strip()]
                elif isinstance(meta_entry, str):
                    return [meta_entry.strip()]
                return []
            
            # 计算每个误推案例的类别重叠
            overlap_results = []
            no_overlap_count = 0
            partial_overlap_count = 0
            full_overlap_count = 0
            
            for case in wrong_cases:
                target_id = str(case.get("target_id", ""))
                target_meta = item_metadata.get(target_id, {})
                target_cat = get_top_category(target_meta)
                target_all_cats = set(get_all_categories(target_meta))
                
                # 获取历史序列物品的类别集合
                history_cats = set()
                history_ids = case.get("predictions_ids", [])
                # 有些案例用 history_text 提取
                if not history_ids:
                    # 从 history_text 推断
                    history_text = case.get("history_text", [])
                    for h_text in history_text:
                        if isinstance(h_text, str) and "[" in h_text:
                            # 格式: "Title [Category]"
                            cat_part = h_text.split("[")[-1].rstrip("]").strip()
                            history_cats.add(cat_part)
                else:
                    for hid in history_ids:
                        h_meta = item_metadata.get(str(hid), {})
                        h_cats = get_all_categories(h_meta)
                        history_cats.update(h_cats)
                
                # 计算重叠
                overlap_cats = target_all_cats & history_cats
                overlap_size = len(overlap_cats)
                
                if overlap_size == 0:
                    no_overlap_count += 1
                elif overlap_size == len(target_all_cats):
                    full_overlap_count += 1
                else:
                    partial_overlap_count += 1
                
                overlap_results.append({
                    "target_id": target_id,
                    "target_category": target_cat,
                    "target_all_categories": list(target_all_cats),
                    "history_categories_count": len(history_cats),
                    "overlap_categories": list(overlap_cats),
                    "overlap_size": overlap_size,
                    "overlap_ratio": round(overlap_size / max(len(target_all_cats), 1), 3),
                })
            
            total = len(wrong_cases)
            stats = {
                "total_wrong_cases": total,
                "no_category_overlap_count": no_overlap_count,
                "no_category_overlap_pct": round(no_overlap_count / total * 100, 2) if total > 0 else 0,
                "partial_overlap_count": partial_overlap_count,
                "partial_overlap_pct": round(partial_overlap_count / total * 100, 2) if total > 0 else 0,
                "full_overlap_count": full_overlap_count,
                "full_overlap_pct": round(full_overlap_count / total * 100, 2) if total > 0 else 0,
                "avg_overlap_ratio": round(sum(r["overlap_ratio"] for r in overlap_results) / total, 3) if total > 0 else 0,
            }
            
            # 样本 (取前 3 条)
            sample = overlap_results[:3]
            
            return {
                "per_case_overlap": overlap_results,
                "statistics": stats,
                "sample": sample,
                "computation_method": "builtin: category overlap between target item and user history sequence",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute category_overlap_stats: {e}")
            return None
    
    def _compute_category_distribution(self, preloaded_data: Dict) -> Optional[Dict]:
        """
        计算物品类别分布 (从元数据统计各顶层类别的物品数量)
        """
        item_metadata = preloaded_data.get("item_text_map") or preloaded_data.get("item_metadata")
        
        if not item_metadata:
            item_metadata = self._load_metadata()
        
        if not item_metadata:
            logger.warning("No metadata available for category_distribution computation")
            return None
        
        try:
            category_count = Counter()
            for item_id, meta in item_metadata.items():
                if isinstance(meta, dict):
                    cat = meta.get("categories", "")
                    if cat and ">" in cat:
                        top_cat = cat.split(">")[0].strip()
                    elif cat:
                        top_cat = cat.strip()
                    else:
                        top_cat = "unknown"
                else:
                    top_cat = "unknown"
                category_count[top_cat] += 1
            
            total_items = sum(category_count.values())
            distribution = {cat: count for cat, count in category_count.most_common()}
            
            stats = {
                "total_items": total_items,
                "unique_categories": len(distribution),
                "top_category": category_count.most_common(1)[0][0] if distribution else "none",
                "top_category_pct": round(category_count.most_common(1)[0][1] / total_items * 100, 2) if total_items > 0 else 0,
                "distribution": distribution,
            }
            
            return {
                "statistics": stats,
                "sample": dict(category_count.most_common(3)),
                "computation_method": "builtin: count items by top-level category from metadata",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute category_distribution: {e}")
            return None
    
    def _compute_wrong_case_category_stats(self, hypothesis: Dict,
                                            preloaded_data: Dict) -> Optional[Dict]:
        """
        计算误推案例中目标物品的类别分布
        
        统计误推目标物品在各类别中的分布，并与全量物品类别分布对比
        """
        wrong_cases = preloaded_data.get("wrong_text_cases", [])
        item_metadata = preloaded_data.get("item_text_map") or preloaded_data.get("item_metadata")
        
        if not item_metadata:
            item_metadata = self._load_metadata()
        
        if not wrong_cases or not item_metadata:
            logger.warning("Missing data for wrong_case_category_stats computation")
            return None
        
        try:
            # 误推目标类别分布
            wrong_target_cats = Counter()
            for case in wrong_cases:
                target_id = str(case.get("target_id", ""))
                target_meta = item_metadata.get(target_id, {})
                if isinstance(target_meta, dict):
                    cat = target_meta.get("categories", "")
                    if cat and ">" in cat:
                        top_cat = cat.split(">")[0].strip()
                    elif cat:
                        top_cat = cat.strip()
                    else:
                        top_cat = "unknown"
                else:
                    top_cat = "unknown"
                wrong_target_cats[top_cat] += 1
            
            # 全量类别分布
            all_cats = Counter()
            for item_id, meta in item_metadata.items():
                if isinstance(meta, dict):
                    cat = meta.get("categories", "")
                    if cat and ">" in cat:
                        top_cat = cat.split(">")[0].strip()
                    elif cat:
                        top_cat = cat.strip()
                    else:
                        top_cat = "unknown"
                else:
                    top_cat = "unknown"
                all_cats[top_cat] += 1
            
            # 计算比率
            wrong_total = sum(wrong_target_cats.values())
            all_total = sum(all_cats.values())
            
            ratio_dict = {}
            for cat, wrong_count in wrong_target_cats.items():
                all_count = all_cats.get(cat, 0)
                wrong_pct = wrong_count / wrong_total * 100 if wrong_total > 0 else 0
                all_pct = all_count / all_total * 100 if all_total > 0 else 0
                ratio_dict[cat] = {
                    "wrong_count": wrong_count,
                    "wrong_pct": round(wrong_pct, 2),
                    "all_count": all_count,
                    "all_pct": round(all_pct, 2),
                    "ratio": round(wrong_pct / all_pct, 2) if all_pct > 0 else 0,
                }
            
            stats = {
                "wrong_case_count": wrong_total,
                "wrong_category_distribution": dict(wrong_target_cats.most_common()),
                "all_category_distribution": dict(all_cats.most_common()),
                "category_ratio_comparison": ratio_dict,
            }
            
            return {
                "statistics": stats,
                "sample": dict(list(ratio_dict.items())[:3]),
                "computation_method": "builtin: compare wrong-case target categories with overall distribution",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute wrong_case_category_stats: {e}")
            return None
    
    def _compute_recommendation_frequency(self, preloaded_data: Dict) -> Optional[Dict]:
        """
        计算模型推荐结果中各物品的推荐频次
        
        从误推案例的预测列表中统计各物品被推荐的次数
        """
        wrong_cases = preloaded_data.get("wrong_text_cases", [])
        
        if not wrong_cases:
            logger.warning("No wrong cases available for recommendation_frequency computation")
            return None
        
        try:
            rec_freq = Counter()
            total_predictions = 0
            
            for case in wrong_cases:
                predictions = case.get("predictions_ids", [])
                if predictions:
                    for pred_id in predictions:
                        rec_freq[str(pred_id)] += 1
                    total_predictions += len(predictions)
            
            # 计算排名 (前 10, 前 20 等)
            top_rec = rec_freq.most_common(20)
            
            stats = {
                "total_prediction_slots": total_predictions,
                "unique_items_in_predictions": len(rec_freq),
                "top_20_recommendations": [
                    {"item_id": item_id, "frequency": freq,
                     "pct": round(freq / total_predictions * 100, 4) if total_predictions > 0 else 0}
                    for item_id, freq in top_rec
                ],
                "max_frequency": rec_freq.most_common(1)[0][1] if rec_freq else 0,
                "mean_frequency": round(sum(rec_freq.values()) / len(rec_freq), 2) if rec_freq else 0,
            }
            
            return {
                "frequency_dict": {k: v for k, v in rec_freq.items()},
                "statistics": stats,
                "sample": {item_id: freq for item_id, freq in top_rec[:3]},
                "computation_method": "builtin: count item occurrences in model prediction lists from wrong cases",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute recommendation_frequency: {e}")
            return None
    
    def _compute_sequence_target_mapping(self, preloaded_data: Dict) -> Optional[Dict]:
        """
        计算用户历史序列与目标物品的关联数据
        
        提取每个案例中: 用户历史序列物品ID、目标物品ID、序列长度、预测物品ID
        """
        wrong_cases = preloaded_data.get("wrong_text_cases", [])
        
        if not wrong_cases:
            logger.warning("No wrong cases available for sequence_target_mapping computation")
            return None
        
        try:
            mappings = []
            for case in wrong_cases:
                target_id = case.get("target_id")
                seq_length = case.get("original_length", 0)
                predictions_ids = case.get("predictions_ids", [])
                user_id = case.get("user_id")
                
                mapping = {
                    "user_id": user_id,
                    "target_id": target_id,
                    "sequence_length": seq_length,
                    "prediction_ids": predictions_ids[:20] if predictions_ids else [],
                    "target_in_predictions": target_id in (predictions_ids or []),
                }
                if mapping["target_in_predictions"]:
                    # 找到目标在预测中的排名
                    try:
                        rank = predictions_ids.index(target_id)
                        mapping["target_rank_in_predictions"] = rank
                    except (ValueError, TypeError):
                        mapping["target_rank_in_predictions"] = -1
                
                mappings.append(mapping)
            
            # 统计摘要
            seq_lengths = [m["sequence_length"] for m in mappings]
            targets_in_pred = sum(1 for m in mappings if m["target_in_predictions"])
            
            stats = {
                "total_cases": len(mappings),
                "avg_sequence_length": round(sum(seq_lengths) / len(seq_lengths), 2) if seq_lengths else 0,
                "min_sequence_length": min(seq_lengths) if seq_lengths else 0,
                "max_sequence_length": max(seq_lengths) if seq_lengths else 0,
                "targets_in_predictions_count": targets_in_pred,
                "targets_in_predictions_pct": round(targets_in_pred / len(mappings) * 100, 2) if mappings else 0,
            }
            
            return {
                "per_case_mapping": mappings,
                "statistics": stats,
                "sample": mappings[:3],
                "computation_method": "builtin: extract user sequence and target item association from wrong cases",
            }
        
        except Exception as e:
            logger.warning(f"Failed to compute sequence_target_mapping: {e}")
            return None
    
    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════
    
    def _load_metadata(self) -> Optional[Dict]:
        """从数据目录加载物品元数据"""
        meta_path = os.path.join(self.data_dir, "id_meta_data.json")
        if not os.path.exists(meta_path):
            # 尝试其他可能的文件名
            if os.path.exists(self.data_dir):
                for fname in os.listdir(self.data_dir):
                    if fname.endswith(".json") and "meta" in fname.lower():
                        meta_path = os.path.join(self.data_dir, fname)
                        break
        
        if not os.path.exists(meta_path):
            return None
        
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            return None
    
    def _save_computed_to_cache(self, data_name: str, data: Dict):
        """将计算结果保存到文件缓存"""
        cache_file = os.path.join(self.cache_dir, f"{data_name}.json")
        try:
            # 序列化时截断过大的数据
            serializable = {"data": data, "name": data_name}
            # 如果 data 中的 per_case_overlap 或 frequency_dict 超过 5000 条，截断
            for key in ["per_case_overlap", "per_case_mapping", "freq_dict", "frequency_dict"]:
                if key in data and isinstance(data[key], (list, dict)):
                    if len(data[key]) > 5000:
                        if isinstance(data[key], list):
                            serializable["data"][key] = data[key][:5000]
                            serializable["data"][key + "_truncated"] = True
                            serializable["data"][key + "_total"] = len(data[key])
                        elif isinstance(data[key], dict):
                            items = list(data[key].items())[:5000]
                            serializable["data"][key] = dict(items)
                            serializable["data"][key + "_truncated"] = True
                            serializable["data"][key + "_total"] = len(data[key])
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(serializable, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved computed data to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to save computed data to cache: {e}")
    
    # ════════════════════════════════════════
    # LLM 计算方法 (用于复杂/新颖的数据需求)
    # ════════════════════════════════════════
    
    def _compute_with_llm(self, missing_data_desc: str,
                           hypothesis: Dict,
                           preloaded_data: Dict) -> Optional[Dict]:
        """
        让 LLM 编写数据计算脚本并执行
        
        Args:
            missing_data_desc: 缺失数据描述 (如 "category_overlap_stats: 目标物品与用户历史序列的类别重叠统计")
            hypothesis: 当前假设
            preloaded_data: 已加载的数据
            
        Returns:
            Computed data dict, or None
        """
        if self.llm is None:
            return None
        
        hyp_id = hypothesis.get("id", "H?")
        data_name = missing_data_desc.split(":")[0].strip() if ":" in missing_data_desc else missing_data_desc.strip()
        
        # 构建原始数据源描述
        raw_data_sources = self._format_raw_data_sources(preloaded_data)
        
        # 创建输出文件路径
        output_file = os.path.join(self.cache_dir, f"{data_name}_raw.json")
        
        prompt = DATA_COMPUTATION_PROMPT.format(
            missing_data_description=missing_data_desc,
            raw_data_sources=raw_data_sources,
            hypothesis_id=hyp_id,
            hypothesis_claim=hypothesis.get("claim", ""),
            verification_thought=hypothesis.get("verification_thought", ""),
            data_needed=json.dumps(hypothesis.get("data_needed", []), ensure_ascii=False),
            output_file_path=output_file,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位 Python 数据工程师，擅长编写数据处理脚本。"
                    "代码必须稳健、能处理文件不存在的情况。"
                    "结果必须以 JSON 格式输出到指定文件。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        
        if response is None:
            logger.error("LLM computation script generation failed - no response")
            return None
        
        # 清理代码
        from .hypothesis_verification_agent import HypothesisVerificationAgent
        # 使用相同的代码清理方法
        code = self._clean_computation_code(response)
        
        # 在代码开头注入必要的头部
        code = self._inject_computation_header(code, output_file, preloaded_data)
        
        # 执行脚本 (带修正循环)
        result = self._execute_computation_with_fix(
            code, data_name, hypothesis, output_file, preloaded_data
        )
        
        return result
    
    def _format_raw_data_sources(self, preloaded_data: Dict) -> str:
        """格式化原始数据源描述"""
        lines = []
        
        # 数据文件
        if os.path.exists(self.data_dir):
            for fname in sorted(os.listdir(self.data_dir)):
                fpath = os.path.join(self.data_dir, fname)
                if os.path.isfile(fpath) and fname.endswith((".txt", ".json")):
                    size_kb = os.path.getsize(fpath) / 1024
                    lines.append(f"- **{fname}** ({size_kb:.0f}KB): 路径 `{fpath}`")
        
        # 已加载的数据路径
        for key, value in preloaded_data.items():
            if isinstance(value, str) and os.path.exists(value):
                lines.append(f"- 已加载: `{key}` = `{value}`")
            elif isinstance(value, dict) and "path" in value:
                lines.append(f"- 已加载: `{key}` = `{value['path']}`")
        
        return "\n".join(lines) if lines else "无可用原始数据文件"
    
    def _clean_computation_code(self, response: str) -> str:
        """清理 LLM 生成的计算代码 (增强版 — 委托给 _clean_code_response)"""
        from .hypothesis_verification_agent import HypothesisVerificationAgent
        return HypothesisVerificationAgent._clean_code_response_static(response)
    
    def _inject_computation_header(self, code: str, output_file: str,
                                    preloaded_data: Dict) -> str:
        """为计算脚本注入必要的头部"""
        # 将 preloaded 中的路径数据保存到临时 JSON 文件
        data_file = os.path.join(self.cache_dir, "_preloaded_paths.json")
        paths_only = {}
        for key, value in preloaded_data.items():
            if isinstance(value, str):
                paths_only[key] = value
            elif isinstance(value, dict) and "path" in value:
                paths_only[key] = value["path"]
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(paths_only, f, ensure_ascii=False)
        
        header = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            "# Data computation script auto-generated",
            "",
            "import json",
            "import os",
            "import sys",
            "from collections import Counter, defaultdict",
            "",
            f'OUTPUT_FILE = "{output_file}"',
            f'PATHS_FILE = "{data_file}"',
            "",
            "try:",
            "    with open(PATHS_FILE, 'r') as f:",
            "        _paths = json.load(f)",
            "except Exception:",
            "    _paths = {}",
            "",
            "def save_result(result_dict):",
            "    try:",
            "        os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)",
            "        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:",
            "            json.dump(result_dict, f, ensure_ascii=False, indent=2)",
            "    except Exception as e:",
            "        sys.stderr.write(f'Failed to save: {e}\\n')",
            "",
            "",
        ]
        
        return "\n".join(header) + "\n" + code
    
    def _execute_computation_with_fix(self, initial_code: str, data_name: str,
                                       hypothesis: Dict, output_file: str,
                                       preloaded_data: Dict) -> Optional[Dict]:
        """执行计算脚本，如果失败则修正重试"""
        script_dir = os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_file = os.path.join(script_dir, f"compute_{data_name}.py")
        
        code = initial_code
        
        for round_num in range(self.MAX_COMPUTE_FIX_ROUNDS):
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing computation script for {data_name} (round {round_num + 1})")
            print(f"      ⚡ 计算数据 {data_name} (round {round_num + 1}/{self.MAX_COMPUTE_FIX_ROUNDS})...")
            
            success, result, error = self._execute_computation_script(script_file)
            
            if success and result is not None:
                # 从结果中提取 data 字段
                data = result.get("data", result)
                return data
            
            if error and round_num < self.MAX_COMPUTE_FIX_ROUNDS - 1 and self.llm:
                logger.warning(f"Computation script failed: {error[:200]}")
                print(f"      ❌ 计算失败: {error[:100]}... 让 LLM 修正")
                
                fixed_code = self._fix_computation_code(code, error, output_file)
                if fixed_code:
                    code = fixed_code
                else:
                    break
        
        return None
    
    def _execute_computation_script(self, script_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """执行计算脚本 (增强版 — 多模式输出检测 + 健壮 JSON 解析)"""
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.MAX_COMPUTE_TIMEOUT,
                cwd=self.project_root,
            )
            
            if result.returncode != 0:
                error = result.stderr or result.stdout or "Unknown execution error"
                return False, None, error
            
            # 从脚本中找输出文件路径 — 多模式匹配
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            output_path = HypothesisVerificationAgent._extract_output_path(script_content)
            
            if output_path and os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                return True, result_data, None
            
            # 尝试从 stdout/stderr 解析 JSON (健壮解析)
            for source_name, source_text in [
                ("stderr", result.stderr),
                ("stdout", result.stdout),
            ]:
                if not source_text or not source_text.strip():
                    continue
                json_str = HypothesisVerificationAgent._extract_json_block(source_text)
                if json_str:
                    parsed = HypothesisVerificationAgent._robust_json_parse(json_str)
                    if parsed is not None:
                        logger.info(f"Computation result extracted from {source_name}")
                        return True, parsed, None
            
            error_msg = "No output file or JSON found"
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            return False, None, error_msg
        
        except subprocess.TimeoutExpired:
            return False, None, f"Computation timed out ({self.MAX_COMPUTE_TIMEOUT}s)"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
    
    def _fix_computation_code(self, original_code: str, error: str,
                               output_file: str) -> Optional[str]:
        """让 LLM 修正计算脚本"""
        if self.llm is None:
            return None
        
        # 提取核心代码
        core_code = self._extract_core_from_computation(original_code)
        
        prompt = VERIFICATION_CODE_FIX_PROMPT.format(
            original_code=core_code,
            error_output=error[:1500],
            output_file_path=output_file,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位 Python 数据工程师，擅长修正数据处理脚本。"
                    "只修正导致错误的部分，保持其他逻辑不变。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        
        if response is None:
            return None
        
        fixed_core = self._clean_computation_code(response)
        return self._inject_computation_header(fixed_core, output_file, {})
    
    def _extract_core_from_computation(self, full_code: str) -> str:
        """从计算脚本中提取核心逻辑 (去掉注入头部, 增强版)"""
        lines = full_code.split("\n")
        core_start = 0
        
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                prev_lines = [l.strip() for l in lines[max(0, i-5):i]]
                if any("sys.stderr" in l for l in prev_lines):
                    core_start = i + 1
                    break
        
        if core_start == 0:
            excluded_prefixes = ("#", "import ", "from ", "OUTPUT_FILE", "PATHS_FILE",
                                 "_paths", "def save_result", "globals()", "data_file",
                                 "_preloaded", "try:", "except")
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not any(stripped.startswith(p) for p in excluded_prefixes):
                    core_start = i
                    break
        
        if core_start > 0:
            return "\n".join(lines[core_start:])
        return full_code
    
    # 已知的数据类型列表 (内置计算生成的数据)
    KNOWN_COMPUTED_DATA_DESCRIPTIONS = {
        "category_overlap_stats": "目标物品与用户历史序列的类别重叠统计",
        "category_distribution": "物品类别分布 (各顶层类别的物品数量)",
        "wrong_case_category_stats": "误推案例中目标物品的类别分布 vs 全量类别分布",
        "recommendation_frequency": "模型推荐结果中各物品的推荐频次",
        "sequence_target_mapping": "用户历史序列与目标物品的关联数据",
        "item_interaction_freq": "训练数据中各物品的交互频率统计",
    }
    
    def format_computed_data_for_prompt(self, computed_data: Dict) -> str:
        """格式化已计算的数据为 LLM prompt 文本"""
        lines = []
        for data_name, data_value in computed_data.items():
            # 添加已知描述
            desc = self.KNOWN_COMPUTED_DATA_DESCRIPTIONS.get(data_name, "")
            if desc:
                lines.append(f"\n### 已计算数据: {data_name} — {desc}")
            else:
                lines.append(f"\n### 已计算数据: {data_name}")
            
            if isinstance(data_value, dict):
                # 显示统计摘要
                if "statistics" in data_value:
                    stats = data_value["statistics"]
                    lines.append("**统计摘要:**")
                    for k, v in stats.items():
                        if isinstance(v, (int, float, str)):
                            lines.append(f"  - {k}: {v}")
                        elif isinstance(v, dict) and len(v) <= 10:
                            lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
                        elif isinstance(v, list) and len(v) <= 5:
                            lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
                
                # 显示样本
                if "sample" in data_value:
                    sample = data_value["sample"]
                    lines.append(f"**数据样本:** {json.dumps(sample, ensure_ascii=False)[:300]}")
                
                # 显示计算方法
                if "computation_method" in data_value:
                    lines.append(f"**计算方法:** {data_value['computation_method']}")
                
                lines.append(f"**可用变量名:** `{data_name}` (直接在验证脚本中使用)")
            
            elif isinstance(data_value, list):
                lines.append(f"**数据:** {len(data_value)} 条记录")
                if data_value:
                    lines.append(f"**样本:** {json.dumps(data_value[:2], ensure_ascii=False)[:200]}")
            
            else:
                lines.append(f"**数据类型:** {type(data_value).__name__}")
        
        return "\n".join(lines) if lines else "暂无已计算的数据"


# ════════════════════════════════════════
# 模型探测引擎 — 从模型推理过程中提取内部数据
# ════════════════════════════════════════

class ModelProbingEngine:
    """
    模型内部数据提取引擎
    
    当假设验证需要模型推理过程中的内部数据 (如注意力权重、隐藏状态、
    嵌入向量等) 时，DataComputationEngine 无法通过简单的数据处理脚本
    获取这些数据。ModelProbingEngine 通过:
    
    1. 分析模型源码结构 (LLM Step 1)
    2. 生成模型探测脚本 (LLM Step 2) — 使用 PyTorch hooks 或临时模型修改
    3. 执行探测脚本并收集结果 (带修正循环)
    
    来提取模型内部的中间计算结果，使假设验证不再因为缺少模型内部数据
    而被标记为"UNVERIFIABLE"。
    
    设计原则:
    - **不修改原始模型文件**: 使用临时目录 + 复制修改的方式，或在脚本内
      使用 register_forward_hook
    - **多步 LLM 调用**: 先分析模型结构，再生成脚本，确保脚本的正确性
    - **修正循环**: 脚本执行失败时自动让 LLM 修正
    - **资源控制**: 只在小样本上运行推理，避免内存溢出
    """
    
    MAX_PROBE_TIMEOUT = 180      # 模型推理超时 (秒) — 比数据处理更长
    MAX_PROBE_FIX_ROUNDS = 3     # 探测脚本修正最大轮数
    MAX_PROBE_SAMPLES = 50       # 最大推理样本数 — 避免内存溢出
    
    # 已知的模型内部数据类型及其描述
    KNOWN_MODEL_DATA_TYPES = {
        "attention_weights": {
            "description": "自注意力权重矩阵 (attention_probs)",
            "keywords": ["注意力权重", "attention weights", "attention_probs", "注意力分布",
                         "注意力坍缩", "attention collapse", "注意力集中", "attention entropy",
                         "注意力重心", "attention centroid"],
            "default_probing_method": "在 SelfAttention 上使用 forward_hook 或临时修改模型返回 attention_probs",
        },
        "attention_entropy": {
            "description": "注意力分布的熵 (衡量注意力集中程度)",
            "keywords": ["注意力熵", "attention entropy", "注意力集中度", "注意力分布熵"],
            "default_probing_method": "从 attention_weights 计算 Shannon 熵",
        },
        "hidden_states": {
            "description": "Transformer 编码器的隐藏状态输出",
            "keywords": ["隐藏状态", "hidden states", "encoder layers", "编码器输出",
                         "中间表示", "intermediate representation"],
            "default_probing_method": "在 EncoderLayer 上使用 forward_hook",
        },
        "item_embeddings": {
            "description": "物品嵌入向量",
            "keywords": ["物品嵌入", "item embeddings", "嵌入向量", "embedding weights",
                         "item embedding"],
            "default_probing_method": "直接读取 model.item_embeddings.weight",
        },
        "position_embeddings": {
            "description": "位置嵌入向量",
            "keywords": ["位置嵌入", "position embeddings"],
            "default_probing_method": "直接读取 model.position_embeddings.weight",
        },
        "model_predictions": {
            "description": "模型的推荐预测结果 (对所有物品的打分)",
            "keywords": ["模型预测", "model predictions", "推荐分数", "prediction scores",
                         "模型输出", "推荐结果"],
            "default_probing_method": "执行模型 forward pass 获取 sequence_output → dot product with item embeddings",
        },
        "gradient_info": {
            "description": "梯度信息 (反向传播梯度)",
            "keywords": ["梯度", "gradient", "梯度分布", "梯度范数"],
            "default_probing_method": "使用 register_backward_hook 或计算 loss.backward() 后检查梯度",
        },
    }
    
    def __init__(self, project_root: str, data_dir: str = None,
                 log_dir: str = None, llm_client=None):
        """
        Args:
            project_root: 项目根目录 (通常为 Recmodel 目录)
            data_dir: 数据目录路径
            log_dir: 日志/缓存目录路径
            llm_client: LLMClient 实例 (用于生成探测脚本)
        """
        self.project_root = project_root
        self.data_dir = data_dir or os.path.join(project_root, "data")
        self.log_dir = log_dir or os.path.join(project_root, "logs")
        self.llm = llm_client
        self._probed_cache = {}
        
        # 缓存目录
        self.cache_dir = os.path.join(self.log_dir, "model_probe_cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # 推测 Recmodel 目录 (可能在 project_root 内或就是 project_root)
        self.recmodel_dir = self._find_recmodel_dir()
    
    def _find_recmodel_dir(self) -> str:
        """查找 Recmodel 目录"""
        # 如果 project_root 本身就是 Recmodel (含有 models.py)
        if os.path.exists(os.path.join(self.project_root, "models.py")):
            return self.project_root
        # 否则查找 project_root/Recmodel/
        if os.path.exists(os.path.join(self.project_root, "Recmodel", "models.py")):
            return os.path.join(self.project_root, "Recmodel")
        # 默认返回 project_root
        return self.project_root
    
    def probe_model_data(self, missing_data: List[str],
                          hypothesis: Dict,
                          preloaded_data: Dict) -> Dict:
        """
        提取模型内部数据
        
        流程:
        1. 发现模型信息 (checkpoint, 模型源码, 数据路径)
        2. LLM 分析模型结构 → 确定探测目标
        3. LLM 生成探测脚本 → 使用 PyTorch hooks / 临时模型修改
        4. 执行探测脚本 → 收集结果
        5. 失败时修正脚本 (最多 3 轮)
        
        Args:
            missing_data: 需要提取的模型内部数据描述列表
            hypothesis: 当前假设
            preloaded_data: 已加载的数据
            
        Returns:
            Dict of extracted data, keyed by data name
        """
        extracted = {}
        remaining_missing = []
        
        for missing_desc in missing_data:
            data_name = missing_desc.split(":")[0].strip() if ":" in missing_desc else missing_desc.strip()
            
            # 检查缓存
            if data_name in self._probed_cache:
                extracted[data_name] = self._probed_cache[data_name]
                logger.info(f"Using cached model-probed data: {data_name}")
                continue
            
            # 检查文件缓存
            cached_file = os.path.join(self.cache_dir, f"{data_name}.json")
            if os.path.exists(cached_file):
                try:
                    with open(cached_file, 'r', encoding='utf-8') as f:
                        cached_result = json.load(f)
                    extracted[data_name] = cached_result.get("data", cached_result)
                    self._probed_cache[data_name] = extracted[data_name]
                    logger.info(f"Using file-cached model-probed data: {data_name}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to load cached {data_name}: {e}")
            
            # 尝试提取
            probe_result = self._probe_single_data(
                data_name, missing_desc, hypothesis, preloaded_data
            )
            
            if probe_result is not None:
                extracted[data_name] = probe_result
                self._probed_cache[data_name] = probe_result
                self._save_probed_to_cache(data_name, probe_result)
                logger.info(f"Model probing succeeded for {data_name}")
            else:
                remaining_missing.append(missing_desc)
                logger.warning(f"Model probing failed for {data_name}")
        
        return extracted
    
    def _probe_single_data(self, data_name: str, missing_desc: str,
                            hypothesis: Dict, preloaded_data: Dict) -> Optional[Dict]:
        """
        对单个缺失数据执行完整的模型探测流程
        
        Step 1: 发现模型信息
        Step 2: LLM 分析模型结构
        Step 3: LLM 生成探测脚本
        Step 4: 执行脚本 (带修正循环)
        """
        hyp_id = hypothesis.get("id", "H?")
        print(f"      🔬 模型探测 {data_name} (假设 {hyp_id})...")
        
        # --- Step 1: 发现模型信息 ---
        model_info = self._discover_model_info()
        
        if not model_info.get("checkpoint_path"):
            logger.warning(f"No model checkpoint found, cannot probe {data_name}")
            print(f"      ⚠️ 未找到模型 checkpoint, 无法执行模型探测")
            return None
        
        # --- Step 2: LLM 分析模型结构 ---
        print(f"      📋 [Step 1] LLM 分析模型结构...")
        model_analysis = self._analyze_model_for_probing(
            missing_desc, hypothesis, model_info
        )
        
        if not model_analysis:
            logger.warning(f"Model analysis failed for {data_name}")
            print(f"      ❌ 模型结构分析失败")
            return None
        
        # --- Step 3: LLM 生成探测脚本 ---
        print(f"      💻 [Step 2] LLM 生成探测脚本...")
        probing_script = self._generate_probing_script(
            missing_desc, hypothesis, model_info, model_analysis
        )
        
        if not probing_script:
            logger.warning(f"Probing script generation failed for {data_name}")
            print(f"      ❌ 探测脚本生成失败")
            return None
        
        # --- Step 4: 执行脚本 (带修正循环) ---
        print(f"      ⚡ [Step 3] 执行探测脚本...")
        result = self._execute_probing_with_fix(
            probing_script, data_name, hypothesis, model_info
        )
        
        return result
    
    def _discover_model_info(self) -> Dict:
        """
        发现模型相关信息: checkpoint, 模型源码, 数据路径
        
        Returns:
            Dict containing model_info
        """
        info = {
            "recmodel_dir": self.recmodel_dir,
            "data_dir": self.data_dir,
            "checkpoint_path": None,
            "model_source_files": {},
            "model_args": None,
        }
        
        # 查找 checkpoint
        info["checkpoint_path"] = self._find_latest_checkpoint()
        
        # 查找模型源码文件
        source_files = {}
        for fname in ["models.py", "modules.py", "trainers.py", "datasets.py", "utils.py"]:
            fpath = os.path.join(self.recmodel_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        source_files[fname] = f.read()
                except Exception as e:
                    logger.warning(f"Failed to read {fname}: {e}")
        info["model_source_files"] = source_files
        
        # 推断模型参数 (从 checkpoint 目录的日志或默认参数)
        info["model_args"] = self._infer_model_args()
        
        return info
    
    def _find_latest_checkpoint(self) -> Optional[str]:
        """
        查找最新的模型 checkpoint
        
        搜索策略:
        1. recmodel_dir/output/
        2. recmodel_dir/
        3. project_root/output/
        4. project_root/Recmodel/output/
        """
        search_dirs = [
            os.path.join(self.recmodel_dir, "output"),
            self.recmodel_dir,
            os.path.join(self.project_root, "output"),
        ]
        
        # 如果 project_root 不是 recmodel_dir，也搜索 project_root/Recmodel/output
        if not self.project_root.endswith("Recmodel"):
            search_dirs.append(os.path.join(self.project_root, "Recmodel", "output"))
        
        pt_files = []
        for search_dir in search_dirs:
            if not os.path.isdir(search_dir):
                continue
            for f in os.listdir(search_dir):
                if f.endswith('.pt'):
                    fpath = os.path.join(search_dir, f)
                    pt_files.append((fpath, os.path.getmtime(fpath)))
        
        if not pt_files:
            logger.warning(f"No checkpoint (.pt) found in: {search_dirs}")
            return None
        
        # 按修改时间排序，选最新的
        pt_files.sort(key=lambda x: x[1], reverse=True)
        latest = pt_files[0][0]
        logger.info(f"Found latest checkpoint: {latest}")
        return latest
    
    def _infer_model_args(self) -> Dict:
        """
        推断模型参数
        
        从数据目录推断 item_size, 从默认值推断其他参数
        """
        args = {
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.5,
            "attention_probs_dropout_prob": 0.5,
            "initializer_range": 0.02,
            "max_seq_length": 50,
            "item_size": None,  # 需要从数据推断
            "data_name": "Beauty",
        }
        
        # 推断 item_size
        train_file = os.path.join(self.data_dir, "Beauty_train.txt")
        if os.path.exists(train_file):
            try:
                max_item = 0
                with open(train_file, 'r') as f:
                    for line in f:
                        items = line.strip().split()
                        for item_id in items:
                            try:
                                item_id_int = int(item_id)
                                if item_id_int > max_item:
                                    max_item = item_id_int
                            except ValueError:
                                pass
                args["item_size"] = max_item + 2  # 加 padding
            except Exception as e:
                logger.warning(f"Failed to infer item_size from train data: {e}")
        
        # 尝试从日志文件推断参数
        log_dir = os.path.join(self.recmodel_dir, "output")
        if os.path.isdir(log_dir):
            for fname in os.listdir(log_dir):
                if fname.endswith('.txt') and 'SASRec' in fname:
                    try:
                        with open(os.path.join(log_dir, fname), 'r') as f:
                            first_line = f.readline()
                            # 解析 args 字符串 (格式: Namespace(...))
                            import argparse
                            ns = argparse.Namespace()
                            for part in first_line.split(','):
                                part = part.strip()
                                if '=' in part:
                                    key = part.split('=')[0].strip()
                                    val_str = part.split('=')[1].strip()
                                    try:
                                        val = eval(val_str)
                                        setattr(ns, key, val)
                                    except:
                                        pass
                            # 将推断到的参数合并
                            for key in ['hidden_size', 'num_hidden_layers', 'num_attention_heads',
                                        'hidden_dropout_prob', 'attention_probs_dropout_prob',
                                        'max_seq_length', 'item_size']:
                                if hasattr(ns, key):
                                    args[key] = getattr(ns, key)
                    except Exception as e:
                        logger.warning(f"Failed to parse log file {fname}: {e}")
        
        return args
    
    def _analyze_model_for_probing(self, missing_desc: str,
                                    hypothesis: Dict,
                                    model_info: Dict) -> Optional[Dict]:
        """
        LLM Step 1: 分析模型结构, 确定探测目标
        
        Returns:
            Dict of model analysis (target modules, extraction strategy, etc.)
        """
        if self.llm is None:
            return None
        
        source_files = model_info.get("model_source_files", {})
        models_source = source_files.get("models.py", "未找到 models.py")
        modules_source = source_files.get("modules.py", "未找到 modules.py")
        
        # 截断过长的源码
        max_len = 2000
        if len(models_source) > max_len:
            models_source = models_source[:max_len] + "\n# ... (截断)"
        if len(modules_source) > max_len:
            modules_source = modules_source[:max_len] + "\n# ... (截断)"
        
        prompt = MODEL_PROBING_ANALYSIS_PROMPT.format(
            missing_data_description=missing_desc,
            hypothesis_id=hypothesis.get("id", "H?"),
            hypothesis_claim=hypothesis.get("claim", ""),
            verification_thought=hypothesis.get("verification_thought", ""),
            models_source=models_source,
            modules_source=modules_source,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位深度学习架构分析专家，擅长分析 Transformer 模型的内部结构。"
                    "请精确分析需要 hook 的模块、hook 类型、数据提取策略。"
                    "输出必须是有效 JSON。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        
        if response is None:
            logger.error("Model analysis LLM call failed - no response")
            return None
        
        # 解析 JSON
        parsed = HypothesisVerificationAgent._robust_json_parse(response)
        if parsed is None:
            logger.error(f"Model analysis JSON parse failed: {response[:200]}")
            return None
        
        logger.info(f"Model analysis result: {json.dumps(parsed, ensure_ascii=False)[:300]}")
        return parsed
    
    def _generate_probing_script(self, missing_desc: str,
                                  hypothesis: Dict,
                                  model_info: Dict,
                                  model_analysis: Dict) -> Optional[str]:
        """
        LLM Step 2: 生成模型探测脚本
        
        Returns:
            Python script code string, or None
        """
        if self.llm is None:
            return None
        
        data_name = missing_desc.split(":")[0].strip() if ":" in missing_desc else missing_desc.strip()
        hyp_id = hypothesis.get("id", "H?")
        
        # 创建输出文件路径
        output_dir = os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"probe_{data_name}.json")
        
        source_files = model_info.get("model_source_files", {})
        modules_source = source_files.get("modules.py", "未找到 modules.py")
        
        # 截断过长的源码 (给 LLM 最多 3000 字符)
        max_len = 3000
        if len(modules_source) > max_len:
            modules_source = modules_source[:max_len] + "\n# ... (截断)"
        
        checkpoint_path = model_info.get("checkpoint_path", "未找到 checkpoint")
        model_args = model_info.get("model_args", {})
        
        prompt = MODEL_PROBING_SCRIPT_PROMPT.format(
            missing_data_description=missing_desc,
            hypothesis_id=hyp_id,
            hypothesis_claim=hypothesis.get("claim", ""),
            verification_thought=hypothesis.get("verification_thought", ""),
            model_analysis_json=json.dumps(model_analysis, indent=2, ensure_ascii=False),
            project_root=self.project_root,
            checkpoint_path=checkpoint_path,
            data_dir=self.data_dir,
            model_args=json.dumps(model_args, ensure_ascii=False),
            modules_source=modules_source,
            output_file_path=output_file,
            recmodel_dir=self.recmodel_dir,
        )
        
        # 尝试最多 2 次生成
        for attempt in range(2):
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位深度学习工程师，擅长编写模型探测脚本。"
                        "脚本必须能独立运行、正确加载模型、提取内部数据并保存为 JSON。"
                        "使用 PyTorch forward_hook 或临时模型修改来提取注意力权重等中间数据。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2 if attempt == 0 else 0.3,
                max_tokens=4096,
            )
            
            if response is None:
                logger.error(f"Probing script generation failed (attempt {attempt+1})")
                continue
            
            # 清理代码
            code = HypothesisVerificationAgent._clean_code_response_static(response)
            
            if code and len(code) > 50:
                return code
        
        logger.error("Probing script generation failed after 2 attempts")
        return None
    
    def _execute_probing_with_fix(self, initial_code: str, data_name: str,
                                   hypothesis: Dict, model_info: Dict) -> Optional[Dict]:
        """
        执行探测脚本，如果失败则修正重试
        
        Args:
            initial_code: 初始探测脚本代码
            data_name: 数据名称
            hypothesis: 当前假设
            model_info: 模型信息
            
        Returns:
            Extracted data dict, or None
        """
        script_dir = os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_file = os.path.join(script_dir, f"probe_{data_name}.py")
        
        code = initial_code
        
        for round_num in range(self.MAX_PROBE_FIX_ROUNDS):
            # 写入脚本
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing probing script for {data_name} (round {round_num + 1})")
            print(f"      ⚡ 执行探测 {data_name} (round {round_num + 1}/{self.MAX_PROBE_FIX_ROUNDS})...")
            
            success, result, error = self._execute_probing_script(script_file, code)
            
            if success and result is not None:
                data = result.get("data", result)
                print(f"      ✅ 模型探测成功: {data_name}")
                # 清理临时目录
                self._cleanup_temp_dirs(script_dir)
                return data
            
            if error and round_num < self.MAX_PROBE_FIX_ROUNDS - 1 and self.llm:
                logger.warning(f"Probing script failed: {error[:300]}")
                print(f"      ❌ 探测失败: {error[:100]}... 让 LLM 修正")
                
                fixed_code = self._fix_probing_script(code, error, data_name, hypothesis, model_info)
                if fixed_code:
                    code = fixed_code
                else:
                    break
        
        # 清理临时目录
        self._cleanup_temp_dirs(script_dir)
        return None
    
    def _execute_probing_script(self, script_path: str,
                                 script_content: str = None) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        执行探测脚本
        
        注意: 使用更长的超时时间 (模型推理需要更多时间)
        """
        try:
            # 检查脚本内容中的输出路径
            output_path = None
            if script_content:
                output_path = HypothesisVerificationAgent._extract_output_path(script_content)
            
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.MAX_PROBE_TIMEOUT,
                cwd=self.recmodel_dir,  # 在 Recmodel 目录中执行 (便于导入模块)
            )
            
            if result.returncode != 0:
                error = result.stderr or result.stdout or "Unknown execution error"
                # 截断过长的错误信息
                if len(error) > 500:
                    error = error[:500] + "... (truncated)"
                return False, None, error
            
            # 检查输出文件
            if output_path and os.path.exists(output_path):
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    return True, result_data, None
                except json.JSONDecodeError as e:
                    return False, None, f"Output JSON parse error: {e}"
            
            # 如果脚本内嵌了输出路径，检查脚本所在目录
            script_dir = os.path.dirname(script_path)
            probe_result_files = [f for f in os.listdir(script_dir)
                                  if f.startswith("probe_") and f.endswith(".json")]
            for rf in probe_result_files:
                try:
                    with open(os.path.join(script_dir, rf), 'r', encoding='utf-8') as f:
                        result_data = json.load(f)
                    return True, result_data, None
                except:
                    continue
            
            # 尝试从 stdout 解析 JSON
            if result.stdout and result.stdout.strip():
                parsed = HypothesisVerificationAgent._robust_json_parse(result.stdout)
                if parsed:
                    return True, parsed, None
            
            error_msg = "No output file or JSON found after model probing"
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            return False, None, error_msg
        
        except subprocess.TimeoutExpired:
            return False, None, f"Model probing timed out ({self.MAX_PROBE_TIMEOUT}s)"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
    
    def _fix_probing_script(self, original_code: str, error: str,
                             data_name: str, hypothesis: Dict,
                             model_info: Dict) -> Optional[str]:
        """
        修正探测脚本
        
        Args:
            original_code: 原始脚本代码
            error: 执行错误信息
            data_name: 数据名称
            hypothesis: 当前假设
            model_info: 模型信息
            
        Returns:
            Fixed code string, or None
        """
        if self.llm is None:
            return None
        
        prompt = MODEL_PROBING_FIX_PROMPT.format(
            original_code=original_code[:3000] if len(original_code) > 3000 else original_code,
            error_message=error,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位深度学习调试专家。请修正探测脚本中的错误。"
                    "特别关注: 模块导入路径、临时文件目录、checkpoint 加载、"
                    "数据路径、内存溢出等问题。"
                    "输出修正后的完整 Python 脚本 (不要只输出 diff)。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=4096,
        )
        
        if response is None:
            return None
        
        fixed_code = HypothesisVerificationAgent._clean_code_response_static(response)
        return fixed_code
    
    def _cleanup_temp_dirs(self, base_dir: str):
        """
        清理探测脚本创建的临时目录
        
        删除 _temp_model_probe 目录 (如果存在)
        """
        # 检查脚本目录下是否有临时目录
        for item in os.listdir(base_dir):
            if item.startswith("_temp_model_probe"):
                temp_path = os.path.join(base_dir, item)
                try:
                    shutil.rmtree(temp_path)
                    logger.info(f"Cleaned up temp dir: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup temp dir: {e}")
        
        # 也检查 Recmodel 目录下的临时目录
        recmodel_temp = os.path.join(self.recmodel_dir, "_temp_model_probe")
        if os.path.exists(recmodel_temp):
            try:
                shutil.rmtree(recmodel_temp)
                logger.info(f"Cleaned up Recmodel temp dir: {recmodel_temp}")
            except Exception as e:
                logger.warning(f"Failed to cleanup Recmodel temp dir: {e}")
    
    def _save_probed_to_cache(self, data_name: str, data: Dict):
        """将提取的数据保存到文件缓存"""
        cached_file = os.path.join(self.cache_dir, f"{data_name}.json")
        try:
            cache_entry = {
                "data": data,
                "timestamp": time.time(),
                "method": "model_probing",
            }
            with open(cached_file, 'w', encoding='utf-8') as f:
                json.dump(cache_entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save cache for {data_name}: {e}")
    
    def is_model_internal_data(self, data_description: str) -> bool:
        """
        判断数据描述是否需要模型内部数据
        
        Args:
            data_description: 数据描述字符串
            
        Returns:
            True if this data requires model probing
        """
        data_text = data_description.lower()
        for data_type, info in self.KNOWN_MODEL_DATA_TYPES.items():
            if any(kw in data_text for kw in info["keywords"]):
                return True
        return False
    
    def format_probed_data_for_prompt(self, probed_data: Dict) -> str:
        """格式化已提取的模型数据为 LLM prompt 文本"""
        lines = []
        for data_name, data_value in probed_data.items():
            # 添加已知描述
            desc = ""
            for dt, info in self.KNOWN_MODEL_DATA_TYPES.items():
                if dt == data_name:
                    desc = info["description"]
                    break
            
            if desc:
                lines.append(f"\n### 模型探测数据: {data_name} — {desc}")
            else:
                lines.append(f"\n### 模型探测数据: {data_name}")
            
            if isinstance(data_value, dict):
                # 显示统计摘要
                if "statistics" in data_value:
                    stats = data_value["statistics"]
                    lines.append("**统计摘要:**")
                    for k, v in stats.items():
                        if isinstance(v, (int, float, str)):
                            lines.append(f"  - {k}: {v}")
                        elif isinstance(v, dict) and len(v) <= 10:
                            lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
                        elif isinstance(v, list) and len(v) <= 5:
                            lines.append(f"  - {k}: {json.dumps(v, ensure_ascii=False)[:200]}")
                
                # 显示样本
                if "sample" in data_value:
                    sample = data_value["sample"]
                    lines.append(f"**数据样本:** {json.dumps(sample, ensure_ascii=False)[:300]}")
                
                # 显示计算方法
                if "computation_method" in data_value:
                    lines.append(f"**提取方法:** {data_value['computation_method']}")
                
                lines.append(f"**可用变量名:** `{data_name}` (直接在验证脚本中使用)")
            
            elif isinstance(data_value, list):
                lines.append(f"**数据:** {len(data_value)} 条记录")
                if data_value:
                    lines.append(f"**样本:** {json.dumps(data_value[:2], ensure_ascii=False)[:200]}")
            
            else:
                lines.append(f"**数据类型:** {type(data_value).__name__}")
        
        return "\n".join(lines) if lines else "暂无模型探测数据"


class HypothesisVerificationAgent:
    """
    假设验证 Agent — 自主式假设验证框架
    
    与旧版 HypothesisVerifier 的关键区别:
    1. 不使用固定验证方法 — LLM 为每个假设自由设计验证方案
    2. 不使用硬编码阈值 — LLM 解读统计结果来判断假设成立与否
    3. Agent 自主写代码执行验证 (而非预定义 Python 函数)
    4. 代码执行失败时自动修正 (self-correction loop)
    
    新增能力 (v2):
    5. 自动计算缺失的派生数据 — 当验证所需数据不存在时，
       Agent 会自动:
       a. 识别哪些数据需要计算
       b. 使用内置方法快速计算常见派生数据 (类别重叠、交互频率等)
       c. 对于复杂需求，让 LLM 编写计算脚本
       d. 执行脚本获取数据并注入到验证流程
    
    新增能力 (v3):
    6. 模型内部数据提取 — 当验证需要模型推理过程中的内部数据
       (注意力权重、隐藏状态、嵌入向量等) 时，DataComputationEngine
       无法通过数据处理脚本获取。新增 ModelProbingEngine：
       a. LLM 分析模型源码结构 → 确定需要 hook 的模块
       b. LLM 生成探测脚本 (使用 PyTorch hooks 或临时模型修改)
       c. 执行探测脚本并收集结果 (带修正循环)
       d. 使假设验证不再因缺少模型内部数据而被标记为"UNVERIFIABLE"
    
    工作流程:
    1. extract_hypotheses() — 从 LLM 分析提取假设
    2. verify_hypotheses() — 对每个假设执行完整的 Agent 验证流程:
       a. generate_verification_plan() — LLM 设计验证方案
       b. discover_and_load_data() — 发现并加载所需数据
       c. [v2] compute_needed_data() — 自动计算缺失的派生数据
       d. [v3] probe_model_data() — 提取模型内部数据 (注意力权重等)
       e. generate_verification_code() — LLM 写验证脚本
       f. execute_verification_code() — 执行脚本 (失败则修正)
       g. analyze_results() — LLM 解读结果, 判断假设
    3. generate_verification_report() — 生成汇总报告
    4. apply_verification_to_analysis() — 将验证结果反馈到分析中
    """
    
    # 验证状态枚举 (与旧版一致)
    CONFIRMED = "CONFIRMED"
    PARTIALLY_CONFIRMED = "PARTIALLY_CONFIRMED"
    REFUTED = "REFUTED"
    UNVERIFIABLE = "UNVERIFIABLE"
    
    # Agent 参数
    MAX_CODE_FIX_ROUNDS = 3    # 代码修正最大轮数
    MAX_EXECUTION_TIMEOUT = 60 # 验证脚本执行超时 (秒)
    
    def __init__(self, llm_client, item_text_map: Dict = None,
                 project_root: str = None, data_dir: str = None,
                 log_dir: str = None):
        """
        Args:
            llm_client: LLMClient 实例
            item_text_map: 物品 ID → 元数据映射
            project_root: 项目根目录路径
            data_dir: 数据目录路径
            log_dir: 日志目录路径
        """
        self.llm = llm_client
        self.item_text_map = item_text_map or {}
        self.project_root = project_root or os.getcwd()
        self.log_dir = log_dir or os.path.join(self.project_root, "logs")
        
        # 数据盘点器
        self.data_inventory = DataInventory(
            project_root=self.project_root,
            data_dir=data_dir,
            log_dir=self.log_dir,
        )
        
        # 旧版 verifier 作为 fallback
        from .hypothesis_verifier import HypothesisVerifier
        self._fallback_verifier = HypothesisVerifier(llm_client, item_text_map)
    
    # ════════════════════════════════════════
    # Phase 1: 假设提取 (新版 — 不限制验证类型)
    # ════════════════════════════════════════
    
    def extract_hypotheses(self, llm_analysis: Dict) -> Optional[List[Dict]]:
        """
        从 LLM 分析结果中提取可验证的假设 (增强版)
        
        与旧版区别:
        - 不限制为固定 6 种验证方法, LLM 自由描述验证思路
        - 新增 JSON 解析失败自动重试 (带错误反馈)
        - 新增多策略健壮 JSON 解析
        - 新增假设结构验证和补全
        - 新增部分假设提取恢复
        
        Args:
            llm_analysis: LLM 案例分析的结果
            
        Returns:
            List of hypothesis dicts, or None if extraction failed
        """
        if not llm_analysis or not llm_analysis.get("parse_success"):
            logger.warning("Cannot extract hypotheses from invalid LLM analysis")
            return None
        
        # 构建 prompt
        analysis_json = json.dumps(llm_analysis, indent=2, ensure_ascii=False)
        if len(analysis_json) > 6000:
            truncated = dict(llm_analysis)
            suggestions = truncated.get("improvement_suggestions", [])
            if suggestions:
                truncated["improvement_suggestions"] = [
                    {k: v for k, v in s.items() if k != "structural_change_detail"}
                    for s in suggestions[:3]
                ]
            analysis_json = json.dumps(truncated, indent=2, ensure_ascii=False)
        
        # 数据盘点
        data_inventory_text = self.data_inventory.format_inventory_for_prompt()
        
        prompt = HYPOTHESIS_EXTRACTION_PROMPT_V2.format(
            llm_analysis_json=analysis_json,
            data_inventory=data_inventory_text,
        )
        
        # 主请求
        response = self._call_llm_for_hypotheses(prompt)
        if response is None:
            logger.error("Hypothesis extraction failed - no LLM response")
            return None
        
        # 解析 + 自动重试 (最多 retry)
        result = self._try_parse_with_retry(response)
        if result is not None:
            return result
        
        # --- 最后手段: 尝试旧版 verifier 作为 fallback ---
        logger.info("All extraction + retry strategies failed, trying fallback verifier")
        try:
            old_hypotheses = self._fallback_verifier.extract_hypotheses(llm_analysis)
            if old_hypotheses:
                logger.info(f"Fallback verifier recovered {len(old_hypotheses)} hypotheses")
                # 补充旧版缺少的字段
                for h in old_hypotheses:
                    if "verification_thought" not in h:
                        h["verification_thought"] = h.get("verification_method", "custom")
                    if "data_needed" not in h:
                        h["data_needed"] = ["item_popularity", "category_metadata", "sequence_data"]
                return old_hypotheses
        except Exception as e2:
            logger.error(f"Fallback verifier also failed: {e2}")
        
        logger.warning(
            "All hypothesis extraction strategies exhausted — returning None. "
            "Log the raw response for debugging."
        )
        return None
    
    def _call_llm_for_hypotheses(self, prompt: str) -> Optional[str]:
        """调用 LLM 提取假设 (可被子方法重写用于测试)"""
        return self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位严谨的数据科学家，擅长从分析结论中识别可验证的假设。"
                    "你的目标是区分LLM的有数据支撑的结论和可能的主观臆断。"
                    "每个假设必须是可以用数据统计来验证的。"
                    "不要局限于固定验证类型，任何可以用数据回答的问题都是可验证假设。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
    
    def _try_parse_with_retry(self, first_response: str,
                               max_retries: int = 2) -> Optional[List[Dict]]:
        """
        解析 LLM 响应, 失败时自动重试带错误反馈
        
        Args:
            first_response: 首次 LLM 响应
            max_retries: 最大重试次数
            
        Returns:
            List of hypothesis dicts, or None
        """
        response = first_response
        
        for attempt in range(1 + max_retries):  # 首次 + max_retries 次重试
            parsed = self._parse_hypothesis_response(response)
            
            if parsed and parsed.get("hypotheses"):
                hypotheses = parsed["hypotheses"]
                if attempt > 0:
                    logger.info(
                        f"Hypotheses recovered on retry #{attempt}: "
                        f"{len(hypotheses)} hypotheses"
                    )
                else:
                    logger.info(f"Extracted {len(hypotheses)} verifiable hypotheses")
                return hypotheses
            
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries + 1} attempts exhausted. "
                    "Raw response (first 500 chars): " +
                    (response[:500] if response else "None")
                )
                return None
            
            # 准备重试: 提取错误信息
            raw_truncated = (response[:3000] + "..." 
                             if response and len(response) > 3000 
                             else (response or "None"))
            parse_error = self._diagnose_json_error(response)
            
            # 提取部分假设用于保留
            partial_text = "无部分解析结果"
            if response:
                partial = self._extract_partial_hypotheses(response)
                if partial:
                    partial_text = json.dumps(
                        [{"id": p.get("id", "?"), "claim": p.get("claim", "")[:60]}
                         for p in partial],
                        ensure_ascii=False
                    )
            
            fix_prompt = HYPOTHESIS_JSON_FIX_PROMPT.format(
                raw_response_truncated=raw_truncated,
                parse_error=parse_error,
                partial_hypotheses_text=partial_text,
            )
            
            logger.info(
                f"Retry #{attempt + 1} with JSON fix prompt "
                f"(parse error: {parse_error[:80]})"
            )
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的数据科学家。你之前输出的假设 JSON 格式有误，"
                        "请重新输出，确保是严格合法的 JSON 格式。"
                    )},
                    {"role": "user", "content": fix_prompt},
                ],
                temperature=0.2,
                max_tokens=2048,
            )
            
            if response is None:
                logger.error("No response from LLM during retry")
                return None
        
        return None
    
    @staticmethod
    def _diagnose_json_error(response: str) -> str:
        """
        诊断 JSON 解析错误的具体原因
        
        Returns:
            人类可读的错误描述
        """
        if not response:
            return "空响应"
        
        # 提取 JSON block
        json_str = HypothesisVerificationAgent._extract_json_block(response)
        if json_str is None:
            return "无法从响应中提取 JSON block (缺少 { 或 ```json 标记)"
        
        # 尝试加载并报告具体错误
        try:
            json.loads(json_str)
            return "标准解析看似正常 (可能 validation 阶段失败)"
        except json.JSONDecodeError as e:
            pos = e.pos
            context_start = max(0, pos - 40)
            context_end = min(len(json_str), pos + 40)
            context = json_str[context_start:context_end]
            
            diagnosis_parts = [f"JSON 解析错误 (位置 {pos}): {e.msg}"]
            diagnosis_parts.append(f"附近上下文: ...{context}...")
            
            # 常见问题诊断
            snippet = json_str[max(0, pos - 5):min(len(json_str), pos + 5)]
            
            if "Expecting ',' delimiter" in str(e):
                # 检查是否在 } 或 ] 后有额外字符
                diagnosis_parts.append("诊断: 可能在对象/数组内缺少逗号分隔符")
            elif "Expecting property name" in str(e) or "Expecting ':' delimiter" in str(e):
                if pos > 0 and json_str[pos-1] == "'":
                    diagnosis_parts.append("诊断: 有单引号未转为双引号")
                else:
                    diagnosis_parts.append("诊断: 键名缺少双引号或冒号")
            elif "Extra data" in str(e):
                diagnosis_parts.append("诊断: JSON 后有额外内容 (多个 JSON 对象)")
            elif "Unterminated string" in str(e):
                diagnosis_parts.append("诊断: 字符串未正确结束 (包含未转义的控制字符)")
            elif "Expecting value" in str(e):
                diagnosis_parts.append("诊断: 预期值位置出现意外字符")
            
            diagnosis_parts.append(f"错误附近字符: ...{snippet}...")
            
            return "\n".join(diagnosis_parts)
    
    def _extract_partial_hypotheses(self, response: str) -> Optional[List[Dict]]:
        """
        从完全不可解析的响应中尝试提取部分假设
        
        正则匹配假设级别的 JSON 对象
        """
        import re
        
        # 尝试提取独立的假设对象
        hypothesis_blocks = re.findall(
            r'\{\s*"id"\s*:\s*"[^"]*"\s*.*?"claim"\s*:\s*"[^"]*"[^}]*\}',
            response,
            re.DOTALL
        )
        
        recovered = []
        for block in hypothesis_blocks:
            parsed = self._robust_json_parse(block)
            if parsed and isinstance(parsed, dict) and "id" in parsed and "claim" in parsed:
                recovered.append(parsed)
        
        if recovered:
            logger.info(f"Partial extraction recovered {len(recovered)} hypotheses via regex")
        
        return recovered if recovered else None
    
    # ════════════════════════════════════════
    # Phase 2: 验证执行 (Agent 核心流程)
    # ════════════════════════════════════════
    
    def verify_hypotheses(self,
                          hypotheses: List[Dict],
                          wrong_text_cases: List[Dict],
                          all_wrong_cases: List[Dict] = None,
                          model_config: Dict = None,
                          item_popularity: Dict = None,
                          overall_metrics: Dict = None,
                          surprise_metrics: Dict = None) -> List[Dict]:
        """
        对每个假设运行自主验证
        
        与旧版区别:
        - 旧版: dispatch 到固定的 _verify_* 方法
        - 新版: 完整的 Agent 流程 (plan → code → execute → analyze)
        
        Args:
            hypotheses: 提取的假设列表
            wrong_text_cases: LLM 使用的文本格式错误案例
            all_wrong_cases: 原始格式的错误案例
            model_config: 模型配置
            item_popularity: 物品热度分布
            overall_metrics: 整体评估指标
            surprise_metrics: 惊喜子集指标
            
        Returns:
            List of verified hypothesis dicts (每个包含 verification_result)
        """
        verified = []
        
        # 预加载可用数据
        preloaded_data = self._prepare_preloaded_data(
            wrong_text_cases, all_wrong_cases, item_popularity,
            overall_metrics, surprise_metrics
        )
        
        for hyp in hypotheses:
            hyp_id = hyp.get("id", "H?")
            claim = hyp.get("claim", "")
            
            logger.info(f"Verifying {hyp_id}: {claim[:80]}...")
            print(f"  🔬 [Agent] 验证假设 {hyp_id}: {claim[:60]}...")
            
            try:
                result = self._verify_single_hypothesis(
                    hyp, preloaded_data, wrong_text_cases
                )
                
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = result
                verified.append(verified_hyp)
                
                status = result.get("status", self.UNVERIFIABLE)
                symbol = {"CONFIRMED": "✅", "PARTIALLY_CONFIRMED": "⚠️",
                          "REFUTED": "❌", "UNVERIFIABLE": "🔍"}.get(status, "?")
                print(f"    {symbol} {hyp_id} → {status}: {result.get('brief', '')[:80]}")
                
            except Exception as e:
                logger.error(f"Agent verification failed for {hyp_id}: {e}")
                traceback.print_exc()
                
                # Fallback: 尝试旧版验证
                print(f"    ⚠ Agent verification failed, trying fallback...")
                result = self._try_fallback_verification(
                    hyp, wrong_text_cases, all_wrong_cases,
                    item_popularity, overall_metrics, surprise_metrics
                )
                
                verified_hyp = dict(hyp)
                verified_hyp["verification_result"] = result
                verified.append(verified_hyp)
        
        return verified
    
    def _verify_single_hypothesis(self,
                                   hypothesis: Dict,
                                   preloaded_data: Dict,
                                   wrong_text_cases: List[Dict]) -> Dict:
        """
        对单个假设执行完整的 Agent 验证流程
        
        流程: Plan → Data → Code → Execute → Analyze
        
        Returns:
            verification result dict
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        # --- Step 1: 生成验证方案 ---
        print(f"    📋 [Step 1] 生成验证方案...")
        verification_plan = self._generate_verification_plan(
            hypothesis, preloaded_data
        )
        if not verification_plan:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "无法生成验证方案",
                "brief": "验证方案生成失败",
                "evidence": None,
            }
        
        # --- Step 2: 发现和准备数据 ---
        print(f"    📊 [Step 2] 准备验证数据...")
        verification_data = self._prepare_verification_data(
            hypothesis, verification_plan, preloaded_data
        )
        
        # --- Step 3: 生成验证代码 ---
        print(f"    💻 [Step 3] 生成验证代码...")
        code = self._generate_verification_code(
            hypothesis, verification_plan, verification_data
        )
        if not code:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "无法生成验证代码",
                "brief": "验证代码生成失败",
                "evidence": None,
            }
        
        # --- Step 4: 执行验证代码 (带修正循环) ---
        print(f"    ⚡ [Step 4] 执行验证代码...")
        execution_result = self._execute_with_correction_loop(
            code, hypothesis, verification_plan, verification_data
        )
        if not execution_result:
            return {
                "status": self.UNVERIFIABLE,
                "reason": "验证代码执行失败 (修正后仍无法运行)",
                "brief": "代码执行失败",
                "evidence": None,
            }
        
        # --- Step 5: 分析结果 ---
        print(f"    🔍 [Step 5] 分析验证结果...")
        analysis_result = self._analyze_results(
            hypothesis, verification_plan, execution_result
        )
        
        return analysis_result
    
    # ════════════════════════════════════════
    # Step 1: 验证方案生成
    # ════════════════════════════════════════
    
    def _generate_verification_plan(self,
                                     hypothesis: Dict,
                                     preloaded_data: Dict) -> Optional[Dict]:
        """
        LLM 为假设生成验证方案 (增强版)
        
        新增: 使用 _call_and_parse_json_with_retry 处理 JSON 解析失败
        新增: 对验证方案进行结构验证
        
        Returns:
            verification plan dict, or None
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        data_inventory_text = self.data_inventory.format_inventory_for_prompt()
        loaded_data_summary = self.data_inventory.format_loaded_data_summary(preloaded_data)
        
        prompt = VERIFICATION_PLAN_PROMPT.format(
            hypothesis_id=hyp_id,
            hypothesis_claim=claim,
            verification_thought=hypothesis.get("verification_thought", ""),
            data_needed=json.dumps(hypothesis.get("data_needed", []), ensure_ascii=False),
            expected_if_true=hypothesis.get("expected_if_true", ""),
            expected_if_false=hypothesis.get("expected_if_false", ""),
            data_inventory=data_inventory_text,
            loaded_data_summary=loaded_data_summary,
        )
        
        def _validate_plan(parsed: Dict) -> bool:
            """验证方案结构完整性"""
            if not isinstance(parsed, dict):
                return False
            plan = parsed.get("verification_plan", parsed)
            if not isinstance(plan, dict):
                return False
            # 检查核心字段
            if not plan.get("method_name") and not plan.get("analysis_steps"):
                logger.warning(f"Verification plan for {hyp_id} missing both method_name and analysis_steps")
                return False
            if not plan.get("data_sources") and not plan.get("analysis_steps"):
                logger.warning(f"Verification plan for {hyp_id} missing data_sources and analysis_steps")
                return False
            return True
        
        result = self._call_and_parse_json_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位数据科学家，擅长设计严谨的统计验证方案。"
                "方案必须具体、可执行，使用项目中可用的数据。"
                "如果某些数据不可用，方案应该说明替代方法。"
            ),
            temperature=0.3,
            max_tokens=2048,
            max_retries=2,
            additional_instructions=(
                "输出格式必须包含: verification_plan 对象 "
                "包含 method_name, method_description, data_sources, "
                "analysis_steps, statistical_method, confirm_criteria, refute_criteria"
            ),
            validate_func=_validate_plan,
        )
        
        if result is None:
            logger.error(f"Verification plan generation failed for {hyp_id}")
            return None
        
        # 标准化: 确保 result 有顶层的 verification_plan
        if "verification_plan" not in result:
            result = {"verification_plan": result}
        
        return result
    
    # ════════════════════════════════════════
    # Step 2: 数据准备
    # ════════════════════════════════════════
    
    def _prepare_verification_data(self,
                                    hypothesis: Dict,
                                    verification_plan: Dict,
                                    preloaded_data: Dict) -> Dict:
        """
        为验证准备数据
        
        将:
        1. 已加载的数据 (preloaded_data)
        2. 从数据盘点发现的数据
        3. 验证方案中指定的数据源
        4. 自动计算的缺失数据 (通过 DataComputationEngine)
        5. [新增 v3] 模型内部数据 (通过 ModelProbingEngine)
        
        合为一个统一的数据描述, 供代码生成使用
        
        新增逻辑 (v2):
        - 如果假设需要的数据不在已加载的数据中，自动识别并计算
        - 内置方法优先 (快速、无需 LLM)
        - 内置方法无法满足时，让 LLM 编写计算脚本
        
        新增逻辑 (v3):
        - 对于需要模型内部数据 (注意力权重、隐藏状态等) 的假设，
          DataComputationEngine 无法通过数据处理脚本获取
        - 新增 ModelProbingEngine，通过:
          a. LLM 分析模型源码结构 → 确定探测目标
          b. LLM 生成探测脚本 (使用 PyTorch hooks 或临时模型修改)
          c. 执行探测脚本并收集结果
          d. 失败时自动修正脚本
        - 使假设验证不再因缺少模型内部数据而被标记为"UNVERIFIABLE"
        """
        verification_data = dict(preloaded_data)
        
        # 根据验证方案补充数据
        plan = verification_plan.get("verification_plan", {})
        data_sources = plan.get("data_sources", [])
        data_needed = hypothesis.get("data_needed", [])
        
        # 尝试从数据盘点加载方案需要的额外数据
        extra_data = self.data_inventory.load_data_for_verification(data_needed)
        for key, value in extra_data.items():
            if key not in verification_data:
                verification_data[key] = value
        
        # --- 识别并准备缺失数据 ---
        missing_data = self.data_inventory.identify_missing_data(data_needed, verification_data)
        
        if missing_data:
            hyp_id = hypothesis.get("id", "H?")
            logger.info(f"Hypothesis {hyp_id} missing {len(missing_data)} data items: {missing_data}")
            print(f"    🔄 识别到 {len(missing_data)} 个缺失数据...")
            
            # ── Phase 1: 数据计算引擎 (处理可从数据文件计算的缺失数据) ──
            computable_missing = self.data_inventory.identify_computable_data(missing_data)
            model_internal_missing = self.data_inventory.identify_model_internal_data(missing_data)
            
            if computable_missing:
                print(f"    🔄 可计算数据: {len(computable_missing)} 个, 尝试自动计算...")
                
                # 获取计算引擎
                engine = self.data_inventory.get_computation_engine(llm_client=self.llm)
                
                # 计算缺失数据
                computed_data = engine.compute_needed_data(
                    computable_missing, hypothesis, verification_data
                )
                
                # 将计算结果合并到验证数据中
                for data_name, data_value in computed_data.items():
                    verification_data[data_name] = data_value
                    print(f"    ✅ 计算完成: {data_name}")
            
            # ── Phase 2: 模型探测引擎 (处理需要模型内部数据的缺失数据) ──
            # 重新检查仍缺失的数据 (Phase 1 可能已解决了部分)
            still_missing = self.data_inventory.identify_missing_data(data_needed, verification_data)
            model_internal_remaining = self.data_inventory.identify_model_internal_data(still_missing)
            
            if model_internal_remaining:
                print(f"    🔬 模型内部数据: {len(model_internal_remaining)} 个, 尝试模型探测...")
                
                # 获取模型探测引擎
                probe_engine = self.data_inventory.get_model_probing_engine(llm_client=self.llm)
                
                # 执行模型探测
                probed_data = probe_engine.probe_model_data(
                    model_internal_remaining, hypothesis, verification_data
                )
                
                # 将探测结果合并到验证数据中
                for data_name, data_value in probed_data.items():
                    verification_data[data_name] = data_value
                    print(f"    ✅ 模型探测完成: {data_name}")
            
            # 记录最终仍然缺失的数据
            final_missing = self.data_inventory.identify_missing_data(data_needed, verification_data)
            if final_missing:
                logger.warning(f"Still missing data for {hyp_id}: {final_missing}")
                print(f"    ⚠️ 仍缺失: {final_missing}")
        
        return verification_data
    
    def _prepare_preloaded_data(self,
                                 wrong_text_cases: List[Dict],
                                 all_wrong_cases: List[Dict],
                                 item_popularity: Dict,
                                 overall_metrics: Dict,
                                 surprise_metrics: Dict) -> Dict:
        """
        准备从 core.py 传入的预加载数据
        
        这些数据已经在内存中, 可以直接传给验证代码
        """
        preloaded = {}
        
        if wrong_text_cases:
            preloaded["wrong_text_cases"] = wrong_text_cases
        
        if item_popularity:
            preloaded["item_popularity"] = item_popularity
        
        if overall_metrics:
            preloaded["overall_metrics"] = overall_metrics
        
        if surprise_metrics:
            preloaded["surprise_metrics"] = surprise_metrics
        
        if self.item_text_map:
            preloaded["item_text_map"] = self.item_text_map
        
        return preloaded
    
    # ════════════════════════════════════════
    # Step 3: 验证代码生成
    # ════════════════════════════════════════
    
    def _generate_verification_code(self,
                                    hypothesis: Dict,
                                    verification_plan: Dict,
                                    verification_data: Dict) -> Optional[str]:
        """
        LLM 根据验证方案和可用数据生成验证脚本 (增强版)
        
        新增: 代码质量验证 (检查包含 output 写入 + 统计计算)
        新增: 代码生成失败自动重试
        
        Returns:
            Python script code string, or None
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        
        # 创建临时输出文件路径
        output_dir = os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"result_{hyp_id}.json")
        
        # 构建可用数据描述
        available_data_desc = self._format_available_data_for_code(verification_data)
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        
        prompt = VERIFICATION_CODE_PROMPT.format(
            hypothesis_claim=claim,
            verification_plan_json=plan_json,
            available_data_description=available_data_desc,
            output_file_path=output_file,
            hypothesis_id=hyp_id,
        )
        
        system_content = (
            "你是一位 Python 数据科学家，擅长编写独立的数据验证脚本。"
            "代码必须稳健、能处理异常、只依赖标准库和 numpy/scipy。"
            "结果必须以 JSON 格式输出到指定文件。"
            "确保你使用 save_result() 函数将结果写入输出文件。"
        )
        
        max_attempts = 3  # 最多尝试 3 次
        last_error = ""
        
        for attempt in range(max_attempts):
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        system_content + (
                            f"\n\n## 上一次尝试的问题\n{last_error}"
                            if last_error else ""
                        )
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2 if attempt == 0 else 0.3,
                max_tokens=4096,
            )
            
            if response is None:
                logger.error(f"Verification code generation failed (attempt {attempt+1})")
                continue
            
            # 清理代码
            code = self._clean_code_response(response)
            
            # 代码质量验证
            validation_errors = self._validate_verification_code(code, output_file)
            
            if not validation_errors:
                # 在代码开头注入数据加载逻辑
                code = self._inject_data_loading(code, verification_data, output_file)
                return code
            
            last_error = "; ".join(validation_errors)
            logger.warning(
                f"Generated code validation failed (attempt {attempt+1}): {last_error}"
            )
            
            if attempt < max_attempts - 1:
                prompt += (
                    f"\n\n## 修正要求\n上一版代码存在以下问题, 请修正:\n"
                    + "\n".join(f"- {e}" for e in validation_errors)
                )
        
        logger.error(f"Verification code generation failed after {max_attempts} attempts")
        return None
    
    @staticmethod
    def _validate_verification_code(code: str, expected_output_file: str) -> List[str]:
        """
        验证生成的代码是否满足基本质量要求
        
        Returns:
            List of validation error messages (empty = valid)
        """
        errors = []
        output_filename = os.path.basename(expected_output_file)
        
        # 1. 检查包含输出文件路径
        if output_filename not in code and expected_output_file not in code:
            errors.append(f"代码中未包含输出文件路径 ({output_filename})")
        
        # 2. 检查使用了 save_result 或 json.dump 来写入结果
        has_save = ("save_result" in code or "json.dump" in code or "json.dumps" in code)
        has_output = ("OUTPUT_FILE" in code or output_filename in code)
        if not has_save and not has_output:
            errors.append("代码中未找到结果保存逻辑 (save_result/json.dump)")
        
        # 3. 检查没有循环 import 或错误代码模式
        if "import subprocess" in code or "import os" not in code:
            pass  # subprocess 可用, os 是必需的但可能写在头部中
        
        # 4. 检查代码包含基本的统计计算
        if not any(kw in code for kw in ["Counter", "statistics", "mean", "sum", "len",
                                          "count", "ratio", "histogram", "统计"]):
            if "statistics" in code or "analysis" in code.lower():
                pass  # 可能是自定义分析
            else:
                errors.append("代码中未检测到统计计算逻辑")
        
        # 5. 检查代码有 try/except 保护
        if "try:" not in code or "except" not in code:
            errors.append("代码缺少异常处理 (try/except)")
        
        return errors
    
    def _format_available_data_for_code(self, verification_data: Dict) -> str:
        """
        格式化可用数据描述, 让 LLM 知道代码中可以使用哪些数据
        
        关键设计: 数据将通过变量注入到代码中, LLM 只需使用变量名
        
        新增 (v2): 对计算生成的派生数据, 展示其统计摘要和样本,
        让 LLM 更好地理解数据结构和含义
        """
        lines = []
        lines.append("以下数据已经作为 Python 变量注入到脚本中, 直接使用即可:")
        lines.append("")
        
        # 已知的数据类型列表 (内置计算生成的数据)
        KNOWN_COMPUTED_DATA = {
            "category_overlap_stats": "目标物品与用户历史序列的类别重叠统计",
            "category_distribution": "物品类别分布 (各顶层类别的物品数量)",
            "wrong_case_category_stats": "误推案例中目标物品的类别分布 vs 全量类别分布",
            "recommendation_frequency": "模型推荐结果中各物品的推荐频次",
            "sequence_target_mapping": "用户历史序列与目标物品的关联数据",
            "item_interaction_freq": "训练数据中各物品的交互频率统计",
        }
        
        # 模型探测数据类型列表 (v3 新增)
        KNOWN_MODEL_PROBED_DATA = {
            "attention_weights": "模型自注意力权重矩阵 (通过模型探测提取)",
            "attention_entropy": "注意力分布熵 (衡量注意力集中程度)",
            "hidden_states": "Transformer编码器各层的隐藏状态输出",
            "item_embeddings": "物品嵌入向量矩阵",
            "model_predictions": "模型对物品的预测分数",
            "gradient_info": "模型梯度信息",
        }
        
        for key, value in verification_data.items():
            # 对于计算生成的数据, 显示更详细的描述
            if key in KNOWN_COMPUTED_DATA and isinstance(value, dict):
                lines.append(f"- `{key}`: **{KNOWN_COMPUTED_DATA[key]}**")
                if "statistics" in value:
                    stats = value["statistics"]
                    lines.append(f"  统计摘要:")
                    for stat_key, stat_val in stats.items():
                        if isinstance(stat_val, (int, float, str)):
                            lines.append(f"    {stat_key}: {stat_val}")
                        elif isinstance(stat_val, dict) and len(stat_val) <= 10:
                            for sk, sv in list(stat_val.items())[:5]:
                                lines.append(f"    {stat_key}.{sk}: {sv}")
                        elif isinstance(stat_val, list) and len(stat_val) <= 5:
                            lines.append(f"    {stat_key}: {json.dumps(stat_val, ensure_ascii=False)[:150]}")
                if "sample" in value:
                    sample = value["sample"]
                    if isinstance(sample, dict):
                        lines.append(f"  样本: {json.dumps(sample, ensure_ascii=False)[:200]}")
                    elif isinstance(sample, list):
                        lines.append(f"  样本: {json.dumps(sample[:2], ensure_ascii=False)[:200]}")
                # 说明可访问的子字段
                if "per_case_overlap" in value:
                    lines.append(f"  每条记录格式: {json.dumps(value['per_case_overlap'][0] if value['per_case_overlap'] else {}, ensure_ascii=False)[:200]}")
                if "per_case_mapping" in value:
                    lines.append(f"  每条记录格式: {json.dumps(value['per_case_mapping'][0] if value['per_case_mapping'] else {}, ensure_ascii=False)[:200]}")
                if "frequency_dict" in value:
                    lines.append(f"  频次字典: {len(value['frequency_dict'])} 个物品")
                if "freq_dict" in value:
                    lines.append(f"  频次字典: {len(value['freq_dict'])} 个物品")
            
            # 对于模型探测数据, 显示更详细的描述 (v3 新增)
            elif key in KNOWN_MODEL_PROBED_DATA and isinstance(value, dict):
                lines.append(f"- `{key}`: **{KNOWN_MODEL_PROBED_DATA[key]}** (模型探测提取)")
                if "statistics" in value:
                    stats = value["statistics"]
                    lines.append(f"  统计摘要:")
                    for stat_key, stat_val in stats.items():
                        if isinstance(stat_val, (int, float, str)):
                            lines.append(f"    {stat_key}: {stat_val}")
                        elif isinstance(stat_val, dict) and len(stat_val) <= 10:
                            for sk, sv in list(stat_val.items())[:5]:
                                lines.append(f"    {stat_key}.{sk}: {sv}")
                        elif isinstance(stat_val, list) and len(stat_val) <= 5:
                            lines.append(f"    {stat_key}: {json.dumps(stat_val, ensure_ascii=False)[:150]}")
                if "sample" in value:
                    sample = value["sample"]
                    if isinstance(sample, dict):
                        lines.append(f"  样本: {json.dumps(sample, ensure_ascii=False)[:200]}")
                    elif isinstance(sample, list):
                        lines.append(f"  样本: {json.dumps(sample[:2], ensure_ascii=False)[:200]}")
                if "computation_method" in value:
                    lines.append(f"  提取方法: {value['computation_method']}")
            
            elif isinstance(value, list):
                lines.append(f"- `{key}`: List, 长度 {len(value)}")
                if value and isinstance(value[0], dict):
                    sample_keys = list(value[0].keys())[:5]
                    lines.append(f"  每个元素的 keys: {sample_keys}")
            elif isinstance(value, dict):
                lines.append(f"- `{key}`: Dict, {len(value)} 条记录")
                if value:
                    sample_key = list(value.keys())[0]
                    lines.append(f"  样本 key: {sample_key}")
            elif isinstance(value, str):
                lines.append(f"- `{key}`: 文件路径字符串 `{value}`")
            else:
                lines.append(f"- `{key}`: {type(value).__name__}")
        
        return "\n".join(lines)
    
    def _inject_data_loading(self, code: str, verification_data: Dict,
                              output_file: str) -> str:
        """
        在代码开头注入数据加载逻辑
        
        策略: 将 preloaded 的数据序列化为 JSON 文件, 让代码从文件加载
        (比直接注入 Python 变量更稳健, 避免 eval 安全问题)
        """
        # 将数据保存为 JSON 文件
        data_dir = os.path.join(self.log_dir, "verification_scripts", "data")
        os.makedirs(data_dir, exist_ok=True)
        data_file = os.path.join(data_dir, "preloaded_data.json")
        
        # 序列化数据 (大列表只保存统计摘要; 计算数据保留完整内容但截断过大字段)
        serializable = {}
        for key, value in verification_data.items():
            try:
                if isinstance(value, list) and len(value) > 100:
                    # 大列表只保存前 50 条作为样本
                    serializable[key + "_sample"] = value[:50]
                    serializable[key + "_count"] = len(value)
                elif isinstance(value, dict) and key in [
                    "category_overlap_stats", "category_distribution",
                    "wrong_case_category_stats", "recommendation_frequency",
                    "sequence_target_mapping", "item_interaction_freq",
                ]:
                    # 计算数据: 保留完整内容, 但截断过大的 per_case 字段
                    computed_dict = dict(value)
                    for subkey in ["per_case_overlap", "per_case_mapping",
                                   "freq_dict", "frequency_dict"]:
                        if subkey in computed_dict:
                            sub_data = computed_dict[subkey]
                            if isinstance(sub_data, list) and len(sub_data) > 200:
                                computed_dict[subkey + "_total"] = len(sub_data)
                                computed_dict[subkey] = sub_data[:200]
                            elif isinstance(sub_data, dict) and len(sub_data) > 2000:
                                computed_dict[subkey + "_total"] = len(sub_data)
                                items = list(sub_data.items())[:2000]
                                computed_dict[subkey] = dict(items)
                    serializable[key] = computed_dict
                else:
                    serializable[key] = value
            except (TypeError, ValueError):
                serializable[key + "_info"] = f"无法序列化: {type(value).__name__}"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(serializable, f, ensure_ascii=False, indent=2)
        
        # 构建注入头部
        injection_lines = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            f"# Verification script auto-generated by HypothesisVerificationAgent",
            "",
            "import json",
            "import os",
            "import sys",
            "import numpy as np",
            "from collections import Counter, defaultdict",
            "from typing import Dict, List",
            "",
            "# ── 加载预置数据 ──",
            f'data_file = "{data_file}"',
            "try:",
            "    with open(data_file, 'r', encoding='utf-8') as f:",
            "        _preloaded = json.load(f)",
            "except Exception as e:",
            "    _preloaded = {}",
            "    print(f'Warning: Failed to load preloaded data: {e}')",
            "",
            f'OUTPUT_FILE = "{output_file}"',
            "",
            "# ── 解析预置数据为变量 ──",
        ]
        
        # 将 preloaded 数据中的每个 key 拆分为变量
        # 重要: 当大列表被截断为 _sample 时, 必须同时创建原始变量名,
        # 让原始变量名指向样本数据, 这样 LLM 生成的代码引用原始变量名时不会报 NameError
        for key, value in verification_data.items():
            if key in serializable:
                injection_lines.append(f'{key} = _preloaded.get("{key}", None)')
            elif key + "_sample" in serializable:
                injection_lines.append(f'{key}_sample = _preloaded.get("{key}_sample", [])')
                injection_lines.append(f'{key}_count = _preloaded.get("{key}_count", 0)')
                # 关键修复: 原始变量名也指向样本数据, 避免引用原始变量名时 NameError
                injection_lines.append(f'{key} = {key}_sample')
        
        injection_lines.extend([
            "",
            "# ── 辅助函数: 安全保存结果 ──",
            "def save_result(result_dict):",
            "    try:",
            "        os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)",
            "        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:",
            "            json.dump(result_dict, f, ensure_ascii=False, indent=2)",
            "    except Exception as e:",
            "        # 如果保存失败, 写入 stderr",
            "        sys.stderr.write(f'Failed to save result: {e}\\n')",
            "        sys.stderr.write(json.dumps(result_dict, ensure_ascii=False) + '\\n')",
            "",
            "",
        ])
        
        return "\n".join(injection_lines) + "\n" + code
    
    # ════════════════════════════════════════
    # Step 4: 代码执行 (带修正循环)
    # ════════════════════════════════════════
    
    def _execute_with_correction_loop(self,
                                       initial_code: str,
                                       hypothesis: Dict,
                                       verification_plan: Dict,
                                       verification_data: Dict) -> Optional[Dict]:
        """
        执行验证代码, 如果失败则修正重试
        
        类似 core.py 中 _fix_code_error 的思路:
        执行 → 检查错误 → 带错误信息让 LLM 修正 → 再执行
        
        Max rounds: self.MAX_CODE_FIX_ROUNDS (默认 3)
        """
        hyp_id = hypothesis.get("id", "H?")
        code = initial_code
        
        # 保存脚本到文件
        script_dir = os.path.join(self.log_dir, "verification_scripts")
        os.makedirs(script_dir, exist_ok=True)
        script_file = os.path.join(script_dir, f"verify_{hyp_id}.py")
        
        for round_num in range(self.MAX_CODE_FIX_ROUNDS):
            # 保存当前代码
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            logger.info(f"Executing verification script for {hyp_id} (round {round_num + 1})")
            print(f"      ⚡ 执行脚本 (round {round_num + 1}/{self.MAX_CODE_FIX_ROUNDS})...")
            
            # 执行脚本
            success, result, error = self._execute_script(script_file)
            
            if success and result is not None:
                logger.info(f"Verification script executed successfully for {hyp_id}")
                return result
            
            if error:
                logger.warning(f"Verification script failed (round {round_num + 1}): {error[:200]}")
                print(f"      ❌ 执行失败: {error[:100]}...")
                
                if round_num < self.MAX_CODE_FIX_ROUNDS - 1:
                    # 修正代码
                    print(f"      🔁 让 LLM 修正代码...")
                    fixed_code = self._fix_verification_code(code, error, hypothesis)
                    if fixed_code:
                        code = fixed_code
                    else:
                        logger.error(f"Code fix failed for {hyp_id}")
                        break
                else:
                    logger.error(f"Max code fix rounds reached for {hyp_id}")
        
        return None
    
    def _execute_script(self, script_path: str) -> Tuple[bool, Optional[Dict], Optional[str]]:
        """
        执行验证脚本 (增强版)
        
        增强: 多模式输出文件路径检测
        增强: stdout JSON 健壮解析
        增强: 结果结构验证
        
        Returns:
            (success, result_dict, error_message)
        """
        try:
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=self.MAX_EXECUTION_TIMEOUT,
                cwd=self.project_root,
            )
            
            if result.returncode != 0:
                error = result.stderr or result.stdout or "Unknown execution error"
                return False, None, error
            
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # ── 读取结果并验证 ──
            # 查找输出文件 — 多模式匹配
            output_file_path = self._extract_output_path(script_content)
            
            if output_file_path and os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                # 关键修复: 检测结果中的 "error" 字段
                # LLM 生成的验证脚本通常用 try/except 捕获异常后写入 {"error": "..."} 
                # 此时退出码为 0 但结果实际是错误, 不能当作成功返回
                if isinstance(result_data, dict) and "error" in result_data:
                    error_msg = result_data["error"]
                    logger.warning(f"Verification script wrote error-result (exitcode=0): {error_msg[:200]}")
                    return False, None, f"Script error-result: {error_msg}"
                return True, result_data, None
            
            # 输出文件不存在 → 尝试从 stdout/stderr 中提取 JSON
            for source_name, source_text in [
                ("stderr", result.stderr),
                ("stdout", result.stdout),
            ]:
                if not source_text or not source_text.strip():
                    continue
                
                # 使用 _robust_json_parse 解析 stdout 中的 JSON
                json_str = self._extract_json_block(source_text)
                if json_str:
                    parsed = self._robust_json_parse(json_str)
                    if parsed is not None:
                        # 同样检测 "error" 字段 — stdout/stderr 中的错误结果也不应视为成功
                        if isinstance(parsed, dict) and "error" in parsed:
                            error_msg = parsed["error"]
                            logger.warning(f"Verification script wrote error-result to {source_name}: {error_msg[:200]}")
                            return False, None, f"Script error-result ({source_name}): {error_msg}"
                        logger.info(f"Result extracted from {source_name}")
                        return True, parsed, None
                
                # 回退: 逐行查找 JSON
                for line in source_text.split("\n"):
                    stripped = line.strip()
                    if stripped and stripped.startswith("{"):
                        parsed = self._robust_json_parse(stripped)
                        if parsed is not None:
                            # 同样检测 "error" 字段
                            if isinstance(parsed, dict) and "error" in parsed:
                                error_msg = parsed["error"]
                                return False, None, f"Script error-result ({source_name}): {error_msg}"
                            logger.info(f"Result extracted from single line in {source_name}")
                            return True, parsed, None
            
            error_msg = "No output file found"
            if output_file_path:
                error_msg += f": {output_file_path}"
            if result.stderr:
                error_msg += f" | stderr: {result.stderr[:200]}"
            return False, None, error_msg
        
        except subprocess.TimeoutExpired:
            return False, None, f"Script execution timed out ({self.MAX_EXECUTION_TIMEOUT}s)"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
    
    @staticmethod
    def _extract_output_path(script_content: str) -> Optional[str]:
        """从脚本代码中提取输出路径 — 多模式匹配"""
        import re
        patterns = [
            # OUTPUT_FILE = "path"
            r'OUTPUT_FILE\s*=\s*["\']([^"\']+)["\']',
            # output_file = "path"  
            r'output_file\s*=\s*["\']([^"\']+)["\']',
            # result_path = "path"
            r'result_path\s*=\s*["\']([^"\']+)["\']',
            # open("path/write", "w") 模式
            r'open\(["\']([^"\']*result[^"\']*\.json)["\']',
            # json.dump(..., open("path", "w"))
            r'json\.dump\([^,]+,\s*open\(["\']([^"\']+\.json)["\']',
        ]
        for pattern in patterns:
            match = re.search(pattern, script_content)
            if match:
                return match.group(1)
        return None
    
    def _fix_verification_code(self, original_code: str, error: str,
                               hypothesis: Dict) -> Optional[str]:
        """
        让 LLM 修正验证代码
        
        Args:
            original_code: 原始代码
            error: 执行错误信息
            hypothesis: 当前假设
            
        Returns:
            Fixed code string, or None
        """
        hyp_id = hypothesis.get("id", "H?")
        
        # 从代码中提取输出路径 (需要保留)
        import re
        output_match = re.search(r'OUTPUT_FILE\s*=\s*"([^"]+)"', original_code)
        output_path = output_match.group(1) if output_match else ""
        
        # 去掉注入的数据加载头部 (保留修正后的核心逻辑)
        # 找到核心逻辑的起始位置 (跳过注入的头部)
        core_code = self._extract_core_code(original_code)
        
        prompt = VERIFICATION_CODE_FIX_PROMPT.format(
            original_code=core_code,
            error_output=error[:1500],  # 截断过长错误
            output_file_path=output_path,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位 Python 专家，擅长根据错误信息修正代码。"
                    "只修正导致错误的部分，保持其他逻辑不变。"
                    "确保代码仍然将结果写入指定 JSON 文件。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
        )
        
        if response is None:
            return None
        
        fixed_core = self._clean_code_response(response)
        
        # 重新注入数据加载头部
        # 需要重建 verification_data (从之前的保存文件加载)
        data_dir = os.path.join(self.log_dir, "verification_scripts", "data")
        data_file = os.path.join(data_dir, "preloaded_data.json")
        
        # 重新构建头部
        injection_lines = [
            "#!/usr/bin/env python3",
            "# -*- coding: utf-8 -*-",
            f"# Verification script auto-generated by HypothesisVerificationAgent (fixed)",
            "",
            "import json",
            "import os",
            "import sys",
            "import numpy as np",
            "from collections import Counter, defaultdict",
            "from typing import Dict, List",
            "",
            "# ── 加载预置数据 ──",
            f'data_file = "{data_file}"',
            "try:",
            "    with open(data_file, 'r', encoding='utf-8') as f:",
            "        _preloaded = json.load(f)",
            "except Exception as e:",
            "    _preloaded = {}",
            "",
            f'OUTPUT_FILE = "{output_path}"',
            "",
            "# ── 从 _preloaded 中提取变量 ──",
            "# (自动注入, 与数据文件中的 keys 对应)",
            "for _key, _value in _preloaded.items():",
            "    globals()[_key] = _value",
            "",
            "# ── 辅助函数: 安全保存结果 ──",
            "def save_result(result_dict):",
            "    try:",
            "        os.makedirs(os.path.dirname(OUTPUT_FILE) or '.', exist_ok=True)",
            "        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:",
            "            json.dump(result_dict, f, ensure_ascii=False, indent=2)",
            "    except Exception as e:",
            "        sys.stderr.write(f'Failed to save result: {e}\\n')",
            "        sys.stderr.write(json.dumps(result_dict, ensure_ascii=False) + '\\n')",
            "",
            "",
        ]
        
        return "\n".join(injection_lines) + "\n" + fixed_core
    
    def _extract_core_code(self, full_code: str) -> str:
        """
        从完整脚本中提取核心验证逻辑 (去掉注入的头部)
        """
        # 找到 "def save_result" 后的空行之后的代码
        # 或者找到第一个非注入的代码行
        
        lines = full_code.split("\n")
        core_start = 0
        
        # 搜索核心逻辑的起始标记
        for i, line in enumerate(lines):
            # 如果行包含 save_result 定义结束后的空行
            if line.strip() == "" and i > 0:
                # 检查之前的行是否是 save_result 函数的一部分
                prev_lines = [l.strip() for l in lines[max(0,i-5):i]]
                if any("sys.stderr" in l for l in prev_lines):
                    core_start = i + 1
                    break
        
        # 如果没找到标记, 尝试找第一个非 import/注释/空行的代码
        if core_start == 0:
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped and not stripped.startswith("#") and \
                   not stripped.startswith("import ") and \
                   not stripped.startswith("from ") and \
                   not stripped.startswith("OUTPUT_FILE") and \
                   not stripped.startswith("data_file") and \
                   not stripped.startswith("_preloaded") and \
                   not stripped.startswith("def save_result") and \
                   not stripped.startswith('globals()'):
                    core_start = i
                    break
        
        if core_start > 0:
            return "\n".join(lines[core_start:])
        return full_code
    
    # ════════════════════════════════════════
    # Step 5: 结果分析
    # ════════════════════════════════════════
    
    def _analyze_results(self,
                         hypothesis: Dict,
                         verification_plan: Dict,
                         execution_result: Dict) -> Dict:
        """
        LLM 解读验证代码的执行结果, 判断假设是否成立 (增强版)
        
        新增: 使用 _call_and_parse_json_with_retry 处理 JSON 解析失败
        新增: 结果结构验证 + 状态有效性检查
        新增: 从非结构化输出中提取部分状态
        
        Returns:
            verification result dict with status, brief, evidence, etc.
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        expected_true = hypothesis.get("expected_if_true", "")
        expected_false = hypothesis.get("expected_if_false", "")
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        result_json = json.dumps(execution_result, indent=2, ensure_ascii=False)
        
        if len(result_json) > 3000:
            result_json = result_json[:3000] + "\n... (截断)"
        
        prompt = RESULT_ANALYSIS_PROMPT.format(
            hypothesis_id=hyp_id,
            hypothesis_claim=claim,
            expected_if_true=expected_true,
            expected_if_false=expected_false,
            verification_plan_json=plan_json,
            execution_result_json=result_json,
        )
        
        VALID_STATUSES = {"CONFIRMED", "PARTIALLY_CONFIRMED", "REFUTED", "UNVERIFIABLE"}
        
        def _validate_result(parsed: Dict) -> bool:
            """验证分析结果结构"""
            if not isinstance(parsed, dict):
                return False
            status = parsed.get("status", "")
            if status not in VALID_STATUSES:
                logger.warning(f"Invalid status '{status}' in result analysis")
                return False
            if not parsed.get("brief") and not parsed.get("detailed_reasoning"):
                logger.warning("Result analysis missing brief and detailed_reasoning")
                return False
            return True
        
        parsed = self._call_and_parse_json_with_retry(
            prompt=prompt,
            system_content=(
                "你是一位严谨的数据科学家，擅长根据统计结果判断假设是否成立。"
                "你必须给出明确判断 (不能模棱两可)。"
                "判断基于数据，而非直觉。如果数据不支持假设, 就判为 REFUTED。"
            ),
            temperature=0.2,
            max_tokens=1024,
            max_retries=2,
            additional_instructions=(
                "输出格式必须包含: status (CONFIRMED|PARTIALLY_CONFIRMED|REFUTED|UNVERIFIABLE), "
                "brief, detailed_reasoning, evidence_summary"
            ),
            validate_func=_validate_result,
        )
        
        if parsed is not None:
            # 补充 evidence 字段
            if "evidence" not in parsed:
                parsed["evidence"] = execution_result.get("statistics", execution_result)
            parsed["method"] = "agent_autonomous"
            return parsed
        
        # 重试耗尽 → 尝试从原始响应中提取部分状态
        logger.warning(f"Result analysis failed for {hyp_id}, trying partial status extraction")
        partial_status = self._extract_partial_status_from_response(
            claim, execution_result
        )
        if partial_status:
            return partial_status
        
        # 完全失败
        return {
            "status": self.UNVERIFIABLE,
            "reason": "LLM 分析结果解析失败 (重试耗尽)",
            "brief": "分析结果解析失败",
            "evidence": execution_result,
            "method": "agent_autonomous",
        }
    
    @staticmethod
    def _extract_partial_status_from_response(claim: str,
                                               execution_result: Dict) -> Optional[Dict]:
        """
        从无法解析的 LLM 响应中提取部分分析结果
        
        检查执行结果中的统计量, 提供最基础的判断
        """
        statistics = execution_result.get("statistics", execution_result)
        if not isinstance(statistics, dict):
            return None
        
        # 如果执行结果本身已经包含有意义的统计数据, 作为 UNVERIFIABLE 返回
        if statistics:
            return {
                "status": "UNVERIFIABLE",
                "reason": "LLM 分析结果格式错误, 但原始统计数据可用",
                "brief": "结果分析格式错误, 请查看原始统计数据",
                "evidence": statistics,
                "method": "agent_autonomous",
            }
        
        return None
    
    # ════════════════════════════════════════
    # Fallback: 旧版验证方法
    # ══════════════════════════
    
    def _try_fallback_verification(self,
                                    hypothesis: Dict,
                                    wrong_text_cases: List[Dict],
                                    all_wrong_cases: List[Dict],
                                    item_popularity: Dict,
                                    overall_metrics: Dict,
                                    surprise_metrics: Dict) -> Dict:
        """
        当 Agent 验证完全失败时, 尝试旧版固定方法验证
        
        将 hypothesis 中的 verification_thought 转换为旧版的 verification_method
        """
        claim = hypothesis.get("claim", "")
        thought = hypothesis.get("verification_thought", "")
        
        # 尝试将自由描述映射到旧版固定方法
        method = self._infer_verification_method(claim, thought)
        hypothesis_copy = dict(hypothesis)
        hypothesis_copy["verification_method"] = method
        
        # 计算旧版统计基线
        baseline = self._fallback_verifier._compute_stats_baseline(
            wrong_text_cases, all_wrong_cases, item_popularity
        )
        
        try:
            if method == "item_popularity":
                result = self._fallback_verifier._verify_item_popularity(hypothesis_copy, baseline)
            elif method == "category_bias":
                result = self._fallback_verifier._verify_category_bias(hypothesis_copy, baseline)
            elif method == "sequence_length":
                result = self._fallback_verifier._verify_sequence_length(hypothesis_copy, baseline)
            elif method == "similarity_bias":
                result = self._fallback_verifier._verify_similarity_bias(hypothesis_copy, baseline, wrong_text_cases)
            elif method == "surprise_score":
                result = self._fallback_verifier._verify_surprise_score(
                    hypothesis_copy, baseline, overall_metrics, surprise_metrics)
            else:
                result = self._fallback_verifier._verify_custom(hypothesis_copy, baseline, wrong_text_cases)
            
            result["method"] = "fallback_fixed"
            result["fallback_reason"] = "Agent autonomous verification failed, using fixed method"
            return result
        
        except Exception as e:
            logger.error(f"Fallback verification also failed: {e}")
            return {
                "status": self.UNVERIFIABLE,
                "reason": f"Both agent and fallback verification failed: {str(e)}",
                "brief": "验证完全失败",
                "evidence": None,
            }
    
    def _infer_verification_method(self, claim: str, thought: str) -> str:
        """
        从假设的自由描述推断最匹配的旧版固定方法
        
        当 Agent 验证失败需要 fallback 时使用
        """
        text = (claim + " " + thought).lower()
        
        if any(kw in text for kw in ["热度", "popularity", "冷门", "热门", "频次", "交互次数", "cold", "hot", "unpopular", "popular"]):
            return "item_popularity"
        if any(kw in text for kw in ["类别", "category", "跨类别", "类别集中", "类型偏差"]):
            return "category_bias"
        if any(kw in text for kw in ["序列长度", "sequence length", "短序列", "长序列", "历史长度"]):
            return "sequence_length"
        if any(kw in text for kw in ["相似性", "similarity", "相似度", "嵌入", "余弦"]):
            return "similarity_bias"
        if any(kw in text for kw in ["惊喜", "surprise", "差异大", "偏离"]):
            return "surprise_score"
        
        return "custom"
    
    # ════════════════════════════════════════
    # Phase 3: 生成验证报告 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def generate_verification_report(self, verified_hypotheses: List[Dict]) -> Dict:
        """
        生成验证报告 (与旧版 HypothesisVerifier.generate_verification_report 一致)
        
        将每个假设的验证结果汇总, 并标注:
        - 哪些 LLM 结论被数据确认
        - 哪些被数据反驳
        - 哪些无法验证
        """
        confirmed = []
        partially_confirmed = []
        refuted = []
        unverifiable = []
        
        for hyp in verified_hypotheses:
            result = hyp.get("verification_result", {})
            status = result.get("status", self.UNVERIFIABLE)
            
            entry = {
                "id": hyp.get("id", "?"),
                "claim": hyp.get("claim", ""),
                "source_field": hyp.get("source_field", ""),
                "verification_method": result.get("method", "agent_autonomous"),
                "status": status,
                "brief": result.get("brief", ""),
                "evidence": result.get("evidence"),
            }
            
            if status == self.CONFIRMED:
                confirmed.append(entry)
            elif status == self.PARTIALLY_CONFIRMED:
                partially_confirmed.append(entry)
            elif status == self.REFUTED:
                refuted.append(entry)
            else:
                unverifiable.append(entry)
        
        # 计算置信度分数
        total = len(verified_hypotheses)
        confirmed_pct = len(confirmed) / total * 100 if total > 0 else 0
        refuted_pct = len(refuted) / total * 100 if total > 0 else 0
        
        # 生成建议
        recommendations = []
        for r in refuted:
            recommendations.append(
                f"⚠ 假设 {r['id']} ({r['claim'][:50]}) 被数据反驳, "
                f"基于此结论的改进建议需要重新审视"
            )
        
        report = {
            "total_hypotheses": total,
            "confirmed": confirmed,
            "confirmed_count": len(confirmed),
            "partially_confirmed": partially_confirmed,
            "partially_confirmed_count": len(partially_confirmed),
            "refuted": refuted,
            "refuted_count": len(refuted),
            "unverifiable": unverifiable,
            "unverifiable_count": len(unverifiable),
            "confirmed_pct": confirmed_pct,
            "refuted_pct": refuted_pct,
            "overall_credibility": (
                "HIGH" if confirmed_pct > 60 else
                "MODERATE" if confirmed_pct > 30 else
                "LOW"
            ),
            "recommendations": recommendations,
            "verified_hypotheses": verified_hypotheses,
            "verification_agent_used": True,  # 标记使用了新版 Agent
        }
        
        # 打印摘要
        print(f"\n  ══════════ 假设验证报告 (Agent 自主验证) ══════════")
        print(f"  总假设数: {total}")
        print(f"  ✅ 已确认: {len(confirmed)} ({confirmed_pct:.1f}%)")
        print(f"  ⚠️ 部分确认: {len(partially_confirmed)}")
        print(f"  ❌ 已反驳: {len(refuted)} ({refuted_pct:.1f}%)")
        print(f"  🔍 无法验证: {len(unverifiable)}")
        print(f"  综合可信度: {report['overall_credibility']}")
        if refuted:
            print(f"  反驳的结论:")
            for r in refuted:
                print(f"    ❌ {r['id']}: {r['brief']}")
        print(f"  ══════════════════════════════════\n")
        
        return report
    
    # ════════════════════════════════════════
    # 将验证结果应用到分析 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def apply_verification_to_analysis(self,
                                        llm_analysis: Dict,
                                        verification_report: Dict) -> Dict:
        """
        将验证结果应用到 LLM 分析结论 (与旧版接口一致)
        
        策略:
        1. 被反驳的结论 → 标注为 REFUTED, 降低其权重
        2. 被确认的结论 → 标注为 CONFIRMED, 增强其可信度
        3. 无法验证的结论 → 保持原状, 但标注为 UNVERIFIABLE
        
        Returns:
            增强后的分析结果
        """
        # 直接使用旧版方法 (逻辑完全相同)
        return self._fallback_verifier.apply_verification_to_analysis(
            llm_analysis, verification_report
        )
    
    # ════════════════════════════════════════
    # 保存验证报告 (与旧版接口一致)
    # ════════════════════════════════════════
    
    def save_verification_report(self, report: Dict, output_path: str):
        """保存验证报告"""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"Saved verification report to {output_path}")
    
    # ════════════════════════════════════════
    # 兼容方法: compute_item_popularity_from_data
    # ════════════════════════════════════════
    
    def compute_item_popularity_from_data(self, train_data) -> Dict:
        """从训练数据计算物品热度分布 (与旧版一致)"""
        return self._fallback_verifier.compute_item_popularity_from_data(train_data)
    
    # ════════════════════════════════════════
    # 辅助方法
    # ════════════════════════════════════════
    
    def _parse_hypothesis_response(self, response: str) -> Optional[Dict]:
        """
        解析 LLM 假设提取回复 (增强版)
        
        多策略:
        1. 提取 JSON block, 尝试健壮解析
        2. 如解析成功, 验证假设结构完整性
        3. 返回 validated dict
        """
        # 提取原始 JSON 字符串
        json_str = self._extract_json_block(response)
        if json_str is None:
            logger.warning("Cannot extract JSON from hypothesis extraction response")
            return None
        
        # 健壮解析
        parsed = self._robust_json_parse(json_str)
        if parsed is not None:
            # 结构验证
            validated = self._validate_hypotheses_structure(parsed)
            if validated is not None:
                return validated
            else:
                logger.warning("JSON parsed but structure validation failed")
        else:
            logger.warning("All JSON parsing strategies failed")
        
        # 如果完全解析失败, 尝试从 raw response 中提取部分假设
        partial = self._extract_partial_hypotheses(response)
        if partial:
            logger.info(f"Recovered {len(partial)} hypotheses from partial extraction")
            return {"hypotheses": partial, "summary": "部分恢复的假设 (存在格式问题)"}
        
        return None
    
    @staticmethod
    def _extract_json_block(response: str) -> Optional[str]:
        """从 LLM 回复中提取 JSON block (不引入顶层变量)"""
        import re
        
        # 优先提取 markdown code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            return json_match.group(1)
        
        # 回退: 找第一个 { 到最后 }
        start = response.find('{')
        end = response.rfind('}')
        if start >= 0 and end > start:
            return response[start:end + 1]
        
        return None
    
    @staticmethod
    def _robust_json_parse(json_str: str) -> Optional[Dict]:
        """
        多策略 JSON 解析, 带模糊修复
        
        Strategy 1: 标准 json.loads
        Strategy 2: ast.literal_eval
        Strategy 3: 修复常见格式问题后重试
        Strategy 4: 修复更复杂的问题 (缺失键引号等)
        """
        # --- Strategy 1: 标准解析 ---
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 2: Python literal ---
        try:
            import ast
            parsed = ast.literal_eval(json_str)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        
        # --- Strategy 3: 修复常见 JSON 格式问题 ---
        fixed = json_str
        
        # 3a: 移除注释 (// 和 /* */)
        fixed = re.sub(r'//[^\n]*', '', fixed)
        fixed = re.sub(r'/\*.*?\*/', '', fixed, flags=re.DOTALL)
        
        # 3b: 移除尾随逗号 (在 } 和 ] 之前)
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # 3c: 将 Python None/True/False 转为 JSON null/true/false
        fixed = fixed.replace(': None', ': null')
        fixed = fixed.replace(': True', ': true')
        fixed = fixed.replace(': False', ': false')
        
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 4: 修复缺失引号的键 ---
        # 匹配: {key: value} → {"key": value} 和 {key: "value"} → {"key": "value"}
        fixed2 = re.sub(
            r'(?<![:"\w])([a-zA-Z_][a-zA-Z0-9_]*)\s*:',
            r'"\1":',
            fixed
        )
        # 但不能把 "hypotheses": [ 中的 " 也加上引号
        # re.sub 已经会跳过已引号的键, 因为用了 (?<![:"\w]) 环视
        try:
            return json.loads(fixed2)
        except json.JSONDecodeError:
            pass
        
        # --- Strategy 5: 单引号 → 双引号 (谨慎: 只替换最外层的) ---
        # 如果上面的都失败, 尝试将单引号替换为双引号
        # 注意: 字符串中的引号嵌套可能造成问题, 但作为最后手段值得尝试
        fixed3 = fixed2.replace("'", '"')
        try:
            return json.loads(fixed3)
        except json.JSONDecodeError:
            pass
        
        return None
    
    @staticmethod
    def _validate_hypotheses_structure(parsed: Dict) -> Optional[Dict]:
        """
        验证和补全假设结构
        
        检查每个 hypothesis 的必要字段, 为缺失的字段设置默认值
        """
        if not isinstance(parsed, dict):
            return None
        
        hypotheses = parsed.get("hypotheses", parsed.get("hypothesis"))
        if hypotheses is None:
            logger.warning("No 'hypotheses' key in parsed response")
            return None
        
        if not isinstance(hypotheses, list):
            logger.warning("'hypotheses' is not a list")
            return None
        
        if len(hypotheses) == 0:
            logger.warning("Hypotheses list is empty")
            return None
        
        # 必填字段
        required_fields = ["id", "claim"]
        optional_fields = {
            "verification_thought": "",
            "verification_method": "custom",
            "data_needed": [],
            "expected_if_true": "",
            "expected_if_false": "",
            "confidence_in_llm": "medium",
            "priority": 3,
            "source_field": "unknown",
        }
        
        validated_hypotheses = []
        for h in hypotheses:
            if not isinstance(h, dict):
                continue
            
            # 检查必填字段
            missing = [f for f in required_fields if f not in h or not h[f]]
            if missing:
                logger.warning(f"Hypothesis missing required fields: {missing}")
                continue
            
            # 补全可选字段
            for key, default in optional_fields.items():
                if key not in h:
                    h[key] = default
            
            # 标准化 data_needed 为 list
            if isinstance(h.get("data_needed"), str):
                h["data_needed"] = [h["data_needed"]]
            elif not isinstance(h.get("data_needed"), list):
                h["data_needed"] = []
            
            validated_hypotheses.append(h)
        
        if not validated_hypotheses:
            logger.warning("No valid hypotheses after structure validation")
            return None
        
        result = {
            "hypotheses": validated_hypotheses,
            "summary": parsed.get("summary", ""),
        }
        
        if len(validated_hypotheses) < len(hypotheses):
            logger.warning(
                f"Filtered {len(hypotheses) - len(validated_hypotheses)} "
                f"invalid hypotheses out of {len(hypotheses)}"
            )
        
        logger.info(
            f"Structure validation passed: {len(validated_hypotheses)} hypotheses"
        )
        return result
    
    def _parse_json_from_response(self, response: str) -> Optional[Dict]:
        """从 LLM 回复中解析 JSON (增强版 — 多策略健壮解析)"""
        import re
        
        # 提取 JSON block
        json_str = self._extract_json_block(response)
        if json_str is None:
            logger.warning("Cannot extract JSON from response")
            return None
        
        # 使用多策略健壮解析
        return self._robust_json_parse(json_str)
    
    def _call_and_parse_json_with_retry(self, prompt: str, system_content: str,
                                          temperature: float = 0.3,
                                          max_tokens: int = 2048,
                                          max_retries: int = 2,
                                          additional_instructions: str = "",
                                          validate_func=None) -> Optional[Dict]:
        """
        通用 JSON 调用 + 解析 + 自动重试工具
        
        模式: LLM 调用 → 健壮 JSON 解析 → 结构验证
              ↓ 解析/验证失败
              错误诊断 + 重试 (带 JSON_FIX_PROMPT_TEMPLATE)
              ↓ 重试耗尽
              None
        
        Args:
            prompt: 调用的 prompt
            system_content: system message 内容
            temperature: LLM temperature
            max_tokens: 最大 token 数
            max_retries: 最大重试次数
            additional_instructions: JSON_FIX_PROMPT 额外说明
            validate_func: 可选的验证函数, 接收 parsed dict, 返回 bool
            
        Returns:
            Parsed JSON dict, or None
        """
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        if response is None:
            logger.error("LLM call returned None")
            return None
        
        for attempt in range(max_retries + 1):  # 首次 + retries
            parsed = self._parse_json_from_response(response)
            
            if parsed is not None:
                # 验证 (如果提供了验证函数)
                if validate_func is None or validate_func(parsed):
                    if attempt > 0:
                        logger.info(f"JSON parsed successfully on retry #{attempt}")
                    return parsed
                else:
                    logger.warning("JSON parsed but validation failed")
                    # 即使验证失败, 也尝试用错误信息重试
                    if attempt >= max_retries:
                        return None
            
            if attempt >= max_retries:
                logger.warning(
                    f"All {max_retries + 1} attempts exhausted. "
                    f"Response preview: {(response[:300] if response else 'None')}..."
                )
                return None
            
            # 准备重试
            raw_truncated = (response[:3000] + "..."
                             if response and len(response) > 3000
                             else (response or "None"))
            parse_error = self._diagnose_json_error(response)
            
            retry_prompt = JSON_FIX_PROMPT_TEMPLATE.format(
                raw_response_truncated=raw_truncated,
                parse_error=parse_error,
                additional_instructions=additional_instructions,
            )
            
            logger.info(
                f"JSON retry #{attempt + 1}/{max_retries} "
                f"(parse error: {parse_error[:60]}...)"
            )
            
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": (
                        "你是一位严谨的数据专家。你之前输出的 JSON 格式有误，"
                        "请重新输出，确保是严格合法的 JSON 格式且内容正确。"
                    )},
                    {"role": "user", "content": retry_prompt},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
            )
            
            if response is None:
                logger.error("No response from LLM during JSON retry")
                return None
        
        return None
    
    def _clean_code_response(self, response: str) -> str:
        """
        清理 LLM 生成的代码回复
        
        去掉:
        - markdown code block 标记 (```python ... ```)
        - 开头/结尾的解释文字
        - 行号前缀
        """
        import re
        
        # 如果有 markdown code block, 提取其中的代码
        code_match = re.search(r'```(?:python)?\s*\n(.*?)\n?```', response, re.DOTALL)
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        # 去掉行号前缀 (如 "  123→" 或 "123|")
        lines = code.split("\n")
        cleaned_lines = []
        for line in lines:
            # 去掉 "数字→" 或 "数字|" 前缀
            line = re.sub(r'^\s*\d+[→|]\s*', '', line)
            cleaned_lines.append(line)
        
        code = "\n".join(cleaned_lines)
        
        # 去掉开头的解释文字 (非代码行)
        # 找到第一个看起来像代码的行
        code_lines = []
        found_code_start = False
        for line in code.split("\n"):
            stripped = line.strip()
            if not found_code_start:
                # 代码行通常以 import, def, class, #, 变量赋值, 空行开头
                if stripped.startswith("import ") or \
                   stripped.startswith("from ") or \
                   stripped.startswith("def ") or \
                   stripped.startswith("class ") or \
                   stripped.startswith("#") or \
                   stripped == "" or \
                   "=" in stripped or \
                   stripped.startswith("if ") or \
                   stripped.startswith("for ") or \
                   stripped.startswith("while ") or \
                   stripped.startswith("try:") or \
                   stripped.startswith("with ") or \
                   stripped.startswith("@"):
                    found_code_start = True
                    code_lines.append(line)
                # 跳过解释文字
            else:
                code_lines.append(line)
        
        return "\n".join(code_lines) if code_lines else code
    
    @staticmethod
    def _clean_code_response_static(response: str) -> str:
        """
        清理 LLM 生成的代码回复 (静态版本, 供 DataComputationEngine 使用)
        
        与 _clean_code_response 逻辑一致, 但不需要 self
        """
        import re
        
        # 提取 markdown code block (支持 python, python3, 或没有语言标记)
        code_match = re.search(
            r'```(?:python[3]?)?\s*\n(.*?)\n?```', response, re.DOTALL
        )
        if code_match:
            code = code_match.group(1)
        else:
            code = response
        
        # 去掉行号前缀
        lines = code.split("\n")
        cleaned = []
        for line in lines:
            line = re.sub(r'^\s*\d+[→|]\s*', '', line)
            cleaned.append(line)
        code = "\n".join(cleaned)
        
        # 去掉开头解释文字
        code_lines = []
        found_start = False
        for line in code.split("\n"):
            stripped = line.strip()
            if not found_start:
                if stripped.startswith("import ") or \
                   stripped.startswith("from ") or \
                   stripped.startswith("def ") or \
                   stripped.startswith("class ") or \
                   stripped.startswith("#") or \
                   stripped == "" or \
                   "=" in stripped or \
                   stripped.startswith("if ") or \
                   stripped.startswith("for ") or \
                   stripped.startswith("while ") or \
                   stripped.startswith("try:") or \
                   stripped.startswith("with ") or \
                   stripped.startswith("@"):
                    found_start = True
                    code_lines.append(line)
            else:
                code_lines.append(line)
        
        return "\n".join(code_lines) if code_lines else code