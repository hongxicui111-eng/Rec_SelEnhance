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


class DataInventory:
    """
    数据盘点器 — 发现项目中可用的数据资源
    
    探索:
    - 训练/测试数据文件
    - 元数据文件 (id_meta_data.json 等)
    - 已计算好的统计量 (item_popularity 等)
    - 训练日志中的指标
    - 模型 checkpoint 信息
    """
    
    def __init__(self, project_root: str, data_dir: str = None, log_dir: str = None):
        self.project_root = project_root
        self.data_dir = data_dir or os.path.join(project_root, "Recmodel", "data")
        self.log_dir = log_dir or os.path.join(project_root, "logs")
        self._inventory = None
    
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


class HypothesisVerificationAgent:
    """
    假设验证 Agent — 自主式假设验证框架
    
    与旧版 HypothesisVerifier 的关键区别:
    1. 不使用固定验证方法 — LLM 为每个假设自由设计验证方案
    2. 不使用硬编码阈值 — LLM 解读统计结果来判断假设成立与否
    3. Agent 自主写代码执行验证 (而非预定义 Python 函数)
    4. 代码执行失败时自动修正 (self-correction loop)
    
    工作流程:
    1. extract_hypotheses() — 从 LLM 分析提取假设
    2. verify_hypotheses() — 对每个假设执行完整的 Agent 验证流程:
       a. generate_verification_plan() — LLM 设计验证方案
       b. discover_and_load_data() — 发现并加载所需数据
       c. generate_verification_code() — LLM 写验证脚本
       d. execute_verification_code() — 执行脚本 (失败则修正)
       e. analyze_results() — LLM 解读结果, 判断假设
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
        从 LLM 分析结果中提取可验证的假设
        
        与旧版区别: 不限制为固定 6 种验证方法, LLM 自由描述验证思路
        
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
        
        response = self.llm.chat(
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
        
        if response is None:
            logger.error("Hypothesis extraction failed - no LLM response")
            return None
        
        parsed = self._parse_hypothesis_response(response)
        if parsed and parsed.get("hypotheses"):
            logger.info(f"Extracted {len(parsed['hypotheses'])} verifiable hypotheses")
            return parsed["hypotheses"]
        
        return None
    
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
        LLM 为假设生成验证方案
        
        Returns:
            verification plan dict, or None
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        thought = hypothesis.get("verification_thought", "")
        data_needed = hypothesis.get("data_needed", [])
        expected_true = hypothesis.get("expected_if_true", "")
        expected_false = hypothesis.get("expected_if_false", "")
        
        data_inventory_text = self.data_inventory.format_inventory_for_prompt()
        loaded_data_summary = self.data_inventory.format_loaded_data_summary(preloaded_data)
        
        prompt = VERIFICATION_PLAN_PROMPT.format(
            hypothesis_id=hyp_id,
            hypothesis_claim=claim,
            verification_thought=thought,
            data_needed=json.dumps(data_needed, ensure_ascii=False),
            expected_if_true=expected_true,
            expected_if_false=expected_false,
            data_inventory=data_inventory_text,
            loaded_data_summary=loaded_data_summary,
        )
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位数据科学家，擅长设计严谨的统计验证方案。"
                    "方案必须具体、可执行，使用项目中可用的数据。"
                    "如果某些数据不可用，方案应该说明替代方法。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
        )
        
        if response is None:
            logger.error("Verification plan generation failed - no LLM response")
            return None
        
        return self._parse_json_from_response(response)
    
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
        
        合并为一个统一的数据描述, 供代码生成使用
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
        LLM 根据验证方案和可用数据生成验证脚本
        
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
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位 Python 数据科学家，擅长编写独立的数据验证脚本。"
                    "代码必须稳健、能处理异常、只依赖标准库和 numpy/scipy。"
                    "结果必须以 JSON 格式输出到指定文件。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度 → 更精确的代码
            max_tokens=4096,
        )
        
        if response is None:
            logger.error("Verification code generation failed - no LLM response")
            return None
        
        # 清理代码 (去掉可能的 markdown 标记)
        code = self._clean_code_response(response)
        
        # 在代码开头注入数据加载逻辑
        code = self._inject_data_loading(code, verification_data, output_file)
        
        return code
    
    def _format_available_data_for_code(self, verification_data: Dict) -> str:
        """
        格式化可用数据描述, 让 LLM 知道代码中可以使用哪些数据
        
        关键设计: 数据将通过变量注入到代码中, LLM 只需使用变量名
        """
        lines = []
        lines.append("以下数据已经作为 Python 变量注入到脚本中, 直接使用即可:")
        lines.append("")
        
        for key, value in verification_data.items():
            if isinstance(value, list):
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
        
        # 序列化数据 (大列表只保存统计摘要)
        serializable = {}
        for key, value in verification_data.items():
            try:
                if isinstance(value, list) and len(value) > 100:
                    # 大列表只保存前 50 条作为样本
                    serializable[key + "_sample"] = value[:50]
                    serializable[key + "_count"] = len(value)
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
        for key, value in verification_data.items():
            if key in serializable:
                injection_lines.append(f'{key} = _preloaded.get("{key}", None)')
            elif key + "_sample" in serializable:
                injection_lines.append(f'{key}_sample = _preloaded.get("{key}_sample", [])')
                injection_lines.append(f'{key}_count = _preloaded.get("{key}_count", 0)')
        
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
        执行验证脚本
        
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
            
            # 查找输出文件
            # 从脚本中找到 OUTPUT_FILE 的路径
            output_dir = os.path.join(self.log_dir, "verification_scripts")
            
            # 尝试从代码中提取输出路径
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # 搜索 OUTPUT_FILE 赋值行
            import re
            output_match = re.search(r'OUTPUT_FILE\s*=\s*"([^"]+)"', script_content)
            if output_match:
                output_file_path = output_match.group(1)
            else:
                # 回退: 在 verification_scripts 目录下搜索 result_*.json
                result_files = [f for f in os.listdir(output_dir) 
                                if f.startswith("result_") and f.endswith(".json")]
                if result_files:
                    output_file_path = os.path.join(output_dir, result_files[0])
                else:
                    return False, None, "No output file found after execution"
            
            # 加载结果
            if os.path.exists(output_file_path):
                with open(output_file_path, 'r', encoding='utf-8') as f:
                    result_data = json.load(f)
                return True, result_data, None
            else:
                # 尝试从 stdout 中提取 JSON
                stdout = result.stdout
                if stdout.strip().startswith("{"):
                    try:
                        result_data = json.loads(stdout.strip())
                        return True, result_data, None
                    except json.JSONDecodeError:
                        pass
                
                return False, None, f"Output file not found: {output_file_path}"
        
        except subprocess.TimeoutExpired:
            return False, None, f"Script execution timed out ({self.MAX_EXECUTION_TIMEOUT}s)"
        except Exception as e:
            return False, None, f"Execution error: {str(e)}"
    
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
        LLM 解读验证代码的执行结果, 判断假设是否成立
        
        与旧版区别: 不使用硬编码阈值, LLM 自主判断
        
        Returns:
            verification result dict with status, brief, evidence, etc.
        """
        hyp_id = hypothesis.get("id", "H?")
        claim = hypothesis.get("claim", "")
        expected_true = hypothesis.get("expected_if_true", "")
        expected_false = hypothesis.get("expected_if_false", "")
        
        plan_json = json.dumps(verification_plan, indent=2, ensure_ascii=False)
        result_json = json.dumps(execution_result, indent=2, ensure_ascii=False)
        
        # 截断过长 JSON
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
        
        response = self.llm.chat(
            messages=[
                {"role": "system", "content": (
                    "你是一位严谨的数据科学家，擅长根据统计结果判断假设是否成立。"
                    "你必须给出明确判断 (不能模棱两可)。"
                    "判断基于数据，而非直觉。如果数据不支持假设, 就判为 REFUTED。"
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,  # 低温度 → 更精确的判断
            max_tokens=1024,
        )
        
        if response is None:
            logger.error("Result analysis failed - no LLM response")
            return {
                "status": self.UNVERIFIABLE,
                "reason": "LLM 分析失败",
                "brief": "结果分析失败",
                "evidence": execution_result,
            }
        
        parsed = self._parse_json_from_response(response)
        if parsed and parsed.get("status"):
            # 补充 evidence 字段 (如果 LMM 输出中没有)
            if "evidence" not in parsed:
                parsed["evidence"] = execution_result.get("statistics", execution_result)
            parsed["method"] = "agent_autonomous"
            return parsed
        
        # 解析失败 → 手动构建结果
        return {
            "status": self.UNVERIFIABLE,
            "reason": "LLM 分析结果解析失败",
            "brief": "分析结果解析失败",
            "evidence": execution_result,
            "method": "agent_autonomous",
        }
    
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
        """解析 LLM 假设提取回复"""
        import re
        
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end + 1]
            else:
                logger.warning("Cannot extract JSON from hypothesis extraction response")
                return None
        
        try:
            parsed = json.loads(json_str)
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error in hypothesis extraction: {e}")
            try:
                import ast
                parsed = ast.literal_eval(json_str)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                pass
            return None
    
    def _parse_json_from_response(self, response: str) -> Optional[Dict]:
        """从 LLM 回复中解析 JSON"""
        import re
        
        # 尝试提取 JSON code block
        json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 尝试直接找 JSON
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end + 1]
            else:
                logger.warning("Cannot extract JSON from response")
                return None
        
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}")
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