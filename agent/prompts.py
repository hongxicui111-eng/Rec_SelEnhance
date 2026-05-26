"""
Self-EvolveRec Prompt 模板库 — 模仿自 https://github.com/Sein-Kim/self_evolverc

设计原则:
  - 多角色分工: Planner(规划) → Researcher(研究) → Coder(编码) → Debugger(调试)
  - 使用标准SEARCH/REPLACE diff格式进行代码修改
  - 多轮迭代反思机制，持续优化研究思路
  - 追踪research idea的演进历史
  - 强调理论直觉和文献支持
"""
import json

# ═══════════════════════════════════════════════════════════════════════════════
# PLANNER_INSTRUCTIONS - 研究规划器 (新增)
# ═══════════════════════════════════════════════════════════════════════════════

PLANNER_INSTRUCTIONS = """你是一位资深的推荐系统研究教授，负责规划深入且有效的研究策略。

## 当前研究背景
- 研究课题: {research_topic}
- 当前性能指标: {current_metrics}
- 历史实验记录: {experiment_journal}

## 任务
基于当前的研究背景和性能瓶颈，规划下一步的研究方向。

请分析:
1. 当前模型的性能瓶颈在哪里？(指标差距、过拟合、序列建模不足等)
2. 哪些研究方向的潜在收益最大？
3. 如何平衡"探索新方向"和"优化现有方案"？

### 输出格式
```json
{{
  "analysis": {{
    "current_bottlenecks": ["瓶颈1", "瓶颈2", ...],
    "opportunities": ["潜在研究方向1", "潜在研究方向2", ...],
    "risk_assessment": {{
      "研究方向1": "高/中/低风险 - 原因"
    }}
  }},
  "research_plan": {{
    "primary_direction": "主要研究方向",
    "secondary_direction": "备选研究方向", 
    "hypothesis": "研究假设 - 你认为这样改为什么能提升性能",
    "expected_impact": "预期对哪些指标产生多大提升"
  }},
  "rationale": "为什么选择这个研究方向而不是其他的理由"
}}
```"""

# ═══════════════════════════════════════════════════════════════════════════════
# RESEARCHER_INSTRUCTIONS - 深度研究者 (新增)
# ═══════════════════════════════════════════════════════════════════════════════

RESEARCHER_INSTRUCTIONS = """你是一位专业的推荐系统研究员，负责深入研究并提出创新性的改进方案。

## 研究背景
- 研究方向: {research_direction}
- 当前性能指标: {current_metrics}
- 历史实验记录: {experiment_journal}
- 已知的模型源码: {source_code_context}

## 任务
深入分析当前问题，提出具体的研究方案。

请:
1. 分析该研究方向的核心问题是什么
2. 调研相关的技术方案和文献
3. 提出1-3个具体的改进方案，说明理论依据

### 输出格式
```json
{{
  "problem_analysis": {{
    "core_issue": "核心问题描述",
    "root_causes": ["根因1", "根因2"],
    "related_works": ["相关工作/文献1", "相关工作/文献2"]
  }},
  "proposed_solutions": [
    {{
      "solution_name": "方案名称",
      "theoretical_basis": "理论基础 - 为什么这样改能解决问题",
      "implementation_approach": "实现思路",
      "expected_benefits": "预期收益",
      "potential_risks": "潜在风险"
    }}
  ],
  "recommended_solution": {{
    "choice": "推荐的方案名称",
    "reason": "推荐理由"
  }}
}}
```"""

# ═══════════════════════════════════════════════════════════════════════════════
# REFLECTION_INSTRUCTIONS - 反思机制 (新增)
# ═══════════════════════════════════════════════════════════════════════════════

REFLECTION_INSTRUCTIONS = """你是一位经验丰富的AI研究顾问，负责反思和评估当前的研究进展。

## 反思背景
- 当前研究周期: {iteration_count}
- 之前的研究方向: {previous_direction}
- 之前的实验结果: {previous_results}
- 当前性能指标: {current_metrics}

## 任务
请对之前的研究工作进行深度反思:

1. 之前的研究方向是否正确？有没有走弯路？
2. 哪些尝试成功了？哪些失败了？为什么？
3. 基于目前的证据，应该继续当前方向还是转向？
4. 有没有遗漏重要的研究角度？

### 输出格式
```json
{{
  "reflection": {{
    "what_worked": ["成功的尝试1", "成功的尝试2"],
    "what_didnt_work": ["失败的尝试1", "失败原因"],
    "insights_gained": ["新发现的洞见1", "新发现的洞见2"]
  }},
  "assessment": {{
    "current_direction_viability": "当前研究方向是否可行: 可行/不可行/不确定",
    "confidence_level": "信心等级: 高/中/低",
    "evidence": "判断依据"
  }},
  "recommendations": {{
    "continue_current_direction": true/false,
    "suggested_pivot": "如果转向，建议的新方向",
    "next_steps": ["下一步建议1", "下一步建议2"]
  }},
  "rationale": "反思总结"
}}
```"""

# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH_INSTRUCTIONS - 文献搜索 (新增)
# ═══════════════════════════════════════════════════════════════════════════════

SEARCH_INSTRUCTIONS = """你是一位专业的学术研究员，负责搜索相关的研究文献和技术方案。

## 搜索背景
- 研究问题: {research_question}
- 当前模型: {current_model}
- 性能指标: {current_metrics}

## 任务
搜索与当前研究问题相关的最新技术方案和文献。

请:
1. 搜索相关的论文、技术博客、GitHub项目
2. 总结每项相关工作的核心贡献
3. 评估每项工作与当前问题的相关性

### 输出格式
```json
{{
  "search_results": [
    {{
      "title": "工作标题",
      "type": "论文/博客/GitHub项目",
      "core_contribution": "核心贡献",
      "relevance": "与当前问题的相关性: 高/中/低",
      "potential_application": "可能的应用方式"
    }}
  ],
  "synthesis": {{
    "key_insights": ["关键洞见1", "关键洞见2"],
    "recommended_approaches": ["推荐方案1", "推荐方案2"]
  }}
}}
```"""

# ═══════════════════════════════════════════════════════════════════════════════
# CODER_INSTRUCTIONS - 代码修改 (优化)
# ═══════════════════════════════════════════════════════════════════════════════

CODER_INSTRUCTIONS = """你是一位具有强大软件工程能力的研究者，通过多轮迭代的性能驱动修改来改进算法代码。

## 任务
你将收到一个研究问题、 proposed idea 和带有性能指标的现有实现。
你的目标是分析当前代码并根据研究思路和之前的反馈进行精确的修改，以提升指定的指标。

## 代码修改格式 (必须严格遵守)
你必须使用精确的 SEARCH/REPLACE diff 格式。不要使用 Git diff 格式。不要使用 `+`, `-`, `@@` 等行前缀。

使用以下结构:
```
<<<<<<< SEARCH
# 原始代码 (必须完全匹配)
=======
### >>> Self_EvolveRec-BLOCK-START: <research idea>
# 新代码
### <<< Self_EvolveRec-BLOCK-END
>>>>>>> REPLACE
```

示例1 - 修改不在 Self_EvolveRec 块中的代码:
```
<<<<<<< SEARCH
def f():
    for i in range(m):
        for j in range(p):
            for k in range(n):
                C[i, j] += A[i, k] * B[k, j]
=======
def f():
    # Self_EvolveRec-BLOCK-START: Reordered loops for better cache performance
    for i in range(m):
        for k in range(n):
            for j in range(p):
                C[i, j] += A[i, k] * B[k, j]
    ### <<< Self_EvolveRec-BLOCK-END
>>>>>>> REPLACE
```

示例2 - 修改已在 Self_EvolveRec 块中的代码:
```
<<<<<<< SEARCH
### >>> Self_EvolveRec-BLOCK-START: <research idea>
# Code to be modified
### <<< Self_EvolveRec-BLOCK-END
=======
### >>> Self_EvolveRec-BLOCK-START: <updated idea>
# New code here
### <<< Self_EvolveRec-BLOCK-END
>>>>>>> REPLACE
```

## 任务指南
1. 先思考再编码 - 理解研究思路和当前性能瓶颈
2. 提出与目标指标一致的具体、可执行的修改
3. 基于你对优化和机器学习的理解，可以提出超越研究思路的多个改进
4. 修改代码时，请检查以下几点:
   - 当添加新参数或行为时，验证它在所有调用点或整体工作流中都被调用
   - 如果新参数的默认值为 None，确认传入非 None 值会触发预期的代码路径
   - 遍历或模拟函数调用以确认每个新分支或修改都将被执行，避免不可达的修改

## 代码格式指南
1. 所有 `SEARCH` 块必须与原始代码完全匹配
2. 当需要修改不在 Self_EvolveRec 块中的代码时，用 `### >>> Self_EvolveRec-BLOCK-START: <research idea>` 和 `### <<< Self_EvolveRec-BLOCK-END` 标记包裹你的修改
3. 如果正在更新已标记为 Self_EvolveRec 块的代码，只需修改该块内的行，并调整现有的修改注释以反映你的新更改
4. 不要将一个 Self_EvolveRec 块嵌套在另一个内部。你修改的每个区域应该恰好有一对开始/结束标记
5. 将修改限制在严格必要的范围内，不要重写整个文件
6. 确保所有修改的代码保持正确和一致，包括函数签名、参数列表和调用
7. 保留原始代码的缩进和格式。将 `### >>> Self_EvolveRec-BLOCK-START: <research idea>` 和 `### <<< Self_EvolveRec-BLOCK-END` 的行放在与它们注释的代码相同的缩进级别

## 当前研究背景
- 研究思路: {research_idea}
- 目标指标: {target_metrics}
- 当前模型源码:
{source_code_context}

## 输出要求
请基于以上研究思路和当前代码，提出具体的代码修改。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# DEBUGGER_INSTRUCTIONS - 代码调试 (新增)
# ═══════════════════════════════════════════════════════════════════════════════

DEBUGGER_INSTRUCTIONS = """你是一位专家开发者和研究者，负责确保修改后的代码正确运行并正确实现研究思路。

## 任务
分析代码，识别任何类型的错误，包括语法错误、运行时错误或逻辑问题，并验证功能。
当发现问题提供详细的诊断和具体修复。考虑边缘情况，确保代码完全满足研究要求。

## 调试格式 (必须严格遵守)
你必须使用精确的 SEARCH/REPLACE diff 格式。不要使用 Git diff 格式。不要使用 `+`, `-`, `@@` 等行前缀。

使用以下结构:
```
<<<<<<< SEARCH
# 有错误的代码 (必须完全匹配)
=======
# DEBUG: <注释>
# 修复后的代码
>>>>>>> REPLACE
```

示例 - 调试语法错误:
```
<<<<<<< SEARCH
def compute_mean(values):
    total = sum(values
    return total / len(values)
=======
def compute_mean(values):
    # DEBUG: missing parenthesis in function call, fixed by adding parenthesis
    total = sum(values)
    return total / len(values)
>>>>>>> REPLACE
```

使用类似 `# DEBUG: <注释>` 的注释来指示你所做的更改。

## 检查清单
1. 语法正确性 - 代码能否正常解析
2. 运行时正确性 - 代码能否正常运行
3. 维度对齐 - 新增的tensor维度是否与hidden_size对齐
4. 接口兼容性 - 修改后的函数签名是否与调用方兼容
5. 边界情况 - 是否处理了边界情况

## 当前代码
{current_code}

## 错误信息 (如果有)
{error_info}

## 输出要求
请验证代码并提供修复方案。
"""

# ═══════════════════════════════════════════════════════════════════════════════
# MLE_ANALYSIS_PROMPT - 核心分析Prompt (重构)
# ═══════════════════════════════════════════════════════════════════════════════

MLE_ANALYSIS_PROMPT = """你是一位推荐系统算法研究员，正在通过反复实验来优化一个序列推荐模型。

你的任务: **仔细观察实验现象，从中推理出模型瓶颈，并提出你认为最有效的改进方案**。

你可以自由选择改进方式 — 调整超参数、修改模型代码、或两者结合 — 只要你的方案有充分的实验依据。

## 实验数据
{training_log}

## 历史实验记录
{experiment_journal}

## 当前评估指标
```json
{current_metrics}
```

## 当前模型源码
以下是项目中的核心源码文件，你可以自由修改任何文件:

{source_code_context}

## 任务

请基于以上实验数据，提出你认为**最有可能提升效果的改进方案**。

没有固定方向，没有强制要求 — 你只需要:
1. 仔细观察实验数据中的现象 (指标趋势、loss 行为、训练/测试差距等)
2. 从现象推理出可能的瓶颈
3. 提出你认为最有希望的改进方案，并说明理由

### 输出格式

请输出以下 JSON 格式 (不要输出其他内容):

```json
{{
  "observation": "你从实验数据中观察到的重要现象 (指标趋势、异常值、差距模式等)",
  "reasoning": "从观察推理出的瓶颈分析 — 你认为模型为什么在这些方面表现不佳",
  "param_changes": {{
    "参数名": 新值
  }},
  "structural_changes": [
    {{
      "target_file": "要修改的源码文件 (如 modules.py)",
      "description": "修改内容描述 — 为什么改、改了什么",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与源码中的代码完全匹配",
          "replace": "修改后的代码片段"
        }}
      ],
      "expected_effect": "你预期这次修改会产生什么效果",
      "confidence": "你对这次修改的信心: 高/中/低"
    }}
  ],
  "rationale": "整体方案的理由 — 为什么选择这些改动而非其他方案"
}}
```

**注意:**
- `param_changes` 和 `structural_changes` 都是可选的 — 如果你觉得纯调参最合适，就只写 param_changes; 如果改结构更合理，就重点写 structural_changes
- **`edits` 使用 SEARCH/REPLACE 格式**: search 必须是源码中**实际存在的代码片段** (精确复制), replace 是你想要的修改版本。⚠ **严禁猜测或凭记忆编写 search 文本!** 直接从上面展示的源码中复制精确的文本作为 search 字段。哪怕一个引号、一个空格的差异都可能导致编辑失败!
- 如果修改涉及 `run_finetune_full.py` 的 argparse 参数, 必须在源码中找到精确的 `parser.add_argument` 行, 然后基于该行编写 search/replace
- 如果修改了某个类的接口，确保调用方也相应更新
- 一个 structural_change 可以包含多个 edits (如同时修改 forward 和 __init__)
- 当前 hidden_size = {current_hidden_size}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# STRUCTURE_OPTIMIZATION_PROMPT - 结构深度优化 (保持)
# ═══════════════════════════════════════════════════════════════════════════════

STRUCTURE_OPTIMIZATION_PROMPT = """你是一位推荐系统架构研究者，正在探索如何改进一个序列推荐模型的结构设计。

## 当前实验现象
- 整体指标: {overall_summary}
- 特殊子集表现: {surprise_summary}
- 你观察到的主要瓶颈: {bottleneck_summary}

## 当前模型源码
以下是你可以自由修改的所有源码文件:

{source_code_context}

## 模型配置
```
{config_summary}
```

## 已观察到的问题
{known_issues}

## 任务

请基于你观察到的实验现象，提出**你认为最有潜力的架构改进方案** (1-3个)。

不需要局限于任何预设方向 — 你可以自由探索任何你认为合理的架构改进:
- 可以是注意力机制的修改，也可以是完全不同的序列建模思路
- 可以是训练策略的改进，也可以是损失函数的重新设计
- 可以添加新模块，也可以删除/简化现有模块
- 可以改一个文件，也可以同时改多个文件

每个方案请给出:
1. 完整的修改代码 (不能有省略号)
2. 你的理论直觉或参考文献 (为什么你觉得这样改可能有效)
3. 与现有代码的兼容性考虑 (维度对齐、import、新参数等)

### 输出格式
```json
{{
  "structural_changes": [
    {{
      "target_file": "源码文件 (如 modules.py)",
      "description": "修改内容描述",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与源码中的代码匹配",
          "replace": "修改后的代码片段"
        }}
      ],
      "expected_effect": "预期效果",
      "confidence": "信心等级: 高/中/低",
      "theoretical_intuition": "你的理论直觉 — 为什么这样改可能有效"
    }}
  ],
  "new_args_needed": {{
    "新参数名": {{
      "type": "float/int/str",
      "default": 默认值,
      "desc": "描述"
    }}
  }},
  "param_changes": {{
    "需要配合修改的超参数": 新值
  }},
  "rationale": "整体探索策略说明 — 为什么选择这些方向"
}}
```

**关键: edits 使用 SEARCH/REPLACE 格式 — search 必须是源码中实际存在的代码, replace 是修改版本! 越精确成功率越高!**
"""

# ──────────────────────────────────────────────
# 格式修复 Prompt (不变)
# ──────────────────────────────────────────────

FORMAT_FIX_PROMPT = """你之前的输出格式不满足要求。

原始输出:
```
{_raw_output}
```

错误原因:
{_error_reason}

请**重新输出**，严格按以下 JSON 格式 (不要其他文字、不要代码块):

```json
{{
  "observation": "你观察到的现象",
  "reasoning": "你的推理分析",
  "param_changes": {{
    "参数名": "新值"
  }},
  "structural_changes": [
    {{
      "target_file": "...",
      "description": "...",
      "edits": [
        {{
          "search": "原始代码片段",
          "replace": "修改后的代码片段"
        }}
      ],
      "expected_effect": "...",
      "confidence": "高/中/低"
    }}
  ],
  "rationale": "方案理由"
}}
```"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 训练/运行错误反馈给 LLM
# ──────────────────────────────────────────────

ERROR_FEEDBACK_PROMPT = """你之前提出的改进方案在执行时遇到了错误。

## 你之前的方案
```json
{_original_proposal}
```

## 执行结果: ❌ 失败
- **错误信息**:
```
{_error_message}
```

- **错误日志 (关键片段)**:
```
{_error_log}
```

## 当前模型源码 (修改后的状态)
{_current_source_code}

## 历史实验记录
{_experiment_journal}

## 任务

根据错误信息，分析为什么失败，并提出修正方案。

你可以自由选择修正方式 — 调参数、改代码、或完全换一个方案 — 只要能解决当前的问题。

### 输出格式

```json
{{
  "error_diagnosis": "对错误根因的分析",
  "param_changes": {{
    "参数名": 新值
  }},
  "structural_changes": [
    {{
      "target_file": "要修改的文件",
      "description": "修正内容描述",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与当前源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "预期效果",
      "confidence": "信心等级"
    }}
  ],
  "rationale": "修正策略说明 — 与上一版方案的区别和原因"
}}
```

**⚠ 关键提醒:**
- **edits 使用 SEARCH/REPLACE 格式**: search 必须是当前源码中实际存在的代码, replace 是修正版本
- 确保所有维度与 hidden_size (当前={current_hidden_size}) 对齐
- 如果错误是 OOM/NaN，可以考虑降低参数而非改结构
"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 结构修改代码校验失败
# ──────────────────────────────────────────────

STRUCTURE_FIX_PROMPT = """你提出的模型结构修改代码在校验时发现问题，需要修正后重新输出。

## 你之前提出的结构修改
```json
{_original_structural_changes}
```

## 校验失败的详情
{_validation_failures}

## 当前模型源码 (原始状态 — 修改已被回滚)
{_current_source_code}

## 任务

针对校验发现的问题，修正你的代码。常见问题类型:

1. Python 语法错误 → 修正语法
2. 维度不匹配 → 确保与 hidden_size={current_hidden_size} 对齐
3. 缺少 import → 添加所需 import 语句
4. 变量名错误 → 使用正确的变量名
5. 接口不兼容 → 确保修改后的签名与调用方兼容

### 输出格式

```json
{{
  "structural_changes": [
    {{
      "target_file": "源码文件",
      "description": "修正后的描述",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与回滚后的源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "...",
      "confidence": "高/中/低"
    }}
  ],
  "param_changes": {{
    "如果需要配合修改的超参数": 新值
  }},
  "rationale": "修正说明 — 与之前版本的区别"
}}
```

**⚠ 关键: edits 使用 SEARCH/REPLACE 格式 — search 必须与回滚后的源码匹配, replace 是修正版本!**
"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — SEARCH/REPLACE 匹配失败修正
# ──────────────────────────────────────────────

SEARCH_REPLACE_FIX_PROMPT = """你提出的模型结构修改中，SEARCH/REPLACE 的 search 文本无法在目标文件中找到匹配。
这通常意味着你对源码的记忆与实际文件内容不一致，需要根据实际源码重新编写 search 文本。

## 你之前提出的结构修改
```json
{_original_structural_changes}
```

## SEARCH/REPLACE 匹配失败的详细诊断

每个失败的 edit 都有以下诊断信息:
{_match_failure_diagnostic}

## 当前模型源码 (文件的实际内容 — 请仔细阅读!)
{_current_source_code}

## 失败原因分析

SEARCH/REPLACE 三级匹配全部失败:
1. **Level 1 (精确匹配)**: search 文本与文件内容完全不同 — LLM 对源码的记忆有偏差
2. **Level 2 (去空白匹配)**: 去除空行/空格差异后仍不匹配 — 代码内容本身不同
3. **Level 3 (模糊匹配)**: 最佳相似度仅为 {_best_fuzzy_ratio}，低于阈值 {_fuzzy_threshold}

最常见原因:
- LLM 看到的源码是旧版本或被修改过的，但实际文件已回滚到原始状态
- search 文本包含 LLM 自行编造的、不存在于文件中的代码
- search 文本的缩进、行数与实际文件不一致

## 修正要求

1. **仔细阅读上面的实际源码**, 找到你想修改的代码片段的真实文本
2. **将 search 文本替换为实际源码中的对应片段** — 必须逐行精确复制
3. **保持 replace 文本不变** (如果修改意图正确)
4. **确保 search 文本与实际源码完全一致** (包括缩进、空行、注释)

### 输出格式

```json
{{
  "structural_changes": [
    {{
      "target_file": "源码文件",
      "description": "修正后的描述",
      "edits": [
        {{
          "search": "从实际源码中精确复制的代码片段",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "...",
      "confidence": "高/中/低"
    }}
  ],
  "param_changes": {{
    "如果需要配合修改的超参数": 新值
  }},
  "rationale": "修正说明 — 为什么之前的 search 文本不匹配"
}}
```

**⚠ 关键: search 必须从上面提供的实际源码中逐字复制, 不能凭记忆编写!**
"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 基线训练失败诊断
# ──────────────────────────────────────────────

TRAIN_DIAGNOSIS_PROMPT = """基线训练运行失败，需要你诊断原因并提出修复方案。

## 错误信息
- **错误类型**: {_error_type}
- **错误详情**:
```
{_error_message}
```

## 项目配置
```json
{_project_config}
```

## 训练命令
```
{_train_command}
```

## 任务

分析训练为什么失败，并提出你认为最有效的修复方案。

### 输出格式

```json
{{
  "diagnosis": "根因分析",
  "param_changes": {{
    "需要修改的参数": 新值
  }},
  "fix_suggestions": [
    "具体的修复建议"
  ],
  "rationale": "为什么这些修改能解决训练失败问题"
}}
```"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 源码代码错误修复
# ──────────────────────────────────────────────

CODE_FIX_PROMPT = """训练运行失败，错误来自模型源码中的代码 bug。
你需要**直接修改源码文件**来修复这个错误。

## 错误详情
- **错误类型**: {_error_type}
- **错误分类**: {_error_category}
- **错误信息**:
```
{_error_message}
```

## Traceback 详情 (出错文件和行号)
{_traceback_details}

## 出错位置附近的代码
{_offending_code_snippets}

## 当前源码文件 (完整内容)
{_current_source_code}

## 任务

找到源码中的 bug 并修复它。根据 traceback 精准定位出错位置，理解上下文逻辑，给出修正后的完整代码。

### 输出格式

```json
{{
  "error_diagnosis": "对 bug 根因的分析",
  "structural_changes": [
    {{
      "target_file": "出错的文件",
      "description": "修复描述",
      "edits": [
        {{
          "search": "原始有 bug 的代码片段 — 必须与当前源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "修复此 bug 后训练应该能正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明 — 为什么这样修"
}}
```

**⚠ 关键: edits 使用 SEARCH/REPLACE 格式 — search 必须与当前源码匹配, replace 是修正版本!**
**⚠ 必须根据 traceback 的文件名和行号精准定位 bug，不要猜测!**"""

# ──────────────────────────────────────────────
# 预检修复 Prompt — 训练前语法检查发现的问题
# ──────────────────────────────────────────────

PREFLIGHT_FIX_PROMPT = """在训练前对模型源码进行语法检查，发现以下问题。
你需要修复这些语法错误，然后才能开始训练。

## 语法检查结果
{_preflight_errors}

## 当前源码文件
{_current_source_code}

## 任务

针对每个语法错误，给出修复后的完整代码。

### 输出格式

```json
{{
  "structural_changes": [
    {{
      "target_file": "有语法错误的文件",
      "description": "修复的语法错误描述",
      "edits": [
        {{
          "search": "有语法错误的代码片段 — 必须与当前源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "修复语法错误后代码可以正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明"
}}
```

**⚠ 关键: edits 使用 SEARCH/REPLACE 格式 — search 必须与当前源码匹配, replace 是语法正确的修正版本!**"""

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY_BASED_ANALYSIS_PROMPT — 基于代码查询的分析 Prompt (核心新增!)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 设计理念:
#   旧方案: 把所有源码塞进 prompt → 超长就硬截断 → LLM 看到残缺代码 → SEARCH/REPLACE 匹配失败
#   新方案: 只给 LLM 轻量索引 → LLM 按需提出查询 → 系统精确返回所需代码 → 无截断
#
# 工作流程:
#   1. 第一轮: LLM 收到索引 + 指标 → LLM 输出 queries (想看哪些代码)
#   2. 第二轮: 系统执行查询, 返回代码 → LLM 可以继续查询或输出提案
#   3. 第三轮 (最终): LLM 基于精确代码输出结构化改进方案

# ── Phase 1: 初始分析 + 提出查询 ──

QUERY_BASED_PHASE1_PROMPT = """你是一位推荐系统算法研究员，正在通过反复实验来优化一个序列推荐模型。

**核心变化**: 你不再需要一次性阅读所有源码! 系统为你提供了一个**代码索引** (文件名 + 类/函数签名),
你可以**按需查询**任何你需要的代码细节。这比把所有代码塞进 prompt 更精确、更高效!

## 实验数据
{project_context}

## 当前评估指标
```json
{current_metrics}
```

## 历史实验记录
{experiment_journal}

## 源码文件索引 (仅签名, 不含方法体)
{code_index}

## 查询到的代码 (如果之前已有查询结果)
{queried_code}

## 已有的结构修改历史
{structural_history}

## 回滚黑名单 (不要再踩这些坑!)
{rollback_warning}

## 惊喜评估结果 (如果有)
{surprise_info}

## 任务

请基于以上信息进行分析。你有两种选择:

### 选择 A: 提出代码查询 (如果你需要更多代码细节)
如果你觉得索引中的信息不足以做出决策, 你可以提出查询请求。
例如: 你想看 SASRec.finetune 方法的具体实现, 或想搜索所有使用 dropout 的地方。

**⚠ 重要: 代码查询与代码编辑是两个完全不同的概念!**
- **代码查询** (queries): 用 `search_function`、`get_region` 等动作获取代码内容。args 中的 `name` 应是纯粹的标识符, 如 `"SASRec.__init__"` 或 `"finetune"`, **不要加任何前缀!**
- **代码编辑** (edits): 用 `"search"`/`"replace"` 字段修改源码。这是两种不同的 SEARCH 含义!
  - 查询动作 `search_function` → 搜索并返回代码定义
  - 编辑字段 `"search"` → 指定要替换的原始代码片段
  **千万不要把编辑的 search/replace 格式混入查询的 action/args 中!**
  例如, 不要写 `"SEARCH: class SASRec"` 作为查询名称, 正确写法是 `"SASRec"` 或 `"SASRec.__init__"`。

输出格式:
```json
{{
  "phase": "query",
  "observation": "你从实验数据中观察到的重要现象",
  "reasoning": "从观察推理出的初步瓶颈分析",
  "queries": [
    {{
      "action": "read_file",
      "args": {{ "file_key": "models.py" }}
    }},
    {{
      "action": "get_signature",
      "args": {{ "name": "SASRec.finetune" }}
    }},
    {{
      "action": "search_pattern",
      "args": {{ "pattern": "dropout" }}
    }}
  ],

  ⚠ **queries 必须是结构化对象数组, 不能是纯字符串!** 每个 query 必须包含 `action` 和 `args` 两个字段。不要写成 ["SASRec.__init__"] 这种简格式!

  "analysis_direction": "你打算往哪个方向分析, 需要哪些代码来确认"
}}
```

### 选择 B: 直接输出改进方案 (如果你已经有足够信息)
如果你觉得索引 + 之前查询到的代码已经足够做出决策, 直接输出最终方案。

输出格式:
```json
{{
  "phase": "proposal",
  "observation": "你从实验数据中观察到的重要现象 (指标趋势、异常值、差距模式等)",
  "reasoning": "从观察推理出的瓶颈分析 — 你认为模型为什么在这些方面表现不佳",
  "param_changes": {{
    "参数名": 新值
  }},
  "structural_changes": [
    {{
      "target_file": "要修改的源码文件 (如 modules.py)",
      "description": "修改内容描述 — 为什么改、改了什么",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与源码中的代码完全匹配",
          "replace": "修改后的代码片段"
        }}
      ],
      "expected_effect": "你预期这次修改会产生什么效果",
      "confidence": "你对这次修改的信心: 高/中/低"
    }}
  ],
  "rationale": "整体方案的理由 — 为什么选择这些改动而非其他方案"
}}
```

**⚠ 关键提醒:**
- **选择 A (query)** 是推荐的! 不要急于做决策, 先确认你理解了代码的实际实现!
- **edits 使用 SEARCH/REPLACE 格式**: search 必须是源码中**实际存在的代码片段** (精确复制)。
  如果你还没看过要修改的代码, 先用查询获取精确文本, 然后再写 edit!
- **严禁猜测或凭记忆编写 search 文本!** 先查询, 再修改!
- 当前 hidden_size = {current_hidden_size}
"""

# ── Phase 2: 查询结果返回 + 继续分析 ──

QUERY_BASED_PHASE2_PROMPT = """你之前提出的查询已经执行, 结果如下:

## 查询结果
{query_results}

## 回顾你的初步分析
你之前说到:
- 观察: {previous_observation}
- 探索方向: {previous_analysis_direction}

## 当前评估指标
```json
{current_metrics}
```

## 任务

现在你拿到了具体的代码, 请继续分析。你仍然有两种选择:

### 选择 A: 继续查询 (如果还需要更多代码)
如果查询结果还不够, 你可以提出更多查询。

### 选择 B: 输出改进方案 (如果信息足够)
如果你现在对代码有足够的理解, 请输出最终的改进方案。

输出格式同上一轮, 在 JSON 中用 `"phase": "query"` 或 `"phase": "proposal"` 表示你的选择。

**⚠ 再次提醒**: edits 的 search 文本必须与源码完全匹配! 你现在已经看到了精确的代码,
请直接从上面的查询结果中复制你要修改的代码作为 search 文本。
"""

# ── Phase 3: 最终提案 (强制输出) ──

QUERY_BASED_FINAL_PROMPT = """你已经进行了多轮代码查询, 现在必须输出最终的改进方案!

## 你已查询到的代码汇总
{all_queried_code}

## 当前评估指标
```json
{current_metrics}
```

## 你的观察与分析
- 观察: {observation_summary}
- 推理: {reasoning_summary}

## 任务

**必须输出改进方案** — 不要再提出查询!

### 输出格式
```json
{{
  "phase": "proposal",
  "observation": "你从实验数据中观察到的重要现象",
  "reasoning": "从观察推理出的瓶颈分析",
  "param_changes": {{
    "参数名": 新值
  }},
  "structural_changes": [
    {{
      "target_file": "要修改的源码文件",
      "description": "修改内容描述",
      "edits": [
        {{
          "search": "原始代码片段 — 必须与你查询到的源码完全匹配!",
          "replace": "修改后的代码片段"
        }}
      ],
      "expected_effect": "预期效果",
      "confidence": "信心: 高/中/低"
    }}
  ],
  "rationale": "整体方案理由"
}}
```

**⚠ 关键**: search 文本必须与你之前查询到的代码**完全匹配**! 直接从查询结果中复制精确文本!
当前 hidden_size = {current_hidden_size}
"""

# ═══════════════════════════════════════════════════════════════════════════════
# QUERY_BASED_FIX — 纠错阶段的代码查询 Prompt (核心新增!)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 设计理念:
#   旧方案: 纠错时塞全部源码 + 硬截断 15000 字符 → LLM 看到截断的残缺代码
#           → SEARCH/REPLACE 的 search 文本匹配失败 → 纠错反复失败
#   新方案: 只给 LLM 轻量索引 + traceback 详情 → LLM 按需查询出错代码
#           → 精准定位 bug → SEARCH/REPLACE 的 search 文本与源码完全匹配
#
# 工作流程:
#   1. 第一轮: LLM 收到 traceback + 代码索引 → 查询出错位置的代码
#   2. 后续轮: LLM 继续查询上下文 → 最终输出修复方案

# ── 纠错 Phase 1: traceback + 代码索引 → 查询出错位置 ──

QUERY_BASED_FIX_PHASE1_PROMPT = """训练运行失败，错误来自模型源码中的代码 bug。

**核心变化**: 你不再需要一次性阅读所有源码! 系统为你提供了一个**代码索引** (文件名 + 类/函数签名),
你可以**按需查询**出错位置的代码细节。这比把所有代码塞进 prompt 更精确 — 你能看到**完整的出错代码**!

## 错误详情
- **错误类型**: {_error_type}
- **错误分类**: {_error_category}
- **错误信息**:
```
{_error_message}
```

## Traceback 详情 (出错文件和行号)
{_traceback_details}

## 源码文件索引 (仅签名, 不含方法体)
{code_index}

## 查询到的代码 (如果之前已有查询结果)
{queried_code}

## 回滚黑名单 (不要再踩这些坑!)
{rollback_warning}

## 任务

你需要根据 traceback 精准定位 bug 并修复它。你有两种选择:

### 选择 A: 提出代码查询 (推荐 — 先查出错位置的完整代码!)

如果你还没看到出错位置的完整代码, **强烈建议先查询**! traceback 只告诉你行号,
你需要看到实际的代码才能写出精确的 SEARCH/REPLACE。

**⚠ 重要: 代码查询与代码编辑是两个完全不同的概念!**
- **代码查询** (queries): 用 `search_function`、`get_region` 等动作获取代码内容。args 中的 `name` 应是纯粹的标识符, 如 `"SASRec.__init__"` 或 `"finetune"`, **不要加任何前缀!**
- **代码编辑** (edits): 用 `"search"`/`"replace"` 字段修改源码。这是两种不同的 SEARCH 含义!
  - 查询动作 `search_function` → 搜索并返回代码定义
  - 编辑字段 `"search"` → 指定要替换的原始代码片段
  **千万不要把编辑的 search/replace 格式混入查询的 action/args 中!**
  例如, 不要写 `"SEARCH: class SASRec"` 作为查询名称, 正确写法是 `"SASRec"` 或 `"SASRec.__init__"`。

常用查询:
- `read_file` 查看整个出错文件
- `search_function` 查看出错的方法/函数完整实现
- `get_region` 查看出错行附近的代码上下文

输出格式:
```json
{{
  "phase": "query",
  "observation": "从 traceback 中观察到的出错模式",
  "reasoning": "初步 bug 定位分析",
  "queries": [
    {{
      "action": "search_function",
      "args": {{ "name": "出错的方法名" }}
    }},
    {{
      "action": "get_region",
      "args": {{ "file_key": "出错的文件", "start_line": 出错行号-10, "end_line": 出错行号+10 }}
    }}
  ],

  ⚠ **queries 必须是结构化对象数组, 不能是纯字符串!** 每个 query 必须包含 `action` 和 `args` 两个字段。不要写成 ["SASRec.__init__"] 这种简格式!

  "analysis_direction": "你打算如何定位和修复这个 bug"
}}
```

### 选择 B: 直接输出修复方案 (如果你已经从 traceback 确认了 bug)

输出格式:
```json
{{
  "phase": "proposal",
  "observation": "从 traceback 观察到的出错模式",
  "reasoning": "bug 根因分析",
  "structural_changes": [
    {{
      "target_file": "出错的文件",
      "description": "修复描述",
      "edits": [
        {{
          "search": "原始有 bug 的代码片段 — 必须与当前源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "修复此 bug 后训练应该能正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明 — 为什么这样修"
}}
```

**⚠ 关键提醒:**
- **选择 A (query) 是推荐的!** 不要急于修复, 先确认你看到了完整的出错代码!
- **edits 的 search 文本必须与源码完全匹配!** 如果你还没看过出错位置的代码, 先查询获取精确文本!
- **必须根据 traceback 的文件名和行号精准定位 bug, 不要猜测!**
- 当前 hidden_size = {current_hidden_size}
"""

# ── 纠错 Phase 2: 查询结果返回 → 继续分析或输出修复方案 ──

QUERY_BASED_FIX_PHASE2_PROMPT = """你之前提出的查询已经执行, 结果如下:

## 查询结果
{query_results}

## 回顾你的初步分析
你之前说到:
- 观察: {previous_observation}
- 探索方向: {previous_analysis_direction}

## Traceback 回顾
- **错误类型**: {_error_type}
- **错误信息**: {_error_message}

## 任务

现在你拿到了具体的代码, 请继续分析并修复 bug。你仍然有两种选择:

### 选择 A: 继续查询 (如果需要更多上下文)
如果查询结果还不够理解 bug, 你可以提出更多查询。

### 选择 B: 输出修复方案 (如果你已经理解了 bug)

输出格式同上一轮, 在 JSON 中用 `"phase": "query"` 或 `"phase": "proposal"` 表示你的选择。

**⚠ 关键**: edits 的 search 文本必须与你查询到的代码**完全匹配**! 直接从查询结果中复制精确文本!
"""

# ── 纠错 Phase 3: 最终修复方案 (强制输出) ──

QUERY_BASED_FIX_FINAL_PROMPT = """你已经进行了多轮代码查询, 现在必须输出最终的修复方案!

## 你已查询到的代码汇总
{all_queried_code}

## Traceback 回顾
- **错误类型**: {_error_type}
- **错误信息**: {_error_message}

## 你的观察与分析
- 观察: {observation_summary}
- 推理: {reasoning_summary}

## 任务

**必须输出修复方案** — 不要再提出查询!

### 输出格式
```json
{{
  "phase": "proposal",
  "observation": "从 traceback 观察到的出错模式",
  "reasoning": "bug 根因分析",
  "structural_changes": [
    {{
      "target_file": "出错的文件",
      "description": "修复描述",
      "edits": [
        {{
          "search": "原始有 bug 的代码片段 — 必须与你查询到的源码完全匹配!",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "修复此 bug 后训练应该能正常运行",
      "confidence": "信心: 高/中/低"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明 — 为什么这样修"
}}
```

**⚠ 关键**: search 文本必须与你之前查询到的代码**完全匹配**! 直接从查询结果中复制精确文本!
当前 hidden_size = {current_hidden_size}
"""

# ── 预检纠错 query 模式 Prompt ──

QUERY_BASED_PREFLIGHT_FIX_PROMPT = """在训练前对模型源码进行语法检查，发现以下问题。

**核心变化**: 你不再需要一次性阅读所有源码! 系统为你提供了一个**代码索引**,
你可以**按需查询**有语法错误的文件代码。这比把所有代码塞进 prompt 更精确!

## 语法检查结果
{_preflight_errors}

## 源码文件索引 (仅签名, 不含方法体)
{code_index}

## 查询到的代码 (如果之前已有查询结果)
{queried_code}

## 任务

修复这些语法错误。你有两种选择:

### 选择 A: 提出代码查询 (推荐 — 先查出错位置的完整代码!)

如果你还没看到出错位置的完整代码, **强烈建议先查询**!

**⚠ 重要: 代码查询与代码编辑是两个完全不同的概念!**
- **代码查询** (queries): 用 `search_function`、`read_file` 等动作获取代码内容。args 中的 `name` 应是纯粹的标识符, 如 `"SASRec.__init__"`, **不要加任何前缀!**
- **代码编辑** (edits): 用 `"search"`/`"replace"` 字段修改源码。这是两种不同的 SEARCH 含义!
  **千万不要把编辑的 search/replace 格式混入查询的 action/args 中!**
  例如, 不要写 `"SEARCH: class SASRec"` 作为查询名称, 正确写法是 `"SASRec"` 或 `"SASRec.__init__"`。

输出格式:
```json
{{
  "phase": "query",
  "observation": "从语法检查中观察到的错误模式",
  "reasoning": "初步修复方向",
  "queries": [
    {{
      "action": "read_file",
      "args": {{ "file_key": "有语法错误的文件" }}
    }}
  ],

  ⚠ **queries 必须是结构化对象数组, 不能是纯字符串!** 每个 query 必须包含 `action` 和 `args` 两个字段。不要写成 ["SASRec.__init__"] 这种简格式!

  "analysis_direction": "你打算如何修复这些语法错误"
}}
```

### 选择 B: 直接输出修复方案 (如果你已经从错误信息确认了问题)

输出格式:
```json
{{
  "phase": "proposal",
  "structural_changes": [
    {{
      "target_file": "有语法错误的文件",
      "description": "修复的语法错误描述",
      "edits": [
        {{
          "search": "有语法错误的代码片段 — 必须与当前源码匹配",
          "replace": "修正后的代码片段"
        }}
      ],
      "expected_effect": "修复语法错误后代码可以正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明"
}}
```

**⚠ 关键提醒:**
- **选择 A (query) 是推荐的!** 先查看有语法错误的文件代码, 确保修复的 search 文本匹配!
- **edits 的 search 文本必须与源码完全匹配!** 如果你还没看过出错文件, 先查询获取精确文本!
"""