"""
LLM Prompt 模板库 — 开放探索式优化

设计原则:
  - 给 LLM 完整的实验现象，让它自己观察和推理
  - 不预设修改方向、参数范围、分析维度
  - LLM 有权选择纯调参、改结构、或两者结合
  - 输出格式保持结构化以方便系统解析，但字段灵活可选
"""
import json

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
      "target_file": "要修改的源码文件路径",
      "target_class_or_function": "要修改的类名或函数名",
      "description": "修改内容描述 — 为什么改、改了什么",
      "new_code": "修改后的完整代码 (Python)",
      "insert_position": "replace_function 或 append_to_file 或 replace_class",
      "expected_effect": "你预期这次修改会产生什么效果",
      "confidence": "你对这次修改的信心: 高/中/低"
    }}
  ],
  "rationale": "整体方案的理由 — 为什么选择这些改动而非其他方案"
}}
```

**注意:**
- `param_changes` 和 `structural_changes` 都是可选的 — 如果你觉得纯调参最合适，就只写 param_changes; 如果改结构更合理，就重点写 structural_changes
- `new_code` 需要是可执行的 Python 代码，不能有省略号或占位符
- 如果修改了某个类的接口，确保调用方也相应更新
- 当前 hidden_size = {current_hidden_size}
"""

# ──────────────────────────────────────────────
# 结构深度优化 Prompt — 给 LLM 更大的探索空间
# ──────────────────────────────────────────────

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
      "target_file": "源码文件路径",
      "target_class_or_function": "类名或函数名",
      "description": "修改内容描述",
      "new_code": "完整的修改后 Python 代码",
      "insert_position": "replace_function / replace_class / append_to_file",
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

**关键: new_code 必须是可直接执行的完整 Python 代码，不要有省略号或占位符!**
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
      "target_class_or_function": "...",
      "description": "...",
      "new_code": "...",
      "insert_position": "...",
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
- **错误类型**: {_error_type}
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
      "target_class_or_function": "要修改的类/函数",
      "description": "修正内容描述",
      "new_code": "修正后的完整 Python 代码",
      "insert_position": "replace_function / replace_class / append_to_file",
      "expected_effect": "预期效果",
      "confidence": "信心等级"
    }}
  ],
  "rationale": "修正策略说明 — 与上一版方案的区别和原因"
}}
```

**⚠ 关键提醒:**
- `new_code` 必须是可执行的完整 Python 代码，不能有省略号
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
      "target_class_or_function": "类/函数",
      "description": "修正后的描述",
      "new_code": "修正后的完整可执行 Python 代码",
      "insert_position": "replace_function / replace_class / append_to_file",
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

**⚠ 关键: new_code 必须是可以直接替换执行的完整代码，不能有任何省略号!**
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
      "target_class_or_function": "出错的类/函数",
      "description": "修复描述",
      "new_code": "修正后的完整函数代码",
      "insert_position": "replace_function",
      "expected_effect": "修复此 bug 后训练应该能正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明 — 为什么这样修"
}}
```

**⚠ 关键: new_code 必须是可以直接替换执行的完整代码，不能有任何省略号!**
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
      "target_class_or_function": "出错的类/函数",
      "description": "修复的语法错误描述",
      "new_code": "修正后的完整代码",
      "insert_position": "replace_function",
      "expected_effect": "修复语法错误后代码可以正常运行",
      "confidence": "高"
    }}
  ],
  "param_changes": {{}},
  "rationale": "修复说明"
}}
```

**⚠ 关键: new_code 必须是语法正确的完整 Python 代码!**"""