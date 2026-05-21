"""
LLM Prompt 模板库 — 支持超参数修改 + 模型结构修改
"""
import json

MLE_ANALYSIS_PROMPT = """你是一位顶尖的推荐系统算法工程师 (MLE)，正在负责优化一个序列推荐系统。

你的任务是**全面分析模型瓶颈，并提出改进方案**。
改进方案包含两大类:
1. **超参数修改**: 调整学习率、dropout、层数等参数
2. **模型结构修改**: 修改模型源码中的类/函数，添加新模块，改变计算逻辑等

**⚠ 重要: 结构修改才是核心!**
如果模型瓶颈是架构性的 (如 Self-Attention 无法捕获惊喜模式、位置编码缺乏时间衰减、FFN 结构不够灵活)，仅调参数无法解决，必须提出结构修改方案。

## 项目背景
{training_log}

## 历史实验记录
以下是 Experiment Journal 中历史最优的 8 条记录:
{experiment_journal}

## 当前评估指标
```json
{current_metrics}
```

## 当前模型源码
以下是模型的核心源码文件，你可以**直接提出修改方案**:

{source_code_context}

## 任务

请分析当前模型瓶颈并提出**改进方案** (必须包含参数修改 + 结构修改):

### 分析维度 (必须覆盖每个维度)

1. **指标短板分析**: NDCG/R/MRR 哪个指标最低？为什么仅靠调参无法改善？
2. **过拟合/欠拟合**: Loss 是否收敛？训练是否充分？如何通过结构修改改善？
3. **架构瓶颈诊断**: 
   - Self-Attention 的 Q/K/V 线性映射是否足够灵活？能否添加偏置/门控？
   - 位置编码是否充分捕捉时间衰减效应？是否需要相对位置编码？
   - FFN 的 4x 扩展比例是否最优？是否需要 GLU 门控？
   - Encoder 层堆叠是否需要跨层连接？
   - 物品嵌入是否有聚类问题？是否需要类别/时间辅助信息？
4. **惊喜交互瓶颈**: 模型为什么无法捕获与历史模式差异大的交互？需要什么结构性改动？
5. **采样与损失**: 当前负采样和损失函数是否有代码级别的优化空间？

### 输出格式 (严格遵守!)

**必须**输出以下 JSON 格式 (不要输出其他内容):

```json
{{
  "analysis": "瓶颈分析 — 必须包含架构层面的诊断，不要只说'指标不够好'",
  "param_changes": {{
    "参数名1": 新值,
    "参数名2": "新值"
  }},
  "structural_changes": [
    {{
      "action_type": "修改类型 (add_module/modify_attention/modify_ffn/modify_position_encoding/modify_embedding/modify_forward_pass/modify_training_logic/add_loss_component/modify_encoder_stack)",
      "target_file": "修改的目标文件 (models.py/modules.py/trainers.py)",
      "target_class_or_function": "修改的类名或函数名 (如 SelfAttention.forward, SASRec.add_position_embedding)",
      "description": "修改的具体描述 — 为什么要改、改了什么逻辑",
      "new_code": "完整的修改后的代码块 (Python 代码，包含类定义或函数定义的完整代码)",
      "insert_position": "插入位置 — 'replace_function' (替换整个函数) 或 'after_class_X' (在类 X 后添加新类) 或 'before_function_Y' (在函数 Y 前插入)",
      "expected_effect": "预期效果 — 会对哪些指标产生什么影响",
      "risk_level": "低/中/高 — 修改可能带来的风险"
    }}
  ],
  "explanation": "每个修改为什么做、预期提升什么指标、以及修改之间的协同关系"
}}
```

### 超参数修改范围
- **超参数**: lr(float, 1e-5~1e-2), batch_size(int, 64~4096), hidden_size(int, 32~512)
- **训练**: epochs(int, 50~1000), weight_decay(float, 0~0.1), seed
- **结构**: num_hidden_layers(1~8), num_attention_heads(1~16), hidden_dropout_prob(0~0.9)
- **损失**: loss_type(str, BCE/BPR)
- **采样**: neg_sampler(str, Uniform/DNS), N(50~1000), M(1~200)
- **对比学习**: CL_type(str, Radical/Gentle), start_epoch(0~200), K(0.01~0.5)

### 结构修改示例
以下是几个结构修改的示例，帮助你理解如何填写 `structural_changes`:

**示例 1: 修改 SelfAttention 添加时间衰减偏置**
```json
{{
  "action_type": "modify_attention",
  "target_file": "modules.py",
  "target_class_or_function": "SelfAttention.forward",
  "description": "在注意力分数上添加时间衰减偏置，使近期的交互获得更高的注意力权重，帮助模型更好地捕捉时间敏感的模式",
  "new_code": "def forward(self, input_tensor, attention_mask):\\n    ...\\n    attention_scores = attention_scores / math.sqrt(self.attention_head_size)\\n    # 添加时间衰减偏置\\n    seq_len = attention_scores.size(-1)\\n    time_decay = torch.exp(-0.1 * torch.arange(seq_len, device=attention_scores.device).float())\\n    time_decay = time_decay.unsqueeze(0).unsqueeze(0).unsqueeze(0)\\n    attention_scores = attention_scores + time_decay.log()\\n    attention_scores = attention_scores + attention_mask\\n    ...",
  "insert_position": "replace_function",
  "expected_effect": "提升 NDCG@5/10，特别改善对近期交互的预测准确度",
  "risk_level": "中"
}}
```

**示例 2: 添加多样性正则化损失**
```json
{{
  "action_type": "add_loss_component",
  "target_file": "trainers.py",
  "target_class_or_function": "_get_loss",
  "description": "在训练损失中添加多样性正则化项，惩罚推荐列表中物品的过度相似，鼓励模型探索更多类别",
  "new_code": "... diversity_loss = ...\\n    total_loss = bce_loss + 0.1 * diversity_loss\\n    ...",
  "insert_position": "replace_function",
  "expected_effect": "提升 Recall@20 和惊喜交互的捕获率",
  "risk_level": "低"
}}
```

### 重要提示
- 超参数范围严格遵守上述限制
- 不要修改参数名，使用精确的名称
- **structural_changes 中的 new_code 必须是可执行的完整 Python 代码**
- new_code 中的变量名和维度必须与 hidden_size (当前={current_hidden_size}) 对齐
- 如果修改了 modules.py 中的类，确保 models.py 中对应的 import 和使用也一致
- 如果添加了新参数 (如 time_decay_rate)，需要在 args 中声明，并在 __init__ 中使用 args.xxx
- **优先提出结构修改!** 如果只在 param_changes 里调 lr/dropout 而没有 structural_changes，说明你没有深入分析模型瓶颈
"""

# ──────────────────────────────────────────────
# 结构修改专用 Prompt (更深入的架构优化)
# ──────────────────────────────────────────────

STRUCTURE_OPTIMIZATION_PROMPT = """你是一位推荐系统架构设计专家，正在对 SASRec 模型进行结构优化。

## 背景
当前模型是 SASRec (Self-Attentive Sequential Recommendation)，基于标准 Transformer Encoder。
它在某些场景下表现不佳，需要通过**修改模型源码**来改善。

## 当前模型性能
- 整体: {overall_summary}
- 惊喜子集: {surprise_summary}
- 主要瓶颈: {bottleneck_summary}

## 当前模型源码
以下是你可以修改的核心源码文件:

{source_code_context}

## 模型配置
```
{config_summary}
```

## 已知的结构性问题
{known_issues}

## 任务
请提出**2-3 个具体的结构修改方案**，每个方案必须:
1. 修改具体的类/函数 (不能只说"应该改进"，必须给出完整代码)
2. 说明修改的理论依据 (引用相关工作或解释数学直觉)
3. 说明修改与现有代码的兼容性 (维度对齐、import 兼容、args 需要哪些新参数)

### 修改方向参考
- **Self-Attention 增强**: 相对位置偏置、时间衰减注意力、多头分组注意力
- **位置编码改进**: 时间衰减位置编码、可学习位置编码、相对位置编码
- **FFN 改进**: GLU 门控、SwiGLU、降低扩展比例
- **嵌入增强**: 类别嵌入辅助、时间嵌入辅助、嵌入正则化
- **训练改进**: 多样性正则化损失、对比学习增强、困难负采样改进
- **编码器改进**: 跨层连接、层间信息融合、残差缩放

### 输出格式
```json
{{
  "structural_changes": [
    {{
      "action_type": "...",
      "target_file": "...",
      "target_class_or_function": "...",
      "description": "...",
      "new_code": "... (完整的 Python 代码)",
      "insert_position": "...",
      "expected_effect": "...",
      "risk_level": "...",
      "theoretical_basis": "理论依据或数学直觉"
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
  "explanation": "整体修改策略说明"
}}
```

**关键要求: new_code 必须是可立即执行的完整 Python 代码，不要有省略号或占位符!**
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
  "analysis": "瓶颈分析",
  "param_changes": {{
    "参数名": "新值/新参数"
  }},
  "structural_changes": [
    {{
      "action_type": "...",
      "target_file": "...",
      "target_class_or_function": "...",
      "description": "...",
      "new_code": "...",
      "insert_position": "...",
      "expected_effect": "...",
      "risk_level": "..."
    }}
  ],
  "explanation": "修改理由"
}}
```"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 训练/运行错误反馈给 LLM
# ──────────────────────────────────────────────

ERROR_FEEDBACK_PROMPT = """你之前提出的改进方案在执行时遇到了错误，需要你**修正方案**后重新提出。

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

## 修正要求

根据错误信息，分析你的方案**为什么失败**，并提出**修正方案**:

### 错误诊断 (必须回答)
1. **错误根因**: 这个错误是因为什么引起的？(维度不匹配/缺少import/变量名错误/逻辑错误/...)
2. **具体代码定位**: 出错的是哪行代码？涉及哪些变量/维度？
3. **修正策略**: 应该如何修改才能避免这个错误？

### 输出格式 (严格遵守!)

```json
{{
  "error_diagnosis": "对错误根因的分析",
  "param_changes": {{
    "参数名": 新值
  }},
  "structural_changes": [
    {{
      "action_type": "修改类型",
      "target_file": "目标文件",
      "target_class_or_function": "目标类/函数",
      "description": "修正后的修改描述 — 说明了与上一版方案的区别和修正内容",
      "new_code": "修正后的完整 Python 代码 — 确保维度对齐、import完整、变量名正确",
      "insert_position": "插入位置",
      "expected_effect": "预期效果",
      "risk_level": "风险等级"
    }}
  ],
  "explanation": "修正说明 — 为什么要改、改了什么、如何避免之前的错误"
}}
```

**⚠ 关键提醒:**
- `new_code` 必须是**可执行的完整 Python 代码**，不要有省略号
- 确保所有维度与 `hidden_size` (当前={current_hidden_size}) 对齐
- 确保所有 import 都包含在内
- 如果之前的结构修改导致了维度不匹配，必须在新代码中修正维度计算
- 如果错误是 OOM/NaN，应该调整参数而非结构
"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 结构修改代码校验失败
# ──────────────────────────────────────────────

STRUCTURE_FIX_PROMPT = """你提出的模型结构修改代码在校验时发现语法/逻辑问题，需要你**修正代码**后重新输出。

## 你之前提出的结构修改
```json
{_original_structural_changes}
```

## 校验失败的详情
{_validation_failures}

## 当前模型源码 (原始状态 — 修改已被回滚)
{_current_source_code}

## 修正要求

请针对每个校验失败的结构修改，输出修正后的版本:

1. **语法错误**: 修正 Python 语法 (如括号不匹配、缩进错误、缺少冒号等)
2. **维度不匹配**: 确保代码中的维度计算与 hidden_size={current_hidden_size} 对齐
3. **Import 缺失**: 添加所有需要的 import 语句
4. **变量名错误**: 使用正确的变量名 (参考当前源码中的命名)
5. **继承/接口不兼容**: 确保修改后的类/函数签名与调用方兼容

### 输出格式 (严格遵守!)

```json
{{
  "structural_changes": [
    {{
      "action_type": "...",
      "target_file": "...",
      "target_class_or_function": "...",
      "description": "修正后的描述 — 说明修正了什么",
      "new_code": "修正后的完整可执行 Python 代码",
      "insert_position": "...",
      "expected_effect": "...",
      "risk_level": "..."
    }}
  ],
  "param_changes": {{
    "如果需要配合修改的超参数": 新值
  }},
  "explanation": "修正说明 — 与之前版本的区别"
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

## 诊断要求

1. **根因分析**: 训练为什么失败？(数据路径错误/配置参数问题/依赖缺失/OOM/...)
2. **修复方案**: 应应该如何修改才能让训练成功？

### 输出格式

```json
{{
  "diagnosis": "根因分析",
  "param_changes": {{
    "需要修改的参数": 新值
  }},
  "fix_suggestions": [
    "具体的修复建议1",
    "具体的修复建议2"
  ],
  "explanation": "为什么这些修改能解决训练失败问题"
}}
```"""

# ──────────────────────────────────────────────
# 自纠错 Prompt — 源码代码错误修复 (核心新增!)
# ──────────────────────────────────────────────

CODE_FIX_PROMPT = """训练运行失败，错误来自**模型源码中的代码 bug**。
你需要**直接修改源码文件**来修复这个错误，然后重新运行训练。

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

## 修复要求

你需要找到源码中的 bug 并修复它。常见错误类型:

1. **SyntaxError**: 语法错误 → 修正语法 (括号、缩进、缺少冒号等)
2. **NameError**: 变量名未定义 → 检查变量名拼写、添加缺失的变量定义
3. **TypeError**: 类型错误 → 检查参数类型、修正类型转换
4. **AttributeError**: 属性不存在 → 检查类/对象的属性名、修正类定义
5. **RuntimeError: size mismatch**: 维度不匹配 → 修正张量维度计算，确保与 hidden_size={current_hidden_size} 对齐
6. **ImportError**: 导入错误 → 添加缺失的 import 语句，修正导入路径
7. **IndentationError**: 缩进错误 → 修正缩进层级

### 修复策略
- **定位**: 根据 traceback 的文件名和行号，找到出错的代码位置
- **分析**: 理解出错行上下文的代码逻辑，判断 bug 的根因
- **修正**: 给出修正后的**完整代码块** (包含出错函数的完整定义，不能有省略号)
- **验证**: 确保修正后的代码: 1)语法正确 2)维度对齐 3)import完整 4)与现有代码兼容

### 输出格式 (严格遵守!)

```json
{{
  "error_diagnosis": "对 bug 根因的分析 — 出错的是哪行代码、为什么出错",
  "structural_changes": [
    {{
      "action_type": "bug_fix",
      "target_file": "出错的文件 (如 modules.py)",
      "target_class_or_function": "出错的类/函数 (如 SelfAttention.forward)",
      "description": "修复描述 — 修了什么 bug、修正了什么逻辑",
      "new_code": "修正后的完整函数代码 — 确保可直接替换执行",
      "insert_position": "replace_function",
      "expected_effect": "修复此 bug 后训练应该能正常运行",
      "risk_level": "低"
    }}
  ],
  "param_changes": {{
    "如果bug修复需要配合调整参数": 新值
  }},
  "explanation": "修复说明 — 为什么这样修、修正了什么"
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

## 修复要求

针对每个语法错误，给出修复后的完整代码:

### 输出格式

```json
{{
  "structural_changes": [
    {{
      "action_type": "syntax_fix",
      "target_file": "有语法错误的文件",
      "target_class_or_function": "出错的类/函数",
      "description": "修复的语法错误描述",
      "new_code": "修正后的完整代码 — 必须语法正确",
      "insert_position": "replace_function",
      "expected_effect": "修复语法错误后代码可以正常运行",
      "risk_level": "低"
    }}
  ],
  "param_changes": {{}},
  "explanation": "修复说明"
}}
```

**⚠ 关键: new_code 必须是语法正确的完整 Python 代码!**"""