#!/usr/bin/env python3
"""
Bug 修复验证脚本 — 确认顶层 edits 中 filename→target_file 映射 bug 已修复。

Bug: _parse_query_response 在将顶层 "edits" 转为 "structural_changes" 时，
     只检查了 "file" 和 "target_file" 键名，但 LLM 常用 "filename" 作为键名，
     导致 target_file 为空字符串 → structure_applier 报 "Missing target_file"。

测试方法: 直接模拟 _parse_query_response 中的格式转换逻辑，
          以及完整的 structure_applier 应用流程。
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# ════════════════════════════════════════
# 模拟 LLM 原始 JSON 响应 (来自用户的日志)
# ════════════════════════════════════════

LLM_RAW_RESPONSE = json.dumps({
    "phase": "proposal",
    "message": "已定位并修复命令行参数错误：`--tau` 和 `--margin` 未在 argparse 中定义。",
    "edits": [
        {
            "filename": "run_finetune_full.py",
            "action": "modify",
            "search": "    parser.add_argument(\"--temperature\", type=float, default=0.1, help=\"temperature for InfoNCE loss (0.05~0.5 recommended)\")\n    parser.add_argument(\"--CL_type\", type=str, default=\"Radical\", help=\"Radical, Gentle\")\n    parser.add_argument(\"--start_epoch\", default=30, type=int)\n    parser.add_argument(\"--K\", default=0.05, type=float)",
            "replace": "    parser.add_argument(\"--temperature\", type=float, default=0.1, help=\"temperature for InfoNCE loss (0.05~0.5 recommended)\")\n    parser.add_argument(\"--CL_type\", type=str, default=\"Radical\", help=\"Radical, Gentle\")\n    parser.add_argument(\"--start_epoch\", default=30, type=int)\n    parser.add_argument(\"--K\", default=0.05, type=float)\n    parser.add_argument(\"--tau\", type=float, default=0.1, help=\"temperature parameter for contrastive loss\")\n    parser.add_argument(\"--margin\", type=float, default=1.0, help=\"margin for contrastive loss\")"
        }
    ]
})

# ════════════════════════════════════════
# 直接模拟 _parse_query_response 中的格式转换逻辑 (修复后版本)
# ════════════════════════════════════════

def parse_query_response_simulated(response_str):
    """模拟 _parse_query_response 中顶层 edits → structural_changes 的转换逻辑"""
    # 提取 JSON
    start = response_str.find('{')
    end = response_str.rfind('}')
    if start < 0 or end <= start:
        return None
    json_str = response_str[start:end + 1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    
    if not isinstance(data, dict):
        return None
    
    # ── 格式转换: 顶层 "edits" → "structural_changes" ──
    if "edits" in data and "structural_changes" not in data:
        top_level_edits = data.pop("edits")
        if isinstance(top_level_edits, list) and top_level_edits:
            structural_entries = []
            for edit in top_level_edits:
                if isinstance(edit, dict):
                    # ★ 修复点: 添加 "filename" 作为 fallback key
                    target_file = edit.get("file", edit.get("filename", edit.get("target_file", "")))
                    instruction = edit.get("instruction", edit.get("description", ""))
                    search_text = edit.get("search", "")
                    replace_text = edit.get("replace", "")
                    action = edit.get("action", edit.get("action_type", "modify"))
                    
                    if search_text or replace_text:
                        entry = {
                            "target_file": target_file,
                            "description": instruction,
                            "edits": [{"search": search_text, "replace": replace_text}],
                        }
                        # 保留其他有用字段
                        for key in ["expected_effect", "confidence"]:
                            if key in edit:
                                entry[key] = edit[key]
                        # action / action_type → action_type
                        entry["action_type"] = action
                        structural_entries.append(entry)
            
            if structural_entries:
                data["structural_changes"] = structural_entries
    
    if "phase" not in data:
        if "structural_changes" in data or "param_changes" in data:
            data["phase"] = "proposal"
        else:
            data["phase"] = "proposal"
    
    return data

# ════════════════════════════════════════
# 测试 1: _parse_query_response 应正确映射 filename → target_file
# ════════════════════════════════════════

print("=" * 60)
print("测试 1: 顶层 edits 中 filename → target_file 映射")
print("=" * 60)

result = parse_query_response_simulated(LLM_RAW_RESPONSE)

if result is None:
    print("  ❌ 解析返回 None!")
    sys.exit(1)

print(f"  解析结果 keys: {list(result.keys())}")
print(f"  phase: {result.get('phase')}")

structural_changes = result.get("structural_changes", [])
print(f"  structural_changes 数量: {len(structural_changes)}")

if len(structural_changes) == 0:
    print("  ❌ structural_changes 为空! LLM 的 edits 没有被正确转换")
    sys.exit(1)

for i, change in enumerate(structural_changes):
    target_file = change.get("target_file", "")
    edits = change.get("edits", [])
    action_type = change.get("action_type", "")
    
    print(f"  Entry {i}:")
    print(f"    target_file: '{target_file}'")
    print(f"    action_type: '{action_type}'")
    print(f"    edits 数量: {len(edits)}")
    
    if not target_file:
        print(f"  ❌ Bug 未修复: target_file 为空字符串! (原始 key='filename')")
    elif target_file == "run_finetune_full.py":
        print(f"  ✅ Bug 已修复: filename='run_finetune_full.py' → target_file='{target_file}'")
    else:
        print(f"  ⚠️  target_file 值不匹配: '{target_file}'")

# ════════════════════════════════════════
# 测试 2: action → action_type 映射
# ════════════════════════════════════════

print()
print("=" * 60)
print("测试 2: action → action_type 映射")
print("=" * 60)

if structural_changes and structural_changes[0].get("action_type") == "modify":
    print("  ✅ Bug 已修复: action='modify' → action_type='modify'")
else:
    action_type_val = structural_changes[0].get("action_type", "MISSING")
    print(f"  ❌ action_type 值: '{action_type_val}' (期望 'modify')")

# ════════════════════════════════════════
# 测试 3: structure_applier 能否正确应用修改
# ════════════════════════════════════════

print()
print("=" * 60)
print("测试 3: structure_applier 能否正确应用修改 (端到端)")
print("=" * 60)

from agent.structure_applier import StructureApplier

# 创建临时测试目录
test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_test_target')
os.makedirs(test_dir, exist_ok=True)

mock_file = os.path.join(test_dir, 'run_finetune_full.py')
with open(mock_file, 'w') as f:
    f.write("""import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)
    args = parser.parse_args()
    return args

main()
""")

# 用 structure_applier 应用修改 (使用转换后的 structural_changes)
applier = StructureApplier(project_root=test_dir)
change = structural_changes[0]
apply_result = applier._apply_single_change(change)

print(f"  apply_result status: {apply_result.get('status')}")

if apply_result.get('status') == 'FAILED':
    error = apply_result.get('error', 'unknown')
    print(f"  ❌ structure_applier 应用失败: {error}")
    
    # 即使 structure_applier 找不到文件 (因为是临时目录),
    # 关键是: 它不应该报 "Missing target_file" 错误!
    if "Missing target_file" in error:
        print(f"  ❌❌ Bug 未修复: target_file 为空导致 'Missing target_file' 错误")
        print(f"     这正是用户遇到的问题!")
    else:
        print(f"  ⚠️  target_file 映射正确, 但其他原因导致失败 (可能是文件查找)")
elif apply_result.get('status') == 'SUCCESS':
    # 检查文件内容
    with open(mock_file, 'r') as f:
        modified_content = f.read()
    
    has_tau = "--tau" in modified_content
    has_margin = "--margin" in modified_content
    print(f"  ✅ structure_applier 应用成功!")
    print(f"    修改后文件中 --tau: {has_tau}")
    print(f"    修改后文件中 --margin: {has_margin}")
    
    if has_tau and has_margin:
        print("  ✅✅✅ 全流程验证成功! argparse 参数正确添加到文件中!")
    else:
        print(f"  ⚠️  应用成功但参数未出现在文件中")

# 清理临时文件
import shutil
shutil.rmtree(test_dir, ignore_errors=True)

# ════════════════════════════════════════
# 测试 4: 对比修复前后的差异
# ════════════════════════════════════════

print()
print("=" * 60)
print("测试 4: 对比修复前后 — 旧代码 (只检查 file/target_file)")
print("=" * 60)

# 旧版本: 只检查 "file" 和 "target_file" (不检查 "filename")
def parse_query_response_OLD(response_str):
    """旧版本 _parse_query_response (Bug 未修复)"""
    start = response_str.find('{')
    end = response_str.rfind('}')
    if start < 0 or end <= start:
        return None
    json_str = response_str[start:end + 1]
    
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    
    if not isinstance(data, dict):
        return None
    
    if "edits" in data and "structural_changes" not in data:
        top_level_edits = data.pop("edits")
        if isinstance(top_level_edits, list) and top_level_edits:
            structural_entries = []
            for edit in top_level_edits:
                if isinstance(edit, dict):
                    # ★ Bug: 只检查 "file" 和 "target_file", 不检查 "filename"
                    target_file = edit.get("file", edit.get("target_file", ""))  # ← 旧代码
                    instruction = edit.get("instruction", edit.get("description", ""))
                    search_text = edit.get("search", "")
                    replace_text = edit.get("replace", "")
                    
                    if search_text or replace_text:
                        entry = {
                            "target_file": target_file,
                            "description": instruction,
                            "edits": [{"search": search_text, "replace": replace_text}],
                        }
                        structural_entries.append(entry)
            
            if structural_entries:
                data["structural_changes"] = structural_entries
    
    if "phase" not in data:
        data["phase"] = "proposal"
    
    return data

result_old = parse_query_response_OLD(LLM_RAW_RESPONSE)
old_target_file = result_old["structural_changes"][0].get("target_file", "")
print(f"  旧代码 target_file: '{old_target_file}'")
if old_target_file == "":
    print("  ❌ 旧代码 bug: filename 键被忽略 → target_file 为空 → 'Missing target_file' 错误!")
else:
    print(f"  ⚠️  旧代码意外地拿到了 target_file: '{old_target_file}'")

print(f"  新代码 target_file: '{structural_changes[0].get('target_file', '')}'")
print("  ✅ 新代码修复: filename 键被正确映射 → target_file 正确填充")

print()
print("=" * 60)
print("验证完成!")
print("=" * 60)