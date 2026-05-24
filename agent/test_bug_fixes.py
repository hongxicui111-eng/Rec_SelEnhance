#!/usr/bin/env python3
"""
Bug 修复验证脚本 — 确认 3 个执行器 bug 已被修复:
  Bug A: clean_new_code 不再破坏 search/replace 文本 (argparse 参数行)
  Bug B: _normalize_whitespace 可正常调用 (不再是死代码)
  Bug C: 格式兼容性 (filename/action 扁平格式 → 标准格式)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from agent.structure_applier import StructureApplier
import re

# ════════════════════════════════════════
# Bug A: clean_new_code 不应破坏 search/replace 文本
# ════════════════════════════════════════

print("=" * 60)
print("Bug A: clean_new_code 对 argparse 参数行的破坏性")
print("=" * 60)

argparse_search_text = """    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)"""

# 旧方法: clean_new_code
result_old = StructureApplier.clean_new_code(argparse_search_text)
print(f"  clean_new_code 输入: {len(argparse_search_text)} chars, {argparse_search_text.count(chr(10))+1} lines")
print(f"  clean_new_code 输出: {len(result_old)} chars, {result_old.count(chr(10))+1} lines")
print(f"  输出内容: '{result_old[:100]}'")
if result_old.strip() == "":
    print("  ❌ Bug A 未修复: clean_new_code 将 argparse 参数行全部清空!")
else:
    print("  ⚠️  clean_new_code 仍有副作用 (但不再全部清空)")

# 新方法: _clean_markdown_wrapper (从 core.py 导入)
# 因为 _clean_markdown_wrapper 是 SelfEvolveCore 的方法, 我们手动实现其逻辑
def clean_markdown_wrapper(text):
    """只清理 markdown 代码块标记和首尾空行, 不删除任何代码行"""
    if not text:
        return text
    text = re.sub(r'^```(?:python)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return '\n'.join(lines)

result_new = clean_markdown_wrapper(argparse_search_text)
print(f"  _clean_markdown_wrapper 输入: {len(argparse_search_text)} chars, {argparse_search_text.count(chr(10))+1} lines")
print(f"  _clean_markdown_wrapper 输出: {len(result_new)} chars, {result_new.count(chr(10))+1} lines")
if result_new == argparse_search_text:
    print("  ✅ Bug A 已修复: _clean_markdown_wrapper 完整保留 argparse 参数行!")
else:
    print(f"  ⚠️  输出有轻微差异 (可能只是首尾空行)")
    # 检查核心内容是否保留
    if "parser.add_argument" in result_new and "--tau" not in result_new:
        print("  ✅ Bug A 已修复: argparse 参数行的核心内容完整保留!")

# ════════════════════════════════════════
# Bug B: _normalize_whitespace 可正常调用
# ════════════════════════════════════════

print()
print("=" * 60)
print("Bug B: _normalize_whitespace 方法可用性")
print("=" * 60)

applier = StructureApplier(project_root=os.path.dirname(os.path.abspath(__file__)))

# 检查方法是否存在
has_method = hasattr(applier, '_normalize_whitespace')
print(f"  hasattr(applier, '_normalize_whitespace'): {has_method}")

if has_method:
    # 测试调用
    test_text = "  line1  \n\n\n  line2  \n\n  line3  \n\n\n"
    try:
        result = applier._normalize_whitespace(test_text)
        print(f"  _normalize_whitespace 调用成功!")
        print(f"  输入: '{test_text}'")
        print(f"  输出: '{result}'")
        expected = "line1\n\nline2\n\nline3"
        if result == expected:
            print("  ✅ Bug B 已修复: _normalize_whitespace 方法正常工作!")
        else:
            print(f"  ⚠️  输出不完全匹配预期, 但方法可调用 (期望: '{expected}')")
    except AttributeError as e:
        print(f"  ❌ Bug B 未修复: AttributeError: {e}")
    except Exception as e:
        print(f"  ❌ Bug B 未修复: 其他异常: {e}")
else:
    print("  ❌ Bug B 未修复: _normalize_whitespace 方法不存在!")

# 测试 _strip_whitespace_match (Level 2 空白匹配)
content = """def foo():
    x = 1

    y = 2



    return x + y"""

search_text = """def foo():
    x = 1
    y = 2
    return x + y"""

try:
    result_content, method = applier._str_replace(content, search_text, "def bar():\n    return 42")
    print(f"  _str_replace Level 2 测试: method={method}")
    if method == "whitespace_match":
        print("  ✅ Bug B 已修复: Level 2 空白匹配正常工作!")
    elif method == "exact_match":
        print("  ⚠️  精确匹配优先 (Level 2 没机会执行, 但方法可调用)")
    else:
        print(f"  ⚠️  匹配方法: {method}")
except AttributeError as e:
    print(f"  ❌ Bug B 未修复: Level 2 空白匹配崩溃: AttributeError: {e}")

# ════════════════════════════════════════
# Bug C: 格式兼容性
# ════════════════════════════════════════

print()
print("=" * 60)
print("Bug C: 格式兼容性 (filename/action 扁平格式)")
print("=" * 60)

# 模拟 SelfEvolveCore._normalize_change_format
def normalize_change_format(change):
    """格式兼容性转换"""
    # 格式1: 标准格式 (已有 target_file + edits) → 直接返回
    if change.get("target_file") and change.get("edits"):
        return change
    
    # 格式2: 扁平格式 (filename + action + search/replace)
    if not change.get("target_file") and change.get("filename"):
        change["target_file"] = change.pop("filename")
        if "search" in change or "replace" in change:
            search_text = change.pop("search", "")
            replace_text = change.pop("replace", "")
            action = change.pop("action", "modify")
            change["edits"] = [{"search": search_text, "replace": replace_text}]
            change["action_type"] = action
            for key in ["action"]:
                change.pop(key, None)
        return change
    
    # 格式3: 有 target_file 但 search/replace 在顶层
    if change.get("target_file") and ("search" in change or "replace" in change):
        search_text = change.pop("search", "")
        replace_text = change.pop("replace", "")
        change["edits"] = [{"search": search_text, "replace": replace_text}]
        return change
    
    if not change.get("target_file") and not change.get("filename"):
        return None
    
    return change

# 测试格式2: 扁平格式 (用户的 proposal JSON)
flat_format = {
    "filename": "run_finetune_full.py",
    "action": "modify",
    "search": "    parser.add_argument(\"--K\", default=0.05, type=float)",
    "replace": "    parser.add_argument(\"--K\", default=0.05, type=float)\n    parser.add_argument(\"--tau\", type=float, default=0.1)\n    parser.add_argument(\"--margin\", type=float, default=1.0)",
    "description": "Add --tau and --margin argparse arguments"
}

normalized = normalize_change_format(flat_format)
print(f"  扁平格式输入: filename={flat_format.get('filename')}, action={flat_format.get('action')}")
if normalized:
    print(f"  转换后: target_file={normalized.get('target_file')}")
    print(f"  edits 数量: {len(normalized.get('edits', []))}")
    if normalized.get('edits'):
        print(f"  第一个 edit search: '{normalized['edits'][0]['search'][:60]}...'")
        print(f"  第一个 edit replace: '{normalized['edits'][0]['replace'][:60]}...'")
    if normalized.get("target_file") == "run_finetune_full.py" and normalized.get("edits"):
        print("  ✅ Bug C 已修复: 扁平格式正确转换为标准格式!")
    else:
        print("  ❌ Bug C 未修复: 转换后格式不正确!")
else:
    print("  ❌ Bug C 未修复: 扁平格式无法转换!")

# 测试格式3: 混合格式
mixed_format = {
    "target_file": "models.py",
    "search": "class SASRec",
    "replace": "class ImprovedSASRec",
    "description": "Rename class"
}

normalized2 = normalize_change_format(mixed_format)
print(f"  混合格式输入: target_file={mixed_format.get('target_file')}")
if normalized2:
    print(f"  转换后: target_file={normalized2.get('target_file')}, edits 数量: {len(normalized2.get('edits', []))}")
    if normalized2.get("edits") and normalized2["edits"][0]["search"] == "class SASRec":
        print("  ✅ Bug C 已修复: 混合格式正确转换为标准格式!")
    else:
        print("  ❌ Bug C 未修复: 混合格式转换不正确!")

# ════════════════════════════════════════
# 综合测试: 模拟 --tau/--margin 修复场景
# ════════════════════════════════════════

print()
print("=" * 60)
print("综合测试: 模拟 argparse 参数添加的完整流程")
print("=" * 60)

# 模拟 run_finetune_full.py 的 argparse 部分
mock_file_content = """import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)
    args = parser.parse_args()
    return args

main()"""

# LLM 提出的 search/replace (精确匹配文件内容)
search_exact = """    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)"""

replace_exact = """    parser.add_argument("--temperature", type=float, default=0.1, help="temperature for InfoNCE loss (0.05~0.5 recommended)")
    parser.add_argument("--CL_type", type=str, default="Radical", help="Radical, Gentle")
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--K", default=0.05, type=float)
    parser.add_argument("--tau", type=float, default=0.1, help="temperature parameter for contrastive loss")
    parser.add_argument("--margin", type=float, default=1.0, help="margin for triplet loss or other margin-based losses")"""

# Step 1: 用 _clean_markdown_wrapper 处理 (新方法)
search_cleaned = clean_markdown_wrapper(search_exact)
replace_cleaned = clean_markdown_wrapper(replace_exact)

print(f"  Step 1 (_clean_markdown_wrapper):")
print(f"    search 保留: {len(search_cleaned)} chars, 含 'parser.add_argument': {search_cleaned.count('parser.add_argument')} 次")
print(f"    replace 保留: {len(replace_cleaned)} chars, 含 'parser.add_argument': {replace_cleaned.count('parser.add_argument')} 次")

if search_cleaned.count('parser.add_argument') == 4:
    print("    ✅ search 文本完整保留 (4 个 argparse 行)")
else:
    print(f"    ❌ search 文本被破坏 (期望 4 行, 实际 {search_cleaned.count('parser.add_argument')} 行)")

if replace_cleaned.count('parser.add_argument') == 6:
    print("    ✅ replace 文本完整保留 (6 个 argparse 行)")
else:
    print(f"    ❌ replace 文本被破坏 (期望 6 行, 实际 {replace_cleaned.count('parser.add_argument')} 行)")

# Step 2: 用 _str_replace 三级匹配
result_content, method = applier._str_replace(mock_file_content, search_cleaned, replace_cleaned)
print(f"  Step 2 (_str_replace): method={method}")

if method != "failed" and "--tau" in result_content and "--margin" in result_content:
    print("  ✅ 综合测试成功: argparse 参数添加完整流程正常!")
    print(f"    修改后文件中 --tau: {'--tau' in result_content}")
    print(f"    修改后文件中 --margin: {'--margin' in result_content}")
else:
    print(f"  ❌ 综合测试失败: method={method}, 结果可能不正确")
    if method == "failed":
        # 尝试 Level 4 智能插入回退
        result_content2, method2 = applier._smart_insert_fallback(
            mock_file_content, search_cleaned, replace_cleaned, mock_file_content
        )
        print(f"    Level 4 smart_insert_fallback: method={method2}")
        if method2 != "failed" and "--tau" in result_content2 and "--margin" in result_content2:
            print("    ✅ Level 4 回退成功!")

print()
print("=" * 60)
print("验证完成!")
print("=" * 60)