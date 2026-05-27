# -*- coding: utf-8 -*-
"""
数据基础设施模块 — 为 LLM 动态生成的代码提供运行环境

所有具体的数据获取、计算、模型探测逻辑都由任务运行时通过 LLM 动态生成代码实现。
此模块只提供基础设施：
  1. 数据发现 — 扫描项目中的数据文件路径、模型 checkpoint 等
  2. 脚本执行 — 执行 LLM 生成的脚本，带超时和错误处理
  3. 缓存管理 — 保存/加载计算或探测结果，避免重复执行
  4. 模型信息 — 发现模型 checkpoint 路径和源码位置
"""

import os
import json
import logging
import subprocess
import tempfile
import shutil
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger("rec_self_evolve.data_infrastructure")


class DataInfrastructure:
    """
    数据基础设施 — 为 LLM 动态生成的代码提供运行环境

    不预定义任何数据处理逻辑。
    所有数据获取、计算、模型探测的具体方法由 LLM 在运行时动态生成。
    """

    DEFAULT_SCRIPT_TIMEOUT = 1200  # 脚本执行超时 (秒)

    def __init__(self, project_root: str, data_dir: str = None, log_dir: str = None,
                 llm_client=None, model_args: Dict[str, Any] = None):
        self.project_root = Path(project_root)
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "data"
        self.log_dir = Path(log_dir) if log_dir else self.project_root / "logs"
        self.llm_client = llm_client
        self.model_args = model_args or {}  # 模型运行参数 (hidden_size, num_hidden_layers, item_size 等)

        # 缓存目录
        self._cache_dir = self.project_root / ".data_cache"
        self._cache_dir.mkdir(exist_ok=True)

        # 内存缓存
        self._memory_cache = {}

        # 数据发现缓存
        self._inventory_cache = None

        # 临时目录追踪（用于清理）
        self._temp_dirs = []

    # ─── 数据发现 ───────────────────────────────────────────────

    def discover_data(self) -> Dict:
        """
        发现项目中可用的数据资源（文件路径、元数据、模型 checkpoint 等）

        Returns:
            Dict 包含各类数据文件的路径信息，供 LLM 生成代码时参考
        """
        if self._inventory_cache is not None:
            return self._inventory_cache

        inventory = {
            "data_files": [],
            "metadata_files": [],
            "model_checkpoints": [],
            "precomputed_results": [],
            "error_case_files": [],
            "logs": []
        }

        # 探索数据目录 — 使用构造函数传入的路径
        scan_dirs = [
            self.data_dir,
            self.log_dir
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    fpath = Path(root) / f
                    try:
                        rel_path = fpath.relative_to(self.project_root)
                    except ValueError:
                        # log_dir 等目录可能在 project_root 外部，直接用绝对路径
                        rel_path = fpath
                    fsize = fpath.stat().st_size

                    if f.endswith(".txt"):
                        inventory["data_files"].append({
                            "path": str(rel_path),
                            "size": fsize,
                            "description": self._describe_file(f)
                        })
                    elif f.endswith(".json"):
                        if "meta" in f.lower():
                            inventory["metadata_files"].append({
                                "path": str(rel_path),
                                "size": fsize,
                                "description": "物品元数据"
                            })
                        else:
                            inventory["precomputed_results"].append({
                                "path": str(rel_path),
                                "size": fsize,
                                "description": self._describe_file(f)
                            })
                    elif f.endswith(".pt") or f.endswith(".pth"):
                        inventory["model_checkpoints"].append({
                            "path": str(rel_path),
                            "size": fsize,
                            "description": "模型检查点"
                        })
                    elif f.endswith(".log"):
                        inventory["logs"].append({
                            "path": str(rel_path),
                            "size": fsize,
                            "description": "训练日志"
                        })

        # 探索错误案例目录
        for error_dir in [self.project_root / "error_cases", self.project_root / "errors"]:
            if error_dir.exists():
                for f in error_dir.glob("*.json"):
                    inventory["error_case_files"].append({
                        "path": str(f.relative_to(self.project_root)),
                        "size": f.stat().st_size,
                        "description": "错误案例"
                    })

        self._inventory_cache = inventory
        return inventory

    def discover_model_info(self) -> Dict[str, Any]:
        """
        发现模型信息（checkpoint 路径、模型源码位置、运行参数、checkpoint 实际维度等）

        ⚠ 关键改进: 除了文件路径和源码，现在还会包含:
          1. model_args — 模型运行时参数 (hidden_size, num_hidden_layers, item_size 等)
          2. checkpoint_shapes — 从 checkpoint 中探测到的实际 tensor 形状
             (这确保了 LLM 生成的代码构建模型时维度与 checkpoint 完全一致)

        Returns:
            包含模型目录、checkpoint 路径、源码文件路径、运行参数、checkpoint 维度等信息的字典
        """
        recmodel_dir = self._find_recmodel_dir()
        checkpoint = self._find_latest_checkpoint()

        model_file = Path(recmodel_dir) / "models.py"
        modules_file = Path(recmodel_dir) / "modules.py"
        trainer_file = Path(recmodel_dir) / "trainers.py"

        model_code = ""
        if model_file.exists():
            with open(model_file, 'r') as f:
                model_code = f.read()

        modules_code = ""
        if modules_file.exists():
            with open(modules_file, 'r') as f:
                modules_code = f.read()

        # ── 新增: 模型运行参数 ──
        # model_args 可能包含 hidden_size, num_hidden_layers, item_size 等关键参数
        # 这些参数决定了模型的维度结构，必须与 checkpoint 一致才能正确 load_state_dict
        effective_model_args = self._resolve_model_args(checkpoint)

        # ── 新增: checkpoint tensor 形状探测 ──
        # 从 checkpoint 文件中读取实际的 tensor 形状，作为模型参数的最终验证
        checkpoint_shapes = self._probe_checkpoint_shapes(checkpoint)

        return {
            # 路径信息
            "project_root": str(self.project_root),
            "recmodel_dir": recmodel_dir,
            "data_dir": str(self.data_dir),
            "checkpoint": checkpoint,
            "model_file": str(model_file),
            "modules_file": str(modules_file),
            "trainer_file": str(trainer_file),
            # 模型运行参数 (最关键! 决定模型维度结构)
            "model_args": effective_model_args,
            # checkpoint 实际 tensor 形状 (用于验证参数是否与 checkpoint 一致)
            "checkpoint_shapes": checkpoint_shapes,
            # 模型源码 (供 LLM 参考模型定义)
            "model_code": model_code,
            "modules_code": modules_code,
        }

    def _resolve_model_args(self, checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """
        合并所有来源的模型运行参数，确保完整且与 checkpoint 一致

        参数来源优先级:
          1. checkpoint_shapes — checkpoint 中实际的 tensor 维度 (最权威)
          2. model_args — 构造时传入的参数 (来自 project_adapter.base_args)
          3. 默认值 — 最基本的 fallback

        Returns:
            合并后的模型参数字典，包含所有模型构造所需的参数
        """
        # 基础: 从构造函数传入的 model_args
        merged = dict(self.model_args)

        # 尝试从 checkpoint 探测来补充/修正参数
        if checkpoint:
            try:
                import torch
                ckpt_data = torch.load(checkpoint, map_location='cpu')
                # 从 checkpoint 的 item_embeddings 权重推断 item_size 和 hidden_size
                if 'item_embeddings.weight' in ckpt_data:
                    emb_shape = ckpt_data['item_embeddings.weight'].shape
                    # item_embeddings: [item_size, hidden_size]
                    merged['item_size'] = emb_shape[0]
                    merged['hidden_size'] = emb_shape[1]
                elif 'item_embeddings.weight' not in ckpt_data:
                    # 有些 checkpoint 可能只保存了部分权重，遍历所有 key 找 embedding
                    for key, tensor in ckpt_data.items():
                        if 'item_embedding' in key.lower() or 'embed' in key.lower():
                            if hasattr(tensor, 'shape') and len(tensor.shape) == 2:
                                # 推断 hidden_size 从 embedding 的第二维
                                if 'hidden_size' not in merged or merged.get('hidden_size') is None:
                                    merged['hidden_size'] = tensor.shape[1]
                                break
                # 从 position_embeddings 推断 max_seq_length
                if 'position_embeddings.weight' in ckpt_data:
                    pos_shape = ckpt_data['position_embeddings.weight'].shape
                    # position_embeddings: [max_seq_length, hidden_size]
                    merged['max_seq_length'] = pos_shape[0]
                    if 'hidden_size' not in merged:
                        merged['hidden_size'] = pos_shape[1]
            except Exception as e:
                logger.warning(f"Failed to probe model args from checkpoint: {e}")

        # 尝试从数据文件推断 item_size (item_size = max_item + 2)
        if 'item_size' not in merged:
            item_size = self._infer_item_size_from_data()
            if item_size is not None:
                merged['item_size'] = item_size

        return merged

    def _probe_checkpoint_shapes(self, checkpoint: Optional[str] = None) -> Dict[str, list]:
        """
        从 checkpoint 中提取所有 tensor 的名称和形状

        这让 LLM 可以知道 checkpoint 中实际的维度，确保构建模型时维度完全匹配。

        Returns:
            {"tensor_name": [shape_dimensions]} 的字典，例如:
            {
                "item_embeddings.weight": [12702, 64],
                "position_embeddings.weight": [50, 64],
                "item_encoder.layers.0.attention.query.weight": [64, 64],
                ...
            }
        """
        if not checkpoint:
            return {}

        try:
            import torch
            ckpt_data = torch.load(checkpoint, map_location='cpu')
            shapes = {}
            for key, tensor in ckpt_data.items():
                if hasattr(tensor, 'shape'):
                    shapes[key] = list(tensor.shape)
            return shapes
        except Exception as e:
            logger.warning(f"Failed to probe checkpoint shapes: {e}")
            return {}

    def _infer_item_size_from_data(self) -> Optional[int]:
        """
        从训练数据文件中推断 item_size (最大 item ID + 2)

        模型构造需要 item_size，但它来自数据而非 argparse。
        这确保 LLM 生成的代码能正确设置 item_size。
        """
        max_item = 0
        data_patterns = ["*_train.txt", "*_val.txt", "*_test.txt"]

        for pattern in data_patterns:
            for data_file in self.data_dir.glob(pattern):
                try:
                    with open(data_file, 'r') as f:
                        for line in f:
                            items = [int(x) for x in line.strip().split()]
                            if items:
                                max_item = max(max_item, max(items))
                except Exception:
                    continue

        if max_item > 0:
            return max_item + 2  # 与 run_finetune_full.py 一致
        return None

    def format_inventory_for_prompt(self, preloaded_data: Optional[Dict] = None) -> str:
        """
        将数据清单格式化为 LLM prompt 可用的文本

        只提供路径信息，不包含任何数据处理逻辑。
        """
        inventory = self.discover_data()

        lines = ["## 可用数据资源"]

        if preloaded_data:
            lines.append("\n### 已预加载数据 (可直接使用)")
            for key, value in preloaded_data.items():
                if value is not None:
                    if isinstance(value, list):
                        lines.append(f"- {key}: {len(value)} 项")
                    elif isinstance(value, dict):
                        lines.append(f"- {key}: {len(value)} 个键")
                    else:
                        lines.append(f"- {key}: {type(value).__name__}")

        lines.append("\n### 数据文件")
        for df in inventory["data_files"]:
            lines.append(f"- {df['path']} ({df.get('description', '')}, {df['size']/1024:.1f}KB)")

        if inventory["metadata_files"]:
            lines.append("\n### 元数据文件")
            for mf in inventory["metadata_files"]:
                lines.append(f"- {mf['path']} ({mf.get('description', '')})")

        if inventory["model_checkpoints"]:
            lines.append("\n### 模型检查点")
            for ckpt in inventory["model_checkpoints"][:5]:
                lines.append(f"- {ckpt['path']}")

        if inventory["error_case_files"]:
            lines.append("\n### 错误案例文件")
            for ef in inventory["error_case_files"]:
                lines.append(f"- {ef['path']}")

        return "\n".join(lines)

    # ─── 脚本执行 ───────────────────────────────────────────────

    def execute_script(self, script_content: str, timeout: int = None,
                       context: Dict[str, Any] = None) -> Optional[Dict]:
        """
        执行脚本（LLM 生成的计算脚本或模型探测脚本）

        Args:
            script_content: Python 脚本代码
            timeout: 执行超时（秒），默认使用 DEFAULT_SCRIPT_TIMEOUT
            context: 执行上下文，需要注入的变量

        Returns:
            执行结果字典，包含 success 和 data/error 字段
        """
        timeout = timeout or self.DEFAULT_SCRIPT_TIMEOUT
        context = context or {}

        script_dir = tempfile.mkdtemp()
        self._temp_dirs.append(script_dir)
        script_path = os.path.join(script_dir, "script.py")

        try:
            # 注入上下文变量
            context_lines = "\n".join([
                f"{k} = {repr(v)}" for k, v in context.items() if v is not None
            ])

            full_script = "# -*- coding: utf-8 -*-\n# Context data\n" + context_lines + "\n\n" + script_content

            with open(script_path, 'w') as f:
                f.write(full_script)

            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.project_root)
            )

            if result.returncode == 0:
                # 尝试从结果文件读取
                result_file = os.path.join(script_dir, "result.json")
                if os.path.exists(result_file):
                    with open(result_file, 'r') as f:
                        return json.load(f)

                # 尝试从 stdout 解析 JSON
                if result.stdout.strip():
                    try:
                        return json.loads(result.stdout)
                    except json.JSONDecodeError:
                        return {"success": True, "data": result.stdout}

                return {"success": True, "data": None}
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "stdout": result.stdout
                }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": f"Execution timeout ({timeout}s)"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ─── 缓存管理 ───────────────────────────────────────────────

    def save_to_cache(self, key: str, data: Any):
        """保存结果到缓存（文件 + 内存）"""
        cache_file = self._cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump(data, f, default=str)
            self._memory_cache[key] = data
        except Exception as e:
            logger.warning(f"Failed to save cache for '{key}': {e}")

    def load_from_cache(self, key: str) -> Optional[Any]:
        """从缓存加载结果（优先内存，其次文件）"""
        if key in self._memory_cache:
            return self._memory_cache[key]

        cache_file = self._cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                self._memory_cache[key] = data
                return data
            except Exception as e:
                logger.warning(f"Failed to load cache for '{key}': {e}")

        return None

    def clear_cache(self):
        """清空缓存"""
        self._memory_cache.clear()
        if self._cache_dir.exists():
            for f in self._cache_dir.glob("*.json"):
                try:
                    f.unlink()
                except:
                    pass

    # ─── 清理 ───────────────────────────────────────────────────

    def cleanup(self):
        """清理所有临时目录"""
        for temp_dir in self._temp_dirs:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
        self._temp_dirs.clear()

    # ─── 内部辅助 ───────────────────────────────────────────────

    def _find_recmodel_dir(self) -> str:
        """找到 Recmodel 目录（直接返回 project_root）"""
        return str(self.project_root)

    def _find_latest_checkpoint(self) -> Optional[str]:
        """查找最新的模型 checkpoint"""
        output_dir = Path(self._find_recmodel_dir()) / "output"
        if not output_dir.exists():
            return None
        checkpoints = list(output_dir.glob("*.pt"))
        if not checkpoints:
            return None
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime))

    def _describe_file(self, fname: str) -> str:
        """根据文件名推断描述"""
        fname_lower = fname.lower()
        if "train" in fname_lower:
            return "训练数据"
        elif "val" in fname_lower:
            return "验证数据"
        elif "test" in fname_lower:
            return "测试数据"
        elif "meta" in fname_lower:
            return "物品元数据"
        else:
            return "数据文件"