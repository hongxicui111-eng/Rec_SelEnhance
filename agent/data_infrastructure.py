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

    def __init__(self, project_root: str, data_dir: str = None, log_dir: str = None, llm_client=None):
        self.project_root = Path(project_root)
        self.data_dir = Path(data_dir) if data_dir else self.project_root / "Recmodel" / "data"
        self.log_dir = Path(log_dir) if log_dir else self.project_root / "logs"
        self.llm_client = llm_client

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

        # 探索数据目录
        scan_dirs = [
            self.project_root / "Recmodel" / "data",
            self.project_root / "data",
            self.project_root / "logs"
        ]

        for scan_dir in scan_dirs:
            if not scan_dir.exists():
                continue
            for root, dirs, files in os.walk(scan_dir):
                for f in files:
                    fpath = Path(root) / f
                    rel_path = fpath.relative_to(self.project_root)
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
        发现模型信息（checkpoint 路径、模型源码位置等）

        Returns:
            包含模型目录、checkpoint 路径、源码文件路径等信息的字典
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

        return {
            "project_root": str(self.project_root),
            "recmodel_dir": recmodel_dir,
            "data_dir": str(self.data_dir),
            "checkpoint": checkpoint,
            "model_file": str(model_file),
            "modules_file": str(modules_file),
            "trainer_file": str(trainer_file),
            "model_code": model_code,
            "modules_code": modules_code,
        }

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
        """找到 Recmodel 目录"""
        recmodel_dir = self.project_root
        if recmodel_dir.exists():
            return str(recmodel_dir)
        for parent in self.project_root.parents:
            candidate = parent / "Recmodel"
            if candidate.exists():
                return str(candidate)
        return str(recmodel_dir)

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