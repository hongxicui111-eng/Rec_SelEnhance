"""
Experiment Journal — 实验记录与历史检索
对应 Google 论文: Shared Context + Experiment Journal
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("rec_self_evolve.journal")


class ExperimentJournal:
    """
    实验日志系统
    - 记录每次进化的完整信息
    - 支持检索历史最优/失败/相似实验
    - 序列化为 JSONL 文件持久化
    """

    def __init__(self, file_path: str = "experiment_journal.jsonl"):
        self.file_path = Path(file_path)
        self.records = []
        self._load()

    def record(self, entry: dict):
        """
        记录一条实验日志
        entry 格式:
        {
            "iteration": int,
            "timestamp": str,
            "status": "SUCCESS" | "FAILED" | "ROLLED_BACK" | ...,
            "metrics": {...},
            "proposal": str,
            "error": str,
            "config": {...},
        }
        """
        if "timestamp" not in entry:
            entry["timestamp"] = datetime.now().isoformat()
        self.records.append(entry)
        self._save(entry)
        logger.info(f"Journal record [{entry['iteration']}]: {entry['status']}")

    def _save(self, entry: dict):
        """追加写入 JSONL"""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def _load(self):
        """从文件加载历史记录"""
        if self.file_path.exists():
            with open(self.file_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            self.records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            logger.info(f"Loaded {len(self.records)} historical records")

    # ════════════════════════════════════════
    # 检索方法
    # ════════════════════════════════════════

    def get_best(self, metric: str = "ndcg@5") -> Optional[dict]:
        """获取历史最优记录"""
        valid = [r for r in self.records if r.get("status") == "SUCCESS"
                 and r.get("metrics", {}).get(metric) is not None]
        if not valid:
            return None
        return max(valid, key=lambda r: r["metrics"][metric])

    def get_latest(self, n: int = 5) -> list:
        """获取最近的 N 条记录"""
        return self.records[-n:]

    def get_successful(self) -> list:
        """获取全部成功记录"""
        return [r for r in self.records if r.get("status") == "SUCCESS"]

    def get_failures(self) -> list:
        """获取失败记录"""
        return [r for r in self.records if r.get("status") != "SUCCESS"]

    def summarize(self, n: int = 10) -> str:
        """
        生成 Experiment Journal 摘要
        供 LLM 在下一轮分析中使用
        
        现在也包含结构修改信息与指标的关联
        """
        if not self.records:
            return "暂无历史实验记录（冷启动阶段）"

        # 最新 N 条
        recent = self.records[-n:]

        lines = [
            f"=== Experiment Journal Summary ===",
            f"Total records: {len(self.records)}",
            f"Successful: {len(self.get_successful())}",
            f"Failed/Skipped: {len(self.get_failures())}",
            "",
            f"--- Recent {len(recent)} iterations ---",
        ]

        for r in reversed(recent):
            ts = r.get("timestamp", "")[-19:]  # 只取时间部分
            it = r.get("iteration", "?")
            status = r.get("status", "?")
            metrics = r.get("metrics", {})

            # 显示主指标
            metric_str = ", ".join([
                f"{k}={float(v):.4f}" for k, v in metrics.items()
                if isinstance(v, (int, float))
            ][:5])

            # 标注是否有结构修改
            struct_tag = ""
            if r.get("structural_changes"):
                num_structs = len(r.get("structural_changes", []))
                struct_tag = f" 🏗️({num_structs}struct)"
            if r.get("status") == "STRUCTURE_ROLLBACK":
                struct_tag = " ↩struct_rollback"

            lines.append(f"[{ts}] iter={it} status={status}{struct_tag} | {metric_str}")

            # 显示结构修改详情 (如果有)
            if r.get("structural_changes"):
                for sc in r.get("structural_changes", [])[:3]:
                    lines.append(f"       🏗️ [{sc.get('target_file', '?')}] "
                                 f"{sc.get('target_class_or_function', '?')}: "
                                 f"{sc.get('description', '?')[:80]}")

            # 显示错误/备注
            if r.get("error"):
                lines.append(f"       error: {r['error'][:100]}")

        return "\n".join(lines)

    def get_trend(self, metric: str = "ndcg@5", window: int = 5) -> Optional[float]:
        """获取指标趋势 (正值 = 上升)"""
        values = [
            r["metrics"][metric] for r in self.records
            if r.get("status") == "SUCCESS"
            and r.get("metrics", {}).get(metric) is not None
        ]
        if len(values) < 3:
            return None

        recent_avg = sum(values[-window:]) / min(window, len(values))
        before_avg = sum(values[-(window * 2):-window]) / min(window, len(values))

        if before_avg == 0:
            return None
        return (recent_avg - before_avg) / before_avg * 100  # 百分比

    def to_dataframe(self):
        """导出为 DataFrame (用于分析)"""
        try:
            import pandas as pd
            return pd.DataFrame(self.records)
        except ImportError:
            return None