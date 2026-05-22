"""
进化质量守卫 — 监控进化方向, 检测退化与停滞
对应 Self-EvolveRec: 方向性反馈验证环
"""
import logging
from typing import Optional

logger = logging.getLogger("rec_self_evolve.quality_guard")


class EvolutionQualityGuard:
    """
    进化质量守卫
    - 检测连续退化 → 回滚到历史最优
    - 检测收敛停滞 → 切换探索策略
    - 维护最优 checkpoint 记录
    """

    def __init__(self, window_size: int = 5,
                 degrade_threshold: float = 0.95,
                 plateau_threshold: float = 0.001,
                 primary_metric: str = "ndcg@5"):
        self.window_size = window_size
        self.degrade_threshold = degrade_threshold
        self.plateau_threshold = plateau_threshold
        self.primary_metric = primary_metric
        self.history = []          # 指标历史
        self.best_index = -1       # 历史最优轮次
        self.best_metrics = {}     # 历史最优指标
        self.best_config = None    # 历史最优配置

    def update(self, iteration: int, metrics: dict, config: Optional[dict] = None):
        """
        更新历史记录
        返回决策:
        {
            "action": "CONTINUE" | "REVERT_TO_BEST" | "SWITCH_STRATEGY",
            "reason": str,
            "best_iteration": int,
            "strategy": str (可选),
        }
        """
        entry = {"iteration": iteration, "metrics": metrics, "config": config}
        self.history.append(entry)

        # ---- 更新最优 ----
        current_value = self._get_primary_value(metrics)
        best_value = self._get_primary_value(self.best_metrics)

        if current_value is not None and (
            best_value is None or current_value > best_value
        ):
            self.best_index = iteration
            self.best_metrics = metrics
            self.best_config = config
            logger.info(f"New best at iteration {iteration}: {self._format_metrics(metrics)}")

        # ---- 需要足够历史才能做决策 ----
        if len(self.history) < 3:
            return {"action": "CONTINUE"}

        # ---- 检查 1: 连续退化 ----
        degrade = self._check_degradation()
        if degrade:
            logger.warning(f"Degradation detected: {degrade['reason']}")
            return {
                "action": "REVERT_TO_BEST",
                "reason": degrade["reason"],
                "best_iteration": self.best_index,
                "best_metrics": self.best_metrics,
            }

        # ---- 检查 2: 收敛停滞 ----
        plateau = self._check_plateau()
        if plateau:
            logger.info(f"Plateau detected: {plateau['reason']}")
            new_strategy = self._select_next_strategy()
            return {
                "action": "SWITCH_STRATEGY",
                "reason": plateau["reason"],
                "strategy": new_strategy,
            }

        return {"action": "CONTINUE"}

    # ════════════════════════════════════════
    # 内部检查
    # ════════════════════════════════════════

    def _check_degradation(self) -> Optional[dict]:
        """
        检查是否连续退化
        策略: 最近 3 轮 vs 前 3 轮, 下降超过阈值
        """
        if len(self.history) < 6:
            return None

        recent = [self._get_primary_value(h["metrics"]) for h in self.history[-3:]]
        before = [self._get_primary_value(h["metrics"]) for h in self.history[-6:-3]]

        recent_values = [v for v in recent if v is not None]
        before_values = [v for v in before if v is not None]

        if len(recent_values) < 2 or len(before_values) < 2:
            return None

        avg_recent = sum(recent_values) / len(recent_values)
        avg_before = sum(before_values) / len(before_values)

        if avg_recent < avg_before * self.degrade_threshold:
            return {
                "type": "degradation",
                "reason": f"指标退化: 近3轮均值 {avg_recent:.4f} < 前3轮均值 {avg_before:.4f} × {self.degrade_threshold} (下降 {(1 - avg_recent / avg_before) * 100:.1f}%)",
                "before_avg": avg_before,
                "recent_avg": avg_recent,
                "drop_pct": (1 - avg_recent / avg_before) * 100,
            }

        if len(recent_values) >= 3:
            if recent_values[0] > recent_values[1] > recent_values[2]:
                return {
                    "type": "monotonic_decrease",
                    "reason": f"连续3轮严格递减: {recent_values[0]:.4f} → {recent_values[1]:.4f} → {recent_values[2]:.4f}",
                    "values": recent_values,
                }

        return None

    def _check_plateau(self) -> Optional[dict]:
        """
        检查是否收敛停滞
        策略: 最近 N 轮提升 < 阈值
        """
        if len(self.history) < self.window_size:
            return None

        recent = self.history[-self.window_size:]
        improvements = []

        for i in range(1, len(recent)):
            v_curr = self._get_primary_value(recent[i]["metrics"])
            v_prev = self._get_primary_value(recent[i - 1]["metrics"])
            if v_curr is not None and v_prev is not None and v_prev > 0:
                improvements.append((v_curr - v_prev) / v_prev)

        if not improvements:
            return None

        if all(abs(delta) < self.plateau_threshold for delta in improvements):
            return {
                "type": "plateau",
                "reason": f"近{self.window_size}轮提升均 < {self.plateau_threshold} (停滞), 最大提升: {max(improvements) if improvements else 0:.4f}",
                "window": self.window_size,
                "max_improvement": max(improvements) if improvements else 0,
            }

        return None

    # ════════════════════════════════════════
    # 策略选择
    # ════════════════════════════════════════

    def _select_next_strategy(self) -> str:
        """根据当前状态选择下一个探索策略"""
        # 轮换策略
        modes = ["aggressive", "explorative", "conservative", "focused"]
        used_count = sum(1 for h in self.history if h.get("strategy"))
        return modes[used_count % len(modes)]

    def _get_primary_value(self, metrics: dict) -> Optional[float]:
        """获取主指标的值"""
        if not metrics:
            return None

        # 尝试多种可能的键名
        for key in [self.primary_metric, "ndcg", "hr", "auc", "recall"]:
            if key in metrics:
                return float(metrics[key])
            # 模糊匹配
            for mkey in metrics:
                if key in mkey.lower():
                    return float(metrics[mkey])

        # 任意非空指标
        for v in metrics.values():
            try:
                return float(v)
            except (ValueError, TypeError):
                continue

        return None

    @staticmethod
    def _format_metrics(metrics: dict) -> str:
        """格式化指标输出"""
        return " | ".join([f"{k}={v:.4f}" for k, v in metrics.items()
                           if isinstance(v, (int, float))])

    def get_summary(self) -> dict:
        """获取进化过程摘要"""
        if not self.history:
            return {"status": "no_data"}
        return {
            "total_iterations": len(self.history),
            "best_iteration": self.best_index,
            "best_metrics": self.best_metrics,
            "latest_metrics": self.history[-1]["metrics"],
        }


class SafetyGuardrails:
    """
    安全护栏系统
    对应 Google 论文: Guardrails (指标阈值守卫)
    """

    def __init__(self, rules: dict = None):
        """
        rules 格式:
        {
            "ndcg@5": {"min": 0.0, "max": 1.0, "regression_limit": 0.05},
        }
        """
        self.rules = rules or {}

    def check_metrics(self, metrics: dict, baseline: Optional[dict] = None) -> list:
        """检查指标是否违反护栏"""
        violations = []
        for metric_name, thresholds in self.rules.items():
            current = metrics.get(metric_name)
            if current is None:
                continue

            if "min" in thresholds and current < thresholds["min"]:
                violations.append(f"{metric_name}={current} < min={thresholds['min']}")
            if "max" in thresholds and current > thresholds["max"]:
                violations.append(f"{metric_name}={current} > max={thresholds['max']}")

            # 对比 baseline 的回退限制
            if baseline and "regression_limit" in thresholds:
                base_val = baseline.get(metric_name)
                if base_val and current < base_val * (1 - thresholds["regression_limit"]):
                    violations.append(
                        f"{metric_name} regression: {base_val} → {current} "
                        f"(limit: {thresholds['regression_limit']})"
                    )

        return violations