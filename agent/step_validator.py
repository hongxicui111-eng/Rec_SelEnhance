# -*- coding: utf-8 -*-
"""
步骤验证器模块 - 验证每个步骤的执行结果

功能：
1. 数据完整性验证 - 检查必要字段是否存在
2. 数据格式验证 - 检查数据类型和范围
3. 统计特征验证 - 检查统计特征是否符合预期
4. 对比验证 - 与基准或预期结果对比
5. LLM辅助验证 - 使用LLM进行复杂判断
"""

import os
import json
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger("rec_self_evolve.step_validator")


class ValidationType(Enum):
    """验证类型"""
    DATA_COMPLETENESS = "data_completeness"  # 数据完整性
    DATA_FORMAT = "data_format"  # 数据格式
    STATISTICS = "statistics"  # 统计特征
    COMPARISON = "comparison"  # 对比验证
    LLM_ASSISTED = "llm_assisted"  # LLM辅助验证
    CUSTOM = "custom"  # 自定义验证


@dataclass
class ValidationRule:
    """验证规则"""
    rule_id: str
    validation_type: ValidationType
    field_path: str  # 要验证的字段路径，如 "output_data.embeddings"
    check_fn: Optional[Callable] = None  # 自定义检查函数
    expected: Any = None  # 期望值
    min_value: Optional[float] = None  # 最小值
    max_value: Optional[float] = None  # 最大值
    not_null: bool = False  # 是否不能为空
    pattern: Optional[str] = None  # 正则模式
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """验证结果"""
    valid: bool
    rule_id: str
    field_path: str
    message: str
    details: Dict[str, Any] = None
    severity: str = "error"  # error, warning, info


class StepValidator:
    """
    步骤验证器 - 验证每个步骤的执行结果

    支持多种验证方式：
    1. 规则验证 - 基于预定义规则的验证
    2. 统计验证 - 基于统计特征的验证
    3. 对比验证 - 与参考数据的对比
    4. LLM验证 - 使用LLM进行智能验证
    """

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.validation_rules: Dict[str, List[ValidationRule]] = {}
        self.validation_history: List[ValidationResult] = []

    def add_rule(self, step_id: str, rule: ValidationRule):
        """为步骤添加验证规则"""
        if step_id not in self.validation_rules:
            self.validation_rules[step_id] = []
        self.validation_rules[step_id].append(rule)

    def add_rules(self, step_id: str, rules: List[ValidationRule]):
        """批量添加验证规则"""
        if step_id not in self.validation_rules:
            self.validation_rules[step_id] = []
        self.validation_rules[step_id].extend(rules)

    def validate_step(
        self,
        step_id: str,
        result_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        验证步骤结果

        Args:
            step_id: 步骤ID
            result_data: 步骤执行结果数据
            context: 额外上下文信息

        Returns:
            验证结果字典
        """
        rules = self.validation_rules.get(step_id, [])
        if not rules:
            # 没有规则，默认通过
            return {"valid": True, "message": "No validation rules defined", "details": {}}

        all_results = []
        for rule in rules:
            validation_result = self._validate_rule(rule, result_data, context)
            all_results.append(validation_result)

        # 记录到历史
        self.validation_history.extend(all_results)

        # 汇总结果
        errors = [r for r in all_results if r.severity == "error" and not r.valid]
        warnings = [r for r in all_results if r.severity == "warning" and not r.valid]

        return {
            "valid": len(errors) == 0,
            "message": "Validation failed" if errors else "Validation passed",
            "errors": [asdict(r) for r in errors],
            "warnings": [asdict(r) for r in warnings],
            "details": {r.rule_id: asdict(r) for r in all_results}
        }

    def _validate_rule(
        self,
        rule: ValidationRule,
        result_data: Dict[str, Any],
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """验证单条规则"""
        try:
            # 获取字段值
            field_value = self._get_nested_field(result_data, rule.field_path)

            # 根据验证类型执行验证
            if rule.validation_type == ValidationType.DATA_COMPLETENESS:
                return self._validate_completeness(rule, field_value)
            elif rule.validation_type == ValidationType.DATA_FORMAT:
                return self._validate_format(rule, field_value)
            elif rule.validation_type == ValidationType.STATISTICS:
                return self._validate_statistics(rule, field_value, context)
            elif rule.validation_type == ValidationType.COMPARISON:
                return self._validate_comparison(rule, field_value, context)
            elif rule.validation_type == ValidationType.LLM_ASSISTED:
                return self._validate_with_llm(rule, field_value, context)
            elif rule.validation_type == ValidationType.CUSTOM and rule.check_fn:
                return rule.check_fn(rule, field_value, context)
            else:
                return ValidationResult(
                    valid=True,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message="No specific validation needed"
                )

        except Exception as e:
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                message=f"Validation error: {str(e)}",
                severity="error"
            )

    def _get_nested_field(self, data: Dict, path: str) -> Any:
        """获取嵌套字段"""
        keys = path.split(".")
        value = data
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            elif isinstance(value, list) and key.isdigit():
                idx = int(key)
                value = value[idx] if idx < len(value) else None
            else:
                return None
        return value

    def _validate_completeness(self, rule: ValidationRule, field_value: Any) -> ValidationResult:
        """验证数据完整性"""
        # 检查是否为空
        if field_value is None:
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_id,
                message=f"Field {rule.field_path} is None",
                severity="error"
            )

        # 检查not_null
        if rule.not_null:
            if isinstance(field_value, (list, dict, str)) and len(field_value) == 0:
                return ValidationResult(
                    valid=False,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message=f"Field {rule.field_path} is empty",
                    severity="error"
                )

        # 检查期望值
        if rule.expected is not None:
            if field_value != rule.expected:
                return ValidationResult(
                    valid=False,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message=f"Field {rule.field_path} expected {rule.expected}, got {field_value}",
                    severity="error"
                )

        return ValidationResult(
            valid=True,
            rule_id=rule.rule_id,
            field_path=rule.field_path,
            message="Completeness check passed"
        )

    def _validate_format(self, rule: ValidationRule, field_value: Any) -> ValidationResult:
        """验证数据格式"""
        # 类型检查
        expected_type = rule.metadata.get("type")
        if expected_type:
            type_map = {
                "int": int,
                "float": (int, float),
                "str": str,
                "list": list,
                "dict": dict,
                "numpy_array": lambda x: isinstance(x, np.ndarray)
            }
            expected = type_map.get(expected_type)
            if expected and not isinstance(field_value, expected):
                return ValidationResult(
                    valid=False,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message=f"Field {rule.field_path} expected type {expected_type}, got {type(field_value).__name__}",
                    severity="error"
                )

        # 范围检查
        if rule.min_value is not None or rule.max_value is not None:
            if isinstance(field_value, (int, float)):
                if rule.min_value is not None and field_value < rule.min_value:
                    return ValidationResult(
                        valid=False,
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        message=f"Field {rule.field_path} value {field_value} less than min {rule.min_value}",
                        severity="error"
                    )
                if rule.max_value is not None and field_value > rule.max_value:
                    return ValidationResult(
                        valid=False,
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        message=f"Field {rule.field_path} value {field_value} greater than max {rule.max_value}",
                        severity="error"
                    )

        # 形状检查（针对numpy数组）
        if isinstance(field_value, np.ndarray):
            expected_shape = rule.metadata.get("shape")
            if expected_shape:
                actual_shape = field_value.shape
                if len(actual_shape) != len(expected_shape):
                    return ValidationResult(
                        valid=False,
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        message=f"Field {rule.field_path} shape {actual_shape} does not match expected {expected_shape}",
                        severity="error"
                    )
                for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
                    if expected != -1 and actual != expected:
                        return ValidationResult(
                            valid=False,
                            rule_id=rule.rule_id,
                            field_path=rule.field_path,
                            message=f"Field {rule.field_path} dimension {i}: {actual} != {expected}",
                            severity="error"
                        )

        return ValidationResult(
            valid=True,
            rule_id=rule.rule_id,
            field_path=rule.field_path,
            message="Format check passed"
        )

    def _validate_statistics(
        self,
        rule: ValidationRule,
        field_value: Any,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """验证统计特征"""
        if not isinstance(field_value, (np.ndarray, list)):
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                message=f"Cannot compute statistics for type {type(field_value).__name__}",
                severity="error"
            )

        # 转换为numpy数组
        arr = np.array(field_value)

        # 计算统计量
        stats = {
            "mean": np.mean(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "median": np.median(arr),
            "shape": arr.shape
        }

        # 检查统计量是否在预期范围内
        checks = rule.metadata.get("checks", {})
        for stat_name, check_range in checks.items():
            if stat_name in stats:
                value = stats[stat_name]
                min_val, max_val = check_range.get("min"), check_range.get("max")
                if min_val is not None and value < min_val:
                    return ValidationResult(
                        valid=False,
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        message=f"Statistic {stat_name}={value} less than min {min_val}",
                        severity=check_range.get("severity", "error"),
                        details={"statistic": stat_name, "value": value, "expected_min": min_val}
                    )
                if max_val is not None and value > max_val:
                    return ValidationResult(
                        valid=False,
                        rule_id=rule.rule_id,
                        field_path=rule.field_path,
                        message=f"Statistic {stat_name}={value} greater than max {max_val}",
                        severity=check_range.get("severity", "error"),
                        details={"statistic": stat_name, "value": value, "expected_max": max_val}
                    )

        return ValidationResult(
            valid=True,
            rule_id=rule.rule_id,
            field_path=rule.field_path,
            message="Statistics check passed",
            details=stats
        )

    def _validate_comparison(
        self,
        rule: ValidationRule,
        field_value: Any,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """验证对比结果"""
        reference = rule.metadata.get("reference")
        if reference:
            # 从context获取参考值
            ref_value = self._get_nested_field(context, reference) if context else None
            if ref_value is not None:
                # 计算差异
                if isinstance(field_value, (int, float)) and isinstance(ref_value, (int, float)):
                    diff = abs(field_value - ref_value)
                    tolerance = rule.metadata.get("tolerance", 0.01)
                    if diff > tolerance:
                        return ValidationResult(
                            valid=False,
                            rule_id=rule.rule_id,
                            field_path=rule.field_path,
                            message=f"Difference {diff} exceeds tolerance {tolerance}",
                            severity="error",
                            details={"value": field_value, "reference": ref_value, "diff": diff}
                        )

        return ValidationResult(
            valid=True,
            rule_id=rule.rule_id,
            field_path=rule.field_path,
            message="Comparison check passed"
        )

    def _validate_with_llm(
        self,
        rule: ValidationRule,
        field_value: Any,
        context: Optional[Dict[str, Any]]
    ) -> ValidationResult:
        """使用LLM验证"""
        if not self.llm_client:
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                message="LLM client not available",
                severity="warning"
            )

        prompt = rule.metadata.get("prompt")
        if not prompt:
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                message="No LLM prompt provided",
                severity="error"
            )

        try:
            response = self.llm_client.chat(prompt.format(
                field_value=field_value,
                context=context,
                **rule.metadata
            ))

            # 解析LLM响应
            if "valid" in response.lower() or "pass" in response.lower():
                return ValidationResult(
                    valid=True,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message="LLM validation passed",
                    details={"llm_response": response}
                )
            else:
                return ValidationResult(
                    valid=False,
                    rule_id=rule.rule_id,
                    field_path=rule.field_path,
                    message=f"LLM validation failed: {response}",
                    severity="warning",
                    details={"llm_response": response}
                )

        except Exception as e:
            return ValidationResult(
                valid=False,
                rule_id=rule.rule_id,
                field_path=rule.field_path,
                message=f"LLM validation error: {str(e)}",
                severity="warning"
            )

    def get_validation_history(self, step_id: Optional[str] = None) -> List[ValidationResult]:
        """获取验证历史"""
        if step_id:
            rules = self.validation_rules.get(step_id, [])
            rule_ids = {r.rule_id for r in rules}
            return [r for r in self.validation_history if r.rule_id in rule_ids]
        return self.validation_history

    def clear_history(self):
        """清空验证历史"""
        self.validation_history.clear()


def create_embedding_validation_rules(step_id: str) -> List[ValidationRule]:
    """创建嵌入向量验证规则（用于H6验证）"""
    return [
        ValidationRule(
            rule_id=f"{step_id}_embeddings_exist",
            validation_type=ValidationType.DATA_COMPLETENESS,
            field_path="output_data.embeddings",
            not_null=True,
            metadata={"description": "检查嵌入向量是否存在"}
        ),
        ValidationRule(
            rule_id=f"{step_id}_embeddings_shape",
            validation_type=ValidationType.DATA_FORMAT,
            field_path="output_data.embeddings",
            metadata={
                "type": "numpy_array",
                "description": "检查嵌入向量形状"
            }
        ),
        ValidationRule(
            rule_id=f"{step_id}_embeddings_nan",
            validation_type=ValidationType.STATISTICS,
            field_path="output_data.embeddings",
            metadata={
                "checks": {
                    "mean": {"min": -10, "max": 10},
                    "std": {"min": 0, "max": 100}
                },
                "description": "检查嵌入向量统计特征"
            }
        ),
        ValidationRule(
            rule_id=f"{step_id}_gradient_log_exist",
            validation_type=ValidationType.DATA_COMPLETENESS,
            field_path="output_data.gradient_log",
            not_null=True,
            metadata={"description": "检查梯度日志是否存在"}
        )
    ]


def create_training_validation_rules(step_id: str) -> List[ValidationRule]:
    """创建训练验证规则"""
    return [
        ValidationRule(
            rule_id=f"{step_id}_loss_decreased",
            validation_type=ValidationType.STATISTICS,
            field_path="output_data.losses",
            metadata={
                "checks": {
                    "mean": {"max": 100, "severity": "warning"}
                },
                "description": "检查损失是否在合理范围"
            }
        ),
        ValidationRule(
            rule_id=f"{step_id}_converged",
            validation_type=ValidationType.COMPARISON,
            field_path="output_data.final_loss",
            metadata={
                "reference": "context.initial_loss",
                "tolerance": 0.5,
                "description": "检查损失是否收敛"
            }
        )
    ]


# 辅助函数：转换为字典
def asdict(obj):
    """将dataclass转换为字典"""
    if hasattr(obj, '__dataclass_fields__'):
        return {k: asdict(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
    elif isinstance(obj, dict):
        return {k: asdict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [asdict(v) for v in obj]
    else:
        return obj
