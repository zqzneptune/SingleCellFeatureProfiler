"""Focused statistical building blocks for canonical feature profiles."""

from .detection import summarize_expression
from .discrimination import discrimination_metrics
from .effect import effect_metrics
from .inference import adjust_p_values, mann_whitney_p_value
from .specificity import target_specificity

__all__ = [
    "adjust_p_values",
    "discrimination_metrics",
    "effect_metrics",
    "mann_whitney_p_value",
    "summarize_expression",
    "target_specificity",
]
