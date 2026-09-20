"""Detection and expression summaries."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExpressionSummary:
    n_cells: int
    n_detected: int
    detection_fraction: float
    mean_expression: float
    median_expression: float


def summarize_expression(
    values: np.ndarray, detection_threshold: float
) -> ExpressionSummary:
    """Summarize one feature in one non-empty population."""
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("Expression summaries require at least one cell.")
    if not np.isfinite(values).all():
        raise ValueError("Expression values must be finite.")
    detected = values > detection_threshold
    return ExpressionSummary(
        n_cells=int(values.size),
        n_detected=int(detected.sum()),
        detection_fraction=float(detected.mean()),
        mean_expression=float(values.mean()),
        median_expression=float(np.median(values)),
    )
