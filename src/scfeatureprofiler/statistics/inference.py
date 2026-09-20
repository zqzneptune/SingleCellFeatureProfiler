"""Cell-level inferential summaries and multiplicity adjustment."""

import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


def mann_whitney_p_value(
    target_values: np.ndarray, alternative_values: np.ndarray
) -> float:
    """Two-sided cell-level Mann–Whitney U p-value."""
    if np.all(target_values == target_values[0]) and np.all(
        alternative_values == target_values[0]
    ):
        return 1.0
    return float(
        mannwhitneyu(
            target_values,
            alternative_values,
            alternative="two-sided",
            method="auto",
        ).pvalue
    )


def adjust_p_values(p_values: np.ndarray) -> np.ndarray:
    """BH-adjust all finite p-values while preserving missing positions."""
    p_values = np.asarray(p_values, dtype=float)
    adjusted = np.full(p_values.shape, np.nan, dtype=float)
    finite = np.isfinite(p_values)
    if finite.any():
        adjusted[finite] = multipletests(p_values[finite], method="fdr_bh")[1]
    return adjusted
