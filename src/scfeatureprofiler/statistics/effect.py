"""Expression-magnitude effect measurements."""

import numpy as np


def effect_metrics(
    target_mean: float,
    alternative_mean: float,
    target_median: float,
    alternative_median: float,
    *,
    fold_change_pseudocount: float,
) -> dict[str, float]:
    """Return signed differences and a guarded log2 arithmetic-mean ratio."""
    if not np.isfinite(fold_change_pseudocount) or fold_change_pseudocount <= 0:
        raise ValueError(
            "`fold_change_pseudocount` must be finite and greater than zero."
        )
    log2_mean_ratio = np.nan
    if (
        target_mean >= 0
        and alternative_mean >= 0
        and not (target_mean == 0 and alternative_mean == 0)
    ):
        log2_mean_ratio = float(
            np.log2(
                (target_mean + fold_change_pseudocount)
                / (alternative_mean + fold_change_pseudocount)
            )
        )
    return {
        "mean_difference": float(target_mean - alternative_mean),
        "median_difference": float(target_median - alternative_median),
        "log2_mean_ratio": log2_mean_ratio,
    }
