"""Target-specific detection specificity."""

import numpy as np


def target_specificity(
    target_detection_fraction: float,
    alternative_detection_fraction: float,
) -> float:
    """Return target detection divided by total target-plus-alternative detection."""
    denominator = target_detection_fraction + alternative_detection_fraction
    if denominator == 0:
        return np.nan
    return float(target_detection_fraction / denominator)
