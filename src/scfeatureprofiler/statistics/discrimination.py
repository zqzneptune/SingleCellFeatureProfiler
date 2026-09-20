"""Cell-level descriptive discrimination measurements."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def discrimination_metrics(
    target_values: np.ndarray, alternative_values: np.ndarray
) -> dict[str, float]:
    """Treat target membership as positive and expression as the score."""
    labels = np.concatenate(
        [
            np.ones(target_values.size, dtype=int),
            np.zeros(alternative_values.size, dtype=int),
        ]
    )
    scores = np.concatenate([target_values, alternative_values])
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
    }
