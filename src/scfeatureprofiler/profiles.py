"""Canonical feature-by-target profiling orchestration."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .contrasts import Contrast, resolve_contrasts
from .inputs import resolve_expression_input
from .models import ExpressionSource, ResolvedExpression
from .statistics import (
    adjust_p_values,
    discrimination_metrics,
    effect_metrics,
    mann_whitney_p_value,
    summarize_expression,
    target_specificity,
)


def _resolve_features(
    available: pd.Index, features: Optional[Sequence[Any]]
) -> list[Any]:
    if features is None:
        return available.tolist()
    if isinstance(features, (str, bytes)):
        raise TypeError("`features` must be a sequence of feature identifiers.")
    selected = list(features)
    if not selected:
        raise ValueError("`features` cannot be empty.")
    if len(pd.Index(selected).unique()) != len(selected):
        raise ValueError("Requested feature identifiers must be unique.")
    missing = [feature for feature in selected if feature not in available]
    if missing:
        raise ValueError(f"Features are not present in the selected source: {missing}")
    return selected


def _validate_profile_configuration(
    detection_threshold: float, fold_change_pseudocount: float
) -> None:
    if not np.isfinite(detection_threshold):
        raise ValueError("`detection_threshold` must be finite.")
    if not np.isfinite(fold_change_pseudocount) or fold_change_pseudocount <= 0:
        raise ValueError(
            "`fold_change_pseudocount` must be finite and greater than zero."
        )


def _validate_execution_configuration(n_jobs: int, feature_chunk_size: int) -> None:
    if (
        isinstance(n_jobs, bool)
        or not isinstance(n_jobs, (int, np.integer))
        or n_jobs == 0
    ):
        raise ValueError("`n_jobs` must be a non-zero integer.")
    if (
        isinstance(feature_chunk_size, bool)
        or not isinstance(feature_chunk_size, (int, np.integer))
        or feature_chunk_size < 1
    ):
        raise ValueError("`feature_chunk_size` must be a positive integer.")


def _profile_one_feature(
    resolved: ResolvedExpression,
    feature: Any,
    prepared_contrasts: Sequence[tuple[Any, np.ndarray, np.ndarray]],
    *,
    detection_threshold: float,
    fold_change_pseudocount: float,
) -> list[dict[str, Any]]:
    """Profile one feature using masks shared across every feature."""
    values = resolved.extract_feature(feature).astype(float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"Feature {feature!r} contains non-finite expression values.")

    rows = []
    for contrast, target_mask, alternative_mask in prepared_contrasts:
        target_values = values[target_mask]
        alternative_values = values[alternative_mask]
        target_summary = summarize_expression(target_values, detection_threshold)
        alternative_summary = summarize_expression(
            alternative_values, detection_threshold
        )
        effect = effect_metrics(
            target_summary.mean_expression,
            alternative_summary.mean_expression,
            target_summary.median_expression,
            alternative_summary.median_expression,
            fold_change_pseudocount=fold_change_pseudocount,
        )
        discrimination = discrimination_metrics(target_values, alternative_values)
        n_comparison_cells = target_summary.n_cells + alternative_summary.n_cells
        rows.append(
            {
                "feature_id": feature,
                "target_group": contrast.target,
                "contrast_type": contrast.contrast_type,
                "alternative_groups": contrast.alternative_groups,
                "n_target_cells": target_summary.n_cells,
                "n_alternative_cells": alternative_summary.n_cells,
                "target_cell_fraction": (target_summary.n_cells / n_comparison_cells),
                "n_target_detected": target_summary.n_detected,
                "n_alternative_detected": alternative_summary.n_detected,
                "target_detection_fraction": target_summary.detection_fraction,
                "alternative_detection_fraction": (
                    alternative_summary.detection_fraction
                ),
                "target_mean_expression": target_summary.mean_expression,
                "alternative_mean_expression": alternative_summary.mean_expression,
                "target_median_expression": target_summary.median_expression,
                "alternative_median_expression": (
                    alternative_summary.median_expression
                ),
                **effect,
                "fold_change_pseudocount": fold_change_pseudocount,
                "target_specificity": target_specificity(
                    target_summary.detection_fraction,
                    alternative_summary.detection_fraction,
                ),
                **discrimination,
                "statistical_test": "mann_whitney_u_two_sided",
                "cell_level_p_value": mann_whitney_p_value(
                    target_values, alternative_values
                ),
                "multiple_testing_method": "benjamini_hochberg",
                "multiple_testing_scope": "complete_profile",
                "expression_source": resolved.source_descriptor,
                "detection_threshold": detection_threshold,
            }
        )
    return rows


def _profile_resolved(
    resolved: ResolvedExpression,
    selected_features: Sequence[Any],
    resolved_contrasts: Sequence[Any],
    *,
    detection_threshold: float,
    fold_change_pseudocount: float,
    n_jobs: int = 1,
    feature_chunk_size: int = 32,
) -> pd.DataFrame:
    """Profile an already resolved matrix and already validated contrasts."""
    _validate_execution_configuration(n_jobs, feature_chunk_size)
    prepared_contrasts = [
        (
            contrast,
            resolved.group_labels == contrast.target,
            np.isin(resolved.group_labels, contrast.alternative_groups),
        )
        for contrast in resolved_contrasts
    ]
    rows = []
    for start in range(0, len(selected_features), feature_chunk_size):
        feature_chunk = selected_features[start : start + feature_chunk_size]
        if n_jobs == 1:
            chunk_rows = [
                _profile_one_feature(
                    resolved,
                    feature,
                    prepared_contrasts,
                    detection_threshold=detection_threshold,
                    fold_change_pseudocount=fold_change_pseudocount,
                )
                for feature in feature_chunk
            ]
        else:
            chunk_rows = Parallel(n_jobs=n_jobs, prefer="threads")(
                delayed(_profile_one_feature)(
                    resolved,
                    feature,
                    prepared_contrasts,
                    detection_threshold=detection_threshold,
                    fold_change_pseudocount=fold_change_pseudocount,
                )
                for feature in feature_chunk
            )
        rows.extend(row for feature_rows in chunk_rows for row in feature_rows)

    result = pd.DataFrame(rows)
    result["cell_level_adjusted_p_value"] = adjust_p_values(
        result["cell_level_p_value"].to_numpy()
    )
    return result


def profile_features(
    data: Any,
    group_by: Any,
    *,
    features: Optional[Sequence[Any]] = None,
    feature_names: Optional[list[str]] = None,
    targets: Any = None,
    contrasts: Optional[Sequence[Contrast]] = None,
    expression_source: Union[str, ExpressionSource] = "X",
    detection_threshold: float = 0.0,
    fold_change_pseudocount: float = 1e-9,
    n_jobs: int = 1,
    feature_chunk_size: int = 32,
) -> pd.DataFrame:
    """Build a canonical feature profile for explicit population contrasts.

    Parallel work uses threads so the resolved expression matrix is shared rather
    than copied into worker processes. Features are submitted in bounded chunks;
    these execution controls do not alter the statistical calculations.
    """
    _validate_profile_configuration(detection_threshold, fold_change_pseudocount)
    _validate_execution_configuration(n_jobs, feature_chunk_size)
    resolved = resolve_expression_input(
        data,
        group_by,
        feature_names=feature_names,
        expression_source=expression_source,
    )
    selected_features = _resolve_features(resolved.feature_ids, features)
    resolved_contrasts = resolve_contrasts(
        resolved.group_labels, targets=targets, contrasts=contrasts
    )
    return _profile_resolved(
        resolved,
        selected_features,
        resolved_contrasts,
        detection_threshold=detection_threshold,
        fold_change_pseudocount=fold_change_pseudocount,
        n_jobs=n_jobs,
        feature_chunk_size=feature_chunk_size,
    )
