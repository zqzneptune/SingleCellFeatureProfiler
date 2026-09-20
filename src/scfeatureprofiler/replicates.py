"""Biological-replicate-aware feature profiling."""

from collections.abc import Sequence
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .contrasts import Contrast, resolve_contrasts
from .inputs import resolve_expression_input
from .models import ExpressionSource, ReplicateProfileResult
from .profiles import (
    _profile_resolved,
    _resolve_features,
    _validate_profile_configuration,
)
from .statistics import effect_metrics, summarize_expression, target_specificity

PROFILE_KEYS = [
    "feature_id",
    "target_group",
    "contrast_type",
    "alternative_groups",
]


def _ordered_unique(values: np.ndarray) -> list[Any]:
    return sorted(
        pd.unique(values).tolist(),
        key=lambda value: (type(value).__name__, repr(value)),
    )


def _effect_direction(effect: float, tolerance: float) -> int:
    if effect > tolerance:
        return 1
    if effect < -tolerance:
        return -1
    return 0


def _sample_summary(
    sample_rows: list[dict[str, Any]],
    *,
    pooled_mean_difference: float,
    effect_direction_tolerance: float,
) -> dict[str, Any]:
    evaluable = [row for row in sample_rows if row["sample_evaluable"]]
    effects = np.asarray(
        [row["sample_mean_difference"] for row in evaluable], dtype=float
    )
    n_available = len(sample_rows)
    n_evaluated = len(evaluable)
    positive = int((effects > effect_direction_tolerance).sum())
    negative = int((effects < -effect_direction_tolerance).sum())
    zero = n_evaluated - positive - negative

    summary = {
        "n_biological_samples_available": n_available,
        "n_biological_samples_evaluated": n_evaluated,
        "n_biological_samples_insufficient_cells": n_available - n_evaluated,
        "n_samples_positive_effect": positive,
        "n_samples_negative_effect": negative,
        "n_samples_zero_effect": zero,
        "effect_recurrence_fraction": np.nan,
        "directional_consistency_fraction": np.nan,
        "sample_mean_difference_mean": np.nan,
        "sample_mean_difference_median": np.nan,
        "sample_mean_difference_standard_deviation": np.nan,
        "sample_mean_difference_minimum": np.nan,
        "sample_mean_difference_maximum": np.nan,
        "sample_effect_dominance": np.nan,
        "dominant_sample_id": None,
    }
    if not n_evaluated:
        return summary

    pooled_direction = _effect_direction(
        pooled_mean_difference, effect_direction_tolerance
    )
    sample_directions = np.asarray(
        [_effect_direction(effect, effect_direction_tolerance) for effect in effects]
    )
    absolute_effects = np.abs(effects)
    absolute_sum = absolute_effects.sum()
    dominant_index = int(np.argmax(absolute_effects))
    summary.update(
        {
            "effect_recurrence_fraction": positive / n_evaluated,
            "directional_consistency_fraction": float(
                np.mean(sample_directions == pooled_direction)
            ),
            "sample_mean_difference_mean": float(effects.mean()),
            "sample_mean_difference_median": float(np.median(effects)),
            "sample_mean_difference_standard_deviation": (
                float(effects.std(ddof=1)) if n_evaluated > 1 else np.nan
            ),
            "sample_mean_difference_minimum": float(effects.min()),
            "sample_mean_difference_maximum": float(effects.max()),
            "sample_effect_dominance": (
                float(absolute_effects[dominant_index] / absolute_sum)
                if absolute_sum > 0
                else np.nan
            ),
            "dominant_sample_id": (
                evaluable[dominant_index]["sample_id"] if absolute_sum > 0 else None
            ),
        }
    )
    return summary


def profile_features_by_sample(
    data: Any,
    group_by: Any,
    sample_by: Any,
    *,
    features: Optional[Sequence[Any]] = None,
    feature_names: Optional[list[str]] = None,
    targets: Any = None,
    contrasts: Optional[Sequence[Contrast]] = None,
    expression_source: Union[str, ExpressionSource] = "X",
    detection_threshold: float = 0.0,
    fold_change_pseudocount: float = 1e-9,
    minimum_cells_per_group: int = 1,
    effect_direction_tolerance: float = 0.0,
) -> ReplicateProfileResult:
    """Profile features with separate pooled and within-sample evidence."""
    _validate_profile_configuration(detection_threshold, fold_change_pseudocount)
    if sample_by is None:
        raise ValueError("`sample_by` is required for biological-replicate profiling.")
    if (
        isinstance(minimum_cells_per_group, bool)
        or not isinstance(minimum_cells_per_group, (int, np.integer))
        or minimum_cells_per_group < 1
    ):
        raise ValueError("`minimum_cells_per_group` must be an integer of at least 1.")
    if not np.isfinite(effect_direction_tolerance) or effect_direction_tolerance < 0:
        raise ValueError(
            "`effect_direction_tolerance` must be finite and non-negative."
        )

    resolved = resolve_expression_input(
        data,
        group_by,
        feature_names=feature_names,
        sample_by=sample_by,
        expression_source=expression_source,
    )
    selected_features = _resolve_features(resolved.feature_ids, features)
    resolved_contrasts = resolve_contrasts(
        resolved.group_labels, targets=targets, contrasts=contrasts
    )
    profiles = _profile_resolved(
        resolved,
        selected_features,
        resolved_contrasts,
        detection_threshold=detection_threshold,
        fold_change_pseudocount=fold_change_pseudocount,
    )
    pooled_effects = {}
    for _, row in profiles.iterrows():
        key = tuple(
            row[name] if name != "alternative_groups" else tuple(row[name])
            for name in PROFILE_KEYS
        )
        pooled_effects[key] = row["mean_difference"]

    sample_rows = []
    summary_rows = []
    for feature in selected_features:
        values = resolved.extract_feature(feature).astype(float, copy=False)
        for contrast in resolved_contrasts:
            target_mask = resolved.group_labels == contrast.target
            alternative_mask = np.isin(
                resolved.group_labels, contrast.alternative_groups
            )
            comparison_mask = target_mask | alternative_mask
            samples = _ordered_unique(resolved.sample_labels[comparison_mask])
            relationship_rows = []
            for sample in samples:
                sample_mask = resolved.sample_labels == sample
                sample_target = target_mask & sample_mask
                sample_alternative = alternative_mask & sample_mask
                target_values = values[sample_target]
                alternative_values = values[sample_alternative]
                evaluable = (
                    target_values.size >= minimum_cells_per_group
                    and alternative_values.size >= minimum_cells_per_group
                )
                row = {
                    "feature_id": feature,
                    "target_group": contrast.target,
                    "contrast_type": contrast.contrast_type,
                    "alternative_groups": contrast.alternative_groups,
                    "sample_id": sample,
                    "sample_n_target_cells": int(target_values.size),
                    "sample_n_alternative_cells": int(alternative_values.size),
                    "sample_evaluable": bool(evaluable),
                    "minimum_cells_per_group": minimum_cells_per_group,
                    "effect_direction_tolerance": effect_direction_tolerance,
                    "expression_source": resolved.source_descriptor,
                    "detection_threshold": detection_threshold,
                }
                measurement_names = [
                    "sample_n_target_detected",
                    "sample_n_alternative_detected",
                    "sample_target_detection_fraction",
                    "sample_alternative_detection_fraction",
                    "sample_target_mean_expression",
                    "sample_alternative_mean_expression",
                    "sample_target_median_expression",
                    "sample_alternative_median_expression",
                    "sample_mean_difference",
                    "sample_median_difference",
                    "sample_log2_mean_ratio",
                    "sample_target_specificity",
                ]
                row.update(dict.fromkeys(measurement_names, np.nan))
                if evaluable:
                    target_summary = summarize_expression(
                        target_values, detection_threshold
                    )
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
                    row.update(
                        {
                            "sample_n_target_detected": target_summary.n_detected,
                            "sample_n_alternative_detected": (
                                alternative_summary.n_detected
                            ),
                            "sample_target_detection_fraction": (
                                target_summary.detection_fraction
                            ),
                            "sample_alternative_detection_fraction": (
                                alternative_summary.detection_fraction
                            ),
                            "sample_target_mean_expression": (
                                target_summary.mean_expression
                            ),
                            "sample_alternative_mean_expression": (
                                alternative_summary.mean_expression
                            ),
                            "sample_target_median_expression": (
                                target_summary.median_expression
                            ),
                            "sample_alternative_median_expression": (
                                alternative_summary.median_expression
                            ),
                            "sample_mean_difference": effect["mean_difference"],
                            "sample_median_difference": effect["median_difference"],
                            "sample_log2_mean_ratio": effect["log2_mean_ratio"],
                            "sample_target_specificity": target_specificity(
                                target_summary.detection_fraction,
                                alternative_summary.detection_fraction,
                            ),
                        }
                    )
                relationship_rows.append(row)
                sample_rows.append(row)

            key = (
                feature,
                contrast.target,
                contrast.contrast_type,
                contrast.alternative_groups,
            )
            summary_rows.append(
                {
                    "feature_id": feature,
                    "target_group": contrast.target,
                    "contrast_type": contrast.contrast_type,
                    "alternative_groups": contrast.alternative_groups,
                    **_sample_summary(
                        relationship_rows,
                        pooled_mean_difference=pooled_effects[key],
                        effect_direction_tolerance=effect_direction_tolerance,
                    ),
                    "minimum_cells_per_group": minimum_cells_per_group,
                    "effect_direction_tolerance": effect_direction_tolerance,
                }
            )

    summaries = pd.DataFrame(summary_rows)
    profiles = profiles.merge(
        summaries, on=PROFILE_KEYS, how="left", validate="one_to_one"
    )
    return ReplicateProfileResult(
        profiles=profiles,
        sample_profiles=pd.DataFrame(sample_rows),
    )
