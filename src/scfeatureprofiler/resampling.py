"""Separate cell-level and biological-sample-level resampling."""

from collections.abc import Sequence
from hashlib import blake2b
from typing import Any, Optional, Union

import numpy as np
import pandas as pd

from .contrasts import Contrast, resolve_contrasts
from .inputs import resolve_expression_input
from .models import ExpressionSource, ResamplingResult
from .profiles import (
    _profile_resolved,
    _resolve_features,
    _validate_profile_configuration,
)
from .replicates import PROFILE_KEYS, _effect_direction, profile_features_by_sample
from .statistics import (
    discrimination_metrics,
    effect_metrics,
    summarize_expression,
    target_specificity,
)

CELL_RESAMPLING_METRICS = [
    "target_detection_fraction",
    "alternative_detection_fraction",
    "mean_difference",
    "median_difference",
    "log2_mean_ratio",
    "target_specificity",
    "roc_auc",
    "average_precision",
]

SAMPLE_RESAMPLING_METRICS = [
    "sample_mean_difference_mean",
    "sample_mean_difference_median",
    "effect_recurrence_fraction",
    "directional_consistency_fraction",
    "sample_effect_dominance",
]


def _validate_resampling_configuration(
    iterations: int, seed: int, confidence_level: float
) -> tuple[int, int, float]:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 1
    ):
        raise ValueError("`iterations` must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("`seed` must be a non-negative integer.")
    if not np.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("`confidence_level` must be strictly between zero and one.")
    return int(iterations), int(seed), float(confidence_level)


def _relationship_key(
    feature: Any,
    target: Any,
    contrast_type: str,
    alternative_groups: tuple[Any, ...],
) -> tuple[Any, Any, str, tuple[Any, ...]]:
    return feature, target, contrast_type, tuple(alternative_groups)


def _relationship_rng(seed: int, relationship: tuple[Any, ...]) -> np.random.Generator:
    stable_bytes = repr(relationship).encode("utf-8")
    digest = blake2b(stable_bytes, digest_size=8).digest()
    first = int.from_bytes(digest[:4], "little")
    second = int.from_bytes(digest[4:], "little")
    return np.random.default_rng(np.random.SeedSequence([seed, first, second]))


def _summarize_distributions(
    distributions: pd.DataFrame,
    *,
    metrics: Sequence[str],
    resampling_unit: str,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    alpha = (1 - confidence_level) / 2
    rows = []
    for relationship, group in distributions.groupby(PROFILE_KEYS, sort=False):
        identity = dict(zip(PROFILE_KEYS, relationship))
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            n_valid = int(finite.size)
            rows.append(
                {
                    **identity,
                    "metric": metric,
                    "resampling_unit": resampling_unit,
                    "iterations": iterations,
                    "seed": seed,
                    "confidence_level": confidence_level,
                    "n_valid_iterations": n_valid,
                    "bootstrap_mean": (float(finite.mean()) if n_valid else np.nan),
                    "bootstrap_standard_error": (
                        float(finite.std(ddof=1)) if n_valid > 1 else np.nan
                    ),
                    "confidence_interval_lower": (
                        float(np.quantile(finite, alpha)) if n_valid else np.nan
                    ),
                    "confidence_interval_upper": (
                        float(np.quantile(finite, 1 - alpha)) if n_valid else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _attach_resampling_provenance(
    profiles: pd.DataFrame,
    *,
    resampling_unit: str,
    iterations: int,
    seed: int,
    confidence_level: float,
) -> pd.DataFrame:
    profiles = profiles.copy()
    profiles["resampling_unit"] = resampling_unit
    profiles["resampling_iterations"] = iterations
    profiles["resampling_seed"] = seed
    profiles["resampling_confidence_level"] = confidence_level
    return profiles


def resample_profile_cells(
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
    iterations: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> ResamplingResult:
    """Bootstrap cells within each side of every explicit contrast."""
    _validate_profile_configuration(detection_threshold, fold_change_pseudocount)
    iterations, seed, confidence_level = _validate_resampling_configuration(
        iterations, seed, confidence_level
    )
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
    profiles = _profile_resolved(
        resolved,
        selected_features,
        resolved_contrasts,
        detection_threshold=detection_threshold,
        fold_change_pseudocount=fold_change_pseudocount,
    )

    rows = []
    for feature in selected_features:
        values = resolved.extract_feature(feature).astype(float, copy=False)
        for contrast in resolved_contrasts:
            target_values = np.sort(values[resolved.group_labels == contrast.target])
            alternative_values = np.sort(
                values[np.isin(resolved.group_labels, contrast.alternative_groups)]
            )
            relationship = _relationship_key(
                feature,
                contrast.target,
                contrast.contrast_type,
                contrast.alternative_groups,
            )
            rng = _relationship_rng(seed, relationship)
            for iteration in range(iterations):
                target_draw = target_values[
                    rng.integers(0, target_values.size, size=target_values.size)
                ]
                alternative_draw = alternative_values[
                    rng.integers(
                        0,
                        alternative_values.size,
                        size=alternative_values.size,
                    )
                ]
                target_summary = summarize_expression(target_draw, detection_threshold)
                alternative_summary = summarize_expression(
                    alternative_draw, detection_threshold
                )
                effect = effect_metrics(
                    target_summary.mean_expression,
                    alternative_summary.mean_expression,
                    target_summary.median_expression,
                    alternative_summary.median_expression,
                    fold_change_pseudocount=fold_change_pseudocount,
                )
                discrimination = discrimination_metrics(target_draw, alternative_draw)
                rows.append(
                    {
                        "feature_id": feature,
                        "target_group": contrast.target,
                        "contrast_type": contrast.contrast_type,
                        "alternative_groups": contrast.alternative_groups,
                        "iteration": iteration,
                        "resampling_unit": "cell",
                        "seed": seed,
                        "n_target_sampling_units": target_values.size,
                        "n_alternative_sampling_units": alternative_values.size,
                        "target_detection_fraction": (
                            target_summary.detection_fraction
                        ),
                        "alternative_detection_fraction": (
                            alternative_summary.detection_fraction
                        ),
                        **effect,
                        "target_specificity": target_specificity(
                            target_summary.detection_fraction,
                            alternative_summary.detection_fraction,
                        ),
                        **discrimination,
                    }
                )

    distributions = pd.DataFrame(rows)
    summaries = _summarize_distributions(
        distributions,
        metrics=CELL_RESAMPLING_METRICS,
        resampling_unit="cell",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )
    profiles = _attach_resampling_provenance(
        profiles,
        resampling_unit="cell",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )
    return ResamplingResult(
        profiles=profiles,
        summaries=summaries,
        distributions=distributions,
        resampling_unit="cell",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )


def resample_profile_samples(
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
    iterations: int = 1000,
    seed: int = 0,
    confidence_level: float = 0.95,
) -> ResamplingResult:
    """Bootstrap evaluable biological samples without resampling cells."""
    iterations, seed, confidence_level = _validate_resampling_configuration(
        iterations, seed, confidence_level
    )
    replicate_result = profile_features_by_sample(
        data,
        group_by,
        sample_by,
        features=features,
        feature_names=feature_names,
        targets=targets,
        contrasts=contrasts,
        expression_source=expression_source,
        detection_threshold=detection_threshold,
        fold_change_pseudocount=fold_change_pseudocount,
        minimum_cells_per_group=minimum_cells_per_group,
        effect_direction_tolerance=effect_direction_tolerance,
    )

    pooled_effects = {}
    for _, row in replicate_result.profiles.iterrows():
        relationship = _relationship_key(
            row["feature_id"],
            row["target_group"],
            row["contrast_type"],
            row["alternative_groups"],
        )
        pooled_effects[relationship] = row["mean_difference"]

    rows = []
    grouped = replicate_result.sample_profiles.groupby(PROFILE_KEYS, sort=False)
    for relationship_values, group in grouped:
        relationship = _relationship_key(*relationship_values)
        effects = group.loc[
            group["sample_evaluable"], "sample_mean_difference"
        ].to_numpy(dtype=float)
        if effects.size == 0:
            raise ValueError(
                "Biological-sample resampling requires at least one evaluable "
                f"sample for relationship {relationship!r}."
            )
        effects = np.sort(effects)
        pooled_direction = _effect_direction(
            pooled_effects[relationship], effect_direction_tolerance
        )
        rng = _relationship_rng(seed, relationship)
        identity = dict(zip(PROFILE_KEYS, relationship))
        for iteration in range(iterations):
            draw = effects[rng.integers(0, effects.size, size=effects.size)]
            directions = np.asarray(
                [
                    _effect_direction(effect, effect_direction_tolerance)
                    for effect in draw
                ]
            )
            absolute_effects = np.abs(draw)
            absolute_sum = absolute_effects.sum()
            rows.append(
                {
                    **identity,
                    "iteration": iteration,
                    "resampling_unit": "biological_sample",
                    "seed": seed,
                    "n_sampling_units": effects.size,
                    "sample_mean_difference_mean": float(draw.mean()),
                    "sample_mean_difference_median": float(np.median(draw)),
                    "effect_recurrence_fraction": float(np.mean(directions == 1)),
                    "directional_consistency_fraction": float(
                        np.mean(directions == pooled_direction)
                    ),
                    "sample_effect_dominance": (
                        float(absolute_effects.max() / absolute_sum)
                        if absolute_sum > 0
                        else np.nan
                    ),
                }
            )

    distributions = pd.DataFrame(rows)
    summaries = _summarize_distributions(
        distributions,
        metrics=SAMPLE_RESAMPLING_METRICS,
        resampling_unit="biological_sample",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )
    profiles = _attach_resampling_provenance(
        replicate_result.profiles,
        resampling_unit="biological_sample",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
    )
    return ResamplingResult(
        profiles=profiles,
        summaries=summaries,
        distributions=distributions,
        resampling_unit="biological_sample",
        iterations=iterations,
        seed=seed,
        confidence_level=confidence_level,
        sample_profiles=replicate_result.sample_profiles,
    )
