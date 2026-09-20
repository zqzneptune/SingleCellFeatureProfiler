"""Comparison of completed profiles from independent datasets."""

from collections.abc import Sequence
from typing import Any, Optional

import numpy as np
import pandas as pd

from .models import ProfileComparisonResult

IDENTITY_COLUMNS = [
    "feature_id",
    "target_group",
    "contrast_type",
    "alternative_groups",
]

DEFAULT_MEASUREMENTS = [
    "target_detection_fraction",
    "alternative_detection_fraction",
    "mean_difference",
    "median_difference",
    "log2_mean_ratio",
    "target_specificity",
    "roc_auc",
    "average_precision",
    "n_biological_samples_available",
    "n_biological_samples_evaluated",
    "n_biological_samples_insufficient_cells",
    "n_samples_positive_effect",
    "n_samples_negative_effect",
    "n_samples_zero_effect",
    "effect_recurrence_fraction",
    "directional_consistency_fraction",
    "sample_mean_difference_mean",
    "sample_mean_difference_median",
    "sample_mean_difference_standard_deviation",
    "sample_mean_difference_minimum",
    "sample_mean_difference_maximum",
    "sample_effect_dominance",
]

DEFAULT_CONFIGURATION_FIELDS = [
    "expression_source",
    "detection_threshold",
    "fold_change_pseudocount",
    "minimum_cells_per_group",
    "effect_direction_tolerance",
]

RELATIONSHIP_COLUMNS = IDENTITY_COLUMNS + [
    "reference_present",
    "comparison_present",
    "relationship_status",
]

MEASUREMENT_COLUMNS = IDENTITY_COLUMNS + [
    "measurement",
    "reference_value",
    "comparison_value",
    "reference_value_status",
    "comparison_value_status",
    "difference_available",
    "difference",
    "absolute_difference",
]

CONFIGURATION_COLUMNS = IDENTITY_COLUMNS + [
    "configuration_field",
    "reference_value",
    "comparison_value",
    "reference_value_status",
    "comparison_value_status",
    "configuration_status",
]


def _validate_dataset_name(value: str, argument: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{argument}` must be a non-empty string.")
    return value


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _identity_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def _stable_identity_key(identity: tuple[Any, ...]) -> tuple[tuple[str, str], ...]:
    return tuple((type(value).__name__, repr(value)) for value in identity)


def _validate_profile(frame: pd.DataFrame, argument: str) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError(f"`{argument}` must be a pandas DataFrame.")
    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"`{argument}` is missing identity columns: {missing}")
    if frame[IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError(
            f"`{argument}` identity columns must not contain missing values."
        )
    try:
        duplicate = frame.duplicated(IDENTITY_COLUMNS)
    except TypeError as error:
        raise TypeError(
            f"`{argument}` identity values must be hashable; "
            "alternative_groups should be tuples."
        ) from error
    if duplicate.any():
        raise ValueError(
            f"`{argument}` contains duplicate feature-target-contrast relationships."
        )
    for _, row in frame[IDENTITY_COLUMNS].iterrows():
        try:
            hash(_identity_key(row))
        except TypeError as error:
            raise TypeError(
                f"`{argument}` identity values must be hashable; "
                "alternative_groups should be tuples."
            ) from error
    return frame.copy()


def _resolve_measurements(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    measurements: Optional[Sequence[str]],
) -> tuple[str, ...]:
    if measurements is None:
        resolved = tuple(
            column
            for column in DEFAULT_MEASUREMENTS
            if column in reference.columns or column in comparison.columns
        )
        if not resolved:
            raise ValueError(
                "Neither profile contains a supported default comparison measurement."
            )
        return resolved
    if isinstance(measurements, (str, bytes)) or not isinstance(measurements, Sequence):
        raise TypeError("`measurements` must be a sequence of column names.")
    resolved = tuple(measurements)
    if not resolved:
        raise ValueError("`measurements` must contain at least one column name.")
    if any(not isinstance(column, str) or not column for column in resolved):
        raise ValueError("Every requested measurement must be a non-empty string.")
    if len(resolved) != len(set(resolved)):
        raise ValueError("Requested measurements must be unique.")
    identity_measurements = set(resolved).intersection(IDENTITY_COLUMNS)
    if identity_measurements:
        raise ValueError(
            "Identity columns cannot be requested as measurements: "
            f"{sorted(identity_measurements)}"
        )
    absent = [
        column
        for column in resolved
        if column not in reference.columns and column not in comparison.columns
    ]
    if absent:
        raise ValueError(
            f"Requested measurements are absent from both profiles: {absent}"
        )
    return resolved


def _coerce_measurements(
    frame: pd.DataFrame, measurements: tuple[str, ...], argument: str
) -> pd.DataFrame:
    for column in measurements:
        if column not in frame.columns:
            continue
        try:
            numeric = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Measurement column {column!r} in `{argument}` must be numeric."
            ) from error
        observed = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(
                f"Measurement column {column!r} in `{argument}` must be "
                "finite or missing."
            )
        frame[column] = numeric
    return frame


def _profile_map(frame: pd.DataFrame) -> dict[tuple[Any, ...], pd.Series]:
    return {_identity_key(row): row for _, row in frame.iterrows()}


def _side_value(
    profile: Optional[pd.Series], field: str, *, unavailable_status: str
) -> tuple[Any, str]:
    if profile is None:
        return np.nan, "relationship_missing"
    if field not in profile.index:
        return np.nan, unavailable_status
    value = profile[field]
    if _is_missing(value):
        return np.nan, "value_missing"
    return value, "observed"


def _values_equal(reference: Any, comparison: Any) -> bool:
    try:
        equal = reference == comparison
    except (TypeError, ValueError):
        return False
    return bool(equal) if isinstance(equal, (bool, np.bool_)) else False


def compare_feature_profiles(
    reference_profiles: pd.DataFrame,
    comparison_profiles: pd.DataFrame,
    *,
    reference_name: str = "reference",
    comparison_name: str = "comparison",
    measurements: Optional[Sequence[str]] = None,
) -> ProfileComparisonResult:
    """Compare completed independent profile tables without joining matrices."""
    reference_name = _validate_dataset_name(reference_name, "reference_name")
    comparison_name = _validate_dataset_name(comparison_name, "comparison_name")
    if reference_name == comparison_name:
        raise ValueError("Reference and comparison dataset names must be distinct.")

    reference = _validate_profile(reference_profiles, "reference_profiles")
    comparison = _validate_profile(comparison_profiles, "comparison_profiles")
    compared_measurements = _resolve_measurements(reference, comparison, measurements)
    reference = _coerce_measurements(
        reference, compared_measurements, "reference_profiles"
    )
    comparison = _coerce_measurements(
        comparison, compared_measurements, "comparison_profiles"
    )

    reference_map = _profile_map(reference)
    comparison_map = _profile_map(comparison)
    identities = sorted(
        set(reference_map).union(comparison_map), key=_stable_identity_key
    )

    relationship_rows = []
    measurement_rows = []
    for identity in identities:
        reference_profile = reference_map.get(identity)
        comparison_profile = comparison_map.get(identity)
        reference_present = reference_profile is not None
        comparison_present = comparison_profile is not None
        if reference_present and comparison_present:
            relationship_status = "both"
        elif reference_present:
            relationship_status = "reference_only"
        else:
            relationship_status = "comparison_only"
        identity_values = dict(zip(IDENTITY_COLUMNS, identity))
        relationship_rows.append(
            {
                **identity_values,
                "reference_present": reference_present,
                "comparison_present": comparison_present,
                "relationship_status": relationship_status,
            }
        )

        for measurement in compared_measurements:
            reference_value, reference_status = _side_value(
                reference_profile,
                measurement,
                unavailable_status="measurement_unavailable",
            )
            comparison_value, comparison_status = _side_value(
                comparison_profile,
                measurement,
                unavailable_status="measurement_unavailable",
            )
            difference_available = (
                reference_status == "observed" and comparison_status == "observed"
            )
            difference = (
                float(comparison_value) - float(reference_value)
                if difference_available
                else np.nan
            )
            measurement_rows.append(
                {
                    **identity_values,
                    "measurement": measurement,
                    "reference_value": (
                        float(reference_value)
                        if reference_status == "observed"
                        else np.nan
                    ),
                    "comparison_value": (
                        float(comparison_value)
                        if comparison_status == "observed"
                        else np.nan
                    ),
                    "reference_value_status": reference_status,
                    "comparison_value_status": comparison_status,
                    "difference_available": difference_available,
                    "difference": difference,
                    "absolute_difference": abs(difference),
                }
            )

    compared_configuration_fields = tuple(
        field
        for field in DEFAULT_CONFIGURATION_FIELDS
        if field in reference.columns or field in comparison.columns
    )
    configuration_rows = []
    for identity in identities:
        identity_values = dict(zip(IDENTITY_COLUMNS, identity))
        reference_profile = reference_map.get(identity)
        comparison_profile = comparison_map.get(identity)
        for field in compared_configuration_fields:
            reference_value, reference_status = _side_value(
                reference_profile,
                field,
                unavailable_status="field_unavailable",
            )
            comparison_value, comparison_status = _side_value(
                comparison_profile,
                field,
                unavailable_status="field_unavailable",
            )
            if reference_status == "observed" and comparison_status == "observed":
                configuration_status = (
                    "match"
                    if _values_equal(reference_value, comparison_value)
                    else "mismatch"
                )
            else:
                configuration_status = "incomplete"
            configuration_rows.append(
                {
                    **identity_values,
                    "configuration_field": field,
                    "reference_value": reference_value,
                    "comparison_value": comparison_value,
                    "reference_value_status": reference_status,
                    "comparison_value_status": comparison_status,
                    "configuration_status": configuration_status,
                }
            )

    return ProfileComparisonResult(
        relationships=pd.DataFrame(relationship_rows, columns=RELATIONSHIP_COLUMNS),
        measurements=pd.DataFrame(measurement_rows, columns=MEASUREMENT_COLUMNS),
        configurations=pd.DataFrame(configuration_rows, columns=CONFIGURATION_COLUMNS),
        reference_name=reference_name,
        comparison_name=comparison_name,
        compared_measurements=compared_measurements,
        compared_configuration_fields=compared_configuration_fields,
    )
