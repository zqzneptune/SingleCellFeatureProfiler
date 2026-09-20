"""Annotation evidence assessment for user-supplied population labels."""

from collections.abc import Iterable
from typing import Any, Optional

import numpy as np
import pandas as pd

from .models import AnnotationEvidenceResult

IDENTITY_COLUMNS = [
    "feature_id",
    "target_group",
    "contrast_type",
    "alternative_groups",
]
REQUIRED_MEASUREMENTS = [
    "target_detection_fraction",
    "alternative_detection_fraction",
    "mean_difference",
]


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _stable_value_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _identity_key(row: pd.Series) -> tuple[Any, ...]:
    return tuple(row[column] for column in IDENTITY_COLUMNS)


def _stable_identity_key(row: pd.Series) -> tuple[tuple[str, str], ...]:
    return tuple(_stable_value_key(row[column]) for column in IDENTITY_COLUMNS)


def _validate_profiles(profiles: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(profiles, pd.DataFrame):
        raise TypeError("`profiles` must be a pandas DataFrame.")
    required = IDENTITY_COLUMNS + REQUIRED_MEASUREMENTS
    missing = [column for column in required if column not in profiles.columns]
    if missing:
        raise ValueError(f"Profile table is missing required columns: {missing}")
    if profiles[IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError("Profile identity columns must not contain missing values.")
    try:
        duplicate = profiles.duplicated(IDENTITY_COLUMNS)
    except TypeError as error:
        raise TypeError(
            "Profile identity values must be hashable; alternative_groups "
            "should be tuples."
        ) from error
    if duplicate.any():
        raise ValueError(
            "Profile table contains duplicate feature-target-contrast relationships."
        )
    for _, row in profiles[IDENTITY_COLUMNS].iterrows():
        try:
            hash(_identity_key(row))
        except TypeError as error:
            raise TypeError(
                "Profile identity values must be hashable; alternative_groups "
                "should be tuples."
            ) from error
        alternatives = row["alternative_groups"]
        if not isinstance(alternatives, tuple) or not alternatives:
            raise ValueError("`alternative_groups` must contain non-empty tuples.")
        contrast_type = row["contrast_type"]
        if contrast_type not in {"target_vs_all", "target_vs_one", "target_vs_set"}:
            raise ValueError(f"Unsupported contrast type {contrast_type!r}.")
        if contrast_type == "target_vs_one" and len(alternatives) != 1:
            raise ValueError(
                "A target_vs_one relationship must contain exactly one alternative."
            )

    result = profiles.copy()
    for column in REQUIRED_MEASUREMENTS:
        try:
            numeric = pd.to_numeric(result[column], errors="raise")
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"Measurement column {column!r} must be numeric."
            ) from error
        observed = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(observed).all():
            raise ValueError(
                f"Measurement column {column!r} must be finite or missing."
            )
        result[column] = numeric
    for column in [
        "target_detection_fraction",
        "alternative_detection_fraction",
    ]:
        observed = result[column].dropna()
        if not observed.between(0, 1).all():
            raise ValueError(f"Measurement column {column!r} must be between 0 and 1.")
    return result


def _validate_proposed_label(proposed_label: Any) -> None:
    if _is_missing(proposed_label):
        raise ValueError("`proposed_label` must not be missing.")
    try:
        hash(proposed_label)
    except TypeError as error:
        raise TypeError("`proposed_label` must be a hashable scalar value.") from error


def _normalize_features(
    features: Optional[Iterable[Any]], argument: str
) -> tuple[Any, ...]:
    if features is None:
        return ()
    if isinstance(features, (str, bytes)):
        values = (features,)
    else:
        try:
            values = tuple(features)
        except TypeError as error:
            raise TypeError(
                f"`{argument}` must be an iterable of feature identifiers."
            ) from error
    for value in values:
        if _is_missing(value):
            raise ValueError(f"`{argument}` must not contain missing identifiers.")
        try:
            hash(value)
        except TypeError as error:
            raise TypeError(
                f"`{argument}` must contain hashable feature identifiers."
            ) from error
    if len(values) != len(set(values)):
        raise ValueError(f"`{argument}` contains duplicate feature identifiers.")
    return tuple(sorted(values, key=_stable_value_key))


def _sort_by_identity(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    ordered = sorted(
        frame.index, key=lambda index: _stable_identity_key(frame.loc[index])
    )
    return frame.loc[ordered].reset_index(drop=True)


def _sort_supporting(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)

    def key(index: Any) -> tuple[Any, ...]:
        row = frame.loc[index]
        specificity = row.get("target_specificity", np.nan)
        specificity_key = (
            -float(specificity) if not _is_missing(specificity) else np.inf
        )
        return (
            -float(row["mean_difference"]),
            specificity_key,
            _stable_identity_key(row),
        )

    return frame.loc[sorted(frame.index, key=key)].reset_index(drop=True)


def _sort_shared(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    ordered = sorted(
        frame.index,
        key=lambda index: (
            -float(frame.loc[index, "alternative_detection_fraction"]),
            -float(frame.loc[index, "target_detection_fraction"]),
            _stable_identity_key(frame.loc[index]),
        ),
    )
    return frame.loc[ordered].reset_index(drop=True)


def _sort_competitors(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.reset_index(drop=True)
    ordered = sorted(
        frame.index,
        key=lambda index: (
            float(frame.loc[index, "mean_difference"]),
            -float(frame.loc[index, "alternative_detection_fraction"]),
            _stable_identity_key(frame.loc[index]),
        ),
    )
    return frame.loc[ordered].reset_index(drop=True)


def _expectation_assessment(row: pd.Series) -> tuple[str, Optional[str]]:
    role = row["expected_role"]
    if role == "unspecified":
        return "not_evaluated", None
    target_detected = row["target_detected"]
    direction = row["effect_direction"]
    if _is_missing(target_detected) or direction == "missing":
        return "unresolved", "required_measurement_missing"
    if role == "expected_support":
        if not bool(target_detected):
            return "contradictory", "expected_support_not_detected"
        if direction != "target_favoring":
            return "contradictory", "expected_support_not_target_favoring"
        return "consistent", None
    if bool(target_detected):
        return "contradictory", "expected_absent_but_target_detected"
    return "consistent", None


def evaluate_annotation_evidence(
    profiles: pd.DataFrame,
    proposed_label: Any,
    *,
    expected_support_features: Optional[Iterable[Any]] = None,
    expected_absent_features: Optional[Iterable[Any]] = None,
    direction_tolerance: float = 0.0,
) -> AnnotationEvidenceResult:
    """Evaluate molecular evidence for one caller-supplied population label."""
    _validate_proposed_label(proposed_label)
    if (
        isinstance(direction_tolerance, (bool, np.bool_))
        or not isinstance(direction_tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(direction_tolerance)
        or direction_tolerance < 0
    ):
        raise ValueError("`direction_tolerance` must be finite and non-negative.")
    tolerance = float(direction_tolerance)
    expected_support = _normalize_features(
        expected_support_features, "expected_support_features"
    )
    expected_absent = _normalize_features(
        expected_absent_features, "expected_absent_features"
    )
    overlap = set(expected_support).intersection(expected_absent)
    if overlap:
        raise ValueError(
            "Expected-support and expected-absence feature sets overlap: "
            f"{sorted(overlap, key=_stable_value_key)!r}"
        )

    validated = _validate_profiles(profiles)
    label_mask = validated["target_group"].map(lambda value: value == proposed_label)
    evaluated = validated[label_mask].copy()
    if evaluated.empty:
        raise ValueError(
            f"Proposed label {proposed_label!r} is not present as a target."
        )
    evaluated = _sort_by_identity(evaluated)

    effects = evaluated["mean_difference"]
    evaluated["effect_direction"] = np.select(
        [effects > tolerance, effects < -tolerance, effects.notna()],
        ["target_favoring", "alternative_favoring", "neutral"],
        default="missing",
    )
    evaluated["target_detected"] = (
        evaluated["target_detection_fraction"]
        .gt(0)
        .where(evaluated["target_detection_fraction"].notna(), pd.NA)
        .astype("boolean")
    )
    evaluated["alternative_detected"] = (
        evaluated["alternative_detection_fraction"]
        .gt(0)
        .where(evaluated["alternative_detection_fraction"].notna(), pd.NA)
        .astype("boolean")
    )
    evaluated["supports_proposed_label"] = (
        evaluated["target_detected"]
        & evaluated["effect_direction"].eq("target_favoring")
    ).astype("boolean")
    evaluated["shared_or_broad"] = (
        evaluated["target_detected"] & evaluated["alternative_detected"]
    ).astype("boolean")

    competitor_groups = []
    competitor_flags = []
    for _, row in evaluated.iterrows():
        alternatives = row["alternative_groups"]
        explicit_one = (
            row["contrast_type"] == "target_vs_one"
            and isinstance(alternatives, tuple)
            and len(alternatives) == 1
        )
        is_competitor = (
            explicit_one and row["effect_direction"] == "alternative_favoring"
        )
        competitor_groups.append(alternatives[0] if is_competitor else None)
        competitor_flags.append(is_competitor)
    evaluated["competitor_group"] = competitor_groups
    evaluated["is_competitor_evidence"] = competitor_flags

    support_set = set(expected_support)
    absent_set = set(expected_absent)
    evaluated["expected_role"] = evaluated["feature_id"].map(
        lambda feature: (
            "expected_support"
            if feature in support_set
            else "expected_absent"
            if feature in absent_set
            else "unspecified"
        )
    )
    assessments = evaluated.apply(_expectation_assessment, axis=1)
    evaluated["expectation_status"] = [item[0] for item in assessments]
    evaluated["expectation_reason"] = [item[1] for item in assessments]

    supporting = _sort_supporting(
        evaluated[evaluated["supports_proposed_label"].fillna(False)].copy()
    )
    shared = _sort_shared(evaluated[evaluated["shared_or_broad"].fillna(False)].copy())
    competitors = _sort_competitors(
        evaluated[evaluated["is_competitor_evidence"]].copy()
    )

    expected_mask = evaluated["expected_role"].ne("unspecified")
    expectation_records = [
        row.to_dict() for _, row in evaluated[expected_mask].iterrows()
    ]
    profiled_features = set(evaluated["feature_id"])
    for feature, role in [
        *((feature, "expected_support") for feature in expected_support),
        *((feature, "expected_absent") for feature in expected_absent),
    ]:
        if feature in profiled_features:
            continue
        row = dict.fromkeys(evaluated.columns, np.nan)
        row.update(
            {
                "feature_id": feature,
                "target_group": proposed_label,
                "contrast_type": None,
                "alternative_groups": None,
                "expected_role": role,
                "expectation_status": "unresolved",
                "expectation_reason": "expected_feature_not_profiled",
            }
        )
        expectation_records.append(row)
    expectation_evidence = pd.DataFrame.from_records(
        expectation_records, columns=evaluated.columns
    )
    expectation_evidence = _sort_by_identity(expectation_evidence)
    contradictory = expectation_evidence[
        expectation_evidence["expectation_status"].eq("contradictory")
    ].reset_index(drop=True)
    unresolved = expectation_evidence[
        expectation_evidence["expectation_status"].eq("unresolved")
    ].reset_index(drop=True)

    return AnnotationEvidenceResult(
        evaluated_profiles=evaluated.reset_index(drop=True),
        supporting_evidence=supporting,
        shared_evidence=shared,
        competitor_evidence=competitors,
        expectation_evidence=expectation_evidence,
        contradictory_evidence=contradictory,
        unresolved_expectations=unresolved,
        proposed_label=proposed_label,
        direction_tolerance=tolerance,
        expected_support_features=expected_support,
        expected_absent_features=expected_absent,
    )
