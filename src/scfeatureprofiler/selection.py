"""Transparent downstream selection from complete feature profiles."""

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from .models import ResamplingResult


@dataclass(frozen=True)
class SelectionCriteria:
    """Explicit independent thresholds for canonical profile measurements."""

    min_target_detection_fraction: Optional[float] = None
    max_alternative_detection_fraction: Optional[float] = None
    min_mean_difference: Optional[float] = None
    min_log2_mean_ratio: Optional[float] = None
    min_target_specificity: Optional[float] = None
    min_roc_auc: Optional[float] = None
    min_average_precision: Optional[float] = None
    max_cell_level_adjusted_p_value: Optional[float] = None
    min_biological_samples_evaluated: Optional[int] = None
    min_effect_recurrence_fraction: Optional[float] = None
    min_directional_consistency_fraction: Optional[float] = None
    max_sample_effect_dominance: Optional[float] = None

    def __post_init__(self) -> None:
        active = self.active_rules()
        if not active:
            raise ValueError("At least one explicit selection criterion is required.")

        fraction_fields = {
            "min_target_detection_fraction",
            "max_alternative_detection_fraction",
            "min_target_specificity",
            "min_roc_auc",
            "min_average_precision",
            "max_cell_level_adjusted_p_value",
            "min_effect_recurrence_fraction",
            "min_directional_consistency_fraction",
            "max_sample_effect_dominance",
        }
        for field_name in fraction_fields:
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise ValueError(
                        f"`{field_name}` must be numeric and between zero and one."
                    )
                if not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"`{field_name}` must be between zero and one.")

        for field_name in {"min_mean_difference", "min_log2_mean_ratio"}:
            value = getattr(self, field_name)
            if value is not None:
                if isinstance(value, (bool, np.bool_)) or not isinstance(
                    value, (int, float, np.integer, np.floating)
                ):
                    raise ValueError(f"`{field_name}` must be finite and numeric.")
                if not np.isfinite(value):
                    raise ValueError(f"`{field_name}` must be finite.")

        sample_count = self.min_biological_samples_evaluated
        if sample_count is not None and (
            isinstance(sample_count, bool)
            or not isinstance(sample_count, (int, np.integer))
            or sample_count < 1
        ):
            raise ValueError(
                "`min_biological_samples_evaluated` must be an integer of at least 1."
            )

    def active_rules(self) -> list[tuple[str, str, str, Any]]:
        specifications = [
            (
                "min_target_detection_fraction",
                "target_detection_fraction",
                "ge",
            ),
            (
                "max_alternative_detection_fraction",
                "alternative_detection_fraction",
                "le",
            ),
            ("min_mean_difference", "mean_difference", "ge"),
            ("min_log2_mean_ratio", "log2_mean_ratio", "ge"),
            ("min_target_specificity", "target_specificity", "ge"),
            ("min_roc_auc", "roc_auc", "ge"),
            ("min_average_precision", "average_precision", "ge"),
            (
                "max_cell_level_adjusted_p_value",
                "cell_level_adjusted_p_value",
                "le",
            ),
            (
                "min_biological_samples_evaluated",
                "n_biological_samples_evaluated",
                "ge",
            ),
            (
                "min_effect_recurrence_fraction",
                "effect_recurrence_fraction",
                "ge",
            ),
            (
                "min_directional_consistency_fraction",
                "directional_consistency_fraction",
                "ge",
            ),
            (
                "max_sample_effect_dominance",
                "sample_effect_dominance",
                "le",
            ),
        ]
        return [
            (field_name, column, operator, getattr(self, field_name))
            for field_name, column, operator in specifications
            if getattr(self, field_name) is not None
        ]


@dataclass(frozen=True)
class RankingCriterion:
    """One explicit lexicographic ranking column and direction."""

    column: str
    ascending: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("A ranking column must be a non-empty string.")
        if not isinstance(self.ascending, bool):
            raise TypeError("`ascending` must be boolean.")


@dataclass(frozen=True)
class FeatureSelectionResult:
    """Complete rule evaluation and the resulting selected profile rows."""

    evaluated_profiles: pd.DataFrame
    selected_profiles: pd.DataFrame
    criteria: SelectionCriteria
    ranking: tuple[RankingCriterion, ...]
    top_n_per_relationship: Optional[int]


IDENTITY_COLUMNS = [
    "feature_id",
    "target_group",
    "contrast_type",
    "alternative_groups",
]
RELATIONSHIP_COLUMNS = [
    "target_group",
    "contrast_type",
    "alternative_groups",
]


def _validate_profile_identity(
    frame: pd.DataFrame, *, require_unique: bool = False
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Profile data must be a pandas DataFrame.")
    missing = [column for column in IDENTITY_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Profile table is missing identity columns: {missing}")
    if frame[IDENTITY_COLUMNS].isna().any(axis=None):
        raise ValueError("Profile identity columns must not contain missing values.")
    if require_unique and frame.duplicated(IDENTITY_COLUMNS).any():
        raise ValueError(
            "Profile table contains duplicate feature-target-contrast relationships."
        )


def _validate_resampling_iterations(frame: pd.DataFrame, iterations: int) -> None:
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 1
    ):
        raise ValueError("`result.iterations` must be a positive integer.")
    if "iteration" not in frame.columns:
        raise ValueError("Resampling distributions must contain an 'iteration' column.")
    if frame.empty:
        raise ValueError("Resampling distributions must contain at least one draw.")
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
        for value in frame["iteration"]
    ):
        raise ValueError("Resampling iteration identifiers must be integers.")
    expected = set(range(int(iterations)))
    for identity, group in frame.groupby(IDENTITY_COLUMNS, sort=False):
        observed = set(group["iteration"].tolist())
        if len(group) != iterations or observed != expected:
            raise ValueError(
                "Every feature-target-contrast relationship must contain exactly "
                f"iterations 0 through {iterations - 1}; invalid relationship "
                f"{identity!r}."
            )


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise ValueError(f"Required measurement column {column!r} is missing.")
    try:
        return pd.to_numeric(frame[column], errors="raise")
    except (TypeError, ValueError) as error:
        raise TypeError(f"Measurement column {column!r} must be numeric.") from error


def _evaluate_rules(frame: pd.DataFrame, criteria: SelectionCriteria) -> pd.DataFrame:
    evaluated = frame.copy()
    pass_columns = []
    for _, column, operator, threshold in criteria.active_rules():
        values = _numeric_column(evaluated, column)
        passed = values.notna()
        if operator == "ge":
            passed &= values >= threshold
        else:
            passed &= values <= threshold
        pass_column = f"passes_{column}"
        evaluated[pass_column] = passed.astype(bool)
        pass_columns.append(pass_column)
    evaluated["passes_all_criteria"] = evaluated[pass_columns].all(axis=1)
    return evaluated


def _validate_ranking(
    frame: pd.DataFrame,
    ranking: Optional[list[RankingCriterion]],
    top_n_per_relationship: Optional[int],
) -> tuple[RankingCriterion, ...]:
    ranking_tuple = tuple(ranking or ())
    if not all(isinstance(item, RankingCriterion) for item in ranking_tuple):
        raise TypeError("Every ranking item must be a RankingCriterion.")
    columns = [item.column for item in ranking_tuple]
    if len(columns) != len(set(columns)):
        raise ValueError("Ranking columns must be unique.")
    for column in columns:
        _numeric_column(frame, column)
    if top_n_per_relationship is not None:
        if not ranking_tuple:
            raise ValueError("Top-N selection requires explicit ranking criteria.")
        if (
            isinstance(top_n_per_relationship, bool)
            or not isinstance(top_n_per_relationship, (int, np.integer))
            or top_n_per_relationship < 1
        ):
            raise ValueError(
                "`top_n_per_relationship` must be an integer of at least 1."
            )
    return ranking_tuple


def _rank_eligible_rows(
    evaluated: pd.DataFrame,
    ranking: tuple[RankingCriterion, ...],
) -> pd.Series:
    ranks = pd.Series(pd.NA, index=evaluated.index, dtype="Int64")
    if not ranking:
        return ranks
    eligible = evaluated[evaluated["passes_all_criteria"]]
    for _, group in eligible.groupby(RELATIONSHIP_COLUMNS, sort=False):
        for item in ranking:
            if group[item.column].isna().any():
                raise ValueError(
                    f"Ranking column {item.column!r} contains missing values among "
                    "rows that pass all criteria."
                )
        sortable = group.assign(
            _feature_identifier_tie_key=group["feature_id"].map(
                lambda value: (type(value).__name__, repr(value))
            )
        )
        columns = [item.column for item in ranking] + ["_feature_identifier_tie_key"]
        ascending = [item.ascending for item in ranking] + [True]
        ordered = sortable.sort_values(columns, ascending=ascending, kind="stable")
        ranks.loc[ordered.index] = np.arange(1, len(ordered) + 1)
    return ranks


def select_profile_features(
    profiles: pd.DataFrame,
    criteria: SelectionCriteria,
    *,
    ranking: Optional[list[RankingCriterion]] = None,
    top_n_per_relationship: Optional[int] = None,
) -> FeatureSelectionResult:
    """Apply explicit independent rules to a complete canonical profile table."""
    if not isinstance(criteria, SelectionCriteria):
        raise TypeError("`criteria` must be a SelectionCriteria object.")
    _validate_profile_identity(profiles, require_unique=True)
    evaluated = _evaluate_rules(profiles, criteria)
    ranking_tuple = _validate_ranking(evaluated, ranking, top_n_per_relationship)
    evaluated["selection_rank"] = _rank_eligible_rows(evaluated, ranking_tuple)
    evaluated["selected"] = evaluated["passes_all_criteria"]
    if top_n_per_relationship is not None:
        evaluated["selected"] &= evaluated["selection_rank"].notna() & (
            evaluated["selection_rank"] <= top_n_per_relationship
        )

    selected = evaluated[evaluated["selected"]].copy()
    if ranking_tuple:
        selected = selected.sort_values(
            RELATIONSHIP_COLUMNS + ["selection_rank"], kind="stable"
        )
    return FeatureSelectionResult(
        evaluated_profiles=evaluated,
        selected_profiles=selected.reset_index(drop=True),
        criteria=criteria,
        ranking=ranking_tuple,
        top_n_per_relationship=top_n_per_relationship,
    )


def summarize_resampled_selection(
    result: ResamplingResult,
    criteria: SelectionCriteria,
    *,
    ranking: Optional[list[RankingCriterion]] = None,
    top_n_per_relationship: Optional[int] = None,
) -> pd.DataFrame:
    """Summarize explicit rule selection and rank variability across draws."""
    if not isinstance(result, ResamplingResult):
        raise TypeError("`result` must be a ResamplingResult.")
    _validate_profile_identity(result.distributions)
    _validate_resampling_iterations(result.distributions, result.iterations)
    evaluated = _evaluate_rules(result.distributions, criteria)
    ranking_tuple = _validate_ranking(evaluated, ranking, top_n_per_relationship)
    evaluated["selection_rank"] = pd.Series(pd.NA, index=evaluated.index, dtype="Int64")

    if ranking_tuple:
        iteration_groups = RELATIONSHIP_COLUMNS + ["iteration"]
        for _, group in evaluated.groupby(iteration_groups, sort=False):
            eligible = group[group["passes_all_criteria"]]
            if eligible.empty:
                continue
            for item in ranking_tuple:
                if eligible[item.column].isna().any():
                    raise ValueError(
                        f"Ranking column {item.column!r} contains missing values "
                        "among resampled rows that pass all criteria."
                    )
            sortable = eligible.assign(
                _feature_identifier_tie_key=eligible["feature_id"].map(
                    lambda value: (type(value).__name__, repr(value))
                )
            )
            columns = [item.column for item in ranking_tuple] + [
                "_feature_identifier_tie_key"
            ]
            ascending = [item.ascending for item in ranking_tuple] + [True]
            ordered = sortable.sort_values(columns, ascending=ascending, kind="stable")
            evaluated.loc[ordered.index, "selection_rank"] = np.arange(
                1, len(ordered) + 1
            )

    evaluated["selected"] = evaluated["passes_all_criteria"]
    if top_n_per_relationship is not None:
        evaluated["selected"] &= evaluated["selection_rank"].notna() & (
            evaluated["selection_rank"] <= top_n_per_relationship
        )

    rows = []
    for identity, group in evaluated.groupby(IDENTITY_COLUMNS, sort=False):
        ranks = pd.to_numeric(group["selection_rank"], errors="coerce").dropna()
        row = {
            **dict(zip(IDENTITY_COLUMNS, identity)),
            "resampling_unit": result.resampling_unit,
            "iterations": result.iterations,
            "seed": result.seed,
            "selection_frequency": float(group["selected"].mean()),
            "n_selected_iterations": int(group["selected"].sum()),
            "n_ranked_iterations": int(len(ranks)),
            "mean_rank": float(ranks.mean()) if len(ranks) else np.nan,
            "rank_standard_deviation": (
                float(ranks.std(ddof=1)) if len(ranks) > 1 else np.nan
            ),
            "minimum_rank": float(ranks.min()) if len(ranks) else np.nan,
            "maximum_rank": float(ranks.max()) if len(ranks) else np.nan,
        }
        rows.append(row)
    return pd.DataFrame(rows)
