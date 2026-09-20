"""Deterministic Matplotlib visualizations for annotation evidence."""

from collections.abc import Iterable
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from .models import AnnotationEvidenceResult

IDENTITY_COLUMNS = [
    "feature_id",
    "target_group",
    "contrast_type",
    "alternative_groups",
]

DISPLAY_PRIORITY = {
    "contradictory": 0,
    "unresolved": 1,
    "competitor": 2,
    "supporting": 3,
    "shared": 4,
    "other": 5,
}

REQUIRED_DISPLAY_COLUMNS = IDENTITY_COLUMNS + [
    "target_detection_fraction",
    "alternative_detection_fraction",
    "mean_difference",
    "effect_direction",
    "supports_proposed_label",
    "shared_or_broad",
    "is_competitor_evidence",
    "expectation_status",
]

TARGET_COLOR = "#0072B2"
ALTERNATIVE_COLOR = "#D55E00"
NEUTRAL_COLOR = "#7F7F7F"


def _is_missing(value: Any) -> bool:
    missing = pd.isna(value)
    return bool(missing) if isinstance(missing, (bool, np.bool_)) else False


def _stable_value_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _stable_identity_key(row: pd.Series) -> tuple[tuple[str, str], ...]:
    return tuple(_stable_value_key(row[column]) for column in IDENTITY_COLUMNS)


def _normalize_features(features: Optional[Iterable[Any]]) -> Optional[tuple[Any, ...]]:
    if features is None:
        return None
    if isinstance(features, (str, bytes)):
        values = (features,)
    else:
        try:
            values = tuple(features)
        except TypeError as error:
            raise TypeError("`features` must be an iterable of identifiers.") from error
    if not values:
        raise ValueError("`features` must contain at least one identifier.")
    for value in values:
        if _is_missing(value):
            raise ValueError("`features` must not contain missing identifiers.")
        try:
            hash(value)
        except TypeError as error:
            raise TypeError("`features` must contain hashable identifiers.") from error
    if len(values) != len(set(values)):
        raise ValueError("`features` contains duplicate feature identifiers.")
    return tuple(sorted(values, key=_stable_value_key))


def _display_priority(row: pd.Series) -> str:
    if row["expectation_status"] == "contradictory":
        return "contradictory"
    if row["expectation_status"] == "unresolved":
        return "unresolved"
    if bool(row["is_competitor_evidence"]):
        return "competitor"
    if not _is_missing(row["supports_proposed_label"]) and bool(
        row["supports_proposed_label"]
    ):
        return "supporting"
    if not _is_missing(row["shared_or_broad"]) and bool(row["shared_or_broad"]):
        return "shared"
    return "other"


def _display_label(row: pd.Series) -> str:
    feature = str(row["feature_id"])
    alternatives = row["alternative_groups"]
    alternative_text = ", ".join(str(value) for value in alternatives)
    if row["contrast_type"] == "target_vs_one":
        return f"{feature} | vs {alternative_text}"
    if row["contrast_type"] == "target_vs_set":
        return f"{feature} | vs {{{alternative_text}}}"
    return f"{feature} | vs all ({alternative_text})"


def _contrast_label(row: pd.Series) -> str:
    alternatives = ", ".join(str(value) for value in row["alternative_groups"])
    return f"{row['contrast_type']}: {alternatives}"


def select_annotation_evidence_rows(
    evidence: AnnotationEvidenceResult,
    *,
    features: Optional[Iterable[Any]] = None,
    max_rows: int = 12,
) -> pd.DataFrame:
    """Select inspectable display rows without creating an evidence score."""
    if not isinstance(evidence, AnnotationEvidenceResult):
        raise TypeError("`evidence` must be an AnnotationEvidenceResult.")
    if (
        isinstance(max_rows, bool)
        or not isinstance(max_rows, (int, np.integer))
        or max_rows < 1
    ):
        raise ValueError("`max_rows` must be a positive integer.")
    requested = _normalize_features(features)
    profiles = evidence.evaluated_profiles.copy()
    missing_columns = [
        column for column in REQUIRED_DISPLAY_COLUMNS if column not in profiles.columns
    ]
    if missing_columns:
        raise ValueError(
            "Evaluated annotation evidence is missing display columns: "
            f"{missing_columns}"
        )
    if profiles.empty:
        raise ValueError("Evaluated annotation evidence must contain at least one row.")
    profiles["display_priority"] = profiles.apply(_display_priority, axis=1)
    profiles["display_label"] = profiles.apply(_display_label, axis=1)

    if requested is not None:
        available = set(profiles["feature_id"])
        missing = [feature for feature in requested if feature not in available]
        if missing:
            raise ValueError(
                f"Requested display features are not present: {missing!r}."
            )
        requested_set = set(requested)
        selected = profiles[profiles["feature_id"].isin(requested_set)].copy()
        ordered = sorted(
            selected.index,
            key=lambda index: _stable_identity_key(selected.loc[index]),
        )
    else:

        def selection_key(index: Any) -> tuple[Any, ...]:
            row = profiles.loc[index]
            effect = row["mean_difference"]
            magnitude_key = -abs(float(effect)) if not _is_missing(effect) else np.inf
            return (
                DISPLAY_PRIORITY[row["display_priority"]],
                magnitude_key,
                _stable_identity_key(row),
            )

        ordered = sorted(profiles.index, key=selection_key)[: int(max_rows)]
        selected = profiles

    result = selected.loc[ordered].reset_index(drop=True)
    result["display_selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _unavailable(axis: Axes, message: str) -> None:
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color=NEUTRAL_COLOR,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _plot_coverage(axis: Axes, selected: pd.DataFrame) -> None:
    positions = np.arange(len(selected))
    height = 0.36
    axis.barh(
        positions - height / 2,
        selected["target_detection_fraction"],
        height=height,
        color=TARGET_COLOR,
        label="Target",
    )
    axis.barh(
        positions + height / 2,
        selected["alternative_detection_fraction"],
        height=height,
        color=ALTERNATIVE_COLOR,
        label="Alternative",
    )
    axis.set_yticks(positions, selected["display_label"])
    axis.set_xlim(0, 1)
    axis.set_xlabel("Detection fraction")
    axis.invert_yaxis()
    axis.legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
    )
    axis.set_title("Detection coverage")


def _plot_specificity(axis: Axes, selected: pd.DataFrame) -> None:
    if "target_specificity" not in selected.columns:
        _unavailable(axis, "Target specificity unavailable")
        axis.set_title("Specificity and pooled effect")
        return
    valid = selected["target_specificity"].notna() & selected["mean_difference"].notna()
    if not valid.any():
        _unavailable(axis, "Target specificity unavailable")
        axis.set_title("Specificity and pooled effect")
        return
    plotted = selected[valid]
    colors = [
        TARGET_COLOR
        if direction == "target_favoring"
        else ALTERNATIVE_COLOR
        if direction == "alternative_favoring"
        else NEUTRAL_COLOR
        for direction in plotted["effect_direction"]
    ]
    axis.scatter(
        plotted["mean_difference"],
        plotted["target_specificity"],
        c=colors,
        s=42,
    )
    for _, row in plotted.iterrows():
        axis.annotate(
            row["display_label"],
            (row["mean_difference"], row["target_specificity"]),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.axvline(0, color=NEUTRAL_COLOR, linewidth=0.8, linestyle="--")
    axis.axhline(0.5, color=NEUTRAL_COLOR, linewidth=0.8, linestyle=":")
    axis.set_ylim(-0.02, 1.08)
    axis.set_xlabel("Mean difference (target − alternative)")
    axis.set_ylabel("Target specificity")
    axis.set_title("Specificity and pooled effect")


def _plot_contrasts(axis: Axes, selected: pd.DataFrame) -> None:
    valid = selected["mean_difference"].notna()
    if not valid.any():
        _unavailable(axis, "Pooled contrast effects unavailable")
        axis.set_title("Explicit contrast effects")
        return
    plotted = selected[valid].copy()
    plotted["_contrast_label"] = plotted.apply(_contrast_label, axis=1)
    features = sorted(pd.unique(plotted["feature_id"]), key=_stable_value_key)
    contrasts = []
    for label in plotted["_contrast_label"]:
        if label not in contrasts:
            contrasts.append(label)
    feature_positions = {feature: index for index, feature in enumerate(features)}
    contrast_positions = {label: index for index, label in enumerate(contrasts)}
    matrix = np.full((len(features), len(contrasts)), np.nan)
    for _, row in plotted.iterrows():
        matrix[
            feature_positions[row["feature_id"]],
            contrast_positions[row["_contrast_label"]],
        ] = row["mean_difference"]
    limit = float(np.nanmax(np.abs(matrix)))
    if limit == 0:
        limit = 1.0
    colormap = plt.get_cmap("RdBu_r").with_extremes(bad="#E6E6E6")
    axis.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        cmap=colormap,
        vmin=-limit,
        vmax=limit,
    )
    axis.set_xticks(np.arange(len(contrasts)), contrasts, rotation=35, ha="right")
    axis.set_yticks(np.arange(len(features)), [str(value) for value in features])
    axis.set_xlabel("Explicit alternative relationship")
    axis.set_ylabel("Feature")
    axis.set_title("Explicit contrast effects")


def _plot_replicates(axis: Axes, selected: pd.DataFrame) -> None:
    required = {
        "effect_recurrence_fraction",
        "directional_consistency_fraction",
    }
    if not required.issubset(selected.columns):
        _unavailable(axis, "Biological-sample evidence unavailable")
        axis.set_title("Biological-sample consistency")
        return
    valid = selected[list(required)].notna().all(axis=1)
    if not valid.any():
        _unavailable(axis, "Biological-sample evidence unavailable")
        axis.set_title("Biological-sample consistency")
        return
    plotted = selected[valid]
    sizes = np.full(len(plotted), 48.0)
    if "sample_effect_dominance" in plotted.columns:
        dominance = plotted["sample_effect_dominance"].to_numpy(dtype=float)
        finite = np.isfinite(dominance)
        sizes[finite] = 35.0 + 70.0 * dominance[finite]
    axis.scatter(
        plotted["effect_recurrence_fraction"],
        plotted["directional_consistency_fraction"],
        s=sizes,
        color=TARGET_COLOR,
        alpha=0.8,
    )
    for _, row in plotted.iterrows():
        axis.annotate(
            row["display_label"],
            (
                row["effect_recurrence_fraction"],
                row["directional_consistency_fraction"],
            ),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    axis.set_xlim(-0.02, 1.08)
    axis.set_ylim(-0.02, 1.08)
    axis.set_xlabel("Effect recurrence fraction")
    axis.set_ylabel("Directional consistency fraction")
    axis.set_title("Biological-sample consistency")


def _plot_ambiguity(
    axis: Axes,
    selected: pd.DataFrame,
    evidence: AnnotationEvidenceResult,
) -> None:
    matrix = np.column_stack(
        [
            selected["supports_proposed_label"].fillna(False).to_numpy(dtype=bool),
            selected["shared_or_broad"].fillna(False).to_numpy(dtype=bool),
            selected["is_competitor_evidence"].to_numpy(dtype=bool),
            selected["expectation_status"].eq("contradictory").to_numpy(),
            selected["expectation_status"].eq("unresolved").to_numpy(),
        ]
    ).astype(int)
    axis.imshow(matrix, aspect="auto", cmap="Greys", vmin=0, vmax=1)
    axis.set_xticks(
        np.arange(5),
        ["Support", "Shared", "Competitor", "Contradictory", "Unresolved"],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(np.arange(len(selected)), selected["display_label"])
    axis.set_title("Evidence ambiguity")
    unprofiled = int(
        evidence.unresolved_expectations["expectation_reason"]
        .eq("expected_feature_not_profiled")
        .sum()
    )
    if unprofiled:
        axis.text(
            0.5,
            -0.22,
            f"Unprofiled expected features: {unprofiled}",
            ha="center",
            va="top",
            transform=axis.transAxes,
            color=NEUTRAL_COLOR,
            clip_on=False,
        )


def plot_annotation_evidence(
    evidence: AnnotationEvidenceResult,
    *,
    features: Optional[Iterable[Any]] = None,
    max_rows: int = 12,
    figsize: Optional[tuple[float, float]] = None,
) -> tuple[Figure, np.ndarray]:
    """Build a five-panel annotation-evidence dashboard without showing it."""
    selected = select_annotation_evidence_rows(
        evidence,
        features=features,
        max_rows=max_rows,
    )
    if figsize is None:
        height = max(8.0, 4.8 + 0.35 * len(selected))
        figsize = (18.0, height)
    figure, grid = plt.subplot_mosaic(
        [
            ["coverage", "specificity", "contrasts"],
            ["replicates", "ambiguity", "ambiguity"],
        ],
        figsize=figsize,
        constrained_layout=True,
    )
    axes = np.asarray(
        [
            grid["coverage"],
            grid["specificity"],
            grid["contrasts"],
            grid["replicates"],
            grid["ambiguity"],
        ],
        dtype=object,
    )

    _plot_coverage(axes[0], selected)
    _plot_specificity(axes[1], selected)
    _plot_contrasts(axes[2], selected)
    _plot_replicates(axes[3], selected)
    _plot_ambiguity(axes[4], selected, evidence)
    figure.suptitle(f"Annotation evidence for {evidence.proposed_label!r}")
    return figure, axes
