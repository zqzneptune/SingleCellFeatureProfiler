#!/usr/bin/env python

"""Thin command-line adapters for the public scFeatureProfiler APIs."""

from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
import typer

from ._logging import setup_logging
from .api import (
    diagnose_clustering_geometry,
    find_marker_features,
    get_feature_activity,
    get_feature_profiles,
    profile_features,
    profile_features_by_sample,
    select_profile_features,
)
from .contrasts import Contrast
from .selection import RankingCriterion, SelectionCriteria

app = typer.Typer(
    name="scfeatureprofiler",
    help="Profile explicit molecular evidence for supplied single-cell labels.",
    add_completion=False,
    pretty_exceptions_show_locals=False,
)


@app.callback()
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(
        True,
        "--verbose/--quiet",
        "-v/-q",
        help="Enable or suppress progress messages.",
    ),
) -> None:
    """Profile molecular evidence without collapsing evidence dimensions."""
    setup_logging(level="INFO" if verbose else "WARNING")
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _abort(message: str, code: int = 2) -> None:
    typer.secho(f"Error: {message}", fg=typer.colors.RED, err=True)
    raise typer.Exit(code=code)


def _read_data(input_file: str) -> Any:
    """Read a cells-by-features CSV or AnnData H5AD file."""
    path = Path(input_file)
    if not path.is_file():
        _abort(f"Input file was not found: {input_file}")
    suffix = path.suffix.lower()
    if suffix == ".h5ad":
        try:
            import anndata
        except ImportError:
            _abort(
                "AnnData support is unavailable. Install the package's anndata "
                "dependencies to read H5AD files."
            )
        return anndata.read_h5ad(path)
    if suffix == ".csv":
        return pd.read_csv(path, index_col=0)
    _abort(f"Unsupported input format {suffix!r}; use a .csv or .h5ad file.")


def _cell_ids(data: Any) -> pd.Index:
    if isinstance(data, pd.DataFrame):
        return data.index
    if hasattr(data, "obs_names"):
        return pd.Index(data.obs_names)
    raise TypeError("CLI input data must be a DataFrame or AnnData object.")


def _read_labels(labels_file: str, cell_ids: pd.Index, argument: str) -> pd.Series:
    path = Path(labels_file)
    if not path.is_file():
        _abort(f"{argument} label file was not found: {labels_file}")
    frame = pd.read_csv(path, index_col=0)
    if frame.shape[1] != 1:
        _abort(f"{argument} label CSV must contain exactly one data column.")
    if frame.index.has_duplicates:
        _abort(f"{argument} label CSV contains duplicate cell identifiers.")
    missing = cell_ids.difference(frame.index)
    extra = frame.index.difference(cell_ids)
    if len(missing) or len(extra):
        _abort(
            f"{argument} label identifiers must exactly match input cells "
            f"({len(missing)} missing, {len(extra)} extra)."
        )
    return frame.iloc[:, 0].reindex(cell_ids)


def _resolve_labels(data: Any, value: str, argument: str) -> Any:
    if Path(value).suffix.lower() == ".csv":
        return _read_labels(value, _cell_ids(data), argument)
    if isinstance(data, pd.DataFrame):
        _abort(f"{argument} must be a label CSV when --input is a CSV matrix.")
    return value


def _parse_features(features: Optional[str]) -> Optional[list[str]]:
    if features is None:
        return None
    path = Path(features)
    if path.is_file():
        values = [
            line.strip() for line in path.read_text().splitlines() if line.strip()
        ]
    else:
        values = [value.strip() for value in features.split(",") if value.strip()]
    if not values:
        _abort("--features must identify at least one feature.")
    if len(values) != len(set(values)):
        _abort("--features contains duplicate identifiers.")
    return values


def _resolve_contrast_arguments(
    targets: Optional[list[str]], alternatives: Optional[list[str]]
) -> tuple[Optional[list[str]], Optional[list[Contrast]]]:
    targets = list(targets or [])
    alternatives = list(alternatives or [])
    if not alternatives:
        return targets or None, None
    if len(targets) != 1:
        _abort("--alternative requires exactly one --target.")
    if len(alternatives) == 1:
        contrast = Contrast.target_vs_one(targets[0], alternatives[0])
    else:
        contrast = Contrast.target_vs_set(targets[0], alternatives)
    return None, [contrast]


def _write_frame(frame: pd.DataFrame, output_file: str, description: str) -> None:
    frame.to_csv(output_file, index=False)
    typer.echo(f"Saved {description} to {output_file}")


def _run_profile(
    *,
    data: Any,
    group_labels: Any,
    features: Optional[list[str]],
    targets: Optional[list[str]],
    alternatives: Optional[list[str]],
    expression_source: str,
    detection_threshold: float,
    fold_change_pseudocount: float,
    n_jobs: int,
    feature_chunk_size: int,
) -> pd.DataFrame:
    resolved_targets, contrasts = _resolve_contrast_arguments(targets, alternatives)
    return profile_features(
        data,
        group_labels,
        features=features,
        targets=resolved_targets,
        contrasts=contrasts,
        expression_source=expression_source,
        detection_threshold=detection_threshold,
        fold_change_pseudocount=fold_change_pseudocount,
        n_jobs=n_jobs,
        feature_chunk_size=feature_chunk_size,
    )


@app.command()
def profile(
    input_file: str = typer.Option(..., "--input", "-i", help="CSV or H5AD input."),
    group_by: str = typer.Option(
        ...,
        "--group-by",
        "-g",
        help="AnnData observation column or one-column label CSV.",
    ),
    output_file: str = typer.Option(
        "feature_profiles.csv", "--output", "-o", help="Canonical profile CSV."
    ),
    features: Optional[str] = typer.Option(
        None,
        "--features",
        "-f",
        help="Comma-separated features or a file containing one feature per line.",
    ),
    target: Annotated[
        Optional[list[str]],
        typer.Option(
            "--target", help="Target label; repeat for multiple target-vs-all rows."
        ),
    ] = None,
    alternative: Annotated[
        Optional[list[str]],
        typer.Option(
            "--alternative",
            help="Explicit alternative; repeat to define a set (requires one target).",
        ),
    ] = None,
    expression_source: str = typer.Option(
        "X", help="Expression source: X, raw, or layer:<name>."
    ),
    detection_threshold: float = typer.Option(
        0.0, help="A cell is detected only above this value."
    ),
    fold_change_pseudocount: float = typer.Option(
        1e-9, help="Positive numerical guard for the mean ratio."
    ),
    n_jobs: int = typer.Option(
        1, "--n-jobs", help="Shared-memory feature workers; -1 uses all CPUs."
    ),
    feature_chunk_size: int = typer.Option(
        32, "--feature-chunk-size", help="Maximum feature tasks submitted at once."
    ),
) -> None:
    """Create canonical pooled feature-target-contrast profiles."""
    data = _read_data(input_file)
    group_labels = _resolve_labels(data, group_by, "group-by")
    try:
        result = _run_profile(
            data=data,
            group_labels=group_labels,
            features=_parse_features(features),
            targets=target,
            alternatives=alternative,
            expression_source=expression_source,
            detection_threshold=detection_threshold,
            fold_change_pseudocount=fold_change_pseudocount,
            n_jobs=n_jobs,
            feature_chunk_size=feature_chunk_size,
        )
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    _write_frame(result, output_file, "canonical profiles")


@app.command(name="profile-samples")
def profile_samples(
    input_file: str = typer.Option(..., "--input", "-i", help="CSV or H5AD input."),
    group_by: str = typer.Option(..., "--group-by", "-g"),
    sample_by: str = typer.Option(..., "--sample-by", "-s"),
    output_file: str = typer.Option("feature_profiles.csv", "--output", "-o"),
    sample_output_file: str = typer.Option(
        "sample_profiles.csv", "--sample-output", help="Per-sample profile CSV."
    ),
    features: Optional[str] = typer.Option(None, "--features", "-f"),
    target: Annotated[Optional[list[str]], typer.Option("--target")] = None,
    alternative: Annotated[Optional[list[str]], typer.Option("--alternative")] = None,
    expression_source: str = typer.Option("X"),
    detection_threshold: float = typer.Option(0.0),
    fold_change_pseudocount: float = typer.Option(1e-9),
    minimum_cells_per_group: int = typer.Option(1),
    effect_direction_tolerance: float = typer.Option(0.0),
) -> None:
    """Create pooled and separate biological-sample feature profiles."""
    data = _read_data(input_file)
    group_labels = _resolve_labels(data, group_by, "group-by")
    sample_labels = _resolve_labels(data, sample_by, "sample-by")
    targets, contrasts = _resolve_contrast_arguments(target, alternative)
    try:
        result = profile_features_by_sample(
            data,
            group_labels,
            sample_labels,
            features=_parse_features(features),
            targets=targets,
            contrasts=contrasts,
            expression_source=expression_source,
            detection_threshold=detection_threshold,
            fold_change_pseudocount=fold_change_pseudocount,
            minimum_cells_per_group=minimum_cells_per_group,
            effect_direction_tolerance=effect_direction_tolerance,
        )
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    _write_frame(result.profiles, output_file, "pooled replicate-aware profiles")
    _write_frame(result.sample_profiles, sample_output_file, "per-sample profiles")


def _parse_ranking(values: Optional[list[str]]) -> list[RankingCriterion]:
    ranking = []
    for value in values or []:
        try:
            column, direction = value.rsplit(":", 1)
        except ValueError:
            _abort("Ranking values must use COLUMN:asc or COLUMN:desc syntax.")
        if not column or direction not in {"asc", "desc"}:
            _abort("Ranking values must use COLUMN:asc or COLUMN:desc syntax.")
        ranking.append(RankingCriterion(column, ascending=direction == "asc"))
    return ranking


@app.command(name="select")
def select_cli(
    profile_file: str = typer.Option(..., "--profiles", "-i"),
    output_file: str = typer.Option("selected_profiles.csv", "--output", "-o"),
    evaluated_output_file: Optional[str] = typer.Option(
        None, "--evaluated-output", help="Optional complete rule-evaluation CSV."
    ),
    min_target_detection_fraction: Optional[float] = typer.Option(None),
    max_alternative_detection_fraction: Optional[float] = typer.Option(None),
    min_mean_difference: Optional[float] = typer.Option(None),
    min_log2_mean_ratio: Optional[float] = typer.Option(None),
    min_target_specificity: Optional[float] = typer.Option(None),
    min_roc_auc: Optional[float] = typer.Option(None),
    min_average_precision: Optional[float] = typer.Option(None),
    max_cell_level_adjusted_p_value: Optional[float] = typer.Option(None),
    min_biological_samples_evaluated: Optional[int] = typer.Option(None),
    min_effect_recurrence_fraction: Optional[float] = typer.Option(None),
    min_directional_consistency_fraction: Optional[float] = typer.Option(None),
    max_sample_effect_dominance: Optional[float] = typer.Option(None),
    rank_by: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Explicit COLUMN:asc or COLUMN:desc; repeat for lexicographic rank."
        ),
    ] = None,
    top_n_per_relationship: Optional[int] = typer.Option(None),
) -> None:
    """Select profile rows with explicit independent measurement rules."""
    try:
        profiles = pd.read_csv(profile_file)
        criteria = SelectionCriteria(
            min_target_detection_fraction=min_target_detection_fraction,
            max_alternative_detection_fraction=max_alternative_detection_fraction,
            min_mean_difference=min_mean_difference,
            min_log2_mean_ratio=min_log2_mean_ratio,
            min_target_specificity=min_target_specificity,
            min_roc_auc=min_roc_auc,
            min_average_precision=min_average_precision,
            max_cell_level_adjusted_p_value=max_cell_level_adjusted_p_value,
            min_biological_samples_evaluated=min_biological_samples_evaluated,
            min_effect_recurrence_fraction=min_effect_recurrence_fraction,
            min_directional_consistency_fraction=(min_directional_consistency_fraction),
            max_sample_effect_dominance=max_sample_effect_dominance,
        )
        result = select_profile_features(
            profiles,
            criteria,
            ranking=_parse_ranking(rank_by),
            top_n_per_relationship=top_n_per_relationship,
        )
    except FileNotFoundError:
        _abort(f"Profile file was not found: {profile_file}")
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    _write_frame(result.selected_profiles, output_file, "selected profiles")
    if evaluated_output_file is not None:
        _write_frame(
            result.evaluated_profiles,
            evaluated_output_file,
            "complete selection evaluation",
        )


@app.command()
def diagnostics(
    input_file: str = typer.Option(..., "--input", "-i", help="AnnData H5AD input."),
    label_key: str = typer.Option(..., "--label-key", "-g"),
    output_prefix: str = typer.Option("clustering_diagnostics", "--output-prefix"),
    use_rep: str = typer.Option("X_pca", help="AnnData obsm representation."),
    metric: str = typer.Option("euclidean", help="Silhouette distance metric."),
) -> None:
    """Write non-mutating clustering geometry diagnostics."""
    data = _read_data(input_file)
    try:
        result = diagnose_clustering_geometry(
            data, label_key, use_rep=use_rep, metric=metric
        )
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    _write_frame(
        result.cell_diagnostics, f"{output_prefix}_cells.csv", "cell diagnostics"
    )
    _write_frame(
        result.group_diagnostics, f"{output_prefix}_groups.csv", "group diagnostics"
    )
    _write_frame(
        result.overall_diagnostics,
        f"{output_prefix}_overall.csv",
        "overall diagnostics",
    )


@app.command(name="legacy-profile", hidden=True)
def legacy_profile(
    ctx: typer.Context,
    input_file: str = typer.Option(..., "--input", "-i"),
    group_by: str = typer.Option(..., "--group-by", "-g"),
    output_file: str = typer.Option("legacy_feature_profiles.csv", "--output", "-o"),
    features: Optional[str] = typer.Option(None, "--features", "-f"),
    condition_by: Optional[str] = typer.Option(None, "--condition-by", "-c"),
    specificity_metric: str = typer.Option("tau"),
    n_jobs: int = typer.Option(-1),
    expression_source: str = typer.Option("X"),
) -> None:
    """Run the historical profiling contract for compatibility."""
    data = _read_data(input_file)
    group_labels = _resolve_labels(data, group_by, "group-by")
    condition_labels = (
        _resolve_labels(data, condition_by, "condition-by")
        if condition_by is not None
        else None
    )
    try:
        result = get_feature_profiles(
            data,
            group_labels,
            features=_parse_features(features),
            condition_by=condition_labels,
            specificity_metric=specificity_metric,
            n_jobs=n_jobs,
            verbose=ctx.obj["verbose"],
            expression_source=expression_source,
        )
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    _write_frame(result, output_file, "legacy profiles")


@app.command(name="find-markers")
def find_markers_cli(
    ctx: typer.Context,
    input_file: str = typer.Option(..., "--input", "-i"),
    group_by: str = typer.Option(..., "--group-by", "-g"),
    output_file: str = typer.Option("ranked_markers.csv", "--output", "-o"),
    condition_by: Optional[str] = typer.Option(None, "--condition-by", "-c"),
    expression_source: str = typer.Option("X"),
    specificity_threshold: float = typer.Option(0.7),
    min_pct_expressing: float = typer.Option(10.0),
    fdr_marker_threshold: float = typer.Option(0.05),
    min_freq: float = typer.Option(0.05),
    max_freq: float = typer.Option(0.90),
    var_mean_ratio_min: float = typer.Option(1.5),
    gap_stat_min: float = typer.Option(1.2),
    right_tail_min: float = typer.Option(2.5),
    cv_min: Optional[float] = typer.Option(0.8),
    n_jobs: int = typer.Option(-1, "--n-jobs"),
) -> None:
    """Run the historical candidate-filtered marker discovery workflow."""
    data = _read_data(input_file)
    group_labels = _resolve_labels(data, group_by, "group-by")
    condition_labels = (
        _resolve_labels(data, condition_by, "condition-by")
        if condition_by is not None
        else None
    )
    try:
        result = find_marker_features(
            data,
            group_labels,
            condition_by=condition_labels,
            n_jobs=n_jobs,
            verbose=ctx.obj["verbose"],
            specificity_threshold=specificity_threshold,
            min_pct_expressing=min_pct_expressing,
            fdr_marker_threshold=fdr_marker_threshold,
            min_freq=min_freq,
            max_freq=max_freq,
            var_mean_ratio_min=var_mean_ratio_min,
            gap_stat_min=gap_stat_min,
            right_tail_min=right_tail_min,
            cv_min=cv_min,
            expression_source=expression_source,
        )
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    if result.empty:
        typer.echo("No legacy marker rows passed the configured filters.")
        return
    _write_frame(result, output_file, "legacy marker profiles")


@app.command()
def activity(
    profile_file: str = typer.Argument(..., help="Legacy profile CSV."),
    fdr_threshold: float = typer.Option(0.05, "--fdr-threshold", "-t"),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n"),
) -> None:
    """Summarize the historical profile activity fields."""
    try:
        profiles = pd.read_csv(profile_file)
        activity_by_feature = get_feature_activity(
            profiles, fdr_presence_threshold=fdr_threshold, top_n=top_n
        )
    except FileNotFoundError:
        _abort(f"Profile file was not found: {profile_file}")
    except (KeyError, TypeError, ValueError) as error:
        _abort(str(error))
    if not activity_by_feature:
        typer.echo("No feature activity passed the configured threshold.")
        return
    for feature in profiles["feature_id"].unique():
        groups = activity_by_feature.get(feature, [])
        typer.echo(f"{feature}: {', '.join(map(str, groups)) or '(none)'}")


if __name__ == "__main__":
    app()
