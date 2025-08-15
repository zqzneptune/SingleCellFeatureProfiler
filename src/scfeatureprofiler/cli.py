#!/usr/bin/env python

"""
Command-Line Interface for SingleCellFeatureProfiler.
"""

from typing import Optional, List
import time

import typer
import pandas as pd

from .api import get_feature_profiles, find_marker_features, get_feature_activity
from ._logging import setup_logging

app = typer.Typer(
    name="scfeatureprofiler",
    help="A CLI for statistical profiling of single-cell feature expression.",
    add_completion=False,
    pretty_exceptions_show_locals=False
)

@app.callback()
def main(verbose: bool = typer.Option(True, "--verbose", "-v", help="Enable verbose progress messages for computation.")):
    """
    scfeatureprofiler: A tool for detailed single-cell feature analysis.
    """
    setup_logging(level="INFO" if verbose else "WARNING")


def _read_data(input_file: str):
    """Helper to read expression data from CSV or H5AD."""
    if input_file.endswith(".h5ad"):
        try:
            import anndata
            return anndata.read_h5ad(input_file)
        except ImportError:
            typer.secho("Error: anndata is not installed. Please run 'pip install scfeatureprofiler[anndata]'", fg=typer.colors.RED)
            raise typer.Exit(code=1)
    elif input_file.endswith(".csv"):
        return pd.read_csv(input_file, index_col=0)
    else:
        typer.secho(f"Error: Unsupported input file format: {input_file}. Use .csv or .h5ad.", fg=typer.colors.RED)
        raise typer.Exit(code=1)


def _read_labels(labels_file: str) -> pd.Series:
    """Helper to read labels from a CSV file."""
    return pd.read_csv(labels_file, index_col=0).squeeze("columns")


@app.command()
def profile(
    ctx: typer.Context, 
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD)."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Path to group labels CSV file OR column name in adata.obs."),
    output_file: str = typer.Option("feature_profiles.csv", "--output", "-o", help="Path to save the output CSV file."),
    features: Optional[str] = typer.Option(
        None, 
        "--features", "-f", 
        help="Path to a feature file (one per line) OR a comma-separated string of feature names."
    ),
    condition_by: Optional[str] = typer.Option(None, "--condition-by", "-c", help="Path to condition labels CSV file OR column name in adata.obs."),
    specificity_metric: str = typer.Option("tau", help="Specificity metric ('tau' or 'gini')."),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all)."),
):
    """
    Generate a detailed statistical profile for features.

    If --features is not provided, a warning will be issued and ALL
    features in the dataset will be profiled.
    """
    verbose = ctx.parent.params.get('verbose', True)
    
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    
    if features is None:
        typer.secho(
            "Warning: No feature file provided. Profiling ALL features in the dataset.",
            fg=typer.colors.YELLOW
        )
        typer.secho("This can be very slow and memory-intensive.", fg=typer.colors.YELLOW)
        for i in range(5, 0, -1):
            typer.echo(f"  Continuing in {i} seconds... (Press CTRL+C to cancel)", nl=False)
            time.sleep(1)
            typer.echo("\r", nl=False)
        typer.echo("                                                          \r", nl=False)

    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    condition_labels = _read_labels(condition_by) if condition_by and ".csv" in condition_by else condition_by
    
    # --- REFACTORED: The logic for parsing 'features' is now in the API. ---
    # We simply pass the string argument directly.
    results_df = get_feature_profiles(
        data=data, 
        group_by=group_labels, 
        features=features,
        condition_by=condition_labels, 
        specificity_metric=specificity_metric, 
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    if results_df.empty:
        typer.secho("Warning: Profiling resulted in an empty DataFrame. No output file will be written.", fg=typer.colors.YELLOW)
    else:
        results_df.to_csv(output_file, index=False)
        typer.echo(f"✅ Success! Profiles saved to {output_file}")


@app.command(name="find-markers")
def find_markers_cli(
    ctx: typer.Context,
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD)."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Group labels source (e.g., 'leiden')."),
    condition_by: Optional[str] = typer.Option(None, "--condition-by", "-c", help="Condition labels for stability analysis (e.g., 'donor'). Optional."),
    output_file: str = typer.Option("ranked_markers.csv", "--output", "-o", help="Path for CSV output."),
    specificity_threshold: float = typer.Option(0.7, help="[Filter] Min specificity score."),
    min_pct_expressing: float = typer.Option(10.0, help="[Filter] Min percentage of expressing cells."),
    fdr_marker_threshold: float = typer.Option(0.05, help="[Filter] Max FDR for marker test."),
    min_freq: float = typer.Option(0.05, help="[Select] Min detection frequency."),
    max_freq: float = typer.Option(0.90, help="[Select] Max detection frequency."),
    var_mean_ratio_min: float = typer.Option(1.5, help="[Select] Min variance-to-mean ratio."),
    gap_stat_min: float = typer.Option(1.2, help="[Select] Min gap statistic."),
    right_tail_min: float = typer.Option(2.5, help="[Select] Min right-tail heaviness."),
    cv_min: Optional[float] = typer.Option(0.8, help="[Select] Min coefficient of variation."),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Number of parallel jobs.")
):
    """
    Find robust marker features and rank them by their stability across conditions.
    """
    verbose = ctx.parent.params.get('verbose', True)
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    condition_labels = _read_labels(condition_by) if condition_by and ".csv" in condition_by else condition_by

    ranked_markers_df = find_marker_features(
        data=data, 
        group_by=group_labels, 
        condition_by=condition_labels,
        n_jobs=n_jobs, 
        verbose=verbose,
        specificity_threshold=specificity_threshold,
        min_pct_expressing=min_pct_expressing,
        fdr_marker_threshold=fdr_marker_threshold,
        min_freq=min_freq, max_freq=max_freq,
        var_mean_ratio_min=var_mean_ratio_min,
        gap_stat_min=gap_stat_min, right_tail_min=right_tail_min, cv_min=cv_min
    )

    if ranked_markers_df.empty:
        typer.secho("Warning: No marker features were found with the given thresholds.", fg=typer.colors.YELLOW)
    else:
        ranked_markers_df.to_csv(output_file, index=False)
        typer.echo(f"✅ Success! Ranked marker table saved to {output_file}")


@app.command()
def activity(
    profile_file: str = typer.Argument(..., help="Path to the CSV file generated by the 'scfeatureprofiler profile' command."),
    fdr_threshold: float = typer.Option(0.05, "--fdr-threshold", "-t", help="FDR threshold to consider a feature 'ON'."),
    top_n: Optional[int] = typer.Option(None, "--top-n", "-n", help="Show only the top N most active groups per feature.")
):
    """
    Summarize a profile CSV to show in which groups features are active.
    """
    try:
        profiles_df = pd.read_csv(profile_file)
    except FileNotFoundError:
        typer.secho(f"Error: Profile file not found at '{profile_file}'", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    required_cols = ['feature_id', 'group', 'fdr_presence', 'norm_score']
    if not all(col in profiles_df.columns for col in required_cols):
        typer.secho(f"Error: The input file is missing required columns: {', '.join(required_cols)}", fg=typer.colors.RED)
        raise typer.Exit(code=1)

    activity_dict = get_feature_activity(
        profiles_df=profiles_df,
        fdr_presence_threshold=fdr_threshold,
        top_n=top_n
    )

    if not activity_dict:
        typer.echo("No significant feature activity found with the given threshold.")
        raise typer.Exit()

    typer.echo("\n--- Feature Activity Summary ---")
    
    all_features_in_profile = profiles_df['feature_id'].unique()
    active_features = activity_dict.keys()
    if not active_features:
        max_len = 0
    else:
        max_len = max(len(feature) for feature in active_features)
    
    for feature in all_features_in_profile:
        active_groups = activity_dict.get(feature)
        feature_part = f"{feature}:".ljust(max_len + 2)
        
        if active_groups:
            groups_str = ", ".join(map(str, active_groups))
            typer.secho(f"{feature_part}{groups_str}", fg=typer.colors.GREEN)
        else:
            typer.secho(f"{feature_part}(No significant activity found)", fg=typer.colors.BRIGHT_BLACK)
    
    typer.echo("-" * 30)


if __name__ == "__main__":
    app()