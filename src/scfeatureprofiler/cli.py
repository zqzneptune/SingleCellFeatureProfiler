#!/usr/bin/env python

"""
Command-Line Interface for SingleCellFeatureProfiler.

This module provides CLI access to the main profiling and analysis functions,
allowing users to run the tool directly from the shell.
"""

from typing import List, Optional

import typer
import pandas as pd

from .api import get_feature_profiles, get_feature_activity, find_marker_features


app = typer.Typer(
    name="scprofiler",
    help="A CLI for statistical profiling of single-cell feature expression.",
    add_completion=False
)

def _read_data(input_file: str):
    """Helper to read expression data from CSV or H5AD."""
    if input_file.endswith(".h5ad"):
        try:
            import anndata
            return anndata.read_h5ad(input_file)
        except ImportError:
            typer.secho("Error: anndata is not installed. Please run 'pip install scfeatureprofiler[anndata]' to use .h5ad files.", fg=typer.colors.RED)
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
    # --- REFACTORED: The logic of these two flags has changed ---
    features: Optional[str] = typer.Option(None, "--features", "-f", help="Specific comma-separated list of features to profile."),
    all_features: bool = typer.Option(False, "--all-features", help="Profile ALL features. Overrides HVG selection."),
    n_hvg: int = typer.Option(3000, "--n-hvg", help="Number of top Highly Variable Genes to profile by default."),
    # --- END REFACTOR ---
    batch_by: Optional[str] = typer.Option(None, "--batch-by", "-b", help="Path to batch labels CSV file OR column name in adata.obs."),
    specificity_metric: str = typer.Option("tau", help="Specificity metric ('tau' or 'gini')."),
    n_jobs: int = typer.Option(-1, help="Number of parallel jobs (-1 for all)."),
    min_freq: float = typer.Option(0.05, help="[Select] Min detection frequency."),
    max_freq: float = typer.Option(0.90, help="[Select] Max detection frequency."),
    var_mean_ratio_min: float = typer.Option(1.5, help="[Select] Min variance-to-mean ratio."),
    gap_stat_min: float = typer.Option(1.2, help="[Select] Min gap statistic."),
    right_tail_min: float = typer.Option(2.5, help="[Select] Min right-tail heaviness."),
    cv_min: Optional[float] = typer.Option(0.8, help="[Select] Min coefficient of variation."),
):
    """
    Generate a statistical profile for features.

    By default, if no feature list is provided via --features, this command
    will run a smart selection algorithm to find and profile promising
    marker candidates.

    To profile every single feature, use the --all-features flag.
    Warning: This can be very slow and memory-intensive.
    """
    verbose = ctx.parent.params.get('verbose', True)
    
    if features and all_features:
        typer.secho("Error: Cannot use --features and --all-features at the same time.", fg=typer.colors.RED)
        raise typer.Exit(code=1)
        
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    batch_labels = _read_labels(batch_by) if batch_by and ".csv" in batch_by else batch_by
    
    feature_list = features.split(',') if features else None

    if verbose and not features and not all_features:
        typer.echo("No feature list provided. Using smart candidate selection by default.")
    
    results_df = get_feature_profiles(
        data=data, group_by=group_labels, features=feature_list,
        all_features=all_features,
        batch_by=batch_labels, specificity_metric=specificity_metric, n_jobs=n_jobs,
        min_freq=min_freq, max_freq=max_freq, var_mean_ratio_min=var_mean_ratio_min,
        gap_stat_min=gap_stat_min, right_tail_min=right_tail_min, cv_min=cv_min,
        verbose=verbose
    )
    
    if results_df.empty:
        typer.secho("Warning: Profiling resulted in an empty DataFrame. No output file will be written.", fg=typer.colors.YELLOW)
    else:
        results_df.to_csv(output_file, index=False)
        typer.echo(f"✅ Success! Profiles saved to {output_file}")

@app.command(name="find-markers")
def find_markers_cli(
    ctx: typer.Context, # <-- ADD THIS
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD)."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Path to group labels CSV file OR column name in adata.obs."),
    output_file: str = typer.Option("markers.csv", "--output", "-o", help="Path to save the output CSV file."),
    # Post-processing options
    specificity_threshold: float = typer.Option(0.7, help="[Filter] Min specificity score for a feature to be a marker."),
    min_pct_expressing: float = typer.Option(10.0, help="[Filter] Min percentage of cells expressing the feature."),
    fdr_marker_threshold: float = typer.Option(0.05, help="[Filter] Max FDR for the one-vs-rest marker test."),
    # Pre-processing options
    min_freq: float = typer.Option(0.05, help="[Select] Min detection frequency."),
    max_freq: float = typer.Option(0.90, help="[Select] Max detection frequency."),
    var_mean_ratio_min: float = typer.Option(1.5, help="[Select] Min variance-to-mean ratio."),
    gap_stat_min: float = typer.Option(1.2, help="[Select] Min gap statistic."),
    right_tail_min: float = typer.Option(2.5, help="[Select] Min right-tail heaviness ratio."),
    cv_min: Optional[float] = typer.Option(0.8, help="[Select] Min coefficient of variation (optional)."),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Number of parallel jobs to use (-1 for all).")
):
    """
    Run the full data-driven pipeline to find marker features for each group.
    """
    verbose = ctx.parent.params.get('verbose', True) # <-- CHANGE THIS
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by

    marker_dict = find_marker_features(
        data=data, group_by=group_labels, n_jobs=n_jobs,
        specificity_threshold=specificity_threshold,
        min_pct_expressing=min_pct_expressing,
        fdr_marker_threshold=fdr_marker_threshold,
        min_freq=min_freq, max_freq=max_freq,
        var_mean_ratio_min=var_mean_ratio_min,
        gap_stat_min=gap_stat_min, right_tail_min=right_tail_min, cv_min=cv_min
    )

    if not any(marker_dict.values()):
        typer.secho("Warning: No marker features were found with the given thresholds.", fg=typer.colors.YELLOW)
    else:
        # Convert dictionary to a long-form DataFrame for easy saving to CSV
        marker_df = pd.DataFrame(
            [(group, feature) for group, features in marker_dict.items() for feature in features],
            columns=['group', 'marker_feature']
        ).sort_values(by=['group', 'marker_feature'])
        
        marker_df.to_csv(output_file, index=False)
        typer.echo(f"✅ Success! Marker list saved to {output_file}")


@app.command()
def activity(
    ctx: typer.Context, # <-- ADD THIS
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD)."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Path to group labels CSV file OR column name in adata.obs."),
    features: str = typer.Option(..., "--features", "-f", help="Comma-separated list of features to check."),
    output_file: str = typer.Option("feature_activity.csv", "--output", "-o", help="Path to save the output CSV file."),
    fdr_threshold: float = typer.Option(0.05, "--fdr-threshold", help="FDR threshold to consider a feature 'ON'.")
):
    """
    Identify in which groups a given list of features are actively ON.
    """
    verbose = ctx.parent.params.get('verbose', True) 
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    feature_list = features.split(',')

    typer.echo("Checking feature activity...")
    results_df = get_feature_activity(
        data=data,
        group_by=group_labels,
        features=feature_list,
        fdr_presence_threshold=fdr_threshold,
        verbose=verbose
    )
    
    if results_df.empty:
        typer.secho("Warning: No features were found to be active with the given threshold.", fg=typer.colors.YELLOW)
    else:
        results_df.to_csv(output_file, index=False)
        typer.echo(f"✅ Success! Activity results saved to {output_file}")


if __name__ == "__main__":
    app()