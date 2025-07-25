#!/usr/bin/env python

"""
Command-Line Interface for SingleCellFeatureProfiler.

This module provides CLI access to the main profiling and analysis functions,
allowing users to run the tool directly from the shell.
"""

from typing import List, Optional

import typer
import pandas as pd

from .api import get_feature_profiles, get_feature_activity

# Create a Typer application
app = typer.Typer(
    name="scprofiler",
    help="A CLI for statistical profiling of single-cell feature expression.",
    add_completion=False
)

def _read_data(input_file: str, feature_names_file: Optional[str] = None):
    """Helper to read expression data from CSV or H5AD."""
    if input_file.endswith(".h5ad"):
        try:
            import anndata
            return anndata.read_h5ad(input_file)
        except ImportError:
            print("Error: anndata is not installed. Please install it to use .h5ad files.")
            raise typer.Exit(code=1)
    elif input_file.endswith(".csv"):
        data = pd.read_csv(input_file, index_col=0)
        if feature_names_file:
            print(f"Warning: --feature-names-file is ignored for CSV input. Using columns from {input_file}.")
        return data
    else:
        print(f"Error: Unsupported input file format: {input_file}. Use .csv or .h5ad.")
        raise typer.Exit(code=1)

def _read_labels(labels_file: str) -> pd.Series:
    """Helper to read labels from a CSV file."""
    return pd.read_csv(labels_file, index_col=0).squeeze("columns")


@app.command()
def profile(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD). For CSV, cells are rows, features are columns."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Path to group labels CSV file (cell_id, group_label) OR column name in adata.obs if input is H5AD."),
    output_file: str = typer.Option("feature_profiles.csv", "--output", "-o", help="Path to save the output CSV file."),
    features: Optional[str] = typer.Option(None, "--features", "-f", help="Comma-separated list of features to profile (e.g., 'CD4,GNLY,MS4A1'). If not provided, all active features are used."),
    batch_by: Optional[str] = typer.Option(None, "--batch-by", "-b", help="Path to batch labels CSV file OR column name in adata.obs."),
    specificity_metric: str = typer.Option("tau", "--specificity-metric", help="Specificity metric to use ('tau' or 'gini')."),
    n_jobs: int = typer.Option(-1, "--n-jobs", help="Number of parallel jobs to use (-1 for all).")
):
    """
    Generate a full statistical profile for features across groups.
    """
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    
    # Handle label/batch inputs
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    batch_labels = _read_labels(batch_by) if batch_by and ".csv" in batch_by else batch_by
    
    feature_list = features.split(',') if features else None

    typer.echo("Running feature profiling...")
    results_df = get_feature_profiles(
        data=data,
        group_by=group_labels,
        features=feature_list,
        batch_by=batch_labels,
        specificity_metric=specificity_metric,
        n_jobs=n_jobs
    )
    
    results_df.to_csv(output_file, index=False)
    typer.echo(f"✅ Success! Profiles saved to {output_file}")


@app.command()
def activity(
    input_file: str = typer.Option(..., "--input", "-i", help="Path to input data file (CSV or H5AD)."),
    group_by: str = typer.Option(..., "--group-by", "-g", help="Path to group labels CSV file OR column name in adata.obs."),
    features: str = typer.Option(..., "--features", "-f", help="Comma-separated list of features to check (e.g., 'CD4,GNLY,MS4A1')."),
    output_file: str = typer.Option("feature_activity.csv", "--output", "-o", help="Path to save the output CSV file."),
    fdr_threshold: float = typer.Option(0.05, "--fdr-threshold", help="FDR threshold to consider a feature 'ON'.")
):
    """
    Identify in which groups a given list of features are actively ON.
    """
    typer.echo(f"Loading data from {input_file}...")
    data = _read_data(input_file)
    
    group_labels = _read_labels(group_by) if ".csv" in group_by else group_by
    feature_list = features.split(',')

    typer.echo("Checking feature activity...")
    results_df = get_feature_activity(
        data=data,
        group_by=group_labels,
        features=feature_list,
        fdr_presence_threshold=fdr_threshold
    )
    
    results_df.to_csv(output_file, index=False)
    typer.echo(f"✅ Success! Activity results saved to {output_file}")


if __name__ == "__main__":
    app()