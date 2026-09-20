#!/usr/bin/env python

"""
Public API for SingleCellFeatureProfiler.
"""

import os
import warnings
from typing import Dict, List, Optional, Union

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from ._engine import _run_profiling_engine
from ._legacy.selection import legacy_select_robust_markers
from ._selection_marker import select_marker_candidates
from ._stability import _calculate_stability_scores
from .comparison import compare_feature_profiles as compare_feature_profiles
from .diagnostics import diagnose_clustering_geometry as diagnose_clustering_geometry
from .evidence import evaluate_annotation_evidence as evaluate_annotation_evidence
from .inputs import resolve_expression_input
from .models import ExpressionSource
from .plotting import plot_annotation_evidence as plot_annotation_evidence
from .plotting import (
    select_annotation_evidence_rows as select_annotation_evidence_rows,
)
from .profiles import profile_features as profile_features
from .replicates import profile_features_by_sample as profile_features_by_sample
from .resampling import resample_profile_cells as resample_profile_cells
from .resampling import resample_profile_samples as resample_profile_samples
from .selection import select_profile_features as select_profile_features
from .selection import summarize_resampled_selection as summarize_resampled_selection

try:
    from anndata import AnnData

    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None


def get_feature_profiles(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    features: Optional[Union[List[str], str]] = None,
    feature_names: Optional[List[str]] = None,
    condition_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
    specificity_metric: str = "tau",
    background_rate: float = 0.01,
    n_jobs: int = -1,
    verbose: bool = True,
    expression_source: Union[str, ExpressionSource] = "X",
) -> pd.DataFrame:
    """
    Provides a complete statistical profile for features across all groups.
    If `condition_by` is provided, the output is a detailed per-condition table.
    """
    resolved = resolve_expression_input(
        data=data,
        group_by=group_by,
        feature_names=feature_names,
        condition_by=condition_by,
        expression_source=expression_source,
    )
    all_f_names = resolved.feature_ids.tolist()

    # --- REFACTORED: API now handles all feature input types directly ---
    feature_list = None
    if features is not None:
        if isinstance(features, list):
            feature_list = features
        elif isinstance(features, str):
            if os.path.exists(features):
                if verbose:
                    print(f"Loading features from file: {features}")
                with open(features, "r") as f:
                    feature_list = [line.strip() for line in f if line.strip()]
            else:
                if verbose:
                    print("Parsing features from comma-separated string.")
                feature_list = [f.strip() for f in features.split(",")]
        else:
            raise TypeError(
                "`features` must be a list of strings, a valid file path, "
                f"or a comma-separated string, but got {type(features)}"
            )

    if feature_list is not None:
        if verbose:
            print(f"Profiling {len(feature_list)} user-provided features.")
        missing = [f for f in feature_list if f not in all_f_names]
        if missing:
            raise ValueError(
                f"The following features were not found in the data: {missing}"
            )
        features_to_analyze = feature_list
    else:
        if verbose:
            print(
                f"Warning: No feature file provided. Profiling all {len(all_f_names)} features. This may be slow and memory-intensive."
            )
        features_to_analyze = all_f_names

    if not features_to_analyze:
        print("Warning: No features to analyze. Returning empty DataFrame.")
        return pd.DataFrame()

    results_df = _run_profiling_engine(
        expression_data=resolved,
        features_to_analyze=features_to_analyze,
        all_feature_names=all_f_names,
        group_labels=resolved.group_labels,
        condition_labels=resolved.condition_labels,
        specificity_metric=specificity_metric,
        background_rate=background_rate,
        n_jobs=n_jobs,
        verbose=verbose,
    )

    if results_df.empty:
        return results_df

    sort_keys = ["feature_id", "group"]
    if "condition" in results_df.columns:
        sort_keys.append("condition")

    return results_df.sort_values(by=sort_keys).reset_index(drop=True)


def find_marker_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    condition_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
    specificity_threshold: float = 0.7,
    min_pct_expressing: float = 10.0,
    fdr_marker_threshold: float = 0.05,
    min_freq: float = 0.05,
    max_freq: float = 0.90,
    var_mean_ratio_min: float = 1.5,
    gap_stat_min: float = 1.2,
    right_tail_min: float = 2.5,
    cv_min: Optional[float] = 0.8,
    verbose: bool = True,
    feature_names: Optional[List[str]] = None,
    expression_source: Union[str, ExpressionSource] = "X",
    **kwargs,
) -> pd.DataFrame:
    """
    Finds robust marker features and ranks them by their stability across conditions.
    """
    if verbose:
        print("Finding robust marker features using a data-driven pipeline...")

    resolved = resolve_expression_input(
        data,
        group_by=group_by,
        feature_names=feature_names,
        condition_by=condition_by,
        expression_source=expression_source,
    )

    if (
        resolved.condition_labels is not None
        and len(np.unique(resolved.condition_labels)) == 1
    ):
        if verbose:
            print(
                "Info: Only one unique condition found. 'stability_score' will be 1.0 for all features."
            )

    candidate_features = select_marker_candidates(
        resolved,
        min_freq=min_freq,
        max_freq=max_freq,
        var_mean_ratio_min=var_mean_ratio_min,
        gap_stat_min=gap_stat_min,
        right_tail_min=right_tail_min,
        cv_min=cv_min,
        verbose=verbose,
    )

    if not candidate_features:
        if verbose:
            print(
                "Warning: No candidate features found after selection. Returning empty DataFrame."
            )
        return pd.DataFrame()

    per_condition_profiles = _run_profiling_engine(
        expression_data=resolved,
        features_to_analyze=candidate_features,
        all_feature_names=resolved.feature_ids.tolist(),
        group_labels=resolved.group_labels,
        condition_labels=resolved.condition_labels,
        verbose=verbose,
        n_jobs=kwargs.get("n_jobs", -1),
        specificity_metric=kwargs.get("specificity_metric", "tau"),
        background_rate=kwargs.get("background_rate", 0.01),
    )

    if per_condition_profiles.empty:
        return pd.DataFrame()

    specificity_metric = kwargs.get("specificity_metric", "tau")
    specificity_col = f"specificity_{specificity_metric}"
    aggregated_markers = _calculate_stability_scores(
        per_condition_profiles, specificity_col
    )

    final_markers_df = aggregated_markers[
        (aggregated_markers[specificity_col] >= specificity_threshold)
        & (aggregated_markers["pct_expressing"] >= min_pct_expressing)
        & (aggregated_markers["fdr_marker"] <= fdr_marker_threshold)
        & (aggregated_markers["log2fc_all"] > 0)
    ].copy()

    return final_markers_df.sort_values(
        by=["group", "fdr_marker", "stability_score"], ascending=[True, True, False]
    ).reset_index(drop=True)


def get_feature_activity(
    profiles_df: pd.DataFrame,
    fdr_presence_threshold: float = 0.05,
    top_n: Optional[int] = None,
) -> Dict[str, List[str]]:
    """
    Summarizes a profile DataFrame to show in which groups features are active.
    """
    required_cols = ["feature_id", "group", "fdr_presence", "norm_score"]
    if not all(col in profiles_df.columns for col in required_cols):
        raise ValueError(
            "Input DataFrame is missing required columns. "
            f"Expected: {', '.join(required_cols)}"
        )

    active_df = profiles_df[
        profiles_df["fdr_presence"] <= fdr_presence_threshold
    ].copy()

    if active_df.empty:
        return {}

    active_df.sort_values(
        by=["feature_id", "norm_score"], ascending=[True, False], inplace=True
    )

    grouped = active_df.groupby("feature_id", observed=True)

    if top_n is not None:
        result_series = (
            grouped.head(top_n)
            .groupby("feature_id", observed=True)["group"]
            .apply(list)
        )
    else:
        result_series = grouped["group"].apply(list)

    return result_series.to_dict()


def evaluate_clustering(
    adata: ad.AnnData, cluster_key: str, use_rep: str = "X_pca", verbose: bool = True
) -> pd.DataFrame:
    """Compatibility wrapper for historical silhouette reporting.

    Args:
        adata (anndata.AnnData): The annotated data matrix.
        cluster_key (str): The key in `adata.obs` where the cluster labels are stored.
        use_rep (str): The representation in `adata.obsm` to use for calculating
            distances (e.g., 'X_pca', 'X_umap'). PCA is recommended.
        verbose (bool): If True, prints a summary of the results.

    Returns:
        pd.DataFrame: A DataFrame with the average silhouette score and size
            for each cluster, sorted by score.
    """
    warnings.warn(
        "`evaluate_clustering()` is deprecated because silhouette values are "
        "geometry diagnostics, not annotation validation. Use "
        "`diagnose_clustering_geometry()` for non-mutating, explicit results.",
        DeprecationWarning,
        stacklevel=2,
    )
    if use_rep not in adata.obsm:
        raise ValueError(f"Representation {use_rep!r} not found in adata.obsm.")
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key {cluster_key!r} not found in adata.obs.")
    result = diagnose_clustering_geometry(
        adata, cluster_key, use_rep=use_rep, metric="euclidean"
    )
    score_column = f"silhouette_{cluster_key}"
    adata.obs[score_column] = result.cell_diagnostics["silhouette_score"].to_numpy()
    groups = result.group_diagnostics.set_index("group_label")
    report_df = groups.rename(
        columns={"mean_silhouette_score": "avg_silhouette_score"}
    )[["avg_silhouette_score", "n_cells"]].sort_values(
        "avg_silhouette_score", ascending=False
    )
    report_df.index.name = cluster_key
    overall_score = result.overall_diagnostics.loc[0, "mean_silhouette_score"]

    if verbose:
        print("--- Clustering Geometry Diagnostics ---")
        print(f"Overall Average Silhouette Score: {overall_score:.3f}\n")
        print("Per-Group Scores:")
        print(report_df)
        print(
            "\nThese values describe cohesion and separation in the selected "
            "representation; they do not establish biological validity."
        )
        print("---------------------------------")

    return report_df


def select_robust_markers(
    ranked_markers_df: pd.DataFrame,
    top_n: int = 10,
    fdr_threshold: float = 0.05,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compatibility wrapper for the historical composite/K-means selector."""
    warnings.warn(
        "`select_robust_markers()` uses the legacy composite-score and K-means "
        "selection behavior. Use `select_profile_features()` with explicit "
        "SelectionCriteria for new analyses.",
        DeprecationWarning,
        stacklevel=2,
    )
    return legacy_select_robust_markers(
        ranked_markers_df,
        top_n=top_n,
        fdr_threshold=fdr_threshold,
        verbose=verbose,
    )
