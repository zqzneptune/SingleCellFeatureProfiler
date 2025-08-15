#!/usr/bin/env python

"""
Public API for SingleCellFeatureProfiler.
"""

from typing import List, Optional, Union, Dict
import os

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.metrics import silhouette_samples, silhouette_score
import anndata as ad
from ._utils import _prepare_and_validate_inputs
from ._engine import _run_profiling_engine
from ._selection_marker import select_marker_candidates
from ._stability import _calculate_stability_scores
from sklearn.cluster import KMeans
from sklearn.preprocessing import minmax_scale

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
    specificity_metric: str = 'tau',
    background_rate: float = 0.01,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Provides a complete statistical profile for features across all groups.
    If `condition_by` is provided, the output is a detailed per-condition table.
    """
    expression_matrix, all_f_names, group_labels, condition_labels = _prepare_and_validate_inputs(
        data=data,
        group_by=group_by,
        feature_names=feature_names,
        condition_by=condition_by
    )

    # --- REFACTORED: API now handles all feature input types directly ---
    feature_list = None
    if features is not None:
        if isinstance(features, list):
            feature_list = features
        elif isinstance(features, str):
            if os.path.exists(features):
                if verbose:
                    print(f"Loading features from file: {features}")
                with open(features, 'r') as f:
                    feature_list = [line.strip() for line in f if line.strip()]
            else:
                if verbose:
                    print("Parsing features from comma-separated string.")
                feature_list = [f.strip() for f in features.split(',')]
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
            raise ValueError(f"The following features were not found in the data: {missing}")
        features_to_analyze = feature_list
    else:
        if verbose:
            print(f"Warning: No feature file provided. Profiling all {len(all_f_names)} features. This may be slow and memory-intensive.")
        features_to_analyze = all_f_names

    if not features_to_analyze:
        print("Warning: No features to analyze. Returning empty DataFrame.")
        return pd.DataFrame()

    results_df = _run_profiling_engine(
        expression_data=data if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else expression_matrix,
        features_to_analyze=features_to_analyze,
        all_feature_names=all_f_names,
        group_labels=group_labels,
        condition_labels=condition_labels, 
        specificity_metric=specificity_metric,
        background_rate=background_rate,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    if results_df.empty:
        return results_df
        
    sort_keys = ['feature_id', 'group']
    if 'condition' in results_df.columns:
        sort_keys.append('condition')
        
    return results_df.sort_values(by=sort_keys).reset_index(drop=True)


def find_marker_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: str,
    condition_by: Optional[str] = None,
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
    **kwargs
) -> pd.DataFrame:
    """
    Finds robust marker features and ranks them by their stability across conditions.
    """
    if verbose:
        print("Finding robust marker features using a data-driven pipeline...")

    expression_matrix, all_f_names, group_labels, condition_labels = _prepare_and_validate_inputs(
        data, group_by=group_by, condition_by=condition_by
    )
    
    if condition_labels is not None and len(np.unique(condition_labels)) == 1:
        if verbose:
            print("Info: Only one unique condition found. 'stability_score' will be 1.0 for all features.")

    candidate_features = select_marker_candidates(
        data, 
        feature_names=None if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else all_f_names,
        min_freq=min_freq, max_freq=max_freq,
        var_mean_ratio_min=var_mean_ratio_min, gap_stat_min=gap_stat_min,
        right_tail_min=right_tail_min, cv_min=cv_min, verbose=verbose
    )

    if not candidate_features:
        if verbose:
            print("Warning: No candidate features found after selection. Returning empty DataFrame.")
        return pd.DataFrame()

    per_condition_profiles = _run_profiling_engine(
        expression_data=data if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else expression_matrix, 
        features_to_analyze=candidate_features,
        all_feature_names=all_f_names,
        group_labels=group_labels,
        condition_labels=condition_labels,
        verbose=verbose,
        n_jobs=kwargs.get('n_jobs', -1),
        specificity_metric=kwargs.get('specificity_metric', 'tau'),
        background_rate=kwargs.get('background_rate', 0.01)
    )
    
    if per_condition_profiles.empty:
        return pd.DataFrame()

    specificity_metric = kwargs.get('specificity_metric', 'tau')
    specificity_col = f'specificity_{specificity_metric}'
    aggregated_markers = _calculate_stability_scores(per_condition_profiles, specificity_col)

    final_markers_df = aggregated_markers[
        (aggregated_markers[specificity_col] >= specificity_threshold) &
        (aggregated_markers['pct_expressing'] >= min_pct_expressing) &
        (aggregated_markers['fdr_marker'] <= fdr_marker_threshold) &
        (aggregated_markers['log2fc_all'] > 0)
    ].copy()

    return final_markers_df.sort_values(
        by=['group', 'fdr_marker', 'stability_score'],
        ascending=[True, True, False]
    ).reset_index(drop=True)


def get_feature_activity(
    profiles_df: pd.DataFrame,
    fdr_presence_threshold: float = 0.05,
    top_n: Optional[int] = None
) -> Dict[str, List[str]]:
    """
    Summarizes a profile DataFrame to show in which groups features are active.
    """
    required_cols = ['feature_id', 'group', 'fdr_presence', 'norm_score']
    if not all(col in profiles_df.columns for col in required_cols):
        raise ValueError(
            "Input DataFrame is missing required columns. "
            f"Expected: {', '.join(required_cols)}"
        )
        
    active_df = profiles_df[profiles_df['fdr_presence'] <= fdr_presence_threshold].copy()

    if active_df.empty:
        return {}

    active_df.sort_values(by=['feature_id', 'norm_score'], ascending=[True, False], inplace=True)

    grouped = active_df.groupby('feature_id', observed=True)

    if top_n is not None:
        result_series = grouped.head(top_n).groupby('feature_id', observed=True)['group'].apply(list)
    else:
        result_series = grouped['group'].apply(list)

    return result_series.to_dict()

def evaluate_clustering(
    adata: ad.AnnData,
    cluster_key: str,
    use_rep: str = 'X_pca',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Evaluates the quality of clustering using the Silhouette Score.

    This function provides a quantitative measure of how dense and well-separated
    the clusters are. It helps identify ambiguous or poorly-defined clusters
    before proceeding with downstream analysis like marker gene detection.

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
    if use_rep not in adata.obsm:
        raise ValueError(f"Representation '{use_rep}' not found in adata.obsm.")
    if cluster_key not in adata.obs:
        raise ValueError(f"Cluster key '{cluster_key}' not found in adata.obs.")

    # Calculate silhouette score for each cell
    X = adata.obsm[use_rep]
    labels = adata.obs[cluster_key]
    
    # Add per-cell scores to adata for detailed inspection
    adata.obs[f'silhouette_{cluster_key}'] = silhouette_samples(X, labels)
    
    # Calculate overall average score
    overall_score = adata.obs[f'silhouette_{cluster_key}'].mean()
    
    # Calculate per-cluster average scores
    cluster_scores = adata.obs.groupby(cluster_key, observed=True)[f'silhouette_{cluster_key}'].mean()
    cluster_sizes = adata.obs[cluster_key].value_counts()
    
    # Create a summary DataFrame
    report_df = pd.DataFrame({
        'avg_silhouette_score': cluster_scores,
        'n_cells': cluster_sizes
    }).sort_values('avg_silhouette_score', ascending=False)

    if verbose:
        print("--- Clustering Quality Report ---")
        print(f"Overall Average Silhouette Score: {overall_score:.3f}\n")
        print("Per-Cluster Scores:")
        print(report_df)
        print("\nInterpretation Guide:")
        print("  > 0.5: Strong, well-separated cluster.")
        print("  > 0.25: Reasonable cluster structure.")
        print("  < 0.1: Weak or overlapping structure. Interpret markers with caution.")
        print("  < 0: Potential misclassifications.")
        print("---------------------------------")
        
    return report_df

def select_robust_markers(
    ranked_markers_df: pd.DataFrame,
    top_n: int = 10,
    fdr_threshold: float = 0.05,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Automatically selects the top N robust markers per group using a dynamic,
    data-driven clustering approach.

    This function calculates a composite score for each potential marker and then
    uses K-Means clustering (k=2) to find a natural cutoff between "good" and
    "exceptional" markers, selecting from the exceptional group.

    Args:
        ranked_markers_df (pd.DataFrame): The DataFrame output from `find_marker_features`.
        top_n (int): The number of top markers to select for each group.
        fdr_threshold (float): Initial filtering step for statistical significance.
        verbose (bool): If True, prints the dynamically determined threshold.

    Returns:
        pd.DataFrame: A filtered and sorted DataFrame containing the top N robust
            markers for each group, with the new 'marker_score' column.
    """
    if ranked_markers_df.empty:
        return pd.DataFrame()

    # Step 1: Baseline filtering on significance
    candidates = ranked_markers_df[ranked_markers_df['fdr_marker'] < fdr_threshold].copy()
    if len(candidates) < 2: # Need at least 2 points to cluster
        if verbose:
            print(f"Warning: Not enough markers ({len(candidates)}) passed the initial FDR threshold of {fdr_threshold} to perform dynamic selection.")
        return candidates

    # Step 2: Create the composite marker score
    spec_col = next((col for col in candidates.columns if 'specificity_' in col), None)
    if not spec_col:
        raise ValueError("Could not find a specificity column in the DataFrame.")
        
    metrics_to_scale = ['log2fc_all', spec_col, 'pct_expressing']
    if 'stability_score' in candidates.columns:
        metrics_to_scale.append('stability_score')

    for metric in metrics_to_scale:
        candidates[f'scaled_{metric}'] = minmax_scale(candidates[metric])
    
    candidates['marker_score'] = candidates[[f'scaled_{m}' for m in metrics_to_scale]].sum(axis=1)

    # Step 3: Use K-Means to find the natural threshold in the marker_score
    scores_for_clustering = candidates[['marker_score']].values
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(scores_for_clustering)
    
    # Identify which cluster label corresponds to "exceptional" markers
    cluster_centers = kmeans.cluster_centers_
    exceptional_cluster_label = np.argmax(cluster_centers)
    
    # The threshold is the minimum score in the exceptional cluster
    exceptional_markers = candidates[kmeans.labels_ == exceptional_cluster_label]
    dynamic_threshold = exceptional_markers['marker_score'].min()

    if verbose:
        print("--- Dynamic Marker Selection via Clustering ---")
        print(f"  - Clustered {len(candidates)} candidate markers into two groups.")
        print(f"  - Identified {len(exceptional_markers)} as 'exceptional'.")
        print(f"  - Learned marker_score threshold: {dynamic_threshold:.3f}")
        print("---------------------------------------------")

    # Step 4: Select all markers above the learned threshold
    final_candidates = candidates[candidates['marker_score'] >= dynamic_threshold].copy()
    
    # Step 5: Rank within each group and select top N
    final_candidates.sort_values(by=['group', 'marker_score'], ascending=[True, False], inplace=True)
    
    top_markers_df = final_candidates.groupby('group').head(top_n).reset_index(drop=True)

    return top_markers_df