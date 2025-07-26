#!/usr/bin/env python

"""
Public API for SingleCellFeatureProfiler.

This module provides the main user-facing functions for feature profiling
and marker discovery.
"""

from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from ._utils import _prepare_and_validate_inputs
from ._engine import _run_profiling_engine
from ._selection_hvg import select_hvg_features
from ._selection_marker import select_marker_candidates

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None


def get_feature_profiles(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    features: Optional[List[str]] = None,
    all_features: bool = False,
    n_hvg: int = 3000,
    feature_names: Optional[List[str]] = None,
    batch_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Provides a complete statistical profile for features across all groups.

    By default, if no specific feature list is provided, this function will
    select the top N highly variable genes (HVGs) for profiling.
    """
    expression_matrix, all_f_names, group_labels, batch_labels = _prepare_and_validate_inputs(
        data=data,
        group_by=group_by,
        feature_names=feature_names,
        batch_by=batch_by
    )

    if features is not None:
        if verbose:
            print(f"Profiling {len(features)} user-provided features.")
        missing = [f for f in features if f not in all_f_names]
        if missing:
            raise ValueError(f"The following features were not found in the data: {missing}")
        features_to_analyze = features
    elif all_features:
        if verbose:
            print(f"Warning: Profiling all {len(all_f_names)} features. This may be slow and memory-intensive.")
        features_to_analyze = all_f_names
    else:
        features_to_analyze = select_hvg_features(
            data, 
            feature_names=feature_names, 
            n_top_features=n_hvg, 
            verbose=verbose
        )

    if not features_to_analyze:
        print("Warning: No features to analyze. Returning empty DataFrame.")
        return pd.DataFrame()

    results_df = _run_profiling_engine(
        expression_data=data if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else expression_matrix,
        features_to_analyze=features_to_analyze,
        all_feature_names=all_f_names,
        group_labels=group_labels,
        batch_labels=batch_labels,
        specificity_metric=specificity_metric,
        background_rate=background_rate,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    return results_df


def find_marker_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
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
) -> Dict[str, List[str]]:
    """
    Finds significant marker features for each group in the data.

    This function uses a dedicated "smart selection" pipeline to find candidate
    markers before profiling.
    """
    if verbose:
        print("Finding marker features using a data-driven marker selection pipeline...")

    # Get batch_by from kwargs if present, for both validation and engine calls
    batch_by = kwargs.get("batch_by")
    
    expression_matrix, all_f_names, group_labels, batch_labels = _prepare_and_validate_inputs(
        data, group_by=group_by, batch_by=batch_by
    )

    # Use None for feature_names if data is AnnData to avoid validation error
    candidate_features = select_marker_candidates(
        data, 
        feature_names=None if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else all_f_names,
        min_freq=min_freq,
        max_freq=max_freq,
        var_mean_ratio_min=var_mean_ratio_min,
        gap_stat_min=gap_stat_min,
        right_tail_min=right_tail_min,
        cv_min=cv_min,
        verbose=verbose
    )

    if not candidate_features:
        if verbose:
            print("Warning: No candidate features found after selection. Returning empty dictionary.")
        return {str(group): [] for group in np.unique(group_labels)}

    # Run the engine on the selected candidates
    all_profiles_df = _run_profiling_engine(
        expression_data=data if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else expression_matrix, 
        features_to_analyze=candidate_features,
        all_feature_names=all_f_names,
        group_labels=group_labels,
        batch_labels=batch_labels,
        verbose=verbose,
        **kwargs
    )
    
    if all_profiles_df.empty:
        return {str(group): [] for group in np.unique(group_labels)}

    specificity_col = [c for c in all_profiles_df.columns if 'specificity' in c][0]
    potential_markers_df = all_profiles_df[
        (all_profiles_df[specificity_col] >= specificity_threshold) &
        (all_profiles_df['pct_expressing'] >= min_pct_expressing) &
        (all_profiles_df['fdr_marker'] <= fdr_marker_threshold) &
        (all_profiles_df['log2fc_all'] > 0)
    ].copy()
    
    if potential_markers_df.empty:
        all_groups = all_profiles_df['group'].unique()
        return {str(group): [] for group in all_groups}

    max_score_indices = potential_markers_df.groupby('feature_id')['norm_score'].idxmax()
    valid_indices = max_score_indices.dropna()
    best_group_indices = potential_markers_df.loc[valid_indices]
    
    marker_dict = best_group_indices.groupby('group')['feature_id'].apply(list).to_dict()

    all_groups = all_profiles_df['group'].unique()
    for group in all_groups:
        group_str = str(group)
        if group_str not in marker_dict:
            marker_dict[group_str] = []
        else:
            if group_str != group:
                marker_dict[group_str] = marker_dict.pop(group)
            marker_dict[group_str].sort()
            
    return marker_dict


def get_feature_activity(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    features: List[str],
    fdr_presence_threshold: float = 0.05,
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    For a given list of features, determines in which groups they are "ON".
    """
    profiles_df = get_feature_profiles(
        data=data, group_by=group_by, features=features, verbose=verbose, **kwargs
    )
    
    if profiles_df.empty:
        return pd.DataFrame()

    activity_df = profiles_df[profiles_df['fdr_presence'] <= fdr_presence_threshold].copy()
    
    return activity_df[['feature_id', 'group', 'norm_score', 'pct_expressing', 'fdr_presence']]