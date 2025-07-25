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

from ._utils import _prepare_and_validate_inputs, get_active_features
from ._engine import _run_profiling_engine

# AnnData is an optional dependency
try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None


def get_feature_profiles(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    features: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
    batch_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01,
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Provides a complete statistical profile for features across all groups.

    This is the main "power-user" function that returns all calculated statistics,
    including normalized expression scores, percentage of expressing cells,
    specificity scores, and significance (FDR) for presence and marker status.

    Parameters
    ----------
    data : AnnData, pd.DataFrame, np.ndarray, or spmatrix
        The single-cell expression matrix (cells x features).
    group_by : str or list-like
        The group labels for each cell (e.g., cell types, clusters).
    features : list of str, optional
        A specific list of features to profile. If None, the function will
        profile all "active" features as determined by `get_active_features`.
        By default None.
    feature_names : list of str, optional
        List of all feature names. Required if `data` is a numpy array or
        sparse matrix. By default None.
    batch_by : str or list-like, optional
        Batch labels for each cell (e.g., donor IDs). If provided, statistics
        are averaged across batches within each group. By default None.
    specificity_metric : str, optional
        The metric for calculating feature specificity ('tau' or 'gini').
        By default 'tau'.
    background_rate : float, optional
        Assumed background expression rate for the presence test. By default 0.01.
    n_jobs : int, optional
        Number of parallel jobs to run. -1 means using all available processors.
        By default -1.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the full statistical profile for the requested features.
    """
    # 1. Validate and prepare all inputs
    expression_matrix, all_f_names, group_labels, batch_labels = _prepare_and_validate_inputs(
        data=data,
        group_by=group_by,
        feature_names=feature_names,
        batch_by=batch_by
    )

    # 2. Determine which features to analyze
    if features is None:
        print("`features` not provided. Identifying active features to profile...")
        # Use the original data object for get_active_features
        features_to_analyze = get_active_features(data, feature_names=feature_names)
        if not features_to_analyze:
            print("Warning: No active features found. Returning empty DataFrame.")
            return pd.DataFrame()
        print(f"Found {len(features_to_analyze)} active features.")
    else:
        # Validate that user-provided features exist in the data
        missing = [f for f in features if f not in all_f_names]
        if missing:
            raise ValueError(f"The following features were not found in the data: {missing}")
        features_to_analyze = features

    # 3. Run the profiling engine
    results_df = _run_profiling_engine(
        expression_data=data if (ANNDATA_AVAILABLE and isinstance(data, AnnData)) else expression_matrix,
        features_to_analyze=features_to_analyze,
        all_feature_names=all_f_names,
        group_labels=group_labels,
        batch_labels=batch_labels,
        specificity_metric=specificity_metric,
        background_rate=background_rate,
        n_jobs=n_jobs
    )
    
    return results_df


def get_feature_activity(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    features: List[str],
    fdr_presence_threshold: float = 0.05,
    **kwargs
) -> pd.DataFrame:
    """
    For a given list of features, determines in which groups they are "ON".

    A feature is considered "ON" in a group if its presence is statistically
    significant (i.e., `fdr_presence` is below the specified threshold).

    Parameters
    ----------
    data : AnnData, pd.DataFrame, np.ndarray, or spmatrix
        The single-cell expression matrix (cells x features).
    group_by : str or list-like
        The group labels for each cell.
    features : list of str
        A specific list of features to check for activity.
    fdr_presence_threshold : float, optional
        The FDR threshold for the presence test. By default 0.05.
    **kwargs
        Additional arguments to pass to `get_feature_profiles` (e.g., `batch_by`, `n_jobs`).

    Returns
    -------
    pd.DataFrame
        A filtered DataFrame showing only the feature-group pairs where the
        feature is considered "ON".
    """
    # Run the full profiling for the given features
    profiles_df = get_feature_profiles(data=data, group_by=group_by, features=features, **kwargs)
    
    if profiles_df.empty:
        return pd.DataFrame()

    # Filter based on the "ON" criterion
    activity_df = profiles_df[profiles_df['fdr_presence'] <= fdr_presence_threshold].copy()
    
    # Return a clean summary
    return activity_df[['feature_id', 'group', 'norm_score', 'pct_expressing', 'fdr_presence']]


# This is the clean, permanent version of the function
def find_marker_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    specificity_threshold: float = 0.7,
    min_pct_expressing: float = 10.0,
    fdr_marker_threshold: float = 0.05,
    **kwargs
) -> Dict[str, List[str]]:
    """
    Finds significant marker features for each group in the data.

    This high-level function identifies markers by first profiling all active
    features and then filtering them based on specificity, expression percentage,
    and marker significance.

    Parameters
    ----------
    (docstring continues...)
    """
    all_profiles_df = get_feature_profiles(data=data, group_by=group_by, features=None, **kwargs)
    
    if all_profiles_df.empty:
        _, _, group_labels, _ = _prepare_and_validate_inputs(data, group_by)
        return {group: [] for group in np.unique(group_labels)}

    specificity_col = [c for c in all_profiles_df.columns if 'specificity' in c][0]
    potential_markers_df = all_profiles_df[
        (all_profiles_df[specificity_col] >= specificity_threshold) &
        (all_profiles_df['pct_expressing'] >= min_pct_expressing) &
        (all_profiles_df['fdr_marker'] <= fdr_marker_threshold) &
        (all_profiles_df['log2fc_marker'] > 0)
    ].copy()
    
    if potential_markers_df.empty:
        all_groups = all_profiles_df['group'].unique()
        return {group: [] for group in all_groups}

    max_score_indices = potential_markers_df.groupby('feature_id')['norm_score'].idxmax()
    valid_indices = max_score_indices.dropna()
    best_group_indices = potential_markers_df.loc[valid_indices]
    
    marker_dict = best_group_indices.groupby('group')['feature_id'].apply(list).to_dict()

    all_groups = all_profiles_df['group'].unique()
    for group in all_groups:
        if group not in marker_dict:
            marker_dict[group] = []
        else:
            marker_dict[group].sort()
            
    return marker_dict