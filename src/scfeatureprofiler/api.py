#!/usr/bin/env python

"""
Public API for SingleCellFeatureProfiler.
"""

from typing import List, Optional, Union, Dict
import os

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from ._utils import _prepare_and_validate_inputs
from ._engine import _run_profiling_engine
from ._selection_marker import select_marker_candidates
from ._stability import _calculate_stability_scores

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

    feature_list = None
    if features is not None:
        if isinstance(features, str) and os.path.exists(features):
            if verbose:
                print(f"Loading features from file: {features}")
            with open(features, 'r') as f:
                feature_list = [line.strip() for line in f if line.strip()]
        elif isinstance(features, list):
            feature_list = features
        else:
            raise TypeError(f"`features` must be a list of strings or a valid file path, but got {type(features)}")

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

    # The engine now correctly returns the detailed per-condition (or per-group) table
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
        
    # Sorting is now done on the aggregated table from stability, but we can do a simple one here
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

    # Step 1: Get the detailed per-condition profiles from the engine
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

    # Step 2: Aggregate results and calculate stability
    specificity_metric = kwargs.get('specificity_metric', 'tau')
    specificity_col = f'specificity_{specificity_metric}'
    aggregated_markers = _calculate_stability_scores(per_condition_profiles, specificity_col)

    # Step 3: Filter the final aggregated table
    final_markers_df = aggregated_markers[
        (aggregated_markers[specificity_col] >= specificity_threshold) &
        (aggregated_markers['pct_expressing'] >= min_pct_expressing) &
        (aggregated_markers['fdr_marker'] <= fdr_marker_threshold) &
        (aggregated_markers['log2fc_all'] > 0)
    ].copy()

    # Final sort is now handled within _calculate_stability_scores, but we can re-sort just in case
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