#!/usr/bin/env python

"""
Internal core engine for single-feature statistical calculations.

This module provides the fundamental function, _analyze_one_feature, which
is the computational core of the package.
"""

from typing import Optional, Dict, List
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ranksums

# Suppress common warnings from numpy/scipy that we handle explicitly
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in log")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")


# In scfeatureprofiler/_core.py

def _calculate_specificity(scores: pd.Series, metric: str) -> float:
    """
    Calculates a specificity score (Tau or Gini).
    """
    # --- FIX: Handle single group edge case ---
    if len(scores) <= 1:
        return 1.0  # By convention, specificity is max if only one group exists

    if scores.max() <= 0:
        return 1.0  # Convention for non-expressed features

    if metric == 'tau':
        # (The rest of the function is unchanged)
        normalized_scores = scores / scores.max()
        return (1 - normalized_scores).sum() / (len(scores) - 1)
    
    elif metric == 'gini':
        sorted_scores = np.sort(scores)
        n = len(scores)
        cumx = np.cumsum(sorted_scores, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
        
    else:
        raise ValueError(f"Unknown specificity metric: {metric}. Use 'tau' or 'gini'.")

def _geometric_mean(series: pd.Series) -> float:
    """Helper for geometric mean, safe for .agg()"""
    positive_values = series[series > 0]
    if positive_values.empty:
        return 0.0
    return np.exp(np.log(positive_values).mean())

def _analyze_one_feature(
    expression_vector: np.ndarray,
    labels_vector: np.ndarray,
    feature_name: str,
    batch_vector: Optional[np.ndarray] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01
) -> pd.DataFrame:
    """
    Performs a full statistical analysis for a single feature.

    This function is the core computational engine. It operates on simple
    numpy arrays for maximum performance and portability.

    Parameters
    ----------
    expression_vector : np.ndarray
        A 1D array of expression values for one feature across all cells.
    labels_vector : np.ndarray
        A 1D array of group labels for each cell.
    feature_name : str
        The name of the feature being analyzed.
    batch_vector : np.ndarray, optional
        A 1D array of batch labels for each cell, by default None.
    specificity_metric : str, optional
        The specificity metric to use ('tau' or 'gini'), by default 'tau'.
    background_rate : float, optional
        The assumed background expression rate for the binomial test, by default 0.01.

    Returns
    -------
    pd.DataFrame
        A DataFrame where each row corresponds to a group, containing all
        calculated statistics for the feature. P-values are raw and need
        FDR correction at a higher level.
    """
    # Create a DataFrame for easy grouping and calculation
    df = pd.DataFrame({
        'expression': expression_vector,
        'group': labels_vector
    })
    if batch_vector is not None:
        df['batch'] = batch_vector

    # --- 1. Calculate Base Statistics per Group (and Batch) ---
    group_keys = ['group', 'batch'] if batch_vector is not None else ['group']
    grouped = df.groupby(group_keys, observed=True)

    # Use .agg for efficient computation
    stats = grouped['expression'].agg(
        n_cells='size',
        n_expressing=lambda x: (x > 0).sum(),
        geo_mean_in_expressing=_geometric_mean
    ).reset_index()

    # Calculate derived metrics
    stats['pct_expressing'] = (stats['n_expressing'] / stats['n_cells']) * 100
    stats['geo_mean_in_expressing'] = grouped['expression'].apply(
        lambda x: np.exp(np.log(x[x > 0]).mean()) if (x > 0).any() else 0
    ).values
    
    # Raw score combines fraction and magnitude
    stats['raw_score'] = stats['pct_expressing'] * stats['geo_mean_in_expressing']
    
    # --- 2. Aggregate across Batches if present ---
    if batch_vector is not None:
        # For scores, we average across batches within a group
        final_stats = stats.groupby('group', observed=True).agg(
            pct_expressing=('pct_expressing', 'mean'),
            raw_score=('raw_score', 'mean')
        ).reset_index()
    else:
        final_stats = stats[['group', 'pct_expressing', 'raw_score']].copy()
    
    # --- 3. Calculate Cross-Group Statistics ---
    all_scores = final_stats.set_index('group')['raw_score']
    
    # a) Normalized Score (0-1 scaling)
    max_score, min_score = all_scores.max(), all_scores.min()
    if max_score > min_score:
        norm_score_series = (all_scores - min_score) / (max_score - min_score)
    else:
        # If all scores are equal, norm_score is 0 for all
        norm_score_series = pd.Series(0.0, index=all_scores.index)

    # Convert the series to a DataFrame for proper merging
    norm_scores_df = norm_score_series.reset_index(name='norm_score')
    final_stats = pd.merge(final_stats, norm_scores_df, on='group')

    # b) Specificity Score
    specificity = _calculate_specificity(all_scores, metric=specificity_metric)
    final_stats[f'specificity_{specificity_metric}'] = specificity

    # --- 4. Perform Statistical Tests ---
    results_list = []
    unique_groups = df['group'].unique()
    
    for group in unique_groups:
        # Prepare data for this group
        mask_group = labels_vector == group
        mask_other = labels_vector != group
        
        expr_group = expression_vector[mask_group]
        
        # a) Presence Test (Binomial)
        n_total_in_group = len(expr_group)
        n_expressing_in_group = (expr_group > 0).sum()
        binom_res = binomtest(
            k=n_expressing_in_group,
            n=n_total_in_group,
            p=background_rate,
            alternative='greater'
        )
        
        # b) Marker Test (Wilcoxon Rank-Sum)
        expr_other = expression_vector[mask_other]
        # Only run test if both groups have data
        if len(expr_group) > 0 and len(expr_other) > 0:
            wilcox_res = ranksums(expr_group, expr_other, alternative='greater')
            p_val_marker = wilcox_res.pvalue
        else:
            p_val_marker = 1.0 # Not meaningful if one group is empty
        
        # c) Log2 Fold Change
        mean_group = np.mean(expr_group) if len(expr_group) > 0 else 0
        mean_other = np.mean(expr_other) if len(expr_other) > 0 else 0
        # Add a pseudocount for stability, common in scRNA-seq
        log2fc = np.log2((mean_group + 1e-9) / (mean_other + 1e-9))
        
        results_list.append({
            'group': group,
            'p_val_presence': binom_res.pvalue,
            'p_val_marker': p_val_marker,
            'log2fc_marker': log2fc
        })

    # --- 5. Combine all results ---
    pvals_df = pd.DataFrame(results_list)
    # Now merge into the corrected final_stats DataFrame
    final_df = pd.merge(final_stats, pvals_df, on='group')
    final_df['feature_id'] = feature_name
    
    # Drop raw_score and reorder columns
    final_df = final_df.drop(columns=['raw_score'])
    final_cols = [
        'feature_id', 'group', 'norm_score', 'pct_expressing',
        'p_val_presence', 'p_val_marker', 'log2fc_marker',
        f'specificity_{specificity_metric}'
    ]
    return final_df[final_cols]