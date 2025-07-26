#!/usr/bin/env python

"""
Internal core engine for single-feature statistical calculations.
"""

from typing import Optional
import warnings

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ranksums

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


def _geometric_mean(series: pd.Series) -> float:
    """Helper for geometric mean, safe for .agg()"""
    positive_values = series[series > 0]
    if positive_values.empty:
        return 0.0
    return np.exp(np.log(positive_values).mean())


def _calculate_specificity(scores: pd.Series, metric: str) -> float:
    # ... (no changes to this function)
    if len(scores) <= 1:
        return 1.0
    if scores.max() <= 0:
        return 1.0
    if metric == 'tau':
        normalized_scores = scores / scores.max()
        return (1 - normalized_scores).sum() / (len(scores) - 1)
    elif metric == 'gini':
        sorted_scores = np.sort(scores)
        n = len(scores)
        cumx = np.cumsum(sorted_scores, dtype=float)
        return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n
    else:
        raise ValueError(f"Unknown specificity metric: {metric}. Use 'tau' or 'gini'.")


def _analyze_one_feature(
    expression_vector: np.ndarray,
    labels_vector: np.ndarray,
    feature_name: str,
    batch_vector: Optional[np.ndarray] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01
) -> pd.DataFrame:
    """Performs a full statistical analysis for a single feature."""
    df = pd.DataFrame({'expression': expression_vector, 'group': labels_vector})
    if batch_vector is not None:
        df['batch'] = batch_vector

    group_keys = ['group', 'batch'] if batch_vector is not None else ['group']
    grouped = df.groupby(group_keys, observed=True)

    # --- ENHANCED: Add more aggregations ---
    stats = grouped['expression'].agg(
        n_cells='size',
        n_expressing=lambda x: (x > 0).sum(),
        mean_all='mean',
        mean_expressing=lambda x: x[x > 0].mean(),
        median_expressing=lambda x: x[x > 0].median(),
        geo_mean_in_expressing=_geometric_mean
    ).reset_index()

    stats['pct_expressing'] = (stats['n_expressing'] / stats['n_cells']) * 100
    stats['raw_score'] = stats['pct_expressing'] * stats['geo_mean_in_expressing']
    
    # Fill NaNs that result from empty groups in agg (e.g., median_expressing of zero values)
    stats = stats.fillna(0)
    
    if batch_vector is not None:
        # Average all numeric columns except n_cells and n_expressing
        agg_cols = {
            'pct_expressing': 'mean', 'raw_score': 'mean', 'mean_all': 'mean',
            'mean_expressing': 'mean', 'median_expressing': 'mean'
        }
        final_stats = stats.groupby('group', observed=True).agg(agg_cols).reset_index()
    else:
        final_stats = stats.copy()

    all_scores = final_stats.set_index('group')['raw_score']
    
    max_score, min_score = all_scores.max(), all_scores.min()
    if max_score > min_score:
        norm_score_series = (all_scores - min_score) / (max_score - min_score)
    else:
        norm_score_series = pd.Series(0.0, index=all_scores.index)

    norm_scores_df = norm_score_series.reset_index(name='norm_score')
    final_stats = pd.merge(final_stats, norm_scores_df, on='group')

    specificity = _calculate_specificity(all_scores, metric=specificity_metric)
    final_stats[f'specificity_{specificity_metric}'] = specificity
    
    results_list = []
    unique_groups = df['group'].unique()
    
    # Calculate overall pct_expressing outside the loop for lift calculation
    pct_expressing_overall = (expression_vector > 0).sum() / len(expression_vector)
    
    for group in unique_groups:
        mask_group = labels_vector == group
        mask_other = labels_vector != group
        expr_group = expression_vector[mask_group]
        expr_other = expression_vector[mask_other]
        
        # --- ENHANCED: Add more stats ---
        mean_group_expressing = np.mean(expr_group[expr_group > 0]) if (expr_group > 0).any() else 0
        mean_other_expressing = np.mean(expr_other[expr_other > 0]) if (expr_other > 0).any() else 0
        log2fc_expressing = np.log2((mean_group_expressing + 1e-9) / (mean_other_expressing + 1e-9))
        
        pct_expressing_group = (expr_group > 0).sum() / len(expr_group)
        pct_expressing_other = (expr_other > 0).sum() / len(expr_other) if len(expr_other) > 0 else 0
        pct_expressing_lift = (pct_expressing_group + 1e-9) / (pct_expressing_other + 1e-9)
        
        n_expressing_in_group = (expr_group > 0).sum()
        binom_res = binomtest(k=n_expressing_in_group, n=len(expr_group), p=background_rate, alternative='greater')
        
        p_val_marker = ranksums(expr_group, expr_other, alternative='greater').pvalue if len(expr_group) > 0 and len(expr_other) > 0 else 1.0
        
        mean_group_all = np.mean(expr_group)
        mean_other_all = np.mean(expr_other)
        log2fc_all = np.log2((mean_group_all + 1e-9) / (mean_other_all + 1e-9))
        
        results_list.append({
            'group': group,
            'p_val_presence': binom_res.pvalue,
            'p_val_marker': p_val_marker,
            'log2fc': log2fc_all,
            'log2fc_expressing': log2fc_expressing,
            'pct_expressing_lift': pct_expressing_lift
        })

    pvals_df = pd.DataFrame(results_list)
    final_df = pd.merge(final_stats, pvals_df, on='group')
    final_df['feature_id'] = feature_name
    
    final_df = final_df.drop(columns=['raw_score'])
    
    # --- ENHANCED: Add more columns to final output ---
    final_cols = [
        'feature_id', 'group', 'norm_score', 'pct_expressing', 'mean_all', 'mean_expressing', 'median_expressing',
        'log2fc', 'log2fc_expressing', 'pct_expressing_lift',
        'p_val_presence', 'fdr_presence', # We will add FDR in the engine
        'p_val_marker', 'fdr_marker',     # We will add FDR in the engine
        f'specificity_{specificity_metric}'
    ]
    # We add placeholder columns for FDR, they will be populated in the engine
    final_df['fdr_presence'] = -1.0
    final_df['fdr_marker'] = -1.0
    
    # Return a subset and reorder
    return final_df[[col for col in final_cols if col in final_df.columns]]