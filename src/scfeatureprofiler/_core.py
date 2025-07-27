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
    """Calculates a specificity score (Tau or Gini)."""
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
    condition_vector: Optional[np.ndarray] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01
) -> pd.DataFrame:
    """Performs a full statistical analysis for a single feature."""
    df = pd.DataFrame({'expression': expression_vector, 'group': labels_vector})
    
    if condition_vector is not None:
        df['condition'] = condition_vector
        group_keys = ['group', 'condition']
    else:
        group_keys = ['group']
    
    grouped = df.groupby(group_keys, observed=True)
    
    per_condition_stats = grouped['expression'].agg(
        n_cells='size',
        n_expressing=lambda x: (x > 0).sum(),
        mean_all='mean',
        mean_expressing=lambda x: x[x > 0].mean(),
        median_expressing=lambda x: x[x > 0].median()
    ).reset_index().fillna(0)
    
    per_condition_stats['pct_expressing'] = (per_condition_stats['n_expressing'] / per_condition_stats['n_cells']) * 100
    
    per_condition_stats['p_val_presence'] = per_condition_stats.apply(
        lambda row: binomtest(
            k=int(row['n_expressing']), n=int(row['n_cells']),
            p=background_rate, alternative='greater'
        ).pvalue,
        axis=1
    )

    group_level_stats = []
    unique_groups = df['group'].unique()
    
    agg_pct = per_condition_stats.groupby('group')['pct_expressing'].mean().reset_index()
    agg_pct['raw_score_proxy'] = agg_pct['pct_expressing']
    
    all_scores = agg_pct.set_index('group')['raw_score_proxy']
    max_score, min_score = all_scores.max(), all_scores.min()
    if max_score > min_score:
        norm_scores = (all_scores - min_score) / (max_score - min_score)
    else:
        norm_scores = pd.Series(0.0, index=all_scores.index)
    
    specificity = _calculate_specificity(all_scores, metric=specificity_metric)
    
    for group in unique_groups:
        mask_group = (labels_vector == group)
        mask_other = (labels_vector != group)
        
        expr_group = expression_vector[mask_group]
        expr_other = expression_vector[mask_other]
        
        p_val_marker = ranksums(expr_group, expr_other, alternative='greater').pvalue if len(expr_group) > 0 and len(expr_other) > 0 else 1.0
        
        mean_group_all = np.mean(expr_group)
        mean_other_all = np.mean(expr_other) if len(expr_other) > 0 else 0
        log2fc_all = np.log2((mean_group_all + 1e-9) / (mean_other_all + 1e-9))
        
        group_level_stats.append({
            'group': group,
            'p_val_marker': p_val_marker,
            'log2fc_all': log2fc_all,
            'norm_score': norm_scores.get(group, 0.0),
            f'specificity_{specificity_metric}': specificity
        })
    
    group_level_df = pd.DataFrame(group_level_stats)

    final_df = pd.merge(per_condition_stats, group_level_df, on='group')
    final_df['feature_id'] = feature_name
    
    if 'condition' not in final_df.columns:
        final_df['condition'] = 'all'
    
    # Return the full per-condition table without FDR columns
    return final_df