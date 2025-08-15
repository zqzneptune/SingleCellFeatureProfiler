#!/usr/bin/env python

"""
Internal core engine for single-feature statistical calculations.
This module is optimized for performance using NumPy and vectorized operations.
"""

from typing import Optional
import warnings

import numpy as np
import pandas as pd
# --- REFACTORED: Import binom instead of binomtest ---
from scipy.stats import binom, ranksums

warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")

def _calculate_specificity(scores: np.ndarray, metric: str) -> float:
    """Calculates a specificity score on a numpy array."""
    if scores.size <= 1:
        return 1.0
    max_score = scores.max()
    if max_score <= 0:
        return 1.0
    if metric == 'tau':
        normalized_scores = scores / max_score
        return np.sum(1 - normalized_scores) / (scores.size - 1)
    elif metric == 'gini':
        sorted_scores = np.sort(scores)
        n = scores.size
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
    """
    Performs a full statistical analysis for a single feature using vectorized operations.
    """
    # --- 1. Setup Indices and Initial DataFrame ---
    unique_groups, group_indices = np.unique(labels_vector, return_inverse=True)
    n_groups = len(unique_groups)
    is_expressing_mask = expression_vector > 0

    if condition_vector is not None:
        unique_conditions, cond_indices = np.unique(condition_vector, return_inverse=True)
        n_conditions = len(unique_conditions)
        
        combined_idx = group_indices * n_conditions + cond_indices
        unique_pairs, pair_indices = np.unique(combined_idx, return_inverse=True)
        n_items = len(unique_pairs)
        
        group_map = np.empty(n_items, dtype=group_indices.dtype)
        cond_map = np.empty(n_items, dtype=cond_indices.dtype)
        for i, pair_val in enumerate(unique_pairs):
            group_map[i] = pair_val // n_conditions
            cond_map[i] = pair_val % n_conditions
        
        indices_to_use = pair_indices
        per_condition_stats = pd.DataFrame({
            'group': unique_groups[group_map],
            'condition': unique_conditions[cond_map],
        })
    else:
        indices_to_use = group_indices
        n_items = n_groups
        per_condition_stats = pd.DataFrame({'group': unique_groups})

    # --- 2. Vectorized Primary Calculations ---
    n_cells = np.bincount(indices_to_use, minlength=n_items).astype(np.int64)
    n_expressing = np.bincount(indices_to_use, weights=is_expressing_mask, minlength=n_items).astype(np.int64)
    sum_expr = np.bincount(indices_to_use, weights=expression_vector, minlength=n_items)
    
    per_condition_stats['n_cells'] = n_cells
    per_condition_stats['n_expressing'] = n_expressing
    per_condition_stats['mean_all'] = np.divide(sum_expr, n_cells, out=np.zeros_like(sum_expr, dtype=float), where=n_cells!=0)
    
    # --- 3. Vectorized Mean/Median Expressing Calculation ---
    if np.any(is_expressing_mask):
        expr_df = pd.DataFrame({
            'expr': expression_vector[is_expressing_mask],
            'idx': indices_to_use[is_expressing_mask]
        })
        grouped_stats = expr_df.groupby('idx')['expr'].agg(['mean', 'median'])
        grouped_stats = grouped_stats.reindex(range(n_items), fill_value=0.0)
        
        per_condition_stats['mean_expressing'] = grouped_stats['mean'].values
        per_condition_stats['median_expressing'] = grouped_stats['median'].values
    else:
        per_condition_stats['mean_expressing'] = 0.0
        per_condition_stats['median_expressing'] = 0.0
    
    # --- 4. Remaining per-condition calculations ---
    # Using safe division for percentage calculation
    pct_expr = np.divide(n_expressing * 100, n_cells, out=np.zeros_like(n_expressing, dtype=float), where=n_cells!=0)
    per_condition_stats['pct_expressing'] = pct_expr
    
    # --- REFACTORED: Vectorized Binomial Test for presence p-value ---
    # The survival function sf(k, n, p) is P(X > k). 
    # The 'greater' alternative in binomtest is P(X >= k), which is equivalent to P(X > k-1).
    per_condition_stats['p_val_presence'] = binom.sf(
        k=n_expressing - 1,
        n=n_cells,
        p=background_rate
    )

    # --- 5. Calculate per-group (cross-condition) statistics ---
    group_level_stats = []
    agg_pct = per_condition_stats.groupby('group')['pct_expressing'].mean()
    
    all_scores = agg_pct.values
    max_score, min_score = all_scores.max(), all_scores.min()
    norm_scores_vals = (all_scores - min_score) / (max_score - min_score) if max_score > min_score else np.zeros_like(all_scores)
    norm_scores_map = dict(zip(agg_pct.index, norm_scores_vals))
    specificity = _calculate_specificity(all_scores, metric=specificity_metric)
    
    for i, group in enumerate(unique_groups):
        mask_group = (group_indices == i)
        expr_group = expression_vector[mask_group]
        expr_other = expression_vector[~mask_group]
        
        p_val_marker = ranksums(expr_group, expr_other, alternative='greater').pvalue if expr_group.size > 0 and expr_other.size > 0 else 1.0
        
        mean_group_all = np.mean(expr_group)
        mean_other_all = np.mean(expr_other) if expr_other.size > 0 else 0
        log2fc_all = np.log2((mean_group_all + 1e-9) / (mean_other_all + 1e-9))
        
        group_level_stats.append({
            'group': group,
            'p_val_marker': p_val_marker,
            'log2fc_all': log2fc_all,
            'norm_score': norm_scores_map.get(group, 0.0),
            f'specificity_{specificity_metric}': specificity
        })
    
    group_level_df = pd.DataFrame(group_level_stats)

    # --- 6. Merge and Finalize ---
    final_df = pd.merge(per_condition_stats, group_level_df, on='group')
    final_df['feature_id'] = feature_name
    
    if 'condition' not in final_df.columns:
        final_df['condition'] = 'all'

    return final_df