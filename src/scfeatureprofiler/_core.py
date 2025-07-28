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

def _geometric_mean(array: np.ndarray) -> float:
    """Helper for geometric mean on a numpy array."""
    positive_values = array[array > 0]
    if positive_values.size == 0:
        return 0.0
    return np.exp(np.log(positive_values).mean())

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
    # --- 1. Calculate per-condition statistics using NumPy ---
    unique_groups, group_indices = np.unique(labels_vector, return_inverse=True)
    n_groups = len(unique_groups)

    if condition_vector is not None:
        unique_conditions, cond_indices = np.unique(condition_vector, return_inverse=True)
        n_conditions = len(unique_conditions)
        
        # Combine group and condition indices for a unique pairing
        combined_idx = group_indices * n_conditions + cond_indices
        unique_pairs, pair_indices = np.unique(combined_idx, return_inverse=True)
        
        # Map back to group and condition
        group_map = np.empty_like(unique_pairs)
        cond_map = np.empty_like(unique_pairs)
        for i, pair_val in enumerate(unique_pairs):
            group_map[i] = pair_val // n_conditions
            cond_map[i] = pair_val % n_conditions
            
        n_pairs = len(unique_pairs)
        
        # Vectorized aggregations using numpy sums on binned data
        n_cells = np.bincount(pair_indices, minlength=n_pairs)
        is_expressing = expression_vector > 0
        n_expressing = np.bincount(pair_indices, weights=is_expressing, minlength=n_pairs)
        sum_expr = np.bincount(pair_indices, weights=expression_vector, minlength=n_pairs)
        
        # --- FIX: Ensure k and n for binomtest are integers ---
        n_cells = n_cells.astype(np.int64)
        n_expressing = n_expressing.astype(np.int64)
        # --- END FIX ---
        
        per_condition_stats = pd.DataFrame({
            'group': unique_groups[group_map],
            'condition': unique_conditions[cond_map],
            'n_cells': n_cells,
            'n_expressing': n_expressing,
            'mean_all': np.divide(sum_expr, n_cells, out=np.zeros_like(sum_expr, dtype=float), where=n_cells!=0)
        })

        # Iterative part for stats that are harder to vectorize
        mean_expressing_list = []
        median_expressing_list = []
        for i in range(n_pairs):
            mask = pair_indices == i
            expr_subset_expressing = expression_vector[mask & is_expressing]
            if expr_subset_expressing.size > 0:
                mean_expressing_list.append(expr_subset_expressing.mean())
                median_expressing_list.append(np.median(expr_subset_expressing))
            else:
                mean_expressing_list.append(0.0)
                median_expressing_list.append(0.0)
        per_condition_stats['mean_expressing'] = mean_expressing_list
        per_condition_stats['median_expressing'] = median_expressing_list
        
    else: # No condition vector
        n_cells = np.bincount(group_indices)
        is_expressing = expression_vector > 0
        n_expressing = np.bincount(group_indices, weights=is_expressing)
        sum_expr = np.bincount(group_indices, weights=expression_vector)
        
        # --- FIX: Ensure k and n for binomtest are integers ---
        n_cells = n_cells.astype(np.int64)
        n_expressing = n_expressing.astype(np.int64)
        # --- END FIX ---
        
        per_condition_stats = pd.DataFrame({
            'group': unique_groups,
            'n_cells': n_cells,
            'n_expressing': n_expressing,
            'mean_all': np.divide(sum_expr, n_cells, out=np.zeros_like(sum_expr, dtype=float), where=n_cells!=0)
        })
        # ... (similar iterative block for mean/median expressing)
        mean_expressing_list = [expression_vector[(group_indices == i) & is_expressing].mean() if n_expressing[i] > 0 else 0 for i in range(n_groups)]
        median_expressing_list = [np.median(expression_vector[(group_indices == i) & is_expressing]) if n_expressing[i] > 0 else 0 for i in range(n_groups)]
        per_condition_stats['mean_expressing'] = mean_expressing_list
        per_condition_stats['median_expressing'] = median_expressing_list

    per_condition_stats['pct_expressing'] = (per_condition_stats['n_expressing'] / per_condition_stats['n_cells']) * 100
    
    # Vectorized binomial test using scipy.stats
    # --- FIX: Use .apply() for the binomial test to ensure scalar integer inputs ---
    per_condition_stats['p_val_presence'] = per_condition_stats.apply(
        lambda row: binomtest(
            k=int(row['n_expressing']), 
            n=int(row['n_cells']),
            p=background_rate, 
            alternative='greater'
        ).pvalue,
        axis=1
    )
    # --- END FIX ---

    # --- 2. Calculate per-group (cross-condition) statistics ---
    group_level_stats = []
    
    # Aggregate per-condition stats to get overall pct_expressing for scoring
    agg_pct = per_condition_stats.groupby('group')['pct_expressing'].mean()
    
    # Calculate norm_score and specificity based on the aggregated scores
    all_scores = agg_pct.values
    max_score, min_score = all_scores.max(), all_scores.min()
    if max_score > min_score:
        norm_scores_vals = (all_scores - min_score) / (max_score - min_score)
    else:
        norm_scores_vals = np.zeros_like(all_scores)
    
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

    # --- 3. Merge and Finalize ---
    final_df = pd.merge(per_condition_stats, group_level_df, on='group')
    final_df['feature_id'] = feature_name
    
    if 'condition' not in final_df.columns:
        final_df['condition'] = 'all'

    return final_df