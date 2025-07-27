#!/usr/bin/env python

"""
Internal module for calculating marker stability scores.
"""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

def _calculate_stability_scores(
    per_condition_results: pd.DataFrame,
    specificity_metric_col: str
) -> pd.DataFrame:
    """
    Calculates aggregated statistics and a marker stability score from a
    full per-condition results table.
    """
    if 'condition' not in per_condition_results.columns:
        per_condition_results['stability_score'] = 1.0
        return per_condition_results

    n_conditions = per_condition_results['condition'].nunique()

    # Define aggregation rules for each column
    agg_rules = {
        'norm_score': 'mean',
        'pct_expressing': 'mean',
        'mean_all': 'mean',
        'mean_expressing': 'mean',
        'median_expressing': 'mean',
        'log2fc_all': 'first',
        'p_val_presence': 'mean', # Aggregate the per-condition p-values
        'fdr_presence': 'mean',   # Aggregate the per-condition FDRs
        'p_val_marker': 'first',
        specificity_metric_col: 'first'
    }
    
    agg_results = per_condition_results.groupby(['feature_id', 'group']).agg(agg_rules).reset_index()

    if n_conditions > 1:
        cv_scores = per_condition_results.groupby(['feature_id', 'group'])['pct_expressing'].agg(
            mean_val='mean',
            std_val='std'
        ).reset_index()
        
        with np.errstate(divide='ignore', invalid='ignore'):
            cv_scores['cv'] = np.nan_to_num(cv_scores['std_val'] / cv_scores['mean_val'])
        
        cv_scores['stability_score'] = 1 - cv_scores['cv']
        
        final_results = pd.merge(
            agg_results,
            cv_scores[['feature_id', 'group', 'stability_score']],
            on=['feature_id', 'group']
        )
    else:
        final_results = agg_results
        final_results['stability_score'] = 1.0

    # Calculate final marker FDR on the aggregated table
    final_results['fdr_marker'] = final_results.groupby('group')['p_val_marker'].transform(lambda x: multipletests(x, method='fdr_bh')[1])

    final_cols = [
        'feature_id', 'group', 'stability_score', 'norm_score', 'pct_expressing', 
        'mean_all', 'log2fc_all',
        'p_val_presence', 'fdr_presence',
        'p_val_marker', 'fdr_marker',
        specificity_metric_col
    ]
    
    for col in final_cols:
        if col not in final_results.columns:
            final_results[col] = np.nan
            
    return final_results[final_cols]