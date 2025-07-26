#!/usr/bin/env python

"""
Internal profiling engine for orchestrating parallel calculations.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from joblib import Parallel, delayed, parallel_backend
from statsmodels.stats.multitest import multipletests

from ._core import _analyze_one_feature

logger = logging.getLogger(__name__)

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None

def _log_and_print(msg: str, verbose: bool):
    """Helper to log and optionally print a message."""
    logger.info(msg)
    if verbose:
        print(msg)

def _fdr_correct_per_group(df: pd.DataFrame, pval_col: str, fdr_col: str) -> pd.DataFrame:
    """Applies FDR correction to p-values within each group."""
    return df.groupby('group', observed=True)[pval_col] \
             .transform(lambda x: multipletests(x, method='fdr_bh')[1])


def _worker_function(
    feature_name: str,
    feature_index: int,
    expression_data: Union[np.ndarray, spmatrix, str], # Can be matrix or path
    is_backed: bool,
    **kwargs
) -> pd.DataFrame:
    """
    Unified worker function for both in-memory and backed data.
    """
    if is_backed:
        import anndata
        adata_backed = anndata.read_h5ad(expression_data, backed='r')
        expression_vector = adata_backed.X[:, feature_index]
    else:
        expression_vector = expression_data[:, feature_index]
    
    if hasattr(expression_vector, "toarray"):
        expression_vector = expression_vector.toarray().flatten()
    
    return _analyze_one_feature(
        expression_vector=np.asarray(expression_vector).flatten(),
        feature_name=feature_name,
        **kwargs
    )


def _run_profiling_engine(
    expression_data: Union[np.ndarray, spmatrix, AnnData],
    features_to_analyze: List[str],
    all_feature_names: List[str],
    group_labels: np.ndarray,
    batch_labels: Optional[np.ndarray] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Orchestrates the statistical analysis for a list of features in parallel.
    """
    feature_index_map = {name: i for i, name in enumerate(all_feature_names)}
    
    common_worker_kwargs = {
        'labels_vector': group_labels,
        'batch_vector': batch_labels,
        'specificity_metric': specificity_metric,
        'background_rate': background_rate
    }
    
    is_backed = ANNDATA_AVAILABLE and isinstance(expression_data, AnnData) and expression_data.isbacked
    
    # Unified argument preparation
    if is_backed:
        expr_data_for_worker = expression_data.filename
    elif ANNDATA_AVAILABLE and isinstance(expression_data, AnnData):
        expr_data_for_worker = expression_data.X
    else:
        expr_data_for_worker = expression_data

    # --- Parallel Execution ---
    n_features = len(features_to_analyze)
    msg = f"Profiling {n_features} features using {n_jobs if n_jobs > 0 else 'all available'} CPU cores..."
    _log_and_print(msg, verbose)

    with parallel_backend('loky', n_jobs=n_jobs):
        list_of_results = Parallel()(
            delayed(_worker_function)(
                feature_name=feature,
                feature_index=feature_index_map[feature],
                expression_data=expr_data_for_worker,
                is_backed=is_backed,
                **common_worker_kwargs
            ) for feature in features_to_analyze
        )
    
    _log_and_print("Parallel computation complete. Finalizing results...", verbose)
    
    if not list_of_results:
        return pd.DataFrame()
        
    full_results_df = pd.concat(list_of_results, ignore_index=True)
    
    # --- Finalization and FDR Correction ---
    _log_and_print("Applying FDR correction...", verbose)
    full_results_df['fdr_presence'] = multipletests(full_results_df['p_val_presence'], method='fdr_bh')[1]
    full_results_df['fdr_marker'] = _fdr_correct_per_group(full_results_df, 'p_val_marker', 'fdr_marker')
    
    full_results_df.rename(columns={'log2fc': 'log2fc_all'}, inplace=True)
    
    final_cols = [
        'feature_id', 'group', 'norm_score', 'pct_expressing', 
        'mean_all', 'mean_expressing', 'median_expressing',
        'log2fc_all', 'log2fc_expressing', 'pct_expressing_lift',
        'p_val_presence', 'fdr_presence',
        'p_val_marker', 'fdr_marker',
        f'specificity_{specificity_metric}'
    ]
    
    for col in final_cols:
        if col not in full_results_df.columns:
            full_results_df[col] = np.nan
            
    return full_results_df[final_cols]