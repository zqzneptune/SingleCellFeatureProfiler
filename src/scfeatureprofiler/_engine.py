#!/usr/bin/env python

"""
Internal profiling engine for orchestrating parallel calculations.

This module contains the _run_profiling_engine function, which manages
the parallel execution of the core statistical analysis over many features
and handles both in-memory and on-disk (backed) data representations.
"""

from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from joblib import Parallel, delayed
from statsmodels.stats.multitest import multipletests

from ._core import _analyze_one_feature

# AnnData is an optional dependency
try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None
    ANNDATA_AVAILABLE = False


def _fdr_correct_per_group(df: pd.DataFrame, pval_col: str, fdr_col: str) -> pd.DataFrame:
    """Applies FDR correction to p-values within each group."""
    df[fdr_col] = df.groupby('group', observed=True)[pval_col] \
                    .transform(lambda x: multipletests(x, method='fdr_bh')[1])
    return df


def _worker_function_in_memory(
    feature_name: str,
    feature_index: int,
    expression_matrix: Union[np.ndarray, spmatrix],
    **kwargs
) -> pd.DataFrame:
    """
    Worker function for in-memory data. Extracts a single feature's
    expression vector and passes it to the core analyzer.
    """
    # Slicing a single column from a sparse or dense matrix is efficient
    expression_vector = expression_matrix[:, feature_index]
    if hasattr(expression_vector, "toarray"):
        expression_vector = expression_vector.toarray().flatten()
    
    return _analyze_one_feature(
        expression_vector=expression_vector,
        feature_name=feature_name,
        **kwargs
    )


def _worker_function_backed(
    feature_name: str,
    feature_index: int,
    adata_path: str, # Pass path to avoid pickling large objects
    **kwargs
) -> pd.DataFrame:
    """
    Worker function for backed AnnData. Reads a single feature's
    expression vector from disk and passes it to the core analyzer.
    """
    # Each worker opens its own read-only handle to the h5ad file
    import anndata
    adata_backed = anndata.read_h5ad(adata_path, backed='r')
    
    expression_vector = adata_backed.X[:, feature_index]
    if hasattr(expression_vector, "toarray"):
        expression_vector = expression_vector.toarray().flatten()
        
    return _analyze_one_feature(
        expression_vector=expression_vector,
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
    n_jobs: int = -1
) -> pd.DataFrame:
    """
    Orchestrates the statistical analysis for a list of features in parallel.

    This function selects the appropriate parallelization strategy based on
    whether the input data is in-memory or a backed AnnData object.

    Parameters
    ----------
    expression_data : np.ndarray, spmatrix, or AnnData
        The expression data (cells x features). Can be a backed AnnData object.
    features_to_analyze : list of str
        A list of feature names to be profiled.
    all_feature_names : list of str
        The full list of all feature names in `expression_data`.
    group_labels : np.ndarray
        A 1D array of group labels for each cell.
    batch_labels : np.ndarray, optional
        A 1D array of batch labels for each cell, by default None.
    specificity_metric : str, optional
        The specificity metric to use ('tau' or 'gini'), by default 'tau'.
    background_rate : float, optional
        Background rate for binomial test, by default 0.01.
    n_jobs : int, optional
        Number of parallel jobs to run, by default -1 (use all available cores).

    Returns
    -------
    pd.DataFrame
        A comprehensive DataFrame containing the full statistical profile
        for all requested features, with FDR-corrected p-values.
    """
    # Create a map for quick index lookup
    feature_index_map = {name: i for i, name in enumerate(all_feature_names)}
    
    # Common arguments for the worker functions
    worker_kwargs = {
        'labels_vector': group_labels,
        'batch_vector': batch_labels,
        'specificity_metric': specificity_metric,
        'background_rate': background_rate
    }
    
    # --- Select execution strategy: in-memory vs. backed ---
    is_backed = ANNDATA_AVAILABLE and isinstance(expression_data, AnnData) and expression_data.isbacked
    
    if is_backed:
        # For backed mode, we pass the file path to workers to avoid pickling issues
        worker_func = _worker_function_backed
        worker_args = {
            'adata_path': expression_data.filename,
            **worker_kwargs
        }
    else:
        # For in-memory data, we pass the data matrix directly
        worker_func = _worker_function_in_memory
        worker_args = {
            'expression_matrix': expression_data if not (ANNDATA_AVAILABLE and isinstance(expression_data, AnnData)) else expression_data.X,
            **worker_kwargs
        }

    # --- Run parallel computation ---
    # Use 'loky' backend for robust process-based parallelism
    list_of_results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(worker_func)(
            feature_name=feature,
            feature_index=feature_index_map[feature],
            **worker_args
        ) for feature in features_to_analyze
    )
    
    if not list_of_results:
        return pd.DataFrame() # Return empty if no features were analyzed
        
    # --- Combine results and apply FDR correction ---
    full_results_df = pd.concat(list_of_results, ignore_index=True)
    
    # FDR for presence test (applied globally across all features and groups)
    full_results_df['fdr_presence'] = multipletests(full_results_df['p_val_presence'], method='fdr_bh')[1]
    
    # FDR for marker test (applied per-group, as is standard)
    full_results_df = _fdr_correct_per_group(full_results_df, 'p_val_marker', 'fdr_marker')
    
    # Final cleanup and column ordering
    final_cols = [
        'feature_id', 'group', 'norm_score', 'pct_expressing',
        'fdr_presence', 'fdr_marker', 'log2fc_marker',
        f'specificity_{specificity_metric}'
    ]
    # Drop raw p-value columns
    full_results_df = full_results_df.drop(columns=['p_val_presence', 'p_val_marker'])
    
    return full_results_df[final_cols]