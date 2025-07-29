#!/usr/bin/env python

"""
Internal profiling engine for orchestrating parallel calculations.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix
from joblib import Parallel, delayed, parallel_backend, cpu_count
from statsmodels.stats.multitest import multipletests

from ._core import _analyze_one_feature
from ._logging import _log_and_print

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None

logger = logging.getLogger(__name__)


def _worker_function(
    feature_name: str,
    feature_index: int,
    expression_data: Union[np.ndarray, spmatrix, str],
    is_backed: bool,
    **kwargs
) -> pd.DataFrame:
    """
    Worker function that runs the per-condition analysis for one feature.
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
    condition_labels: Optional[np.ndarray] = None,
    specificity_metric: str = 'tau',
    background_rate: float = 0.01,
    n_jobs: int = -1,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Orchestrates analysis and returns the full per-condition table.
    FDR correction is done here, but aggregation is left to downstream functions.
    """
    feature_index_map = {name: i for i, name in enumerate(all_feature_names)}
    
    common_worker_kwargs = {
        'labels_vector': group_labels,
        'condition_vector': condition_labels,
        'specificity_metric': specificity_metric,
        'background_rate': background_rate
    }
    
    is_backed = ANNDATA_AVAILABLE and isinstance(expression_data, AnnData) and expression_data.isbacked
    
    if is_backed:
        expr_data_for_worker = expression_data.filename
    elif ANNDATA_AVAILABLE and isinstance(expression_data, AnnData):
        expr_data_for_worker = expression_data.X
    else:
        expr_data_for_worker = expression_data

    actual_jobs = n_jobs if n_jobs > 0 else cpu_count()
    n_jobs_str = f"{n_jobs} (resolved to {actual_jobs} workers)" if n_jobs == -1 else str(actual_jobs)
    msg = (
        f"\n--- Profiling {len(features_to_analyze)} features across {len(np.unique(group_labels))} groups "
        f"and {len(np.unique(condition_labels)) if condition_labels is not None else 1} condition(s) ---\n"
        f"Starting parallel computation with n_jobs={n_jobs_str}..."
    )
    _log_and_print(msg, verbose)

    with parallel_backend('loky', n_jobs=n_jobs):
        list_of_per_condition_results = Parallel(verbose=0)(
            delayed(_worker_function)(
                feature_name=feature,
                feature_index=feature_index_map[feature],
                expression_data=expr_data_for_worker,
                is_backed=is_backed,
                **common_worker_kwargs
            ) for feature in features_to_analyze
        )
    
    _log_and_print("... Parallel computation complete.", verbose)
    
    if not list_of_per_condition_results:
        return pd.DataFrame()
        
    per_condition_df = pd.concat(list_of_per_condition_results, ignore_index=True)
    
    _log_and_print("\n--- Finalizing results ---", verbose)
    _log_and_print(f"  - Collected {len(per_condition_df)} per-condition observations.", verbose)
    
    _log_and_print("  - Applying FDR correction for presence...", verbose)
    per_condition_df['fdr_presence'] = multipletests(per_condition_df['p_val_presence'], method='fdr_bh')[1]
    
    return per_condition_df