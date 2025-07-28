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
from ._stability import _calculate_stability_scores
from ._logging import _log_and_print
from joblib import Parallel, delayed, parallel_backend, cpu_count
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
    Simplified worker function that only runs the per-condition analysis.
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
    Orchestrates analysis, builds the full per-condition table in memory,
    then performs FDR and stability calculations.
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

    # --- Enhanced Verbosity ---
    n_features = len(features_to_analyze)
    n_groups = len(np.unique(group_labels))
    n_conditions = len(np.unique(condition_labels)) if condition_labels is not None else 1
    # --- FIX: Accurately report the number of workers ---
    actual_jobs = n_jobs if n_jobs > 0 else cpu_count()
    msg = (
        f"\n--- Profiling {n_features} features across {n_groups} groups "
        f"and {n_conditions} condition(s) ---\n"
        f"Using {actual_jobs} CPU cores for parallel computation..."
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
    _log_and_print("\n--- Finalizing results ---", verbose)
    
    if not list_of_per_condition_results:
        return pd.DataFrame()
        
    per_condition_df = pd.concat(list_of_per_condition_results, ignore_index=True)
    _log_and_print(f"  - Collected {len(per_condition_df)} per-condition observations.", verbose)
    
    _log_and_print("  - Applying FDR correction for presence...", verbose)
    per_condition_df['fdr_presence'] = multipletests(per_condition_df['p_val_presence'], method='fdr_bh')[1]
    
    _log_and_print("  - Aggregating results and calculating stability scores...", verbose)
    specificity_col = f'specificity_{specificity_metric}'
    final_df = _calculate_stability_scores(per_condition_df, specificity_col)
    _log_and_print(f"  - Final aggregated table has {len(final_df)} rows.", verbose)
    
    return final_df