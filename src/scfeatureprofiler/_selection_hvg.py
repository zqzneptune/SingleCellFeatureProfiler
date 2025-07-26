#!/usr/bin/env python

"""
Internal module for vanilla Highly Variable Gene (HVG) selection.
"""

from typing import List, Optional, Union
import logging

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse
from statsmodels.nonparametric.smoothers_lowess import lowess

from ._utils import _prepare_and_validate_inputs

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


def _calculate_mean_variance(X: Union[np.ndarray, spmatrix]):
    """Calculates mean and variance in a sparse-aware manner."""
    if issparse(X):
        mean = np.asarray(X.mean(axis=0)).flatten()
        # E[X^2] - (E[X])^2
        mean_sq = np.asarray(X.power(2).mean(axis=0)).flatten()
        var = mean_sq - (mean ** 2)
    else:
        mean = X.mean(axis=0)
        var = X.var(axis=0)
    return mean, var


def select_hvg_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    feature_names: Optional[List[str]] = None,
    n_top_features: int = 3000,
    verbose: bool = True
) -> List[str]:
    """
    Selects Highly Variable Genes (HVGs) using a vanilla implementation.

    This function implements the 'seurat_v3' method. It automatically checks
    if the data appears to be log-transformed and applies normalization and
    log-transformation only if needed.
    """
    _log_and_print(f"--- Selecting top {n_top_features} Highly Variable Genes (HVGs) ---", verbose)

    expr_matrix, f_names, _, _ = _prepare_and_validate_inputs(
        data, group_by=np.arange(data.shape[0]), feature_names=feature_names
    )

    # --- NEW: Check if data is already log-transformed ---
    # Heuristic: if max value is small (e.g., < 20), assume it's logged.
    # Raw counts or CPM will typically have much larger max values.
    max_val = expr_matrix.max()
    is_logged = max_val < 20 

    if is_logged:
        _log_and_print("  Data appears to be already log-transformed. Skipping normalization.", verbose)
        norm_matrix = expr_matrix.copy()
    else:
        _log_and_print("  Data appears to be raw counts. Applying library size normalization and log-transformation...", verbose)
        if issparse(expr_matrix):
            norm_matrix = expr_matrix.copy().astype(np.float32)
            lib_size = np.asarray(norm_matrix.sum(axis=1)).flatten()
            median_lib_size = np.median(lib_size[lib_size > 0])
            if median_lib_size == 0: median_lib_size = 1e4
            
            # Normalize per cell
            scaling_factors = median_lib_size / lib_size
            scaling_factors[np.isinf(scaling_factors)] = 0.
            norm_matrix = norm_matrix.multiply(scaling_factors[:, np.newaxis])
            
            norm_matrix.data = np.log1p(norm_matrix.data)
        else:
            norm_matrix = expr_matrix.copy().astype(np.float32)
            lib_size = norm_matrix.sum(axis=1)
            median_lib_size = np.median(lib_size[lib_size > 0])
            if median_lib_size == 0: median_lib_size = 1e4
            
            # Normalize per cell, handling division by zero for empty cells
            scaling_factors = np.divide(median_lib_size, lib_size, out=np.zeros_like(lib_size, dtype=float), where=lib_size!=0)
            norm_matrix = norm_matrix * scaling_factors[:, np.newaxis]
            
            norm_matrix = np.log1p(norm_matrix)

    # 3. Calculate mean and variance for each feature
    mean, var = _calculate_mean_variance(norm_matrix)
    
    feature_stats = pd.DataFrame({
        'mean': mean,
        'variance': var,
        'feature': f_names
    })
    
    feature_stats = feature_stats[feature_stats['variance'] > 0]
    
    # 4. Fit LOESS regression: variance ~ log(mean)
    feature_stats['log_mean'] = np.log1p(feature_stats['mean'])
    
    smoothed = lowess(feature_stats['variance'], feature_stats['log_mean'], frac=0.3)
    
    loess_map = pd.DataFrame(smoothed, columns=['log_mean_sm', 'variance_sm'])
    # --- FIX: Ensure the merge key dtypes match ---
    feature_stats['log_mean'] = feature_stats['log_mean'].astype('float64')
    
    merged_stats = pd.merge_asof(
        feature_stats.sort_values('log_mean'),
        loess_map.sort_values('log_mean_sm'),
        left_on='log_mean',
        right_on='log_mean_sm',
        direction='nearest'
    )
    
    _log_and_print("  LOESS model fitted to mean-variance trend.", verbose)

    # 5. Standardize variance
    merged_stats['expected_std'] = np.sqrt(merged_stats['variance_sm'])
    merged_stats['variance_std'] = (
        (merged_stats['variance'] - merged_stats['variance_sm']) / 
        merged_stats['expected_std']
    )
    
    # --- FIX: Replace inplace clip with direct assignment ---
    # This is the recommended, future-proof way to perform this operation.
    merged_stats['variance_std'] = merged_stats['variance_std'].clip(
        upper=np.sqrt(norm_matrix.shape[0])
    )
    # --- END FIX ---
    
    # 6. Rank features and select top N
    merged_stats.sort_values('variance_std', ascending=False, inplace=True)
    
    hvg_df = merged_stats.head(n_top_features)
    hvg_list = hvg_df['feature'].tolist()

    _log_and_print(f"  Selected {len(hvg_list)} HVGs.", verbose)
    _log_and_print("-" * 35, verbose)
    
    return hvg_list