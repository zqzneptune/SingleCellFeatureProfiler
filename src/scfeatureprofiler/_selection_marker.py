#!/usr/bin/env python

"""
Internal module for sophisticated marker candidate selection.
"""

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse

from ._utils import _prepare_and_validate_inputs

logger = logging.getLogger(__name__)

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


def _log_and_print(msg: str, verbose: bool):
    """Helper to log and optionally print a message."""
    logger.info(msg)
    if verbose:
        print(msg)


def _calculate_sparse_variance(matrix: spmatrix) -> np.ndarray:
    """Calculates column-wise variance for a sparse matrix efficiently."""
    mean_of_squares = np.asarray(matrix.power(2).mean(axis=0)).flatten()
    mean = np.asarray(matrix.mean(axis=0)).flatten()
    return mean_of_squares - (mean ** 2)


def select_marker_candidates(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    feature_names: Optional[List[str]] = None,
    min_freq: float = 0.05,
    max_freq: float = 0.90,
    var_mean_ratio_min: float = 1.5,
    gap_stat_min: float = 1.2,
    right_tail_min: float = 2.5,
    cv_min: Optional[float] = 0.8,
    expression_threshold: float = 0.0,
    verbose: bool = True
) -> List[str]:
    """
    Selects candidate marker features using a tiered, data-driven filtering approach.
    """
    _log_and_print("--- Selecting marker candidates ---", verbose)

    expression_matrix, f_names, _, _ = _prepare_and_validate_inputs(
        data=data,
        group_by=np.arange(data.shape[0]),
        feature_names=feature_names
    )

    if issparse(expression_matrix) and expression_matrix.format != 'csr':
        expression_matrix = expression_matrix.tocsr()

    n_cells, n_features = expression_matrix.shape
    _log_and_print(f"Total features to start: {n_features}", verbose)

    # --- Tier 1: Primary Filters (Vectorized) ---
    _log_and_print("Running Tier 1 filters (Frequency & Variance)...", verbose)
    
    n_cells_detected = np.asarray((expression_matrix > expression_threshold).sum(axis=0)).flatten()
    freq = n_cells_detected / n_cells
    
    mean_expr = np.asarray(expression_matrix.mean(axis=0)).flatten()
    var_expr = _calculate_sparse_variance(expression_matrix) if issparse(expression_matrix) else np.var(expression_matrix, axis=0)
    
    # Use np.errstate to avoid warnings for division by zero, which we handle
    with np.errstate(divide='ignore', invalid='ignore'):
        var_mean_ratio = np.nan_to_num(var_expr / mean_expr)

    # Combine all Tier 1 boolean masks
    candidate_mask = (
        (freq >= min_freq) &
        (freq <= max_freq) &
        (var_mean_ratio > var_mean_ratio_min)
    )
    
    tier1_indices = np.where(candidate_mask)[0]
    _log_and_print(f"  {len(tier1_indices)} features passed Tier 1.", verbose)

    # --- Tiers 2 & 3: Distribution Shape Filters (Iterative) ---
    if not tier1_indices.any():
        _log_and_print("No features passed Tier 1. Returning empty list.", verbose)
        return []
        
    _log_and_print("Running Tiers 2 & 3 filters (Distribution Shape)...", verbose)
    
    final_candidate_indices = []
    for i in tier1_indices:
        if issparse(expression_matrix):
            # .data is the most efficient way to get non-zero values from a CSR/CSC column slice
            non_zero_expr = expression_matrix[:, i].data
        else:
            col_data = expression_matrix[:, i]
            non_zero_expr = col_data[col_data > expression_threshold]

        if len(non_zero_expr) < 2:
            continue

        p10, p50, p90 = np.percentile(non_zero_expr, [10, 50, 90])

        # Gap Statistic
        if p10 <= gap_stat_min * (expression_threshold + 1e-9):
            continue

        # Right-Tail Heaviness
        if p50 <= 1e-9 or (p90 / p50) <= right_tail_min:
            continue
            
        # Optional CV Filter
        if cv_min is not None:
            mean_val = np.mean(non_zero_expr)
            if mean_val <= 1e-9 or (np.std(non_zero_expr) / mean_val) <= cv_min:
                continue

        final_candidate_indices.append(i)

    _log_and_print(f"  {len(final_candidate_indices)} features passed Tiers 2 & 3.", verbose)
    _log_and_print("-" * 35, verbose)

    return [f_names[i] for i in final_candidate_indices]