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
from ._logging import _log_and_print

logger = logging.getLogger(__name__)

try:
    from anndata import AnnData
except ImportError:
    AnnData = None


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
    
    # --- FIX: Add a safeguard for empty input data ---
    if data.shape[0] == 0:
        _log_and_print("  Input data contains zero cells. Returning empty list.", verbose)
        return []
    # --- END FIX ---
    
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
    
    freq_mask = (freq >= min_freq) & (freq <= max_freq)
    _log_and_print(f"  - Passed frequency filter ({min_freq} <= freq <= {max_freq}): {freq_mask.sum()} features remaining.", verbose)
    
    mean_expr = np.asarray(expression_matrix.mean(axis=0)).flatten()
    var_expr = _calculate_sparse_variance(expression_matrix) if issparse(expression_matrix) else np.var(expression_matrix, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        var_mean_ratio = np.nan_to_num(var_expr / mean_expr)

    var_mean_mask = var_mean_ratio > var_mean_ratio_min
    
    candidate_mask = freq_mask & var_mean_mask
    _log_and_print(f"  - Passed var/mean filter (>{var_mean_ratio_min}): {candidate_mask.sum()} features remaining.", verbose)

    # --- Tiers 2 & 3: Distribution Shape Filters (Iterative) ---
    tier1_indices = np.where(candidate_mask)[0]
    if not tier1_indices.any():
        _log_and_print("\nNo features passed Tier 1. Returning empty list.", verbose)
        return []
        
    _log_and_print("\nRunning Tiers 2 & 3 filters (Distribution Shape)...", verbose)
    
    candidate_indices = tier1_indices
    
    # Tier 2a: Gap Statistic
    passed_indices = []
    for i in candidate_indices:
        non_zero_expr = expression_matrix[:, i].data if issparse(expression_matrix) else expression_matrix[expression_matrix[:, i] > expression_threshold, i]
        if len(non_zero_expr) < 2: continue
        if np.percentile(non_zero_expr, 10) > gap_stat_min * (expression_threshold + 1e-9):
            passed_indices.append(i)
    _log_and_print(f"  - Passed Gap Statistic filter (>{gap_stat_min}): {len(passed_indices)} features remaining.", verbose)
    candidate_indices = passed_indices

    # Tier 2b: Right-Tail Heaviness
    passed_indices = []
    for i in candidate_indices:
        non_zero_expr = expression_matrix[:, i].data if issparse(expression_matrix) else expression_matrix[expression_matrix[:, i] > expression_threshold, i]
        if len(non_zero_expr) < 2: continue
        p50, p90 = np.percentile(non_zero_expr, [50, 90])
        if p50 > 1e-9 and (p90 / p50) > right_tail_min:
            passed_indices.append(i)
    _log_and_print(f"  - Passed Right-Tail filter (>{right_tail_min}): {len(passed_indices)} features remaining.", verbose)
    candidate_indices = passed_indices

    # Tier 3: Coefficient of Variation
    if cv_min is not None:
        passed_indices = []
        for i in candidate_indices:
            non_zero_expr = expression_matrix[:, i].data if issparse(expression_matrix) else expression_matrix[expression_matrix[:, i] > expression_threshold, i]
            if len(non_zero_expr) < 2: continue
            mean_val = np.mean(non_zero_expr)
            if mean_val > 1e-9 and (np.std(non_zero_expr) / mean_val) > cv_min:
                passed_indices.append(i)
        _log_and_print(f"  - Passed CV filter (>{cv_min}): {len(passed_indices)} features remaining.", verbose)
        candidate_indices = passed_indices

    final_candidate_indices = candidate_indices
    _log_and_print("-" * 35, verbose)

    return [f_names[i] for i in final_candidate_indices]