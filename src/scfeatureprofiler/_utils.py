#!/usr/bin/env python

"""
Internal and public utility functions for SingleCellFeatureProfiler.

This module contains helper functions for data validation, preparation, and
pre-filtering of features.
"""

from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix, issparse

# AnnData is an optional dependency
try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None  # Define AnnData as None if it's not available
    ANNDATA_AVAILABLE = False


def _prepare_and_validate_inputs(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    batch_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
) -> Tuple[Union[np.ndarray, spmatrix], List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Validates and standardizes all input data for the profiling engine.

    This is the single entry point for data ingestion. It handles multiple
    input formats and returns a consistent set of objects for downstream processing.

    Parameters
    ----------
    data : AnnData, pd.DataFrame, np.ndarray, or spmatrix
        The single-cell expression matrix (cells x features).
        - If AnnData, `group_by` and `batch_by` are looked up in `.obs`.
        - If DataFrame, columns are used as `feature_names`.
        - If array/sparse matrix, `feature_names` must be provided.
    group_by : str or list-like
        The group labels for each cell.
        - If `data` is AnnData, this is the column name in `adata.obs`.
        - Otherwise, a list, numpy array, or pandas Series of labels.
    feature_names : list of str, optional
        List of feature names. Required if `data` is a numpy array or sparse matrix.
        By default None.
    batch_by : str or list-like, optional
        The batch labels for each cell (e.g., donor IDs).
        - If `data` is AnnData, this is the column name in `adata.obs`.
        - Otherwise, a list, numpy array, or pandas Series of labels.
        By default None.

    Returns
    -------
    tuple
        A tuple containing:
        - expression_matrix: The validated expression data (cells x features).
          Can be a numpy array, sparse matrix, or a view of a backed AnnData object.
        - feature_names: A list of strings with the feature names.
        - group_labels: A 1D numpy array of group labels for each cell.
        - batch_labels: A 1D numpy array of batch labels, or None.

    Raises
    ------
    ValueError
        If inputs have inconsistent dimensions, missing required arguments,
        or contain invalid values (like NaNs in labels).
    TypeError
        If input types are not supported.
    """
    # --- 1. Process Expression Data and Feature Names ---
    n_cells, n_features = data.shape

    if ANNDATA_AVAILABLE and isinstance(data, AnnData):
        expression_matrix = data.X
        feature_names_processed = data.var_names.tolist()
        if feature_names is not None:
            raise ValueError(
                "Do not provide `feature_names` when `data` is an AnnData object. "
                "Feature names are taken from `adata.var_names`."
            )
        feature_names = feature_names_processed

    elif isinstance(data, pd.DataFrame):
        expression_matrix = data.values
        feature_names_processed = data.columns.tolist()
        if feature_names is not None:
            raise ValueError(
                "Do not provide `feature_names` when `data` is a pandas DataFrame. "
                "Feature names are taken from DataFrame columns."
            )
        feature_names = feature_names_processed

    elif isinstance(data, (np.ndarray, spmatrix)):
        expression_matrix = data
        if feature_names is None:
            raise ValueError(
                "`feature_names` must be provided when `data` is a numpy array or sparse matrix."
            )
        if len(feature_names) != n_features:
            raise ValueError(
                f"Length of `feature_names` ({len(feature_names)}) does not match "
                f"the number of features in `data` ({n_features})."
            )
    else:
        raise TypeError(f"Unsupported type for `data`: {type(data)}. "
                        "Expected AnnData, pandas DataFrame, numpy array, or sparse matrix.")

    # --- 2. Process Group and Batch Labels ---
    def process_labels(labels_arg: Any, n_cells_ref: int, label_name: str) -> Optional[np.ndarray]:
        """Helper to validate and convert any label vector."""
        if labels_arg is None:
            return None

        # Extract labels from AnnData if `labels_arg` is a string key
        if ANNDATA_AVAILABLE and isinstance(data, AnnData) and isinstance(labels_arg, str):
            if labels_arg not in data.obs.columns:
                raise ValueError(f"`{labels_arg}` not found in `adata.obs` columns.")
            labels = data.obs[labels_arg].values
        else:
            labels = labels_arg

        # Convert to numpy array
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        if labels.ndim != 1:
            raise ValueError(f"`{label_name}` must be a 1D array-like object, but has {labels.ndim} dimensions.")

        if len(labels) != n_cells_ref:
            raise ValueError(
                f"Length of `{label_name}` ({len(labels)}) does not match "
                f"the number of cells in `data` ({n_cells_ref})."
            )

        if pd.isna(labels).any():
            raise ValueError(f"`{label_name}` contains NaN or missing values.")
        
        return labels

    group_labels = process_labels(group_by, n_cells, 'group_by')
    if group_labels is None:
        raise ValueError("`group_by` is a required argument and cannot be None.")
        
    batch_labels = process_labels(batch_by, n_cells, 'batch_by')

    return expression_matrix, feature_names, group_labels, batch_labels


def get_active_features(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    feature_names: Optional[List[str]] = None,
    min_cells: int = 10,
    min_expression: float = 0.0
) -> List[str]:
    """
    Identify actively expressed features based on expression thresholds.

    This utility function filters features based on the number of cells expressing
    them above a minimum threshold. It is useful for reducing the feature space
    before running more intensive profiling.

    Parameters
    ----------
    data : AnnData, pd.DataFrame, np.ndarray, or spmatrix
        The single-cell expression matrix (cells x features).
    feature_names : list of str, optional
        List of feature names. Required if `data` is a numpy array or sparse matrix.
        By default None.
    min_cells : int, default 10
        Minimum number of cells that must express a feature above `min_expression`
        for the feature to be considered "active".
    min_expression : float, default 0.0
        Minimum expression threshold. A value of 0 means any non-zero expression
        is counted.

    Returns
    -------
    list of str
        A list of active feature names.
    
    Examples
    --------
    >>> import numpy as np
    >>> from scfeatureprofiler._utils import get_active_features
    >>>
    >>> expression = np.random.poisson(0.5, (100, 50))
    >>> features = [f"Feature_{i}" for i in range(50)]
    >>> active = get_active_features(expression, feature_names=features, min_cells=5)
    >>> print(f"Found {len(active)} active features.")
    """
    # Use the main validator to standardize inputs, we only need the matrix and feature names
    expression_matrix, f_names, _, _ = _prepare_and_validate_inputs(
        data=data,
        group_by=np.arange(data.shape[0]), # Provide a dummy group_by to pass validation
        feature_names=feature_names
    )

    # Efficiently count expressing cells
    if issparse(expression_matrix):
        # For sparse matrices, counting non-zero elements is fast
        expressing_cells_per_feature = expression_matrix.getnnz(axis=0) if min_expression <= 0 \
            else (expression_matrix > min_expression).sum(axis=0)
        # sum() on a sparse matrix returns a 1xN matrix, so convert to array
        expressing_cells_per_feature = np.asarray(expressing_cells_per_feature).flatten()
    else:
        # For dense matrices, direct comparison is fine
        expressing_cells_per_feature = (expression_matrix > min_expression).sum(axis=0)

    # Create boolean mask and filter feature names
    active_mask = expressing_cells_per_feature >= min_cells
    active_feature_names = [f_names[i] for i, is_active in enumerate(active_mask) if is_active]
    
    return active_feature_names