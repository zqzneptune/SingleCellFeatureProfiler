#!/usr/bin/env python

"""
Internal utility functions for data validation and preparation.
"""

from typing import List, Optional, Union, Tuple, Any

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    AnnData = None
    ANNDATA_AVAILABLE = False


def _prepare_and_validate_inputs(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    condition_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
) -> Tuple[Union[np.ndarray, spmatrix, AnnData], List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Validates and standardizes all input data for the profiling engine.
    """
    n_cells, n_features = data.shape

    if ANNDATA_AVAILABLE and isinstance(data, AnnData):
        expression_matrix = data.X if not data.isbacked else data
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

    def process_labels(labels_arg: Any, n_cells_ref: int, label_name: str) -> Optional[np.ndarray]:
        """Helper to validate and convert any label vector."""
        if labels_arg is None:
            return None

        if ANNDATA_AVAILABLE and isinstance(data, AnnData) and isinstance(labels_arg, str):
            if labels_arg not in data.obs.columns:
                raise ValueError(f"`{labels_arg}` not found in `adata.obs` columns.")
            labels = data.obs[labels_arg].values
        else:
            labels = labels_arg

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
        
    condition_labels = process_labels(condition_by, n_cells, 'condition_by')

    return expression_matrix, feature_names, group_labels, condition_labels