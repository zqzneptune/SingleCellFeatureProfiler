#!/usr/bin/env python

"""
Internal utility functions for data validation and preparation.
"""

from typing import List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import spmatrix

from .inputs import AnnData, resolve_expression_input


def _prepare_and_validate_inputs(
    data: Union[AnnData, pd.DataFrame, np.ndarray, spmatrix],
    group_by: Union[str, list, np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    condition_by: Optional[Union[str, list, np.ndarray, pd.Series]] = None,
) -> Tuple[Union[np.ndarray, spmatrix, AnnData], List[str], np.ndarray, Optional[np.ndarray]]:
    """
    Validates and standardizes all input data for the profiling engine.
    """
    resolved = resolve_expression_input(
        data,
        group_by,
        feature_names=feature_names,
        condition_by=condition_by,
    )
    return (
        resolved.expression_matrix,
        resolved.feature_ids.tolist(),
        resolved.group_labels,
        resolved.condition_labels,
    )
