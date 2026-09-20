"""Canonical input and AnnData expression-source resolution."""

from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse

from .models import ExpressionSource, ResolvedExpression

try:
    from anndata import AnnData

    ANNDATA_AVAILABLE = True
except ImportError:  # pragma: no cover - package currently depends on scanpy/anndata
    AnnData = None
    ANNDATA_AVAILABLE = False


def _is_sparse_like(matrix: Any) -> bool:
    """Recognize SciPy and AnnData-backed sparse matrix implementations."""
    return sparse.issparse(matrix) or (
        getattr(matrix, "format", None) in {"csr", "csc"}
        and "sparse_dataset" in type(matrix).__module__
    )


def _select_anndata_matrix(data: Any, source: ExpressionSource) -> tuple[Any, pd.Index]:
    if source.kind == "X":
        return data.X, data.var_names.copy()
    if source.kind == "raw":
        if data.raw is None:
            raise ValueError("AnnData has no `.raw` expression source.")
        return data.raw.X, data.raw.var_names.copy()
    if source.layer not in data.layers:
        raise ValueError(f"AnnData layer {source.layer!r} was not found.")
    return data.layers[source.layer], data.var_names.copy()


def _resolve_labels(
    labels_arg: Any,
    *,
    obs: pd.DataFrame,
    n_cells: int,
    label_name: str,
    allow_none: bool,
) -> Optional[np.ndarray]:
    if labels_arg is None:
        if allow_none:
            return None
        raise ValueError(f"`{label_name}` is a required argument and cannot be None.")

    if isinstance(labels_arg, str):
        if labels_arg not in obs.columns:
            raise ValueError(
                f"`{labels_arg}` not found in observation metadata columns."
            )
        labels = obs[labels_arg].to_numpy()
    else:
        labels = np.asarray(labels_arg)

    if labels.ndim != 1:
        raise ValueError(
            f"`{label_name}` must be a 1D array-like object, but has "
            f"{labels.ndim} dimensions."
        )
    if len(labels) != n_cells:
        raise ValueError(
            f"Length of `{label_name}` ({len(labels)}) does not match the number "
            f"of cells in `data` ({n_cells})."
        )
    if pd.isna(labels).any():
        raise ValueError(f"`{label_name}` contains NaN or missing values.")
    return labels


def resolve_expression_input(
    data: Any,
    group_by: Any,
    *,
    feature_names: Optional[list[str]] = None,
    condition_by: Any = None,
    sample_by: Any = None,
    expression_source: Union[ExpressionSource, str] = "X",
) -> ResolvedExpression:
    """Resolve supported inputs into the package's canonical representation."""
    source = ExpressionSource.parse(expression_source)

    if ANNDATA_AVAILABLE and isinstance(data, AnnData):
        if feature_names is not None:
            raise ValueError(
                "Do not provide `feature_names` when `data` is an AnnData object. "
                "Feature names are taken from the selected AnnData source."
            )
        matrix, feature_ids = _select_anndata_matrix(data, source)
        cell_ids = data.obs_names.copy()
        obs = data.obs.copy(deep=False)
        is_backed = bool(data.isbacked)
        backing_path = Path(data.filename) if is_backed else None
    elif isinstance(data, pd.DataFrame):
        if source.kind != "X":
            raise ValueError("Non-AnnData inputs only support expression_source='X'.")
        if feature_names is not None:
            raise ValueError(
                "Do not provide `feature_names` when `data` is a pandas DataFrame. "
                "Feature names are taken from DataFrame columns."
            )
        feature_ids = data.columns.copy()
        cell_ids = data.index.copy()
        if all(isinstance(dtype, pd.SparseDtype) for dtype in data.dtypes):
            matrix = data.sparse.to_coo().tocsr()
        else:
            matrix = data.to_numpy(copy=False)
        obs = pd.DataFrame(index=cell_ids)
        is_backed = False
        backing_path = None
    elif isinstance(data, np.ndarray) or sparse.issparse(data):
        if source.kind != "X":
            raise ValueError("Non-AnnData inputs only support expression_source='X'.")
        if feature_names is None:
            raise ValueError(
                "`feature_names` must be provided when `data` is a numpy array "
                "or sparse matrix."
            )
        matrix = data
        feature_ids = pd.Index(feature_names)
        cell_ids = pd.RangeIndex(data.shape[0], name="cell_index")
        obs = pd.DataFrame(index=cell_ids)
        is_backed = False
        backing_path = None
    else:
        raise TypeError(
            f"Unsupported type for `data`: {type(data)}. Expected AnnData, "
            "pandas DataFrame, numpy array, or sparse matrix."
        )

    if getattr(matrix, "ndim", 2) != 2:
        raise ValueError(
            "Expression data must be a two-dimensional cells-by-features matrix."
        )
    n_cells, n_features = matrix.shape
    if len(feature_ids) != n_features:
        raise ValueError(
            f"Length of `feature_names` ({len(feature_ids)}) does not match the "
            f"number of features in `data` ({n_features})."
        )
    if feature_ids.has_duplicates:
        duplicates = feature_ids[feature_ids.duplicated()].unique().tolist()
        raise ValueError(
            f"Feature identifiers must be unique; duplicates: {duplicates}"
        )

    group_labels = _resolve_labels(
        group_by,
        obs=obs,
        n_cells=n_cells,
        label_name="group_by",
        allow_none=False,
    )
    condition_labels = _resolve_labels(
        condition_by,
        obs=obs,
        n_cells=n_cells,
        label_name="condition_by",
        allow_none=True,
    )
    sample_labels = _resolve_labels(
        sample_by,
        obs=obs,
        n_cells=n_cells,
        label_name="sample_by",
        allow_none=True,
    )

    return ResolvedExpression(
        expression_matrix=matrix,
        feature_ids=pd.Index(feature_ids),
        cell_ids=pd.Index(cell_ids),
        obs=obs,
        group_labels=group_labels,
        condition_labels=condition_labels,
        sample_labels=sample_labels,
        expression_source=source,
        is_sparse=_is_sparse_like(matrix),
        is_backed=is_backed,
        backing_path=backing_path,
    )
