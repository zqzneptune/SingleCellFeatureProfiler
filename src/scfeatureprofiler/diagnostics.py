"""Descriptive clustering geometry diagnostics."""

from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_samples

from .models import ClusteringDiagnosticsResult


def _stable_value_key(value: Any) -> tuple[str, str]:
    return type(value).__name__, repr(value)


def _validate_text(value: str, argument: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"`{argument}` must be a non-empty string.")
    return value


def _validate_representation(matrix: Any, n_cells: int) -> Any:
    if not hasattr(matrix, "shape") or len(matrix.shape) != 2:
        raise ValueError(
            "The selected representation must be a two-dimensional matrix."
        )
    if matrix.shape[0] != n_cells:
        raise ValueError(
            "The selected representation must contain one row for every cell."
        )
    if matrix.shape[1] < 1:
        raise ValueError(
            "The selected representation must contain at least one dimension."
        )
    try:
        observed = matrix.data if sparse.issparse(matrix) else np.asarray(matrix)
        finite = np.isfinite(observed).all()
    except TypeError as error:
        raise TypeError("The selected representation must be numeric.") from error
    if not finite:
        raise ValueError("The selected representation must contain only finite values.")
    return matrix


def _validate_labels(labels: pd.Series) -> np.ndarray:
    if labels.isna().any():
        raise ValueError("Group labels must not contain missing values.")
    values = labels.to_numpy(dtype=object)
    for value in values:
        try:
            hash(value)
        except TypeError as error:
            raise TypeError("Group labels must be hashable scalar values.") from error
    unique = set(values)
    if len(unique) < 2:
        raise ValueError("Silhouette diagnostics require at least two observed groups.")
    if len(unique) >= len(values):
        raise ValueError("Silhouette diagnostics require fewer groups than cells.")
    return values


def _summarize_scores(scores: np.ndarray) -> dict[str, float]:
    return {
        "mean_silhouette_score": float(np.mean(scores)),
        "median_silhouette_score": float(np.median(scores)),
        "silhouette_score_standard_deviation": (
            float(np.std(scores, ddof=1)) if len(scores) > 1 else np.nan
        ),
        "minimum_silhouette_score": float(np.min(scores)),
        "maximum_silhouette_score": float(np.max(scores)),
        "negative_silhouette_fraction": float(np.mean(scores < 0)),
    }


def diagnose_clustering_geometry(
    adata: ad.AnnData,
    label_key: str,
    *,
    use_rep: str = "X_pca",
    metric: str = "euclidean",
) -> ClusteringDiagnosticsResult:
    """Describe label geometry in one explicitly selected representation.

    Silhouette values quantify cohesion and separation in the chosen geometry.
    They do not establish the biological validity of the supplied labels.
    """
    if not isinstance(adata, ad.AnnData):
        raise TypeError("`adata` must be an AnnData object.")
    label_key = _validate_text(label_key, "label_key")
    use_rep = _validate_text(use_rep, "use_rep")
    metric = _validate_text(metric, "metric")
    if use_rep not in adata.obsm:
        raise ValueError(f"Representation {use_rep!r} was not found in `adata.obsm`.")
    if label_key not in adata.obs:
        raise ValueError(f"Label key {label_key!r} was not found in `adata.obs`.")

    representation = _validate_representation(adata.obsm[use_rep], adata.n_obs)
    labels = _validate_labels(adata.obs[label_key])
    groups = sorted(set(labels), key=_stable_value_key)
    group_codes = {group: index for index, group in enumerate(groups)}
    label_codes = np.asarray([group_codes[label] for label in labels])
    try:
        scores = silhouette_samples(representation, label_codes, metric=metric)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Silhouette calculation failed for distance metric {metric!r}: {error}"
        ) from error

    cell_diagnostics = pd.DataFrame(
        {
            "cell_id": adata.obs_names.to_numpy(copy=True),
            "group_label": labels,
            "silhouette_score": scores,
        }
    )
    group_rows = []
    for group in groups:
        group_scores = scores[labels == group]
        group_rows.append(
            {
                "group_label": group,
                "n_cells": int(len(group_scores)),
                **_summarize_scores(group_scores),
            }
        )
    group_diagnostics = pd.DataFrame(group_rows)
    overall_diagnostics = pd.DataFrame(
        [
            {
                "n_cells": int(adata.n_obs),
                "n_groups": int(len(groups)),
                **_summarize_scores(scores),
                "label_key": label_key,
                "representation": use_rep,
                "metric": metric,
            }
        ]
    )
    return ClusteringDiagnosticsResult(
        cell_diagnostics=cell_diagnostics,
        group_diagnostics=group_diagnostics,
        overall_diagnostics=overall_diagnostics,
        label_key=label_key,
        representation=use_rep,
        metric=metric,
    )
