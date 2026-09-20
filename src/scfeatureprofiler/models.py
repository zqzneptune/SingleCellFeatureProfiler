"""Canonical data models used by the profiling pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from scipy import sparse


@dataclass(frozen=True)
class ExpressionSource:
    """An explicit selector for an AnnData expression matrix."""

    kind: str = "X"
    layer: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in {"X", "raw", "layer"}:
            raise ValueError("Expression source kind must be 'X', 'raw', or 'layer'.")
        if self.kind == "layer" and not self.layer:
            raise ValueError(
                "A layer expression source requires a non-empty layer name."
            )
        if self.kind != "layer" and self.layer is not None:
            raise ValueError("A layer name is only valid when source kind is 'layer'.")

    @classmethod
    def parse(cls, source: Union["ExpressionSource", str]) -> "ExpressionSource":
        """Parse the public ``X``, ``raw``, or ``layer:<name>`` syntax."""
        if isinstance(source, cls):
            return source
        if not isinstance(source, str):
            raise TypeError("`expression_source` must be a string or ExpressionSource.")
        if source == "X":
            return cls("X")
        if source == "raw":
            return cls("raw")
        if source.startswith("layer:"):
            return cls("layer", source.split(":", 1)[1])
        raise ValueError("`expression_source` must be 'X', 'raw', or 'layer:<name>'.")

    @property
    def descriptor(self) -> str:
        return f"layer:{self.layer}" if self.kind == "layer" else self.kind


@dataclass(frozen=True)
class ResolvedExpression:
    """A resolved cells-by-features matrix and its analysis metadata."""

    expression_matrix: Any
    feature_ids: pd.Index
    cell_ids: pd.Index
    obs: pd.DataFrame
    group_labels: np.ndarray
    condition_labels: Optional[np.ndarray]
    sample_labels: Optional[np.ndarray]
    expression_source: ExpressionSource
    is_sparse: bool
    is_backed: bool
    backing_path: Optional[Path] = None

    @property
    def shape(self) -> tuple:
        return self.expression_matrix.shape

    @property
    def source_descriptor(self) -> str:
        return self.expression_source.descriptor

    def feature_index(self, feature: Any) -> int:
        """Return the unambiguous column position for a feature identifier."""
        try:
            location = self.feature_ids.get_loc(feature)
        except KeyError as error:
            raise KeyError(
                f"Feature {feature!r} is not present in the selected source."
            ) from error
        if not isinstance(location, (int, np.integer)):
            raise ValueError(
                "Feature identifiers must be unique for name-based extraction."
            )
        return int(location)

    def extract_feature_at(self, index: int) -> np.ndarray:
        """Extract one feature as a dense vector without densifying the full matrix."""
        index = int(index)
        if index < 0 or index >= self.shape[1]:
            raise IndexError("Feature column index is out of range.")
        vector = self.expression_matrix[:, index]
        if sparse.issparse(vector) or hasattr(vector, "toarray"):
            vector = vector.toarray()
        return np.asarray(vector).reshape(-1)

    def extract_feature(self, feature: Any) -> np.ndarray:
        """Extract one feature by identifier, including an integer identifier."""
        return self.extract_feature_at(self.feature_index(feature))


@dataclass(frozen=True)
class ReplicateProfileResult:
    """Pooled feature profiles and individually accessible per-sample evidence."""

    profiles: pd.DataFrame
    sample_profiles: pd.DataFrame


@dataclass(frozen=True)
class ResamplingResult:
    """Point estimates, bootstrap summaries, and individual resampling draws."""

    profiles: pd.DataFrame
    summaries: pd.DataFrame
    distributions: pd.DataFrame
    resampling_unit: str
    iterations: int
    seed: int
    confidence_level: float
    sample_profiles: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class ProfileComparisonResult:
    """Aligned relationships, measurements, and recorded configurations."""

    relationships: pd.DataFrame
    measurements: pd.DataFrame
    configurations: pd.DataFrame
    reference_name: str
    comparison_name: str
    compared_measurements: tuple[str, ...]
    compared_configuration_fields: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationEvidenceResult:
    """Complete annotation evidence and non-exclusive category views."""

    evaluated_profiles: pd.DataFrame
    supporting_evidence: pd.DataFrame
    shared_evidence: pd.DataFrame
    competitor_evidence: pd.DataFrame
    expectation_evidence: pd.DataFrame
    contradictory_evidence: pd.DataFrame
    unresolved_expectations: pd.DataFrame
    proposed_label: Any
    direction_tolerance: float
    expected_support_features: tuple[Any, ...]
    expected_absent_features: tuple[Any, ...]


@dataclass(frozen=True)
class ClusteringDiagnosticsResult:
    """Cell, group, and overall geometry summaries for supplied labels."""

    cell_diagnostics: pd.DataFrame
    group_diagnostics: pd.DataFrame
    overall_diagnostics: pd.DataFrame
    label_key: str
    representation: str
    metric: str
