#!/usr/bin/env python

"""Single-cell feature expression profiling."""

from ._legacy import legacy_select_robust_markers
from .api import (
    compare_feature_profiles,
    diagnose_clustering_geometry,
    evaluate_annotation_evidence,
    evaluate_clustering,
    find_marker_features,
    get_feature_activity,
    get_feature_profiles,
    plot_annotation_evidence,
    profile_features,
    profile_features_by_sample,
    resample_profile_cells,
    resample_profile_samples,
    select_annotation_evidence_rows,
    select_profile_features,
    select_robust_markers,
    summarize_resampled_selection,
)
from .contrasts import Contrast
from .inputs import resolve_expression_input
from .models import (
    AnnotationEvidenceResult,
    ClusteringDiagnosticsResult,
    ExpressionSource,
    ProfileComparisonResult,
    ReplicateProfileResult,
    ResamplingResult,
    ResolvedExpression,
)
from .selection import (
    FeatureSelectionResult,
    RankingCriterion,
    SelectionCriteria,
)

__all__ = [
    "get_feature_profiles",
    "plot_annotation_evidence",
    "get_feature_activity",
    "find_marker_features",
    "evaluate_clustering",
    "diagnose_clustering_geometry",
    "evaluate_annotation_evidence",
    "select_robust_markers",
    "compare_feature_profiles",
    "ExpressionSource",
    "ResolvedExpression",
    "resolve_expression_input",
    "Contrast",
    "profile_features",
    "profile_features_by_sample",
    "ReplicateProfileResult",
    "resample_profile_cells",
    "resample_profile_samples",
    "ResamplingResult",
    "ProfileComparisonResult",
    "AnnotationEvidenceResult",
    "ClusteringDiagnosticsResult",
    "SelectionCriteria",
    "RankingCriterion",
    "FeatureSelectionResult",
    "select_profile_features",
    "select_annotation_evidence_rows",
    "summarize_resampled_selection",
    "legacy_select_robust_markers",
]

__version__ = "2.0.0"
