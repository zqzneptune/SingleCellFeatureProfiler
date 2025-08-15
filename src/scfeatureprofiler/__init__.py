#!/usr/bin/env python

"""
SingleCellFeatureProfiler: A Python package for single-cell feature expression profiling.
"""

from .api import (
    get_feature_profiles,
    get_feature_activity,
    find_marker_features,
    evaluate_clustering,
    select_robust_markers,
)

__all__ = [
    "get_feature_profiles",
    "get_feature_activity",
    "find_marker_features",
    "evaluate_clustering",
    "select_robust_markers",
]

__version__ = "1.1.2"
