#!/usr/bin/env python

"""
SingleCellFeatureProfiler: A Python package for single-cell gene expression profiling.

This package provides tools for comprehensive statistical analysis of single-cell
RNA sequencing data, focusing on gene expression profiling across different cell
groups and marker gene discovery.

The main public API consists of:
- GeneProfiler: Deep-dive statistical profiling for specific genes with marker detection
- MarkerFinder: Discovery of marker genes across cell groups (coming soon)
- get_active_genes: Utility function for identifying active genes
"""

from ._gene_profiler import GeneProfiler
from ._utils import get_active_genes

# Public API - only expose user-facing classes
__all__ = [
    "GeneProfiler",
    "get_active_genes",
]

# Package metadata
__version__ = "0.1.0"
__author__ = "SingleCellFeatureProfiler Team"
__email__ = "contact@scfeatureprofiler.org"
__description__ = "Statistical profiling tools for single-cell gene expression data"
