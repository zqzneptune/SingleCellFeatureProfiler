#!/usr/bin/env python

"""
Internal module for Highly Variable Gene (HVG) selection using Scanpy.
"""

from typing import List
import logging

try:
    import anndata
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    anndata = None

from ._logging import _log_and_print

logger = logging.getLogger(__name__)


def select_hvg_features(
    adata_proc: "anndata.AnnData",
    n_top_features: int = 3000,
    verbose: bool = True
) -> List[str]:
    """
    Selects Highly Variable Genes (HVGs) from a preprocessed AnnData object.
    
    This function uses `scanpy.pp.highly_variable_genes` and assumes the
    input AnnData object has already been normalized and log-transformed.
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("AnnData and Scanpy are required for HVG selection.")
    
    _log_and_print(f"--- Selecting top {n_top_features} Highly Variable Genes (HVGs) ---", verbose)

    # Work on a copy to avoid modifying the input object passed from the API
    adata_hvg = adata_proc.copy()

    sc.pp.highly_variable_genes(
        adata_hvg,
        n_top_genes=n_top_features,
        flavor='seurat_v3',
        subset=True
    )
    
    hvg_list = adata_hvg.var_names.tolist()
    
    _log_and_print(f"  Selected {len(hvg_list)} HVGs.", verbose)
    _log_and_print("-" * 35, verbose)
    
    return hvg_list