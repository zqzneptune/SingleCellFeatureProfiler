#!/usr/bin/env python

"""
Internal utility functions for SingleCellFeatureProfiler.

This module contains helper functions that support the main API classes
and provide standalone utility functionality for gene expression analysis.
"""

from typing import List, Optional, Union, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

try:
    from anndata import AnnData
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    AnnData = None


def get_active_genes(
    expression_data: Union[np.ndarray, sparse.csr_matrix, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    min_cells: int = 10,
    min_expression: float = 0.0,
    return_mask: bool = False
) -> Union[List[str], Tuple[List[str], np.ndarray]]:
    """
    Identify actively expressed genes based on expression thresholds.
    
    This utility function filters genes based on the number of cells expressing
    them above a minimum threshold. It's useful for preprocessing steps and
    quality control in single-cell analysis pipelines.
    
    Parameters
    ----------
    expression_data : numpy.ndarray, scipy.sparse.csr_matrix, or pandas.DataFrame
        Gene expression matrix with cells as rows and genes as columns.
        For DataFrame input, gene names are taken from column names.
    gene_names : list of str, optional
        List of gene names corresponding to columns in expression_data.
        Required if expression_data is not a DataFrame.
    min_cells : int, default 10
        Minimum number of cells that must express a gene above min_expression
        for the gene to be considered "active".
    min_expression : float, default 0.0
        Minimum expression threshold. Genes must be expressed above this
        value in at least min_cells to be considered active.
    return_mask : bool, default False
        If True, returns both the list of active genes and a boolean mask
        indicating which genes are active.
        
    Returns
    -------
    list of str or tuple
        If return_mask is False: List of active gene names.
        If return_mask is True: Tuple of (active_gene_names, boolean_mask).
        
    Examples
    --------
    >>> import numpy as np
    >>> from scfeatureprofiler import get_active_genes
    >>> 
    >>> # Example with numpy array
    >>> expression = np.random.poisson(0.5, (100, 50))  # 100 cells, 50 genes
    >>> gene_names = [f"Gene_{i}" for i in range(50)]
    >>> active_genes = get_active_genes(expression, gene_names, min_cells=5)
    >>> print(f"Found {len(active_genes)} active genes")
    
    >>> # Example with pandas DataFrame
    >>> import pandas as pd
    >>> df = pd.DataFrame(expression, columns=gene_names)
    >>> active_genes = get_active_genes(df, min_cells=5)
    
    >>> # Get both genes and mask
    >>> active_genes, mask = get_active_genes(df, min_cells=5, return_mask=True)
    >>> print(f"Active genes: {len(active_genes)}, Total genes: {len(mask)}")
    """
    # Handle DataFrame input
    if isinstance(expression_data, pd.DataFrame):
        if gene_names is not None:
            raise ValueError("gene_names should not be provided when expression_data is a DataFrame")
        gene_names = expression_data.columns.tolist()
        expression_matrix = expression_data.values
    else:
        if gene_names is None:
            raise ValueError("gene_names must be provided when expression_data is not a DataFrame")
        if len(gene_names) != expression_data.shape[1]:
            raise ValueError(f"Length of gene_names ({len(gene_names)}) must match number of columns in expression_data ({expression_data.shape[1]})")
        expression_matrix = expression_data
    
    # Convert sparse matrix to dense for threshold operations if needed
    if sparse.issparse(expression_matrix):
        # For sparse matrices, we can efficiently count non-zero elements above threshold
        if min_expression <= 0:
            # Count non-zero elements
            expressing_cells_per_gene = np.array((expression_matrix > min_expression).sum(axis=0)).flatten()
        else:
            # Need to convert to dense for threshold comparison
            expressing_cells_per_gene = np.array((expression_matrix.toarray() > min_expression).sum(axis=0))
    else:
        # Dense matrix - direct threshold comparison
        expressing_cells_per_gene = (expression_matrix > min_expression).sum(axis=0)
    
    # Create boolean mask for active genes
    active_mask = expressing_cells_per_gene >= min_cells
    
    # Get active gene names
    active_gene_names = [gene_names[i] for i in range(len(gene_names)) if active_mask[i]]
    
    if return_mask:
        return active_gene_names, active_mask
    else:
        return active_gene_names


def _validate_expression_data(
    expression_data: Union[np.ndarray, sparse.csr_matrix, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    group_labels: Optional[Union[List[str], pd.Series]] = None
) -> Tuple[np.ndarray, List[str], Optional[pd.Series]]:
    """
    Internal helper function to validate and standardize expression data inputs.
    
    This function ensures consistent data formats across the package and performs
    basic validation checks on input data dimensions and types.
    
    Parameters
    ----------
    expression_data : numpy.ndarray, scipy.sparse.csr_matrix, or pandas.DataFrame
        Gene expression matrix with cells as rows and genes as columns.
    gene_names : list of str, optional
        List of gene names corresponding to columns in expression_data.
    group_labels : list of str or pandas.Series, optional
        Group/cell type labels for each cell (row) in expression_data.
        
    Returns
    -------
    tuple
        Tuple containing (expression_matrix, gene_names, group_labels) where:
        - expression_matrix is a numpy.ndarray
        - gene_names is a list of strings
        - group_labels is a pandas.Series or None
        
    Raises
    ------
    ValueError
        If input data dimensions are inconsistent or invalid.
    TypeError
        If input data types are not supported.
    """
    # Handle DataFrame input
    if isinstance(expression_data, pd.DataFrame):
        if gene_names is not None:
            raise ValueError("gene_names should not be provided when expression_data is a DataFrame")
        gene_names = expression_data.columns.tolist()
        expression_matrix = expression_data.values
    elif isinstance(expression_data, (np.ndarray, sparse.csr_matrix)):
        if gene_names is None:
            raise ValueError("gene_names must be provided when expression_data is not a DataFrame")
        if len(gene_names) != expression_data.shape[1]:
            raise ValueError(f"Length of gene_names ({len(gene_names)}) must match number of columns in expression_data ({expression_data.shape[1]})")
        
        # Convert sparse to dense for consistency
        if sparse.issparse(expression_data):
            expression_matrix = expression_data.toarray()
        else:
            expression_matrix = expression_data
    else:
        raise TypeError(f"Unsupported expression_data type: {type(expression_data)}")
    
    # Validate dimensions
    if expression_matrix.ndim != 2:
        raise ValueError(f"expression_data must be 2-dimensional, got {expression_matrix.ndim} dimensions")
    
    if expression_matrix.shape[0] == 0 or expression_matrix.shape[1] == 0:
        raise ValueError("expression_data cannot be empty")
    
    # Handle group labels
    if group_labels is not None:
        if isinstance(group_labels, list):
            group_labels = pd.Series(group_labels)
        elif not isinstance(group_labels, pd.Series):
            raise TypeError(f"group_labels must be a list or pandas.Series, got {type(group_labels)}")
        
        if len(group_labels) != expression_matrix.shape[0]:
            raise ValueError(f"Length of group_labels ({len(group_labels)}) must match number of rows in expression_data ({expression_matrix.shape[0]})")
    
    return expression_matrix, gene_names, group_labels


def _prepare_data_inputs(
    expression_data: Union[np.ndarray, sparse.csr_matrix, pd.DataFrame],
    gene_names: Optional[List[str]] = None,
    group_labels: Optional[Union[List[str], pd.Series]] = None
) -> Tuple[np.ndarray, List[str], pd.Series]:
    """
    Prepare and validate input data for analysis classes.
    
    This is the main data preparation function used by GeneProfiler and other
    analysis classes to ensure consistent input handling.
    
    Parameters
    ----------
    expression_data : numpy.ndarray, scipy.sparse.csr_matrix, or pandas.DataFrame
        Gene expression matrix with cells as rows and genes as columns.
    gene_names : list of str, optional
        List of gene names corresponding to columns in expression_data.
    group_labels : list of str or pandas.Series, optional
        Group/cell type labels for each cell (row) in expression_data.
        
    Returns
    -------
    tuple
        Tuple containing (expression_matrix, gene_names, group_labels) where:
        - expression_matrix is a numpy.ndarray
        - gene_names is a list of strings  
        - group_labels is a pandas.Series
        
    Raises
    ------
    ValueError
        If group_labels is None or if input data is invalid.
    """
    expression_matrix, gene_names, group_labels = _validate_expression_data(
        expression_data, gene_names, group_labels
    )
    
    if group_labels is None:
        raise ValueError("group_labels is required for analysis")
    
    return expression_matrix, gene_names, group_labels
