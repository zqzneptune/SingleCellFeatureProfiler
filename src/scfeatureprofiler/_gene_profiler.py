#!/usr/bin/env python

"""
Internal implementation of the GeneProfiler class.

This module contains the GeneProfiler class which performs deep-dive statistical
profiling for a specific list of genes across different cell groups.
"""

from typing import List, Union, Optional, Dict, Any
import warnings

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from ._core import _Profiler
from ._utils import _prepare_data_inputs


class GeneProfiler:
    """
    Performs deep-dive statistical profiling for a specific list of genes.

    This class is optimized for answering the question: "For this specific set of
    genes, what are their expression profiles and statistics across all cell groups?"

    The GeneProfiler accepts expression data in multiple formats (AnnData, numpy arrays,
    or sparse matrices) and provides comprehensive statistical analysis including
    expression fractions, geometric means, normalized scores, and significance testing.

    Parameters
    ----------
    expression_data : Union[AnnData, np.ndarray, csr_matrix]
        The single-cell expression matrix. If AnnData, uses the .X attribute.
        If array-like, should be cells x genes format.
    group_labels : np.ndarray
        A 1D array of group labels (e.g., cell type names), one for each cell.
        Must have the same length as the number of cells in expression_data.
    gene_names : Optional[List[str]], optional
        A list of gene names corresponding to the columns in expression_data.
        Required if expression_data is not an AnnData object. By default None.
    donor_labels : Optional[np.ndarray], optional
        A 1D array of donor IDs, one for each cell. If provided, results will
        be averaged across donors within each group. By default None.

    Attributes
    ----------
    n_cells : int
        Number of cells in the dataset.
    n_genes : int
        Number of genes in the dataset.
    gene_names : List[str]
        List of gene names.
    group_names : List[str]
        Unique group names in the dataset.

    Examples
    --------
    >>> import numpy as np
    >>> from scfeatureprofiler import GeneProfiler
    >>> 
    >>> # Create sample data
    >>> expression = np.random.poisson(2, (1000, 500))  # 1000 cells, 500 genes
    >>> groups = np.random.choice(['TypeA', 'TypeB', 'TypeC'], 1000)
    >>> genes = [f'Gene_{i}' for i in range(500)]
    >>> 
    >>> # Initialize profiler
    >>> profiler = GeneProfiler(expression, groups, genes)
    >>> 
    >>> # Run analysis on specific genes
    >>> results = profiler.run(['Gene_1', 'Gene_2', 'Gene_10'])
    >>> print(results.head())
    """

    def __init__(
        self,
        expression_data: Union[AnnData, np.ndarray, csr_matrix],
        group_labels: np.ndarray,
        gene_names: Optional[List[str]] = None,
        donor_labels: Optional[np.ndarray] = None,
    ):
        """
        Initialize the GeneProfiler with expression data and group labels.

        Parameters
        ----------
        expression_data : Union[AnnData, np.ndarray, csr_matrix]
            The single-cell expression matrix (cells x genes).
        group_labels : np.ndarray
            A 1D array of group labels, one for each cell.
        gene_names : Optional[List[str]], optional
            A list of gene names. Required if expression_data is not an AnnData object.
            By default None.
        donor_labels : Optional[np.ndarray], optional
            A 1D array of donor IDs, one for each cell. By default None.

        Raises
        ------
        ValueError
            If input validation fails or dimensions don't match.
        TypeError
            If input types are not supported.
        """
        # Validate and prepare the input data
        self._prepared_data = self._prepare_data_inputs(
            expression_data, group_labels, gene_names, donor_labels
        )
        
        # Set public attributes for easy access
        self.n_cells = self._prepared_data['expression'].shape[0]
        self.n_genes = self._prepared_data['expression'].shape[1]
        self.gene_names = self._prepared_data['genes']
        self.group_names = sorted(list(set(self._prepared_data['group_labels'])))
        
        # Store donor information if provided
        self.has_donors = self._prepared_data['donor_labels'] is not None
        if self.has_donors:
            self.donor_names = sorted(list(set(self._prepared_data['donor_labels'])))

    def _prepare_data_inputs(
        self,
        expression_data: Union[AnnData, np.ndarray, csr_matrix],
        group_labels: np.ndarray,
        gene_names: Optional[List[str]] = None,
        donor_labels: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Validate and prepare input data for the profiler.

        Parameters
        ----------
        expression_data : Union[AnnData, np.ndarray, csr_matrix]
            The expression matrix in various formats.
        group_labels : np.ndarray
            Array of group labels for each cell.
        gene_names : Optional[List[str]], optional
            List of gene names, by default None.
        donor_labels : Optional[np.ndarray], optional
            Array of donor labels for each cell, by default None.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing validated and prepared data components.

        Raises
        ------
        ValueError
            If validation fails or dimensions don't match.
        TypeError
            If input types are not supported.
        """
        prepared = {}
        
        # Handle different input formats for expression data
        if isinstance(expression_data, AnnData):
            # Extract from AnnData object
            prepared['expression'] = expression_data.X
            prepared['genes'] = expression_data.var_names.tolist()
            n_cells, n_genes = expression_data.shape
            
        elif isinstance(expression_data, (np.ndarray, csr_matrix)) or issparse(expression_data):
            # Handle numpy arrays and sparse matrices
            if gene_names is None:
                raise ValueError(
                    "gene_names must be provided when expression_data is not an AnnData object"
                )
            
            prepared['expression'] = expression_data
            prepared['genes'] = list(gene_names)
            n_cells, n_genes = expression_data.shape
            
            # Validate gene_names length
            if len(gene_names) != n_genes:
                raise ValueError(
                    f"Length of gene_names ({len(gene_names)}) does not match "
                    f"number of genes in expression_data ({n_genes})"
                )
        else:
            raise TypeError(
                f"expression_data must be AnnData, numpy.ndarray, or scipy.sparse matrix, "
                f"got {type(expression_data)}"
            )
        
        # Validate group_labels
        if not isinstance(group_labels, np.ndarray):
            group_labels = np.array(group_labels)
        
        if group_labels.ndim != 1:
            raise ValueError("group_labels must be a 1D array")
        
        if len(group_labels) != n_cells:
            raise ValueError(
                f"Length of group_labels ({len(group_labels)}) does not match "
                f"number of cells in expression_data ({n_cells})"
            )
        
        prepared['group_labels'] = group_labels
        
        # Validate donor_labels if provided
        if donor_labels is not None:
            if not isinstance(donor_labels, np.ndarray):
                donor_labels = np.array(donor_labels)
            
            if donor_labels.ndim != 1:
                raise ValueError("donor_labels must be a 1D array")
            
            if len(donor_labels) != n_cells:
                raise ValueError(
                    f"Length of donor_labels ({len(donor_labels)}) does not match "
                    f"number of cells in expression_data ({n_cells})"
                )
        
        prepared['donor_labels'] = donor_labels
        
        # Check for any NaN or infinite values in group labels
        if np.any(pd.isna(group_labels)):
            raise ValueError("group_labels contains NaN values")
        
        # Check for empty groups
        unique_groups = np.unique(group_labels)
        if len(unique_groups) < 2:
            warnings.warn(
                f"Only {len(unique_groups)} unique group(s) found. "
                "Statistical comparisons may not be meaningful.",
                UserWarning
            )
        
        return prepared

    def run(self, genes: List[str], fdr_threshold: float = 0.05) -> pd.DataFrame:
        """
        Run the profiling analysis for a given list of genes.

        This method performs comprehensive statistical profiling of the specified genes
        across all cell groups, including expression fractions, geometric means,
        normalized scores, specificity measures, and significance testing.

        Parameters
        ----------
        genes : List[str]
            A list of gene IDs to profile. All genes must be present in the dataset.
        fdr_threshold : float, optional
            The False Discovery Rate threshold for significance testing, by default 0.05.
            Must be between 0 and 1.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the full statistical profile for the requested genes.
            Columns include:
            - gene_id: Gene identifier
            - group_name: Cell group name
            - norm_score: Normalized expression score (0-1)
            - pct_expressing: Percentage of cells expressing the gene
            - fdr_presence: FDR-corrected p-value for presence test
            - fdr_marker: FDR-corrected p-value for marker test (if applicable)
            - log2fc_marker: Log2 fold change for marker test (if applicable)
            - specificity_tau: Tau specificity score (0-1, higher = more specific)

        Raises
        ------
        ValueError
            If genes are not found in the dataset or fdr_threshold is invalid.
        RuntimeError
            If the profiling calculation fails.

        Examples
        --------
        >>> # Run analysis on specific genes
        >>> results = profiler.run(['ACTB', 'GAPDH', 'CD3E'])
        >>> 
        >>> # Filter for significant results
        >>> significant = results[results['fdr_presence'] < 0.05]
        >>> 
        >>> # Find highly specific genes
        >>> specific = results[results['specificity_tau'] > 0.8]
        """
        # Validate inputs
        if not isinstance(genes, list):
            raise TypeError("genes must be a list of gene names")
        
        if not genes:
            raise ValueError("genes list cannot be empty")
        
        if not (0 < fdr_threshold <= 1):
            raise ValueError("fdr_threshold must be between 0 and 1")
        
        # Check that all requested genes are in the dataset
        missing_genes = [gene for gene in genes if gene not in self.gene_names]
        if missing_genes:
            raise ValueError(
                f"The following genes were not found in the dataset: {missing_genes}"
            )
        
        # Instantiate the core engine
        engine = _Profiler(
            expression_data=self._prepared_data['expression'],
            group_labels=self._prepared_data['group_labels'],
            gene_names=self._prepared_data['genes'],
            donor_labels=self._prepared_data['donor_labels']
        )
        
        # Execute the engine's calculation method
        try:
            results_df = engine.run_calculations(
                gene_subset=genes,
                fdr_threshold=fdr_threshold
            )
        except Exception as e:
            raise RuntimeError(f"Profiling calculation failed: {str(e)}") from e
        
        return results_df

    def find_markers_by_specificity(
        self, 
        genes: List[str], 
        specificity_threshold: float = 0.7,
        expression_threshold: float = 10.0,
        fdr_threshold: float = 0.05
    ) -> Dict[str, List[str]]:
        """
        Find marker genes for each group using specificity_tau cutoff.

        This method identifies marker genes for each cell group by applying
        specificity and expression thresholds. Genes with high specificity_tau
        scores (closer to 1) are considered more specific to particular groups.

        Parameters
        ----------
        genes : List[str]
            A list of gene IDs to evaluate for marker potential.
        specificity_threshold : float, optional
            Minimum specificity_tau score for a gene to be considered a marker,
            by default 0.7. Values closer to 1 indicate higher specificity.
        expression_threshold : float, optional
            Minimum percentage of cells expressing the gene in the target group,
            by default 10.0.
        fdr_threshold : float, optional
            Maximum FDR-corrected p-value for presence significance,
            by default 0.05.

        Returns
        -------
        Dict[str, List[str]]
            A dictionary mapping each group name to its list of marker genes.
            Groups with no markers will have empty lists.

        Raises
        ------
        ValueError
            If input parameters are invalid or genes are not found.

        Examples
        --------
        >>> # Find markers using default thresholds
        >>> markers = profiler.find_markers_by_specificity(['Gene1', 'Gene2', 'Gene3'])
        >>> print(f"TypeA markers: {markers['TypeA']}")
        >>> 
        >>> # Use stricter thresholds
        >>> strict_markers = profiler.find_markers_by_specificity(
        ...     genes=['Gene1', 'Gene2'], 
        ...     specificity_threshold=0.8,
        ...     expression_threshold=20.0
        ... )
        """
        # Validate inputs
        if not isinstance(genes, list):
            raise TypeError("genes must be a list of gene names")
        
        if not genes:
            raise ValueError("genes list cannot be empty")
        
        if not (0 <= specificity_threshold <= 1):
            raise ValueError("specificity_threshold must be between 0 and 1")
        
        if not (0 <= expression_threshold <= 100):
            raise ValueError("expression_threshold must be between 0 and 100")
        
        if not (0 < fdr_threshold <= 1):
            raise ValueError("fdr_threshold must be between 0 and 1")
        
        # Run the full profiling analysis
        results_df = self.run(genes, fdr_threshold)
        
        # Initialize markers dictionary for all groups
        markers_dict = {group: [] for group in self.group_names}
        
        # Filter results based on thresholds
        filtered_results = results_df[
            (results_df['specificity_tau'] >= specificity_threshold) &
            (results_df['pct_expressing'] >= expression_threshold) &
            (results_df['fdr_presence'] <= fdr_threshold)
        ]
        
        # For each gene, find the group with highest normalized score
        # (this identifies which group the gene is most specific to)
        for gene_id in genes:
            gene_data = filtered_results[filtered_results['gene_id'] == gene_id]
            
            if not gene_data.empty:
                # Find the group with the highest normalized expression score
                best_group_idx = gene_data['norm_score'].idxmax()
                best_group = gene_data.loc[best_group_idx, 'group_name']
                
                # Add gene as marker for this group
                markers_dict[best_group].append(gene_id)
        
        # Sort marker lists for consistent output
        for group in markers_dict:
            markers_dict[group].sort()
        
        return markers_dict
