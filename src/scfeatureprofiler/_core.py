#!/usr/bin/env python

"""
Internal core profiling engine.

This module contains the _Profiler class which implements the core statistical
calculations for gene expression profiling. This is an internal implementation
detail and should not be used directly by end users.
"""

from typing import List, Union, Optional, Dict, Any, Tuple
import warnings

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

# Suppress runtime warnings about invalid values, which we handle explicitly
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered")


class _Profiler:
    """
    Internal profiling engine for statistical calculations.
    
    This class encapsulates the core logic for computing expression statistics,
    performing significance tests, and generating the final results DataFrame.
    It is designed to be used internally by the public API classes.
    
    Parameters
    ----------
    expression_data : Union[np.ndarray, csr_matrix]
        The expression matrix (cells x genes).
    group_labels : np.ndarray
        Array of group labels for each cell.
    gene_names : List[str]
        List of gene names.
    donor_labels : Optional[np.ndarray], optional
        Array of donor labels for each cell, by default None.
    """
    
    def __init__(
        self,
        expression_data: Union[np.ndarray, csr_matrix],
        group_labels: np.ndarray,
        gene_names: List[str],
        donor_labels: Optional[np.ndarray] = None,
    ):
        """Initialize the profiler with validated data."""
        self.expression_data = expression_data
        self.group_labels = group_labels
        self.gene_names = gene_names
        self.donor_labels = donor_labels
        self.expression_threshold = 0  # Threshold for considering a gene "expressed"
        
        # Convert to dense array if sparse for easier processing
        if issparse(self.expression_data):
            self.expression_dense = self.expression_data.toarray()
        else:
            self.expression_dense = self.expression_data
    
    @staticmethod
    def _geometric_mean(x: pd.Series) -> float:
        """
        Calculate geometric mean of positive values in a series.
        
        Parameters
        ----------
        x : pd.Series
            Series of expression values.
            
        Returns
        -------
        float
            Geometric mean of positive values, 0.0 if no positive values.
        """
        x_positive = x[x > 0]
        return np.exp(np.log(x_positive).mean()) if not x_positive.empty else 0.0
    
    @staticmethod
    def _calculate_tau(x: pd.Series) -> float:
        """
        Calculate tau specificity score.
        
        The tau score measures tissue/group specificity, ranging from 0 (broadly expressed)
        to 1 (highly specific). Higher values indicate more specific expression.
        
        Parameters
        ----------
        x : pd.Series
            Series of expression values across groups.
            
        Returns
        -------
        float
            Tau specificity score (0-1).
        """
        if x.max() <= 0:
            return 1.0  # Convention for non-expressed genes
        x_norm = x / x.max()
        return (1 - x_norm).sum() / (len(x) - 1)
    
    def _compute_stats_for_chunk(
        self, 
        expr_chunk_df: pd.DataFrame, 
        group_keys: List[str]
    ) -> pd.DataFrame:
        """
        Calculate core metrics for a chunk of expression data.
        
        This is the core computational method that efficiently calculates
        expression statistics using pandas groupby operations.
        
        Parameters
        ----------
        expr_chunk_df : pd.DataFrame
            DataFrame containing both metadata (group_keys) and expression data.
        group_keys : List[str]
            List of column names to group by (e.g., ['group_name'] or ['group_name', 'donor_id']).
            
        Returns
        -------
        pd.DataFrame
            DataFrame with calculated statistics for each gene-group combination.
        """
        # Melt to long format for efficient grouping
        melted_df = expr_chunk_df.melt(
            id_vars=group_keys, 
            var_name='gene_id', 
            value_name='expression'
        )
        
        # Group by all keys and calculate statistics
        grouped = melted_df.groupby(group_keys + ['gene_id'], observed=True)
        
        # Use .agg for efficient, parallel computation of stats
        agg_results = grouped['expression'].agg(
            frac_expr=lambda x: (x > self.expression_threshold).mean(),
            geo_mean=self._geometric_mean
        ).reset_index()
        
        # Calculate derived scores
        agg_results['raw_score'] = agg_results['geo_mean'] * agg_results['frac_expr']
        agg_results['pct_expressing'] = agg_results['frac_expr'] * 100
        
        return agg_results.drop(columns=['frac_expr', 'geo_mean'])
    
    def _compute_patterns_in_chunks(
        self, 
        gene_subset: List[str], 
        chunk_size: int = 500
    ) -> pd.DataFrame:
        """
        Compute expression patterns for genes in memory-efficient chunks.
        
        Parameters
        ----------
        gene_subset : List[str]
            List of genes to analyze.
        chunk_size : int, optional
            Number of genes to process per chunk, by default 500.
            
        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with expression statistics.
        """
        # Get indices of requested genes
        gene_indices = [self.gene_names.index(gene) for gene in gene_subset]
        
        # Prepare metadata DataFrame
        if self.donor_labels is not None:
            obs_df = pd.DataFrame({
                'group_name': self.group_labels,
                'donor_id': self.donor_labels
            })
            group_keys = ['group_name', 'donor_id']
        else:
            obs_df = pd.DataFrame({'group_name': self.group_labels})
            group_keys = ['group_name']
        
        # Process genes in chunks
        num_chunks = int(np.ceil(len(gene_indices) / chunk_size))
        all_results_long = []
        
        for i, chunk_start in enumerate(range(0, len(gene_indices), chunk_size)):
            chunk_end = chunk_start + chunk_size
            chunk_gene_indices = gene_indices[chunk_start:chunk_end]
            chunk_gene_names = [gene_subset[j] for j in range(chunk_start, min(chunk_end, len(gene_subset)))]
            
            # Extract expression data for this chunk
            expr_chunk = self.expression_dense[:, chunk_gene_indices]
            expr_chunk_df = pd.DataFrame(expr_chunk, columns=chunk_gene_names)
            
            # Combine with metadata
            df_for_computation = pd.concat([obs_df, expr_chunk_df], axis=1)
            
            # Compute statistics for this chunk
            chunk_summary = self._compute_stats_for_chunk(df_for_computation, group_keys)
            all_results_long.append(chunk_summary)
        
        # Combine all chunks
        df_long = pd.concat(all_results_long, ignore_index=True)
        
        # If we have donors, aggregate across donors within each group
        if self.donor_labels is not None:
            agg_keys = ['gene_id', 'group_name']
            df_long = df_long.groupby(agg_keys, observed=True).agg(
                raw_score=('raw_score', 'mean'),
                pct_expressing=('pct_expressing', 'mean')
            ).reset_index()
        
        return df_long
    
    def _calculate_normalized_scores_and_specificity(
        self, 
        df_long: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate normalized scores and specificity measures.
        
        Parameters
        ----------
        df_long : pd.DataFrame
            Long-format DataFrame with raw scores.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with added normalized scores and specificity measures.
        """
        # Pivot to calculate normalized scores across groups
        raw_score_pivot = df_long.pivot(
            index='gene_id', 
            columns='group_name', 
            values='raw_score'
        )
        
        # Calculate normalized scores (0-1 scaling within each gene)
        norm_scores = raw_score_pivot.apply(
            lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) > 0 else 0, 
            axis=1
        )
        norm_scores_long = norm_scores.stack().reset_index(name='norm_score')
        
        # Calculate specificity (tau) scores
        specificity = pd.DataFrame({
            'specificity_tau': raw_score_pivot.apply(self._calculate_tau, axis=1)
        }, index=raw_score_pivot.index)
        
        # Merge back with original data
        df_long = pd.merge(df_long, norm_scores_long, on=['gene_id', 'group_name'])
        df_long = pd.merge(df_long, specificity, on='gene_id')
        
        # Drop raw_score as it's not needed in final output
        df_long = df_long.drop(columns=['raw_score'])
        
        return df_long
    
    def _calculate_presence_significance(
        self, 
        gene_subset: List[str], 
        background_rate: float = 0.01
    ) -> pd.DataFrame:
        """
        Calculate statistical significance of gene presence using binomial test.
        
        Parameters
        ----------
        gene_subset : List[str]
            List of genes to test.
        background_rate : float, optional
            Expected background expression rate, by default 0.01.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with presence test results.
        """
        # Get gene indices
        gene_indices = [self.gene_names.index(gene) for gene in gene_subset]
        
        # Count cells per group
        unique_groups, group_counts = np.unique(self.group_labels, return_counts=True)
        cell_counts = dict(zip(unique_groups, group_counts))
        
        # Calculate expression presence matrix
        expr_subset = self.expression_dense[:, gene_indices]
        is_expressed = expr_subset > self.expression_threshold
        
        # Create DataFrame for easier grouping
        presence_df = pd.DataFrame(is_expressed, columns=gene_subset)
        presence_df['group_name'] = self.group_labels
        
        # Count expressing cells per group
        expressing_counts = presence_df.groupby('group_name', observed=True).sum(numeric_only=True)
        
        # Perform binomial tests
        binomial_results = []
        for group_name in expressing_counts.index:
            n_total_cells = cell_counts[group_name]
            for gene in expressing_counts.columns:
                k_expressing_cells = int(expressing_counts.loc[group_name, gene])
                result = binomtest(
                    k=k_expressing_cells, 
                    n=n_total_cells, 
                    p=background_rate, 
                    alternative='greater'
                )
                binomial_results.append({
                    'gene_id': gene, 
                    'group_name': group_name, 
                    'p_value': result.pvalue
                })
        
        # Apply FDR correction
        pvals_df = pd.DataFrame(binomial_results)
        pvals_df['fdr_presence'] = multipletests(pvals_df['p_value'], method='fdr_bh')[1]
        
        return pvals_df[['gene_id', 'group_name', 'fdr_presence']]
    
    def _calculate_marker_significance(
        self, 
        gene_subset: List[str]
    ) -> pd.DataFrame:
        """
        Calculate marker gene significance using Wilcoxon rank-sum test.
        
        This method performs one-vs-rest comparisons for each group to identify
        genes that are significantly upregulated in that group compared to all others.
        
        Parameters
        ----------
        gene_subset : List[str]
            List of genes to test.
            
        Returns
        -------
        pd.DataFrame
            DataFrame with marker test results.
        """
        try:
            import scanpy as sc
            from anndata import AnnData
            
            # Get gene indices
            gene_indices = [self.gene_names.index(gene) for gene in gene_subset]
            
            # Create temporary AnnData object for scanpy
            expr_subset = self.expression_dense[:, gene_indices]
            temp_adata = AnnData(X=expr_subset)
            temp_adata.var_names = gene_subset
            temp_adata.obs['group_name'] = self.group_labels
            
            # Run Wilcoxon test
            sc.tl.rank_genes_groups(
                temp_adata, 
                groupby='group_name', 
                method='wilcoxon', 
                use_raw=False
            )
            
            # Extract results
            de_results = temp_adata.uns['rank_genes_groups']
            marker_stats = []
            
            for group in de_results['names'].dtype.names:
                for i, gene in enumerate(de_results['names'][group]):
                    if gene in gene_subset:  # Only include requested genes
                        marker_stats.append({
                            'gene_id': gene,
                            'group_name': group,
                            'fdr_marker': de_results['pvals_adj'][group][i],
                            'log2fc_marker': de_results['logfoldchanges'][group][i]
                        })
            
            return pd.DataFrame(marker_stats)
            
        except ImportError:
            # If scanpy is not available, return empty results
            warnings.warn(
                "scanpy not available, skipping marker gene analysis", 
                UserWarning
            )
            
            # Create empty results with correct structure
            unique_groups = np.unique(self.group_labels)
            empty_results = []
            for group in unique_groups:
                for gene in gene_subset:
                    empty_results.append({
                        'gene_id': gene,
                        'group_name': group,
                        'fdr_marker': np.nan,
                        'log2fc_marker': np.nan
                    })
            
            return pd.DataFrame(empty_results)
    
    def run_calculations(
        self, 
        gene_subset: List[str], 
        fdr_threshold: float = 0.05,
        background_rate: float = 0.01,
        chunk_size: int = 500
    ) -> pd.DataFrame:
        """
        Run the complete profiling calculation pipeline.
        
        This is the main method that orchestrates all the statistical calculations
        and returns the final results DataFrame.
        
        Parameters
        ----------
        gene_subset : List[str]
            List of genes to profile.
        fdr_threshold : float, optional
            FDR threshold for significance, by default 0.05.
        background_rate : float, optional
            Background expression rate for presence test, by default 0.01.
        chunk_size : int, optional
            Chunk size for memory-efficient processing, by default 500.
            
        Returns
        -------
        pd.DataFrame
            Complete results DataFrame with all statistics.
        """
        # Step 1: Compute core expression patterns
        df_long = self._compute_patterns_in_chunks(gene_subset, chunk_size)
        
        # Step 2: Calculate normalized scores and specificity
        df_long = self._calculate_normalized_scores_and_specificity(df_long)
        
        # Step 3: Calculate presence significance
        presence_results = self._calculate_presence_significance(gene_subset, background_rate)
        df_long = pd.merge(df_long, presence_results, on=['gene_id', 'group_name'])
        
        # Step 4: Calculate marker significance
        marker_results = self._calculate_marker_significance(gene_subset)
        df_long = pd.merge(df_long, marker_results, on=['gene_id', 'group_name'], how='left')
        
        # Step 5: Ensure consistent column order and types
        final_cols = [
            'gene_id', 'group_name', 'norm_score', 'pct_expressing',
            'fdr_presence', 'fdr_marker', 'log2fc_marker', 'specificity_tau'
        ]
        
        # Fill any missing values
        df_long['fdr_marker'] = df_long['fdr_marker'].fillna(1.0)
        df_long['log2fc_marker'] = df_long['log2fc_marker'].fillna(0.0)
        
        return df_long[final_cols]
