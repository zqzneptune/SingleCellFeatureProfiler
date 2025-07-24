#!/usr/bin/env python

"""
Main profiler CLI for SingleCellFeatureProfiler.

This script provides a command-line interface for running comprehensive
gene expression profiling analysis using the SingleCellFeatureProfiler package.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    from anndata import AnnData
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    AnnData = object  # Placeholder for type hints
    sc = None

from scfeatureprofiler import GeneProfiler, get_active_genes


def load_data(file_path: str, backed: bool = False) -> object:
    """
    Load single-cell data from various formats.
    
    Parameters
    ----------
    file_path : str
        Path to the input data file.
    backed : bool, default False
        Whether to load data in backed mode for memory efficiency.
        
    Returns
    -------
    AnnData
        Loaded AnnData object.
        
    Raises
    ------
    ImportError
        If anndata/scanpy is not available.
    FileNotFoundError
        If the file doesn't exist.
    ValueError
        If the file format is not supported.
    """
    if not ANNDATA_AVAILABLE:
        raise ImportError("anndata and scanpy are required for data loading. Install with: pip install anndata scanpy")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    
    file_path = Path(file_path)
    
    try:
        if file_path.suffix.lower() == '.h5ad':
            adata = sc.read_h5ad(file_path, backed='r' if backed else None)
        elif file_path.suffix.lower() in ['.h5', '.hdf5']:
            adata = sc.read_h5(file_path)
        elif file_path.suffix.lower() == '.xlsx':
            adata = sc.read_excel(file_path)
        elif file_path.suffix.lower() in ['.csv', '.tsv', '.txt']:
            delimiter = '\t' if file_path.suffix.lower() in ['.tsv', '.txt'] else ','
            adata = sc.read_csv(file_path, delimiter=delimiter)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
    except Exception as e:
        raise ValueError(f"Failed to load data from {file_path}: {e}")
    
    return adata


def validate_metadata_columns(adata: object, group_col: str, donor_col: Optional[str] = None) -> None:
    """
    Validate that required metadata columns exist in the AnnData object.
    
    Parameters
    ----------
    adata : AnnData
        The AnnData object to validate.
    group_col : str
        Name of the group/cell type column.
    donor_col : str, optional
        Name of the donor/sample column.
        
    Raises
    ------
    ValueError
        If required columns are missing.
    """
    available_cols = list(adata.obs.columns)
    
    if group_col not in available_cols:
        raise ValueError(
            f"Group column '{group_col}' not found in data. "
            f"Available columns: {available_cols}"
        )
    
    if donor_col and donor_col not in available_cols:
        raise ValueError(
            f"Donor column '{donor_col}' not found in data. "
            f"Available columns: {available_cols}"
        )


def preprocess_data(adata: object, normalize: bool = True, log_transform: bool = True) -> object:
    """
    Preprocess the single-cell data.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    normalize : bool, default True
        Whether to perform library size normalization.
    log_transform : bool, default True
        Whether to apply log1p transformation.
        
    Returns
    -------
    AnnData
        Preprocessed AnnData object.
    """
    # Work on a copy to avoid modifying the original
    adata_proc = adata.copy()
    
    # Check if data appears to be raw counts
    is_raw = np.issubdtype(adata_proc.X.dtype, np.integer) and adata_proc.X.max() > 20
    
    if is_raw and normalize:
        print("INFO: Raw integer counts detected. Performing library size normalization.")
        sc.pp.normalize_total(adata_proc, target_sum=1e4)
    
    if is_raw and log_transform:
        print("INFO: Applying log1p transformation.")
        sc.pp.log1p(adata_proc)
    
    return adata_proc


def filter_data(
    adata: object,
    group_col: str,
    min_cells_per_group: int = 50,
    min_genes_per_cell: int = 200,
    min_cells_per_gene: int = 10
) -> object:
    """
    Filter cells and genes based on quality criteria.
    
    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    group_col : str
        Name of the group column.
    min_cells_per_group : int, default 50
        Minimum number of cells required per group.
    min_genes_per_cell : int, default 200
        Minimum number of genes detected per cell.
    min_cells_per_gene : int, default 10
        Minimum number of cells expressing each gene.
        
    Returns
    -------
    AnnData
        Filtered AnnData object.
    """
    print(f"INFO: Starting with {adata.n_obs} cells and {adata.n_vars} genes")
    
    # Filter cells by gene count
    if min_genes_per_cell > 0:
        sc.pp.filter_cells(adata, min_genes=min_genes_per_cell)
        print(f"INFO: After filtering cells with <{min_genes_per_cell} genes: {adata.n_obs} cells")
    
    # Filter genes by cell count
    if min_cells_per_gene > 0:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)
        print(f"INFO: After filtering genes expressed in <{min_cells_per_gene} cells: {adata.n_vars} genes")
    
    # Filter groups by cell count
    if min_cells_per_group > 0:
        group_counts = adata.obs[group_col].value_counts()
        valid_groups = group_counts[group_counts >= min_cells_per_group].index
        
        if len(valid_groups) < len(group_counts):
            print(f"INFO: Filtering groups with <{min_cells_per_group} cells")
            print(f"INFO: Keeping {len(valid_groups)} of {len(group_counts)} groups")
            
            mask = adata.obs[group_col].isin(valid_groups)
            adata = adata[mask, :].copy()
            
            # Remove unused categories if categorical
            if hasattr(adata.obs[group_col], 'cat'):
                adata.obs[group_col] = adata.obs[group_col].cat.remove_unused_categories()
    
    print(f"INFO: Final dataset: {adata.n_obs} cells and {adata.n_vars} genes")
    return adata


def run_analysis(
    adata: object,
    group_col: str,
    donor_col: Optional[str] = None,
    genes: Optional[List[str]] = None,
    fdr_threshold: float = 0.05,
    max_genes: Optional[int] = None
) -> pd.DataFrame:
    """
    Run the gene profiling analysis.
    
    Parameters
    ----------
    adata : AnnData
        Preprocessed AnnData object.
    group_col : str
        Name of the group column.
    donor_col : str, optional
        Name of the donor column.
    genes : List[str], optional
        Specific genes to analyze. If None, analyzes active genes.
    fdr_threshold : float, default 0.05
        FDR threshold for significance testing.
    max_genes : int, optional
        Maximum number of genes to analyze (for computational efficiency).
        
    Returns
    -------
    pd.DataFrame
        Analysis results.
    """
    # Determine genes to analyze
    if genes is None:
        print("INFO: No specific genes provided, identifying active genes...")
        active_genes = get_active_genes(
            adata.X,
            gene_names=adata.var_names.tolist(),
            min_cells=max(10, adata.n_obs // 100),  # Adaptive threshold
            min_expression=0.0
        )
        
        if max_genes and len(active_genes) > max_genes:
            print(f"INFO: Limiting analysis to {max_genes} most variable genes")
            # Use scanpy to find highly variable genes
            adata_temp = adata.copy()
            sc.pp.highly_variable_genes(adata_temp, n_top_genes=max_genes)
            hvg_genes = adata_temp.var[adata_temp.var.highly_variable].index.tolist()
            genes = [g for g in hvg_genes if g in active_genes][:max_genes]
        else:
            genes = active_genes
        
        print(f"INFO: Analyzing {len(genes)} active genes")
    else:
        print(f"INFO: Analyzing {len(genes)} user-specified genes")
    
    # Initialize profiler
    group_labels = adata.obs[group_col].values
    donor_labels = adata.obs[donor_col].values if donor_col else None
    
    profiler = GeneProfiler(
        expression_data=adata,
        group_labels=group_labels,
        donor_labels=donor_labels
    )
    
    # Run analysis
    print("INFO: Running statistical analysis...")
    results = profiler.run(genes, fdr_threshold=fdr_threshold)
    
    return results


def save_results(results: pd.DataFrame, output_prefix: str, metadata: dict) -> None:
    """
    Save analysis results and metadata.
    
    Parameters
    ----------
    results : pd.DataFrame
        Analysis results.
    output_prefix : str
        Output file prefix.
    metadata : dict
        Analysis metadata.
    """
    # Save main results
    results_file = f"{output_prefix}_results.tsv"
    results.to_csv(results_file, sep='\t', index=False, float_format='%.6g')
    print(f"INFO: Results saved to {results_file}")
    
    # Save metadata
    metadata_file = f"{output_prefix}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"INFO: Metadata saved to {metadata_file}")


def main():
    """Main entry point for the profiler CLI."""
    parser = argparse.ArgumentParser(
        description="SingleCellFeatureProfiler: Comprehensive gene expression profiling",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/output
    parser.add_argument(
        "input_file",
        help="Path to input single-cell data file (.h5ad, .h5, .csv, .tsv, .xlsx)"
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Prefix for output files"
    )
    
    # Data columns
    parser.add_argument(
        "--group-col",
        required=True,
        help="Column name for cell groups/types in the data"
    )
    parser.add_argument(
        "--donor-col",
        help="Column name for donor/sample IDs (optional)"
    )
    
    # Gene selection
    parser.add_argument(
        "--genes",
        nargs='+',
        help="Specific genes to analyze (if not provided, analyzes active genes)"
    )
    parser.add_argument(
        "--max-genes",
        type=int,
        default=5000,
        help="Maximum number of genes to analyze (for computational efficiency)"
    )
    
    # Analysis parameters
    parser.add_argument(
        "--fdr-threshold",
        type=float,
        default=0.05,
        help="FDR threshold for significance testing"
    )
    
    # Filtering parameters
    parser.add_argument(
        "--min-cells-per-group",
        type=int,
        default=50,
        help="Minimum cells required per group"
    )
    parser.add_argument(
        "--min-genes-per-cell",
        type=int,
        default=200,
        help="Minimum genes detected per cell"
    )
    parser.add_argument(
        "--min-cells-per-gene",
        type=int,
        default=10,
        help="Minimum cells expressing each gene"
    )
    
    # Preprocessing options
    parser.add_argument(
        "--skip-normalization",
        action='store_true',
        help="Skip library size normalization (assume data is pre-normalized)"
    )
    parser.add_argument(
        "--skip-log-transform",
        action='store_true',
        help="Skip log transformation (assume data is already log-transformed)"
    )
    
    # Performance options
    parser.add_argument(
        "--backed",
        action='store_true',
        help="Load data in backed mode for memory efficiency"
    )
    
    args = parser.parse_args()
    
    start_time = time.time()
    metadata = {
        "command_line_args": vars(args),
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scfeatureprofiler_version": "0.1.0"
    }
    
    try:
        print("=== SingleCellFeatureProfiler Analysis ===")
        print(f"Input file: {args.input_file}")
        print(f"Output prefix: {args.output_prefix}")
        print()
        
        # Load data
        print("Loading data...")
        adata = load_data(args.input_file, backed=args.backed)
        metadata["original_shape"] = adata.shape
        print(f"Loaded {adata.n_obs} cells and {adata.n_vars} genes")
        
        # Validate metadata columns
        validate_metadata_columns(adata, args.group_col, args.donor_col)
        
        # Convert to memory if backed
        if args.backed:
            print("Converting to memory for processing...")
            adata = adata.to_memory()
        
        # Preprocess data
        print("\nPreprocessing data...")
        adata = preprocess_data(
            adata,
            normalize=not args.skip_normalization,
            log_transform=not args.skip_log_transform
        )
        
        # Filter data
        print("\nFiltering data...")
        adata = filter_data(
            adata,
            args.group_col,
            min_cells_per_group=args.min_cells_per_group,
            min_genes_per_cell=args.min_genes_per_cell,
            min_cells_per_gene=args.min_cells_per_gene
        )
        metadata["filtered_shape"] = adata.shape
        
        # Run analysis
        print("\nRunning analysis...")
        results = run_analysis(
            adata,
            args.group_col,
            args.donor_col,
            args.genes,
            args.fdr_threshold,
            args.max_genes
        )
        
        metadata["n_results"] = len(results)
        metadata["n_genes_analyzed"] = results['gene_id'].nunique()
        metadata["n_groups"] = results['group_name'].nunique()
        
        # Save results
        print("\nSaving results...")
        metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        metadata["total_runtime_seconds"] = time.time() - start_time
        
        save_results(results, args.output_prefix, metadata)
        
        print(f"\n=== Analysis Complete ===")
        print(f"Runtime: {metadata['total_runtime_seconds']:.1f} seconds")
        print(f"Results: {len(results)} entries for {results['gene_id'].nunique()} genes")
        print(f"Output files: {args.output_prefix}_results.tsv, {args.output_prefix}_metadata.json")
        
    except (FileNotFoundError, ValueError, ImportError) as e:
        print(f"Error: {e}", file=sys.stderr)
        metadata["error"] = str(e)
        metadata["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Save error metadata if possible
        try:
            error_file = f"{args.output_prefix}_error_metadata.json"
            with open(error_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            print(f"Error metadata saved to {error_file}", file=sys.stderr)
        except:
            pass
        
        sys.exit(1)
    
    except KeyboardInterrupt:
        print("\nAnalysis cancelled by user.", file=sys.stderr)
        sys.exit(1)
    
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
