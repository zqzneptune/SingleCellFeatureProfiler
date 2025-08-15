# Single-Cell Feature Profiler

[![PyPI version](https://badge.fury.io/py/scfeatureprofiler.svg)](https://badge.fury.io/py/scfeatureprofiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/pypi/pyversions/scfeatureprofiler.svg)](https://pypi.org/project/scfeatureprofiler)

A powerful, fast, and user-friendly Python package for deep characterization of single-cell feature expression patterns.

`scfeatureprofiler` provides a suite of statistical tools to analyze single-cell data (e.g., scRNA-seq, CITE-seq) and answer fundamental biological questions:

1.  **Cluster Quality:** Are my clusters well-defined and biologically meaningful?
2.  **Marker Discovery:** Which features are robust and specific markers for each cell group?
3.  **Feature Activity:** In which cell groups is a specific feature actively expressed?

The package is designed for performance, with a parallelized backend that can handle extremely large datasets, including out-of-core analysis for data that doesn't fit into memory.

## Key Features

-   **Multi-Interface:** Use it as a Python library in your Jupyter notebooks or as a command-line tool for script-based workflows.
-   **Flexible Input:** Works directly with `AnnData` objects, `pandas.DataFrame`, or `numpy` arrays.
-   **Robust Cluster Validation:** Includes an `evaluate_clustering` function using silhouette scores to quantify cluster quality *before* marker discovery.
-   **Data-Driven Marker Selection:** Implements a dynamic, clustering-based method to automatically identify the best markers without arbitrary thresholds.
-   **High Performance:** Parallelized using `joblib` to use all available CPU cores for rapid analysis.
-   **Scalable:** Supports out-of-core computation for memory-mapped `AnnData` objects, enabling analysis of millions of cells.

## Installation

You can install `scfeatureprofiler` directly from PyPI:

```bash
pip install scfeatureprofiler
```

To include support for `AnnData` objects (recommended), install with the `[anndata]` extra:

```bash
pip install scfeatureprofiler[anndata]
```

To install all dependencies for development, use:
```bash
# Clone the repository first
git clone https://github.com/zqzneptune/SingleCellFeatureProfiler.git
cd SingleCellFeatureProfiler
pip install -e ".[all]"
```

## Quick Start

`scfeatureprofiler` is designed to be intuitive. Here are two examples for the most common use cases.

### 1. Python API: The Complete Marker Discovery Workflow

This is the recommended workflow inside a Jupyter notebook after you have performed clustering.

```python
import scanpy as sc
from scfeatureprofiler import evaluate_clustering, find_marker_features, select_robust_markers

# 1. Load your clustered single-cell data
adata = sc.read_h5ad("path/to/your_clustered_data.h5ad")

# 2. (Recommended) Evaluate clustering quality first
#    This helps ensure your clusters are meaningful before finding markers.
cluster_report = evaluate_clustering(adata, cluster_key='leiden')
#    A good cluster should have a silhouette score > 0.25.

# 3. Find all potential marker features for your clusters
#    This returns a comprehensive pandas DataFrame for deep exploration.
all_markers_df = find_marker_features(
    data=adata,
    group_by='leiden'
)

# 4. Automatically select the top 10 best markers per cluster
#    This function uses a data-driven method to find natural cutoffs.
top_markers_df = select_robust_markers(all_markers_df, top_n=10)

print("--- Top 5 Robust Markers for each Cluster ---")
print(top_markers_df.groupby('group').head(5))

# 5. Convert to a dictionary for Scanpy plotting functions
top_markers_dict = top_markers_df.groupby('group')['feature_id'].apply(list).to_dict()
sc.pl.dotplot(adata, top_markers_dict, groupby='leiden')
```

### 2. Command-Line (CLI): Find and Rank Markers

If you prefer to work from the terminal, you can perform the entire marker discovery pipeline with a single command.

**Input File:**
-   `my_data.h5ad`: An AnnData file with clustering results in `adata.obs['leiden']`.

**Command:**

```bash
scfeatureprofiler find-markers \
    --input my_data.h5ad \
    --group-by leiden \
    --output ranked_markers.csv
```

**Output (`ranked_markers.csv`):**
This produces a detailed CSV file with all statistically significant markers, ranked by group and significance.

```csv
feature_id,group,stability_score,norm_score,pct_expressing,log2fc_all,fdr_marker,...
CD8A,CD8 T-cell,1.0,1.0,95.4,8.2,1.2e-250,...
GZMB,CD8 T-cell,1.0,0.98,92.1,7.5,4.5e-245,...
MS4A1,B-cell,1.0,1.0,98.2,9.5,8.1e-280,...
...
```

## Available CLI Commands

-   **`scfeatureprofiler find-markers`**: A full pipeline to select, profile, and rank robust marker features.
-   **`scfeatureprofiler profile`**: Generate a detailed statistical profile for a user-provided list of features.
-   **`scfeatureprofiler activity`**: Summarize a profile to show in which groups features are "ON".

Use `scfeatureprofiler --help` or `scfeatureprofiler find-markers --help` for a full list of options.