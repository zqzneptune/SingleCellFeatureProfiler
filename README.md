# Single-Cell Feature Profiler

[![PyPI version](https://badge.fury.io/py/scfeatureprofiler.svg)](https://badge.fury.io/py/scfeatureprofiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python versions](https://img.shields.io/pypi/pyversions/scfeatureprofiler.svg)](https://pypi.org/project/scfeatureprofiler)

A powerful, fast, and multi-interface Python package for deep characterization of single-cell feature expression patterns.

`scfeatureprofiler` provides a suite of statistical tools to analyze single-cell data (e.g., scRNA-seq, CITE-seq) and answer two fundamental questions:
1.  **Feature Activity:** In which cell groups is a feature actively expressed?
2.  **Marker Discovery:** Which features are specific markers for each cell group?

The package is designed for performance, with a parallelized backend that can handle extremely large datasets, including out-of-core analysis for data that doesn't fit into memory.

## Key Features

-   **Multi-Interface:** Use it as a Python library in your Jupyter notebooks or as a command-line tool for script-based workflows.
-   **Flexible Input:** Works directly with `AnnData` objects, `pandas.DataFrame`, or `numpy` arrays.
-   **Comprehensive Statistics:** Calculates normalized expression scores, percentage of expressing cells, specificity scores (Tau and Gini), and robust statistical significance (FDR).
-   **High Performance:** Parallelized using `joblib` to use all available CPU cores for rapid analysis.
-   **Scalable:** Natively supports out-of-core computation for `AnnData` objects stored on disk, enabling analysis of millions of cells.
-   **Lightweight:** Minimal dependencies, making it easy to integrate into existing analysis environments.

## Installation

You can install `scfeatureprofiler` directly from PyPI:

```bash
pip install scfeatureprofiler
```

To include support for `AnnData` objects, install with the `[anndata]` extra:

```bash
pip install scfeatureprofiler[anndata]
```

To install all dependencies for development, use:
```bash
# Clone the repository first
git clone https://github.com/zqzneptune/SingleCellFeatureProfiler.git
cd SingleCellFeatureProfiler
pip install -e .[all]
```

## Quick Start

`scfeatureprofiler` is designed to be intuitive. Here are two examples for the most common use cases.

### 1. Python API: Find Marker Genes for Clusters

This is the most common use case inside a Jupyter notebook after you have performed clustering.

```python
import anndata
from scfeatureprofiler import find_marker_features

# 1. Load your clustered single-cell data
#    (This example assumes you have an AnnData object)
adata = anndata.read_h5ad("path/to/your_clustered_data.h5ad")

# 2. Find marker features for your clusters
#    'leiden' is the column in adata.obs containing cluster labels.
marker_dict = find_marker_features(
    data=adata,
    group_by='leiden'
)

# 3. Print the results
for cluster, markers in marker_dict.items():
    print(f"Cluster {cluster} Markers: {', '.join(markers[:10])}...")

# 4. (Optional) Use the results directly with Scanpy for plotting
import scanpy as sc
sc.pl.dotplot(adata, marker_dict, groupby='leiden')
```

### 2. Command-Line (CLI): Get Full Profiles for a Gene List

If you have a CSV file of expression data and want to get a detailed statistical report for a few genes of interest without writing a script.

**Input Files:**
-   `expression.csv`: A cells-by-genes matrix.
-   `cell_groups.csv`: A file mapping cell IDs to group labels.

**Command:**

```bash
scprofiler profile \
    --input expression.csv \
    --group-by cell_groups.csv \
    --features "CD4,CD8A,GNLY,MS4A1" \
    --output gene_profiles.csv
```

**Output (`gene_profiles.csv`):**
This will produce a detailed CSV file with statistics for each gene in each cell group, ready for analysis in Excel or another program.

```
feature_id,group,norm_score,pct_expressing,fdr_presence,fdr_marker,...
CD4,T-cell Helper,0.98,85.4,1.2e-50,3.4e-30,...
CD4,B-cell,0.05,2.1,0.89,1.0,...
CD8A,T-cell Cytoxic,0.99,92.1,4.5e-60,8.1e-45,...
...
```

## Available CLI Commands

-   `scprofiler profile`: Generate a full statistical profile for features.
-   `scprofiler activity`: Identify in which groups a list of features are "ON".

Use `scprofiler --help` or `scprofiler profile --help` for a full list of options.
