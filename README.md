# Single-Cell Feature Profiler

[![PyPI version](https://badge.fury.io/py/scfeatureprofiler.svg)](https://pypi.org/project/scfeatureprofiler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python versions](https://img.shields.io/pypi/pyversions/scfeatureprofiler.svg)](https://pypi.org/project/scfeatureprofiler)

`scfeatureprofiler` is an interpretable framework for measuring molecular
evidence connecting features to supplied cellular identities. It keeps coverage,
specificity, effect, discrimination, cell-level statistical evidence,
biological-sample reproducibility, and resampling uncertainty individually
accessible instead of forcing them into one marker score.

## Why separate evidence dimensions?

- High expression does not imply specificity.
- High specificity does not imply broad target coverage.
- A tiny effect can be highly significant when many cells are treated as
  observations.
- A pooled effect can be driven by one donor rather than reproduced across
  biological samples.
- One-versus-rest can hide behavior against the most relevant alternative.
- Clustering geometry in one representation does not establish biological label
  correctness.

The package therefore requires explicit contrasts, records the selected AnnData
expression source, retains ambiguous or contradictory evidence, and keeps cell
and biological-sample resampling distinct.

## Installation

```bash
pip install scfeatureprofiler
```

For development:

```bash
git clone https://github.com/zqzneptune/SingleCellFeatureProfiler.git
cd SingleCellFeatureProfiler
pip install -e ".[dev]"
```

The methodological refactor is versioned as `2.0.0`; the major version signals
the changed primary CLI profile contract. Historical Python workflows remain
available through compatibility interfaces described below.

## Canonical Python workflow

```python
import anndata as ad
from scfeatureprofiler import (
    Contrast,
    SelectionCriteria,
    evaluate_annotation_evidence,
    profile_features_by_sample,
    select_profile_features,
)

adata = ad.read_h5ad("cells.h5ad")
contrast = Contrast.target_vs_one("CD8 T-cell", "CD4 T-cell")

result = profile_features_by_sample(
    adata,
    group_by="cell_type",
    sample_by="donor",
    features=["CD8A", "GZMB", "NKG7"],
    contrasts=[contrast],
    expression_source="layer:log1p",
)

# Apply only thresholds chosen for this analysis; none are biological defaults.
selection = select_profile_features(
    result.profiles,
    SelectionCriteria(
        min_target_detection_fraction=0.5,
        min_mean_difference=0.5,
        min_effect_recurrence_fraction=0.75,
    ),
)

# Assess a user-supplied label and expectations; this does not auto-annotate.
evidence = evaluate_annotation_evidence(
    result.profiles,
    proposed_label="CD8 T-cell",
    expected_support_features=["CD8A"],
)
```

`result.profiles` contains pooled feature-target-contrast evidence.
`result.sample_profiles` separately retains every within-sample relationship,
including insufficient samples. Selection returns both complete rule evaluation
and selected rows; failed or contradictory evidence is not erased.

## Explicit contrasts and expression sources

Canonical profiles support target-versus-all, target-versus-one, and
target-versus-set contrasts. AnnData users choose `X`, `raw`, or
`layer:<name>` explicitly; unavailable sources are errors rather than fallbacks.
Dense NumPy, pandas, SciPy sparse, and AnnData inputs have equivalent numerical
semantics where applicable.

## Command line

The CLI calls the same Python APIs:

```bash
scfeatureprofiler profile \
    --input cells.h5ad \
    --group-by cell_type \
    --target "CD8 T-cell" \
    --alternative "CD4 T-cell" \
    --features CD8A,GZMB,NKG7 \
    --expression-source layer:log1p \
    --output profiles.csv

scfeatureprofiler select \
    --profiles profiles.csv \
    --min-target-detection-fraction 0.5 \
    --min-mean-difference 0.5 \
    --output selected.csv
```

`profile-samples` writes pooled and per-sample evidence separately.
`diagnostics` writes cell, group, and overall clustering-geometry tables.
Historical `find-markers`, `activity`, `get_feature_profiles()`, and
`select_robust_markers()` interfaces remain for compatibility; new analyses
should use canonical profiles and explicit selection.

## Performance boundaries

Canonical profiling supports bounded feature chunks and shared-memory joblib
workers. Sparse matrices remain sparse between feature extractions, while the
statistics for each active feature use a dense cell-length vector. Backed
AnnData `X`, raw data, and layers are tested, but this is not a general
out-of-core guarantee. Small profiles may be faster with the serial default;
benchmark representative data before increasing `n_jobs`.

## Documentation

- [Documentation index](docs/README.md)
- [Seven reproducible validation scenarios](docs/tutorials/seven-validation-scenarios.md)
- [Canonical output schema](docs/reference/canonical-profile-schema.md)
- [Biological-replicate evidence](docs/reference/replicate-profile-schema.md)
- [Resampling and uncertainty](docs/reference/resampling.md)
- [Annotation evidence and visualization](docs/reference/annotation-evidence.md)
- [Clustering geometry diagnostics](docs/reference/clustering-diagnostics.md)
- [CLI reference](docs/reference/cli.md)
- [Measured performance behavior](docs/reference/performance.md)

## Scope

The package does not download CELLxGENE data, infer annotations or ontologies,
integrate modalities, model trajectories, or impose a formal cell-type/state
solution. Independent datasets can be compared through completed profile tables
without integrating their expression matrices.
