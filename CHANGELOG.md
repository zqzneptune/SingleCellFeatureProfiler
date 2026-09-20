# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2026-09-20

### Added

- Canonical feature profiles with explicit target-versus-all, one, and set
  contrasts and separate coverage, effect, specificity, discrimination, and
  cell-level statistical evidence.
- Biological-replicate summaries, distinct cell/sample resampling, transparent
  downstream selection, independent profile comparison, annotation-evidence
  evaluation and visualization, and clustering geometry diagnostics.
- Explicit AnnData `X`, raw, and layer selection; tested backed access; bounded
  feature chunking and shared-memory profiling workers.
- Thin canonical CLI adapters, complete reference documentation, and a runnable
  seven-scenario scientific tutorial.

### Changed

- The primary `profile` CLI command now returns the canonical profile schema.
  The prior result contract remains available through the hidden
  `legacy-profile` command and `get_feature_profiles()` Python function.
- `select_robust_markers()` and `evaluate_clustering()` are compatibility
  wrappers with deprecation warnings; canonical alternatives avoid composite
  marker scoring and biological-validity claims from clustering geometry.
- Runtime metadata now declares `anndata` and `scikit-learn` directly instead of
  relying on the unused `scanpy` dependency to provide them transitively.

### Versioning

- Reconciled the observed `1.1.2` baseline and the historical product-plan
  discrepancy as a `2.0.0` release. The major version signals the breaking
  primary CLI profile-contract change.

## [1.1.0] - 2025-07-27

### Added
- **Novel Marker Stability Score**: The `find-markers` command now calculates a stability score (1 - CV) for each marker, quantifying its robustness across different biological conditions. This provides a powerful metric for selecting reliable biomarkers.
- **Enhanced Verbose Output**: Added detailed, step-by-step progress messages to the `find-markers` pipeline and improved the clarity of `joblib`'s parallel processing logs.

### Changed
- **BREAKING CHANGE**: The `find-markers` command now **requires** a `--condition-by` argument to perform stability analysis. The `batch_by` argument has been removed and replaced with this clearer, more biologically relevant parameter.
- **BREAKING CHANGE**: The output of `find-markers` is now a ranked `pandas.DataFrame` instead of a dictionary, including the new `stability_score`.
- **Workflow Refactoring**:
    - The `profile` command no longer defaults to HVG selection. It now profiles all features by default, with a clear warning and a 5-second countdown to give users control over this computationally intensive task.
    - The `activity` command is now a lightweight post-processing utility that operates on the CSV output of `profile`, rather than performing redundant computations.
- **Performance Optimization**: The core statistical function (`_analyze_one_feature`) was refactored to use vectorized NumPy operations, significantly improving performance by removing pandas overhead in the main parallel loop.

### Fixed
- **Memory Outage**: Resolved a critical memory leak in the stability score calculation by moving aggregation logic from the main process into the parallel workers.
- **Data Type Errors**: Fixed a recurring `TypeError` in `binomtest` by ensuring its inputs are always correctly cast as integers.
- **API and CLI Bugs**: Corrected numerous `ImportError`, `NameError`, and `KeyError` issues that arose during the refactoring process, ensuring all commands and API calls work as intended.

## [1.0.0] - 2025-07-25

This is a major redesign of the package with a new, more powerful, and user-friendly API.

### Added
- New, intuitive user-facing API functions: `get_feature_profiles`, `get_feature_activity`, and `find_marker_features`.
- A full Command-Line Interface (CLI) accessible via the `scprofiler` command.
- High-performance parallel backend using `joblib` for massive speed improvements.
- Support for out-of-core computation on "backed" AnnData objects, enabling analysis of datasets larger than memory.
- Support for Gini coefficient as an alternative specificity metric (`specificity_metric='gini'`).
- Comprehensive unit and integration test suite using `pytest`.
- Official support for Python 3.9 through 3.12.

### Changed
- **BREAKING CHANGE:** Complete overhaul of the original `GeneProfiler` class-based API.
- Renamed all "gene" terminology to the more generic "feature" to reflect broader applicability (e.g., CITE-seq).
- Project structure now uses a `src` layout.
- `pyproject.toml` updated with optimized dependencies and configurations.

### Removed
- **BREAKING CHANGE:** The `GeneProfiler` class has been removed in favor of the new functional API.
- Removed the hard dependency on `scanpy` for statistical tests. Wilcoxon tests are now handled natively with `scipy`.
- `anndata` is now an optional dependency, making the core package lighter.
