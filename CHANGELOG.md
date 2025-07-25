# Changelog

All notable changes to this project will be documented in this file.

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