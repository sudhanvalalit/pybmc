# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Heteroscedastic error models: the noise variance can now depend on the
  distance from the training data in principal-component space
  (`pc_dist`) and/or on the disagreement among model predictions
  (`model_var`), linearly or quadratically (new `error_model` parameter
  of `BayesianModelCombination`, new `pybmc.error_models` module and
  `gibbs_sampler_heteroscedastic` Gibbs-within-Metropolis sampler with
  burn-in proposal adaptation toward a target acceptance rate)
- Posterior predictive sampling with per-point variances
  (`rndm_m_heteroscedastic_calculator`)
- Calibration diagnostics: `coverage_quality` (mean |empirical - nominal|
  coverage) and `diagnose_coverage_shape` (under-/over-dispersion
  classification)
- `mh_acceptance_rate_` attribute on `BayesianModelCombination` after
  heteroscedastic training

### Fixed
- `get_weights()` now slices the coefficient columns by the number of
  kept components instead of assuming a single trailing noise parameter
- `evaluate()` now excludes points without truth values, as documented
- Comprehensive docstrings for all public classes and functions
- CONTRIBUTING.md with contribution guidelines
- CHANGELOG.md to track project changes
- AUTHORS.md to credit contributors
- Improved README with better documentation and examples

### Changed
- Updated API documentation in docs/api_reference.md
- Improved usage examples in docs/usage.md
- Standardized docstrings to Google style throughout the codebase

### Fixed
- Minor bug fixes in orthogonalization implementation
- Improved error handling in data loading functions

## [0.1.0] - 2025-07-24

### Added
- Initial release of pybmc
- Core functionality for Bayesian model combination
- Data loading and preprocessing capabilities
- Orthogonalization using SVD
- Gibbs sampling implementation
- Prediction with uncertainty quantification
- Basic documentation structure
