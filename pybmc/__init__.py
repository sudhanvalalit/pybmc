# This file makes the `pybmc` directory a package.
"""
pybmc: Bayesian Model Combination toolkit

Classes:
- Model: A model defined by input/output data
- Dataset: Handles loading and preparing nuclear model datasets
- BayesianModelCombination: Combines models using Bayesian inference

Error models (see `pybmc.error_models`):
- homoscedastic (constant variance) and six heteroscedastic variants
  whose variance depends on distance in principal-component space
  and/or the spread among model predictions.
"""

from .data import Dataset
from .bmc import BayesianModelCombination
from .inference_utils import (
    gibbs_sampler,
    gibbs_sampler_simplex,
    gibbs_sampler_heteroscedastic,
    USVt_hat_extraction,
)
from .sampling_utils import (
    coverage,
    coverage_quality,
    diagnose_coverage_shape,
    mace,
    reduced_chi_square,
    DEFAULT_PREDICTIVE_SEED,
)
from .error_models import (
    VARIANCE_MODELS,
    HeteroscedasticMetrics,
    required_metrics,
    variance_basis,
    variance_parameter_names,
)


__all__ = [
    "Model",
    "Dataset",
    "BayesianModelCombination",
    "gibbs_sampler",
    "gibbs_sampler_simplex",
    "gibbs_sampler_heteroscedastic",
    "USVt_hat_extraction",
    "coverage",
    "coverage_quality",
    "diagnose_coverage_shape",
    "mace",
    "reduced_chi_square",
    "DEFAULT_PREDICTIVE_SEED",
    "VARIANCE_MODELS",
    "HeteroscedasticMetrics",
    "required_metrics",
    "variance_basis",
    "variance_parameter_names",
]
