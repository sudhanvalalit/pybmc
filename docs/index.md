# pyBMC Documentation

Welcome to the official documentation for pyBMC, a Python package for Bayesian Model Combination (BMC) with a focus on nuclear mass predictions.

## Overview

pyBMC provides a comprehensive framework for combining multiple predictive models using Bayesian statistics. Key features include:

- **Data Management**: Load and preprocess nuclear mass data from HDF5 and CSV files
- **Orthogonalization**: Transform model predictions using Singular Value Decomposition (SVD)
- **Bayesian Inference**: Perform Gibbs sampling for model combination
- **Uncertainty Quantification**: Generate predictions with credible intervals
- **Model Evaluation**: Calculate coverage statistics for model validation

## Getting Started

### Installation

```bash
pip install pybmc
```

### Quick Start

```python
from pybmc import Dataset, BayesianModelCombination

# Load nuclear mass data
dataset = Dataset("nuclear_data.h5")
data_dict = dataset.load_data(
    models=["FRDM2012", "WS4", "HFB32", "D1M", "UNEDF1", "BCPM"],
    keys=["Binding_Energy"],
    domain_keys=["N", "Z"]
)

# Initialize BMC
bmc = BayesianModelCombination(
    models_list=["FRDM2012", "WS4", "HFB32", "D1M", "UNEDF1", "BCPM"],
    data_dict=data_dict,
    truth_column_name="Binding_Energy"
)

# Split data
train_df, val_df, test_df = dataset.split_data(
    data_dict,
    "Binding_Energy",
    splitting_algorithm="random",
    train_size=0.6,
    val_size=0.2,
    test_size=0.2
)

# Orthogonalize model predictions
bmc.orthogonalize("Binding_Energy", train_df, components_kept=3)

# Train the model combination
bmc.train(training_options={
    'iterations': 50000,
    'sampler': 'gibbs_sampling'
})

# Make predictions
rndm_m, lower_df, median_df, upper_df = bmc.predict2("Binding_Energy")

# Evaluate model performance
coverage_results = bmc.evaluate()
```

## Documentation Contents

- [Usage Guide](usage.md): Detailed examples and tutorials
- [API Reference](api_reference.md): Complete documentation of all classes and functions
- [Theory Background](theory.md): Mathematical foundations of Bayesian model combination
- [Contributing](CONTRIBUTING.md): How to contribute to pyBMC

## Support

For questions or support, please open an issue on our [GitHub repository](https://github.com/ascsn/pybmc/issues).

## License

This project is licensed under the GPL V3 License - see the [LICENSE](../LICENSE) file for details.
