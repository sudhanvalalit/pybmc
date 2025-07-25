# pyBMC: Bayesian Model Combination for Nuclear Mass Predictions

pyBMC is a Python package for performing Bayesian Model Combination (BMC) on nuclear mass models. It provides tools for data handling, orthogonalization, Gibbs sampling, and prediction with uncertainty quantification.

## Features

- **Data Management**: Load and preprocess nuclear mass data from HDF5 and CSV files
- **Orthogonalization**: Transform model predictions using Singular Value Decomposition (SVD)
- **Bayesian Inference**: Perform Gibbs sampling for model combination
- **Uncertainty Quantification**: Generate predictions with credible intervals
- **Model Evaluation**: Calculate coverage statistics for model validation

## Installation

```bash
pip install pybmc
```

## Quick Start

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

# Split data into training, validation, and test sets
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

# Make predictions with uncertainty quantification
rndm_m, lower_df, median_df, upper_df = bmc.predict2("Binding_Energy")

# Evaluate model performance
coverage_results = bmc.evaluate()
```

## Documentation

Comprehensive documentation is available at [https://ascsn.github.io/pybmc/](https://ascsn.github.io/pybmc/), including:

- API Reference
- Usage Guides
- Theory Background
- Tutorial Notebooks

## Contributing

We welcome contributions! Please see our [Contribution Guidelines](CONTRIBUTING.md) for details on how to contribute to the project.

## License

This project is licensed under the GPL V3 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use pyBMC in your research, please cite:

```bibtex
@software{pybmc,
  title = {pyBMC: Bayesian Model Combination},
  author = {Kyle Godbey and Troy Dasher and An Le},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/ascsn/pybmc}}
}
```

## Support

For questions or support, please open an issue on our [GitHub repository](https://github.com/ascsn/pybmc/issues).
