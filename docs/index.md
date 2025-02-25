# Welcome to pybmc Documentation

<div class="grid cards" markdown>

- :material-function-variant: **Bayesian model combination**
- :material-math-integral: **Statistical inference**
- :material-chart-bar: **Ensemble modeling**
- :material-code-array: **Simple Python API**

</div>

## Introduction

**pybmc** is a Python package that implements a Bayesian model combination strategy with the following features:

- **Versatile**: Works with any set of models on a given domain
- **Flexible**: Handles vector inputs and predictions
- **Powerful**: Includes optional orthogonalization steps
- **Complete**: Training, validation, and testing functions built-in

<div class="grid" markdown>
<div markdown>

## Why Bayesian Model Combination?

Bayesian model combination (BMC) provides a principled approach to combining predictions from multiple models. Unlike simple averaging or voting techniques, BMC:

1. Accounts for correlations between models
2. Assigns optimal weights based on model performance
3. Provides uncertainty estimates for predictions
4. Is robust against overfitting

</div>
<div markdown>

```python
# Quick example
from pybmc import BayesianModelCombination, Model

# Create models
model1 = Model("linear", predictions_train, targets_train)
model2 = Model("neural", predictions_train, targets_train)

# Set up BMC
bmc = BayesianModelCombination([model1, model2])
bmc.train()

# Get combined predictions
predictions = bmc.predict(new_data)
```

</div>
</div>

## Installation

Install the package using pip:

```bash
pip install pybmc
```

Or using poetry:

```bash
poetry add pybmc
```

## Features

| Feature | Description |
| ------- | ----------- |
| Multiple models | Combine any number of pre-trained models |
| Custom domains | Works with any input domain X (can be a vector) |
| Vector predictions | Y(X) can be a vector itself (e.g., masses, radii) |
| Orthogonalization | Optional step to improve model combination |
| Flexible datasets | Training, validation, and testing with partial data |

## Usage Example

```python
import numpy as np
from pybmc.models import Model
from pybmc.data import Dataset
from pybmc.bmc import BayesianModelCombination

# Create models
model1 = Model("model1", np.array([1, 2, 3]), np.array([10, 20, 30]))
model2 = Model("model2", np.array([1, 2, 3]), np.array([15, 25, 35]))

# Load data
data_source = "path/to/data_source"
dataset = Dataset(data_source)
data = dataset.load_data(data_source)

# Split data
train_data, val_data, test_data = dataset.split_data(train_size=0.6, val_size=0.2, test_size=0.2)

# Create Bayesian model combination
bmc = BayesianModelCombination(models=[model1, model2], options={'use_orthogonalization': True})

# Orthogonalize models (optional)
bmc.orthogonalize(train_data)

# Train the model combination
bmc.train(train_data)

# Predict using the model combination
X = np.array([1, 2, 3])
predictions = bmc.predict(X)

# Evaluate the model combination
evaluation = bmc.evaluate(val_data)
```

!!! tip "Getting Started"
    Check out the [Usage](usage.md) page for more detailed examples and the [API Reference](api_reference.md) for complete documentation.
