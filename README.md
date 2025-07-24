# pybmc [![CI Status](https://github.com/ascsn/pybmc/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/ascsn/pybmc/actions/workflows/ci-cd.yml)

Simple package for Bayesian model combination

## Overview
This package implements a Bayesian model combination strategy with the following features:
1. A set of models
2. Valid on a given domain X
   - X can be a vector
3. With a prediction Y
   - Y(X) can, in principle, be a vector itself (masses, radii, etc)
4. Optional orthogonalization step
5. Training, validation, and optional testing sets defined on a subset of X
   - It's also possible that we don't have a full Y(X) vector across each of these sets
6. A training method that determines model weights given a training set
7. A prediction method that, given a valid X vector, produces Y with the learned model weights

## Installation

### From PyPI (after release)
```bash
pip install pybmc
```

### From source
```bash
git clone https://github.com/ascsn/pybmc.git
cd pybmc
poetry install
```

## Documentation
Full documentation is available at: [https://ascsn.github.io/pybmc/](https://ascsn.github.io/pybmc/)

## Usage
Here is an example of how to use the package:

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

## License
This project is licensed under the terms of the GNU General Public License v3.0. See the LICENSE file for details.
