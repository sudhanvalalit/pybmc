# pybmc
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
To install the package, you can use `poetry`:

```sh
poetry add pybmc
```

## Usage
Here is an example of how to use the package:

```python
from pybmc.models import Model
from pybmc.data import Dataset
from pybmc.bmc import BayesianModelCombination

# Create models
model1 = Model("model1")
model2 = Model("model2")

# Load data
dataset = Dataset(data_source)
data = dataset.load_data()

# Split data
train_data, val_data, test_data = dataset.split_data(train_size=0.6, val_size=0.2, test_size=0.2)

# Orthogonalize models (optional)
model1.orthogonalize(train_data)
model2.orthogonalize(train_data)

# Create Bayesian model combination
bmc = BayesianModelCombination(models=[model1, model2])

# Train the model combination
bmc.train(train_data)

# Predict using the model combination
predictions = bmc.predict(X)

# Evaluate the model combination
evaluation = bmc.evaluate(val_data)
```

## License
This project is licensed under the terms of the GNU General Public License v3.0. See the LICENSE file for details.
