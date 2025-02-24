# Usage

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

## Explanation of Each Step

1. **Create models**: We create two instances of the `Model` class, `model1` and `model2`, with their respective domains and outputs.
2. **Load data**: We create an instance of the `Dataset` class and load data from a specified source.
3. **Split data**: We split the loaded data into training, validation, and testing sets.
4. **Create Bayesian model combination**: We create an instance of the `BayesianModelCombination` class with the created models and an option to use orthogonalization.
5. **Orthogonalize models (optional)**: We orthogonalize the models using the training data if the orthogonalization option is enabled.
6. **Train the model combination**: We train the Bayesian model combination using the training data.
7. **Predict using the model combination**: We use the trained model combination to make predictions for a given input.
8. **Evaluate the model combination**: We evaluate the performance of the model combination using the validation data.
