# API Reference

## Model Class

### `Model`

A class representing a model with a domain (x) and an output (y).

#### Attributes

- `name` (str): The name of the model.
- `x` (np.ndarray): The domain of the model.
- `y` (np.ndarray): The output of the model.

#### Methods

- `__init__(self, name, x, y)`: Initialize the Model object.

  - **Args:**
    - `name` (str): The name of the model.
    - `x` (array-like): The domain of the model.
    - `y` (array-like): The output of the model.

## Dataset Class

### `Dataset`

A class representing a dataset.

#### Attributes

- `data` (any): The data of the dataset.

#### Methods

- `__init__(self, data)`: Initialize the Dataset object.

  - **Args:**
    - `data` (any): The data of the dataset.

- `load_data(self, source)`: Load data from a given source.

  - **Args:**
    - `source` (str): The source of the data.

- `split_data(self, train_size, val_size, test_size)`: Split data into training, validation, and testing sets.

  - **Args:**
    - `train_size` (float): The proportion of the data to include in the training set.
    - `val_size` (float): The proportion of the data to include in the validation set.
    - `test_size` (float): The proportion of the data to include in the testing set.

- `get_subset(self, domain_X)`: Return a subset of data for a given domain X.

  - **Args:**
    - `domain_X` (any): The domain for which to return the subset of data.

## BayesianModelCombination Class

### `BayesianModelCombination`

A class representing a Bayesian model combination.

#### Attributes

- `models` (list or np.ndarray): The list or array of models.
- `options` (dict): The options for the model combination.
- `weights` (any): The weights of the models.

#### Methods

- `__init__(self, models, options=None)`: Initialize the BayesianModelCombination object.

  - **Args:**
    - `models` (list or np.ndarray): The list or array of models.
    - `options` (dict, optional): The options for the model combination. Defaults to None.

- `train(self, training_data)`: Train the model combination using training data.

  - **Args:**
    - `training_data` (any): The training data.

- `predict(self, X)`: Produce predictions using the learned model weights.

  - **Args:**
    - `X` (any): The input data.

- `evaluate(self, data)`: Evaluate the model combination on validation or testing data.

  - **Args:**
    - `data` (any): The data to evaluate the model combination on.

- `orthogonalize(self, data)`: Orthogonalize the models using the given data.

  - **Args:**
    - `data` (any): The data to use for orthogonalization.
