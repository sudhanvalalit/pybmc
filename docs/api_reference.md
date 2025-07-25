# API Reference

## pybmc.data

### `Dataset`
```python
class Dataset:
    """
    Handles loading and preprocessing of nuclear mass data.

    Args:
        data_source (str): Path to data file (HDF5 or CSV)

    Methods:
        load_data(models, keys, domain_keys): Load data for specified models
        split_data(data_dict, target_key, splitting_algorithm, **kwargs): Split data into train/val/test sets
    """
```

## pybmc.bmc

### `BayesianModelCombination`
```python
class BayesianModelCombination:
    """
    Main class for Bayesian Model Combination.

    Args:
        models_list (list): List of model names
        data_dict (dict): Dictionary containing model predictions
        truth_column_name (str): Name of column containing true values

    Methods:
        orthogonalize(target_key, train_df, components_kept): Perform SVD orthogonalization
        train(training_options): Train the model combination
        predict2(target_key): Generate predictions with uncertainty
        evaluate(): Calculate coverage statistics
    """
```

## pybmc.inference_utils

### `gibbs_sampler`
```python
def gibbs_sampler(y, X, iterations, prior_info):
    """
    Performs Gibbs sampling for Bayesian linear regression.

    Args:
        y (np.ndarray): Response vector (centered)
        X (np.ndarray): Design matrix
        iterations (int): Number of sampling iterations
        prior_info (tuple): Prior parameters

    Returns:
        np.ndarray: Posterior samples [beta, sigma]
    """
```

### `gibbs_sampler_simplex`
```python
def gibbs_sampler_simplex(
    y, X, Vt_hat, S_hat, iterations, prior_info, burn=10000, stepsize=0.001
):
    """
    Performs Gibbs sampling with simplex constraints on model weights.

    Args:
        y (np.ndarray): Centered response vector
        X (np.ndarray): Design matrix of principal components
        Vt_hat (np.ndarray): Normalized right singular vectors
        S_hat (np.ndarray): Singular values
        iterations (int): Number of sampling iterations
        prior_info (list): [nu0, sigma20] - prior parameters for variance
        burn (int, optional): Burn-in iterations (default: 10000)
        stepsize (float, optional): Proposal step size (default: 0.001)

    Returns:
        np.ndarray: Posterior samples [beta, sigma]
    """
```

### `USVt_hat_extraction`
```python
def USVt_hat_extraction(U, S, Vt, components_kept):
    """
    Extracts reduced-dimensionality matrices from SVD results.

    Args:
        U (np.ndarray): Left singular vectors
        S (np.ndarray): Singular values
        Vt (np.ndarray): Right singular vectors (transposed)
        components_kept (int): Number of components to retain

    Returns:
        tuple: (U_hat, S_hat, Vt_hat, Vt_hat_normalized)
    """
```

## pybmc.sampling_utils

### `coverage`
```python
def coverage(percentiles, rndm_m, models_output, truth_column):
    """
    Calculates coverage percentages for credible intervals.

    Args:
        percentiles (list): Percentiles to evaluate
        rndm_m (np.ndarray): Posterior samples of predictions
        models_output (pd.DataFrame): DataFrame containing true values
        truth_column (str): Name of column with true values

    Returns:
        list: Coverage percentages for each percentile
    """
```

### `rndm_m_random_calculator`
```python
def rndm_m_random_calculator(filtered_model_predictions, samples, Vt_hat):
    """
    Generates posterior predictive samples and credible intervals.

    Args:
        filtered_model_predictions (np.ndarray): Model predictions
        samples (np.ndarray): Gibbs samples [beta, sigma]
        Vt_hat (np.ndarray): Normalized right singular vectors

    Returns:
        tuple: 
            - rndm_m (np.ndarray): Posterior predictive samples
            - [lower, median, upper] (list): Credible interval arrays
    """
