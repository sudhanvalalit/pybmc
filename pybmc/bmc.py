import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from .inference_utils import gibbs_sampler, USVt_hat_extraction
from .sampling_utils import coverage, rndm_m_random_calculator


class BayesianModelCombination:
    """
    The main idea of this class is to perform BMM on the set of models that we choose 
    from the dataset class. What should this class contain:
    + Orthogonalization step.
    + Perform Bayesian inference on the training data that we extract from the Dataset class.
    + Predictions for certain isotopes.
    """

    def __init__(self, models_truth, selected_models_dataset, weights=None):
        """ 
        Initialize the BayesianModelCombination class.

        :param models_truth: List of model names including 'truth' for experimental data.
        :param selected_models_dataset: Pandas DataFrame containing model predictions and truth values.
        :param weights: Optional initial weights for the models.
        """

        if not isinstance(models_truth, list) or not all(isinstance(model, str) for model in models_truth):
            raise ValueError("The 'models' should be a list of model names (strings) for Bayesian Combination.")    
        if not isinstance(selected_models_dataset, pd.DataFrame):
            raise ValueError("The 'selected_models_dataset' should be a pandas dataframe") 
        if not set(models_truth).issubset(selected_models_dataset.columns):
            raise KeyError("One or more selected models are missing in the dataset.")
        if 'truth' not in models_truth:
            raise KeyError("We need a 'truth' data column for the training algorithm")

        self.selected_models_dataset = selected_models_dataset # Dataset used for Bayesian Model Mixing
        self.models_truth = models_truth # Models and truth values of the BMC dataset
        self.models = models_truth.remove('truth') # This is just the set of models without experimental data
        self.weights = weights if weights is not None else None # Weights of the models 


    def train(self, training_options=None):
        """
        Train the model combination using training data and optional training parameters.

        :param training_data: Placeholder (not used).
        :param training_options: Dictionary of training options. Keys:
            - 'iterations': (int) Number of Gibbs iterations (default 50000)
            - 'b_mean_prior': (np.ndarray) Prior mean vector (default zeros)
            - 'b_mean_cov': (np.ndarray) Prior covariance matrix (default diag(S_hat²))
            - 'nu0_chosen': (float) Degrees of freedom for variance prior (default 1.0)
            - 'sigma20_chosen': (float) Prior variance (default 0.02)
        """
        training_options = {}

        iterations = training_options.get('iterations', 50000)
        num_components = self.U_hat.shape[1]
        S_hat = self.S_hat

        b_mean_prior = training_options.get('b_mean_prior', np.zeros(num_components))
        b_mean_cov = training_options.get('b_mean_cov', np.diag(S_hat**2))
        nu0_chosen = training_options.get('nu0_chosen', 1.0)
        sigma20_chosen = training_options.get('sigma20_chosen', 0.02)

        self.samples = self.gibbs_sampler(self.centered_experiment_train, self.U_hat, iterations, [b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen])

    
    def predict(self, X):
        """
        Predict with full Bayesian posterior sampling, including model weight and data noise.

        :param X: A NumPy array or DataFrame of shape (n_points, n_models), containing
                the model predictions at each evaluation point.
                This should already be filtered and aligned with the trained model order.
        :return: 
            - rndm_m: array of shape (n_samples, n_points), containing full predictive draws
            - (lower, median, upper): 95% credible interval at each point (2.5, 50, 97.5 percentiles)
        """

        if self.samples is None or self.Vt_hat is None:
            raise ValueError("Must call `orthogonalize()` and `train()` before predicting.")

        if isinstance(X, pd.DataFrame):
            X = X.values  

        rndm_m, (lower, median, upper) = self.rndm_m_random_calculator(X, self.samples)

        return rndm_m, (lower, median, upper)
 



    def evaluate(self, method="coverage", domain_filter=None):
        """
        Evaluate the model combination using coverage and/or random sampling.

        :param method: "coverage", "random", or list of both.
        :param domain_filter: dict with optional 'Z' and 'N' ranges, e.g., {"Z": (20, 30), "N": (20, 40)}
        :return: dictionary with keys: "random", "coverage"
        """
        if isinstance(method, str):
            method = [method]

        # Filter data if domain_filter is provided
        df = self.selected_models_dataset.copy()
        if domain_filter:
            # from pybmc.data apply_filters
            def apply_filters(df, filters):
                result = df.copy()
                for column, condition in filters.items():
                    if column == 'multi' and callable(condition):
                        result = result[result.apply(condition, axis=1)]
                    elif callable(condition):
                        result = result[condition(result[column])]
                    elif isinstance(condition, tuple) and len(condition) == 2:
                        result = result[(result[column] >= condition[0]) & (result[column] <= condition[1])]
                    elif isinstance(condition, list):
                        result = result[result[column].isin(condition)]
                    else:
                        result = result[result[column] == condition]
                return result

            df = apply_filters(df, domain_filter)

        filtered_predictions = df[self.models].values

        results = {}

        if "random" in method:
            rndm_m, [lower, median, upper] = self.rndm_m_random_calculator(filtered_predictions, self.samples)
            results["random"] = [lower, median, upper]
        if "coverage" in method:
            results["coverage"] = self._coverage(np.arange(0, 101, 5), rndm_m, df)

        return results

    def orthogonalize(self, data, components_kept):
        """
        Perform orthogonalization on the input data using Singular Value Decomposition (SVD).
        Stores centered truth values, principal components, and related matrices for training.

        :param data: DataFrame containing the training data, with model and 'truth' columns.
        :param components_kept: Number of principal components to retain (default: 3).
        """
        
        # Extract model predictions and experimental truth
        model_matrix = data[self.models].values  # shape: (n_points, n_models)
        truth_vector = data['truth'].values      # shape: (n_points,)

        # Compute mean model prediction for each point
        mean_prediction = np.mean(model_matrix, axis=1)

        # Center the experimental values and model predictions
        centered_experiment = truth_vector - mean_prediction
        centered_models = model_matrix - mean_prediction[:, None]

        # Perform SVD on centered model predictions
        U, S, Vt = np.linalg.svd(centered_models)

        # Extract reduced SVD components
        U_hat, S_hat, Vt_hat, Vt_hat_normalized = self.USVt_hat_extraction(U, S, Vt, components_kept)

        # Save attributes needed for training
        self.centered_experiment_train = centered_experiment
        self.U_hat = U_hat
        self.S_hat = S_hat
        self.Vt_hat = Vt_hat
        self.Vt_hat_normalized = Vt_hat_normalized
        self._predictions_mean_train = mean_prediction


