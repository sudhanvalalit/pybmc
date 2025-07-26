import numpy as np
import pandas as pd
from .inference_utils import (
    gibbs_sampler,
    gibbs_sampler_simplex,
    USVt_hat_extraction,
)
from .sampling_utils import coverage, rndm_m_random_calculator


class BayesianModelCombination:
    """
    Implements Bayesian Model Combination (BMC) for aggregating predictions from multiple models.

    This class performs orthogonalization of model predictions, trains the model combination
    using Gibbs sampling, and provides methods for prediction and evaluation.

    Args:
        models_list (list[str]): List of model names to combine.
        data_dict (dict[str, pandas.DataFrame]): Dictionary from `load_data()` where keys are property names and values are DataFrames.
        truth_column_name (str): Name of the column containing ground truth values.
        weights (list[float], optional): Initial weights for models. Defaults to equal weights.

    Attributes:
        models_list (list[str]): List of model names.
        data_dict (dict[str, pandas.DataFrame]): Loaded data dictionary.
        truth_column_name (str): Ground truth column name.
        weights (list[float]): Current model weights.
        samples (numpy.ndarray): Posterior samples from Gibbs sampling.
        current_property (str): Current property being processed.
        centered_experiment_train (numpy.ndarray): Centered experimental values.
        U_hat (numpy.ndarray): Reduced left singular vectors from SVD.
        Vt_hat (numpy.ndarray): Normalized right singular vectors.
        S_hat (numpy.ndarray): Retained singular values.
        Vt_hat_normalized (numpy.ndarray): Original right singular vectors.
        _predictions_mean_train (numpy.ndarray): Mean predictions across models.

    Example:
        >>> bmc = BayesianModelCombination(
                models_list=["model1", "model2"],
                data_dict=data,
                truth_column_name="truth"
            )
    """

    def __init__(self, models_list, data_dict, truth_column_name, weights=None):
        """
        Initializes the BMC instance.

        Args:
            models_list (list[str]): List of model names to combine.
            data_dict (dict[str, pandas.DataFrame]): Dictionary of DataFrames from Dataset.load_data().
            truth_column_name (str): Name of column containing ground truth values.
            weights (list[float], optional): Initial model weights. Defaults to None (equal weights).

        Raises:
            ValueError: If `models_list` is not a list of strings or `data_dict` is invalid.
        """

        if not isinstance(models_list, list) or not all(
            isinstance(model, str) for model in models_list
        ):
            raise ValueError(
                "The 'models' should be a list of model names (strings) for Bayesian Combination."
            )
        if not isinstance(data_dict, dict) or not all(
            isinstance(df, pd.DataFrame) for df in data_dict.values()
        ):
            raise ValueError(
                "The 'data_dict' should be a dictionary of pandas DataFrames, one per property."
            )

        self.data_dict = data_dict
        self.models_list = models_list
        self.models = [m for m in models_list if m != "truth"]
        self.weights = weights if weights is not None else None
        self.truth_column_name = truth_column_name

    def orthogonalize(self, property, train_df, components_kept):
        """
        Performs orthogonalization of model predictions using SVD.

        This method centers model predictions, performs SVD decomposition, and retains
        the specified number of components for subsequent training.

        Args:
            property (str): Nuclear property to orthogonalize (e.g., 'BE').
            train_df (pandas.DataFrame): Training data from Dataset.split_data().
            components_kept (int): Number of SVD components to retain.

        Note:
            This method must be called before training. Results are stored in instance attributes.
        """
        # Store selected property
        self.current_property = property

        # Extract the relevant DataFrame for that property
        df = self.data_dict[property].copy()
        self.selected_models_dataset = df  # Store for train() and predict()

        # Extract model outputs (only the model columns)
        models_output_train = train_df[self.models]
        model_predictions_train = models_output_train.values

        # Mean prediction across models (per nucleus)
        predictions_mean_train = np.mean(model_predictions_train, axis=1)

        # Experimental truth values for the property
        centered_experiment_train = (
            train_df[self.truth_column_name].values - predictions_mean_train
        )

        # Center model predictions
        model_predictions_train_centered = (
            model_predictions_train - predictions_mean_train[:, None]
        )

        # Perform SVD
        U, S, Vt = np.linalg.svd(model_predictions_train_centered)

        # Dimensionality reduction
        U_hat, S_hat, Vt_hat, Vt_hat_normalized = USVt_hat_extraction(U, S, Vt, components_kept)  # type: ignore

        # Save for training
        self.centered_experiment_train = centered_experiment_train
        self.U_hat = U_hat
        self.Vt_hat = Vt_hat
        self.S_hat = S_hat
        self.Vt_hat_normalized = Vt_hat_normalized
        self._predictions_mean_train = predictions_mean_train

    def train(self, training_options=None):
        """
        Trains the model combination using Gibbs sampling.

        Args:
            training_options (dict, optional): Training configuration. Options:
                - iterations (int): Number of Gibbs iterations (default: 50000).
                - sampler (str): 'gibbs_sampling' or 'simplex' (default: 'gibbs_sampling').
                - burn (int): Burn-in iterations for simplex sampler (default: 10000).
                - stepsize (float): Proposal step size for simplex sampler (default: 0.001).
                - b_mean_prior (numpy.ndarray): Prior mean vector (default: zeros).
                - b_mean_cov (numpy.ndarray): Prior covariance matrix (default: diag(S_hat²)).
                - nu0_chosen (float): Degrees of freedom for variance prior (default: 1.0).
                - sigma20_chosen (float): Prior variance (default: 0.02).

        Note:
            Requires prior call to `orthogonalize()`. Stores posterior samples in `self.samples`.
        """

        if training_options is None:
            training_options = {}

        # functions defined so that whenever a key not specified, we print out the default value for users
        def get_option(key, default):
            if key not in training_options:
                print(f"[INFO] Using default value for '{key}': {default}")
            return training_options.get(key, default)

        iterations = get_option("iterations", 50000)
        sampler = get_option("sampler", "gibbs_sampling")
        burn = get_option("burn", 10000)
        stepsize = get_option("stepsize", 0.001)

        S_hat = self.S_hat
        num_components = self.U_hat.shape[1]

        b_mean_prior = get_option("b_mean_prior", np.zeros(num_components))
        b_mean_cov = get_option("b_mean_cov", np.diag(S_hat**2))
        nu0_chosen = get_option("nu0_chosen", 1.0)
        sigma20_chosen = get_option("sigma20_chosen", 0.02)

        if sampler == "simplex":
            self.samples = gibbs_sampler_simplex(
                self.centered_experiment_train,
                self.U_hat,
                self.Vt_hat,
                self.S_hat,
                iterations,
                [
                    nu0_chosen,
                    sigma20_chosen,
                ],  # Note: no b_mean_prior/b_mean_cov needed
                burn=burn,
                stepsize=stepsize,
            )
        else:
            self.samples = gibbs_sampler(
                self.centered_experiment_train,
                self.U_hat,
                iterations,
                [b_mean_prior, b_mean_cov, nu0_chosen, sigma20_chosen],
            )

    def predict(self, X):
        """
        Predicts values using the trained model combination with uncertainty quantification.

        Args:
            X (pandas.DataFrame): Input data containing model predictions and domain information.

        Returns:
            tuple[numpy.ndarray, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: Contains:
                - rndm_m (numpy.ndarray): Full posterior draws (n_samples, n_points).
                - lower_df (pandas.DataFrame): Lower bounds (2.5th percentile) with domain keys.
                - median_df (pandas.DataFrame): Median predictions with domain keys.
                - upper_df (pandas.DataFrame): Upper bounds (97.5th percentile) with domain keys.

        Raises:
            ValueError: If `orthogonalize()` and `train()` haven't been called.
        """
        if self.samples is None or self.Vt_hat is None:
            raise ValueError(
                "Must call `orthogonalize()` and `train()` before predicting."
            )

        if not isinstance(X, pd.DataFrame):
            raise ValueError(
                "X must be a pandas DataFrame containing model predictions and domain info."
            )

        # Infer model columns vs. domain columns
        model_cols = self.models
        domain_keys = [col for col in X.columns if col not in model_cols]

        model_preds = X[model_cols].values
        rndm_m, (lower, median, upper) = rndm_m_random_calculator(
            model_preds, self.samples, self.Vt_hat
        )

        domain_df = X[domain_keys].reset_index(drop=True)

        lower_df = domain_df.copy()
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df

    def predict2(self, property):
        """
        Predicts values for a specific property using the trained model combination.

        This version uses the property name instead of a DataFrame input.

        Args:
            property (str): Property name to predict (e.g., 'ChRad').

        Returns:
            tuple[numpy.ndarray, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: Contains:
                - rndm_m (numpy.ndarray): Full posterior draws (n_samples, n_points).
                - lower_df (pandas.DataFrame): Lower bounds (2.5th percentile) with domain keys.
                - median_df (pandas.DataFrame): Median predictions with domain keys.
                - upper_df (pandas.DataFrame): Upper bounds (97.5th percentile) with domain keys.

        Raises:
            ValueError: If `orthogonalize()` and `train()` haven't been called.
            KeyError: If property not found in `data_dict`.
        """
        if self.samples is None or self.Vt_hat is None:
            raise ValueError(
                "Must call `orthogonalize()` and `train()` before predicting."
            )

        if property not in self.data_dict:
            raise KeyError(f"Property '{property}' not found in data_dict.")

        df = self.data_dict[property].copy()

        # Infer domain and model columns
        full_model_cols = self.models
        domain_keys = [
            col
            for col in df.columns
            if col not in full_model_cols and col != self.truth_column_name
        ]

        # Determine which models are present
        available_models = [m for m in df.columns if m in self.models]

        # Sets
        trained_models_set = set(self.models)
        available_models_set = set(available_models)

        missing_models = trained_models_set - available_models_set
        extra_models = available_models_set - trained_models_set
        print(f"Available models: {available_models_set}")
        print(f"Trained models: {trained_models_set}")

        if len(extra_models) > 0:
            raise ValueError(
                f"ERROR: Property '{property}' contains extra models not present during training: {list(extra_models)}. "
                "You must retrain if using a larger model space."
            )

        if len(missing_models) > 0:
            print(
                f"WARNING: Predicting on property '{property}' with missing models: {list(missing_models)}"
            )
            print(
                "         The trained model weights include these models — prediction will proceed, but results may not be statistically accurate."
            )

        if len(available_models) == 0:
            raise ValueError(
                "No available trained models are present in prediction DataFrame."
            )

        # Filter predictions and model weights
        model_preds = df[available_models].values
        domain_df = df[domain_keys].reset_index(drop=True)

        # Find indices of available models in training order
        model_indices = [self.models.index(m) for m in available_models]

        # Reduce Vt_hat and samples to only use available models
        Vt_hat_reduced = self.Vt_hat[:, model_indices]

        rndm_m, (lower, median, upper) = rndm_m_random_calculator(
            model_preds, self.samples, Vt_hat_reduced
        )

        # Build output DataFrames
        lower_df = domain_df.copy()
        lower_df["Predicted_Lower"] = lower

        median_df = domain_df.copy()
        median_df["Predicted_Median"] = median

        upper_df = domain_df.copy()
        upper_df["Predicted_Upper"] = upper

        return rndm_m, lower_df, median_df, upper_df

    def evaluate(self, domain_filter=None):
        """
        Evaluates model performance using coverage calculation.

        Args:
            domain_filter (dict, optional): Filtering rules for domain columns.
                Example: {"Z": (20, 30), "N": (20, 40)}.

        Returns:
            list[float]: Coverage percentages for each percentile in [0, 5, 10, ..., 100].
        """
        df = self.data_dict[self.current_property]

        if domain_filter:
            # Inline optimized filtering
            for col, cond in domain_filter.items():
                if col == "multi" and callable(cond):
                    df = df[df.apply(cond, axis=1)]
                elif callable(cond):
                    df = df[cond(df[col])]
                elif isinstance(cond, tuple) and len(cond) == 2:
                    df = df[df[col].between(*cond)]
                elif isinstance(cond, list):
                    df = df[df[col].isin(cond)]
                else:
                    df = df[df[col] == cond]

        preds = df[self.models].to_numpy()
        rndm_m, (lower, median, upper) = rndm_m_random_calculator(
            preds, self.samples, self.Vt_hat
        )

        return coverage(
            np.arange(0, 101, 5),
            rndm_m,
            df,
            truth_column=self.truth_column_name,
        )
