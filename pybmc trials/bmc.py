import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

class BayesianModelCombination:
    """
    The main idea of this class is to perform BMM on the set of models that we choose 
    from the dataset class. What should this class contain:
    + Orthogonalization step.
    + Perform Bayesian inference on the training data that we extract from the Dataset class.
    + Predictions for certain isotopes.
    """
    def __init__(self, models_truth, selected_models_dataset, options = None, weights = None):
        if not isinstance(models_truth, list) or not all(isinstance(model, str) for model in models_truth):
            raise ValueError("The 'models' should be a list of model names (strings) for Bayesian Combination.")
        
        if not isinstance(selected_models_dataset, pd.DataFrame):
            raise ValueError("The 'selected_models_dataset' should be a pandas dataframe")
        
        if not set(models_truth).issubset(selected_models_dataset.columns):
            raise KeyError("One or more selected models are missing in the dataset.")
        if 'truth' not in models_truth:
            raise KeyError("We need a 'truth' data column for the training algorithm")

        
        self.selected_models_dataset = selected_models_dataset[models_truth]


        self.options = options if options is not None else {'use_orthogonalization': True}
        self.weights = weights if weights is not None else None

    def train(self, train_index, models, components_kept):
        if self.options.get('use_orthogonalization', False):
            # This gives you the dataframe containing only the training index
            models_truth_train  = self.selected_models_dataset.iloc[train_index]
            # This gives you the dataframe containing only the models
            models_output_train = models_truth_train[models]

            U, S, Vt, predicions_mean_train = self.orthogonalize(models_output_train)
        # This extracts dimensionality-reduction matrices from SVD
        U_hat, S_hat, Vt_hat, Vt_hat_normalized = self.USVt_hat_extraction(U, S, Vt, components_kept)

        # Centered experimental data
        y = self.selected_models_dataset['truth'].values - predicions_mean_train

        gibbs_sampler(y, X, iterations,prior_info)


    def orthogonalize(self, models_output_train):
        """ 
        We are going to assume for now that the experimental data is not included into the models'
        dataframe, but that is not needed for the orthogonalization step.
        """
        model_predictions_train = models_output_train.values
        # This is the mean predictions of all models
        predictions_mean_train = np.mean(model_predictions_train, axis = 1)

        model_predictions_train_centered = model_predictions_train - predictions_mean_train[:,None]

        U, S, Vt = np.linalg.svd(model_predictions_train_centered)

        return U, S, Vt, predictions_mean_train
    
    def USVt_hat_extraction(self, U,S,Vt, components_kept):
        U_hat = np.array([U.T[i] for i in range(components_kept)]).T
        S_hat = S[:components_kept]
        Vt_hat = np.array([Vt[i]/S[i] for i in range(components_kept)])
        Vt_hat_normalized = np.array([Vt[i] for i in range(components_kept)])
        return U_hat, S_hat, Vt_hat, Vt_hat_normalized


    # def orthogonalize(self, selected_models_dataset = None):
    #     selected_models_dataset = selected_models_dataset if selected_models_dataset is not None else self.selected_models_dataset