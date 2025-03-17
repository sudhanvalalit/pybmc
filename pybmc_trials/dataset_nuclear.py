import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


class Dataset:
    """
    The main idea of this class is to handle data construction and organization of the set of models
    that we choose to analyze. What should this class contain:
    + Methods that help us extract model data from the data source.
    + Methods that select valid domains (e.g., even-even nuclei).
    + Methods that make our data self-consistent among all models.
    """

    models_selected_default = ['ME2', 'MEdelta', 'PC1', 'NL3S', 'SKMS', 'SKP', 'SLY4', 'SV', 'UNEDF0', 'UNEDF1']
    keys_default = ['N', 'Z', 'BE', 'ChRad']

    def __init__(self, data_source=None, models=None, keys = None):
        """
        Initialize the dataset with models and extract relevant data.

        :param data_source: Path to the HDF5 file containing the data.
        :param models: List of model names to extract data for.
        :param keys: Variable arguments specifying which data attributes to extract.
        """
        self.data_source = data_source if data_source else "./data/selected_data.h5"           
        if not os.path.exists(self.data_source): #
            raise FileNotFoundError(f"Data source {self.data_source} not found.")
        self.data_source = data_source if data_source else "./data/selected_data.h5"
        if models is None: # Ensures models is a list of strings 
            self.models = Dataset.models_selected_default
        elif not isinstance(models, list) or not all(isinstance(m, str) for m in models):
            raise ValueError("models must be a list of strings.")
        else:
            self.models = models
        if keys is None: # Ensures keys is a list of strings
            self.keys = Dataset.keys_default
        elif not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise ValueError("keys must be a list of strings.")
        else:
            self.keys = keys
        self.raw_data_sets = self.load_data(self.data_source, self.models, self.keys)

    def load_data(self, data_source=None, models=None, keys=None):
        """
        Loads data from the given HDF5 or CSV file and organizes it into a dictionary.
        :param data_source: Path to the HDF5 or CSV file (if not provided, uses instance attribute).
        :param models: List of model names to extract data for (if not provided, uses instance attribute).
        :param keys: List of data attributes to extract (if not provided, uses instance attribute).
        :return: Dictionary where keys are model names and values are dictionaries of extracted data.
        """
        # Use instance attributes if arguments are not provided
        data_source = data_source if data_source is not None else self.data_source
        models = models if models is not None else self.models
        keys = keys if keys is not None else self.keys

        # Ensure the file exists
        if not os.path.exists(data_source):
            raise FileNotFoundError(f"Data source '{data_source}' not found.")

        data_dict = {}
        if data_source.endswith('.h5'):
            for model in models:
                data_values = pd.read_hdf(data_source, key=model)
                data_dict[model] = {key: data_values[key] for key in keys}
        elif data_source.endswith('.csv'):
            data_values = pd.read_csv(data_source)
            for model in models:
                model_data = data_values[data_values['model'] == model]
                data_dict[model] = {key: model_data[key].values for key in keys}
        else:
            raise ValueError("Unsupported file format. Only .h5 and .csv are supported.")
        
        return data_dict

    def domain_synchronization(self, models_data_sets=None, models_selected=None):
        """
        Synchronizes the dataset across selected models, ensuring that only common isotopes remain.

        :return: A numpy array of synchronized isotopes.
        """
        return np.array(self.unified_NZ_extraction(self.extract_common_isotopes(), models_data_sets, models_selected))

    def extract_common_isotopes(self, models_data_sets=None, models_selected=None):
        """
        Extracts and returns a NumPy array of (N, Z) values that are common across all selected models.

        :param models_data_sets: Dictionary of model datasets (defaults to self.raw_data_sets).
        :param models_selected: List of models to extract isotopes from (defaults to self.models).
        :return: NumPy array of isotopes [(N1, Z1), (N2, Z2), ...] that exist in all selected models.
        """
        models_data_sets = models_data_sets if models_data_sets is not None else self.raw_data_sets
        models_selected = models_selected if models_selected is not None else self.models

        # Start with the isotope list from the first model
        if not models_selected:
            raise ValueError("No models selected for isotope extraction.")

        if not all(model in models_data_sets for model in models_selected):
            raise KeyError("One or more selected models are missing in the dataset.")

        # Convert each model’s (N, Z) data into a DataFrame
        model_dfs = [
            pd.DataFrame(models_data_sets[model], columns=['N', 'Z'])
            for model in models_selected
        ]

        # Merge all models to keep only isotopes that appear in every model
        from functools import reduce
        common_isotopes_df = reduce(lambda left, right: left.merge(right, on=['N', 'Z'], how='inner'), model_dfs)

        return common_isotopes_df.to_numpy()

    
    def selected_models_data_sets_extraction(self, df, models_data_sets=None, models_selected=None, property=None):
        """
        Extracts and organizes model predictions for a given nuclear property.

        :param df: DataFrame containing (N, Z) values to filter.
        :param property: The nuclear property to extract (e.g., 'BE', 'ChRad').
        :return: A DataFrame where each column is a model's prediction for the given property.
        """
        models_data_sets = models_data_sets if models_data_sets is not None else self.raw_data_sets
        models_selected = models_selected if models_selected is not None else self.models

        if property is None:
            raise ValueError("Property must be specified.")

        result_df = df.copy()  # Start with filtered (N, Z) data

        for model in models_selected:
            if property not in models_data_sets[model]:
                raise KeyError(f"Property '{property}' not found in model '{model}' dataset.")

            model_df = pd.DataFrame(models_data_sets[model], columns=['N', 'Z', property])
            result_df = result_df.merge(model_df, on=['N', 'Z'], how='inner', suffixes=("", f"_{model}"))

        return result_df

    
    def split_data(self, train_size, val_size, test_size):
        """
        Split data into training, validation, and testing sets.

        :param train_size: Proportion of the data to include in the training set.
        :param val_size: Proportion of the data to include in the validation set.
        :param test_size: Proportion of the data to include in the testing set.
        :return: A tuple containing the training, validation, and testing sets.
        """
        if train_size + val_size + test_size != 1.0:
            raise ValueError("train_size, val_size, and test_size must sum to 1.0")

        # Combine all data into a single DataFrame
        combined_data = pd.concat([pd.DataFrame(self.raw_data_sets[model]) for model in self.models], ignore_index=True)
        if combined_data.empty:
            raise ValueError("No data available for splitting.")

        train_data, temp_data = train_test_split(combined_data, train_size=train_size, random_state=1)
        
        # Calculate validation and test sizes relative to the temp set
        val_size_relative = val_size / (val_size + test_size)
        
        # Split temp data into validation and test sets
        val_data, test_data = train_test_split(temp_data, test_size=val_size_relative, random_state=1)
        
        return train_data, val_data, test_data
        

    def get_subset(self, domain_X=None, N_range=None, Z_range=None): 
        """
        Return a subset of data for a given domain X and/or a range of N and Z.

        :param domain_X: String specifying the domain type ('even-even', 'even-odd', 'odd-even', 'odd-odd').
        :param N_range: Tuple specifying the range of neutron numbers (N_min, N_max).
        :param Z_range: Tuple specifying the range of proton numbers (Z_min, Z_max).
        :return: A DataFrame containing the subset of data.
        """
        combined_data = pd.concat([pd.DataFrame(self.raw_data_sets[model]) for model in self.models], ignore_index=True)

        if domain_X:
            if domain_X == 'even-even':
                subset_data = combined_data[(combined_data['N'] % 2 == 0) & (combined_data['Z'] % 2 == 0)]
            elif domain_X == 'even-odd':
                subset_data = combined_data[(combined_data['N'] % 2 == 0) & (combined_data['Z'] % 2 != 0)]
            elif domain_X == 'odd-even':
                subset_data = combined_data[(combined_data['N'] % 2 != 0) & (combined_data['Z'] % 2 == 0)]
            elif domain_X == 'odd-odd':
                subset_data = combined_data[(combined_data['N'] % 2 != 0) & (combined_data['Z'] % 2 != 0)]
            else:
                raise ValueError("Invalid domain_X value. Choose from 'even-even', 'even-odd', 'odd-even', 'odd-odd'.")
        else:
            subset_data = combined_data

        if N_range:
            N_min, N_max = N_range
            subset_data = subset_data[(subset_data['N'] >= N_min) & (subset_data['N'] <= N_max)]

        if Z_range:
            Z_min, Z_max = Z_range
            subset_data = subset_data[(subset_data['Z'] >= Z_min) & (subset_data['Z'] <= Z_max)]

        return subset_data
    




    

