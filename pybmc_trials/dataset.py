import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


class Dataset:
    """
    A general-purpose dataset class for loading and managing model data
    for Bayesian model combination workflows.
    
    Supports .h5 and .csv files, and provides data splitting functionality.
    """

    def __init__(self, data_source=None):
        """
        Initialize the Dataset object.

        :param data_source: Path to the data file (.h5 or .csv).
        """
        self.data_source = data_source
        self.data = {}  # Dictionary of model to DataFrame

    def load_data(self, models, keys=None, domain_keys=None):
        """
        Load data for specified models from a file and synchronize their domains.

        :param models: List of model names (for HDF5 keys or filtering CSV).
        :param keys: List of columns to extract (optional).
        :param domain_keys: List of columns used to define the common domain (ex: ['N', 'Z']).
        :return: Dictionary with model names as keys and synchronized DataFrames as values.
        """
        if self.data_source is None:
            raise ValueError("Data source must be specified.")

        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"Data source '{self.data_source}' not found.")

        if domain_keys is None:
            domain_keys = []

        data_dict = {}

        if self.data_source.endswith('.h5'):
            for model in models:
                df = pd.read_hdf(self.data_source, key=model)
                data_dict[model] = df[keys] if keys else df

        elif self.data_source.endswith('.csv'):
            df = pd.read_csv(self.data_source)
            for model in models:
                model_df = df[df['model'] == model]
                data_dict[model] = model_df[keys] if keys else model_df

        else:
            raise ValueError("Unsupported file format. Only .h5 and .csv are supported.")

        if domain_keys:
            reference_model = models[0]
            reference_df = data_dict[reference_model]
            reference_domain = reference_df[domain_keys].drop_duplicates()

            for model in models[1:]:
                df = data_dict[model]
                merged = df.merge(reference_domain, on=domain_keys, how='inner')
                data_dict[model] = merged

            data_dict[reference_model] = reference_df.merge(reference_domain, on=domain_keys, how='inner')

        self.data = data_dict
        return data_dict

    def split_data(self, data, train_size=0.6, val_size=0.2, test_size=0.2):
        """
        Split a dataset into train, validation, and test sets.

        :param data: DataFrame to split.
        :param train_size: Proportion of data for training.
        :param val_size: Proportion of data for validation.
        :param test_size: Proportion of data for testing.
        :return: Tuple (train, val, test) as DataFrames.
        """
        if not np.isclose(train_size + val_size + test_size, 1.0):
            raise ValueError("train_size + val_size + test_size must equal 1.0")

        train_data, temp_data = train_test_split(data, train_size=train_size, random_state=1)
        val_size_rel = val_size / (val_size + test_size)
        val_data, test_data = train_test_split(temp_data, test_size=1 - val_size_rel, random_state=1)

        return train_data, val_data, test_data



    def get_subset(self, data=None, filters=None, apply_to_all_models=False):
            """
            Return a subset of data based on flexible filtering criteria.

            :param data: A single DataFrame to filter (optional if using apply_to_all_models).
            :param filters: Dictionary of filtering rules.
                            - key = column name, value = single value, tuple (min, max), list of values, or callable
                            - special key 'multi' allows row-wise lambda functions
            :param apply_to_all_models: If True, apply filters to all models in self.data
            :return: Filtered DataFrame or dict of DataFrames per model
            """
            if filters is None:
                raise ValueError("Filters must be provided.")

            def apply_filters(df): # Function to apply filters to a DataFrame
                result = df.copy()
                for column, condition in filters.items(): # Loops through the filters dictionary 
                    if column == 'multi' and callable(condition): # "multi" means the user wants to use custom logic across multiple columns, callable(condition) makes sure it's a function.
                        result = result[result.apply(condition, axis=1)]
                    elif callable(condition): # If the value is a function and not "multi", apply it to the column only.
                        result = result[condition(result[column])]
                    elif isinstance(condition, tuple) and len(condition) == 2: # If the condition is a tuple like (10, 20), check if each value is in that range.
                        result = result[(result[column] >= condition[0]) & (result[column] <= condition[1])]
                    elif isinstance(condition, list): # If the condition is a list, keep rows where the column matches any value in that list.
                        result = result[result[column].isin(condition)]
                    else:
                        result = result[result[column] == condition] # Default case: filter to rows where the column exactly equals that value.
                return result # Returns dataframe with filters applied, only for one model 

            if apply_to_all_models: # Returns a dictionary, where each key is a model name, each value is a filtered DataFrame corresponding to that model's data
                return {model: apply_filters(df) for model, df in self.data.items()}
            else:
                if data is None:
                    raise ValueError("Data must be provided when apply_to_all_models is False.")
                return apply_filters(data)