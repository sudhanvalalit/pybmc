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
        if not os.path.exists(self.data_source): # This ensures that the file exist in our system
            raise FileNotFoundError(f"Data source {self.data_source} not found.") 
        self.data_source = data_source if data_source else "./data/selected_data.h5" # This is the default file \
            # path that we are using for the moment
        if models is None: # Ensures models is a list of strings 
            self.models = Dataset.models_selected_default
        elif not isinstance(models, list) or not all(isinstance(m, str) for m in models): # This ensures that the \
            # models is a list of string
            raise ValueError("models must be a list of strings.")
        else:
            self.models = models
        if keys is None: # Ensures keys is a list of strings
            self.keys = Dataset.keys_default
        elif not isinstance(keys, list) or not all(isinstance(k, str) for k in keys): # This ensures that the keys \
            # must be a list of strings
            raise ValueError("keys must be a list of strings.")
        else:
            self.keys = keys
        # Next line extract the raw data set that we extract from the source. Right now the load_data method is effective for \
        # h5 file only. Needs further check
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
        elif data_source.endswith('.csv'): # In our case right now, the csv file only contains the experimental data\
            # for charge radii so it does not contains the data for the model yet. Will work on it later.
            data_dict = pd.read_csv(data_source)
        else:
            raise ValueError("Unsupported file format. Only .h5 and .csv are supported.")
        
        return data_dict

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

        # Only work with models that exist in self.models (models that we initialize our data with)
        if not all(model in self.models for model in models_selected):
            raise KeyError("One or more selected models are missing in the dataset.")

        # Convert each model’s (N, Z) data into a DataFrame. This create a list in which each element is a dataframe\
        # that contains isotopes for each model
        model_dfs = [
            pd.DataFrame(models_data_sets[model], columns=['N', 'Z'])
            for model in models_selected
        ]

        # Merge all models to keep only isotopes that appear in every model
        from functools import reduce # reduce is  a function from functools  that recursively applies a function to \
        # a sequence, reducing it to a single output.
        common_isotopes_df = reduce(lambda left, right: left.merge(right, on=['N', 'Z'], how='inner'), model_dfs) # lambda\
        # left, right merge left and right dataframes into one using how = 'inner', which only extract coommon isotopes.

        
        return common_isotopes_df.to_numpy(), common_isotopes_df # This equal to df.values - extract numpy array out from our dataframe

    
    def selected_models_data_sets_extraction(self, initial_NZ_df=None, models_data_sets=None, models_selected=None, property=None):
        """
        Extracts and organizes model predictions for a given nuclear property.

        :param df: DataFrame containing (N, Z) values to filter.
        :param property: The nuclear property to extract (e.g., 'BE', 'ChRad'). Note that it should only extract 1 property at a time
        :return: A DataFrame where each column is a model's prediction for the given property.
        """
        models_data_sets = models_data_sets if models_data_sets is not None else self.raw_data_sets
        models_selected = models_selected if models_selected is not None else self.models
        
        if property is None:
                raise ValueError("Property must be specified.")
        
        # This will ensure that if we have a subset of data that we want to extract from all the models then our dataset will work\
        # with this dataset
        if initial_NZ_df is not None:
             selected_models_data_sets_df  = initial_NZ_df.copy()
        # Otherwise, we will extract dataset that have common isotopes of all models
        else:
            # Extract common isotopes from our dataset and use that to merge all all models together
            initial_NZ, initial_NZ_df = self.extract_common_isotopes(models_data_sets, models_selected)


            selected_models_data_sets_df = initial_NZ_df.copy()  # Start with filtered (N, Z) data

        for i, model in enumerate(models_selected):
            if property not in models_data_sets[model]:
                raise KeyError(f"Property '{property}' not found in model '{model}' dataset.")

            # Rename the property column to the model name
            model_df = pd.DataFrame(models_data_sets[model], columns=['N', 'Z', property]).rename(
                columns={property: model}  # Rename column to just model name
            )

            # Merge on N and Z to keep common isotopes
            selected_models_data_sets_df = selected_models_data_sets_df.merge(model_df, on=['N', 'Z'], how='inner')

        return selected_models_data_sets_df

    
    def split_data(self, subset_data, splitting_algorithm, models_data_sets=None, models_selected=None, **kwargs):
        """
        Split data into training, validation, and testing sets.

        :param train_size: Proportion of the data to include in the training set.
        :param val_size: Proportion of the data to include in the validation set.
        :param test_size: Proportion of the data to include in the testing set.
        :return: A tuple containing the training, validation, and testing sets.
        """
        if not isinstance(subset_data, np.ndarray):
            print("subset_data must be a numpy array")

        if splitting_algorithm not in ('inside_to_outside', 'random'):
            raise ValueError("splitting_algorithm must be either 'inside_to_outside' or 'random'")
        
        # Use the the subset_data after we have sorted using the get_subset method
        
        
        if splitting_algorithm == 'random':
            # Ensure required parameters exist
            required_params = ['train_size', 'val_size', 'test_size']
            if not all(param in kwargs for param in required_params):
                raise ValueError(f"Missing parameters for 'random' split: {required_params}")
            
            train_size = kwargs['train_size']
            val_size = kwargs['val_size']
            test_size = kwargs['test_size']

            # Validate that the sizes sum to 1
            if train_size + val_size + test_size != 1.0:
                raise ValueError("train_size, val_size, and test_size must sum to 1.0")

            train_idx, temp_idx = train_test_split(subset_data.index, train_size=train_size, random_state=1)
            val_size_relative = val_size / (val_size + test_size)  # Relative validation size
            val_idx, test_idx = train_test_split(temp_idx, test_size=val_size_relative, random_state=1)

        elif splitting_algorithm == 'inside_to_outside':
            # Ensure required parameters exist
            required_params = ['stable_isotopes', 'distance_1', 'distance_2']
            if not all(param in kwargs for param in required_params):
                raise ValueError(f"Missing parameters for 'random' split: {required_params}")
            
            stable_isotopes = kwargs['stable_isotopes']
            distance_1 = kwargs['distance_1']
            distance_2 = kwargs['distance_2'] 

            # We will only work with indexes of train, valid, and test data beccause we will only extract data from these indexes
            train_idx, val_idx, test_idx = self.separate_points_distance_allSets(subset_data, stable_isotopes, distance_1, distance_2)

        return np.array(train_idx), np.array(val_idx), np.array(test_idx)
        

    def get_subset(self, combined_data_df, domain_X=None, N_range=None, Z_range=None): 
        """
        Return a subset of data for a given domain X  and/or a range of N and Z. The input should be a dataframe that contain
        common isotopes that all models have

        :param domain_X: String specifying the domain type ('even-even', 'even-odd', 'odd-even', 'odd-odd').
        :param N_range: Tuple specifying the range of neutron numbers (N_min, N_max).
        :param Z_range: Tuple specifying the range of proton numbers (Z_min, Z_max).
        :return: A DataFrame containing the subset of data.
        """

        if domain_X:
            if domain_X == 'even-even':
                subset_data_df = combined_data_df[(combined_data_df['N'] % 2 == 0) & (combined_data_df['Z'] % 2 == 0)]
            elif domain_X == 'even-odd':
                subset_data_df = combined_data_df[(combined_data_df['N'] % 2 == 0) & (combined_data_df['Z'] % 2 != 0)]
            elif domain_X == 'odd-even':
                subset_data_df = combined_data_df[(combined_data_df['N'] % 2 != 0) & (combined_data_df['Z'] % 2 == 0)]
            elif domain_X == 'odd-odd':
                subset_data_df = combined_data_df[(combined_data_df['N'] % 2 != 0) & (combined_data_df['Z'] % 2 != 0)]
            else:
                raise ValueError("Invalid domain_X value. Choose from 'even-even', 'even-odd', 'odd-even', 'odd-odd'.")
        else:
            subset_data_df = combined_data_df

        if N_range:
            N_min, N_max = N_range
            subset_data_df = subset_data_df[(subset_data_df['N'] >= N_min) & (subset_data_df['N'] <= N_max)]

        if Z_range:
            Z_min, Z_max = Z_range
            subset_data_df = subset_data_df[(subset_data_df['Z'] >= Z_min) & (subset_data_df['Z'] <= Z_max)]

        # This reset the index of the dataframe from the old dataframe
        subset_data_df.reset_index(drop = True, inplace = True)

        return subset_data_df.to_numpy(), subset_data_df
    


    # Below are the algorithms that we used to separate our data inside-out from the stable isotopes
    def separate_points_random(self, list1,random_chance):
            """
            Separates points in list1 into two groups randomly

            """
            train = []
            test = []

            train_list_coordinates=[]
            test_list_coordinates=[]


            for i in range(len(list1)):
                point1=list1[i]
                val=np.random.rand()
                if val<=random_chance:
                    train.append(point1)
                    train_list_coordinates.append(i)
                else:
                    test.append(point1)
                    test_list_coordinates.append(i)

            return np.array(train), np.array(test), np.array(train_list_coordinates), np.array(test_list_coordinates)

    def separate_points_distance(self, list1, list2, distance):
        """
        Separates points in list1 into two groups based on their proximity to any point in list2.

        :param list1: List of (x, y) tuples.
        :param list2: List of (x, y) tuples.
        :param distance: The threshold distance to determine proximity.
        :return: Two lists - close_points and distant_points.
        """
        train = []
        test = []

        train_list_coordinates=[]
        test_list_coordinates=[]

        for i in range(len(list1)):
            point1=list1[i]
            close = False
            for point2 in list2:
                if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance:
                    close = True
                    break
            if close:
                train.append(point1)
                train_list_coordinates.append(i)
            else:
                test.append(point1)
                test_list_coordinates.append(i)

        return np.array(train), np.array(test), np.array(train_list_coordinates), np.array(test_list_coordinates)

    def separate_points_distance_allSets(self, list1, list2, distance1, distance2):
        """
        Separates points in list1 into three groups based on their proximity to any point in list2.

        :param list1: List of (x, y) tuples.
        :param list2: List of (x, y) tuples.
        :param distance: The threshold distance to determine proximity.
        :return: Two lists - close_points and distant_points.
        """
        train = []
        validation=[]
        test = []

        train_list_coordinates=[]
        validation_list_coordinates=[]
        test_list_coordinates=[]

        for i in range(len(list1)):
            point1=list1[i]
            close = False
            for point2 in list2:
                if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance1:
                    close = True
                    break
            if close:
                train.append(point1)
                train_list_coordinates.append(i)
            else:
                close2=False
                for point2 in list2:
                    if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance2:
                        close2 = True
                        break
                if close2==True:
                    validation.append(point1)
                    validation_list_coordinates.append(i)
                else:
                    test.append(point1)
                    test_list_coordinates.append(i)                

        return train_list_coordinates, validation_list_coordinates, test_list_coordinates
    
uploaded_data = Dataset(data_source= r'C:\Users\congn\OneDrive\Desktop\An Le Materials\ModelOrthogonalization\data\selected_data.h5',models = None,keys = None)

# We have checked that it does extract common isotopes for you
common_isotopes, common_isotopes_df = uploaded_data.extract_common_isotopes()

# This does give you the subset of data that we will work with
selected_NZ, selected_NZ_df = uploaded_data.get_subset( common_isotopes_df, domain_X='even-even', N_range=[8, 300], Z_range= [8, 300])
print('The number of common isotopes are' + str(selected_NZ_df.shape) + 'isotopes')

# We want to check if the get_subset method could handle more than just the dataset of original models
stable_isotopes_full=np.loadtxt(r"C:\Users\congn\OneDrive\Desktop\An Le Materials\ModelOrthogonalization\Stable-Isotopes.txt")
stable_isotopes_full_df = pd.DataFrame({ 'N' : stable_isotopes_full[:, 0], 'Z' : stable_isotopes_full[:, 1]  })
selected_stable_isotopes, selected_stable_isotopes_df = uploaded_data.get_subset(stable_isotopes_full_df,\
                                                                                  domain_X='even-even', N_range=[8, 300], Z_range= [8, 300])
print('The number of selected stable isotopes are' + str(selected_stable_isotopes.shape) + 'isotopes')

# This check that the code is able to extract dataframe of models 
selected_models_data_sets_mass_df = uploaded_data.selected_models_data_sets_extraction(initial_NZ_df= selected_NZ_df, property= 'ChRad')
print(selected_models_data_sets_mass_df)