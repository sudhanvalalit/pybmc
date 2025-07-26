import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # type: ignore
import os


class Dataset:
    """
    Manages datasets for Bayesian model combination workflows.

    Supports loading data from HDF5 and CSV files, splitting data, and filtering.

    Attributes:
        data_source (str): Path to data file.
        data (dict[str, pandas.DataFrame]): Dictionary of loaded data by property.
        domain_keys (list[str]): Domain columns used for data alignment.
    """

    def __init__(self, data_source=None):
        """
        Initializes the Dataset instance.

        Args:
            data_source (str, optional): Path to data file (.h5 or .csv).
        """
        self.data_source = data_source
        self.data = {}  # Dictionary of model to DataFrame
        self.domain_keys = ["X1", "X2"]  # Default domain keys

    def load_data(self, models, keys=None, domain_keys=None, model_column="model"):
        """
        Loads data for multiple properties and models.

        Args:
            models (list[str]): Model names to load.
            keys (list[str]): Property names to extract.
            domain_keys (list[str], optional): Domain columns (default: ['N', 'Z']).
            model_column (str, optional): CSV column identifying models (default: 'model').

        Returns:
            dict[str, pandas.DataFrame]: Dictionary of DataFrames keyed by property name.

        Raises:
            ValueError: If `data_source` not specified or `keys` missing.
            FileNotFoundError: If `data_source` doesn't exist.

        Example:
            >>> dataset = Dataset('data.h5')
            >>> data = dataset.load_data(
                    models=['model1', 'model2'],
                    keys=['BE', 'Rad'],
                    domain_keys=['Z', 'N']
                )
        """
        self.domain_keys = domain_keys

        if self.data_source is None:
            raise ValueError("Data source must be specified.")
        if not os.path.exists(self.data_source):
            raise FileNotFoundError(f"Data source '{self.data_source}' not found.")
        if keys is None:
            raise ValueError("You must specify which properties to extract via 'keys'.")

        result = {}

        for prop in keys:
            dfs = []
            skipped_models = []

            if self.data_source.endswith(".h5"):
                for model in models:
                    df = pd.read_hdf(self.data_source, key=model)
                    # Check required columns
                    missing_cols = [
                        col for col in domain_keys + [prop] if col not in df.columns
                    ]
                    if missing_cols:
                        print(
                            f"[Skipped] Model '{model}' missing columns {missing_cols} for property '{prop}'."
                        )
                        skipped_models.append(model)
                        continue
                    temp = df[domain_keys + [prop]].copy()
                    temp.rename(columns={prop: model}, inplace=True)  # type: ignore
                    dfs.append(temp)
            elif self.data_source.endswith(".csv"):
                df = pd.read_csv(self.data_source)
                for model in models:
                    if model_column not in df.columns:
                        raise ValueError(
                            f"Expected column '{model_column}' not found in CSV."
                        )
                    model_df = df[df[model_column] == model]
                    missing_cols = [
                        col
                        for col in domain_keys + [prop]
                        if col not in model_df.columns
                    ]
                    if missing_cols:
                        print(
                            f"[Skipped] Model '{model}' missing columns {missing_cols} for key '{prop}'."
                        )
                        skipped_models.append(model)
                        continue
                    temp = model_df[domain_keys + [prop]].copy()
                    temp.rename(columns={prop: model}, inplace=True)
                    dfs.append(temp)
            else:
                raise ValueError(
                    "Unsupported file format. Only .h5 and .csv are supported."
                )

            if not dfs:
                print(
                    f"[Warning] No models with property '{prop}'. Resulting DataFrame will be empty."
                )
                result[prop] = pd.DataFrame(
                    columns=domain_keys + [m for m in models if m not in skipped_models]
                )
                continue

            # Intersect domain for this property
            common_df = dfs[0]
            for other_df in dfs[1:]:
                common_df = pd.merge(common_df, other_df, on=domain_keys, how="inner")

            result[prop] = common_df
            self.data = result
        return result

    def view_data(self, property_name=None, model_name=None):
        """
        Provides flexible data viewing options.

        Args:
            property_name (str, optional): Specific property to view.
            model_name (str, optional): Specific model to view.

        Returns:
            Union[dict[str, Union[pandas.DataFrame, str]], pandas.DataFrame, pandas.Series]:
                - If no args: dict of available properties/models.
                - If only `model_name`: dict of `{property: DataFrame}`.
                - If only `property_name`: DataFrame for property.
                - If both: Series of model values for property.

        Raises:
            RuntimeError: If no data loaded.
            KeyError: If property or model not found.
        """

        if not self.data:
            raise RuntimeError("No data loaded. Run `load_data(...)` first.")

        if property_name is None and model_name is None:
            props = list(self.data.keys())
            models = sorted(
                set(
                    col
                    for prop_df in self.data.values()
                    for col in prop_df.columns
                    if col not in self.domain_keys
                )
            )

            return {"available_properties": props, "available_models": models}

        if model_name is not None and property_name is None:
            # Return a dictionary: {property: Series of model values}
            result = {}
            for prop, df in self.data.items():
                if model_name in df.columns:
                    cols = self.domain_keys + [model_name]
                    result[prop] = df[cols]
                else:
                    result[prop] = f"[Model '{model_name}' not available]"
            return result

        if property_name is not None:
            if property_name not in self.data:
                raise KeyError(f"Property '{property_name}' not found.")

            df = self.data[property_name]

            if model_name is None:
                return df  # Full property DataFrame

            if model_name not in df.columns:
                raise KeyError(
                    f"Model '{model_name}' not found in property '{property_name}'."
                )

            return df[model_name]

    def separate_points_distance_allSets(self, list1, list2, distance1, distance2):
        """
        Separates points into groups based on proximity thresholds.

        Args:
            list1 (list[tuple[float, float]]): Points to classify as (x, y) tuples.
            list2 (list[tuple[float, float]]): Reference points as (x, y) tuples.
            distance1 (float): First proximity threshold.
            distance2 (float): Second proximity threshold.

        Returns:
            tuple[list[int], list[int], list[int]]: Three lists of indices from `list1`:
                - Within `distance1` of any point in `list2`.
                - Within `distance2` but not `distance1`.
                - Beyond `distance2`.
        """
        train = []
        validation = []
        test = []

        train_list_coordinates = []
        validation_list_coordinates = []
        test_list_coordinates = []

        for i in range(len(list1)):
            point1 = list1[i]
            close = False
            for point2 in list2:
                if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance1:
                    close = True
                    break
            if close:
                train.append(point1)
                train_list_coordinates.append(i)
            else:
                close2 = False
                for point2 in list2:
                    if np.linalg.norm(np.array(point1) - np.array(point2)) <= distance2:
                        close2 = True
                        break
                if close2:
                    validation.append(point1)
                    validation_list_coordinates.append(i)
                else:
                    test.append(point1)
                    test_list_coordinates.append(i)

        return (
            train_list_coordinates,
            validation_list_coordinates,
            test_list_coordinates,
        )

    def split_data(
        self, data_dict, property_name, splitting_algorithm="random", **kwargs
    ):
        """
        Splits data into training, validation, and test sets.

        Args:
            data_dict (dict[str, pandas.DataFrame]): Output from `load_data()`.
            property_name (str): Property to use for splitting.
            splitting_algorithm (str): 'random' or 'inside_to_outside'.
            **kwargs: Algorithm-specific parameters:
                - `random`: `train_size` (float), `val_size` (float), `test_size` (float).
                - `inside_to_outside`: `stable_points` (list[tuple[float, float]]), `distance1` (float), `distance2` (float).

        Returns:
            tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]: (train, validation, test) DataFrames.

        Raises:
            ValueError: For invalid algorithm or missing parameters.
        """
        if property_name not in data_dict:
            raise ValueError(
                f"Property '{property_name}' not found in the provided data dictionary."
            )

        data = data_dict[property_name]

        if isinstance(data, pd.DataFrame):
            indexable_data = data.reset_index(drop=True)
            point_list = list(indexable_data.itertuples(index=False, name=None))
        else:
            raise TypeError(
                "Data for the specified property must be a pandas DataFrame."
            )

        if splitting_algorithm == "random":
            required = ["train_size", "val_size", "test_size"]
            if not all(k in kwargs for k in required):
                raise ValueError(f"Missing required kwargs for 'random': {required}")

            train_size = kwargs["train_size"]
            val_size = kwargs["val_size"]
            test_size = kwargs["test_size"]

            if not np.isclose(train_size + val_size + test_size, 1.0):
                raise ValueError("train_size + val_size + test_size must equal 1.0")

            # Random split using indexes
            train_idx, temp_idx = train_test_split(
                indexable_data.index, train_size=train_size, random_state=1
            )
            val_rel = val_size / (val_size + test_size)
            val_idx, test_idx = train_test_split(
                temp_idx, test_size=1 - val_rel, random_state=1
            )

        elif splitting_algorithm == "inside_to_outside":
            required = ["stable_points", "distance1", "distance2"]
            if not all(k in kwargs for k in required):
                raise ValueError(
                    f"Missing required kwargs for 'inside_to_outside': {required}"
                )

            stable_points = kwargs["stable_points"]
            distance1 = kwargs["distance1"]
            distance2 = kwargs["distance2"]

            (
                train_idx,
                val_idx,
                test_idx,
            ) = self.separate_points_distance_allSets(
                point_list, stable_points, distance1, distance2
            )
        else:
            raise ValueError(
                "splitting_algorithm must be either 'random' or 'inside_to_outside'"
            )

        train_data = indexable_data.iloc[train_idx]
        val_data = indexable_data.iloc[val_idx]
        test_data = indexable_data.iloc[test_idx]

        return train_data, val_data, test_data

    def get_subset(self, property_name, filters=None, models_to_include=None):
        """
        Returns a filtered subset of data for a property.

        Args:
            property_name (str): Property to filter.
            filters (dict, optional): Domain filtering rules.
            models_to_include (list[str], optional): Models to include.

        Returns:
            pandas.DataFrame: Filtered DataFrame.

        Raises:
            ValueError: If property not found.
        """
        if property_name not in self.data:
            raise ValueError(f"Property '{property_name}' not found in dataset.")

        df = self.data[property_name].copy()

        # Apply row-level filters (domain-based)
        if filters:
            for column, condition in filters.items():
                if column == "multi" and callable(condition):
                    df = df[df.apply(condition, axis=1)]
                elif callable(condition):
                    df = df[condition(df[column])]
                elif isinstance(condition, tuple) and len(condition) == 2:
                    df = df[(df[column] >= condition[0]) & (df[column] <= condition[1])]
                elif isinstance(condition, list):
                    df = df[df[column].isin(condition)]
                else:
                    df = df[df[column] == condition]

        # Optionally restrict to a subset of models
        if models_to_include is not None:
            domain_keys = [col for col in ["N", "Z"] if col in df.columns]
            allowed_cols = domain_keys + [
                m for m in models_to_include if m in df.columns
            ]
            df = df[allowed_cols]

        return df
