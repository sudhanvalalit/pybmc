import pandas as pd
from dataset_nuclear import Dataset

dataset = Dataset()
data = dataset.load_data()

# Check loaded data
# print("Loaded Data:", data)

# # Check extracted common isotopes
# common_isotopes = dataset.extract_common_isotopes()
# print("Common Isotopes:", common_isotopes)

# # Check train/val/test split
# train, val, test = dataset.split_data(0.6, 0.2, 0.2)
# print("Train Data:", train.head())
# print("Validation Data:", val.head())
# print("Test Data:", test.head())

# # Check subset retrieval
subset = dataset.get_subset(domain_X="even-even", N_range=(2, 20), Z_range=(2, 10))
print("Subset Data:", subset.head())
