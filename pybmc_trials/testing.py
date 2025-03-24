import pandas as pd
from dataset import Dataset

dataset = Dataset()
file_path = 'data/selected_data.h5'

ds = Dataset(data_source=file_path)
models_to_load = ["ME2", "PC1", "UNEDF1"]
data = ds.load_data(models=models_to_load)

# LOADING DATA

# for model_name, df in data.items():
#     print(f"{model_name} shape:", df.shape)
#     print(df.head())


# FILTERING DATA


# multi

# filtered_multi = ds.get_subset(
#     data=data["ME2"],
#     filters={
#         "multi": lambda row: (row["Z"] % 2 == 0) and (row["N"] % 2 == 0) and (row["BE"] > 1800)
#     },
#     apply_to_all_models=True
# )
# print(filtered_multi)


# callable (lamda)

# filtered = ds.get_subset(
#     data=data["ME2"],
#     filters={
#         "BE": lambda col: col > 1500,
#         "Z": (70, 90)
#     }
# )
# print(filtered.head())


# range condition

# filtered_range = ds.get_subset(
#     data=data["ME2"],
#     filters={"Z": (20, 40)}
# )
# print(filtered_range.head())


# SPLITTING

# model_data = data["ME2"]

# train, val, test = ds.split_data(model_data, train_size=0.7, val_size=0.15, test_size=0.15)

# print("Train size:", len(train))
# print("Val size:", len(val))
# print("Test size:", len(test))







