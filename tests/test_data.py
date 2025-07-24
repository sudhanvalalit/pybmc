import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch
from pybmc.data import Dataset

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.sample_df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4],
                "y": [1, 2, 3, 4],
                "target": [10, 20, 30, 40],
                "modelA": [9, 19, 29, 39],
                "modelB": [11, 21, 31, 41],
            }
        )
        self.dataset = Dataset(data_source="fake_path.h5")
        self.dataset.data = {"target": self.sample_df}

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_csv")
    def test_load_data_csv(self, mock_read_csv, mock_exists):
        mock_read_csv.return_value = pd.DataFrame(
            {
                "x": [1, 2],
                "y": [1, 2],
                "target": [10, 20],
                "model": ["modelA", "modelB"],
            }
        )

        dataset = Dataset(data_source="fake_path.csv")
        result = dataset.load_data(
            models=["modelA", "modelB"],
            keys=["target"],
            domain_keys=["x", "y"],
            model_column="model",
        )

        self.assertIn("target", result)
        self.assertTrue(
            all(
                col in result["target"].columns
                for col in ["x", "y", "modelA", "modelB"]
            )
        )

    @patch("pybmc.data.os.path.exists", return_value=True)
    @patch("pybmc.data.pd.read_hdf")
    def test_load_data_h5(self, mock_read_hdf, mock_exists):
        model_data = pd.DataFrame(
            {"x": [1, 2], "y": [1, 2], "target": [10, 20]}
        )
        mock_read_hdf.side_effect = lambda file, key: model_data

        dataset = Dataset(data_source="fake_path.h5")
        result = dataset.load_data(
            models=["modelA", "modelB"],
            keys=["target"],
            domain_keys=["x", "y"],
        )

        self.assertIn("target", result)
        self.assertIsInstance(result["target"], pd.DataFrame)
        self.assertTrue(
            all(
                col in result["target"].columns
                for col in ["x", "y", "modelA", "modelB"]
            )
        )

    def test_split_data_random(self):
        data_dict = {"target": self.sample_df}
        train, val, test = self.dataset.split_data(
            data_dict=data_dict,
            property_name="target",
            splitting_algorithm="random",
            train_size=0.6,
            val_size=0.2,
            test_size=0.2,
        )
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(self.sample_df))

    def test_split_data_inside_to_outside(self):
        coords_only = self.sample_df[["x", "y"]].copy()
        self.dataset.data = {"target": coords_only}

        train, val, test = self.dataset.split_data(
            data_dict={"target": coords_only},
            property_name="target",
            splitting_algorithm="inside_to_outside",
            stable_points=[(1, 1)],
            distance1=0.1,
            distance2=100,
        )
        total = len(train) + len(val) + len(test)
        self.assertEqual(total, len(coords_only))
        self.assertTrue(
            any(
                (row["x"] == 1 and row["y"] == 1)
                for _, row in train.iterrows()
            )
        )

    def test_get_subset_basic_filter(self):
        filters = {"x": lambda x: x > 2}
        result = self.dataset.get_subset(
            property_name="target",
            filters=filters,
            models_to_include=["modelA", "modelB"],
        )
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(
            (result["modelA"].notna()).all()
        )  # Check filtered rows exist
        self.assertTrue(
            all(col in result.columns for col in ["modelA", "modelB"])
        )


if __name__ == "__main__":
    unittest.main()
