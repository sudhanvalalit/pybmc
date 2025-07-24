import unittest
import pandas as pd
from pybmc.bmc import BayesianModelCombination


class TestBayesianModelCombination(unittest.TestCase):
    def setUp(self):
        # Expanded mock dataset (6 rows)
        self.df = pd.DataFrame(
            {
                "x": [1, 2, 3, 4, 5, 6],
                "y": [10, 11, 12, 13, 14, 15],
                "truth": [11, 21, 31, 41, 51, 61],
                "model1": [10, 20, 30, 40, 50, 60],
                "model2": [15, 25, 35, 45, 55, 65],
                "model3": [12, 30, 32, 43, 58, 67],
            }
        )

        self.property = "target"
        self.data_dict = {self.property: self.df}
        self.models = ["model1", "model2", "model3", "truth"]

        self.bmc = BayesianModelCombination(
            models_list=self.models,
            data_dict=self.data_dict,
            truth_column_name="truth",
        )

        # Manually designate training data (first 4 rows)
        self.train_df = self.df.iloc[:4]

    def test_orthogonalize(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.assertTrue(hasattr(self.bmc, "centered_experiment_train"))
        self.assertEqual(self.bmc.U_hat.shape[0], self.train_df.shape[0])
        self.assertEqual(self.bmc.Vt_hat.shape[1], len(self.bmc.models))

    def test_train(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        self.assertEqual(
            self.bmc.samples.shape[1], 3
        )  # 2 components + 1 sigma

    def test_predict(self):
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        # Use all rows for prediction input
        X = self.df[["x", "y", "model1", "model2", "model3"]].copy()
        rndm_m, lower_df, median_df, upper_df = self.bmc.predict2(
            self.property
        )

        self.assertEqual(rndm_m.shape[1], len(X))
        self.assertIn("Predicted_Lower", lower_df.columns)
        self.assertIn("Predicted_Median", median_df.columns)
        self.assertIn("Predicted_Upper", upper_df.columns)

    def test_evaluate(self):
        # Run orthogonalization and training first
        self.bmc.orthogonalize(
            property=self.property, train_df=self.train_df, components_kept=2
        )
        self.bmc.train()
        eval_results = self.bmc.evaluate()

        # Assertions
        self.assertIsInstance(eval_results, list)
        self.assertEqual(len(eval_results), 21)
        self.assertTrue(all(isinstance(x, (int, float)) for x in eval_results))


if __name__ == "__main__":
    unittest.main()
