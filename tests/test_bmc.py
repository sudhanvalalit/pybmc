import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pybmc.bmc import BayesianModelCombination
from pybmc.models import Model
from pybmc.sampling_utils import coverage, rndm_m_random_calculator


class TestBayesianModelCombination(unittest.TestCase):
    def setUp(self):
        self.models = [
            Model("model1", np.array([1, 2, 3]), np.array([10, 20, 30])),
            Model("model2", np.array([1, 2, 3]), np.array([15, 25, 35])),
            Model("model3", np.array([1, 2, 3]), np.array([12, 22, 32]))
        ]
        self.options = {'use_orthogonalization': True}
        self.bmc = BayesianModelCombination(self.models, self.options)

    def test_train(self):
        training_data = [1, 2, 3, 4, 5]
        result = self.bmc.train(training_data)
        # Add assertions to check the result of train method
        self.assertIsNone(result)
        # Add more specific assertions based on the expected behavior of
        # train method

    def test_predict(self):
        X = [1, 2, 3, 4, 5]
        result = self.bmc.predict(X)
        # Add assertions to check the result of predict method
        self.assertIsNotNone(result)
        # Add more specific assertions based on the expected behavior of
        # predict method

    def test_evaluate(self):
        # Create a dummy DataFrame
        data = {
            'N': [10, 20, 30],
            'Z': [20, 30, 40],
            'model1': [1.0, 2.0, 3.0],
            'model2': [1.5, 2.5, 3.5],
            'truth': [1.2, 2.3, 3.1]
        }
        df = pd.DataFrame(data)

        # Instantiate the BMC object
        bmc = BayesianModelCombination(models_truth=['model1', 'model2', 'truth'], selected_models_dataset=df)
        bmc.models = ['model1', 'model2']  # ensure this is available
        bmc.samples = np.random.rand(100, 3)  # dummy samples for evaluation

        with patch("pybmc.bmc.rndm_m_random_calculator") as mock_rndm, \
            patch("pybmc.bmc.coverage") as mock_coverage:

            # Define fake return values for the mocks
            mock_rndm.return_value = (np.array([0.0, 0.0, 0.0]), [0.1, 0.2, 0.3])
            mock_coverage.return_value = 0.85

            result = bmc.evaluate(method=["random", "coverage"])

            self.assertIn("random", result)
            self.assertIn("coverage", result)
            self.assertEqual(len(result["random"]), 3)
            self.assertIsInstance(result["coverage"], float)


    def test_orthogonalize(self):
        data = [1, 2, 3, 4, 5]
        result = self.bmc.orthogonalize(data)
        # Add assertions to check the result of orthogonalize method
        self.assertIsNone(result)
        # Add more specific assertions based on the expected behavior of
        # orthogonalize method

    def test_constructor_requires_models(self):
        with self.assertRaises(ValueError):
            BayesianModelCombination("invalid_model_list")

    def test_options_flag(self):
        self.bmc.options['use_orthogonalization'] = False
        data = [1, 2, 3, 4, 5]
        result = self.bmc.orthogonalize(data)
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
