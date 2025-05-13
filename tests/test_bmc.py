import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from pybmc.bmc import BayesianModelCombination
from pybmc.sampling_utils import coverage, rndm_m_random_calculator


class TestBayesianModelCombination(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset as a pandas DataFrame
        self.selected_models_dataset = pd.DataFrame({
            'model1': [10, 20, 30],
            'model2': [15, 25, 35],
            'model3': [12, 22, 32],
            'truth':  [11, 21, 31],
            'N': [1, 2, 3],  
            'Z': [10, 10, 10]
        })

        # Define models used for training 
        self.models = ['model1', 'model2', 'model3', 'truth']
        self.bmc = BayesianModelCombination(self.models, self.selected_models_dataset)
    def test_orthogonalize(self):
        # Prepare input training data (a subset of the dataset)
        data = self.selected_models_dataset.iloc[[0, 1, 2]]  
        components_kept = 2

        result = self.bmc.orthogonalize(data, components_kept=2)

        self.assertIsNone(result)

        # Check required attributes were set
        self.assertTrue(hasattr(self.bmc, 'centered_experiment_train'))
        self.assertTrue(hasattr(self.bmc, 'U_hat'))
        self.assertTrue(hasattr(self.bmc, 'S_hat'))
        self.assertTrue(hasattr(self.bmc, 'Vt_hat'))
        self.assertTrue(hasattr(self.bmc, 'Vt_hat_normalized'))
        self.assertTrue(hasattr(self.bmc, '_predictions_mean_train'))

        # Check expected shapes
        n_points = data.shape[0]
        self.assertEqual(self.bmc.U_hat.shape, (n_points, components_kept))
        self.assertEqual(self.bmc.S_hat.shape[0], components_kept)
        self.assertEqual(self.bmc.Vt_hat.shape, (components_kept, len(self.bmc.models)))

    def test_train(self):
        train_data = self.selected_models_dataset.iloc[[0, 1, 2]]
        components_kept = 2
        self.bmc.orthogonalize(train_data, components_kept)

        training_options = {
            'iterations': 100,
            'b_mean_prior': np.zeros(components_kept),
            'b_mean_cov': np.diag(self.bmc.S_hat[:components_kept]**2),
            'nu0_chosen': 1.0,
            'sigma20_chosen': 0.02
        }

        self.bmc.train(training_options)

        self.assertTrue(hasattr(self.bmc, 'samples'))
        self.assertEqual(self.bmc.samples.shape[1], components_kept + 1)  # +1 for sigma


    def test_predict(self):
        # Prepare training (same as test_train)
        data = self.selected_models_dataset.iloc[[0, 1, 2]]  # or use a defined `train_data`
        components_kept = 2

        self.bmc.orthogonalize(data, components_kept)

        training_options = {
            'iterations': 100,
            'b_mean_prior': np.zeros(components_kept),
            'b_mean_cov': np.diag(self.bmc.S_hat[:components_kept]**2),
            'nu0_chosen': 1.0,
            'sigma20_chosen': 0.02
        }
        self.bmc.train(training_options)

        # Prepare aligned test data (must match training models' order)
        X_input = self.selected_models_dataset.iloc[:3][['model1', 'model2', 'model3']]

        rndm_m, (lower, median, upper) = self.bmc.predict(X_input)

        # Assertions
        self.assertEqual(rndm_m.shape[1], len(X_input))  # should have one column per point
        self.assertEqual(len(lower), len(X_input))
        self.assertEqual(len(median), len(X_input))
        self.assertEqual(len(upper), len(X_input))

        self.assertTrue(np.all(lower <= median))
        self.assertTrue(np.all(median <= upper))

    def test_evaluate(self):
        self.bmc.models = ['model1', 'model2']
        self.bmc.samples = np.random.rand(100, 3)
        self.bmc.Vt_hat = np.ones((2, 2)) 

        with patch("pybmc.bmc.rndm_m_random_calculator") as mock_rndm, \
            patch("pybmc.bmc.coverage") as mock_coverage:

            mock_rndm.return_value = (np.array([0.0, 0.0, 0.0]), [0.1, 0.2, 0.3])
            mock_coverage.return_value = 0.85

            result = self.bmc.evaluate(method=["random", "coverage"])

            self.assertIn("random", result)
            self.assertIn("coverage", result)
            self.assertEqual(len(result["random"]), 3)
            self.assertIsInstance(result["coverage"], float)


if __name__ == '__main__':
    unittest.main()
