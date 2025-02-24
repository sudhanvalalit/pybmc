import unittest
import numpy as np
from pybmc.models import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model("test_model", np.array([1, 2, 3, 4, 5]), np.array([10, 20, 30, 40, 50]))

    def test_x(self):
        np.testing.assert_array_equal(self.model.x, np.array([1, 2, 3, 4, 5]))

    def test_y(self):
        np.testing.assert_array_equal(self.model.y, np.array([10, 20, 30, 40, 50]))


if __name__ == '__main__':
    unittest.main()
