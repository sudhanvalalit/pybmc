import unittest
from pybmc.models import Model


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model("test_model")

    def test_domain(self):
        self.model.domain = [1, 2, 3, 4, 5]
        self.assertEqual(self.model.domain, [1, 2, 3, 4, 5])

    def test_range(self):
        self.model.range = [10, 20, 30, 40, 50]
        self.assertEqual(self.model.range, [10, 20, 30, 40, 50])


if __name__ == '__main__':
    unittest.main()
