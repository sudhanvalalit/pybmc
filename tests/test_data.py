import unittest
from pybmc.data import Dataset


class TestDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset(data=[1, 2, 3, 4, 5])

    def test_load_data(self):
        source = "data_source"
        result = self.dataset.load_data(source)
        # Add assertions to check the result of load_data method
        self.assertIsNotNone(result)
        # Add more specific assertions based on the expected behavior of
        # load_data method

    def test_split_data(self):
        train_size, val_size, test_size = 0.6, 0.2, 0.2
        result = self.dataset.split_data(train_size, val_size, test_size)
        # Add assertions to check the result of split_data method
        self.assertIsNotNone(result)
        # Add more specific assertions based on the expected behavior of
        # split_data method

    def test_get_subset(self):
        domain_X = [1, 2]
        result = self.dataset.get_subset(domain_X)
        # Add assertions to check the result of get_subset method
        self.assertIsNotNone(result)
        # Add more specific assertions based on the expected behavior of
        # get_subset method


if __name__ == '__main__':
    unittest.main()
