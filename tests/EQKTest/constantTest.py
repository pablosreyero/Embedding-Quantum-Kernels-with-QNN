from src.EQK import Constant
import unittest
from src.Sampler import Sampler
import numpy as np


class ConstantEQKTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_set, cls.train_labels = Sampler().circle(n_points=500)
        cls.test_set, cls.test_labels = Sampler().circle(n_points=30)

        cls.predictor = Constant()

    def test_train(self):
        self.predictor.train(self.train_set, self.train_labels, self.test_set, self.test_labels, silent=False)
        self.assertTrue(self.predictor.trained)

    def test_predict(self):
        self.predictor.train(self.train_set, self.train_labels, self.test_set, self.test_labels)
        actual = self.predictor.predict(self.test_set)

        try:
            constant = 0
            self.assertTrue((actual, np.full(shape=actual.shape, fill_value=constant)))
        except AssertionError:
            constant = 1
            self.assertTrue((actual, np.full(shape=actual.shape, fill_value=constant)))

        print(f'{constant = }')

    def test_get_accuracy(self):
        self.predictor.train(self.train_set, self.train_labels, self.test_set, self.test_labels)
        acc = self.predictor.get_accuracy(self.test_set, self.test_labels)
        self.assertTrue(0 <= acc <= 1)


if __name__ == '__main__':
    unittest.main()

