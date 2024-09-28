import unittest
from src.utils import get_function
from src.Sampler import Sampler


class UtilsTest(unittest.TestCase):

    def test_get_function(self):
        circle_loaded = get_function('circle', Sampler)

        actual_points, actual_labels = circle_loaded(n_points=100)
        expected_points, expected_labels = Sampler().circle(n_points=100)

        self.assertEqual(actual_points.shape, expected_points.shape)
        self.assertEqual(actual_labels.shape, expected_labels.shape)
