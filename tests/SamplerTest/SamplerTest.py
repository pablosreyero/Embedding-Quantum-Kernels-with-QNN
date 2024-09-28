from src.Sampler import Sampler
import unittest
from icecream import ic
from src.visualization import plot_dataset


class Samplertest(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.n_points = 3000


    def test_circle(self):
        data, labels = Sampler().circle(radius=0.5, n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='circle')


    def test_multi_circle(self):
        data, labels = Sampler().multi_circle(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='multi_circle')


    def test_annulus(self):
        data, labels = Sampler().annulus(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='annulus')


    def test_sinus(self):
        data, labels = Sampler().sinus(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='sinus')

    def test_sinus_dif(self):
        data, labels = Sampler().sinus_dif(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='sinus dif')


    def test_corners(self):
        data, labels = Sampler().corners(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='corners')
        
    def test_spiral(self):
        data, labels = Sampler().spiral(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='spiral')

    def test_checkerboard(self):
        data, labels = Sampler().checkerboard(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='checkerboard')

    def test_rectangle(self):
        data, labels = Sampler().rectangle(n_points=self.n_points)
        self.assertEqual(data.shape, (self.n_points, 2))
        self.assertEqual(labels.shape, (self.n_points,))
        plot_dataset(data, labels, title='rectangle')


if __name__ == '__main__':
    unittest.main()
