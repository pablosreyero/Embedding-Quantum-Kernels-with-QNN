import unittest
from icecream import ic
from src.Sampler import Sampler3D
from src.visualization import plot_dataset

n_points = 2000

class Sampler3DTest(unittest.TestCase):

    def test_sphere(self):
        points, labels = Sampler3D().sphere(n_points)
        plot_dataset(points, labels, title='sphere')

    def test_shell(self):
        points, labels = Sampler3D().shell(n_points)
        plot_dataset(points, labels, title='shell')

    def test_torus(self):
        points, labels = Sampler3D().torus(n_points)
        plot_dataset(points, labels, title='toris')

    def test_cube(self):
        points, labels = Sampler3D().cube(n_points)
        plot_dataset(points, labels, title='cube')

    def test_spheres(self):
        points, labels = Sampler3D().union_spheres(n_points)
        plot_dataset(points, labels, title='spheres')

    def test_cylinder(self):
        points, labels = Sampler3D().cylinder(n_points)
        plot_dataset(points, labels, title='cylinder')

    def test_ellipsoid(self):
        points, labels = Sampler3D().ellipsoid(n_points)
        plot_dataset(points, labels, title='ellipsoid')

    def test_pyramid(self):
        points, labels = Sampler3D().cylinder(n_points)
        plot_dataset(points, labels, title='pyramid')

if __name__ == '__main__':
    unittest.main()
