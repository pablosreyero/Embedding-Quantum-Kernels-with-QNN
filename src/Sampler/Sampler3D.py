import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from mpl_toolkits.mplot3d import Axes3D

DEFAULT_N_POINTS = 100
DEFAULT_SPREAD = 1

def generate_random_points(n_points, spread):
    return  spread * (2 * np.random.rand(n_points, 3) - 1)

class Sampler3D():

    @staticmethod
    def torus(n_points:int=DEFAULT_N_POINTS, inner_radius:float=0.25, outer_radius:float=0.75, spread:float=DEFAULT_SPREAD):
        assert inner_radius < outer_radius

        def is_inside_torus(point):
            x, y, z = point
            dist_from_center = np.sqrt(x ** 2 + y ** 2)
            return (dist_from_center - outer_radius) ** 2 + z ** 2 < inner_radius **2

        # Function to generate points on a torus
        points = generate_random_points(n_points, spread)
        labels = np.array([is_inside_torus(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def sphere(n_points: int = DEFAULT_N_POINTS, radius: float = 1.0, center: tuple = (0, 0, 0),
               spread: float = DEFAULT_SPREAD):
        def is_inside_sphere(point):
            x, y, z = point
            cx, cy, cz = center
            return (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 < radius ** 2

        # Generate random points
        points = generate_random_points(n_points, spread)
        # Classify points
        labels = np.array([is_inside_sphere(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def shell(n_points: int = DEFAULT_N_POINTS, inner_radius: float = 0.4, outer_radius: float = 0.7,
              center: tuple = (0, 0, 0), spread: float = DEFAULT_SPREAD):
        def is_inside_spheres(point):
            x, y, z = point
            cx, cy, cz = center
            dist1 = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2
            dist2 = dist1  # assume spheres share the same center
            return dist1 < inner_radius ** 2 or dist2 < outer_radius ** 2

        # Generate random points
        points = generate_random_points(n_points, spread)

        # Classify points
        labels = np.array([is_inside_spheres(p) for p in points], dtype=int)

        return points, labels

    @staticmethod
    def cube(n_points:int=100, side_length:float=1, spread:float=DEFAULT_SPREAD):
        half_side = side_length / 2
        points = generate_random_points(n_points, spread)
        labels = np.array([(abs(x) <= half_side and abs(y) <= half_side and abs(z) <= half_side) for x, y, z in points],
                          dtype=int)
        return points, labels

    @staticmethod
    def union_spheres(n_points:int=DEFAULT_N_POINTS, centers:list=[(0,0,0), (0,0.2,0)], radius:list=[0.3, 0.5],
                      spread:float=DEFAULT_SPREAD):

        def is_inside_any_sphere(point):
            return any(np.linalg.norm(point - center) < r for center, r in zip(centers, radius))

        points = generate_random_points(n_points, spread)
        labels = np.array([is_inside_any_sphere(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def cylinder(n_points:int=DEFAULT_N_POINTS, radius:float=0.8, height:float=1.5, spread:float=DEFAULT_SPREAD):
        def is_inside_cylinder(point):
            x, y, z = point
            return (x ** 2 + y ** 2 < radius ** 2) and (abs(z) < height / 2)

        points = generate_random_points(n_points, spread)
        labels = np.array([is_inside_cylinder(p) for p in points], dtype=int)
        return points, labels

    @staticmethod
    def ellipsoid(n_points:int=DEFAULT_N_POINTS, a:float=1, b:float=2, c:float=0.2, spread:float=DEFAULT_SPREAD):
        def is_inside_ellipsoid(point):
            x, y, z = point
            return (x ** 2 / a ** 2 + y ** 2 / b ** 2 + z ** 2 / c ** 2) < 1

        points = generate_random_points(n_points, spread)
        labels = np.array([is_inside_ellipsoid(p) for p in points], dtype=int)
        return points, labels


    @staticmethod
    def pyramid(n_points:int=DEFAULT_N_POINTS, base_size:float=0.7, height:float=1.2, spread:float=DEFAULT_SPREAD):
        def is_inside_pyramid(point):
            x, y, z = point
            return (z >= 0) and (z <= height) and (abs(x) <= (1 - z / height) * base_size / 2) and (
                        abs(y) <= (1 - z / height) * base_size / 2)

        points = generate_random_points(n_points, spread)
        labels = np.array([is_inside_pyramid(p) for p in points], dtype=int)
        return points, labels
