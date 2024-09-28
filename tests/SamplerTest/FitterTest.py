from icecream import ic
import sys
import os

# Get the current script directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Define the parent directory containing the 'src' folder
parent_dir = os.path.abspath(os.path.join(current_dir, '../..'))

# Add the parent directory to sys.path
sys.path.append(parent_dir)
import unittest
from src.Sampler import Sampler, fit_argument, Sampler3D
from src.EQK import Constant as Constant

target_per = 0.5
tol_per = 0.1

def test_circle():
    def func(radius: float):
        return Sampler().circle(n_points=1000, radius=radius)

    kwargs = {'radius': 0.1}
    kwargs = fit_argument(func, target_percentage=target_per, tolerance=tol_per,  **kwargs)

    points, labels = func(**kwargs)

    constant = Constant()
    constant.train(points, labels)
    acc = constant.get_accuracy(points, labels)

    ic(acc)
    assert target_per - tol_per <= acc <= target_per + tol_per


def test_Toris():
    def func(radius: float):
        return Sampler().annulus(n_points=1000, inner_radius=radius, outer_radius=0.2)
    kwargs = {'radius': 0.1}
    ic(kwargs)
    kwargs = fit_argument(func, target_percentage=target_per, tolerance=tol_per, **kwargs)

    points, labels = func(**kwargs)

    constant = Constant()
    constant.train(points, labels)
    acc = constant.get_accuracy(points, labels)

    ic(acc)
    assert target_per - tol_per <= acc <= target_per + tol_per

if __name__ == '__main__':
    test_circle()
    test_Toris()