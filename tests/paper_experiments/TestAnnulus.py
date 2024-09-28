"""
En este script quiero ver si obtengo los mismos resultados que obtiene Pablo en su paper para la red de 1 neurona
"""
import unittest
import numpy as np
from src.Sampler import Sampler
from src.visualization import plot_predictions_2d
from src.dataReuploading import SingleQNN, MultiQNN


class SingleQNNExperiment(unittest.TestCase):

    def test_circles(self):
        """
        Results from Appendix G TABLE I
        """
        points, labels = Sampler().annulus(n_points=100)
        points_test, labels_test = Sampler().annulus(n_points=100)
        optimizer_params = {0: {'learning_rate': 0.05}, -1: {'learning_rate':0.005}}
        batch_size = 24
        n_epochs = {0: 30, -1: 10}

        qnn = MultiQNN(num_qubits=1, )

