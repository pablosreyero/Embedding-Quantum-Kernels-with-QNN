from src.EQK import EQKn
from src.Sampler import Sampler
import unittest
from src.dataReuploading import MultiQNN
from src.utils import increase_dimensions
import os
import numpy as np
from icecream import ic
import pickle
from src.visualization import plot_predictions_2d
from src.utils import get_accuracy
from time import time
import matplotlib.pyplot as plt


class EQKnTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_set, cls.train_labels = Sampler().circle(n_points=100)
        cls.test_set, cls.test_labels = Sampler().circle(n_points=30)

        cls.train_set = increase_dimensions(cls.train_set)
        cls.test_set = increase_dimensions(cls.test_set)

        qnn_path = 'data/qnn_circle.pkl'
        if not os.path.isfile(qnn_path):
            cls.qnn = MultiQNN(num_qubits=1, num_layers=2)
            cls.qnn.train(cls.train_set, cls.train_labels, silent=False, batch_size=50)
            with open(qnn_path, 'wb') as qnn_file:
                qnn_to_save = cls.qnn
                pickle.dump(qnn_to_save, qnn_file)
        else:
            with open(qnn_path, 'rb') as qnn_file:
                cls.qnn = pickle.load(qnn_file)

        cls.eqk = EQKn(cls.qnn)

    def test_loaded_qnn(self):
        points, labels = Sampler().circle(n_points=100)
        accuracy = self.qnn.get_accuracy(points, labels)
        predictions = self.qnn.predict(points)
        ic(accuracy, predictions)
        plot_predictions_2d(points, labels, predictions)


    def test_kernel_function(self):
        k_00 = self.eqk._kernel_function(self.train_set[0], self.train_set[0])
        k_01 = self.eqk._kernel_function(self.train_set[0], self.train_set[1])
        k_10 = self.eqk._kernel_function(self.train_set[1], self.train_set[0])

        # Assert k_00 = 1
        self.assertTrue(np.isclose(k_00, 1, rtol=1e-2))
        # Assert k_01 is a probability value
        self.assertGreaterEqual(k_01, 0)
        self.assertLessEqual(k_01, 1)
        # Assert symmetry
        self.assertAlmostEqual(k_10, k_01)

    def test_compute_kernel_matrix(self):
        matrix = self.eqk._construct_train_kernel_matrix(self.train_set)
        # check shapes
        self.assertEquals(matrix.shape, (len(self.train_set), len(self.train_set)))

        # Check diagonal is full of 1
        diags = np.diag(matrix).round().astype(int)
        self.assertTrue((diags == np.ones_like(diags)).all())

        # Check a random element and compare with kernel function
        k_25 = self.eqk._kernel_function(self.train_set[2], self.train_set[5])
        self.assertEqual(k_25, matrix[2, 5])
        self.assertEqual(k_25, matrix[5, 2])

    def test_train(self):
        self.eqk.train(self.train_set, self.train_labels)

    @unittest.skip('Parallel pending to be fixed')
    def test_parallel_computation_kernel_matrix(self):

        # Create a shorter data_set not to wait to mucho
        train_set, train_labels = Sampler().circle(n_points=10)


        t_0 = time()
        matrix_parallel = self.eqk._construct_train_kernel_matrix(train_set, parallel=True, n_jobs=6)
        t_parallel = time() - t_0

        t_0 = time()
        matrix_normal = self.eqk._construct_train_kernel_matrix(train_set, parallel=False)
        t_normal = time() - t_0

        np.testing.assert_array_almost_equal(matrix_normal, matrix_parallel)
        self.assertLess(t_parallel, t_normal)

    def test_predict(self):

        # Create a shorter data_set not to wait to mucho
        train_set, train_labels = Sampler().circle(n_points=20)
        test_set, test_labels = Sampler().circle(n_points=10)

        self.eqk.train(train_set, train_labels)
        actual = self.eqk.predict(test_set)

        ic(train_labels)
        ic(actual, test_labels)

        # assert size
        self.assertEquals(len(test_labels), len(actual))
        # assert not constant
        self.assertNotEqual(np.min(actual), np.max(actual))


    def test_eqk_improves_qnn(self):

        # Create a new data_set not to get blind by overfitting
        train_set, train_labels = Sampler().circle(n_points=200)
        test_set, test_labels = Sampler().circle(n_points=100)

        # Train an predict eqk
        self.eqk.train(train_set, train_labels, silent=False)
        eqk_prediction = self.eqk.predict(test_set)
        acc_eqk = get_accuracy(eqk_prediction, test_labels)

        # predict with used qnn
        qnn_prediction = self.qnn.predict(test_set)
        acc_qnn = get_accuracy(qnn_prediction, test_labels)

        fig, ax = plt.subplots(2, 1)
        plot_predictions_2d(test_set, test_labels, qnn_prediction, axis=ax[0],
                            title=f"qnn prediction - accuracy {acc_qnn}")
        plot_predictions_2d(test_set, test_labels, eqk_prediction, axis=ax[1],
                            title=f"eqk prediction - accuracy {acc_eqk}")
        plt.show()
        plt.savefig('data/results_test_qnn_vs_eqk.png')

        self.assertGreater(acc_eqk, acc_qnn)


if __name__ == '__main__':
    unittest.main()
