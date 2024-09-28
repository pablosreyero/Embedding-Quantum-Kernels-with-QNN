import unittest
from src.dataReuploading import MultiQNN, SingleQNN
from src.Sampler import Sampler
from pennylane import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from warnings import warn
from time import time
from src.visualization import plot_predictions_2d
from src.utils import increase_dimensions


class SingleQNNTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_set, cls.train_label = Sampler().circle(n_points=100)
        cls.test_set, cls.test_label = Sampler().circle(n_points=20)

        cls.n_qubits = 2
        cls.n_layers = 7
        cls.qnn = MultiQNN(num_qubits=cls.n_qubits, num_layers=cls.n_layers)
    def test_train(self):
        # run on simpler dataset
        train_set, train_label = Sampler().circle(n_points=20)
        test_set, test_label = Sampler().circle(n_points=10)
        df = self.qnn.train(train_set, train_label, test_set, test_label, silent=False)
        print('Final stats')
        print(df)

    def test_train_without_test(self):
        df = self.qnn.train(self.train_set, self.train_label, silent=False)
        print('Final stats')
        print(df)

    def test_cost_depends_on_parameters(self):

        set = increase_dimensions(self.train_set)

        params_before = [
                        np.random.uniform(size=(self.n_layers, 3), requires_grad=True)
                            for _ in range(2 * self.n_qubits - 1)
                        ]
        cost_before = self.qnn.cost(params_before, set, self.train_label)

        params_after = [
                        np.random.uniform(size=(self.n_layers, 3), requires_grad=True)
                            for _ in range(2 * self.n_qubits - 1)
                        ]
        cost_after = self.qnn.cost(params_after, set, self.train_label)

        for p_before, p_after in zip(params_before, params_after):
            self.assertFalse((p_before == p_after).all())

        self.assertNotEquals(cost_after, cost_before)

    @unittest.skip('This test takes too long')
    def test_performance(self):
        train_set, train_label = Sampler().circle(n_points=400)
        test_set, test_label = Sampler().circle(n_points=50)
        validation_set, validation_label = Sampler().circle(n_points=100)
        qnn = MultiQNN(num_qubits=4, num_layers=6)

        stats = qnn.train(train_set, train_label, test_set, test_label, n_epochs=5, batch_size=20, silent=False)
        print(stats)
        predictions = qnn.predict(validation_set)

        fig, ax = plt.subplots(1,2)
        for point, prediction, valid_label in zip(validation_set, predictions, validation_label):
            color_pred = 'red' if prediction == 0 else 'blue'
            color_expected = 'red' if valid_label == 0 else 'blue'
            ax[0].scatter(point[0], point[1], color=color_pred)
            ax[1].scatter(point[0], point[1], color=color_expected)

        ax[0].set_title('Predictions')
        ax[1].set_title('Real')

        plt.show()

    def test_save_and_load(self):

        path = 'data/test_save_and_load'

        self.qnn.train(self.train_set, self.train_label, self.test_set, self.test_label, silent=False)
        self.qnn.save_qnn(path)

        qnn_loaded = MultiQNN().load_qnn(path)

        self.assertEqual(self.qnn.training_info, qnn_loaded.training_info)

    def test_loaded_qnn_is_able_to_predict(self):
        path = 'data/test_save_and_load'
        qnn = MultiQNN().load_qnn(path)
        accuracy = qnn.get_accuracy(self.test_set, self.test_label)
        ic(accuracy)


    def test_multiqnn_equals_singleqnn_circles(self):
        """
        Compare whether MultiQNN with num_qubits = 1 works similarly to SingleQNN
        """
        data, labels = Sampler().circle(n_points=200)
        data_test, labels_test = Sampler().circle(n_points=200)

        layers = 5

        multiqnn = MultiQNN(num_layers=layers, num_qubits=1)
        singleqnn = SingleQNN(num_layers=layers)

        df_multi = multiqnn.train(data, labels, data_test, labels_test, silent=False)
        df_single = singleqnn.train(data, labels, data_test, labels_test, silent=False)

        ic(df_single, df_multi)

        # Ensure that final loss does not differ in more than a 7.5%
        loss_single = df_single['loss'].values[-1]
        loss_multi = df_multi['loss'].values[-1]

        # Print predictions
        prediction_multi = multiqnn.predict(data_test)
        prediction_single = singleqnn.predict(data_test)
        fig, ax = plt.subplots(1,2)
        plot_predictions_2d(data_test, prediction_multi, labels_test, axis=ax[0],
                            title=f"multiqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plot_predictions_2d(data_test, prediction_single, labels_test, axis=ax[1],
                            title=f"singleqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plt.show()


        try:
            self.assertLessEqual(abs(loss_single - loss_multi)/loss_single, 0.075)
        except AssertionError as e:   # If test fails but multi gets better results, it is ok
            if loss_multi < loss_single:
                warn('MultiQNN gets results better than the tolerance percentage. '
                     'Test is valid though we should check results', stacklevel=2)
            else:
                raise e


    def test_multiqnn_equals_singleqnn_corners(self):
        """
        Compare whether MultiQNN with num_qubits = 1 works similarly to SingleQNN
        """
        data, labels = Sampler().corners(n_points=200)
        data_test, labels_test = Sampler().circle(n_points=50)

        layers = 5

        multiqnn = MultiQNN(num_layers=layers, num_qubits=1)
        singleqnn = SingleQNN(num_layers=layers)

        df_multi = multiqnn.train(data, labels, data_test, labels_test, silent=False)
        df_single = singleqnn.train(data, labels, data_test, labels_test, silent=False)

        ic(df_single, df_multi)

        # Ensure that final loss does not differ in more than a 7.5%
        loss_single = df_single['loss'].values[-1]
        loss_multi = df_multi['loss'].values[-1]

        # Print predictions
        prediction_multi = multiqnn.predict(data_test)
        prediction_single = singleqnn.predict(data_test)
        fig, ax = plt.subplots(1,2)
        plot_predictions_2d(data_test, prediction_multi, labels_test, axis=ax[0],
                            title=f"multiqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plot_predictions_2d(data_test, prediction_single, labels_test, axis=ax[1],
                            title=f"singleqnn prediction - accuracy {multiqnn.get_accuracy(data, labels)}")
        plt.show()


        try:
            self.assertLessEqual(abs(loss_single - loss_multi)/loss_single, 0.075)
        except AssertionError as e:   # If test fails but multi gets better results, it is ok
            if loss_multi < loss_single:
                warn('MultiQNN gets results better than the tolerance percentage. '
                     'Test is valid though we should check results', stacklevel=2)
            else:
                raise e


    def test_parallelization(self):
        points, labels = Sampler().annulus(n_points=100)
        ic(points[1], labels[1])

        qnn_parallel = MultiQNN(num_qubits=3, num_layers=3, parallel=True, n_jobs=5)
        t_0 = time()
        qnn_parallel.train(points, labels)
        t_parallel = time() - t_0

        qnn_series = MultiQNN(num_qubits=3, num_layers=3, parallel=False)
        t_0 = time()
        qnn_series.train(points, labels)
        t_series = time() - t_0

        ic(t_parallel, t_series)
        self.assertLess(t_parallel, t_series)


    def test_parallelization_can_run_circuits(self):
        qnn = MultiQNN(num_qubits=3, num_layers=3, parallel=True, n_jobs=2)
        dm = qnn.create_dm_labels(3)
        ic(qnn.qnn(qnn.params, [0.5, 0.5], dm[0]))

    def test_forward(self):
        result = self.qnn.forward(self.train_set, n_shots=10)
        self.assertEqual(self.train_set.shape[0], result.shape[0])

        # Mode is right for just one shot
        result = self.qnn.forward(self.train_set, n_shots=1)
        self.assertEqual(self.train_set.shape[0], result.shape[0])

    def test_forward_is_correct(self):
        # I compare with the old version

        # Old forward
        t0 = time()
        expected = []
        set_ = increase_dimensions(self.train_set)
        for i in range(len(set_)):
            fidel_function = lambda y: self.qnn.qnn(params=self.qnn.params, x=set_[i], y=y)
            dm_labels = self.qnn.create_dm_labels(int((len(self.qnn.params) - 1) / 2) + 1)
            fidelities = [fidel_function(dm) for dm in dm_labels]
            best_fidel = np.argmax(fidelities)
            expected.append(best_fidel)
        t_old = time() - t0

        # New forward
        t0 = time()
        actual = self.qnn.forward(self.train_set, 1000)   # Even with 1000 shots it's faster!
        t_new = time() - t0

        self.assertLess(t_new, t_old)
        self.assertGreater(sum(actual == expected)/len(actual), 0.95)    # 5% of tolerance


if __name__ == '__main__':
    unittest.main()