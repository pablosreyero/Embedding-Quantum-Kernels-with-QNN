import pennylane as qml
from pennylane import numpy as np
from src.utils import get_optimizer, iterate_minibatches, accuracy_score, increase_dimensions
import pandas as pd
from tqdm import tqdm
import pickle
from dask.distributed import Client
from typing import Union
import os
from collections import Counter
from icecream import ic

DEFAULT_N_EPOCHS = 3


class MultiQNN:

    def __init__(self, num_qubits: int = 3, num_layers: int = 5, parallel=False, n_jobs: int = 1):

        if num_qubits > num_layers + 1:
            raise ValueError("The number of qubits cannot be greater than the number of layers + 1")

        self.num_qubits = num_qubits
        self.name = 'MultiQNN'
        self.num_layers = num_layers
        self.params = self.initialize_parameters()

        self.trained = False
        self.training_info = None

        if parallel:
            raise NotImplementedError('Not prepared yet')
            self.dask_client = Client()
            self.dev = qml.device('qiskit.aer', wires=num_qubits,
                                  backend='aer_simulator', executor=self.dask_client, max_job_size=n_jobs)
        else:
            self.dev = qml.device("default.qubit", wires=num_qubits)

    def qnn(self, params, x, y):
        qnode = qml.QNode(self._qnn, self.dev)
        return qnode(params, x, y)


    def _qnn(self, params, x, y):
        """
        Params va a contener los parametros desde 0 hasta el qubit que quieres entrenar
        Args:
            params: is a list of len == n_qubits containing dictionaries.
                    Each qubit has its own dictionary. For qubit 0, only key == 'single'. For the rest also 'controlled'
            x: data_point
            y: data_label
        """
        n_qubits = int((len(params) - 1) / 2) + 1
        self._base_circuit(x, params)
        return qml.expval(qml.Hermitian(y, wires=range(n_qubits)))

    def forward(self, point_set, n_shots:int=1):

        point_set_ = increase_dimensions(point_set)

        results = []
        for x in point_set_:

            dev_shots = qml.device("default.qubit", wires=self.num_qubits, shots=n_shots)
            qnode = qml.QNode(self._qnn_shots, dev_shots)
            output = qnode(x)

            # get the mode of the output
            if n_shots > 1:
                # get the most frequent result
                count = Counter(output)
                mode_output = count.most_common(1)[0][0]
                results.append(mode_output)
            else:
                results.append(output)

        return np.array(results, requires_grad=False).astype(np.int8)

    def _qnn_shots(self, x):
        """
        Params va a contener los parametros desde 0 hasta el qubit que quieres entrenar
        Args:
            params: is a list of len == n_qubits containing dictionaries.
                    Each qubit has its own dictionary. For qubit 0, only key == 'single'. For the rest also 'controlled'
            x: data_point
            y: data_label
        """
        self._base_circuit(x)
        return qml.sample(wires=0)

    def _base_circuit(self, x, params=None):
        if params is None:
            params = self.params

        n_qubits = int((len(params) - 1) / 2) + 1
        for layer in range(self.num_layers):
            for qubit in range(n_qubits):
                qml.Rot(*x, wires=qubit)
                if qubit != 0:
                    qml.Rot(*params[2 * qubit - 1][layer, :], wires=qubit)
                    qml.CRot(*params[2 * qubit][layer, :], wires=[qubit, qubit - 1])
                else:
                    qml.Rot(*params[qubit][layer, :], wires=qubit)

    def create_dm_labels(self, qubits):
        matrix1 = np.kron([[1, 0], [0, 0]], np.eye(2 ** (qubits - 1)))
        matrix2 = np.kron([[0, 0], [0, 1]], np.eye(2 ** (qubits - 1)))

        return np.array([matrix1, matrix2], requires_grad=False)

    def initialize_parameters(self):
        """
        Los parametros creados estan en una gran matriz. Esta matriz tiene longitud 2n_qubits -1
        El primer indice es para el qubit 0
        El segundo y tercer indice es para el qubit 1. El segundo son los parametros que no entrelazan y el tercero
            los parametros que entrelanzan
        ...
        Al qubit k != 0 le corresponde los parametros 2k-1 (parametros no entrelazantes) y 2k (parametros entrelazantes)
        """
        params = np.zeros(shape=(2 * self.num_qubits - 1, self.num_layers, 3), requires_grad=True)
        params[0, :, :] = np.random.uniform(size=(self.num_layers, 3), requires_grad=True)  # el primer qubit si tiene
        return params

    def cost(self, params, x, y):

        loss = 0
        for i in range(len(x)):
            dm_labels = self.create_dm_labels(
                int((len(params) - 1) / 2) + 1)  # what is inside = num_qubits of param selection
            f = self.qnn(params, x[i], dm_labels[int(y[i])])
            loss += (1 - f) ** 2
        return loss / len(x)

    def test(self, x, params=None):
        # MEto como parametro tambien los params para solo usar aquellos que estan entrenados

        # if params is None:
        #     params = self.params

        # fidelity_values = []
        # predicted = []
        # for i in range(len(x)):
        #     fidel_function = lambda y: self.qnn(params, x[i], y)
        #     dm_labels = self.create_dm_labels(int((len(params) - 1) / 2) + 1)
        #     fidelities = [fidel_function(dm) for dm in dm_labels]
        #     best_fidel = np.argmax(fidelities)
        #     predicted.append(best_fidel)
        #     fidelity_values.append(fidelities)
        predicted = self.forward(x, n_shots=500)


        return np.array(predicted)
        # return np.array(predicted), np.array(fidelity_values)

    def predict(self, x):
        return self.forward(x, n_shots=100)

    def train(self, data_points_train,
              data_labels_train,
              data_points_test=None,
              data_labels_test=None,
              n_epochs: Union[int, dict] = 3,
              batch_size: Union[int, dict] = 24,
              optimizer: str = 'adam',
              optimizer_parameters: dict = {},
              silent: bool = False):
        """

        Args:
            data_points_train:
            data_labels_train:
            data_points_test:
            data_labels_test:
            n_epochs: number of epochs to use. It can specify a common number (int) or different for each qubit training
                Example: n_epochs = 4, n_epochs = {0: 3, 1: 3, -1: 2}
            batch_size:  Example: batch_size = 40, n_epochs = {0: 30, 1: 40, -1: 20}
            optimizer:
            optimizer_parameters: (dictionary). Examples:
                 {'learning_rate': 0.1, 'beta1': 1}, {0: {'learning_rate': 0.1}, -1: {'learning_rate': 0.01}},
                 {0: {'learning_rate': 0.1}}
            silent:

        Returns:

        """

        stats = []

        # Prepare training and test points -> if 2d point, extend to 3d
        if data_points_train.shape[1] == 2:  # in case two dimensional points
            data_points_train_ = increase_dimensions(data_points_train)

            # In case I have provided tests points, if everything is fine, they should be 2d as well -> extend to 3d
            if data_points_test is not None:
                data_points_test_ = increase_dimensions(data_points_test)
        else:
            data_points_train_ = data_points_train
            if data_points_test is not None:
                data_points_test_ = data_points_test
                
        range_qubit = range(self.num_qubits)
        if silent:             # Keep track of the progress bar if do not display qnn training stats
            range_qubit = tqdm(range_qubit)

        for qubit_ in range_qubit:
            if not silent: print(f'%%% Training qubit {qubit_} %%%')

            ##### Get an optimizer #####
            # I can define one per qubit,
            if qubit_ in optimizer_parameters:  # opt for qubit specified
                opt_params_qubit = optimizer_parameters[qubit_]
            elif -1 in optimizer_parameters:  # opt for every qubit not explicitly specified
                opt_params_qubit = optimizer_parameters[-1]
            elif 'learning_rate' in optimizer_parameters:  # a common for all of them
                opt_params_qubit = optimizer_parameters
            else:  # Use of default optimizer
                opt_params_qubit = {}

            opt = get_optimizer(optimizer, opt_params_qubit)

            ##### Get the number of epochs to train this qubit #####
            if isinstance(n_epochs, int):
                n_epochs_ = n_epochs
            elif isinstance(n_epochs, dict):
                if qubit_ in n_epochs:
                    n_epochs_ = n_epochs[qubit_]
                elif -1 in n_epochs:
                    n_epochs_ = n_epochs[-1]
                else:
                    n_epochs_ = DEFAULT_N_EPOCHS  # default value
            else:
                raise ValueError('n_epochs must be either a int or a dictionary')

            ##### get the batch size to train the qubit #####
            if isinstance(batch_size, int):
                batch_size_ = batch_size
            elif isinstance(batch_size, dict):
                if qubit_ in batch_size:
                    batch_size_ = batch_size[qubit_]
                elif -1 in batch_size:
                    batch_size_ = batch_size[-1]
                else:
                    batch_size_ = len(data_points_train) // DEFAULT_N_EPOCHS
            else:
                raise ValueError('batch_size must be either a int or a dictionary')

            if len(data_points_train) < batch_size_:
                batch_size_ = len(data_points_train)
                n_epochs_ = 1
            else:
                batch_size_ = batch_size

            ##### Start training #####
            for it in range(n_epochs_):
                for Xbatch, ybatch in iterate_minibatches(data_points_train_, data_labels_train, batch_size=batch_size_):
                    params_up_to_qubit = self.params[:2 * qubit_ + 1]  # Take params from qubit 0 up to train qubit
                    trained_params, _, _ = opt.step(self.cost, params_up_to_qubit, Xbatch, ybatch)
                    self.params[:2 * qubit_ + 1] = trained_params

                predicted_train = self.test(data_points_train_, trained_params)
                accuracy_train = accuracy_score(data_labels_train, predicted_train)

                loss = self.cost(trained_params, data_points_train_, data_labels_train)

                if data_labels_test is not None and data_labels_train is not None:
                    predicted_test = self.test(data_points_test_, trained_params)
                    accuracy_test = accuracy_score(data_labels_test, predicted_test)
                else:
                    accuracy_test = 0

                res = [it + 1, loss, accuracy_train, accuracy_test]
                if not silent:
                    print("Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(*res))

            stats.append([qubit_] + res[1:])  # only save final results

        # Save training info
        self.training_info = res[1:]

        df = pd.DataFrame(stats)  # Coger solo stats finales
        df.columns = ['qubit', 'loss', 'train_accuracy', 'test_accuracy']
        df.set_index('qubit', inplace=True)

        self.trained = True

        return df

    def get_accuracy(self, test_data, test_label):
        predicted_train = self.test(test_data)
        accuracy_test = accuracy_score(predicted_train, test_label)
        return accuracy_test.item(0)

    def save_qnn(self, path):
        path_ = f'{path}.pkl' if path[-4:] != '.pkl' else path

        if not self.trained:
            raise Exception('The model is not trained yet, so you must not save it')

        os.makedirs(os.path.dirname(path_), exist_ok=True)
        with open(path_, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load_qnn(path):
        path_ = f'{path}.pkl' if path[-4:] != '.pkl' else path
        with open(path_, 'rb') as file:
            return pickle.load(file)
