import pennylane as qml
from pennylane import numpy as np
from src.utils import iterate_minibatches, accuracy_score, get_optimizer
from icecream import ic
import pandas as pd


class SingleQNN:

    def __init__(self, num_layers: int = 5):
        self.num_qubits = 1
        self.name = 'SingleQNN'
        self.num_layers = num_layers
        self.params = self.initialize_parameters()
        self.dm_labels = self.create_dm_labels()

        self.trained = False

        self.trained = False

        dev = qml.device("default.qubit", wires=self.num_qubits)

        @qml.qnode(dev)
        def qnn(params, x, y):
            for p in params:
                qml.Rot(*x, omega=0, wires=0)
                qml.Rot(*p, wires=0)
            return qml.expval(qml.Hermitian(y, wires=[0]))

        self.qnn = qnn

    def create_dm_labels(self):
        qubits = self.num_qubits
        matrix1 = np.kron([[1, 0], [0, 0]], np.eye(2 ** (qubits - 1)))
        matrix2 = np.kron([[0, 0], [0, 1]], np.eye(2 ** (qubits - 1)))

        return np.array([matrix1, matrix2], requires_grad=False)

    def initialize_parameters(self):
        return np.random.uniform(size=(self.num_layers, 3), requires_grad=True)

    def cost(self, params, x, y):
        loss = 0.0
        for i in range(len(x)):
            f = self.qnn(params, x[i], self.dm_labels[int(y[i])])
            loss += (1 - f) ** 2
        return loss / len(x)

    def test(self, x):
        fidelity_values = []
        predicted = []

        for i in range(len(x)):
            fidel_function = lambda y: self.qnn(self.params, x[i], y)
            fidelities = [fidel_function(dm) for dm in self.dm_labels]
            best_fidel = np.argmax(fidelities)

            predicted.append(best_fidel)
            fidelity_values.append(fidelities)

        return np.array(predicted), np.array(fidelity_values)

    def train(self, data_points_train, data_labels_train, data_points_test=None, data_labels_test=None,
              n_epochs: int = 3, batch_size: int = 24,
              optimizer:str='adam',
              opt_params: dict = {},
              silent: bool = False):

        opt = get_optimizer(optimizer, opt_params)

        stats = []
        for it in range(n_epochs):
            for Xbatch, ybatch in iterate_minibatches(data_points_train, data_labels_train,
                                                      batch_size=batch_size):  # AFTER INTRODUCING A BATCH, PARAMETERS ARE UPDATED
                self.params, _, _ = opt.step(self.cost, self.params, Xbatch, ybatch)

            predicted_train, fidel_train = self.test(data_points_train)

            accuracy_train = accuracy_score(data_labels_train, predicted_train)
            loss = self.cost(self.params, data_points_train, data_labels_train)
            if data_labels_test is not None and data_labels_train is not None:
                predicted_test, fidel_test = self.test(data_points_test)
                accuracy_test = accuracy_score(data_labels_test, predicted_test)
            else:
                accuracy_test = None
            res = [it + 1, loss, accuracy_train, accuracy_test]
            stats.append(res)
            if not silent:
                print("Epoch: {:2d} | Loss: {:3f} | Train accuracy: {:3f} | Test accuracy: {:3f}".format(*res))

        df = pd.DataFrame(stats)
        df.columns = ['epoch', 'loss', 'train_accuracy', 'test_accuracy']
        df.set_index('epoch')

        self.trained = True
        return df


    def predict(self, x):
        predicted = []
        for i in range(len(x)):
            fidel_function = lambda y: self.qnn(self.params, x[i], y)
            fidelities = [fidel_function(dm) for dm in self.dm_labels]
            best_fidel = np.argmax(fidelities)
            predicted.append(best_fidel)
        return np.array(predicted)
