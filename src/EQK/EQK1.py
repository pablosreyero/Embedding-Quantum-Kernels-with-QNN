import pennylane as qml
from src.dataReuploading import MultiQNN
from icecream import ic
from sklearn.svm import SVC
from pennylane import numpy as np


class EQK1:
    """
    Embedding Quantum Kernel (EQK) relying on in a Single Quantum Neural Network (SingleQNN).
    Note that I will construct the SingleQNN from MultiQNN because it is more generalized and then I can modify
    everything more easily
    """

    def __init__(self, qnn: MultiQNN, num_qubits: int = 4, entanglement_mode: str = 'cnot'):

        # Check arguments
        if not qnn.trained:
            raise ValueError("qnn must be trained beforehand")
        if not qnn.num_qubits == 1:
            raise ValueError("qnn must be a SingleQNN")
        if entanglement_mode not in ['cnot', 'cz', 'strongly']:
            raise ValueError("entanglement_mode must be 'cnot', 'cz' or 'strongly'")

        # Save variables
        self.name = "EQKn"
        self.qnn = qnn  # Already trained Quantum Neural Networkç

        self.num_qubits = num_qubits
        self.entanglement_mode = entanglement_mode

        self.trained = False
        self.svm = None  # Variable to store support vector machine
        self.trained = False
        self.params = None
        self.train_points = None

    def _forward_qnn_with_entanglement(self, point: np.ndarray):

        for layer in range(self.qnn.num_layers):
            for qubit in range(self.num_qubits):
                qml.Rot(*point, omega=0, wires=qubit)
                qml.Rot(*self.qnn.params[0][layer, :], wires=qubit)
            if self.entanglement_mode == 'cnot':
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            if self.entanglement_mode == 'cz':
                for qubit in range(self.num_qubits - 1):
                    qml.CZ(wires=[qubit, qubit + 1])
            if self.entanglement_mode == 'strongly':
                shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=self.num_qubits)
                weights = np.random.random(size=shape)
                qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))

    def _kernel_function(self, x1, x2):
        """
        Given two points, it returns the kernel value
        """


        inverse_circuit = qml.adjoint(self._forward_qnn_with_entanglement)

        @qml.qnode(self.qnn.dev)
        def kernel_circuit(point_i, point_j):
            self._forward_qnn_with_entanglement(point_j)
            inverse_circuit(point_i)
            return qml.probs(wires=range(self.qnn.num_qubits))  # expected value of all zeros
        return kernel_circuit(x1, x2)[0].item()

    def _construct_train_kernel_matrix(self, data_set):
        kernel_matrix = qml.kernels.square_kernel_matrix(data_set, self._kernel_function, assume_normalized_kernel=True)
        return kernel_matrix

    def _construct_kernel_matrix(self, data_set1, data_set2):
        kernel_matrix = qml.kernels.kernel_matrix(data_set1, data_set2, self._kernel_function)
        return kernel_matrix

    def train(self, train_set, label_train, test_set=None, label_test=None, silent:bool=True):

        self.svm = SVC(kernel='precomputed')
        if not silent:
            print('[INFO] EQK creating kernel_matrix to train')
        kernel_matrix = self._construct_train_kernel_matrix(train_set)
        if not silent:
            print('[INFO] Finished!')
            print('[INFO] Fitting model to kernel_matrix')
        self.svm.fit(kernel_matrix, label_train)
        self.trained = True
        self.train_points = train_set
        if not silent:
            print('[INFO] Finished!')

        if test_set is not None and label_test is not None:
            if not silent:
                print('[INFO] Evaluating model on test_set')
                print('[INFO] Creating kernel_matrix training vs test points')
            k_test = self._construct_kernel_matrix(train_set, test_set)
            if not silent:
                print('[INFO] Finished')
                print('[INFO] Predicting points')
            predictions = self.svm.predict(k_test)

            percentage = 0
            for expected, actual in zip(label_test, predictions):
                if expected == actual:
                    percentage += 1
            percentage /= len(predictions)
            print("[INFO] Testing finished!")
            print(f"[INFO] Accuracy: {percentage}")


    def predict(self, data_set):
        if self.trained == False:
            raise ValueError("EQK must be trained beforehand")

        kernel_matrix = self._construct_kernel_matrix(data_set, self.train_points)
        predictions = self.svm.predict(kernel_matrix)
        return predictions

    def __str__(self):
        class_name = self.__class__.__name__
        qnn_name = str(self.qnn)
        return f'1-to-n Embedding Quantum Kernel {class_name} using 1-QNN: {qnn_name}\nTrained: {"yes" if self.trained else "no"}'
