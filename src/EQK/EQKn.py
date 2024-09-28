import pennylane as qml
from src.dataReuploading import MultiQNN
from icecream import ic
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor
from pennylane import numpy as np
from pandas import DataFrame
from ..utils import increase_dimensions


class EQKn:

    def __init__(self, qnn: MultiQNN):
        """
        Embedding Quantum Kernel (EQK) relying in a Multiqubit Quantum Neural Network (MultiQNN).
        """

        if not qnn.trained:
            raise ValueError("qnn must be trained beforehand")

        self.name = "EQKn"
        self.qnn = qnn  # Already trained Quantum Neural Network

        self.trained = False
        self.svm = None  # Variable to store support vector machine
        self.trained = False
        self.params = None
        self.train_points = None

    def git_kernel_function(self, x1, x2):
        """
        Given two points, it returns the kernel value
        """

        inverse_circuit = qml.adjoint(self.qnn._base_circuit)

        @qml.qnode(self.qnn.dev)
        def kernel_circuit(point_i, point_j):
            self.qnn._base_circuit(point_j)
            inverse_circuit(point_i)
            return qml.probs(wires=range(self.qnn.num_qubits))  # expected value of all zeros

        return kernel_circuit(x1, x2)[0].item()

    def _compute_row_kernel(self, data_set, row_idx):
        n_samples = data_set.shape[0]
        row = np.zeros(n_samples, requires_grad=True)
        for i in range(n_samples):
            if row_idx == i:
                row[i] = 1
            else:
                row[i] = self._kernel_function(data_set[row_idx], data_set[i])
        return row

    def _compute_row_kernel_batch(self, data_set, start_idx, end_idx):
        n_samples = data_set.shape[0]
        kernel_matrix_batch = np.zeros((end_idx - start_idx, n_samples), requires_grad=True)
        for i in range(start_idx, end_idx):
            kernel_matrix_batch[i - start_idx] = np.array(
                [self._kernel_function(data_set[i], data_set[j]) for j in range(n_samples)])
        return kernel_matrix_batch

    def _construct_train_kernel_matrix(self, data_set, parallel:bool=False, n_jobs:int=3, batch_size: int = 10):
        # if parallel:
        #     # Number of processes to use
        #     num_processes = multiprocessing.cpu_count()
        #     if n_jobs > num_processes:
        #         raise ValueError(f'Number jobs can be at most {num_processes}')
        #
        #     # Create a Pool with the specified number of processes
        #     pool = multiprocessing.Pool(processes=n_jobs)
        #     results = [pool.apply_async(self._compute_row_kernel, (data_set, i)) for i in range(len(data_set))]
        #     pool.close()
        #     pool.join()
        #     kernel_matrix = np.array([res.get() for res in results])

        if parallel:
            raise ValueError('Do not set parallel to True. It does not give any advantage')  #TODO: fix parallelization
        n_samples = len(data_set)
        if parallel:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = []
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    futures.append(executor.submit(self._compute_row_kernel_batch, data_set, start_idx, end_idx))
                kernel_matrix_batches = [future.result() for future in futures]
                kernel_matrix = np.concatenate(kernel_matrix_batches, axis=0)
        else:
            kernel_matrix = qml.kernels.square_kernel_matrix(data_set, self._kernel_function, assume_normalized_kernel=True)
        return kernel_matrix

    def _construct_kernel_matrix(self, data_set1, data_set2, parallel:bool=False, n_jobs:int=3):
        kernel_matrix = qml.kernels.kernel_matrix(data_set1, data_set2, self._kernel_function)
        return kernel_matrix

    def train(self, train_set, label_train, test_set=None, label_test=None, silent:bool=True) -> DataFrame:

        self.svm = SVC(kernel='precomputed')
        if not silent:
            print('[INFO] EQK creating kernel_matrix to train')
            
        # Adapts training and test sets in case 2D points
        train_set_ = increase_dimensions(train_set)
        test_set_ = increase_dimensions(test_set)
            
        
        kernel_matrix = self._construct_train_kernel_matrix(train_set_)
        if not silent:
            print('[INFO] Finished!')
            print('[INFO] Fitting model to kernel_matrix')
        self.svm.fit(kernel_matrix, label_train)
        self.trained = True
        self.train_points = train_set_
        
        # Get accuracy of traning set
        predictions = self.svm.predict(kernel_matrix)
        percentage_training = (label_train == predictions).sum() / len(predictions)

        if not silent:
            print(f'[INFO] Finished!\n[INFO] Accuracy of training set: {percentage_training}')


        if test_set is not None and label_test is not None:
            if not silent:
                print('[INFO] Evaluating model on test_set')
                print('[INFO] Creating kernel_matrix training vs test points')
            ic(train_set_.shape, test_set_.shape)
            k_test = self._construct_kernel_matrix(train_set_, test_set_)
            try:
                ic(k_test.shape)
            except:
                pass
            
            
            if not silent:
                print('[INFO] Finished')
                print('[INFO] Predicting points')
            
            
            predictions = self.svm.predict(k_test)
            percentage_test= (label_test == predictions).sum() / len(predictions)
            
            print(f'[INFO] Testing finished!\n[INFO] Accuracy of training set: {percentage_test}')
        else:
            percentage_test = np.nan
            
        return DataFrame(data=[[percentage_training, percentage_test]], columns=['train_accuracy', 'test_accuracy'])
    
    def get_accuracy(self, data_set, label):
        if self.trained == False:
            raise ValueError("EQK must be trained beforehand")

        predictions = self.predict(data_set)
        accuracy = accuracy_score(label, predictions)
        return accuracy


    def predict(self, data_set):
        if self.trained == False:
            raise ValueError("EQK must be trained beforehand")
        kernel_matrix = self._construct_kernel_matrix(data_set, self.train_points)
        predictions = self.svm.predict(kernel_matrix)
        return predictions

    def __str__(self):
        class_name = self.__class__.__name__
        qnn_name = str(self.qnn)
        return f'n-to-n Embedding Quantum Kernel ({class_name}) using n-QNN: {qnn_name}.\nTrained: {"yes" if self.trained else "no"}'
