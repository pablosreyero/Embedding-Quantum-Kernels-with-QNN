from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
import numpy as np
import os
from pennylane import numpy as np_p
from icecream import ic


### OPTIMIZERS


DEFAULT_ADAM_PARAMS = {'learning_rate': 0.05,
                       'beta1': 0.9,
                       'beta2': 0.999}

def get_optimizer(optimizer='adam', params=DEFAULT_ADAM_PARAMS):
    if optimizer == 'adam':
        learning_rate = params.get('learning_rate', DEFAULT_ADAM_PARAMS['learning_rate'])
        beta1 = params.get('beta1', DEFAULT_ADAM_PARAMS['beta1'])
        beta2 = params.get('beta2', DEFAULT_ADAM_PARAMS['beta2'])
        opt = AdamOptimizer(stepsize=learning_rate, beta1=beta1, beta2=beta2)
    elif optimizer == 'gd' or optimizer == 'gradient':
        learning_rate = params.get('learning_rate', DEFAULT_ADAM_PARAMS['learning_rate'])
        opt = GradientDescentOptimizer(stepsize=learning_rate)
    else:
        raise ValueError('Unknown optimizer. Available options: [gradient, adam]')
    return opt

#### PROCESS POINTS
def increase_dimensions(dataset):
    if dataset.shape[1] == 3:
        return dataset
    dataset_ = np_p.zeros((dataset.shape[0], 3), requires_grad=False)
    dataset_[:, :2] = dataset
    return dataset_


#### OPTIMIZATION

def iterate_minibatches(inputs, targets, batch_size):
    """
    A generator for batches of the input data

    Args:
        inputs (array[float]): input data
        targets (array[float]): targets

    Returns:
        inputs (array[float]): one batch of input data of length `batch_size`
        targets (array[float]): one batch of targets of length `batch_size`
    """
    for start_idx in range(0, inputs.shape[0] - batch_size + 1, batch_size):
        idxs = slice(start_idx, start_idx + batch_size)
        yield inputs[idxs], targets[idxs]


### CHECK PREDICTIONS

def accuracy_score(y_true, y_pred):
    """Accuracy score.

    Args:
        y_true (array[float]): 1-d array of targets
        y_pred (array[float]): 1-d array of predictions

    Returns:
        score (float): the fraction of correctly classified samples
    """
    score = y_true == y_pred #Cuando predicción y true coinciden añado 1 a la suma
    return score.sum() / len(y_true)


def get_accuracy(actual:np.ndarray, expected:np.ndarray):
    return (actual == expected).sum() / len(actual)


### SAVE TO FILE
def save_array_to_csv(arr, filename):
    if filename[:-4] != '.csv':
        filename = filename + '.csv'
    create_path_if_missing(filename)
    np.savetxt(filename, arr, delimiter=',')

def create_path_if_missing(path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

### RANDOM

def print_gray(text):
    # ANSI escape code for gray text
    gray_code = "\033[90m"
    reset_code = "\033[0m"
    # Print the text in gray
    print(f"{gray_code}{text}{reset_code}")


def get_function(method_name, Class):
    # Get all attributes (including methods) of the class
    attributes = vars(Class)

    # Check if the method exists
    if method_name in attributes:
        func = getattr(Class, method_name)
        return func
    # If method not found or not static, return None
    raise ModuleNotFoundError('Method {} not found in class {}'.format(method_name, Class))