import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import ptp
from icecream import ic

def plot_predictions_3d(points, predictions, real_labels=None, axis=None, title: str=None):
    assert len(points) == len(predictions)
    if real_labels is not None:
        assert len(points) == len(real_labels)

    if axis is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = axis

    # Plot points with predictions
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=predictions, cmap='viridis', marker='o', label='Predictions')

    # Compare to real labels
    if real_labels is not None:
        # Separate the points based on correct/incorrect predictions
        correct_predictions = (real_labels == predictions)
        incorrect_predictions = ~correct_predictions

        # Highlight points with incorrect predictions
        ax.scatter(points[incorrect_predictions][:, 0], points[incorrect_predictions][:, 1],
                   points[incorrect_predictions][:, 2], color='k', marker='x', s=100, label='Incorrect')

    ax.legend()

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()
    if axis is None:
        plt.show()

def plot_predictions_2d(points, predictions, real_labels=None, axis=None, title:str=None):

    assert len(points) == len(predictions)
    if real_labels is not None:
        assert len(points) == len(real_labels)

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis

    # plot points with predictions
    _plot_dataset_2d(points, predictions, ax, title)

    # Compare to real labels
    if real_labels is not None:
        # Separate the points based on correct/incorrect predictions
        correct_predictions = (real_labels == predictions)
        incorrect_predictions = ~correct_predictions

        # Highlight points with incorrect predictions
        ax.scatter(points[incorrect_predictions][:, 0], points[incorrect_predictions][:, 1],
                   color='k', marker='x', s=100, label='Incorrect')

    # Ensure equal aspect ratio
    ax.set_box_aspect(1)

    ax.legend()

    if title is not None:
        ax.set_title(title)

    # plt.tight_layout()
    if axis is None:
        plt.show()

def plot_dataset(points, labels, fig=None, axis=None, title:str=None, show=False):

    assert len(points) == len(labels)

    if points.shape[1] == 2:
        if axis is None:
            fig, ax = plt.subplots()
        else:
            ax = axis
        _plot_dataset_2d(points, labels, ax, title)

    else:
        if axis is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            if not isinstance(axis, Axes3D):   # TODO: arreglar esta mierda
                ic(fig.__dict__)
                ic(axis.shape)
                num_rows, num_cols = fig.get_subplotspec().get_geometry()
                position = axis.get_subplotspec().num1 + 1
                ax = fig.add_subplot(int(f'{num_rows}{num_cols}{position}'), projection='3d')
            else:
                ax = axis
        _plot_dataset_3d(points, labels, ax, title)

    ax.legend()

    if title is not None:
        ax.set_title(title)

    # plt.tight_layout()
    if axis is None and show:
        plt.show()

def _plot_dataset_2d(points, labels, ax, title):

    # Separate the points based on predicted labels
    points_class0 = points[labels == 0]
    points_class1 = points[labels == 1]

    # Plot points with true label 0
    ax.scatter(points_class0[:, 0], points_class0[:, 1],
               c='r', marker='o', label='Class 0', alpha=0.5)

    # Plot points with true label 1
    ax.scatter(points_class1[:, 0], points_class1[:, 1],
               c='b', marker='^', label='Class 1', alpha=0.5)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')


def _plot_dataset_3d(points, labels, ax, title=None):
    # Separate the points based on predicted labels
    points_class0 = points[labels == 0]
    points_class1 = points[labels == 1]

    # Plot points with true label 0
    ax.scatter(points_class0[:, 0], points_class0[:, 1], points_class0[:, 2],
               c='r', marker='o', label='Class 0', alpha=0.2)

    # Plot points with true label 1
    ax.scatter(points_class1[:, 0], points_class1[:, 1], points_class1[:, 2],
               c='b', marker='^', label='Class 1', alpha=0.5)

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal scaling for axis and adjust box aspect ratio
    ax.set_box_aspect([ptp(points[:, 0]), ptp(points[:, 1]), ptp(points[:, 2])])

