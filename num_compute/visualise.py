import matplotlib.pyplot as plt
import numpy as np


def plot_metric_over_time(metric_values, title, ylabel):
    """
    Plots the value of a metric as chunks of data arrive.

    Args:
        metric_values (np.array): 1D array of metric values.
        title (str): Title of the plot.
        ylabel (str): Label of the y-axis.
    """

    # Exception handling
    if not isinstance(metric_values, np.ndarray):
        raise TypeError("metric_values must be a numpy array")
    if np.ndim(metric_values) != 1:
        raise ValueError("metric_values must be a 1D array")
    if len(metric_values) == 0:
        raise ValueError("metric_values must not be empty")

    # Produce plot
    x = np.arange(len(metric_values))
    plt.plot(x, metric_values)
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def compare_models(metric1, metric2, labels, title, ylabel):
    """
    Produces plot that compares the performance of two models
    according to a given metric as chunks of data arrive.

    Args:
        metric1 (np.array): 1D array of metric values for the first model.
        metric2 (np.array): 1D array of metric values for the second model.
        labels (list): List of labels for the two models.
        title (str): Title of the plot.
        ylabel (str): Label of the y-axis.
    """

    # Exception handling
    if not isinstance(metric1, np.ndarray) or not isinstance(metric2, np.ndarray):
        raise TypeError("metric_values must be a numpy array")
    if np.ndim(metric1) != 1 or np.ndim(metric2) != 1:
        raise ValueError("metric_values must be a 1D array")
    if len(metric1) != len(metric2):
        raise ValueError("metric_values must have the same length")
    if len(metric1) == 0 or len(metric2) == 0:
        raise ValueError("metric_values must not be empty")
    if not isinstance(labels, list):
        raise TypeError("labels must be a list")
    if len(labels) != 2:
        raise ValueError("labels must have length 2")


    # Produce plot
    x = np.arange(len(metric1))
    plt.plot(x, metric1, label=labels[0])
    plt.plot(x, metric2, label=labels[1])
    plt.xlabel("Time")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()


def plot_predictions_vs_ground_truth(y_true, y_pred, num_classes):
    """
    Produces a grouped histogram showing the frequency of each class
    in the ground truth and predicted labels.

    Args:
        y_true (np.array): 1D array of ground truth labels.
        y_pred (np.array): 1D array of predicted labels.
        num_classes (int): Number of classes
    """

    # Exception handling
    if not isinstance(y_true, np.ndarray) or not isinstance(y_pred, np.ndarray):
        raise TypeError("y_true and y_pred must be numpy arrays")
    if not isinstance(num_classes, int):
        raise TypeError("num_classes must be an integer")
    if np.ndim(y_true) != 1 or np.ndim(y_pred) != 1:
        raise ValueError("y_true and y_pred must be 1D arrays")
    

    # Produce plot
    bin_edges = np.arange(-0.5, num_classes + 1.5, 1)
    plt.hist([y_true, y_pred], bins = bin_edges, rwidth = 0.8)
    plt.xticks(np.arange(num_classes))
    plt.xlim([-0.75, num_classes - 0.25])
    plt.legend(["Ground Truth", "Predictions"])
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Ground Truth vs Predictions")
    plt.show()