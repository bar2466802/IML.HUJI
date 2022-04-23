#################################################################
# FILE : loss_functions.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 2
# DESCRIPTION: Implement MSE function for models
#################################################################

import numpy as np


def mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate MSE loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    MSE of given predictions
    """
    return np.float(np.mean(np.power((y_true - y_pred), 2)))


def misclassification_error(y_true: np.ndarray, y_pred: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    # y_diff = y_true * y_pred
    # error = len(y_diff[y_diff < 0])
    error = sum(1 for y1, y2 in zip(y_true, y_pred) if y1 != y2)
    if normalize:
        error = error / len(y_true)
    return error


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Accuracy of given predictions
    """
    y_diff = y_true + y_pred
    positive_count = len(y_true[y_true == 1])
    negative_count = len(y_true[y_true == -1])
    true_positive_count = len(y_diff[y_diff == 2])
    true_negative_count = len(y_diff[y_diff == -2])
    return (true_positive_count + true_negative_count) / (positive_count + negative_count)


def cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the cross entropy of given predictions

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values

    Returns
    -------
    Cross entropy of given predictions
    """
    raise NotImplementedError()
