#################################################################
# FILE : utils.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 2
# DESCRIPTION: utils for IML classes - implement split_train_test for models
#################################################################

from typing import Tuple
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split


def split_train_test(X: pd.DataFrame, y: pd.Series, train_proportion: float = .75) \
        -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Randomly split given sample to a training- and testing sample

    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Data frame of samples and feature values.

    y : Series of shape (n_samples, )
        Responses corresponding samples in data frame.

    train_proportion: Fraction of samples to be split as training set

    Returns
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples

    """
    # Shuffle your dataset
    shuffle_df = X.sample(frac=1)
    y = y.reindex_like(shuffle_df)
    train_size = int(len(shuffle_df) * train_proportion)
    train_X = shuffle_df[:train_size]
    train_y = y[:train_size]
    test_X = shuffle_df[train_size:]
    test_Y = y[train_size:]

    # train_X, test_X, train_y, test_Y  = train_test_split(X, y, test_size=train_proportion, random_state=0)
    return train_X, train_y, test_X, test_Y


def confusion_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Compute a confusion matrix between two sets of integer vectors

    Parameters
    ----------
    a: ndarray of shape (n_samples,)
        First vector of integers

    b: ndarray of shape (n_samples,)
        Second vector of integers

    Returns
    -------
    confusion_matrix: ndarray of shape (a_unique_values, b_unique_values)
        A confusion matrix where the value of the i,j index shows the number of times value `i` was found in vector `a`
        while value `j` vas found in vector `b`
    """
    raise NotImplementedError()
