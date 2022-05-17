#################################################################
# FILE : perceptron.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 3
# DESCRIPTION: Perceptron class - fit a halfspace classifier according to the algorithm's pseudocode as seen in class
#################################################################

from __future__ import annotations
from typing import Callable
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


def default_callback(fit: Perceptron, x: np.ndarray, y: int):
    pass


class Perceptron(BaseEstimator):
    """
    Perceptron half-space classifier

    Finds a separating hyperplane for given linearly separable data.

    Attributes
    ----------
    include_intercept: bool, default = True
        Should fitted model include an intercept or not

    max_iter_: int, default = 1000
        Maximum number of passes over training data

    coefs_: ndarray of shape (n_features,) or (n_features+1,)
        Coefficients vector fitted by Perceptron algorithm. To be set in
        `Perceptron.fit` function.

    callback_: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
    """

    def __init__(self,
                 include_intercept: bool = True,
                 max_iter: int = 1000,
                 callback: Callable[[Perceptron, np.ndarray, int], None] = default_callback):
        """
        Instantiate a Perceptron classifier

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        max_iter: int, default = 1000
            Maximum number of passes over training data

        callback: Callable[[Perceptron, np.ndarray, int], None]
            A callable to be called after each update of the model while fitting to given data
            Callable function should receive as input a Perceptron instance, current sample and current response
        """
        super().__init__()
        self.include_intercept_ = include_intercept
        self.max_iter_ = max_iter
        self.callback_ = callback
        self.coefs_ = None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit a halfspace to to given samples. Iterate over given data as long as there exists a sample misclassified
        or that did not reach `self.max_iter_`

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.fit_intercept_`
        """

        x_copy = np.copy(X)
        if self.include_intercept_:
            x_copy = np.c_[np.ones(len(X)), X]

        self.coefs_ = np.zeros(x_copy.shape[1])
        # From Gilad's post in the forum:
        # "You can therefore change `fitted_` to `True` in the `Perceptron.fit` after changing w for the first time"
        self.fitted_ = True
        diff = np.sign(x_copy @ self.coefs_) - y  # if the true value is 0 then * won't work sue I used - instead
        index = 0

        # Iterate over given data as long as that did not reach `self.max_iter_`
        # and as long as there exists a sample misclassified
        while index < self.max_iter_ and diff.any():
            nonzero = np.where(diff != 0)
            # The first index is necessary because the vector is within a tuple
            last_non_zero_index = nonzero[0][-1]
            self.coefs_ += (x_copy[last_non_zero_index] * y[last_non_zero_index])
            self.callback_(self, x_copy, y[last_non_zero_index])
            diff = np.sign(x_copy @ self.coefs_) - y
            index += 1
        print("number of iterations: " + str(index))

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        x_copy = np.copy(X)
        if self.include_intercept_:
            x_copy = np.c_[np.ones(len(X)), X]

        return np.sign(x_copy @ self.coefs_)

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
