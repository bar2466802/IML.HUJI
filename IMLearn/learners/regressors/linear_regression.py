from __future__ import annotations
from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import pandas as pd
from numpy.linalg import pinv

from ...metrics import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        # Check if we need to convert to array
        x_arr = X
        if type(X) == pd.DataFrame:
            x_arr = X.to_numpy()

        # If we need to include intercept then add a zero-th coordinate to every sample xi = (1, x1, ... , xd)
        if self.include_intercept_:
            new_column = np.full(x_arr.shape[0], 1)
            np.insert(x_arr, 0, new_column, axis=1)
        moore_penrose_pseudo_inv = np.linalg.pinv(x_arr)
        self.coefs_ = moore_penrose_pseudo_inv.dot(y)

        # from sklearn.linear_model import LinearRegression
        # reg = LinearRegression().fit(X, y)
        # score = reg.score(X, y)

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
        return X @ self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        return mean_square_error(y, self.predict(X))

