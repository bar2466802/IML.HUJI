#################################################################
# FILE : linear_discriminant_analysis.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 3
# DESCRIPTION: LDA class
#################################################################

from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv
import pandas as pd


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        # x_pd = pd.DataFrame(X)
        # y_pd = pd.DataFrame(y)
        # self.pi_ = np.array(y_pd.value_counts(normalize=True))
        # self.mu_ = np.array(x_pd.groupby(by=y).mean())
        self.pi_ = np.zeros(shape=(len(self.classes_)))
        self.mu_ = np.zeros(shape=(len(self.classes_), X.shape[1]))
        self.cov_ = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(self.classes_):
            x_i = X[y == group]
            # mu_i = self.mu_[idx]
            mu_i = np.mean(x_i, axis=0)
            self.mu_[idx] = mu_i
            pi_i = (y == group).sum() / len(y)
            self.pi_[idx] = pi_i
            matrix = x_i - mu_i
            self.cov_ += matrix.T @ matrix
        self.cov_ /= len(y)
        self._cov_inv = np.linalg.inv(self.cov_)

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
        return self.classes_[np.argmax(self.likelihood(X), axis=1)]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")
        likelihood = np.zeros(shape=(X.shape[0], len(self.classes_)))
        for i, x in enumerate(X):
            likelihood_i = np.zeros(shape=len(self.classes_))
            for idx in range(len(self.classes_)):
                a_k = self._cov_inv @ self.mu_[idx]
                b_k = np.log(self.pi_[idx]) - 0.5 * (self.mu_[idx] @ self._cov_inv @ self.mu_[idx].T)
                likelihood_i[idx] = a_k.T @ x + b_k
            likelihood[i] = likelihood_i
        return likelihood

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
