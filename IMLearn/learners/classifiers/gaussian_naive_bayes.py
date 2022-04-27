#################################################################
# FILE : gaussian_naive_bayes.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 3
# DESCRIPTION: GNB class
#################################################################

from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.zeros(shape=(len(self.classes_)))
        self.mu_ = np.zeros(shape=(len(self.classes_), X.shape[1]))
        self.vars_ = np.zeros(shape=(self.classes_.shape[0], X.shape[1]))
        for idx, group in enumerate(self.classes_):
            x_i = X[y == group]
            mu_i = np.mean(x_i, axis=0)
            self.mu_[idx] = mu_i
            pi_i = (y == group).sum() / len(y)
            self.pi_[idx] = pi_i
            self.vars_[idx] = np.var(x_i, axis=0)

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
        for i, key in enumerate(self.classes_):
            mu_i = self.mu_[i]
            var_i = self.vars_[i]
            cov_i = np.diag(var_i)
            inv_cov_i = np.linalg.inv(cov_i)
            pi_i = self.pi_[i]
            d = X[:, np.newaxis, :] - mu_i
            mahalanobis = np.sum(d.dot(inv_cov_i) * d, axis=2).flatten()
            pdf = np.exp(-.5 * mahalanobis) / np.sqrt((2 * np.pi) ** len(X) * np.linalg.det(cov_i)) * pi_i
            likelihood[:, i] = pdf
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

    def _calculate_likelihood(self, class_idx, x):
        mean = self.mu_[class_idx]
        var = self.vars_[class_idx]
        num = np.exp(- (x - mean) ** 2 / (2 * var))
        denom = np.sqrt(2 * np.pi * var)
        return num / denom
