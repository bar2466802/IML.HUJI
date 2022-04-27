from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
import pandas as pd


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
        x_pd = pd.DataFrame(X)
        y_pd = pd.DataFrame(y)
        self.pi_ = np.array(y_pd.value_counts(normalize=True))
        self.mu_ = np.array(x_pd.groupby(by=y).mean())
        # self.vars_ = np.array(x_pd.groupby(by=y).var())
        self.vars_ = np.zeros(shape=(self.classes_.shape[0], X.shape[1]))
        for idx, group in enumerate(self.classes_):
            x_i = X[y == group]
            self.vars_[idx] = np.var(x_i, axis=0)

        # from sklearn.naive_bayes import GaussianNB
        # clf = GaussianNB()
        # clf.fit(X, y)

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
        return np.argmax(self.likelihood(X), axis=1)

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
        for x in X:
            for i in range(len(self.classes_)):
                mean = self.mu_[i]
                var = self.vars_[i]
                log_li = -0.5 * (np.log(2 * np.pi * var)) - 0.5 * ((x - mean) ** 2) / var
                pi_log = np.log(self.pi_)
                likelihood[i] = log_li.sum() + pi_log

        # posteriors = []
        # for i, c in enumerate(self.classes_):
        #     prior = np.log(self.pi_[i])
        #     posterior = np.sum(np.log(self._calculate_likelihood(i, X)))
        #     posterior = prior + posterior
        #     posteriors.append(posterior)
        # posteriors = np.array(posteriors)
        # # return the class with highest posterior probability
        # if np.equal(posteriors, likelihood):
        #     print("yes!")
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
