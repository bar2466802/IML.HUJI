#################################################################
# FILE : cross_validate.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 5
# DESCRIPTION: K-fold cross-validation
#################################################################

from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    x_folds = np.array_split(X, cv)
    y_folds = np.array_split(y, cv)

    train_scores = []  # to check if we needed it or not
    validation_scores = []
    for k, (x_fold, y_fold) in enumerate(zip(x_folds, y_folds)):
        data = np.stack(np.delete(x_folds, k))
        labels = np.stack(np.delete(y_folds, k))
        estimator.fit(data, labels)
        score = scoring(y_fold, estimator.predict(x_fold))
        validation_scores.append(score)

    validation_scores = np.array(validation_scores)
    return np.float(np.mean(validation_scores)), np.float(np.std(validation_scores))


