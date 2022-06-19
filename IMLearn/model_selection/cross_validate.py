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
    ids = np.arange(X.shape[0])

    # Randomly split samples into `cv` folds
    folds = np.array_split(ids, cv)

    train_score, validation_score = .0, .0
    for fold_ids in folds:
        train_msk = ~np.isin(ids, fold_ids)
        fit = deepcopy(estimator).fit(X[train_msk], y[train_msk])

        train_score += scoring(y[train_msk], fit.predict(X[train_msk]))
        validation_score += scoring(y[fold_ids], fit.predict(X[fold_ids]))

    return train_score / cv, validation_score / cv
    # x_folds = np.array_split(X, cv)
    # y_folds = np.array_split(y, cv)
    #
    # train_scores = []  # to check if we needed it or not
    # validation_scores = []
    # for k, (x_fold, y_fold) in enumerate(zip(x_folds, y_folds)):
    #     x_folds_copy = deepcopy(x_folds)
    #     y_folds_copy = deepcopy(y_folds)
    #     x_folds_copy.pop(k)
    #     data = np.concatenate(x_folds_copy)
    #     y_folds_copy.pop(k)
    #     labels = np.concatenate(y_folds_copy)
    #     estimator.fit(data, labels)
    #     # calc train score
    #     score = scoring(estimator.predict(data), labels)
    #     train_scores.append(score)
    #     # calc validation score
    #     score = scoring(estimator.predict(x_fold), y_fold)
    #     validation_scores.append(score)
    #
    # train_scores = np.array(train_scores)
    # validation_scores = np.array(validation_scores)
    # return np.float(np.mean(train_scores)), np.float(np.mean(validation_scores))


