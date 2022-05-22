#################################################################
# FILE : perform_model_selection.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 5
# DESCRIPTION: Compare Ridge, Lasso and Linear Regression models
#################################################################

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn import BaseEstimator
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
import matplotlib.pyplot as plt
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    response = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    X = np.linspace(-1.2, 2, n_samples)
    y = pd.Series(response(X))
    eps = np.random.normal(0, noise, n_samples)
    y_ = pd.Series(y + eps)
    X = pd.DataFrame(X)
    # Split X and noisy y to train and test
    train_X, train_y, test_X, test_Y = split_train_test(X, y_, 2 / 3)

    plt.title("Scatter plot of noiseless, train and test sets \n n_samples = " + str(n_samples)
              + " , noise = " + str(noise))
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(train_X, train_y, c='b', marker='x', label='Train')
    plt.scatter(test_X, test_Y, c='r', marker='.', label='Test')
    plt.scatter(X, y, c='pink', marker='*', label='Noiseless')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    plt.cla()
    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    scores = []
    for k in range(11):
        poly_model = PolynomialFitting(k)
        train_score, validation_score = cross_validate(estimator=poly_model, X=train_X.to_numpy(), y=train_y.to_numpy(),
                                                       scoring=mean_square_error)
        scores.append({'train_score': train_score, 'validation_score': validation_score, 'k': k})

    scores_df = pd.DataFrame(scores)
    plt.title("Average training and validation errors Vs. polynomial degree \n n_samples = " + str(n_samples)
              + " , noise = " + str(noise))
    plt.xlabel("k values (polynomial degree)")
    plt.ylabel("Average errors")
    plt.plot(scores_df['k'], scores_df['train_score'], c='b', label='Average train error')
    plt.plot(scores_df['k'], scores_df['validation_score'], c='r', label='Average validation error')
    plt.legend(loc='upper left')
    plt.show()
    plt.clf()
    plt.cla()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_model = scores_df[scores_df['validation_score'] == scores_df['validation_score'].min()]
    best_k = int(best_model['k'])
    print('Best k-degree is: ' + str(best_k))
    best_poly_model = PolynomialFitting(best_k)
    best_poly_model.fit(train_X.to_numpy(), train_y.to_numpy())
    y_pred = best_poly_model.predict(test_X.to_numpy())
    test_error = mean_square_error(test_Y.to_numpy(), y_pred)
    # form = "{0:.3g}"
    form = "{0:.2f}"
    print('Test error over entire train set with k-degree = ' + str(best_k) + ': ' + str(form.format(test_error)))
    print('Validation error previously achieved with 5-fold cross-validation: ' + str(
        form.format(float(best_model['validation_score']))))


def add_subplot(fig, df, row=1, col=1):
    fig.append_trace(go.Scatter(x=df['k'], y=df['validation_score'], mode="lines",
                                name='lasso validation score'), row=row, col=col)
    fig.append_trace(go.Scatter(x=df['k'], y=df['train_score'], mode="lines",
                                name='lasso train score'), row=row, col=col)
    fig.update_xaxes(title_text="k", row=row, col=col)
    fig.update_yaxes(title_text="Score", row=row, col=col)


# def compare_models(X: np.ndarray, y: np.ndarray, k_range: np.ndarray, n_samples: int = 50,
#                    k_fold: bool = False, fig_title: str = ""):
#     scores_lasso, scores_ridge = [], []
#     for k in k_range:
#         # lasso
#         lasso = Lasso(alpha=k)
#         # ridge
#         ridge = RidgeRegression(lam=k)
#         if k_fold:
#             lasso_train, lasso_validation = cross_validate(estimator=lasso, X=X.to_numpy(), y=y.to_numpy(),
#                                                            scoring=mean_square_error)
#             ridge_train, ridge_validation = cross_validate(estimator=ridge, X=X.to_numpy(), y=y.to_numpy(),
#                                                            scoring=mean_square_error)
#         else:
#             train_proportion = n_samples / len(X)
#             train_X, train_y, test_X, test_Y = split_train_test(X, y, train_proportion)
#             # Get train and validation scores for lasso
#             lasso.fit(train_X, train_y)
#             lasso_train = mean_square_error(lasso.predict(train_X), train_y)
#             lasso_validation = mean_square_error(lasso.predict(test_X), test_Y)
#             # Get train and validation scores for ridge
#             ridge.fit(train_X, train_y)
#             ridge_train = mean_square_error(ridge.predict(train_X), train_y)
#             ridge_validation = mean_square_error(ridge.predict(test_X), test_Y)
#
#         scores_lasso.append({'train_score': lasso_train, 'validation_score': lasso_validation, 'k': k})
#         scores_ridge.append({'train_score': ridge_train, 'validation_score': ridge_validation, 'k': k})
#
#     lasso_df = pd.DataFrame(scores_lasso)
#     ridge_df = pd.DataFrame(scores_ridge)
#     titles = ["Lasso", "Ridge"]
#     fig = make_subplots(subplot_titles=titles, rows=1, cols=2, horizontal_spacing=0.05, vertical_spacing=.09)
#     add_subplot(fig, lasso_df)
#     add_subplot(fig, ridge_df, row=1, col=2)
#     fig.update_layout(title=fig_title, title_pad_b=100, title_pad_l=15, margin=dict(b=40))
#     fig.show()


def test_model(fig, model_name, X: np.ndarray, y: np.ndarray, k_range: np.ndarray, n_samples: int = 50,
               k_fold: bool = False, row: int = 1, col: int = 1) -> pd.DataFrame:
    scores = []
    for k in k_range:
        if model_name == "Lasso":
            # lasso
            estimator = Lasso(alpha=k)
        else:
            # ridge
            estimator = RidgeRegression(lam=k)

        if k_fold:
            train_score, validation_score = cross_validate(estimator=estimator, X=X.to_numpy(), y=y.to_numpy(),
                                                           scoring=mean_square_error)
        else:
            train_proportion = n_samples / len(X)
            train_X, train_y, test_X, test_Y = split_train_test(X, y, train_proportion)
            # Get train and validation scores
            estimator.fit(train_X, train_y)
            train_score = mean_square_error(estimator.predict(train_X), train_y.to_numpy())
            validation_score = mean_square_error(estimator.predict(test_X), test_Y.to_numpy())

        scores.append({'train_score': train_score, 'validation_score': validation_score, 'k': k})

    df = pd.DataFrame(scores)
    add_subplot(fig, df, row, col)
    return df


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    data, labels = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_proportion = n_samples / len(data)
    train_X, train_y, test_X, test_Y = split_train_test(data, labels, train_proportion)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    # Check what is the best range for the Hyperparameters
    titles = ["Lasso", "Ridge"]
    fig = make_subplots(subplot_titles=titles, rows=1, cols=2, horizontal_spacing=0.05, vertical_spacing=.09)
    title = "CV for different values of the regularization parameter for Ridge and Lasso regressions"
    test_model(fig=fig, model_name=titles[0], X=data, y=labels, k_range=np.linspace(1e-5, 5, n_evaluations),
               n_samples=n_samples, k_fold=True)
    test_model(fig=fig, model_name=titles[1], X=data, y=labels, k_range=np.linspace(1e-5, 50, n_evaluations),
               n_samples=n_samples, k_fold=True, row=1, col=2)
    fig.update_layout(title=title, title_pad_b=100, title_pad_l=15, margin=dict(b=40))
    fig.show()
    # Test the best range of the Hyperparameters
    title = "Train and validation errors as a function of the tested regularization parameter value"
    fig = make_subplots(subplot_titles=titles, rows=1, cols=2, horizontal_spacing=0.05, vertical_spacing=.09)
    lasso_k_rng = np.linspace(1e-5, 2.5, n_evaluations)
    lasso_df = test_model(fig=fig, model_name=titles[0], X=data, y=labels, k_range=lasso_k_rng, n_samples=n_samples,
                          k_fold=False)
    ridge_k_rng = np.linspace(1e-5, 30, n_evaluations)
    ridge_df = test_model(fig=fig, model_name=titles[1], X=data, y=labels, k_range=ridge_k_rng, n_samples=n_samples,
                          k_fold=False, row=1, col=2)
    fig.update_layout(title=title, title_pad_b=100, title_pad_l=15, margin=dict(b=40))
    fig.show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    print("Answers for Question 8:")
    best_lam = lasso_k_rng[np.argmin(lasso_df['validation_score'])]
    ridge = RidgeRegression(lam=best_lam)
    ridge.fit(train_X.to_numpy(), train_y.to_numpy())
    test_score = mean_square_error(ridge.predict(test_X.to_numpy()), test_Y.to_numpy())
    form = "{0:.2f}"
    print("Test error score of Ridge model: " + str(form.format(test_score)))

    best_alpha = ridge_k_rng[np.argmin(ridge_df['validation_score'])]
    lasso = Lasso(alpha=best_alpha)
    lasso.fit(train_X, train_y)
    test_score = mean_square_error(lasso.predict(test_X), test_Y.to_numpy())
    print("Test error score of Lasso model: " + str(form.format(test_score)))

    linear_reg = LinearRegression()
    linear_reg.fit(train_X.to_numpy(), train_y.to_numpy())
    test_score = mean_square_error(linear_reg.predict(test_X.to_numpy()), test_Y.to_numpy())
    print("Test error score of Least Squares model: " + str(form.format(test_score)))


if __name__ == '__main__':
    np.random.seed(0)
    # Part 1 - Cross Validation For Selecting Polynomial Degree
    print("Question 3 Answer, n_samples = 100, noise = 5:")
    select_polynomial_degree(n_samples=100, noise=5)
    print("")
    # Question 4 - Repeat the questions above but using a noise level of 0
    print("Question 4 Answer, n_samples = 100, noise = 0:")
    select_polynomial_degree(n_samples=100, noise=0)
    print("")
    # Question 5 - Repeat the questions above while generating m = 1500 samples using a noise level of 10
    print("Question 5 Answer, n_samples = 1500, noise = 10:")
    select_polynomial_degree(n_samples=1500, noise=10)
    print("")

    # Part 2 - Choosing Regularization Parameters Using Cross Validation
    select_regularization_parameter()
