from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
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
    X = np.linspace(-1.2, 2, 100)
    y = pd.Series(response(X))
    eps = np.random.normal(0, noise)
    y_ = pd.Series(y + eps)
    X = pd.DataFrame(X)
    # Split X and noisy y to train and test
    train_X, train_y, test_X, test_Y = split_train_test(X, y_, 2 / 3)

    plt.title("Scatter plot of noiseless, train and test sets \n n_samples = " + str(n_samples)
              + " , noise = " + str(noise))
    plt.xlabel("Y")
    plt.ylabel("X")
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
                                                       scoring=poly_model.loss)
        scores.append({'train_score': train_score, 'validation_score': validation_score, 'k': k})

    scores_df = pd.DataFrame(scores)
    plt.title("Average training and validation errors Vs. polynomial degree \n n_samples = " + str(n_samples)
              + " , noise = " + str(noise))
    plt.xlabel("k values (polynomial degree)")
    plt.ylabel("Average errors")
    plt.scatter(scores_df['k'], scores_df['train_score'], c='b', marker='x', label='Average train error')
    plt.scatter(scores_df['k'], scores_df['validation_score'], c='r', marker='.', label='Average validation error')
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
    test_error = best_poly_model.loss(test_X.to_numpy(), test_Y.to_numpy())
    form = "{:.2f}"
    print('Test error over entire train set: ' + str(form.format(test_error)))
    print('Validation error previously achieved with 5-fold cross-validation: ' + str(
        form.format(float(best_model['validation_score']))))


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
    train_proportion = 50 / len(data)
    train_X, train_y, test_X, test_Y = split_train_test(data, labels, train_proportion)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    scores_lasso = []
    scores_ridge = []
    for k in np.linspace(0, 555, 5000):
        lasso = Lasso(alpha=k)
        train_score, validation_score = cross_validate(estimator=lasso, X=data.to_numpy(), y=labels.to_numpy(),
                                                       scoring=mean_square_error)
        scores_lasso.append({'train_score': train_score, 'validation_score': validation_score, 'k': k})
        ridge = RidgeRegression(lam=k)
        train_score, validation_score = cross_validate(estimator=ridge, X=data.to_numpy(), y=labels.to_numpy(),
                                                       scoring=mean_square_error)
        scores_ridge.append({'train_score': train_score, 'validation_score': validation_score, 'k': k})

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model



if __name__ == '__main__':
    np.random.seed(0)
    # Part 1 - Cross Validation For Selecting Polynomial Degree
    print("Question 3 Answer:")
    print("Question 4 Answer, n_samples = 100, noise = 5:")
    select_polynomial_degree()
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

