import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, NoReturn

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from plotly.subplots import make_subplots
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path - {title}"))


def get_gd_state_recorder_callback(module_type: Type[BaseModule], init: np.ndarray, etas: Tuple[float]) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    value, values = [], []
    descent_paths, descent_path = [], []

    def gd_callback(solver: GradientDescent, weights: np.ndarray, val: np.ndarray, grad: np.ndarray, t: int,
                    eta: float, delta: float) -> NoReturn:
        """
            Plot the descent path of the gradient descent algorithm

            Parameters:
            -----------
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)
        """
        descent_path.append(weights)
        value.append(val)

    for rate in etas:
        descent_path, value = [], []
        module = module_type(weights=init)
        fixed_lr = FixedLR(base_lr=rate)
        gd = GradientDescent(learning_rate=fixed_lr, callback=gd_callback)
        gd.fit(X=None, y=None, f=module)
        descent_paths.append(descent_path)
        values.append(value)

    return np.array(descent_paths), np.array(values)


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [L1, L2]
    modules_titles = ["L1", "L2"]
    titles = []
    for i, t in enumerate(modules_titles):
        module_title_arr = []
        for j, eta in enumerate(etas):
            title = "Module: " + t + ", eta: " + str(eta) + ", init: " + str(init)
            module_title_arr.append(title)
        titles.append(module_title_arr)
    titles = np.array(titles)
    for i, module in enumerate(modules):
        descent_paths, values = get_gd_state_recorder_callback(module_type=module, init=init, etas=etas)
        for j, path in enumerate(descent_paths):
            title = titles[i][j]
            plot_descent_path(module=module, descent_path=path, title=title).show()
        # figs = np.array(figs)
        # fig = make_subplots(rows=int(len(etas) / 2), cols=int(len(etas) / 2))
        # rng = range(len(etas))
        # fig.add_traces(figs, rows=np.repeat(rng, len(etas)), cols=np.tile(rng, len(etas)))
        # fig.show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    raise NotImplementedError()

    # Plot algorithm's convergence for the different values of gamma
    raise NotImplementedError()

    # Plot descent path for gamma=0.95
    raise NotImplementedError()


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
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
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    # fit_logistic_regression()
    print("fin ex6!")
