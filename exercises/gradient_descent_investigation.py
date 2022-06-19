import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type, NoReturn

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from plotly.subplots import make_subplots
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test
from IMLearn.base import BaseModule, BaseLR

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


def get_gd_state_recorder_callback(module_type: Type[BaseModule], init: np.ndarray, etas: Tuple[float],
                                   gammas: Tuple[float] = None) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    weights: List[np.ndarray]
        Recorded parameters
    values: List[np.ndarray]
        Recorded objective values
    achieved_points: List[np.ndarray]
        Values returned from GD
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

    achieved_points = []
    for i, rate in enumerate(etas):
        descent_path, value, = [], []
        module = module_type(weights=init)
        if gammas is None:
            learning_rate = FixedLR(base_lr=rate)
        else:
            gamma = gammas[i]
            learning_rate = ExponentialLR(base_lr=rate, decay_rate=gamma)
        gd = GradientDescent(learning_rate=learning_rate, callback=gd_callback)
        val_achieved = gd.fit(X=None, y=None, f=module)
        descent_paths.append(descent_path)
        values.append(value)
        achieved_points.append(val_achieved)

    return np.array(descent_paths, dtype=object), np.array(values, dtype=object), np.array(achieved_points,
                                                                                           dtype=object)


def print_min(module, module_name, achieved_points, etas, gammas=None):
    form = "{:.3f}"
    print("For module " + module_name)
    # print("points returned from GD = ")
    # print(str(achieved_points))
    print("values computed from returned points from GD = ")
    values_achieved = np.array([module(weights=w).compute_output() for w in achieved_points])
    print(str(values_achieved))
    min_loss = values_achieved.min()
    min_loss_i = int(np.argmin(values_achieved))
    best_eta = etas[min_loss_i]
    result = "Lowest loss achieved = " + form.format(min_loss) + ", with eta = " + str(best_eta)
    if gammas is not None:
        best_gamma = gammas[min_loss_i]
        result += " with gamma = " + str(best_gamma)
    print(result + "\n")


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    modules = [L1, L2]
    modules_titles = ["L1", "L2"]
    """
    3.1.1 Comparing Fixed learning rates - question 1
    Plot the descent trajectory for each of the settings described above
    """
    titles = []
    for i, t in enumerate(modules_titles):
        module_title_arr = []
        for j, eta in enumerate(etas):
            title = "FixedLR Module - " + t + ", eta - " + str(eta)
            module_title_arr.append(title)
        titles.append(module_title_arr)
    titles = np.array(titles)

    for i, module in enumerate(modules):
        descent_paths, values, achieved_points = get_gd_state_recorder_callback(module_type=module, init=init,
                                                                                etas=etas)
        for j in range(len(descent_paths)):
            path = descent_paths[j]
            val = values[j]
            title = titles[i][j]
            fig1 = plot_descent_path(module=module, descent_path=path, title=title)
            # fig1.show()  # TODO uncomment
            fig1.write_image("Gradient Descent Path " + title + ".png", engine='kaleido', format='png')

            """
                3.1.1 Comparing Fixed learning rates - question 3
                For each of the modules, plot the convergence rate (i.e. the norm as a function of the GD
                iteration) for all specified learning rates. Explain your results
            """
            fig2 = go.Figure(go.Scatter(x=np.array(range(len(val))), y=val, mode="markers+lines", marker_color="plum"),
                             layout=go.Layout(title=f"GD Descent Convergence Rate - {title}", xaxis_title="Iteration",
                                              yaxis_title="Loss"))
            fig2.show()  # TODO uncomment
            fig2.write_image("GD Descent Convergence " + title + ".png", engine='kaleido', format='png')

        """
           What is the lowest loss achieved when minimizing each of the modules? Explain the differences
        """
        print_min(module=module, module_name=modules_titles[i], etas=etas, achieved_points=achieved_points)


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    # and Plot algorithm's convergence for the different values of gamma
    """
    3.1.2 Comparing Exponentially Decaying learning rates
    Plot the convergence rate for all decay rates in a single plot. Explain your results
    """
    etas = np.full(shape=len(gammas), fill_value=eta)
    modules = [L1, L2]
    modules_titles = ["L1", "L2"]
    titles = []
    for i, t in enumerate(modules_titles):
        module_title_arr = []
        for j, gamma in enumerate(gammas):
            title = "ExponentialLR Module - " + t + ", eta - " + str(eta) + ", gamma - " + str(gamma)
            module_title_arr.append(title)
        titles.append(module_title_arr)
    titles = np.array(titles)

    for k, module in enumerate(modules):
        descent_paths, values, achieved_points = get_gd_state_recorder_callback(module_type=module, init=init,
                                                                                etas=etas, gammas=gammas)
        title = modules_titles[k] + " - Loss Vs. Iteration - convergence rate for all decay rates"
        fig = go.Figure(layout=go.Layout(title=f"GD Descent Convergence Rate - {title}", xaxis_title="Iteration",
                                         yaxis_title="Loss"))
        plots_colors = ["blue", "red", "green", "pink"]
        for i in range(len(values)):
            value = values[i]
            color = plots_colors[i]
            gamma = gammas[i]
            iterations = np.array(range(len(value)))
            plot = go.Scatter(x=iterations, y=value, mode="markers+lines", marker_color=color, name=str(gamma))
            fig.add_trace(plot)
        # fig.show()  # TODO uncomment
        fig.write_image("Gradient Descent Convergence " + title + ".png", engine='kaleido', format='png', scale=1,
                        width=1550,
                        height=1000)

        # Plot descent path for all gammas
        for j in range(len(descent_paths)):
            path = np.array(descent_paths[j])
            title = titles[k][j]
            fig1 = plot_descent_path(module=module, descent_path=path, title=title)
            # fig1.show()  # TODO uncomment
            fig1.write_image("Gradient Descent Path " + title + ".png", engine='kaleido', format='png', scale=1,
                             width=1550,
                             height=1000)

        print_min(module=module, module_name=modules_titles[k], etas=etas, achieved_points=achieved_points,
                  gammas=gammas)


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


    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    r


if __name__ == '__main__':
    np.random.seed(0)
    print("compare_fixed_learning_rates")
    compare_fixed_learning_rates()
    print("compare_exponential_decay_rates")
    compare_exponential_decay_rates()
    print("fit_logistic_regression")
    fit_logistic_regression()
    print("fin ex6!")
