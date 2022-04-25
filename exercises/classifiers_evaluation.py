#################################################################
# FILE : classifiers_evaluation.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 3
# DESCRIPTION: Test Perceptron class
#################################################################
import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets


    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    data, labels,  losses = [], [], []

    def callback_func(fit: Perceptron, x: np.ndarray, y: int):
        loss = fit.loss(data, np.array(labels))
        losses.append(loss)
    prev_path = "../datasets/"
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        data, labels = load_dataset(prev_path + f)
        # Fit Perceptron and record loss in each fit iteration
        perceptron = Perceptron(callback=callback_func)
        perceptron.fit(data, labels)

        # Plot figure of loss as function of fitting iteration
        name = n + " - Loss vs Training Iterations"
        # fig = px.line(x=range(len(losses)), y=losses, title=name)
        # fig.show()
        plt.title(name)
        plt.xlabel("Training Iterations")
        plt.ylabel("Training Loss")
        plt.plot(range(len(losses)), losses, color="green")
        plot_name = name + ".png"
        plt.savefig(plot_name)
        plt.show()
        plt.clf()
        plt.cla()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    prev_path = "../datasets/"
    models = [LDA(), GaussianNaiveBayes()]
    models_names = ["LDA", "Gaussian Naive Bayes"]
    symbols = np.array(["circle", "x"])
    plots = []

    fig = make_subplots(rows=1, cols=2, subplot_titles=[rf"$\textbf{{{m}}}$" for m in models_names],
                        horizontal_spacing=0.01, vertical_spacing=.03)

    for i, (f, model) in enumerate(zip(["gaussian1.npy", "gaussian2.npy"], models)):
        # Load dataset
        data, labels = load_dataset(prev_path + f)
        lims = np.array([data.min(axis=0), data.max(axis=0)]).T + np.array([-.4, .4])
        title = "Dataset: " + f
        # Fit models and predict over training set
        # model.fit(data, labels)
        # model.predict(data)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig.add_traces([decision_surface(model.fit(data, labels).predict, lims[0], lims[1], showscale=False),
                        go.Scatter(x=data[:, 0], y=data[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=labels, symbol=symbols[labels], colorscale=[custom[0], custom[-1]],
                                               line=dict(color="black", width=1)))],
                       rows=(i // 3) + 1, cols=(i % 3) + 1)

        fig.update_layout(title=rf"$\textbf{{(2) Decision Boundaries Of Models - {title} Dataset}}$",
                          margin=dict(t=100)) \
            .update_xaxes(visible=False).update_yaxes(visible=False)

        # Add traces for data-points setting symbols and colors
        # raise NotImplementedError()

        # Add `X` dots specifying fitted Gaussians' means
        # raise NotImplementedError()

        # Add ellipses depicting the covariances of the fitted Gaussians
        # raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    print("Ex3: This is the end!")
