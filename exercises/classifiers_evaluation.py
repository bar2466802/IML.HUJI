#################################################################
# FILE : classifiers_evaluation.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 3
# DESCRIPTION: Test classes: Perceptron, LDA and GNB
#################################################################
import numpy as np
import pandas as pd
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
    data, labels, losses = [], [], []

    def callback_func(fit: Perceptron, x: np.ndarray, y: int):
        loss = fit.loss(data, np.array(labels))
        losses.append(loss)

    prev_path = "../datasets/"
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
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

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black", showlegend=False)


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    prev_path = "../datasets/"
    models = [LDA(), GaussianNaiveBayes()]
    models_names = ["LDA", "Gaussian Naive Bayes"]
    datasets = ["gaussian1.npy", "gaussian2.npy"]
    symbols = np.array(["circle", "bowtie", "hexagram", "triangle-up", "pentagon", "star"])
    colors = np.array(["red", "blue", "LightGreen", "pink", "yellow", "purple"])
    from IMLearn.metrics import accuracy
    df_q2 = {
        'model_name': [],
        'dataset_name': [],
        'data': [],
        'labels': [],
        'mu_': [],
        'cov_': [],
        'classes': [],
        'accuracy': [],
        'y_pred': []
    }
    for i, f in enumerate(datasets):
        # Load dataset
        data, labels = load_dataset(prev_path + f)
        for model, model_name in zip(models, models_names):
            model.fit(data, labels)
            y_pred = model.predict(data)
            accuracy_val = accuracy(y_true=labels, y_pred=y_pred)
            # Update Dataframe
            df_q2['accuracy'].append(accuracy_val)
            from sklearn.naive_bayes import GaussianNB
            clf = GaussianNB()
            clf.fit(data, labels)
            y_p = clf.predict(data)
            score = accuracy(y_true=labels, y_pred=y_p)
            df_q2['y_pred'].append(y_pred)
            df_q2['model_name'].append(model_name)
            df_q2['data'].append(data)
            df_q2['labels'].append(labels)
            df_q2['mu_'].append(model.mu_)
            if hasattr(model, 'cov_'):
                df_q2['cov_'].append(model.cov_)
            else:
                df_q2['cov_'].append(model.vars_)
            df_q2['classes'].append(model.classes_)
            df_q2['dataset_name'].append(f)

    df_q2 = pd.DataFrame(df_q2)
    for i, (group_name, df_group) in enumerate(df_q2.groupby('dataset_name')):
        fig = make_subplots(
            subplot_titles=[rf"$\textbf Model: {{{item['model_name']}}} - Accuracy: {{{item['accuracy']}}}$" for
                            idx, item in df_group.iterrows()],
            rows=1, cols=2, horizontal_spacing=0.01, vertical_spacing=.03)
        classes_options = df_group.classes.values[0]
        # Add legend
        for idx, class_name in enumerate(classes_options):
            pred_text = "color of pred :" + str(classes_options[idx])
            true_text = "symbol of true :" + str(classes_options[idx])
            fig.append_trace(go.Scatter(x=[None], y=[None], mode='markers', legendgroup=true_text, showlegend=True,
                                        marker=dict(size=10, symbol=symbols[idx], color="black"), name=true_text),
                             row=1, col=1)
            fig.append_trace(go.Scatter(x=[None], y=[None], mode='markers', legendgroup=pred_text, showlegend=True,
                                        marker=dict(size=10, color=colors[idx]), name=pred_text), row=1, col=1)

        for idx, cell_data in df_group.iterrows():
            title = "Dataset: " + cell_data['dataset_name']
            # Add traces for data-points setting symbols and colors
            main_plot = go.Scatter(x=cell_data.data[:, 0], y=cell_data.data[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=colors[cell_data.y_pred], symbol=symbols[cell_data.labels], size=8,
                                               line=dict(width=1, color="violet")))
            fig.append_trace(main_plot, row=1, col=(idx % 2) + 1)
            fig.update_layout(title=rf"$\textbf{{Decision Boundaries Of Models - LDA & GNB {title}}}$",
                              margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
            # Add ellipses
            for j, mu in enumerate(cell_data.mu_):
                # Add ellipses depicting the covariances of the fitted Gaussians
                cov = cell_data.cov_
                if cov.shape[0] != cov.shape[1]:  # check if real cov or var
                    cov = np.diag(cov[j])
                fig.append_trace(get_ellipse(mu, cov), row=1, col=(idx % 2) + 1)
                # Add `X` dots specifying fitted Gaussians' means
                fig.append_trace(go.Scatter(x=[mu[0]], y=[mu[1]], mode='markers', showlegend=False,
                                            marker=dict(size=14, symbol="x", color="black"),
                                            line=dict(width=5, color='DarkSlateGrey')), row=1, col=(idx % 2) + 1)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    print("Ex3: This is the end!")
