from turtledemo.__main__ import font_sizes

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression
import IMLearn.utils.utils as utils
from typing import NoReturn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from utils import animation_to_gif

pio.templates.default = "simple_white"


def load_data(filename: str) -> (pd.DataFrame, pd.DataFrame):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename).dropna().drop_duplicates()
    return preprocess_data(df)


def preprocess_data(data: pd.DataFrame) -> (pd.DataFrame, pd.DataFrame):
    """
    Preprocess data.
    Parameters
    ----------
    data: pd.DataFrame
        House prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    # Drop houses with negative number of bedrooms/bathrooms
    data = data[data['bedrooms'] >= 0]
    data = data[data['bathrooms'] >= 0]
    data = data[data['sqft_living'] > 0]
    data = data[data['floors'] > 0]
    data['year_sold'] = pd.to_datetime(data['date']).dt.year
    data['yr_renovated'] = data['yr_renovated'].astype(int)
    data['renovated_bool'] = np.where(data['year_sold'] - data['yr_renovated'] <= 30, 1, 0)
    data = pd.get_dummies(data, columns=['year_sold'])
    fields_to_drop = ["id", "date", "long", "lat", "yr_renovated"]
    data = data.drop(columns=fields_to_drop)
    # for field in fields_to_drop:
    #     data = data.drop([field], axis=1)

    # Check all range properties are within their value ranges
    data = data[data['view'].isin(np.arange(0, 5, 1))]
    data = data[data['condition'].isin(np.arange(0, 6, 1))]
    data = data[data['grade'].isin(np.arange(0, 14, 1))]

    data['grade_bad'] = np.where(data['grade'] <= 5, 1, 0)
    data['grade_medium'] = np.where(data['grade'].isin(np.arange(6, 11, 1)), 1, 0)
    data['grade_great'] = np.where(data['grade'] >= 11, 1, 0)
    data = pd.get_dummies(data, columns=['zipcode'])

    # Create labels and features to return
    labels = data["price"]
    features = data.drop(columns="price")
    return features, labels


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    # Loop on each feature in data frame X:
    corr_data = {'features': [], 'corr': []}
    for (feature, feature_data) in X.iteritems():
        # Pearson's correlation coefficient = covariance(X, Y) / (stdv(X) * stdv(Y))
        pearson_corr = (np.cov(feature_data, y) / (np.std(feature_data) * np.std(y)))[1][0]
        # Save correlations to get best and worst later
        corr_data['features'].append(feature)
        corr_data['corr'].append(np.abs(pearson_corr))

        title = "Scatterplot of Correlation btw Feature: " + feature + " Vs. Response"
        subtitle = "The Pearson's correlation btw them is: " + str(pearson_corr)
        plt.title(subtitle, fontsize=10)
        plt.suptitle(title, fontsize=12)
        plt.xlabel("Feature: " + feature)
        plt.ylabel("Response")
        plt.scatter(feature_data, y, color="green")
        # plt.tight_layout()
        plt.show()
        plot_name = output_path + "corr_btw_response_and_" + feature + ".png"
        # plt.savefig(plot_name)
        plt.clf()
        plt.cla()

    df = pd.DataFrame(corr_data)
    df = df.sort_values(by='corr', ascending=False)
    print("Feature with best corr is: " + str(df['features'].iloc[0]) + " it's corr is: " + str(df['corr'].iloc[0]))
    print("Feature with worst corr is: " + str(df['features'].iloc[-1]) + " it's corr is: " + str(df['corr'].iloc[-1]))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    filename = "../datasets/house_prices.csv"
    X, y = load_data(filename)

    # Question 2 - Feature evaluation with respect to response
    plots_path = "../exercises/"
    # feature_evaluation(X, y, plots_path)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = np.arange(10, 101, 1)
    loss_list = []
    for percentage in (percentages.repeat(10) / 100):
        X_train_p = train_X.sample(frac=percentage)
        y_train_p = train_y[train_y.index.isin(X_train_p.index)]
        model = LinearRegression(include_intercept=True)
        model.fit(X_train_p.to_numpy(), y_train_p.to_numpy())
        loss = model.loss(test_X.to_numpy(), test_y.to_numpy())
        loss_list.append(loss)

    loss_arr = np.array(loss_list).reshape(-1, 10)
    mean_loss = np.mean(loss_arr, axis=1)
    std_loss = np.std(loss_arr, axis=1)

    y1 = mean_loss - 2 * std_loss
    y2 = mean_loss + 2 * std_loss

    data_for_fig = (
        go.Scatter(x=percentages, y=mean_loss, mode="markers+lines", name="Mean Loss", line=dict(dash="dash"),
                   marker=dict(color="green", opacity=.7)),
        go.Scatter(x=percentages, y=y1, fill=None, mode="lines", line=dict(color="lightgrey"),
                   showlegend=False),
        go.Scatter(x=percentages, y=y2, fill='tonexty', mode="lines",
                   line=dict(color="lightgrey"),
                   showlegend=False),)

    layout = go.Layout(
        title_text="Scatterplot of average loss as function of training size",
        xaxis={"title": "percentage training size"},
        yaxis={"title": "average loss"})

    fig = go.Figure(data=data_for_fig, layout=layout)

    fig.show()

    # frames = []
    # for i in range(10):
    #     frames.append(go.Frame(
    #         data=[
    #             go.Scatter(x=mean_pred, y=plot_data['test_size'], mode="markers+lines", name="Predicted Points",
    #                        marker=dict(color="blue", opacity=.7))]))
    #
    # for i in range(len(frames)):
    #     frames[i]["data"] = (go.Scatter(x=plot_data['test_size'], y=mean_pred, mode="markers+lines",
    #                                     name="Mean Prediction",
    #                                     line=dict(dash="dash"), marker=dict(color="green", opacity=.7)),
    #                          go.Scatter(x=plot_data['test_size'], y=mean_pred - 2 * var_pred, fill=None,
    #                                     mode="lines",
    #                                     line=dict(color="lightgrey"), showlegend=False),
    #                          go.Scatter(x=plot_data['test_size'], y=mean_pred + 2 * var_pred, fill='tonexty',
    #                                     mode="lines",
    #                                     line=dict(color="lightgrey"), showlegend=False),) + frames[i]["data"]
    #
    # fig = go.Figure(data=frames[0]["data"],
    #                 frames=frames[1:],
    #                 layout=go.Layout(
    #                     title=dict(text="Scatterplot of average loss as function of training size"),
    #                     xaxis=dict(title="percentage training size"),
    #                     yaxis=dict(title="average loss"),
    #                     updatemenus=[dict(visible=True,
    #                                       type="buttons",
    #                                       buttons=[dict(label="Play",
    #                                                     method="animate",
    #                                                     args=[None, dict(frame={"duration": 1000})])])]))
    # # animation_to_gif(fig, f"../ex2_q4.gif", 1000)
    # fig.show()

    print("The end!")
