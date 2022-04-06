#################################################################
# FILE : city_temperature_prediction.py
# WRITER : Bar Melinarskiy
# EXERCISE : Intro to Machine Learning - 67577 - Exercise 2
# DESCRIPTION: Testing PolynomialFitting class - predict temperature of cities.
#################################################################

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt

pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filepath_or_buffer=filename, parse_dates=['Date']).dropna().drop_duplicates()
    return preprocess_data(df)


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
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
    # Check all range properties are within their value ranges: Drop non-valid months or days
    data = data[data['Month'].isin(np.arange(1, 13, 1))]
    data = data[data['Day'].isin(np.arange(1, 32, 1))]

    # Remove Too high or low temperatures
    data = data[data['Temp'] >= -50]
    data = data[data['Temp'] <= 50]

    data['DayOfYear'] = pd.to_datetime(data['Date']).dt.dayofyear
    # format categorical fields with get_dummies
    # data = pd.get_dummies(data, columns=['Month'])
    # data = pd.get_dummies(data, columns=['Day'])

    # Drop non relevant fields
    fields_to_drop = ["Date"]
    data = data.drop(columns=fields_to_drop)
    return data


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    filename = "../datasets/City_Temperature.csv"
    city_temps_df = load_data(filename)

    # Question 2 - Exploring data for specific country
    israel_temps_df = city_temps_df[city_temps_df['Country'] == 'Israel']

    """"
    Plot a scatter plot showing this relation, and color code the dots by the different years
    (make sure color scale is discrete and not continuous).
    """
    q2_1_title = "Scatter plot of average daily temperature Vs. DayOfYear"
    q2_df = israel_temps_df.astype({"Year": str})
    fig = px.scatter(q2_df, x="DayOfYear", y="Temp", color="Year", title=q2_1_title)
    fig.show()

    """"
    Group the samples by `Month` and plot a bar plot showing for each month the standard deviation of the daily
    temperatures. Suppose you fit a polynomial model (with the correct degree) over data sampled uniformly at random
    from this dataset, and then use it to predict temperatures from random days across the year.
    """
    q2_2_title = "Bar plot Month Vs. std of the daily temperatures"
    month_groups_df = israel_temps_df.groupby(by='Month')
    months = np.array(list(month_groups_df.groups.keys()))
    month_temp_std = np.array(month_groups_df['Temp'].std())
    plt.bar(months, month_temp_std, color='green')
    plt.xlabel("Month")
    plt.ylabel("standard deviation of the daily temperature")
    plt.title(q2_2_title)
    plt.tight_layout()
    plt.show()

    # Question 3 - Exploring differences between countries
    """"
    Returning to the full dataset, group the samples according to `Country` and `Month` and
    calculate the average and standard deviation of the temperature. Plot a line plot of the average
    monthly temperature, with error bars (using the standard deviation) color coded by the country.
    """
    q2_3_title = "Line plot of the mean monthly temperature, with error bars (std) color coded by the country"
    df_group_by_country_month = city_temps_df.groupby(['Country', 'Month'])
    df_q3 = df_group_by_country_month.agg({'Temp': ['mean', 'std']})
    df_q3 = df_q3.xs('Temp', axis=1, drop_level=True).reset_index()
    fig = px.line(df_q3, x="Month", y="mean", error_y="std", color="Country", title=q2_3_title)
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    """
        Over the subset containing observations only from Israel perform the following:
            • Randomly split the dataset into a training set (75%) and test set (25%).
            • For every value k ∈ [1,10], fit a polynomial model of degree k using the training set.
            • Record the loss of the model over the test set, rounded to 2 decimal places.
        Print the test error recorded for each value of k. In addition plot a bar plot showing the test
        error recorded for each value of k.
    """
    only_day_of_year_df = israel_temps_df['DayOfYear']
    train_X, train_y, test_X, test_y = split_train_test(only_day_of_year_df, israel_temps_df['Temp'])
    loss_list = []
    k_arr = range(1, 11)
    for k in k_arr:
        poly_model = PolynomialFitting(k)
        poly_model.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = poly_model.loss(test_X.to_numpy(), test_y.to_numpy())
        loss = round(loss, 2)
        loss_list.append(loss)
        form = "{:.2f}"
        print("The loss for k = " + str(k) + " is: " + form.format(loss))

    loss_arr = np.array(loss_list)
    q2_4_title = "Bar plot showing the test error recorded for each value of k"
    plt.bar(np.array(k_arr), loss_arr, color='pink')
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Loss")
    plt.title(q2_4_title)
    plt.tight_layout()
    plt.show()
    # Question 5 - Evaluating fitted model on different countries
    """
    Fit a model over the entire subset of records from Israel using the k chosen above. Plot a bar
    plot showing the model’s error over each of the other countries.
    """
    # Fit a model over the entire subset of records from Israel using the k = 4
    poly_model = PolynomialFitting(k=4)
    poly_model.fit(israel_temps_df['DayOfYear'].to_numpy(), israel_temps_df['Temp'].to_numpy())

    # Group by country
    df_group_by_country = city_temps_df.groupby(['Country'])
    countries = np.array(list(df_group_by_country.groups.keys()))  # get all the countries keys for plot
    loss_list = []
    # Loop on each country's data and predict based on the model fitted to Israel only
    for country_key, country_df in df_group_by_country:
        loss = poly_model.loss(country_df['DayOfYear'], country_df['Temp'])
        loss_list.append(loss)

    loss_arr = np.array(loss_list)
    q2_5_title = "Bar plot showing the model’s error over each of the other countries for k = 4"
    plt.bar(countries, loss_arr, color='red')
    plt.xlabel("Country")
    plt.ylabel("Loss")
    plt.title(q2_4_title)
    plt.tight_layout()
    plt.show()

    print("The end city temperature prediction!")
