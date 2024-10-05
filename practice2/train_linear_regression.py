from practice.models import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import fire

def train_regession_least_squares(path_to_data: str):
    data = pd.read_csv(path_to_data)
    print(data)

    X = data[['area', 'bathrooms', 'bedrooms', 'parking']].values
    y = data['price'].values

    model = LinearRegression(
        regularization_type='l2',
        alpha=0.1
    )

    model.fit(X, y, method='least_squares')

    y_pred = model.predict(X)

    # plot points and final regression
    plt.scatter(data['area'], y, color='blue')
    plt.scatter(data['area'], y_pred, color='red')
    plt.show()

def train_regression_gradient_descent(path_to_data: str):
    data = pd.read_csv(path_to_data)
    print(data)

    X = data[['area']].values
    y = data['price'].values

    # standardize data
    X_mean, X_std = X.mean(), X.std()
    X = (X - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()
    y = (y - y_mean) / y_std

    model = LinearRegression(
        regularization_type='l2',
        alpha=0.1,
        max_iter=1000,
        tol=1e-3
    )

    model.fit(X, y, method='gradient_descent', learning_rate=0.01)

    y_pred = model.predict(X)

    # destandardize
    y = y * y_std + y_mean
    y_pred = y_pred * y_std + y_mean
    X = X * X_std + X_mean

    # plot points and final regression
    plt.scatter(X, y, color='blue')
    plt.scatter(X, y_pred, color='red')
    plt.show()

def train_polynomial_regression(path_to_data: str):
    data = pd.read_csv(path_to_data)
    print(data)

    X = data[['area']].values
    y = data['price'].values

    # make polynomials of deg 3 from data
    X_poly = np.hstack([X, X ** 2, X ** 3])

    # standardize
    X_mean, X_std = X_poly.mean(axis=0), X_poly.std(axis=0)
    X_poly = (X_poly - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()

    model = LinearRegression(
        regularization_type='l2',
        alpha=0.1,
        max_iter=1000,
        tol=1e-3
    )
    model.fit(X_poly, y, method='gradient_descent', learning_rate=0.01)

    y_pred = model.predict(X_poly)
    y_pred = y_pred * y_std + y_mean
    X = X * X_std[0] + X_mean[0]
    y = y * y_std + y_mean

    # plot points and final model
    plt.scatter(X, y, color='blue')
    plt.scatter(X, y_pred, color='red')
    plt.show()


def wave_data_regression():
    # create data
    X = np.linspace(0, 1, 100)
    y = np.sin(2 * np.pi * X) + np.random.normal(0, 0.1, 100)

    # plot data
    # plt.scatter(X, y)
    # plt.show()

    # reshape for model
    # X = X.reshape(-1, 1)

    # polynomial features
    def make_poly_features(X: np.ndarray, n: int) -> np.ndarray:
        res = [X**i for i in range(1, n+1)]
        return np.vstack(res).T


    # max 20 else overflow
    X = make_poly_features(X, 19)

    # standardize
    X_mean, X_std = X.mean(axis=0), X.std(axis=0)
    X = (X - X_mean) / X_std

    y_mean, y_std = y.mean(), y.std()


    model = LinearRegression(
        regularization_type='l2',
        alpha=0.1,
        max_iter=15000,
        tol=1e-9
    )
    model.fit(X, y, method='gradient_descent', learning_rate=0.1)

    y_pred = model.predict(X)

    # destandadize
    y = y * y_std + y_mean
    y_pred = y_pred * y_std + y_mean
    X = X * X_std + X_mean

    # plot points and final model
    plt.scatter(X[:, 0], y, color='blue')
    plt.scatter(X[:, 0], y_pred, color='red')
    plt.show()

    # plot weights bar diagram
    plt.bar(range(len(model._weights)), model._weights)
    plt.show()

if __name__ == '__main__':
     fire.Fire(train_regession_least_squares)
     fire.Fire(train_regression_gradient_descent)
     fire.Fire(train_polynomial_regression)
     fire.Fire(wave_data_regression)