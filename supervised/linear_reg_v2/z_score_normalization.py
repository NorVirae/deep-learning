import numpy as np
import matplotlib.pyplot as plt
import os
from utils_multi import load_house_data, run_gradient_descent
from utils_multi import norm_plot, plt_equal_scale, plot_cost_i_w
from utils_common import dlc
import math
from libs.grad_descent_work import compute_gradient_descent_external

print(f"Path: {os.getcwd()}")
np.set_printoptions(precision=2)

plt.style.use("./lib/deep.mplstyle")

X_train, y_train = load_house_data()
X_features = ["size(sqft)", "bedrooms", "floors", "age"]

# fig,ax=plt.subplots(1, 4, figsize=(12, 3), sharey=True)
# for i in range(len(ax)):
#     print(i, "Check")
#     ax[i].scatter(X_train[:,i],y_train)
#     ax[i].set_xlabel(X_features[i])
# ax[0].set_ylabel("Price (1000's)")

# _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha=9.9e-7)

# plot_cost_i_w(X_train, y_train, hist)

# _, _, hist = run_gradient_descent(X_train, y_train, 10, alpha=1e-7)

# plot_cost_i_w(X_train, y_train, hist)


def mean(X):
    uj = np.zeros((X.shape[1],))
    m = X.shape[0]
    n = X.shape[1]
    for j in range(n):
        for i in range(m):
            xj_i = X[i][j]
            uj[j] += xj_i
        uj[j] = uj[j] / m
    return uj


def std(X):
    uj = mean(X)
    m = X.shape[0]
    n = X.shape[1]
    sigma = np.zeros((X.shape[1],))  # use sigma instead

    for j in range(n):
        for i in range(m):
            xj_i = X[i][j]
            sigma_i = (xj_i - uj[j]) ** 2
            sigma[j] += sigma_i
        sigma[j] = math.sqrt(sigma[j] / m)

    return sigma


def z_score_normalize(X):
    """
    Docstring for z_score_normalize

    :param X: Description

    xj_i = (xj_i - uj)/𝜎j

    uj = 1/m * ∑(xj_i) i.e i = 0 to m

    𝜎j^2 = 1/m * ∑(xj_i - uj)^2 i.e i = 0 to m

    """
    # X_mu = mean(X)
    X_mu = np.mean(X, axis=0)
    # sigma = std(X)  # use sigma instead
    sigma = np.std(X, axis=0)

    X_norm = (X - X_mu) / sigma

    return X_norm, X_mu, sigma


def predict(X_house, X_mu, sigma, w_norm, b_norm):
    """
    Docstring for predict

    :param X_house: Description
    :param X_mu: Description
    :param sigma: Description
    xj_i = (xj_i - uj)/𝜎j
    """
    # first normalize X_house
    X_house_norm = (X_house - X_mu) / sigma
    print(X_house_norm, "X House Norm", w_norm, "JIMA", b_norm)
    price = np.dot(X_house_norm, w_norm) + b_norm
    return price


X_norm, X_mu, sigma = z_score_normalize(X_train)
w_norm, b_norm, _, _ = compute_gradient_descent_external(
    X_norm, y_train, np.zeros(X_norm.shape[1])
)
X_house = np.array([1200, 3, 1, 40])
print(
    predict(X_house, X_mu, sigma, w_norm, b_norm),
    "Price Prediction",
)
