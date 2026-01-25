import numpy as np
import matplotlib.pyplot as plt
import math, copy
import os
import pathlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLE_PATH = os.path.join(BASE_DIR, "deep.mplstyle")
print(BASE_DIR)
plt.style.use(STYLE_PATH)
np.set_printoptions(precision=2)


X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

print(f"X train shape: {X_train.shape} the type is {type(X_train)}")
print(X_train)

print(f"Y train shape {y_train.shape} the type is {type(y_train)} ")
print(y_train)


b_init = 785.1811367994083
w_init = np.array([0.39133535, 18.75376741, -53.36032453, -26.42131618])

print(f"w_nit shape is {w_init.shape} b_init type is {type(b_init)}")


def predict_single_loop(x, w, b):
    """
    Docstring for predict_single_loop

    :param x: Description
    :param w: Description
    :param b: Description
    """
    print(x.shape)
    n = x.shape[0]
    o_pred = 0

    for i in range(n):
        p_i = w[i] * x[i]
        o_pred += p_i

    o_pred = o_pred + b

    return o_pred


def predict(x, w, b):
    """
    Docstring for predict

    :param x: Description
    :param w: Description
    :param b: Description
    """

    p = np.dot(x, w) + b
    return p


x_vector = X_train[0, :]
print(f"The current X first row[0]{x_vector}")

result = predict_single_loop(x_vector, w_init, b_init)
print(f"this is the prediction {result}")

pred_dot_result = predict(x_vector, w_init, b_init)
print(f"this is the result from dot product {pred_dot_result}")


def compute_cost(X, y, w, b):
    """
    Docstring for compute_cost

    :param X: Description
    :param y: Description
    :param w: Description
    :param b: Description

    remember that cost is
    J(w,b) = 1/2m*∑(f_wb(x[i]) - y[i])**2
    f_wb = wx + b
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        f_wb_i = np.dot(X[i], w) + b
        cost = cost + (f_wb_i - y[i]) ** 2

    cost = cost / (2 * m)

    return cost


def compute_gradient(X, y, w, b):
    """
    Docstring for compute_gradient

    :param X: Description
    :param y: Description
    :param w: Description
    :param b: Description

    dj_dw = 1/m ∑(f_wb(x[i]) - y[i]) * x[i]
    dj_db = 1/m ∑(f_wb(x[i]) - y[i])
    """
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0

    for i in range(m):
        dj_dw_i = ((np.dot(X[i], w) + b) - y[i]) * X[i]
        dj_dw += dj_dw_i
        dj_db_i = (np.dot(X[i], w) + b) - y[i]
        dj_db += dj_db_i
    dj_db = dj_db / m
    dj_dw = dj_dw / m
    return dj_dw, dj_db


def compute_gradient_descent(X, y, w, b, compute_gradient, num_iters):
    """
    Docstring for compute_gradient_descent

    :param X: X train Dataset
    :param y: y Train Dataset
    :param w: A list of weight
    :param b: the Bias
    :param compute_gradient: A function that calculates the gradient


    """



computed_cost = compute_cost(X_train, y_train, w_init, b_init)
print(f"This is the computed cost {computed_cost}")

dj_dw, dj_db = compute_gradient(X_train, y_train, w_init, b_init)

print(f"this is dj_dw: {dj_dw} and this is dj_db: {dj_db}")
