"""
lab_utils_multi.py
    routines used in Course 1, Week2, labs4-9 dealing with multiple variables
"""

import copy
import math
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from utils_common import compute_cost, compute_gradient_matrix
from utils_common import dlblue, dlorange, dldarkred, dlmagenta, dlpurple

plt.style.use("./lib/deep.mplstyle")

##########################################################
# Plotting Routines
##########################################################


# plots p1 vs p2. Prange is an array of entries [min, max, steps]. In feature scaling lab.
def plt_contour_multi(x, y, w, b, ax, prange, p1, p2, title="", xlabel="", ylabel=""):
    """ """
    contours = [
        1e2,
        2e2,
        3e2,
        4e2,
        5e2,
        6e2,
        7e2,
        8e2,
        1e3,
        1.25e3,
        1.5e3,
        1e4,
        1e5,
        1e6,
        1e7,
    ]
    px, py = np.meshgrid(np.linspace(*(prange[p1])), np.linspace(*(prange[p2])))
    z = np.zeros_like(px)
    for i in range(px.shape[0]):
        for j in range(px.shape[1]):
            w_ij = w
            b_ij = b
            if p1 <= 3:
                w_ij[p1] = px[i, j]
            if p1 == 4:
                b_ij = px[i, j]
            if p2 <= 3:
                w_ij[p2] = py[i, j]
            if p2 == 4:
                b_ij = py[i, j]

            z[i][j] = compute_cost(x, y, w_ij, b_ij)
    CS = ax.contour(
        px,
        py,
        z,
        contours,
        linewidths=2,
        colors=[dlblue, dlorange, dldarkred, dlmagenta, dlpurple],
    )
    ax.clabel(CS, inline=1, fmt="%1.2e", fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=14)


def plt_equal_scale(X_train, X_norm, y_train):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    prange = [
        [0.238 - 0.045, 0.238 + 0.045, 50],
        [-25.77326319 - 0.045, -25.77326319 + 0.045, 50],
        [-50000, 0, 50],
        [-1500, 0, 50],
        [0, 200000, 50],
    ]
    w_best = np.array([0.23844318, -25.77326319, -58.11084634, -1.57727192])
    b_best = 235
    plt_contour_multi(
        X_train,
        y_train,
        w_best,
        b_best,
        ax[0],
        prange,
        0,
        1,
        title="Unnormalized, J(w,b), vs w[0],w[1]",
        xlabel="w[0] (size(sqft))",
        ylabel="w[1] (# bedrooms)",
    )
    #
    w_best = np.array([111.1972, -16.75480051, -28.51530411, -37.17305735])
    b_best = 376.949151515151
    prange = [
        [111 - 50, 111 + 50, 75],
        [-16.75 - 50, -16.75 + 50, 75],
        [-28.5 - 8, -28.5 + 8, 50],
        [-37.1 - 16, -37.1 + 16, 50],
        [376 - 150, 376 + 150, 50],
    ]
    plt_contour_multi(
        X_norm,
        y_train,
        w_best,
        b_best,
        ax[1],
        prange,
        0,
        1,
        title="Normalized, J(w,b), vs w[0],w[1]",
        xlabel="w[0] (normalized size(sqft))",
        ylabel="w[1] (normalized # bedrooms)",
    )
    fig.suptitle("Cost contour with equal scale", fontsize=18)
    # plt.tight_layout(rect=(0,0,1.05,1.05))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def norm_plot(ax, data):
    scale = (np.max(data) - np.min(data)) * 0.2
    x = np.linspace(np.min(data) - scale, np.max(data) + scale, 50)
    _, bins, _ = ax.hist(data, x, color="xkcd:azure")
    # ax.set_ylabel("Count")

    mu = np.mean(data)
    std = np.std(data)
    dist = norm.pdf(bins, loc=mu, scale=std)

    axr = ax.twinx()
    axr.plot(bins, dist, color="orangered", lw=2)
    axr.set_ylim(bottom=0)
    axr.axis("off")


def plot_cost_i_w(X, y, hist):
    ws = np.array([p[0] for p in hist["params"]])
    rng = max(abs(ws[:, 0].min()), abs(ws[:, 0].max()))
    wr = np.linspace(-rng + 0.27, rng + 0.27, 20)
    cst = [
        compute_cost(X, y, np.array([wr[i], -32, -67, -1.46]), 221)
        for i in range(len(wr))
    ]

    fig, ax = plt.subplots(1, 2, figsize=(12, 3))
    ax[0].plot(hist["iter"], (hist["cost"]))
    ax[0].set_title("Cost vs Iteration")
    ax[0].set_xlabel("iteration")
    ax[0].set_ylabel("Cost")
    ax[1].plot(wr, cst)
    ax[1].set_title("Cost vs w[0]")
    ax[1].set_xlabel("w[0]")
    ax[1].set_ylabel("Cost")
    ax[1].plot(ws[:, 0], hist["cost"])
    plt.show()


##########################################################


# This version saves more values and is more verbose than the assigment versons
def gradient_descent_houses(
    X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters
):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : intiial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (scalar,int)   : number of iterations to run gradient descent
    Returns
      w (ndarray (n,))    : Updated values of parameters
      b (ndarray (scalar)  Updated value of parameter
    """

    # An array to store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []

    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    save_interval = np.ceil(
        num_iters / 10000
    )  # prevent resource exhaustion for long runs

    print(
        f"Iteration Cost          w0       w1       w2       w3       b       djdw0    djdw1    djdw2    djdw3    djdb  "
    )
    print(
        f"---------------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|"
    )

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(
                f"{i:9d} {cst:0.5e} {w[0]: 0.1e} {w[1]: 0.1e} {w[2]: 0.1e} {w[3]: 0.1e} {b: 0.1e} {dj_dw[0]: 0.1e} {dj_dw[1]: 0.1e} {dj_dw[2]: 0.1e} {dj_dw[3]: 0.1e} {dj_db: 0.1e}"
            )

    return w, b, hist  # return w,b and history for graphing


def run_gradient_descent(X, y, iterations=1000, alpha=1e-6):
    """sets up and runs gradient_descent_houses"""

    _, n = X.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    # run gradient descent
    w_out, b_out, hist_out = gradient_descent_houses(
        X,
        y,
        initial_w,
        initial_b,
        compute_cost,
        compute_gradient_matrix,
        alpha,
        iterations,
    )
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.2f}")

    return w_out, b_out, hist_out


# compact extaction of hist data
# x = hist["iter"]
# J  = np.array([ p    for p in hist["cost"]])
# ws = np.array([ p[0] for p in hist["params"]])
# dj_ws = np.array([ p[0] for p in hist["grads"]])

# bs = np.array([ p[1] for p in hist["params"]])


def run_gradient_descent_feng(X, y, iterations=1000, alpha=1e-6):
    _, n = X.shape
    # initialize parameters
    initial_w = np.zeros(n)
    initial_b = 0
    # run gradient descent
    w_out, b_out, _ = gradient_descent(
        X,
        y,
        initial_w,
        initial_b,
        compute_cost,
        compute_gradient_matrix,
        alpha,
        iterations,
    )
    print(f"w,b found by gradient descent: w: {w_out}, b: {b_out:0.4f}")

    return w_out, b_out


def gradient_descent(
    X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters
):
    """
    Performs batch gradient descent to learn theta. Updates theta by taking
    num_iters gradient steps with learning rate alpha

    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters
      b_in (scalar)       : intiial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (scalar,int)   : number of iterations to run gradient descent
    Returns
      w (ndarray (n,))    : Updated values of parameters
      b (ndarray (scalar)  Updated value of parameter
    """

    # An array to store values at each iteration primarily for graphing later
    hist = {}
    hist["cost"] = []
    hist["params"] = []
    hist["grads"] = []
    hist["iter"] = []

    w = copy.deepcopy(w_in)  # avoid modifying global w within function
    b = b_in
    save_interval = np.ceil(
        num_iters / 10000
    )  # prevent resource exhaustion for long runs

    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w, b)

        # Update Parameters using w, b, alpha and gradient
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        # Save cost J,w,b at each save interval for graphing
        if i == 0 or i % save_interval == 0:
            hist["cost"].append(cost_function(X, y, w, b))
            hist["params"].append([w, b])
            hist["grads"].append([dj_dw, dj_db])
            hist["iter"].append(i)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i % math.ceil(num_iters / 10) == 0:
            # print(f"Iteration {i:4d}: Cost {cost_function(X, y, w, b):8.2f}   ")
            cst = cost_function(X, y, w, b)
            print(f"Iteration {i:9d}, Cost: {cst:0.5e}")
    return w, b, hist  # return w,b and history for graphing


def load_house_data():
    data = np.loadtxt("./lib/houses.txt", delimiter=",", skiprows=1)
    X = data[:, :4]
    y = data[:, 4]
    return X, y


def zscore_normalize_features(X, rtn_ms=False):
    """
    returns z-score normalized X by column
    Args:
      X (ndarray (m,n)) :
    Returns
      X_norm (ndarray (m,n)): input normalized by column
    """
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma

    if rtn_ms:
        return X_norm, mu, sigma
    return X_norm
