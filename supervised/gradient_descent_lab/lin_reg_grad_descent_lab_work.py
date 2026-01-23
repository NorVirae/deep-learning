import numpy as np
import matplotlib.pyplot as plt
import math

"""
Firstly you need a way to compute

- The cost:
-- Linear Model
f_wb = wx(i) + b

-- cost
J_wb = 1/2m Σ(f_wb(x(i)) - y(i))^2


- The gradient: partial derivative
d_dw_f_wb(x(i)) = 1/m * Σ(f_wb(x(i)) - y(i)) * x(i)
d_db_f_wb(x(i)) = 1/m * Σ(f_wb(x(i)) - y(i))

- Gradient Descent:
w = w - α(d_dw_f_wb(x(i)))
b = b - α(d_db_f_wb(x(i)))


For Predictions of Y:
since w and b is known
y = wx + b
"""


def calculate_cost(x, y, w, b):
    """
    Docstring for calculate_cost

    :param x: Description
    :param y: Description
    :param w: Description
    :param b: Description

    - The cost:
        -- Linear Model
        f_wb = wx(i) + b

        -- cost
        J_wb = 1/2m Σ(f_wb(x(i)) - y(i))^2
    """
    m = x.shape[0]
    sum = 0
    J_wb = 0
    for i in range(m):
        sum += ((w * x[i] + b) - y[i]) ** 2

    J_wb = 1 / 2 * m * sum
    return J_wb


def calculate_gradient(x, y, w, b):
    """
    Docstring for calculate_gradient

    :param x: Description
    :param y: Description
    :param calculate_cost: Description

    for
    - The gradient: partial derivative
        d_dw_f_wb(x(i)) = 1/m * Σ(f_wb(x(i)) - y(i)) * x(i)
        d_db_f_wb(x(i)) = 1/m * Σ(f_wb(x(i)) - y(i))
    """

    m = x.shape[0]

    dj_dw = 0
    dj_db = 0

    sum_b = 0
    sum_w = 0

    for i in range(m):
        sum_w += ((w * x[i] + b) - y[i]) * x[i]
        sum_b += (w * x[i] + b) - y[i]

    dj_dw = 1 / m * sum_w
    dj_db = 1 / m * sum_b

    return dj_dw, dj_db


def calculate_gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    """
    Docstring for calculate_gradeient_descent

    :param x: Description
    :param y: Description
    :param w: Description
    :param b: Description

    - Gradient Descent:
        w = w - α(d_dw_f_wb(x(i)))
        b = b - α(d_db_f_wb(x(i)))
    """
    w = w_in
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = calculate_gradient(x, y, w, b)
        tmp_w = w - alpha * dj_dw
        tmp_b = b - alpha * dj_db

        w = tmp_w
        b = tmp_b

    return w, b


def predict(x, w, b):
    y = w * x + b

    return y


w_in = 0
b_in = 0

num_iters = 100000
alpha = 1.0e-2
x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w, b = calculate_gradient_descent(x_train, y_train, w_in, b_in, alpha, num_iters)

print(w, b)
print(round(w), round(b))
print(predict(1.6, round(w), round(b)))

# plt.plot(x_train, y_train, "--", zorder=9000)

# plt.show()
