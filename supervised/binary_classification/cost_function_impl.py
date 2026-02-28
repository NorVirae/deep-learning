import numpy as np
import matplotlib.pyplot as plt


X_train = np.array(
    [[0.5, 1.5], [1, 1], [1.5, 0.5], [2, 2], [3, 0.5], [1, 2.5]]
)  # (m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])


# fig, ax = plt.subplots(1, 1, figsize=(4, 4))
neg = y_train == 0
pos = y_train == 1

# print(X_train[pos, 0], neg)
# ax.scatter(X_train[pos, 0], X_train[pos, 1], s=80, marker="x", c="red")
# ax.scatter(X_train[neg, 0], X_train[neg, 1], s=80, marker="o", c="blue")
# ax.axis([0, 4, 0, 3.5])

# ax.set_ylabel("$x_1$", fontsize=12)
# ax.set_xlabel("$x_0$", fontsize=12)


def compute_cost_logistic(X, y, w, b):
    """
    X: training set
    y: Prediction set
    w: Weight
    b: bias

    so Cost is J(w, b)
    J(w,b) = 1/m ∑[loss(F(w,b))]
    loss(F(w,b)) = -y^(i)log(F(w, b)^(i)) - (1 - y^(i))log(1 - F(w, b)(i))
    F(w,b) = g(z^(i))
    z(i) = wx^(i) + b
    g(z(i)) = 1/(1 + e^-(wx^(i) + b))
    """
    m = X.shape[0]
    loss = 0.0
    cost = 0.0

    for i in range(m):
        z_i = np.dot(w, X[i]) + b
        f_wb_i = 1 / (1 + np.exp(-z_i))
        loss += -y[i] * np.log(f_wb_i) - (1 - y[i]) * np.log(1 - f_wb_i)
    cost = loss / m
    return cost


w_tmp = np.array([1, 1])
b_tmp = -3
cost = compute_cost_logistic(X_train, y_train, w_tmp, b_tmp)
print(compute_cost_logistic(X_train, y_train, w_tmp, -4), "cost with -4")
print(cost)
# plt.show()


"""
Lets Plot the decison boundary
remeber 
w0x0 + w1x1 + b = 0
w0 = 1
w1 = 1
so
x0 + x1 + b = 0
x1 = -b-x0
"""

x0 = np.arange(0, 6)
x1 = -b_tmp - x0
x1_diff = 4 - x0

print(x0)
print(x1)

fig, ax = plt.subplots(1, 1, figsize=(4, 4))

ax.plot(x0, x1, c="blue",label="$b$=-3")
ax.plot(x0, x1_diff, c="red", label="$b$=-4")


ax.scatter(X_train[pos, 0], X_train[pos, 1], s=80, marker="x", c="red")
ax.scatter(X_train[neg, 0], X_train[neg, 1], s=80, marker="o", c="blue")
ax.axis([0, 4, 0, 3.5])

ax.set_ylabel("$x_1$", fontsize=12)
ax.set_xlabel("$x_0$", fontsize=12)

plt.show()
