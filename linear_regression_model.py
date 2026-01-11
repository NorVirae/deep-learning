import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train =  {y_train}")

m = x_train.shape[0]
print(f"Number of training examples = {m}")

i = 0

while i < m:
    x_i = x_train[i]
    y_i = y_train[i]

    print(f"this is x{i} = {x_i} and this is y{i} = {y_i}")
    i += 1

"""
wasn't put in the course i was using but to get the bias and weight you solve the system of
two equations where 
y = wx + b
shown as
w + b = 300
2w + b = 500

solving the systems of equations will give

w = 200
b = 100

"""

w = 200
b = 100


def compute_model_output(x, w, b):

    m = x.shape[0]
    f_wb = np.zeros(m)
    print(f_wb)
    for i in range(m):
        f_wb[i] = w * x[i] + b

    return f_wb

tmp_f_wb = compute_model_output(x_train, w, b)
print(f"Check {tmp_f_wb}")

# Actual Values
plt.plot(x_train, tmp_f_wb, c="b", label="Predictions")

plt.scatter(x_train, y_train, marker="x", c="r", label="Actual values")
plt.title("Housing Prices!")
plt.ylabel("Price (in 1000 of dollars)")
plt.xlabel("Size (1000 sqft)")


plt.show()
