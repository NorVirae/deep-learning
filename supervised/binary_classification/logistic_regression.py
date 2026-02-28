import numpy as np
import matplotlib.pyplot as plt


# checkout nupy exponential e^z
array_el = np.arange(-8, 10, 1)
array_exp = np.exp(array_el)

# print(f"Array elements : {array_el} \nArray Exponential: {array_exp}")

# also does for single numbers
single_el = 0
single_exp = np.exp(single_el)
print(f" Single Element: {single_el} \n Exponential Single Element: {single_exp}")


def sigmoid(z):
    """
    The Sigmoid function has got the formula

    g(z) = 1/(1 + e^-z)

    Implementation:
    """
    g = 1 / (1 + np.exp(-z))
    return g


# checking out sigmoid function passing through the center

sigmoid_single = sigmoid(single_el)
# print(f"Single Element: {single_el} sigmoid single: {sigmoid_single}")

y = sigmoid(array_el)
# print(f"Array Element: {array_el} sigmoid array: {sigmoid_array}")

print("Elements  ----->  Sigmoid")
print(np.c_[array_el, y])

# Lets plot this thing

fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(array_el, y)
ax.set_ylabel("Sigmoid [Y] (1) or (0)")
ax.set_xlabel("X or Array Elements")


plt.show()