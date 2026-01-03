import numpy as np
import matplotlib.pyplot as plt


def plotlines(A, b):
    x = np.linspace(-10, 10, 400)

    for i in range(len(b)):
        a1, a2 = A[i]
        y = (b[i] - a1 * x) / a2
        plt.plot(x, y)

    plt.axhline(0, color="black")
    plt.axvline(0, color="black")
    plt.grid(True)
    plt.show()
