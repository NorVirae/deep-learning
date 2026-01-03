import numpy as np
import matplotlib.pyplot as plt
from utils import plotlines

"""
{
−𝑥1+3𝑥2=7,
3𝑥1+2𝑥2=1,
(1)

[
[-1 3 7],
[3 2 1]
]

repesent each side as it's own matrix
we have 
matrix A
[[-1 3]
[3 2]]

vector b
[[7],[1]]
"""
# above is a system of equation, representing it as a matrix in numpy we have

A = np.array([[-1, 3], [3, 2]], dtype=np.dtype(float))
b = np.array([7, 1], dtype=np.dtype(float))

ans = np.linalg.solve(A, b)

print(f"Solition to linear equation x1: {ans[0]}, x2: {ans[1]}")

d = np.linalg.det(A)

print(f"The determinant to the matrix A is {d}")


# trying to rep 2X 2 as plot line
A_system = 