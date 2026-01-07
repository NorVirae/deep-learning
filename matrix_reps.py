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


# trying to rep 2 x 2 as plot line
A_system = np.hstack((A, b.reshape((2, 1))))
print(A_system)

plotlines(A, b)

'''
Lets try this
{−𝑥1+3𝑥2=7
,3𝑥1−9𝑥2=1,(2)

A = [-1 3
    3 -9]

b = [7, 1]

'''

B = np.array([[4, 3], [1, -5]], dtype=np.dtype(float))
g = np.array([6, 8], dtype=np.dtype(float))

sol_b = np.linalg.solve(B, g)
print(f"solution to eq2 {sol_b}")

det_b = np.linalg.det(B)
print(f'det for eq2 {det_b}')


plot_b = np.hstack((B, g.reshape(2, 1)))
plotlines(B, g)


try:
    x_2 = np.linalg.solve(B, g)
except np.linalg.LinAlgError as err:
    print(err)

    