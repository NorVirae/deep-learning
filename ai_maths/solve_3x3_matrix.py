import numpy as np
import numpy.linalg as la

"""

4𝑥1−3𝑥2+𝑥3=−10,
2𝑥1+𝑥2+3𝑥3=0,
−𝑥1+2𝑥2−5𝑥3=17,(1)

"""

A = np.array(
    [
        [4, -3, 1],
        [2, 1, 3],
        [-1, 2, -5],
    ],
    dtype=np.dtype(float),
)

b = np.array([20, 0, 17])

print(f"this is the shape of A:\n{A.shape}")

print(f"this is the shape of b:\n{b.shape}")

x = la.solve(A, b)

print(f"X1: {x[0]}, x2: {x[1]}, x3: {x[2]}")

print("the Determinant of Matrix A")
det = la.det(A)
print(det)

# what happens when a system has got no solution?

"""

"""
A2 = np.array([[1, 1, 1], [0, 1, -3], [2, 1, 5]])
b2 = np.array([2, 1, 0])

# confirm shape.
print(f"A2 shape is {A2.shape}, b2 shape is {b2.shape}")

sol = la.solve(A2, b2)
print(f"solution to A2 b2 is {sol}")

# the determinant is
det  = la.det(A2)
print(f"Determinant is {det}")