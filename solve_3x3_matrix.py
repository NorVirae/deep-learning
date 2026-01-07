import numpy as np
import numpy.linalg as la

"""

4𝑥1−3𝑥2+𝑥3=−10,
2𝑥1+𝑥2+3𝑥3=0,
−𝑥1+2𝑥2−5𝑥3=17,(1)

"""

A = np.array([
    [4, -3, 1],
    [2, 1, 3],
    [-1, 2, -5],
], dtype=np.dtype(float))

b = np.array([20, 0, 17])

print(f"this is the shape of A:\n{A.shape}")

print(f"this is the shape of b:\n{b.shape}")

x = la.solve(A, b)

print(f"X1: {x[0]}, x2: {x[1]}, x3: {x[2]}")

print("the Determinant of Matrix A")
det = la.det(A)
print(det)