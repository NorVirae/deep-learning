import numpy as np

one_dimensional_arr = np.array([10, 12])
# print(one_dimensional_arr)


# array print
array = np.array([1, 3, 5])
# print(array)

# arrange
b = np.arange(9)
# print(b)

# np arrange increment by 3
c = np.arange(0, 21, 7)
# print(c)

d = np.linspace(0, 21, 4, dtype=int)
# print(d)

# char array
g = np.array(["Norbert frank mba!"])
# print(g)

# ones arr
ones = np.ones(3)
# print(ones)

zeros = np.zeros(4)
# print(zeros)

empty = np.empty(4)
# print(empty)

random = np.random.rand(40)
# print(random)

# multi dimensional array
ondD = np.array(
    [
        1,
        2,
        3,
        4,
        5,
        6,
    ]
)
multi = np.reshape(ondD, (2, 3))

# print(multi.shape)

tdarray1 = np.array([2, 2, 2])
tdarray1 = np.reshape(tdarray1, (3, 1))
# print(tdarray1)
tdarray2 = np.array([[3, 3, 3], [4, 4, 4]])

# print(tdarray1)


one_dim_arr = np.array([1, 2, 3, 4, 5, 6])
multi_dim_arr = np.reshape(one_dim_arr, (2, 3))
# print(multi_dim_arr * 2, "HELLO")

# print(multi_dim_arr.ndim)
# print(multi_dim_arr.shape)
# print(multi_dim_arr.size)


two_dim = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(two_dim[0:1])
sect1 = two_dim[:, 1:3]
least_sect1 = sect1[1:3]
sect1_real = 1 * (
    (least_sect1[0][0] * least_sect1[1][1]) - (least_sect1[0][1] * least_sect1[1][0])
)
print(least_sect1)
print(sect1_real)


sect2 = two_dim[:, 0:3:2]
least_sect2 = sect2[1:3]
sect2_real = -1 * (
    (least_sect2[0][0] * least_sect2[1][1]) - (least_sect2[0][1] * least_sect2[1][0])
)
print(least_sect2)
print(sect2_real)

sect3 = two_dim[:, 0:2:1]
least_sect3 = sect3[1:3]
sect3_real = 1 * (
    (least_sect3[0][0] * least_sect3[1][1]) - (least_sect3[0][1] * least_sect3[1][0])
)
print(least_sect3)
print(sect3_real)

det = sect1_real + sect2_real + sect3_real
print("DETERMINANT for matrix {}  is  {:.2}".format(two_dim, float(det)))


a = np.array([0, 1, 2, 3, 4, 5, 6])
n_4 = a[::1]
# print(n_4)


# STACKING OF ARTAYs

a = np.array([[1, 1], [2, 2]])
b = np.array([[3, 3], [4, 4]])

vstack = np.vstack((a, b))
hstack = np.hstack((a, b))
splithstack = np.hsplit((a), 2)
splitvstack = np.vsplit((a), 2)

top, bottom = splithstack
# print(vstack, "VSTACK")
# print(hstack, "HSTACK")
print("TOP", top,"\n", "BOTTOM", bottom)