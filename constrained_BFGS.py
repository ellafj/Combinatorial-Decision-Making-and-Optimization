import matplotlib.pyplot as plt
import numpy as np

def initialize_variables(x):
    # x is an array with five entries, where the first three are A11, A12=A21 and A22.
    # The last two entries is coordinate of center for initial guess
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    vec = np.array([x[3], x[4]])
    return A, vec


# Calculates the residual for f1
def res1(zi, A, c):
    return np.matmul(np.matmul((zi - c).T, A), zi - c) - 1


# Calculates the residual for f2
def res2(zi, A, b):
    return np.matmul(np.matmul(zi.T, A), zi) - np.matmul(zi.T, b) - 1


# Calculating value for f_i function
def f(x, z, i, inner_points):
    A, c = initialize_variables(x)
    value = 0
    for ind in range(len(z)):
        if ind in inner_points:
            if i == 1:
                value += max(res1(z[ind], A, c), 0) ** 2
            else:
                value += max(res2(z[ind], A, c), 0) ** 2
        else:
            if i == 1:
                value += min(res1(z[ind], A, c), 0) ** 2
            else:
                value += min(res2(z[ind], A, c), 0) ** 2
    return value


# Generates the randomly scattered points
def generate_points(x, scale1 = 1, scale2 = 2 * 10 ** (-1),  size = 500):
    A, c = initialize_variables(x)
    all_points = np.random.multivariate_normal(c, scale1 * np.linalg.inv(A), size = size)
    inner_points = []
    n = len(all_points)
    x = np.random.normal(0, scale2, n)
    y = np.random.normal(0, scale2, n)

    for i in range(n):
        if res1(all_points[i], A, c) <= 0:
            inner_points.append(i)

    all_points[:, 0] = all_points[:, 0] + x
    all_points[:, 1] = all_points[:, 1] + y

    return all_points, inner_points


# The contraint functions
def constraints(x):
    def c1(x, y1, y2):
        return x[0] - y1
    def c2(x, y1, y2):
        return y2 - x[0]
    def c3(x, y1, y2):
        return x[2] - y1
    def c4(x, y1, y2):
        return y2 - x[2]
    def c5(x, y1, y2):
        return (np.abs(x[0] * x[2])) ** (1/2) - (y1 ** 2 + x[1] ** 2) ** (1/2)

    return [c1, c2, c3, c4, c5]


