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


# Calculates the residual for f
def res(zi, A, c):
    return np.matmul(np.matmul((zi - c).T, A), zi - c) - 1


# Calculating value for f_i function
def f(x, z, inner_points):
    A, c = initialize_variables(x)
    value = 0
    for ind in range(len(z)):
        if ind in inner_points:
            value += max(res(z[ind], A, c), 0) ** 2
        else:
            value += min(res(z[ind], A, c), 0) ** 2
    return value


# Calculating the gradient function of f_i
def df(x, z, inner_points):
    # Initializing variables for dr/dA and dr/dc
    dr_dA = np.zeros((2,2))
    dr_dc = np.zeros(2)

    A, c = initialize_variables(x)

    for ind in range(len(z)):
        # Calculates dr/dA for this data point
        grad_A = np.matmul((z[ind]-c).T, z[ind]-c)
        # Calculates dr/dc for this data point
        grad_c = -2 * np.matmul(A, z[ind]-c)
        # Calculates the residual for this data point
        ri = res(z[ind], A, c)

        # Checks if data point is an inner point
        if ind in inner_points and ri > 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c

        # Checks if data point is an outer point
        if ind not in inner_points and ri < 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c
    
    return np.array([dr_dA[0,0], dr_dA[0,1], dr_dA[1,1], dr_dc[0], dr_dc[1]])


# A function that makes f and df easier to call
def problem_definition(x, z, inner_points):
    f_func = lambda x: f(x, z, inner_points)
    df_func = lambda x: df(x, z, inner_points)
    return f_func, df_func



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



