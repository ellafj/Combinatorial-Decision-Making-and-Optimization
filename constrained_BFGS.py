import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

#######################################################################
#################### Functions needed for plotting ####################
#######################################################################

# Generates the data points for the problem
def generate_points(x, size, perturb = False):
    # Initializing variables
    A, c = initialize_ellipse(x)
    area = 5
    all_points = area * (2 * np.random.rand(size, 2) - np.ones((2,)))
    inner_points = np.zeros(len(all_points))

    # Marks relevant points as inner points
    for ind in range(len(all_points)):
        if res(all_points[ind], A, c) <= 0:
            inner_points[ind] = 1

    # Perturbs the data points
    if perturb:
        for ind in range(len(all_points)):
            all_points[ind] += np.random.uniform(-1, 1, size = (1, 2))[0]

    return all_points, inner_points


# Plots the data points
def plot_points(z, inner_points):
    for ind in range(len(z)):
        if inner_points[ind] == 1:
            color = "C0"
        else:
            color = "C3"
        plt.scatter(z[ind,0], z[ind,1], color=color)


# Plot the ellipses
def plot_ellipses(x_k, color='blue'):
    A, c = initialize_ellipse(x_k)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    r = np.zeros((100, 100))
    for i in range(len(x)):
        for j in range(len(y)):
            r[i, j] = res(np.array([X[i, j], Y[i, j]]), A, c)
    plt.contour(X, Y, r, levels = [0], alpha=1.0, color=color)


#######################################################################
################## Functions needed for calculations ##################
#######################################################################

# Initializing variables needed for ellipse in correct
def initialize_ellipse(x):
    # x is an array with five entries, where the first three are A11, A12=A21 and A22.
    # The last two entries is coordinate of center of the ellipse
    A = np.zeros((2, 2))
    A[0] = [x[0], x[1]]
    A[1] = [x[1], x[2]]
    vec = np.array([x[3], x[4]])
    return A, vec


# Calculates the residual for f.
# Returns value <= 0 if points is contained in ellipse
# and value > 0 if outside of ellipse
def res(zi, A, c):
    return (zi - c).T @ A @ (zi - c) - 1


# Calculating value for f function
def f(x, z, inner_points):
    # Initializing variables
    A, c = initialize_ellipse(x)
    value = 0

    for ind in range(len(z)):
        # Calculates residual
        ri = res(z[ind], A, c)

        # Checks if points are wrongly classified and adds penalties
        if inner_points[ind] == 1 and ri > 0:
            value += ri ** 2
        if inner_points[ind] == 0 and ri < 0:
            value += ri ** 2
    return value


# Calculating the gradient function of f_i
def df(x, z, inner_points):
    # Initializing variables for dr/dA and dr/dc
    dr_dA = np.zeros((2,2))
    dr_dc = np.zeros(2)

    A, c = initialize_ellipse(x)

    for ind in range(len(z)):
        # Calculates dr/dA for this data point
        grad_A = np.outer((z[ind]-c), (z[ind]-c))
        # Calculates dr/dc for this data point
        grad_c = -2 * (A @ (z[ind]-c))
        # Calculates the residual for this data point
        ri = res(z[ind], A, c)

        # # Checks if points are wrongly classified and adds penalties
        if inner_points[ind] == 1 and ri > 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c
        if inner_points[ind] == 0 and ri < 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c

    return np.array([dr_dA[0,0], dr_dA[0,1], dr_dA[1,1], dr_dc[0], dr_dc[1]])


# The constraint functions
def c(x, y1, y2):
    c = np.zeros(5)
    c[0] = x[0] - y1
    c[1] = y2 - x[0]
    c[2] = x[2] - y1
    c[3] = y2 - x[2]
    c[4] = np.abs(x[0] * x[2]) ** (1/2) - (y1 ** 2 + x[1] ** 2) ** (1/2)
    return c


# The gradient of the constraint functions
def dc(x, y1, y2):
    dc = np.zeros((5,5))
    dc[0] = [1, 0, 0, 0, 0]
    dc[1] = [-1, 0, 0, 0, 0]
    dc[2] = [0, 0, 1, 0, 0]
    dc[3] = [0, 0, -1, 0, 0]
    dc[4] = [
        0.5 * x[2] / np.sqrt(x[0] * x[2]),
        - x[1] / np.sqrt(y1**2 + x[1]**2),
        0.5 * x[0] / np.sqrt(x[0] * x[2]),
        0,
        0
    ]
    #[0.5 * (x[2]/x[0])**(1/2), -x[1]/(y1**2 + x[1]**2)**(1/2), 0.5 * (x[0]/x[2])**(1/2), 0, 0]
    return dc


# The Jacobian of the constraint functions
def c_jacobian(x, y1, y2):
    J_c = np.zeros((5,5))
    grad_c = dc(x, y1, y2)
    J_c[0] = grad_c[0]
    J_c[1] = grad_c[1]
    J_c[2] = grad_c[2]
    J_c[3] = grad_c[3]
    J_c[4] = grad_c[4]
    return J_c


# The new constrained function by the primal barrier method
def f_constrained(x, z, inner_points, y1, y2, beta):
    if np.any(c(x, y1, y2) < 0):
       return np.inf
    return f(x, z, inner_points) - beta * np.sum(np.log(c(x, y1, y2)))


# The gradient function of the new constrained function by the primal barrier method
# Will be of the form df_constrained = df - beta * sum(dc / c)
def df_constrained(x, z, inner_points, y1, y2, beta):
    grad_f = df(x, z, inner_points)
    c_array = c(x, y1, y2)
    dc_array = dc(x, y1, y2)

    for ind in range(len(grad_f)):
        grad_f -= beta * dc_array[ind]/c_array[ind]

    return grad_f


#######################################################################
########################### The BFGS method ###########################
#######################################################################


# Function for backtracking line search
def backtracking(f_k, df_k, x_k, z, inner_points, y1, y2, beta, p_k, alpha=1, rho=0.5, const=0.2):
    while True:
        # Checking if we are in the feasible set
        if np.any(c(x_k + alpha * p_k, y1, y2) == np.nan) or np.any(c(x_k + alpha * p_k, y1, y2) < 0):
            alpha = rho * alpha

        # Checking if Wolfe condition is fulfilled
        elif f_constrained(x_k + alpha * p_k, z, inner_points, y1, y2, beta) <= f_k + const * alpha * df_k.dot(p_k): #df_k.T @ p_k:
            break

        # Else we update our alpha value
        else:
            alpha = rho * alpha

    return alpha


# Function for iterating with BFGS method
def BFGS(x0, z, inner_points, y1, y2, beta=1, TOL=1e-7):
    # Initializing variables
    x_k = x0
    I = np.eye(len(x0))
    H_k = I
    k = 0

    while True:
        while la.norm(df_constrained(x_k, z, inner_points, y1, y2, beta)) > TOL:
            print("Iteration: ", k)
            f_k = f_constrained(x_k, z, inner_points, y1, y2, beta)
            df_k = df_constrained(x_k, z, inner_points, y1, y2, beta)
            p_k = - H_k @ df_k
            alpha_k = backtracking(f_k, df_k, x_k, z, inner_points, y1, y2, beta, p_k)
            x_new = x_k + alpha_k * p_k
            s_k = x_new - x_k
            df_new = df_constrained(x_new, z, inner_points, y1, y2, beta)
            y_k = df_new - df_k

            # A failsafe to ensure that Hessian update will be ok
            if s_k.dot(y_k) > 0:
                rho_k = 1 / s_k.dot(y_k)
                H_k = (I - rho_k * np.outer(s_k, y_k)) @ H_k @ (I - rho_k * np.outer(y_k, s_k))+ rho_k * np.outer(s_k, s_k)

            # Updating variables
            x_k = x_new
            k += 1

            if alpha_k < TOL:
                break

        if beta < TOL:
            print('\n initial ellipse', x0)
            print('final ellipse', x_k)
            return x_k
        else:
            print('beta', beta/2)
            beta = beta/2


#######################################################################
########################### Test functions  ###########################
#######################################################################


x0 = [2, 1, 1, 0, 1] #[1/1.2**2, 0, 1/2**2, 0, 0] #[5, 1, 5, 0, 0] #[3,1,3,0,0]
z, inner = generate_points(x0, 150, perturb=True)
y1 = 0.1
y2 = 100
x = [3, 2, 2, 0, 0]
plot_points(z, inner)
plot_ellipses(x)
new_x = BFGS(x, z, inner, y1, y2)
plot_ellipses(new_x)
plt.show()
