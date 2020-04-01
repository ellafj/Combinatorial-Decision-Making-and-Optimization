import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

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
    #print('zi', zi)
    #print('c', c)
    #print('A', A)
    return (zi - c).T @ A @ (zi - c) - 1


# Calculating value for f_i function
def f(x, z, inner_points):
    A, c = initialize_variables(x)
    value = 0
    for ind in range(len(z)):
        ri = res(z[ind], A, c)
        if (inner_points[ind] == 1) and (ri > 0):
            value += ri ** 2
        if (inner_points[ind] == 0) and (ri < 0):
            value += ri ** 2
    return value

# Calculating the gradient function of f_i
def df(x, z, inner_points):
    # Initializing variables for dr/dA and dr/dc
    dr_dA = np.zeros((2,2))
    dr_dc = np.zeros(2)

    A, c = initialize_variables(x)

    for ind in range(len(z)):
        # Calculates dr/dA for this data point
        grad_A = np.outer((z[ind]-c), (z[ind]-c)) #np.matmul((z[ind]-c).T, z[ind]-c)
        # Calculates dr/dc for this data point
        grad_c = -2 * (A @ (z[ind]-c)) #-2 * np.matmul(A, z[ind]-c)
        # Calculates the residual for this data point
        ri = res(z[ind], A, c)

        # Checks if data point is an inner point
        if inner_points[ind] == 1 and ri > 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c

        # Checks if data point is an outer point
        if inner_points[ind] == 0 and ri < 0:
            dr_dA += 2 * ri * grad_A
            dr_dc += 2 * ri * grad_c

    return np.array([dr_dA[0,0], dr_dA[0,1], dr_dA[1,1], dr_dc[0], dr_dc[1]])

# Generates the randomly scattered points
def generate_points(x, scale = 1e-1, size = 500):
    A, c = initialize_variables(x)
    all_points = np.random.uniform(-2, 2, ((size, 2))) #np.random.multivariate_normal(c, scale1 * np.linalg.inv(A), size = size)
    inner_points = []
    n = len(all_points)
    x = np.random.normal(0, scale, n)
    y = np.random.normal(0, scale, n)

    for i in range(n):
        if res(all_points[i], A, c) <= 0:
            inner_points.append(i)

    all_points[:, 0] = all_points[:, 0] + x
    all_points[:, 1] = all_points[:, 1] + y

    return all_points, inner_points

def new_generate_points(x, size, perturb = False):
    A, c = initialize_variables(x)
    area = 5
    all_points = area * (2 * np.random.rand(size, 2) - np.ones((2,)))
    inner_points = np.zeros(len(all_points))
    for ind in range(len(all_points)):
        if res(all_points[ind], A, c) <= 0:
            inner_points[ind] = 1

    if perturb:
        for ind in range(len(all_points)):
            all_points[ind] += np.random.uniform(-1, 1, size = (1, 2))[0]

    return all_points, inner_points


# Plots the points
def plot_points(z, inner_points):
    for ind in range(len(z)):
        if inner_points[ind] == 1:
            color = "C0"
        else:
            color = "C3"
        plt.scatter(z[ind,0], z[ind,1], color=color)


# Plot the ellipses
def plot_ellipses(x_k):
    A, c = initialize_variables(x_k)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    r = np.zeros((100, 100))
    for i in range(len(x)):
        for j in range(len(y)):
            r[i, j] = res(np.array([X[i, j], Y[i, j]]), A, c)
    plt.contour(X, Y, r, levels = [0], alpha=1.0)


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



x0 = [2, 1, 1, 0, 1] #[1/1.2**2, 0, 1/2**2, 0, 0] #[5, 1, 5, 0, 0] #[3,1,3,0,0]
z, inner = new_generate_points(x0, 150, perturb=True)
y1 = 0.1
y2 = 100
x = [3, 2, 2, 0, 0]
plot_points(z, inner)
plot_ellipses(x)
new_x = BFGS(x, z, inner, y1, y2)
plot_ellipses(new_x)
plt.show()


# Not the problem
def primal_barrier(x0, z, w, gamma_small, gamma_big,
                  n=5, beta=1.0, tol=1e-7, problem=1, silent=False):
    '''Incorporating the primal logarithmic barrier in order to solve
    the constrained case using BFGS. The ojective function is assumed
    to be of form f(x), and is modified by constrained_functions.P to
    be of the form P(x, beta) = f(x) - beta * sum(c(x))

    Arguments:
        x0 {Array} -- Inital guess
        z {array} -- coordinates of points
        w {array} -- labels of corresponding points
        gamma_small {float} -- lower constant in constraints
        gamma_big {float} -- upper constant in constraints

    Keyword Arguments:
        n {int} -- number of components of x (default: {5})
        beta {float} -- initial barrier value (default: {1.0})
        tol {float} -- stopping criteria for BFGS (default: {1e-5})
        problem {int} -- which problem to optimize.
                         Valid choices are 1 and 2. (default: {1})
        silent {bool} -- if True, the function will not print
                         a progress report as the algorithm runs
    '''
    # Create array to store computed ellipses
    ellipses = x0

    I = np.eye(n)
    x_k = x0
    iteration = 0
    while True:
        # begin BFGS loop
        H_k = I
        df_k = df_constrained(x_k, z, w, gamma_small, gamma_big, beta)
        while la.norm(df_k) > tol:
            df_k = df_constrained(x_k, z, w, gamma_small, gamma_big, beta)
            f_k = f_constrained(x_k, z, w, gamma_small, gamma_big, beta)
            p_k = - H_k @ df_k
            alpha = backtracking(f_k, df_k, x_k, z, w, gamma_small, gamma_big, beta, p_k)
            prev_xk = x_k
            x_k = x_k + alpha * p_k
            s_k = x_k - prev_xk
            prev_dfk = df_k
            df_k = df_constrained(x_k, z, w, gamma_small, gamma_big, beta)
            y_k = df_k - prev_dfk
            prev_Hk = H_k
            # Check that Hessian update is okay
            if s_k.dot(y_k) > 0:
                rho = 1 / s_k.dot(y_k)
                H_k = ((I - rho * np.outer(s_k, y_k)) @ H_k @ (I - rho * np.outer(y_k, s_k))
                      + rho * np.outer(s_k, s_k))
            if alpha < 1e-7:
                iteration += 1
                break
            iteration += 1
        # end BFGS loop

        # Check accuracy
        if beta < tol:
            z_k = beta / c(x_k, gamma_small, gamma_big)
            KKT = df_k - (c_jacobian(x_k, gamma_small, gamma_big).T).dot(z_k)
            print(f'*** Algorithm complete ***')
            print(f'The 2-norm of the vector computed as the KKT condition is {la.norm(KKT)}.')
            print('The final results of the algorithm are as follows:')
            print(f'Objective function with barrier value {f_k}.')
            print(f'x_k: {x_k}')
            ellipses = np.vstack([ellipses, x_k])
            return ellipses

        else:
            beta = beta / 2
            print('beta', beta)
            ellipses = np.vstack([ellipses, x_k])
            if not silent:
                print(f'Downscaling at iteration {iteration}.' \
                       'β = {np.format_float_scientific(beta, precision=3)}.')
                print(f'Objective function with barrier value {f_k}.')
                print(f'x_k: {x_k}')
                print('---------------------------------------------------------')
def backtrack(f_k, df_k, x, z, w, gamma_small, gamma_big, beta, p, alpha=1, rho=0.5, c1=0.2):
    while True:
        # Make sure we stay in the feasible set
        if np.any(c(x + alpha * p, gamma_small, gamma_big) == np.nan) or np.any(c(x + alpha * p, gamma_small, gamma_big) < 0):
            alpha = rho * alpha
        elif f_constrained(x + alpha*p, z, w, gamma_small, gamma_big, beta) <= f_k + c1 * alpha * df_k.dot(p):
            return alpha
        else:
            alpha = rho * alpha
def df_constrained_new(x, z, w, gamma_small, gamma_big, beta):
    dP_array = df(x, z, w)
    c_array = c(x, gamma_small, gamma_big)
    dc_array = dc(x, gamma_small, gamma_big)
    for i in range(5):
        dP_array -= beta /c_array[i] * dc_array[i]

    return dP_array
def f_constrained_new(x):
    return 0
def c_jacobian_new(x, gamma_small, gamma_big):
    A = np.zeros((5, 5))
    A[0, 0] = 1
    A[1, 0] = -1
    A[2, 2] = 1
    A[3, 2] = -1
    A[4] = np.array([
        0.5 * np.sqrt(x[2] / x[0]),
        - x[1] / np.sqrt(gamma_small**2 + x[1]**2),
        0.5 * np.sqrt(x[0] / x[2]),
        0,
        0
    ])
    return A
def df_new(x, z, w):
    A, c = initialize_variables(x)
    g1 = np.zeros((2,2))
    g2 = np.zeros(2)
    for i in range(len(z)):
        ri = np.transpose(z[i]-c)@A@(z[i]-c) - 1
        g11 = np.outer((z[i]-c), (z[i]-c))
        g22 = -2*(A@(z[i]-c))

        if (w[i] == 1) and (ri > 0):
            g1 += 2*ri*g11
            g2 += 2*ri*g22

        if (w[i] == 0) and (ri < 0):
            g1 += 2*ri*g11
            g2 += 2*ri*g22

    assert g1.shape == (2, 2), "Wrong"
    assert len(g2) == 2, "Wrong"

    df = np.array([g1[0][0], g1[0][1], g1[1][1], g2[0], g2[1]])

    return df
