import numpy as np

def quadratic_function(x, Q, hessian_needed=True):
    f = 0.5 * np.dot(x.T, np.dot(Q, x))
    g = np.dot(Q, x)
    H = Q if hessian_needed else None
    return f, g, H

# Q as identity matrix (contour lines are circles)
Q_i = np.array([[1, 0], [0, 1]])

# Q with different scaling (contour lines are axis aligned ellipses)
Q_ii = np.array([[1, 0], [0, 100]])

# Q as a rotated ellipse (contour lines are rotated ellipses)
A = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
D = np.array([[100, 0], [0, 1]])
Q_iii = np.dot(A.T, np.dot(D, A))

def rosenbrock(x, hessian_needed=False):
    x1, x2 = x
    f = 100 * (x2 - x1**2)**2 + (1 - x1)**2
    dfdx1 = -400 * x1 * (x2 - x1**2) - 2 * (1 - x1)
    dfdx2 = 200 * (x2 - x1**2)
    g = np.array([dfdx1, dfdx2])
    h = None
    if hessian_needed:
        H11 = 1200 * x1**2 - 400 * x2 + 2
        H22 = 200
        H12 = H21 = -400 * x1
        h = np.array([[H11, H12], [H21, H22]])
    return f, g, h

def linear_function(x, a, hessian_needed=False):
    f = np.dot(a.T, x)
    g = a
    h = np.zeros((len(a), len(a))) if hessian_needed else None
    return f, g, h

# Define a non-zero vector 'a'
a = np.array([2, 3])

def exponential_function(x, hessian_needed=False):
    x1, x2 = x
    f = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
    dfdx1 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) - np.exp(-x1 - 0.1)
    dfdx2 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)
    g = np.array([dfdx1, dfdx2])
    h = None
    if hessian_needed:
        H11 = np.exp(x1 + 3*x2 - 0.1) + np.exp(x1 - 3*x2 - 0.1) + np.exp(-x1 - 0.1)
        H22 = 9 * np.exp(x1 + 3*x2 - 0.1) + 9 * np.exp(x1 - 3*x2 - 0.1)
        H12 = H21 = 3 * np.exp(x1 + 3*x2 - 0.1) - 3 * np.exp(x1 - 3*x2 - 0.1)
        h = np.array([[H11, H12], [H21, H22]])
    return f, g, h
