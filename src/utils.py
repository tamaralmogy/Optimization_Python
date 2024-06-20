import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os

def plot_contours(f, x_limits, y_limits, title="Loss Function Contour", paths=None, path_labels=None, filename=None):
    x = np.linspace(x_limits[0], x_limits[1], 500)
    y = np.linspace(y_limits[0], y_limits[1], 500)
    X, Y = np.meshgrid(x, y)
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(16, 12))
    levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 30) if "Linear" not in title else np.linspace(np.min(Z), np.max(Z), 30)
    cp = plt.contour(X, Y, Z, levels=levels, cmap='cividis')
    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    if paths is not None and path_labels is not None:
        for path, label in zip(paths, path_labels):
            plt.plot(path[:, 0], path[:, 1], marker='o' if "Newton" in label else 'x', label=label, alpha=0.8)
        plt.legend()

    if filename is not None:
        plt.savefig(filename)
    plt.show()

def plot_function_values(methods, title='Function Value vs. Iterations', filename=None):
    plt.figure()
    for method in methods:
        plt.plot(method['values'], label=method['name'])
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Function Value')
    plt.legend()
    if filename is not None:
        plt.savefig(filename)
    plt.show()


def plot_qp_result(result, path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the feasible region (triangle)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan  # Feasible region where Z >= 0

    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='magenta')

    # Plot the path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-')

    # Plot the final point
    ax.scatter(result[0], result[1], result[2], color='blue', marker='x')

    ax.set_title('QP Path and Final Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


def plot_lp_result(result, path):
    fig, ax = plt.subplots()

    # Define the range for x and y
    x = np.linspace(0, 2.5, 400)
    y1 = -x + 1
    y2 = np.ones_like(x)
    y3 = np.zeros_like(x)

    # Plot the feasible region
    ax.fill_between(x, np.maximum(y1, y3), y2, where=(x <= 2), color='yellow', alpha=0.5, label='Feasible Region')

    # Plot the constraint lines for better visualization
    ax.plot(x, y1, 'b-', label='$y \geq -x + 1$')
    ax.plot(x, y2, 'g-', label='$y \leq 1$')
    ax.plot([2, 2], [0, 1], 'm-', label='$x \leq 2$')
    ax.plot(x, y3, 'c-', label='$y \geq 0$')

    # Plot the path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'ro-', label='Path')

    # Plot the final point
    ax.scatter(result[0], result[1], color='blue', marker='x', s=100, label='Final Point')

    # Set titles and labels
    ax.set_title('LP Path and Final Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.5)
    ax.grid(True)
    ax.legend()
    plt.show()

def plot_obj_vs_iter(outer_obj_values):
    plt.plot(outer_obj_values, 'bo-')
    plt.title('Objective Value vs Outer Iteration')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)
    plt.show()
    
def plot_qp_result(result, path, filename=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the feasible region (triangle)
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = 1 - X - Y
    Z[Z < 0] = np.nan  # Feasible region where Z >= 0

    ax.plot_surface(X, Y, Z, alpha=0.5, rstride=100, cstride=100, color='magenta')

    # Plot the path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], 'ro-')

    # Plot the final point
    ax.scatter(result[0], result[1], result[2], color='blue', marker='x')

    ax.set_title('QP Path and Final Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_lp_result(result, path, filename=None):
    fig, ax = plt.subplots()

    # Define the range for x and y
    x = np.linspace(0, 2.5, 400)
    y1 = -x + 1
    y2 = np.ones_like(x)
    y3 = np.zeros_like(x)

    # Plot the feasible region
    ax.fill_between(x, np.maximum(y1, y3), y2, where=(x <= 2), color='yellow', alpha=0.5, label='Feasible Region')

    # Plot the constraint lines for better visualization
    ax.plot(x, y1, 'b-', label='$y \geq -x + 1$')
    ax.plot(x, y2, 'g-', label='$y \leq 1$')
    ax.plot([2, 2], [0, 1], 'm-', label='$x \leq 2$')
    ax.plot(x, y3, 'c-', label='$y \geq 0$')

    # Plot the path
    path = np.array(path)
    ax.plot(path[:, 0], path[:, 1], 'ro-', label='Path')

    # Plot the final point
    ax.scatter(result[0], result[1], color='blue', marker='x', s=100, label='Final Point')

    # Set titles and labels
    ax.set_title('LP Path and Final Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.5)
    ax.grid(True)
    ax.legend()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close(fig)

def plot_obj_vs_iter(outer_obj_values, filename=None):
    plt.plot(outer_obj_values, 'bo-')
    plt.title('Objective Value vs Outer Iteration')
    plt.xlabel('Outer Iteration')
    plt.ylabel('Objective Value')
    plt.grid(True)

    if filename:
        plt.savefig(filename)
    else:
        plt.show()
    plt.close()