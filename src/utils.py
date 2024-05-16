import matplotlib.pyplot as plt
import numpy as np
def plot_contours(f, x_limits, y_limits, title="Loss Function Contour", paths=None, path_labels=None, filename=None):
    # Create grid / plot space based on limits input
    x = np.linspace(x_limits[0], x_limits[1], 500)
    y = np.linspace(y_limits[0], y_limits[1], 500)
    X, Y = np.meshgrid(x, y)

    # Get function values for points on plot space / grid
    Z = np.array([[f(np.array([X[i, j], Y[i, j]]))[0] for j in range(X.shape[1])] for i in range(X.shape[0])])

    plt.figure(figsize=(16, 12))

    # Handle linear differently without log scale/spacing for contours
    if "Linear" in title:
        levels = np.linspace(np.min(Z), np.max(Z), 30)
    else: # not linear, log scale/spacing for visualization
        levels = np.logspace(np.log10(np.min(Z)), np.log10(np.max(Z)), 30)

    # plot contour lines
    cp = plt.contour(X, Y, Z, levels=levels, cmap='cividis')

    plt.colorbar(cp)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Plot optimization paths per function
    if paths is not None and path_labels is not None:
        for path, label in zip(paths, path_labels):
            if "Newton" in label:
                plt.plot(path[:, 0], path[:, 1], marker='o', label=label, alpha=0.8)
            else:
                plt.plot(path[:, 0], path[:, 1], marker='x', label=label, markersize=10)
        plt.legend()

    if filename is not None:
        plt.savefig(filename)
    plt.show()


# Function to plot function values over iterations
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