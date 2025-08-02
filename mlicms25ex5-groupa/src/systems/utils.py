from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

###############################################################################
# Data loading
###############################################################################


def load_function(filename: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Loads evaluations of f(x) from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data. The data should have
        d + 1 columns, where the first d columns are the input data
        and the last column is the target data.

    Returns:
    --------
    tuple[npt.ArrayLike, npt.ArrayLike]
        A tuple of input data of shape (N, d) and target data of
        shape (N,), where N is the number of rows in the file.
    """
    data = np.loadtxt(filename)
    x, y = data[:, :-1], data[:, -1]
    return x, y


def load_vectorfield(filename: str) -> tuple[npt.ArrayLike, npt.ArrayLike]:
    """Loads evaluations of a dynamical system from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data. The data should
        have 2*d columns, where the first d columns are the
        x0 coordinates and the last d columns are the x1
        coordinates.

    Returns:
    --------
    tuple[npt.ArrayLike, npt.ArrayLike], shape [(N, d), (N, d)]
        x0 and x1 positions for the dynamical system.
    """
    data = np.loadtxt(filename)
    if data.shape[1] % 2 != 0:
        raise ValueError("The number of columns in the file should be even.")
    d = data.shape[1] // 2
    x0, x1 = data[:, :d], data[:, d:]
    return x0, x1


def load_manifold(filename: str) -> npt.ArrayLike:
    """Loads coordinates of manifold points from a file.

    Parameters:
    -----------
    filename : str
        Path to the file containing the data.

    Returns:
    --------
    npt.ArrayLike
        Array of the shape (N, d), where N is the neumber of row
        in the file and d is the dimensionality of the embedding space.
    """
    data = np.loadtxt(filename)
    return data


###############################################################################
# Plotting. You can change these functions to adjust your plots.
###############################################################################


def plot_function(
    ax: plt.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    y_pred: npt.ArrayLike | None = None,
    scatter: bool = False,
    xlabel: str = "x",
    ylabel: str = "y",
):
    """Plots a 1D function and its prediction.

    Parameters:
    -----------
        ax: plt.Axes
            A matplotlib axis to plot on.
        x: npt.ArrayLike
            A 1D array of arguments.
        y: npt.ArrayLike
            A 1D array of function values.
        y_pred: npt.ArrayLike | None = None
            An optional 1D array of predicted values.
        scatter: bool = False
            If true, use ax.scatter instead of ax.plot.
        xlabel: str = "x"
            The label of the x-axis.
        ylabel: str = "y"
            The label of the y-axis.
    """
    if scatter:
        plotter = partial(ax.scatter, s=1)
    else:
        plotter = ax.plot
        # Sort the arguments.
        argsort = np.argsort(x.flatten())
        x = x[argsort]
        y = y[argsort]
        if y_pred is not None:
            y_pred = y_pred[argsort]

    # Plot points as dots so no overlapping with predictions happen
    # Adjust marker size and colors for distinction
    plotter(x, y, 'o', label='True labels', markersize=4, color='blue')

    if y_pred is not None:
        plotter(x, y_pred, label='Predicted labels', markersize=10, color='orange')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


def plot_vector_field(ax: plt.Axes, x0: npt.ArrayLike, x1: npt.ArrayLike):
    """Plots a 2D vector field.

    Parameters:
    -----------
    ax: plt.Axes
        A matplotlib axis to plot on.
    x0: npt.ArrayLike
        Initial states of the dynamical system.
    x1: npt.ArrayLike
        Advanced states.
    """
    ax.scatter(x0[:, 0], x0[:, 1], s=1)
    ax.scatter(x1[:, 0], x1[:, 1], s=1)
    ax.quiver(
        x0[:, 0],
        x0[:, 1],
        x1[:, 0] - x0[:, 0],
        x1[:, 1] - x0[:, 1],
        angles="xy",
        scale_units="xy",
        scale=1,
    )
    ax.set_xlabel("x[0]")
    ax.set_ylabel("x[1]")

def plot_2D_trajectory(ax, traj, **kwargs):
    """Line plot of a 2-D trajectory.
    traj : (N,2) array with columns x(t), y(t)
    """
    ax.plot(traj[:,0], traj[:,1], **kwargs)
    ax.set_xlabel("x[0]"); ax.set_ylabel("x[1]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().set_aspect('equal', adjustable='box') 


def plot_3D_trajectory(
    ax: plt.Axes,
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
):
    """Plots a 3D trajectory of a dynamical system

    Parameters:
    -----------
    ax: plt.Axes
        A matplotlib axis to plot on.
    x: npt.ArrayLike
        The x-coordinate of the trajectory
    y: npt.ArrayLike
        The y-coordinate of the trajectory
    z: npt.ArrayLike
        The z-coordinate of the trajectory
    c: npt.ArrayLike | None, default = None
        If present, this array is used to speficy colors of the points.
    """
    ax.scatter(x, y, z, s=1, c=c)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


###############################################################################
