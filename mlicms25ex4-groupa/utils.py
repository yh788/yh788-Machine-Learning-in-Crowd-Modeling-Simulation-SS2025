from typing import List, Tuple, Type
from numpy.typing import ArrayLike
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.figure
from scipy.integrate import solve_ivp

from dynamical_system import *

# TODO: add all the util functions here, e.g. plot functions.
# You should use the same plot function for the bifurcation plots in task 2 and 4


def plot_bifurcation_1d(
    system_class: Type[DynamicalSystem],
    par_range: ArrayLike,
    x_range: List[int],
    ax: plt.Axes,
    par_name: str = "alpha",
) -> matplotlib.figure.Figure:
    """Plot a bifurcation diagram for a 1D dynamical system.
    Args:
        system_class (Type[DynamicalSystem]):
            The class of the dynamical system to simulate.
        par_range (ArrayLike):
            Range of parameter values to evaluate.
        x_range (List[int]):
            Range of state variable values.
        ax (plt.Axes):
            Axes object to plot on.
        par_name (str):
            Name of the parameter for labeling.
    Returns:
        matplotlib.figure.Figure: The generated plot figure.
    """
    stable_points = []
    unstable_points = []
    par_stable_values = []
    par_unstable_values = []

    for par in par_range:
        system = system_class(par=par, xrange=[x_range])

        # we find the fixed points by finding roots
        # can use a grid search and then a root finder or analytical
        # for such specific tasks, we can find analytical solutions

        if system_class == Task21:
            if par >= 0:
                x_fixed = [np.sqrt(par), -np.sqrt(par)]
            else:
                x_fixed = []
        elif system_class == Task22:
            if par >= 3:
                x_fixed = [np.sqrt((par - 3) / 2), -np.sqrt((par - 3) / 2)]
            else:
                x_fixed = []
        else:
            # for general
            pass

        for x in x_fixed:
            # determine stability by checking derivative
            # f' < 0 stable
            # f' > 0 unstable

            # defining h
            h = 1e-5
            # numerical derivative
            dfdx = (system.fun(0, x + h) - system.fun(0, x - h)) / (2 * h)

            if dfdx < 0:
                stable_points.append(x)
                par_stable_values.append(par)
            elif dfdx > 0:
                unstable_points.append(x)
                par_unstable_values.append(par)

    # Plotting
    ax.plot(par_stable_values, stable_points, "b.", label="Stable")
    ax.plot(par_unstable_values, unstable_points, "r.", label="Unstable")
    ax.set_xlabel(f"${par_name}$")
    ax.set_ylabel("$x^*$")
    ax.legend()
    ax.set_title(f"Bifurcation Diagram for {system_class.__name__}")
    ax.grid(True)
    return ax.get_figure()


def euler_solver(
    system: DynamicalSystem, x0: ArrayLike, t_span: List[float], dt: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Explicit Euler method solver for dynamical systems
    Args:
        system: DynamicalSystem instance
        x0: initial state
        t_span: [t_start, t_end]
        dt: time step
    Returns:
        t: time points
        x: solution array, shape (n_steps, dim_state)
    """
    t = np.arange(t_span[0], t_span[1], dt)
    x = np.zeros((len(t), len(x0)))
    x[0] = x0

    for i in range(1, len(t)):
        x[i] = x[i - 1] + dt * np.array(system.fun(t[i - 1], x[i - 1]))

    return t, x


def phase_portrait_andronov_hopf(system: Task3, ax=None):
    """
    Plots the phase portrait of the Andronov-Hopf system
    Args:
        system: Task3 instance
        ax: matplotlib axes (optional)
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    X1, X2 = system.X

    # Calculate vector field
    dX = np.zeros_like(system.X)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            dX[0][i, j], dX[1][i, j] = system.fun(0, [X1[i, j], X2[i, j]])

    # Plot streamlines
    ax.streamplot(X1, X2, dX[0], dX[1], density=2)
    ax.set_aspect(1)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Phase Portrait (α = {system.par:.2f})")
    ax.grid(True)

    return ax


def plot_trajectory(system, x0):
    """Plot trajectory for dynamical system
    Args:
        system: dynamical system instance
        x0: initial state
    Returns:
        matplotlib.axes.Axes: The axes object with the plot
    """
    t, sol = euler_solver(system, x0, [0, 10], 0.001)

    ax = plt.gca()
    ax.plot(sol[:, 0], sol[:, 1], "--", label=f"Start at {x0}")
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title(f"Phase Portrait (α = {system.par})")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal")

    return ax


def plot_cusp_bifurcation(
    x_range: List[float], alpha2_range: List[float], num_points: int = 100
) -> plt.Axes:
    """
    Visualize the cusp bifurcation surface where dx/dt = 0
    for the system dx/dt = α₁ + α₂x - x³

    Args:
        x_range: [x_min, x_max] range for x values
        alpha2_range: [alpha2_min, alpha2_max] range for α₂ values
        num_points: number of points to sample in each dimension

    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    # Create grid of points
    x = np.linspace(x_range[0], x_range[1], num_points)
    alpha2 = np.linspace(alpha2_range[0], alpha2_range[1], num_points)
    X, A2 = np.meshgrid(x, alpha2)

    # Calculate α₁ from equilibrium condition dx/dt = 0
    A1 = X**3 - A2 * X

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Plot surface
    surf = ax.plot_surface(A1, A2, X, cmap="viridis")

    # Labels and title
    ax.set_xlabel("α₁")
    ax.set_ylabel("α₂")
    ax.set_zlabel("x")
    ax.set_title("Cusp Bifurcation Surface")

    # Add colorbar
    plt.colorbar(surf)
    return ax


def plot_phase_portrait(system: Task1, ax: plt.Axes, title: str = ""):
    """Plots the phase portrait of a 2D linear system defined by dx/dt = A x"""
    X1, X2 = system.X
    U, V = system.fun(0, np.array([X1, X2]))

    ax.streamplot(X1, X2, U, V, color="lightgray")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.set_aspect("equal")


def plot_cusp_projection():
    """Plot the 2D projection of the cusp bifurcation set into the parameter space."""
    x = np.linspace(-2, 2, 400)
    alpha2 = 3 * x**2
    alpha1 = -2 * x**3

    plt.figure(figsize=(8, 6))
    plt.plot(alpha2, alpha1, "b-", label="Fold curve (bifurcation set)")
    plt.xlabel(r"$\alpha_2$")
    plt.ylabel(r"$\alpha_1$")
    plt.title("Cusp Bifurcation: 2D Projection into Parameter Space")
    plt.grid(True)
    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)
    plt.legend()
    plt.axis("equal")
    plt.show()


def plot_logistic_map_bifurcation(r_min, r_max, n_itr, n_plot, n_r):
    """Plot a bifurcation diagram for logistic map system.
    Args:
        r_min(float): lowest possible r value
        r_max(float): highest possible r value
        n_itr(int): number of iterations before plotting
        n_plot(int): number of iterations used for plotting
        n_r(int): number of r values under calculation
    Returns:
        matplotlib.axes.Axes: The plot axes
    """
    r = np.linspace(r_min, r_max, n_r)

    # Initialization
    x = 0.5 * np.ones(n_r)
    r_points_list = []
    x_points_list = []

    # Iterate to let the system settle on the attractor
    for i in range(n_itr):
        x = r * x * (1 - x)

    # Iterate and collect points for plotting
    for i in range(n_plot):
        x = r * x * (1 - x)
        r_points_list.append(r)
        x_points_list.append(x)

    # Convert lists to numpy arrays after the loop finishes
    r_points = np.concatenate(r_points_list)
    x_points = np.concatenate(x_points_list)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(r_points, x_points, ",k", alpha=0.15)
    ax.set_title("Bifurcation Diagram of Logistic Map", fontsize=16)
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    return ax


def plot_lorenz(system: Task42, x_0: ArrayLike, t_end: float, n_points: int = 30000):
    """Solves and plots a 3D trajectory of the Lorenz system.
    Args:
        system (Task42): An instance of the dynamical system to be solved.
        x_0 (ArrayLike): The initial state vector [X_0, Y_0, Z_0].
        t_end (float): The end time for the simulation, starting from t=0.
        n_points (int): The number of points to evaluate and plot along the trajectory.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            - A 1D array of time points of shape
            - A 2D array representing the trajectory of shape
    """
    t_span = [0, t_end]
    t_eval = np.linspace(0, t_end, n_points)

    solution = solve_ivp(
        system.fun, t_span, x_0, args=(), dense_output=True, t_eval=t_eval
    )
    # Unpack the trajectory data
    x_t, y_t, z_t = solution.y

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(x_t, y_t, z_t, lw=0.5)
    ax.set_title("Lorenz Attractor", fontsize=16)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.zaxis.labelpad = 0
    ax.view_init(elev=None, azim=-75)

    # Export trajectory data points
    return solution.t, solution.y
