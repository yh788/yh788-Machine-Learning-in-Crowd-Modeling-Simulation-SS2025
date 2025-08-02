from typing import List, Callable, Union
from numpy.typing import ArrayLike
import numpy as np
import scipy


class DynamicalSystem:
    """This class defines a dynamical system

    Methods
    --------
    solve_system(fun: Callable, init_state: ArrayLike, t_eval: ArrayLike):
        Solve the dynamical system
    """

    def __init__(self, xrange: List[List[int]], discrete: bool = False):
        """Parameters
        -------------
        xrange: List[List[int]]
            Is used to set the domain of the dynamical system.
            Specify start, end, number of points for each dimension
        discrete: bool = False
            If true, the dynamical system is time-discrete
        """
        self.xrange = xrange
        self.discrete = discrete

        self.X = self._set_grid_coordinates(xrange=self.xrange)

    def solve_system(self, init_state: ArrayLike, t_eval: ArrayLike) -> ArrayLike:
        """Solve the dynamical system

        Given the evolution rules, the initial point, and the time steps, we
        obtain the trajectory of the point. The solving method is different
        for time-discrete system, so two methods are implemented here.

        Parameters
        ----------
        init_state: ArrayLike
            Initial state of the system
        t_span : List[float]
            Time span of the integration (t_start, t_end)
        t_eval: ArrayLike
            Time steps of the trajectory

        Returns
        -------
        trajectory: ArrayLike
            Trajectory of the inital point in time
        """
        if not self.discrete:
            solution = scipy.integrate.solve_ivp(
                fun=self.fun,
                t_span=(t_eval[0], t_eval[-1]),
                y0=init_state,
                t_eval=t_eval,
            )
            return solution.y
        else:
            # TODO: implement solve_system method
            trajectory = np.zeros((len(t_eval), len(init_state)))
            trajectory[0] = init_state
            for i in range(1, len(t_eval)):
                trajectory[i] = self.fun(i, trajectory[i - 1])
            # Here we assume that the fun method returns the next state
            # given the current state and the time step
            solution = trajectory.T
            return solution

    def _set_grid_coordinates(self, xrange: List[List[int]]) -> List[np.ndarray]:
        """Set up the coordinates. For multidimensional cases use meshgrid"""
        match len(xrange):
            case 1:
                return np.linspace(xrange[0][0], xrange[0][1], xrange[0][2])
            case 2:
                X1, X2 = np.meshgrid(
                    np.linspace(xrange[0][0], xrange[0][1], xrange[0][2]),
                    np.linspace(xrange[1][0], xrange[1][1], xrange[1][2]),
                )
                return [X1, X2]
            case 3:
                X1, X2, X3 = np.meshgrid(
                    np.linspace(xrange[0][0], xrange[0][1], xrange[0][2]),
                    np.linspace(xrange[1][0], xrange[1][1], xrange[1][2]),
                    np.linspace(xrange[2][0], xrange[2][1], xrange[2][2]),
                )
                return [X1, X2, X3]


class Task1(DynamicalSystem):
    """This class defines a dynamical system for Task 1
    Attributes:
        xrange: List[List[int]]
            Is used to set the domain of the dynamical system.
            Specify start, end, number of points for each dimension
        discrete: bool = False
            If true, the dynamical system is time-discrete
        X: List[np.ndarray]
            Coordinates of the grid points in the domain
        t_eval: ArrayLike
            Time steps of the trajectory
        t_span: List[float]
            Time span of the integration (t_start, t_end)
        par: ArrayLike
            A 2D matrix that defines the evolution rule of the system
    Methods:
        fun(t: float, x: ArrayLike) -> ArrayLike:
            Defines the evolution rule of the system
    """

    def __init__(self, par, *args, **kwargs):
        """Hint: par is a 2D matrix here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """Hint: t is not necessarily used"""
        """TODO: implement dx/dt = Ax"""
        return np.einsum("ij,j...->i...", self.par, x)


class Task21(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """TODO: implement dx/dt = alpha - x**2"""
        """Hint: t is not necessarily used"""

        return self.par - x**2


class Task22(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """TODO: implement dx/dt = alpha - 2*x**2 - 3"""
        """Hint: t is not necessarily used"""

        return self.par - 2 * x**2 - 3


class Task3(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """
        Initialize Andronov-Hopf system
        Args:
            par: bifurcation parameter alpha
        """
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """
        Andronov-Hopf system ODEs
        dx₁/dt = αx₁ - x₂ - x₁(x₁² + x₂²)
        dx₂/dt = x₁ + αx₂ - x₂(x₁² + x₂²)
        """
        alpha = self.par
        return np.array(
            [
                alpha * x[0] - x[1] - x[0] * (x[0] ** 2 + x[1] ** 2),
                x[0] + alpha * x[1] - x[1] * (x[0] ** 2 + x[1] ** 2),
            ]
        )


class Task41(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a float/int here"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """TODO: implement the discrete system x_n+1 = r*x_n*(1-x_n)"""
        """Hint: t is not necessarily used"""

        return self.par * x * (1 - x)


class Task42(DynamicalSystem):
    def __init__(self, par, *args, **kwargs):
        """Hint: par is a list [sigma, beta, rho]"""
        super().__init__(*args, **kwargs)
        self.par = par

    def fun(self, t: float, x: ArrayLike) -> ArrayLike:
        """TODO: implement the Lorenz system"""
        """Hint: t is not necessarily used"""
        sigma, beta, rho = self.par

        # x[0] is x while x[1] is y and x[2] is z
        dxdt = sigma * (x[1] - x[0])
        dydt = x[0] * (rho - x[2]) - x[1]
        dzdt = x[0] * x[1] - beta * x[2]

        return [dxdt, dydt, dzdt]
