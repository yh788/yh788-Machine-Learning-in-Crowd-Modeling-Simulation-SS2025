import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.integrate as spi
from sklearn.base import RegressorMixin

try:
    from ..models.approximator import BaseApproximator
except ImportError:
    # Fallback for when running from notebooks
    from models.approximator import BaseApproximator


class BaseDynamicalSystem(abc.ABC):
    """Abstract base class for all dynamical systems.

    This class defines the fundamental interface for a dynamical system, which
    is characterized by its state and a rule for how that state evolves over
    time. This is an abstract class and is not meant to be instantiated directly.
    Subclasses must implement the `_get_tangent` method.
    """

    @abc.abstractmethod
    def _get_tangent(self, t: float, state: npt.ArrayLike) -> npt.ArrayLike:
        """Computes the tangent corresponding to the dynamical system.

        Parameters:
        -----------
            t: float
                Time at which the tangent is evaluated.
            state: npt.ArrayLike, shape (d,)
                State of the dynamical system.

        Returns:
        --------
            npt.ArrayLike, shape (d,)
                Tangent vector corresponding to the state.
        """
        tangent = np.zeros_like(state)
        return tangent
        

    def simulate(
        self, state: npt.ArrayLike, t_max: float, n_evals: int | None = None
    ) -> npt.ArrayLike:
        """Simulate the dynamical system for a given time.

        Parameters:
        -----------
            state: npt.ArrayLike, shape (d,)
                Initial state of the system.
            t_max: float
                Time to simulate the system.
            n_evals: int | None, default = None
                Number of timesteps to evaluatte at. The timestamps are
                equally spaced between 0 and t. If n_evals is None, then
                only the final state is returned.

        Returns:
        --------
            npt.ArrayLike, shape (n_evals, d)
                States at evaluated timestamps.
        """

        if n_evals is None:
            t_eval = [t_max]  # Only evaluate at final time
        else:
            t_eval = np.linspace(0, t_max, n_evals)

        sol = spi.solve_ivp(
            self._get_tangent,
            (0, t_max),
            state,
            t_eval=t_eval,
            method="RK45",
            vectorized=False,
        )
        if sol.success:
            if n_evals is None:
                # If n_evals is None, return the last state as a 1D array
                # with shape (d,).
                return sol.y[:, -1]

            # if n_evals == 1:
            #     # If n_evals is None or 1, return the last state as a 2D array
            #     # with shape (1, d).
            #     return sol.y[:, -1].reshape(1, -1)
            else:
                # If n_evals is specified, return the states at the evaluated timestamps
                # as a 2D array with shape (n_evals, d).
                return sol.y.T  # Transpose to get shape (n_evals, d)
            # return sol.y.T
        else:
            raise RuntimeError(
                f"Simulation failed with message: {sol.message}. "
                "Please check the initial state and the system parameters."
            )
        
    def batch_simulate(
        self, states: npt.ArrayLike, t_max: float, n_evals: int = None
    ) -> npt.ArrayLike:
        """Simulate the dynamical system for multiple initial states.
        Parameters:
        -----------
            states: npt.ArrayLike, shape (N, d)
                Initial states of the system, where N is the number of states
                and d is the dimension of the state.
            t_max: float
                Time to simulate the system.
            n_evals: int | None, default = None
                Number of timesteps to evaluate at. The timestamps are
                equally spaced between 0 and t. If n_evals is None, then
                only the final state for each initial state is returned.    
        Returns:
        --------
            npt.ArrayLike, shape (N, n_evals, d)
                States at evaluated timestamps for each initial state.      
        """ 

        states = np.asarray(states)
        if states.ndim == 1:
            states = states[None, :]  # Ensure states has shape (N, d)

        results = []
        for state in states:
            result = self.simulate(state=state, t_max=t_max, n_evals=n_evals)
            # Always ensure result has 3 dimensions (N, n_evals, d) for consistency
            if n_evals is None:
                # result shape is (d,), reshape to (1, 1, d) to indicate 1 eval at 1 timestep
                result = result.reshape(1, 1, -1)
            elif n_evals == 1:
                # result shape is (1, d), reshape to (1, 1, d) for consistency
                result = result.reshape(1, 1, -1)
            else:
                # result shape is (n_evals, d), reshape to (1, n_evals, d)
                result = result.reshape(1, *result.shape)
            results.append(result)

        # Stack results along the first axis (N dimension)
        results_array = np.vstack(results) if n_evals is None else np.concatenate(results, axis=0)
        
        # Squeeze unnecessary dimensions if n_evals is None
        if n_evals is None:
            results_array = results_array.squeeze(axis=1)  # Shape becomes (N, d)

        return results_array



@dataclass
class LorenzSystem(BaseDynamicalSystem):
    """Represents the classic Lorenz chaotic dynamical system.

    This class implements the Lorenz attractor, a system of three coupled,
    non-linear ordinary differential equations.
    The system is defined by the equations:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """

    sigma: float = 10
    rho: float = 28
    beta: float = 8 / 3

    def _get_tangent(self, _, state: npt.ArrayLike) -> npt.ArrayLike:
        state_array = np.asarray(state)
        x, y, z = state_array
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        tangent = np.array([dx, dy, dz])
        return tangent


@dataclass
class TrainableDynamicalSystem(BaseDynamicalSystem):
    """A dynamical system whose vector field is learned from data.

    This class acts as a bridge between observed data and a continuous
    dynamical model. It wraps an approximator object and uses it to learn
    the underlying vector field of the system's state.

    Once fitted, it can be used to simulate trajectories.
    """

    approximator: BaseApproximator

    def _infer_tangent(
        self, x0: npt.ArrayLike, x1: npt.ArrayLike, delta_t: float
    ) -> npt.ArrayLike:
        """Approximates the tangent map of a dynamical system from snapshots.

        Parameters:
        -----------
            x0, x1: npt.ArrayLike, shape (N, d)
                N snapshots of the dynamical system.
            delta_t: float
                Time step between the snapshots.

        Returns:
        --------
            npt.ArrayLike, shape (N, d)
                N approximated tangent vectors.
                The i-th row of the output array is the approximation of the
                ith position.
        """
        return (x1 - x0) / delta_t
    
    
    def fit(
        self, x0: npt.ArrayLike, x1: npt.ArrayLike, delta_t: float
    ) -> "TrainableDynamicalSystem":
        """Fit the approximator to the data.
        Parameters:
        -----------
            x0, x1: npt.ArrayLike, shape (N, d)
                N snapshots of the dynamical system.
            delta_t: float
                Time step between the snapshots.    
        Returns:
        --------
            TrainableDynamicalSystem
                The fitted dynamical system.
        """
        tangent_vectors = self._infer_tangent(x0, x1, delta_t)
        self.approximator.fit(x0, tangent_vectors)
        return self

    def _get_tangent(self, _, state: npt.ArrayLike) -> npt.ArrayLike:
        tangent = self.approximator.predict(state.reshape(1, -1)).flatten()
        if tangent.ndim != 1:
            raise ValueError(
                "The output of the approximator must be a 1D array."
                f"Got {tangent.ndim}D array instead."
            )
        return tangent


class GridSearchDynamicalSystem(RegressorMixin):
    """Wrapper for TrainableDynamicalSystem.

    This class wraps TrainableDynamicalSystem to make it compatible with
    the GridSearchCV.
    """

    def __init__(self, delta_t, approximator_cls, **approximator_params):
        self.delta_t = delta_t
        approximator = approximator_cls(**approximator_params)
        self.system = TrainableDynamicalSystem(approximator)

    def fit(self, x0, x1):
        self.system.fit(x0, x1, self.delta_t)
        return self

    def predict(self, x):
        return self.system.batch_simulate(x, self.delta_t, n_evals=2)[:, -1, :]

    def get_params(self, deep=True):
        approximator_cls = self.system.approximator.__class__
        approximator_params = self.system.approximator.get_params(deep)
        return approximator_params | {
            "delta_t": self.delta_t,
            "approximator_cls": approximator_cls,
        }

    def set_params(self, **params):
        self.__init__(**params)
        return self
