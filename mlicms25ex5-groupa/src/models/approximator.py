import abc
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin

from . import utils


@dataclass
class BaseApproximator(abc.ABC, RegressorMixin):
    """Abstract base class for approximators based on basis functions.
    Parameters
    ----------
    rcond : float
        Cut-off ratio passed to the least-squares solver for numerical
        stability.
    Attributes
    ----------
    _weights : npt.ArrayLike
        The learned coefficient matrix. This stores the weight
        parameters after fit() method is called.
    Notes
    -----
    This is an abstract class and should not be instantiated directly.
    Use one of its subclasses, such as `LinearApproximator` or
    `RBFApproximator`.
    """

    rcond: float = 0.0
    _weights: npt.ArrayLike = None

    @abc.abstractmethod
    def construct_basis(self, x: npt.ArrayLike) -> npt.ArrayLike:
        pass

    def fit(self, x: npt.ArrayLike, y: npt.ArrayLike) -> "BaseApproximator":
        """Fit the model to the data.
        Parameters
        ----------
        x : npt.ArrayLike
            The input data points, shape (N, D), where N is the number of data
            points and D is the dimensionality of each point.
        y : npt.ArrayLike
            The target values, shape (N, 1) or (N,).    
        Returns
        -------
        BaseApproximator
            The fitted model.
        """

        basis_matrix = self.construct_basis(x)

        self._weights = utils.linear_solve(basis_matrix, y, rcond=self.rcond)

        return self

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Predict the target values for the given input data.
        Parameters
        ----------
        x : npt.ArrayLike
            The input data points, shape (N, D), where N is the number of data
            points and D is the dimensionality of each point.
        Returns
        -------
        npt.ArrayLike
            The predicted target values, shape (N, 1) or (N,).
        """
        if self._weights is None:
            raise ValueError(
                "The model weights are not set. "
                "You should call the fit(...) method first."
            )
        
        basis_matrix = self.construct_basis(x)
        return basis_matrix @ self._weights

    def get_params(self, deep=True):
        return {"rcond": self.rcond}

    def set_params(self, **params):
        self.rcond = params["rcond"]
        return self


class LinearApproximator(BaseApproximator):
    """Approximate a function using a linear model of the form f(x) = Ax.
    Parameters
    ----------
    rcond : float
        Cut-off ratio passed to the least-squares solver for numerical
        stability.
    Attributes
    ----------
    _weights : npt.ArrayLike
        The learned coefficient matrix, shape (d_in, d_out). This stores the weight
        parameters after fit() method is called.
    """
    def construct_basis(self, x):
        """Construct the basis for the linear model.
        Parameters
        ----------
        x : npt.ArrayLike
            The input data points, shape (N, D), where N is the number of data
            points and D is the dimensionality of each point.
        Returns
        -------
        npt.ArrayLike
            The basis matrix, shape (N, D), where N is the number of data points
            and D is the dimensionality of each point.
        """
        return np.asarray(x)

@dataclass
class RBFApproximator(BaseApproximator):
    """Approximate a function using a linear combination of radial basis functions.
    Parameters
    ----------
    L : int
        The number of radial basis functions to use. The centers are chosen randomly
        from the training data.
    eps : float
        The bandwidth of the Gaussian RBFs.
    rcond : float
        Cut-off ratio passed to the least-squares solver for numerical
        stability.
    Attributes
    ----------
    _weights : npt.ArrayLike
        The learned coefficient matrix, shape (L, d_out). This stores the weight
        parameters after fit() method is called.
    _centers : npt.ArrayLike
        The positions of the RBF centers, shape (L, d_in). These are
        randomly selected from the data.
    """
    L: int = 10
    eps: float = 1e-1
    _centers: npt.ArrayLike = None

    def get_centers(self, x: npt.ArrayLike, seed: int = 42):
        """Randomly get L centers from the dataset.
        Parameters
        ----------
        x : npt.ArrayLike
            The input data points, shape (N, D), where N is the number of data
            points and D is the dimensionality of each point.
        Returns
        -------
        npt.ArrayLike
            An array containing L selected center points, shape (L, D), where L is the number of
            center points and D is the dimensionality of each point.
        """

        if self.L > x.shape[0]:
            raise ValueError(
                "The number of centers should be "
                "less than the number of data points. "
                f"Got: L={self.L}, N={x.shape[0]}."
            )
        rng = np.random.RandomState(seed)
        indices = rng.choice(x.shape[0], size=self.L, replace=False)

        return x[indices, :]

    def construct_basis(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Construct the basis for the radial basis functions model.
        Parameters
        ----------
        x : npt.ArrayLike
            The input data points, shape (N, D), where N is the number of data
            points and D is the dimensionality of each point.
        Returns
        -------
        npt.ArrayLike
            The basis matrix for the RBF model, shape (N, M)， where N is the 
            number of data points and M is the number of rbf centers.
        """
        # If only one data point is given, turn it into a 2D array.
        if x.ndim == 1:
            x = x[None, :]
        # Only compute centers once.
        if self._centers is None:
            self._centers = self.get_centers(x)

        return utils.rbf(x, self._centers, self.eps)


    def get_params(self, deep=True):
        return super().get_params(deep) | {"L": self.L, "eps": self.eps}

    def set_params(self, **params):
        self.L = params["L"]
        self.eps = params["eps"]
        return super().set_params(**params)
