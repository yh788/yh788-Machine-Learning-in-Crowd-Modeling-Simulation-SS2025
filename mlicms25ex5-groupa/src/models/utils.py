import numpy as np
import numpy.typing as npt
from sklearn.base import RegressorMixin
from sklearn.model_selection import GridSearchCV, KFold


def rbf(x: npt.ArrayLike, centers: npt.ArrayLike, eps: float) -> npt.ArrayLike:
    """Computes the radial basis functions for the provided set of inputs.

    Parameters:
    -----------
    x : npt.ArrayLike, shape (N, d)
        N input points.
    centers : npt.ArrayLike, shape (M, d)
        M centers of the radial basis functions.
    eps : float
        The scaling factor of the radial basis functions.

    Returns:
    --------
    npt.Arraylike, shape (N, M)
        Evaluations of the radial basis functions.
    """
    # Calculate squared distance by expanding the equation
    x_sq_norms = np.sum(x ** 2, axis=1)[:, np.newaxis]
    centers_sq_norms = np.sum(centers ** 2, axis=1)
    dot_product = np.dot(x, centers.T)
    # Broadcasting
    sq_dists = x_sq_norms - 2 * dot_product + centers_sq_norms
    # Return rbf function
    return np.exp(-sq_dists / (eps ** 2))


def linear_solve(x: npt.ArrayLike, y: npt.ArrayLike, rcond: float) -> npt.ArrayLike:
    """Computes the least-square solution for the given input and target values.

    Parameters:
    -----------
    x : npt.ArrayLike, shape (N, d_in)
        Feature matrix.
    y : npt.ArrayLike, shape (N, d_out)
        Target values.
    rcond : float
        Regualarization for the solver.

    Returns:
    --------
    npt.Arraylike, shape (d_in, d_out)
        Solution of the least-square problem.
    """
    # Solution, residuals, rank, singular_values = np.linalg.lstsq(x, y, rcond=rcond), the solution is [0] of tuple
    solution = np.linalg.lstsq(x, y, rcond=rcond)[0]
    return solution


def compute_mse(y_true: npt.ArrayLike, y_pred: npt.ArrayLike) -> float:
    """Computes the mean squared error between the true and predicted values.

    Parameters:
    -----------
    y_true : npt.ArrayLike, shape (N, d_out)
        True target values.
    y_pred : npt.ArrayLike, shape (N, d_out)
        Predicted target values.

    Returns:
    --------
    float
        Mean squared error between the true and predicted values.
    """
    # Calculate squared error 
    sq_err = (y_true - y_pred)**2
    # Calculate MSE
    mse = np.mean(np.sum(sq_err, axis=1))
    # Return MSE
    return mse


def grid_search(
    parameters: dict[str, list],
    model: RegressorMixin,
    data: tuple[npt.ArrayLike, npt.ArrayLike],
    scoring: str = "neg_mean_squared_error",
    n_splits: int = 5,
    **kwargs,
) -> GridSearchCV:
    """Performs a grid search using sklearn.model_selection.GridSearchCV.

    After fitting, the best parameters and the best model can be accessed
    with cv.best_params_ and cv.best_estimator_.

    Parameters:
    -----------
        parameters : dict[str, list]
            Dictionary of possible hyperparameters for the model.
        model : RegressorMixin
            Model to be evaluated. The model should have parameters as attributes.
        data, (x, y) : tuple[npt.ArrayLike, npt.ArrayLike]
            Training data for the model.
        scoring : str, default="accuracy"
            Scoring metric for the model.
        n_splits : int, default=5
            Number of splits for cross-validation.
        **kwargs:
            Additional keyword arguments to be passed to model.fit(...).

    Returns:
    --------
    GridSearchCV
        GridSearchCV object with the results of the grid search.
    """
    # Optional: Implement the grid search with GridSearchCV
    # Unpack the training data
    x, y = data

    # Create a cross-validation splitter.
    cv_splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Instantiate the GridSearchCV object.
    grid_search_cv = GridSearchCV(
        estimator=model,
        param_grid=parameters,
        scoring=scoring,
        cv=cv_splitter,
        verbose=1,
        n_jobs=-1
    )

    # Execute the grid search.
    grid_search_cv.fit(x, y, **kwargs)
    return grid_search_cv
