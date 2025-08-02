import scipy as sp
import scipy.spatial
from scipy.linalg import eigh
import numpy as np

""" This script contains the implementation of the diffusion map algorithm. 
"""


def create_distance_matrix(X, max_distance=200):
    """Compute a sparse distance matrix using scipy.spatial.KDTree. Set max_distance as 200.

    Args:
        X (npt.NDArray[np.float]): Data matrix.
        max_distance (int, optional): Computes a distance matrix leaving as zero any distance greater than max_distance. Defaults to 200.

    Returns:
        npt.NDArray[np.float]: Distance Matrix (Hint: You may have to use (D).toarray()!, output shape = (np.shape(D)[0], np.shape(D)[0]))
    """
    # TODO: Implement method
    # Hints: using scipy.spatial.KDTree, set max_distance as 200, you may have to use .toarray() to the array you are returning!)
    tree = sp.spatial.KDTree(X)

    # Compute sparse distance matrix and return
    sp_dist_matrix = tree.sparse_distance_matrix(tree, max_distance)
    distance_matrix = sp_dist_matrix.toarray()

    return distance_matrix


def set_epsilon(p, distance_matrix):
    """Set scalar epsilon as 'p' % of the diameter of the dataset.
    (Step 2 of the algorithm mentioned in the worksheet.)

    Args:
        p (np.float64): percentage.
        distance_matrix (npt.NDArray[np.float]): Distance matrix.

    Returns:
        np.float64: returns epsilon.
    """
    # TODO: Implement method (Hint: p is a float between 1-100, you have to divide by 100)
    # Diameter of the dataset is the largest distance of distance matrix
    epsilon = (p / 100) * np.max(distance_matrix)
    return epsilon


def create_kernel_matrix(D, eps):
    """Create the Kernel matrix.

    Args:
        D (npt.NDArray[np.float]): Distance matrix
        eps (np.float64): epsilon.

    Returns:
        npt.NDArray[np.float]: Kernel matrix. (output shape = (np.shape(D)[0], np.shape(D)[0]))
    """
    # TODO: Form the kernel matrix W (Step 3 of the algorithm from the worksheet)
    W = np.exp(-(D**2) / eps)
    # TODO: Form the diagonal normalization matrix (Step 4 of the algorithm from the worksheet)
    # Diagonal elements are the sum value of each row
    diag_element = W.sum(axis=1)
    P = np.diag(diag_element)
    # TODO: Normalize W to form the kernel matrix K (Step 5 of the algorithm from the worksheet)
    P_inv = np.linalg.inv(P)
    K = P_inv @ W @ P_inv
    return K


def diffusion_map(X, n_eig_vals=5):
    """Implementation of the diffusion map algorithm.
        Please refer to the algorithm in the worksheet for the following.
        The step numbers in the following refer to the steps of the algorithm in the worksheet.

    Args:
        X (npt.NDArray[np.float]): Data matrix (each row represents one data point)
        n_eig_vals (int, optional): The number of eigenvalues and eigenvectors of the Laplace-Beltrami operator defined on the manifold close to the data to be computed. Default is 10.

    Returns:
        tuple(npt.NDArray[np.float], npt.NDArray[np.float]): eigenvalues, eigenvector of the Laplace-Beltrami operator
        output shapes: (n_eig_vals + 1, ), (np.shape(X)[0], n_eig_vals + 1)
    """

    # TODO: Compute distance matrix. Use method create_distance_matrix(..) defined in this script. (Step 1 from the algorithm in the worksheet)
    D = create_distance_matrix(X, max_distance=200)

    # TODO: Use function set_epsion(.., ..) defined in this script to set epsilon to 5% of the diameter of the dataset (Step 1 from the algorithm in the worksheet).
    # Epsilon = 5% and therefore p = 5
    eps = set_epsilon(5, D)

    # TODO: Form Kernel matrix K. Use function create_kernel_matrix(.., ..) defined in this script. (Steps 3-5 from the algorithm in the worksheet)
    K = create_kernel_matrix(D, eps)

    # TODO: Form the diagonal normalization matrix (Step 6 from the algorithm in the worksheet)
    q_diag_element = K.sum(axis=1)
    Q = np.diag(q_diag_element)

    # TODO: Form symmetric matrix T_hat (Step 7 from the algorithm in the worksheet)
    # Take each element on the diagonal line of Q and to the power of -1/2
    q_inv_sqrt = q_diag_element ** (-1 / 2)
    Q_inv_sqrt = np.diag(q_inv_sqrt)
    T_hat = Q_inv_sqrt @ K @ Q_inv_sqrt

    # TODO: Find the L + 1 largest eigenvalues and the corresponding eigenvectors of T_hat (Step 8 from the algorithm in the worksheet)
    # Use eigh function to get eigenvalues and eigenvectors in ascending order, then take L+1 values from the end
    L = n_eig_vals
    T_hat_eva, T_hat_eve = eigh(T_hat)
    T_hat_largest_eva = T_hat_eva[-(L + 1) :]
    T_hat_largest_eve = T_hat_eve[:, -(L + 1) :]

    # TODO: Compute the eigenvalues of T_hat^(1/ε) in DESCENDING ORDER (Hint: You can use np.flip(..))!! (Step 9 from the algorithm in the worksheet)
    # Flip the squared eigenvalues to make in descending order. Then take square root for final eigen values.
    T_hat_eps_eva_sq = T_hat_largest_eva ** (1 / eps)
    T_hat_eps_eva_sq_des = np.flip(T_hat_eps_eva_sq)
    lambda_value = np.sqrt(T_hat_eps_eva_sq_des)
    # TODO: Compute the eigenvectors of the matrix T (Hint: You can use np.flip(..) with an appropriate axis) (Step 10 from the algorithm in the worksheet)
    # Make eigenvectors also in descending order, corresponding to eigenvalues.
    T_hat_largest_eve_des = np.flip(T_hat_largest_eve, axis=1)
    T_eve = Q_inv_sqrt @ T_hat_largest_eve_des

    return lambda_value, T_eve
