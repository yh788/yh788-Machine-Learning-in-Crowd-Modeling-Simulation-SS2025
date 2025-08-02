import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import utils

""" Task2.1: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Create a periodic dataset with the details described in the task-sheet
# Generate dataset via column_stack, to convert a tuple into an array
N = 1000
t_k = 2 * np.pi * np.arange(1, N + 1) / (N + 1)
X = np.column_stack((np.cos(t_k), np.sin(t_k)))
# Print(X.shape) to check data shape, should be (1000, 2)

# TODO: Visualize data-set
plt.scatter(X[:, 0], X[:, 1], s=10)
plt.title("Periodic Dataset Visualization")
plt.xlabel("x")
plt.ylabel("y")
plt.axis("equal")
plt.grid(True)

# TODO: Plot 5 eigenfunctions associated to the largest eigenvalues using the function diffusion_map() implemented in utils.py
n_eig_vals = 5
eigenvalues, eigenvectors = utils.diffusion_map(X, n_eig_vals)

# TODO: Plot 5 eigenfunctions
plt.figure(figsize=(10, 8))
plt.title(
    "5 Eigenfunctions Associated to the Largest Eigenvalues", fontsize=16
)
for i in range(1, n_eig_vals + 1):
    sort_indices = np.argsort(t_k)
    plt.plot(
        t_k[sort_indices],
        eigenvectors[sort_indices, i],
        label=f"$\phi_{{{i}}}$",
    )
plt.xlabel("$t_k$")
plt.ylabel("Eigenfunction Value")
plt.grid(True)
plt.legend()
plt.show()
plt.pause(100)
