import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import PCA
import utils

""" Task2.2: In this script, we compute eigenfunctions of the Laplace Beltrami operator on the
“swiss roll” manifold.  We need functions defined in utils.py for this script.
"""

# TODO: Generate swiss roll dataset
# Simply generate dataset with make_swiss_roll, random state is set randomly as 608 to make the result reproducible
X, t = make_swiss_roll(n_samples=5000, noise=0, random_state=608, hole=False)

# Calculate 3 PCs, and export explained variance ratio
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
print("Principal Components:", pca.components_)
print("Explained variance ratio:", pca.explained_variance_ratio_)

# TODO: Visualize data-set
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t)
plt.title("Swiss Roll Dataset Visualization")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.grid(True)

# TODO: Use function diffusion_map() defined in utils to compute first ten eigenfunctions (corresponding to 10 largest eigenvalues) of the Laplace Beltrami operator on the “swiss roll” manifold

n_eig_vals = 10
eigenvalues, eigenvectors = utils.diffusion_map(X, n_eig_vals)

# TODO: Plot the first non-constant eigenfunction φ1 against the other eigenfunctions
phi_1 = eigenvectors[:, 1]

# We will plot phi_1 against phi_2 to phi_10, 9 subplots in total
num_plots = n_eig_vals - 1
grid_size = int(np.ceil(np.sqrt(num_plots)))

fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
fig.suptitle(
    "Plot of other Eigenfunctions ($\phi_l$) against $\phi_1$", fontsize=16
)
axes = axes.flatten()

for i in range(num_plots):
    l = i + 2
    phi_l = eigenvectors[:, l]
    ax = axes[i]
    # Use a scatter plot
    scatter = ax.scatter(phi_1, phi_l, c=t, s=5)
    ax.set_title(f"$\phi_{{{l}}}$ against $\phi_1$")
    ax.set_xlabel("$\phi_1$")
    ax.set_ylabel(f"$\phi_{{{l}}}$")
    ax.grid(True)

fig.subplots_adjust(hspace=0.5, wspace=0.3)
plt.show()
plt.pause(100)
