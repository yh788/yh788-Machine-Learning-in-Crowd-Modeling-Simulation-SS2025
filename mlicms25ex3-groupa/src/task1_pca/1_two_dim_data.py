import utils
import numpy as np
import matplotlib.pyplot as plt
import os

""" Task 1.1: In this script, we apply principal component analysis to two-dimensional data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load the dataset from the file pca_dataset.txt
data_file_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "pca_dataset.txt"
    )
)  # Get file path
data = np.loadtxt(data_file_path)  # Load data

# TODO: Compute mean of the data
mean = np.mean(data, axis=0)  # Compute mean using numpy mean

# TODO: Center data
centered_data = utils.center_data(
    data
)  # Center data using centering function from utils

# TODO: Compute SVD
U, S, V_t = utils.compute_svd(
    centered_data
)  # Compute SVD using SVD computation function from utils

# TODO: Plot principal components
# Plot settings
plt.figure(figsize=(8, 8))
plt.scatter(data[:, 0], data[:, 1], alpha=0.7, label="Data")
plt.axhline(0, color="black", lw=0.5)
plt.axvline(0, color="black", lw=0.5)

scale = 2.5
for i in range(2):  # For the first two principal components
    direction = V_t[i] * scale  # Scale the direction (V component of SVD)
    # Plot direction with labels
    plt.quiver(
        *mean,
        *direction,
        angles="xy",
        scale_units="xy",
        scale=1,
        color=["r", "g"][i],
        label=f"PC{i + 1}",
    )

# Plot settings
plt.title("PCA on Dataset")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.axis("equal")
plt.legend()
plt.grid(True)
plt.show()

# TODO: Analyze the energy captured by the first two principal components using utils.compute_energy()
pc1_energy = utils.compute_energy(
    S, 1
)  # Get energy of the first principal component using energy computation function from utils
pc2_energy = utils.compute_energy(
    S, 2
)  # Get energy of the second principal component using energy computation function from utils

print(
    f"Energy in PC1: {pc1_energy:.2f}%"
)  # Print energy of the first principal component
print(
    f"Energy in PC2: {pc2_energy:.2f}%"
)  # Print energy of the second principal component
