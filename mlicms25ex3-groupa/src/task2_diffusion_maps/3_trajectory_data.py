import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import utils

"""Task2.3: In this script, we demonstrate the similarity of Diffusion Maps and Fourier analysis using a periodic dataset.
We need functions defined in utils.py for this script.
"""

# TODO: Load data and apply algorithm
data = np.loadtxt('../../data/data_DMAP_PCA_vadere.txt')
print(data.shape)
# TODO: Compute eigenfunctions associated to the largest eigenvalues using function diffusion_map() implemented in utils.py

eigenvalues, eigenvectors = utils.diffusion_map(data, n_eig_vals=5)
# φ0 is constant, we analyse φ1 to φ5
phi_1 = eigenvectors[:, 1]
phi_2 = eigenvectors[:, 2]
phi_3 = eigenvectors[:, 3]
phi_4 = eigenvectors[:, 4]
phi_5 = eigenvectors[:, 5]

# TODO: Plot the first non-constant eigenfunction φ1 against the other eigenfunctions
# Use 4 sub figures to plot φ1 to φ2, φ3, φ4 and φ5
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.title('Plot of $\phi_2$ against $\phi_1$')
plt.scatter(phi_1, phi_2, s=10)
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_2$')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.title('Plot of $\phi_3$ against $\phi_1$')
plt.scatter(phi_1, phi_3, s=10)
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_3$')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.title('Plot of $\phi_4$ against $\phi_1$')
plt.scatter(phi_1, phi_4, s=10)
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_4$')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.title('Plot of $\phi_5$ against $\phi_1$')
plt.scatter(phi_1, phi_5, s=10)
plt.xlabel('$\phi_1$')
plt.ylabel('$\phi_5$')
plt.grid(True)

plt.tight_layout()
plt.show()
plt.pause(100)
