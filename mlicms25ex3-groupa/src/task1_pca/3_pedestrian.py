import os
import numpy as np
import matplotlib.pyplot as plt
import utils

""" Task 1.3: In this script, we apply principal component analysis to pedestrian trajectory data. 
We need functions defined in utils.py for this script.
"""

# TODO: Load trajectory data in data_DMAP_PCA_Vadere.txt. (Hint: You may need to use a space as delimiter)
data_file_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "data",
        "data_DMAP_PCA_vadere.txt",
    )
)  # Get file path
data = np.loadtxt(data_file_path, delimiter=" ")  # Load data

# TODO: Center the data by subtracting the mean
centered_data = utils.center_data(
    data
)  # Center data using centering function from utils

# TODO: Extract positions of pedestrians 1 and 2
pedestrian_1 = data[
    :, 0:2
]  # Get first two columns of the data for pedestrian 1 (x1, y1)
pedestrian_2 = data[
    :, 2:4
]  # Get the third and fourth columns of the data for pedestrian 2 (x2, y2)

# TODO: Visualize trajectories of first two pedestrians (Hint: You can optionally use utils.visualize_traj_two_pedestrians() )
# Use given function to visualize trajectories (with all principal components)
utils.visualize_traj_two_pedestrians(
    pedestrian_1, pedestrian_2, ("Pedestrian Trajectories", "x", "y")
)
plt.show()  # Show plot, it's missing in the utils function

# TODO: Compute SVD of the data using utils.compute_svd()
U, S, V_t = utils.compute_svd(
    centered_data
)  # Compute SVD using SVD computation function from utils

# TODO: Reconstruct data by truncating SVD using utils.reconstruct_data_using_truncated_svd()
reconstructed_data = utils.reconstruct_data_using_truncated_svd(
    U, S, V_t, 2
)  # Compute data with the first two principal components

# TODO: Visualize trajectories of the first two pedestrians in the 2D space defined by the first two principal components
recon_pedestrian_1 = reconstructed_data[
    :, 0:2
]  # Get first two columns of the RECONSTRUCTED data for pedestrian 1 (x1, y1)
recon_pedestrian_2 = reconstructed_data[
    :, 2:4
]  # Get the third and fourth columns of the RECONSTRUCTED data for pedestrian 2 (x2, y2)
# Use given function to visualize trajectories (with the first two principal components)
utils.visualize_traj_two_pedestrians(
    recon_pedestrian_1,
    recon_pedestrian_2,
    ("Reconstructed Trajectories (2 PCs)", "x", "y"),
)
plt.show()  # Show plot, it's missing in the utils function

# TODO: Answer the questions in the worksheet with the help of utils.compute_cumulative_energy(), utils.compute_num_components_capturing_threshold_energy()
pc1_energy = utils.compute_energy(
    S, 1
)  # Get energy of the first principal component using energy computation function from utils
pc2_energy = utils.compute_energy(
    S, 2
)  # Get energy of the second principal component using energy computation function from utils
cumulative_energy_2 = utils.compute_cumulative_energy(
    S, 2
)  # Get the cumulative energy of the first two principal components
n_components_90 = utils.compute_num_components_capturing_threshold_energy(
    S, 0.90
)  # Get how many components are needed to reach a threshold of 0.9 energy

print(
    f"Energy in 1st principal component: {pc1_energy:.2f}%"
)  # Print energy of the first principal component
print(
    f"Energy in 2nd principal component: {pc2_energy:.2f}%"
)  # Print energy of the second principal component
print(
    f"Cumulative energy in first 2 components: {cumulative_energy_2:.2f}%"
)  # Print cumulative energy of the first two principal components
print(
    f"Number of components for >90% energy: {n_components_90}"
)  # Print how many components are needed for a threshold of 0.9 energy
