import utils

""" Task 1.2: In this script, we apply principal component analysis to a racoon image. 
We need functions defined in utils.py for this script.
"""

# TODO: Load and resize the racoon image in grayscale
image = utils.load_resize_image()  # Load image using function from utils

# TODO: Compute Singular Value Decomposition (SVD) using utils.compute_svd()
U, S, V_t = utils.compute_svd(
    image
)  # Compute SVD using SVD computation function from utils

# TODO: Reconstruct images using utils.reconstruct_images
utils.reconstruct_images(
    U, S, V_t
)  # Reconstruct image using function from utils

# TODO: Compute the number of components where energy loss is smaller than 1% using utils.compute_num_components_capturing_threshold_energy()
# Compute how many components are needed to capture the threshold energy using function from utils
n_components = utils.compute_num_components_capturing_threshold_energy(S)
print(
    f"Number of components to retain 99% of the energy: {n_components}"
)  # Print amount of components
