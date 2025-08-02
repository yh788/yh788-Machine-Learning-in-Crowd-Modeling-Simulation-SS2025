import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from utils import *

""" This script is used to train and test the VAE.
"""

############################################################
## Subtasks 3.3 & 3.4 in the worksheet ##
############################################################
# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the MNIST dataset: Train and test
train_dataset = MNIST(".", train=True, download=True, transform=ToTensor())
test_dataset = MNIST(".", train=False, download=True, transform=ToTensor())


# TODO: Set the learning rate, batch size and no. of epochs
learning_rate = 0.001
batch_size = 128
num_epochs = 50
hidden_dim = 256  # Number of neurons in the hidden layer


# TODO: Create an instance of Dataloader for train_dataset using torch.utils.data, use appropriate batch size, keep shuffle=True.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# TODO: Create an instance of Dataloader for test_dataset using torch.utils.data, use appropriate batch size, keep shuffle=False.
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# TODO: Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
input_dim = 784  # 28x28 images flattened
latent_dim = 2  # Latent space dimension for the first part

# TODO: Instantiate the VAE model with a latent dimension of 2, using the utility function instantiate_vae() from utils
vae_model = instantiate_vae(input_dim, latent_dim, hidden_dim, device)

# TODO. Set up an appropriate optimizer from torch.optim with an appropriate learning rate
optimizer = optim.Adam(vae_model.parameters(), lr=learning_rate)

plots_at_epochs = [1, 5, 25, 50]  # generate plots at epoch numbers


# TODO: Compute train and test losses by performing the training loop using the utility function training_loop() from utils
train_losses, test_losses = training_loop(
    vae_model,
    optimizer,
    train_loader,
    test_loader,
    num_epochs,
    plots_at_epochs,
    device,
)

# TODO: Plot the loss curve using the utility function plot_loss() from utils
plot_loss(train_losses, test_losses)


##############################################################
##### Subtask 3.5 in the worksheet #######
##############################################################
# Create the VAE model with a latent dimension of 32
# TODO: Repeat the above steps with the latent dimension of 32 and compute train and test losses
print("\nTraining VAE with latent dimension of 32...")
latent_dim_32 = 32  # Latent space dimension for the second part
vae_model_32 = instantiate_vae(input_dim, latent_dim_32, hidden_dim, device)
optimizer_32 = optim.Adam(vae_model_32.parameters(), lr=learning_rate)
train_losses_32, test_losses_32 = training_loop(
    vae_model_32,
    optimizer_32,
    train_loader,
    test_loader,
    num_epochs,
    plots_at_epochs,
    device,
)

# TODO: (5a) Compare 15 generated digits using the utility function reconstruct_digits()
# Done in training loop

# TODO: (5b) Plot the loss curve using the utility function plot_loss()
plot_loss(train_losses_32, test_losses_32)
