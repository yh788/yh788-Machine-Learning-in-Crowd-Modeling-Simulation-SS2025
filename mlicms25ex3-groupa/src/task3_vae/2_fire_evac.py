import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model import VAE
from utils import *
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from decimal import Decimal


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" This script is used to train and test the VAE on the fire_evac dataset. 
(Bonus: You can simulate trajectories with Vadere, for bonus points.) 
Not included in automated tests, as it's open-ended.
"""


# TODO: Download the FireEvac dataset
def download_fire_evac_dataset() -> dict:
    """Load data from the 'data' directory."""
    try:
        # the downloaded data exists here
        train_data = np.load("data/FireEvac_train_set.npy")
        test_data = np.load("data/FireEvac_test_set.npy")
        print(
            f"Loaded FireEvac dataset: train shape {train_data.shape}, test shape {test_data.shape}"
        )
        return {"train": train_data, "test": test_data}
    except FileNotFoundError:
        raise FileNotFoundError(
            "FireEvac dataset files not found. Ensure the files are run from outside src and accesible through data."
        )


data = download_fire_evac_dataset()
train_data = torch.from_numpy(data["train"]).float()
test_data = torch.from_numpy(data["test"]).float()

data_min = train_data.min(0, keepdims=True)[0]
data_max = train_data.max(0, keepdims=True)[0]


# TODO: Make a scatter plot to visualise it....as asked here
# so not putting in utils
def plot_fire_evac_data(data, title):
    """Create a scatter plot of the FireEvac data.

    Args:
        data (np.ndarray): FireEvac data from dataset.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[:, 0], data[:, 1], alpha=0.5)
    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid()
    plt.savefig(f"{title}.png")
    plt.show()


plot_fire_evac_data(train_data, title="FireEvac Train Data")
plot_fire_evac_data(test_data, title="FireEvac Test Data")

# TODO: Train a VAE on the FireEvac data

train_data_scaled = scale_data(train_data, data_min, data_max)
test_data_scaled = scale_data(test_data, data_min, data_max)

# dataloaders creation
train_dataset = TensorDataset(train_data_scaled)
test_dataset = TensorDataset(test_data_scaled)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# visualising initila data
print("Initial test dataset visualisation\n")
plt.figure(figsize=(8, 8))
plt.scatter(test_data[:, 0], test_data[:, 1])
plt.title("Initial FireEvac Test Dataset")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.axis("equal")
plt.savefig("fire_evac_initial_data.png")
plt.show()


class VAE_FireEvac(VAE):
    """VAE subclass with a Tanh activation for the FireEvac task."""

    def __init__(
        self,
        d_in: int,
        d_latent: int,
        d_hidden_layer: int,
        device: torch.device,
    ):
        super().__init__(d_in, d_latent, d_hidden_layer, device)
        # Override the decoder to use Tanh activation for the [-1, 1] range
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden_layer),
            nn.ReLU(),
            nn.Linear(d_hidden_layer, d_hidden_layer),
            nn.ReLU(),
            nn.Linear(d_hidden_layer, d_in),
            nn.Tanh(),
        )


# for training data
input_dim, latent_dim, hidden_dim = 2, 2, 64
learning_rate, epochs = 0.001, 200

vae = VAE_FireEvac(input_dim, latent_dim, hidden_dim, device).to(device)
optimizer = optim.Adam(vae.parameters(), lr=learning_rate)

plots_at_epochs = []

beta = 0.05
method = "MSE"
train_losses, test_losses = training_loop(
    vae,
    optimizer,
    train_loader,
    test_loader,
    epochs,
    plots_at_epochs,
    device,
    special_loss=True,
    beta=beta,
)

# TODO: Make a scatter plot of the reconstructed test set
# relative error test losses to train losses
relative_error = np.abs(
    np.array(test_losses) - np.array(train_losses)
) / np.array(train_losses)
print(
    "relative erorr",
    relative_error.min(),
    relative_error.max(),
    relative_error.mean(),
)
plot_loss(
    train_losses,
    test_losses,
    title=f"FireEvac VAE Loss Curve with {method} Loss function and $\\beta$ = {'%.2E' % Decimal(beta)}",
    save_path=f"Loss_curve_{method}_beta_{'%.2E' % Decimal(beta)}.png",
)

# TODO: Make a scatter plot of 1000 generated samples.
# Visualisation of results
vae.eval()
with torch.no_grad():
    # Reconstruct test data
    reconstructed_scaled, _, _ = vae(
        scale_data(test_data, data_min, data_max).to(device)
    )
    reconstructed_unscaled = descale_data(
        reconstructed_scaled.cpu(), data_min, data_max
    )
    # Generate new samples
    generated_scaled = vae.generate_data(num_samples=1000)
    generated_unscaled = descale_data(
        generated_scaled.cpu(), data_min, data_max
    )

# Plotting reconstructed data
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
ax1.scatter(test_data[:, 0], test_data[:, 1], s=5, alpha=0.7)
ax1.set_title("Original Test Data")
ax1.grid(True)
ax1.axis("equal")
ax2.scatter(
    reconstructed_unscaled[:, 0],
    reconstructed_unscaled[:, 1],
    s=5,
    alpha=0.7,
    c="r",
)
ax2.set_title("Reconstructed Test Data")
ax2.grid(True)
ax2.axis("equal")
title = f"Reconstructed vs Original Test Data for {method} Loss function with $\\beta$ ={'%.2E' % Decimal(beta)}"
plt.suptitle(title)
plt.savefig(
    f"fire_evac_reconstructed_data_loss_curve_{method}_beta_{'%.2E' % Decimal(beta)}.png"
)
plt.show()

# Plotting generated data
plt.figure(figsize=(8, 8))
plt.scatter(
    generated_unscaled[:, 0], generated_unscaled[:, 1], s=10, alpha=0.7, c="g"
)
plt.title("1000 Generated Samples")
plt.xlabel("X-position")
plt.ylabel("Y-position")
plt.grid(True)
plt.axis("equal")
plt.savefig("fire_evac_generated_samples.png")
plt.show()

# TODO: Generate data to estimate the critical number of people for the MI building
# Estimating the critical number of people in a sensitive area
print("Estimating the critical number of people...")
x_min, x_max, y_min, y_max = 130, 150, 50, 70
critical_threshold = 100
people_in_area, total_people_generated = 0, 0

with torch.no_grad():
    while people_in_area <= critical_threshold:
        generated_unscaled = descale_data(
            vae.generate_data(num_samples=100).cpu(), data_min, data_max
        )
        total_people_generated += 100

        in_area_mask = (
            generated_unscaled[:, 0].ge(x_min)
            & generated_unscaled[:, 0].le(x_max)
            & generated_unscaled[:, 1].ge(y_min)
            & generated_unscaled[:, 1].le(y_max)
        )

        people_in_area += torch.sum(in_area_mask).item()
        print(
            f"  ... generated {total_people_generated} people, {people_in_area} are in the sensitive area.",
            end="\r",
        )

print("\n\n--- Estimation Result ---")
print(
    f"The number of people in the sensitive area (count: {people_in_area}) exceeded the threshold of {critical_threshold}."
)
print(
    f"Total number of people needed to exceed the critical number: ~{total_people_generated}"
)
