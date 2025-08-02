import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from model import VAE
from torch.utils.data import DataLoader


# Define a loss function that combines binary cross-entropy and Kullback-Leibler divergence
def reconstruction_loss(
    x_reconstructed: torch.Tensor, x: torch.Tensor
) -> torch.Tensor:
    """Compute the reconstruction loss.

    Args:
        x_reconstructed (torch.Tensor): Reconstructed data
        x (torch.Tensor): raw/original data

    Returns:
        (torch.Tensor): reconstruction loss
    """
    # from hint using binary_cross_entropy
    # Flatten both tensors to ensure same dimensions
    x = x.view(-1, 784)
    x_reconstructed = x_reconstructed.view(-1, 784)
    return F.binary_cross_entropy(x_reconstructed, x, reduction="sum")


def kl_loss(logvar: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
    """Compute the Kullback-Leibler (KL) divergence loss using the encoded data into the mean and log-variance.

    Args:
        logvar (torch.Tensor): log of variance (from the output of the encoder)
        mu (torch.Tensor): mean (from the output of the encoder)

    Returns:
        (torch.Tensor): KL loss
    """
    # since the kl divergence is given by this formula
    # log valr is log(sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return kld


# Function to compute ELBO loss
def elbo_loss(
    x: torch.Tensor,
    x_reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
):
    """Compute Evidence Lower BOund (ELBO) Loss by combining the KL loss and reconstruction loss.

    Args:
        x (torch.Tensor): raw/original data
        x_reconstructed (torch.Tensor): Reconstructed data
        mu (torch.Tensor): mean (from the output of the encoder)
        logvar (torch.Tensor): log of variance (from the output of the encoder)

    Returns:
        (torch.Tensor): ELBO loss
    """
    # usually flattened the input x
    # as pdf mentions elbo loss to be sum of two losses...so adding them
    recon_loss = reconstruction_loss(x_reconstructed, x)
    kld = kl_loss(logvar, mu)
    # adding them
    return recon_loss + kld


# Loss function suitable for 2D coordinate data (using l1), for task 4
def elbo_loss_l1(
    x: torch.Tensor,
    x_reconstructed: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """ELBO loss using L1 (Mean absolute error) for reconstruction.
    This version uses 'mean' reduction for more stable hyperparameter tuning.
    """
    # Use 'mean' and then scale by the number of features (input dimensions).
    # For (x,y) data, x.shape[1] is 2.
    recon_loss = F.l1_loss(x_reconstructed, x, reduction="sum")
    # recon_loss = F.mse_loss(x_reconstructed, x, reduction='sum')

    kld = kl_loss(logvar, mu)

    # We now divide the KLD by the batch size to keep it on a similar
    # per-sample scale as the mean-based reconstruction loss.

    return recon_loss + (beta * kld)


# Function for training the VAE
def train_epoch(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    dataloader: DataLoader,
    device,
    special_loss: bool,
    beta=0.1,
) -> np.float64:
    """Train the vae for one epoch and return the training loss on the epoch.

    Args:
        model (object): The model (of class VAE)
        optimizer (object): Adam optimizer (from torch.optim)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu') on which the training is to be done.
        special_loss: Using special loss function for task 4
    Returns:
        np.float64: training loss
    """

    total_loss = 0.0  # Initialize total loss for the epoch
    samples = 0  # init the number of samples
    for batch in dataloader:
        optimizer.zero_grad()  # Zero the gradients before each batch
        x = batch[0].to(device)

        # TODO: Perform forward pass of the VAE

        reconstructed_x, mu, logvar = model(x)

        # TODO Compute the ELBO loss
        if special_loss:
            loss = elbo_loss_l1(x, reconstructed_x, mu, logvar, beta)
        else:
            loss = elbo_loss(x, reconstructed_x, mu, logvar)

        # TODO Compute gradients
        loss.backward()

        # TODO Perform an optimization step
        optimizer.step()

        # TODO: Compute total_loss and return the total_loss/len(dataloader.dataset)
        total_loss += loss.item()  # Accumulate the loss for the epoch
        if not special_loss:
            samples += x.size(0)
    return np.float64(
        total_loss / len(dataloader.dataset)
    )  # Average loss per sample
    pass


def evaluate(
    model: VAE,
    optimizer: object,
    dataloader: DataLoader,
    device,
    special_loss: bool,
    beta: float = 1.0,
) -> np.float64:
    """Evaluate the model on the test data and return the test loss.

    Args:
        model (object): The model (of class VAE)
        dataloader (object): Data loader combines a dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        device: The device (e.g., 'cuda' or 'cpu')
        special_loss: Using special loss function for task 4
        beta: Float for the special loss
    Returns:
        np.float64: test loss.
    """
    # TODO: Implement method!
    # Hint: Do not forget to deactivate the gradient calculation!
    model.eval()
    total_loss = 0.0  # Initialize total loss for the evaluation
    samples = 0  # Initialize the number of samples
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch in dataloader:
            x = batch[0].to(device)

            reconstructed_x, mu, logvar = model(x)  # Forward pass
            if special_loss:
                loss = elbo_loss_l1(x, reconstructed_x, mu, logvar, beta)
            else:
                loss = elbo_loss(x, reconstructed_x, mu, logvar)
            total_loss += loss.item()  # Accumulate the loss for the evaluation
            if not special_loss:
                samples += x.size(0)
    # Return the average loss over the dataset
    return np.float64(total_loss / len(dataloader.dataset))


# Function to plot latent representation of the first batch with different classes colored
def latent_representation(
    model: VAE, optimizer: object, dataloader: DataLoader, device, epoch: int
) -> None:
    """Plot the latent representation with different classes colored."""
    model.eval()
    with torch.no_grad():
        # Lists to store all encodings and labels
        all_z = []
        all_labels = []

        # Iterate through all batches
        for data, labels in dataloader:
            x = data.to(device)
            # Encode data
            mu, logvar = model.encode_data(x)
            z = model.reparameterize(mu, logvar)

            # Store encodings and labels
            all_z.append(z.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        # Concatenate all batches
        all_z = np.concatenate(all_z, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Create scatter plot
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            all_z[:, 0], all_z[:, 1], c=all_labels, cmap="tab10", alpha=0.6
        )
        plt.colorbar(scatter)
        plt.title(f"Latent Space Representation (All Samples) - Epoch {epoch}")
        plt.xlabel("z₁")
        plt.ylabel("z₂")
        plt.savefig(f"scatter_all_{epoch}.png")
        plt.show()
        plt.close()


# Function to plot original and reconstructed digits side by side
def reconstruct_digits(
    model: VAE,
    optimizer: object,
    dataloader: DataLoader,
    device,
    epoch: int,
    num_digits: int = 15,
) -> None:
    """Plot original and reconstructed digits side by side."""
    model.eval()
    with torch.no_grad():
        # Set fixed seed for reproducibility
        torch.manual_seed(42)

        # Get first batch
        first_batch = next(iter(dataloader))[0][:num_digits].to(device)
        reconstructed_x, _, _ = model(first_batch)

        # Create figure
        plt.figure(figsize=(20, 8))
        for i in range(num_digits):
            # Original
            plt.subplot(2, num_digits, i + 1)
            plt.imshow(
                first_batch[i].cpu().numpy().reshape(28, 28), cmap="gray"
            )
            plt.axis("off")
            if i == 0:
                plt.title("Original")

            # Reconstructed
            plt.subplot(2, num_digits, num_digits + i + 1)
            plt.imshow(
                reconstructed_x[i].cpu().numpy().reshape(28, 28), cmap="gray"
            )
            plt.axis("off")
            if i == 0:
                plt.title("Reconstructed")

        plt.suptitle(f"Original vs Reconstructed Digits - Epoch {epoch}")
        plt.savefig(f"original_vs_reconstructed_{epoch}.png")
        plt.show()
        plt.close()


# Function to plot generated digits
def generate_digits(
    model: VAE,
    optimizer: object,
    dataloader: DataLoader,
    device,
    epoch: int,
    num_samples: int = 15,
) -> None:
    """Generate digits by encoding and decoding samples from the dataset.

    Args:
        model (VAE): The VAE model
        optimizer (object): The optimizer
        dataloader (DataLoader): Dataset loader
        device: The device to run on
        epoch (int): Current epoch number
        num_samples (int, optional): Number of samples to generate. Defaults to 15.
    """
    # Using only the first batch to generate digits for simplicity, speed and reproducibility
    model.eval()
    with torch.no_grad():
        # Get first batch and encode it
        x = next(iter(dataloader))[0][:num_samples].to(device)
        mu, logvar = model.encode_data(x)

        # Sample from the latent space using the encoded parameters
        z = model.reparameterize(mu, logvar)

        # Generate new samples by decoding
        generated_samples = model.decode_data(z)

        # Plot the generated digits
        plt.figure(figsize=(10, 10))
        for i in range(num_samples):
            plt.subplot(3, 5, i + 1)
            plt.imshow(
                generated_samples[i].cpu().numpy().reshape(28, 28), cmap="gray"
            )
            plt.axis("off")

        plt.suptitle(f"Generated Digits from Encoded Space - Epoch {epoch}")
        plt.tight_layout()
        plt.savefig(f"generated_{epoch}.png")
        plt.show()
        plt.close()

    pass


# Function to plot the loss curve
def plot_loss(
    train_losses, test_losses, title="Loss Curve", save_path="loss_curve.png"
) -> None:
    """Plot the training and test loss curves.
    Args:
        train_losses (list): List of training losses.
        test_losses (list): List of test losses.
        title (str, optional): Title of the plot. Defaults to "Loss Curve".
        save_path (str, optional): Path to save the plot. Defaults to "loss_curve.png".
    """
    # Plot the training and test losses
    epochs = len(train_losses)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, label="Train")
    plt.plot(range(1, epochs + 1), test_losses, label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("ELBO Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    plt.close()


def training_loop(
    vae: VAE,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int,
    plots_at_epochs: list,
    device,
    special_loss=None,
    beta: float = 1.0,
) -> tuple[list, list]:
    """Train the vae model.

    Args:
        vae (object): The model (of class VAE).
        optimizer (object): Adam optimizer (from torch.optim).
        train_loader (object): A data loader that combines the training dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        test_loader (object): A data loader that combines the test dataset and a sampler, and provides an iterable over the given dataset (from torch.utils.data).
        epochs (int): No. of epochs to train the model.
        plots_at_epochs (list): List of integers containing epoch numbers at which the plots are to be made.
        device: The device (e.g., 'cuda' or 'cpu').
        special_loss: Bool for determing which loss function to use
        beta (float): Weight for the KL divergence term.

    Returns:
        tuple [list, list]: Lists train_losses, test_losses containing train and test losses at each epoch.
    """
    # Lists to store the training and test losses
    train_losses = []
    test_losses = []
    for epoch in range(1, epochs + 1):
        # TODO: Compute training loss for one epoch
        train_loss = train_epoch(
            vae, optimizer, train_loader, device, special_loss, beta
        )

        # TODO: Evaluate loss on the test dataset
        test_loss = evaluate(
            vae, optimizer, test_loader, device, special_loss, beta
        )

        # TODO: Append train and test losses to the lists train_losses and test_losses respectively
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        print(
            f"Epoch {epoch}, Train Loss: {train_loss.item()}, Test Loss: {test_loss.item()}"
        )

        # TODO: For specific epoch numbers described in the worksheet, plot latent representation, reconstructed digits, generated digits after specific epochs
        if epoch in plots_at_epochs:
            latent_representation(vae, optimizer, train_loader, device, epoch)
            reconstruct_digits(vae, optimizer, test_loader, device, epoch)
            generate_digits(vae, optimizer, test_loader, device, epoch)

    # TODO: return train_losses, test_losses
    return train_losses, test_losses
    pass


def instantiate_vae(
    d_in,
    d_latent,
    d_hidden_layer,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    """Instantiate the variational autoencoder.

    Args:
        d_in (int): Input dimension.
        d_latent (int): Latent dimension.
        d_hidden_layer (int): Number of neurons in each hidden layer of the encoder and decoder.
        device: e.g., 'cuda' or 'cpu'. Defaults to torch.device('cuda' if torch.cuda.is_available() else 'cpu').

    Returns:
        object: An object of class VAE
    """
    return VAE(d_in, d_latent, d_hidden_layer, device).to(device)


# using the functions in 2_fire_evac so better practice to put it here
# scaling the data: Rescale data to [-1, 1] as suggested
def scale_data(
    data: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor
) -> torch.Tensor:
    # we scale from [0,1] then to [-1, 1]
    return 2 * ((data - data_min) / (data_max - data_min)) - 1


# descaling the data: Rescale data to original
def descale_data(
    data: torch.Tensor, data_min: torch.Tensor, data_max: torch.Tensor
) -> torch.Tensor:
    # we scale from [-1, 1] to original
    return ((data + 1) / 2) * (data_max - data_min) + data_min
