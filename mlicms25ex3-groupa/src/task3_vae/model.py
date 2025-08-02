import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(self, d_in: int, d_latent: int, d_hidden_layer: int, device):
        """Initialiize the VAE.

        Args:
            d_in (int): Input dimension
            d_latent (int): Latent dimension.
            d_hidden_layer (int): Number of neurons in the hidden layers of encoder and decoder.
            device: 'cpu' or 'cuda
        """
        super(VAE, self).__init__()

        # Set device
        self.device = device

        # TODO: Set dimensions: input dim, latent dim, and no. of neurons in the hidden layer
        self.input_dim = d_in  # Input dimension
        self.d_latent = d_latent  # Latent space dimension
        self.hidden_dim = (
            d_hidden_layer  # Number of neurons in the hidden layer
        )

        # TODO: Initialize the encoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )

        # TODO: Initialize a linear layer for computing the mean (one of the outputs of the encoder)
        self.fc_mu = nn.Linear(self.hidden_dim, self.d_latent)

        # TODO: Initialize a linear layer for computing the variance (one of the outputs of the encoder)
        self.fc_logvar = nn.Linear(self.hidden_dim, self.d_latent)

        # TODO: Initialize the decoder using nn.Sequential with appropriate layer dimensions, types (linear, ReLu, Sigmoid etc.).
        self.decoder = nn.Sequential(
            nn.Linear(self.d_latent, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim, self.input_dim
            ),  # Output reconstructed data
            nn.Sigmoid(),  # Use Sigmoid for output to ensure values are in [0, 1]
        )

    def encode_data(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass throguh the encoder.

        Args:
            x (torch.Tensor): Input data

        Returns:
            tuple[torch.Tensor, torch.Tensor]: mean, log of variance
        """
        # TODO: Implement method!!
        x = x.to(self.device)
        # Flatten input if needed
        if x.dim() == 4:  # (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)

        encoded = self.encoder(x)  # Pass through the encoder
        mu = self.fc_mu(encoded)  # Compute mean (fc_mu)
        logvar = self.fc_logvar(encoded)  # Compute log variance (fc_logvar)
        return mu, logvar
        pass

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Use the reparameterization trick for sampling from a Gaussian distribution.

        Args:
            mu (torch.Tensor): Mean of the Gaussian distribution.
            logvar (torch.Tensor): Log variance of the Gaussian distribution.

        Returns:
            torch.Tensor: Sampled latent vector.
        """
        # TODO: Implement method!!
        mu = mu.to(self.device)
        logvar = logvar.to(self.device)

        # Sample from a standard normal distribution
        std = torch.exp(0.5 * logvar)  # Standard deviation
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
        pass

    def decode_data(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vectors to reconstruct data.

        Args:
            z (torch.Tensor): Latent vector.

        Returns:
            torch.Tensor: Reconstructed data.
        """
        # TODO: Implement method!!

        z = z.to(self.device)
        reconstructed_data = self.decoder(z)
        return reconstructed_data
        pass

    def generate_data(self, num_samples: int) -> torch.Tensor:
        """Generate data by sampling and decoding 'num_samples' vectors in the latent space.

        Args:
            num_samples (int): Number of generated data samples.

        Returns:
            torch.Tensor: generated samples.
        """
        # TODO: Implement method!!
        # Hint (You may need to use .to(self.device) for sampling the latent vector!)
        z = torch.randn(num_samples, self.d_latent).to(
            self.device
        )  # Sample from standard normal distribution
        reconstructed_data = self.decode_data(z)
        return reconstructed_data
        pass

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass of the VAE.

        Args:
            x (torch.Tensor): Input data.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: reconstructed data, mean of gaussian distribution (encoder), variance of gaussian distribution (encoder)
        """
        # TODO: Implement method!!
        # Ensure proper input shape
        if len(x.shape) == 4:  # (batch_size, channels, height, width)
            x = x.view(x.size(0), -1)
        x = x.to(self.device)

        mu, logvar = self.encode_data(x)
        z = self.reparameterize(mu, logvar)
        x_reconstructed = self.decode_data(z)
        return x_reconstructed, mu, logvar
        pass
