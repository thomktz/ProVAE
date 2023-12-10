import torch
import torch.nn as nn
import torch.optim as optim
from .encoder import Encoder
from .decoder import Decoder


class ProVAE(nn.Module):
    """
    Progressive-Growing Variational Autoencoder.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the latent space.
    config : list of dicts
        The configuration for the progressive growing.
    device : str
        The device to run the model on.
    """

    def __init__(self, latent_dim, config, device):
        super().__init__()
        self.latent_dim = latent_dim
        self.config = config
        self.device = device

        self.encoder = Encoder(config, latent_dim)
        self.decoder = Decoder(config, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick for sampling from a Gaussian distribution.

        Parameters
        ----------
        mu : torch.Tensor
            The mean of the distribution.
        logvar : torch.Tensor
            The log variance of the distribution.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """Forward pass through the encoder and decoder."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def _train_init(self, learning_rate):
        self.criterion = nn.MSELoss()  # or any other loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.to(self.device)

    def _train_one_epoch(self, dataloader, epoch_index):
        self.train()  # set the model to training mode
        total_loss = 0

        for batch in dataloader:
            # Assuming batch is a tuple of (inputs, targets)
            inputs, _ = batch
            inputs = inputs.to(self.device)

            # Forward pass
            reconstructed, mu, logvar = self(inputs)

            # Compute loss
            # Customize the loss function as per your requirements
            loss = self.criterion(reconstructed, inputs)  # Example loss

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch_index + 1}] complete. Average loss: {average_loss}")
