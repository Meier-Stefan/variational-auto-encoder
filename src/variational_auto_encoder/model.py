"""Variational Autoencoder (VAE) model and loss functions for MNIST reconstruction."""

import torch
from torch import nn


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dimension: int, hidden_dimension: int = 200, z_dimension: int = 20) -> None:
        super().__init__()

        # Encoder
        self.img_to_hidden = nn.Linear(input_dimension, hidden_dimension)
        self.hidden_to_mean = nn.Linear(hidden_dimension, z_dimension)
        self.hidden_to_std = nn.Linear(hidden_dimension, z_dimension)

        # Decoder
        self.z_to_hidden = nn.Linear(z_dimension, hidden_dimension)
        self.hidden_to_img = nn.Linear(hidden_dimension, input_dimension)

        self.relu = nn.ReLU()

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.relu(self.img_to_hidden(x))
        mean = self.hidden_to_mean(hidden)
        std = self.hidden_to_std(hidden)
        return mean, std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.z_to_hidden(z))
        return torch.sigmoid(self.hidden_to_img(hidden))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, std = self.encode(x)
        epsilon = torch.randn_like(std)
        z_new = mean + std * epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mean, std


def kl_divergence(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    return -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))


def vae_loss(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    reconstruction_loss = nn.BCELoss(reduction="sum")(recon_x, x)
    kl_div = kl_divergence(mu, sigma)
    return reconstruction_loss + kl_div
