import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
from tqdm import tqdm
from torch import nn, optim
from model import VariationalAutoEncoder
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils.device import get_device
from utils.determinism import setup_determinism
from inference import generate

import torchvision.datasets as datasets


def train(
    input_dim=784,
    hidden_dim=200,
    z_dim=20,
    epochs=10,
    batch_size=32,
    lr=3e-4,
    device=None,
    save_path="vae_mnist.pth"
):
    """
    Train a VAE on MNIST dataset.
    
    Args:
        input_dim: Input dimension (default: 784 for 28x28)
        hidden_dim: Hidden layer dimension (default: 200)
        z_dim: Latent space dimension (default: 20)
        epochs: Number of training epochs (default: 10)
        batch_size: Batch size (default: 32)
        lr: Learning rate (default: 3e-4)
        device: Device to train on (default: cuda if available)
        save_path: Path to save model weights (default: vae_mnist.pth)
    
    Returns:
        Trained model
    """
    setup_determinism()
    print("Determinism setup complete!")

    if device is None:
        device = get_device()

    train_dataset = datasets.MNIST(
        root="dataset/", train=True, transform=transforms.ToTensor(), download=True
    )
    test_dataset = datasets.MNIST(
        root="dataset/", train=False, transform=transforms.ToTensor(), download=True
    )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    model = VariationalAutoEncoder(input_dim, hidden_dim, z_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="sum")

    model.train()
    for epoch in range(epochs):
        training_loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (x, _) in training_loop:
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)

            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loop.set_postfix(loss=loss.item())

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}!")

    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)

            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))
            test_loss += reconstruction_loss + kl_div

    test_loss /= len(test_dataset)
    print(f"Test loss: {test_loss:.4f}")

    return model


if __name__ == "__main__":
    model = train(epochs=10, save_path="vae_mnist.pth")
    generate(model, num_samples=16)
