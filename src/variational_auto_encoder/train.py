# Ensure CUBLAS workspace is configured as early as possible.
# This module is imported by the CLIs before they do any CUDA work.
from .utils import setup_cublas_workspace

setup_cublas_workspace()

from pathlib import Path
from typing import Protocol

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

from .model import VariationalAutoEncoder, vae_loss
from .utils import resolve_device, setup_determinism, ModelConfig
from .data_tools.load_npz_dataset import load_npz_dataset


class TrainConfig(Protocol):
    batch_size: int
    epochs: int
    lr: float
    input_dim: int
    hidden_dim: int
    z_dim: int
    device: str
    data_dir: str
    output_path: str
    npz_path: str | None


def create_dataloaders(data_dir: str, batch_size: int = 32, npz_path: str | None = None) -> tuple[DataLoader, DataLoader]:
    if npz_path is not None:
        train_dataset = load_npz_dataset(npz_path, train=True)
        test_dataset = load_npz_dataset(npz_path, train=False)
    else:
        train_dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=True,
        )
        test_dataset = datasets.MNIST(
            root=data_dir,
            train=False,
            transform=transforms.ToTensor(),
            download=True,
        )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def create_model(
    input_dim: int, hidden_dim: int, z_dim: int, device: torch.device
) -> VariationalAutoEncoder:
    model = VariationalAutoEncoder(input_dim, hidden_dim, z_dim).to(device)
    return model


def train_one_epoch(
    model: VariationalAutoEncoder,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    input_dim: int,
) -> float:
    """
    Train for one epoch and return mean loss per sample (matching original behavior).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, _ in tqdm(train_loader, desc="Training", leave=False):
        x = x.to(device).view(x.shape[0], input_dim)
        x_reconstructed, mu, sigma = model(x)
        loss = vae_loss(x_reconstructed, x, mu, sigma)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = x.size(0)
        total_loss += loss.item()
        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: VariationalAutoEncoder,
    test_loader: DataLoader,
    device: torch.device,
    input_dim: int,
) -> float:
    """
    Evaluate and return mean loss per sample (matching original behavior).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(device).view(x.shape[0], input_dim)
            x_reconstructed, mu, sigma = model(x)
            loss = vae_loss(x_reconstructed, x, mu, sigma)

            batch_size = x.size(0)
            total_loss += loss.item()
            total_samples += batch_size

    return total_loss / total_samples


def train_loop(cfg: TrainConfig) -> VariationalAutoEncoder:
    setup_determinism()

    device = resolve_device(cfg.device)

    npz_path = getattr(cfg, "npz_path", None)
    train_loader, test_loader = create_dataloaders(cfg.data_dir, cfg.batch_size, npz_path)
    model = create_model(cfg.input_dim, cfg.hidden_dim, cfg.z_dim, device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for epoch in range(cfg.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, cfg.input_dim)
        val_loss = evaluate(model, test_loader, device, cfg.input_dim)
        print(f"Epoch {epoch + 1}/{cfg.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    output_path = Path(cfg.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            # Store model config for future use (dimension consistency).
            "model_config": {
                "input_dim": cfg.input_dim,
                "hidden_dim": cfg.hidden_dim,
                "z_dim": cfg.z_dim,
            },
        },
        output_path,
    )
    print(f"Model saved to {output_path}")

    return model


def generate(
    model: VariationalAutoEncoder | None = None,
    num_samples: int = 16,
    z_dim: int = 20,
    save_path: str = "generated.png",
    device: torch.device | None = None,
    input_dim: int = 784,
    hidden_dim: int = 200,
    checkpoint_path: str = "vae_mnist.pth",
) -> None:
    from torchvision.utils import save_image

    if device is None:
        device = resolve_device("auto")

    if model is None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Support both new (with model_config) and old (pure state_dict) checkpoints.
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            cfg_dict = checkpoint.get("model_config")
            if cfg_dict is not None:
                input_dim = cfg_dict.get("input_dim", input_dim)
                hidden_dim = cfg_dict.get("hidden_dim", hidden_dim)
                z_dim = cfg_dict.get("z_dim", z_dim)
        else:
            # Old format: raw state_dict
            state_dict = checkpoint

        model = VariationalAutoEncoder(input_dim, hidden_dim, z_dim)
        model.load_state_dict(state_dict)
        model.to(device)
        print(f"Loaded model from {checkpoint_path}")

    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        samples = model.decode(z)
        samples = samples.view(-1, 1, 28, 28)
        save_image(samples, save_path)
        print(f"Generated {num_samples} images saved to {save_path}")
