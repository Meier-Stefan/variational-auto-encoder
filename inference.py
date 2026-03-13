import torch
from torchvision.utils import save_image
from model import VariationalAutoEncoder
from utils.device import get_device


def load_model(input_dim=784, hidden_dim=200, z_dim=20, weights_path="vae_mnist.pth", device=None):
    """
    Load a trained VAE model from weights file.
    
    Args:
        input_dim: Input dimension (default: 784 for 28x28)
        hidden_dim: Hidden layer dimension (default: 200)
        z_dim: Latent space dimension (default: 20)
        weights_path: Path to model weights file
        device: Device to load model on (default: cuda if available)
    
    Returns:
        Loaded model on specified device
    """
    if device is None:
        device = get_device()

    model = VariationalAutoEncoder(input_dim, hidden_dim, z_dim)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def generate(
    model=None,
    num_samples=16,
    z_dim=20,
    save_path="generated.png",
    device=None,
    weights_path="vae_mnist.pth"
):
    """
    Generate new images using a trained VAE.
    
    Args:
        model: Trained VAE model (if None, loads from weights_path)
        num_samples: Number of samples to generate (default: 16)
        z_dim: Latent dimension used during training (default: 20)
        save_path: Path to save generated images (default: generated.png)
        device: Device to run on (default: cuda if available)
        weights_path: Path to model weights if loading model (default: vae_mnist.pth)
    """
    if device is None:
        device = get_device()

    if model is None:
        model = load_model(
            input_dim=784,
            hidden_dim=200,
            z_dim=z_dim,
            weights_path=weights_path,
            device=device
        )
        print(f"Loaded model from {weights_path}")

    model.eval()
    with torch.no_grad():
        z = torch.randn(num_samples, z_dim).to(device)
        samples = model.decode(z)
        samples = samples.view(-1, 1, 28, 28)
        save_image(samples, save_path)
        print(f"Generated {num_samples} images saved to {save_path}")
