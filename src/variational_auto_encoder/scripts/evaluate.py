"""CLI script for evaluating a trained VAE."""

import argparse
from dataclasses import dataclass

import torch

from variational_auto_encoder.train import evaluate, create_dataloaders
from variational_auto_encoder.model import VariationalAutoEncoder
from variational_auto_encoder.utils import resolve_device, ModelConfig


@dataclass
class EvalConfig:
    batch_size: int
    input_dim: int
    hidden_dim: int
    z_dim: int
    device: str
    data_dir: str
    checkpoint_path: str


def cli(argv: list[str] | None = None) -> int:
    model_cfg = ModelConfig()

    parser = argparse.ArgumentParser(description="Evaluate a trained VAE on MNIST test set")
    parser.add_argument("--data-dir", default="dataset/", help="Directory containing dataset")
    parser.add_argument("--checkpoint-path", "-c", default="vae_mnist.pth", help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--input-dim", type=int, default=model_cfg.input_dim, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=model_cfg.hidden_dim, help="Hidden dimension")
    parser.add_argument("--z-dim", type=int, default=model_cfg.z_dim, help="Latent dimension")
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on (e.g. "auto", "cpu", "cuda", "cuda:0", "mps")',
    )

    args = parser.parse_args(argv)

    cfg = EvalConfig(
        batch_size=args.batch_size,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        device=args.device,
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint_path,
    )

    device = resolve_device(cfg.device)

    _, test_loader = create_dataloaders(cfg.data_dir, cfg.batch_size)

    model = VariationalAutoEncoder(cfg.input_dim, cfg.hidden_dim, cfg.z_dim)
    checkpoint = torch.load(cfg.checkpoint_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)

    test_loss = evaluate(model, test_loader, device, cfg.input_dim)
    print(f"Test loss: {test_loss:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
