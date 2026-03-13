"""CLI script for training VAE on MNIST."""

import argparse
from dataclasses import dataclass

from variational_auto_encoder.train import train_loop
from variational_auto_encoder.utils import ModelConfig


@dataclass
class SimpleConfig:
    batch_size: int
    epochs: int
    lr: float
    input_dim: int
    hidden_dim: int
    z_dim: int
    device: str
    data_dir: str
    output_path: str


def main(argv: list[str] | None = None) -> int:
    model_cfg = ModelConfig()

    parser = argparse.ArgumentParser(description="Train a VAE on MNIST")
    parser.add_argument("--data-dir", default="dataset/", help="Directory to store/load dataset")
    parser.add_argument("--output-path", default="vae_mnist.pth", help="Path to save model weights")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--input-dim",
        type=int,
        default=model_cfg.input_dim,
        help="Input dimension (default: 784 for 28x28)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=model_cfg.hidden_dim,
        help="Hidden layer dimension",
    )
    parser.add_argument(
        "--z-dim",
        type=int,
        default=model_cfg.z_dim,
        help="Latent space dimension",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to train on (e.g. "auto", "cpu", "cuda", "cuda:0", "mps")',
    )

    args = parser.parse_args(argv)

    cfg = SimpleConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        z_dim=args.z_dim,
        device=args.device,
        data_dir=args.data_dir,
        output_path=args.output_path,
    )

    train_loop(cfg)
    return 0


def cli(argv: list[str] | None = None) -> int:
    return main(argv)


if __name__ == "__main__":
    raise SystemExit(cli())
