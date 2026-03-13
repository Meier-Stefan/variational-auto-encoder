"""CLI script for generating images with a trained VAE."""

import argparse

import torch

from variational_auto_encoder.train import generate
from variational_auto_encoder.utils import resolve_device, ModelConfig


def cli(argv: list[str] | None = None) -> int:
    model_cfg = ModelConfig()

    parser = argparse.ArgumentParser(description="Generate images with a trained VAE")
    parser.add_argument("--checkpoint-path", "-c", default="vae_mnist.pth", help="Path to model checkpoint")
    parser.add_argument("--num-samples", "-n", type=int, default=16, help="Number of samples to generate")
    parser.add_argument("--save-path", "-o", default="generated.png", help="Path to save generated images")
    parser.add_argument("--z-dim", type=int, default=model_cfg.z_dim, help="Latent dimension")
    parser.add_argument("--input-dim", type=int, default=model_cfg.input_dim, help="Input dimension")
    parser.add_argument("--hidden-dim", type=int, default=model_cfg.hidden_dim, help="Hidden dimension")
    parser.add_argument(
        "--device",
        default="auto",
        help='Device to run on (e.g. "auto", "cpu", "cuda", "cuda:0", "mps")',
    )

    args = parser.parse_args(argv)

    device = resolve_device(args.device)

    # The library-level `generate` accepts a torch.device or None
    generate(
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        save_path=args.save_path,
        z_dim=args.z_dim,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
