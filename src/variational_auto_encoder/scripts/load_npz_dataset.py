"""CLI script for loading and inspecting numpy dataset files."""

import argparse
from pathlib import Path

from variational_auto_encoder.data_tools.load_npz_dataset import load_npz_dataset


def main(npz_path: Path | str, train: bool = True) -> None:
    """Load and validate a numpy dataset."""
    dataset = load_npz_dataset(npz_path, train=train)
    print(f"Loaded dataset with {len(dataset)} samples")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect numpy image datasets"
    )
    parser.add_argument("npz_path", help="Path to the .npz dataset file")
    parser.add_argument(
        "--train",
        "-t",
        action="store_true",
        default=True,
        help="Load training split (default: True)",
    )
    parser.add_argument(
        "--test",
        "-T",
        action="store_false",
        dest="train",
        help="Load test split instead of training",
    )
    return parser.parse_args(argv)


def cli(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    main(args.npz_path, train=args.train)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
