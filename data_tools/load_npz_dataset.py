"""Dataset utilities for loading numpy-based image datasets."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class _NpzDataset((Dataset[tuple[torch.Tensor, torch.Tensor]])):
    """Internal PyTorch Dataset for numpy arrays."""

    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[idx]
        y = self.labels[idx]

        x = torch.from_numpy(x).float().unsqueeze(0) / 255.0
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def load_npz_dataset(npz_path: Path | str, train: bool = True) -> _NpzDataset:
    """Create a PyTorch Dataset from a numpy .npz file.

    Loads either training or test splits from a numpy archive file
    containing MNIST-style data with x_train/y_train and x_test/y_test arrays.
    """
    npz_path = Path(npz_path)
    data = np.load(npz_path)

    if train:
        images = data["x_train"]
        labels = data["y_train"]
    else:
        images = data["x_test"]
        labels = data["y_test"]

    return _NpzDataset(images, labels)
