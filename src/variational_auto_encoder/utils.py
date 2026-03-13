import os
from dataclasses import dataclass

import torch


# Determinism / env utils

def setup_cublas_workspace() -> None:
    """
    Ensure CUBLAS_WORKSPACE_CONFIG is set for deterministic cuBLAS behavior
    on CUDA >= 10.2.

    Must be called before any CUDA work starts. In practice, importing a module
    that calls this at import time is sufficient.
    """
    # Only set if not already set from the outside, so users can override.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def setup_determinism(seed: int = 42) -> None:
    """
    Configure PyTorch for deterministic behavior as far as possible.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Device resolution

def resolve_device(device_str: str) -> torch.device:
    """
    Resolve a user-facing device string into a torch.device.

    - "auto": prefer CUDA, then MPS, then CPU.
    - Otherwise: use the passed string directly (e.g. "cpu", "cuda", "cuda:0", "mps").
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_str)


# Shared model config (to keep dimensions in sync across scripts)

@dataclass
class ModelConfig:
    input_dim: int = 784
    hidden_dim: int = 200
    z_dim: int = 20
