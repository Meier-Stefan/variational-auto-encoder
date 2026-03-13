import torch


def setup_determinism(seed=42):
    """
    Sets up a fully deterministic environment.
    
    Args:
        seed: Random seed for reproducibility (default: 42)
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
