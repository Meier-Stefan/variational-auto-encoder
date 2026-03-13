import torch


def get_device():
    """
    Returns the appropriate torch device (cuda if available, else cpu).
    
    Returns:
        torch.device: The device to use for computations
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
