import os

# CRITICAL: Set this BEFORE importing torch or doing any CUDA operations
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
from torch import nn


def setup_determinism(seed=42):
    """Sets up a fully deterministic environment."""
    # Set seeds for all libraries
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True)
    # Disable cuDNN's non-deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
setup_determinism()
print("Determinism setup complete! ")

class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_dimension, hidden_dimension=200, z_dimension=20):
        super().__init__()
        
        # encoder
        self.img_2hid = nn.Linear(input_dimension, hidden_dimension)
        self.hiddenToMean = nn.Linear(hidden_dimension, z_dimension)
        self.hiddenToStandardDeviation = nn.Linear(hidden_dimension, z_dimension)

        # decoder
        self.z_2hid = nn.Linear(z_dimension, hidden_dimension)
        self.hid_2img = nn.Linear(hidden_dimension, input_dimension)

        self.relu = nn.ReLU()

    def encode(self, x):
        hidden_dimension = self.relu(self.img_2hid(x))
        mean, standardDeviation = self.hiddenToMean(hidden_dimension), self.hiddenToStandardDeviation(hidden_dimension)
        return mean, standardDeviation

    def decode(self, z):
        hidden_dimension = self.relu(self.z_2hid(z))
        return torch.sigmoid(self.hid_2img(hidden_dimension))

    def forward(self, x):
        mean, standardDeviation = self.encode(x)
        epsilon = torch.randn_like(standardDeviation)
        z_new = mean + standardDeviation*epsilon
        x_reconstructed = self.decode(z_new)
        return x_reconstructed, mean, standardDeviation

if __name__ == "__main__":
    x = torch.randn(4, 512*512)
    vae = VariationalAutoEncoder(input_dimension=262144)
    x_reconstructed, mean, standardDeviation = vae(x)
    print(x_reconstructed.shape)
    print(mean.shape)
    print(standardDeviation.shape)