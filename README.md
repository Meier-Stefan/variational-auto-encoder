# Variational Auto Encoder to Create Synthetic Retina Fundus images

The goal of this project is to create synthetic data for models that will be trained on retina fundus images to detect diabetic retinopathy.

The first step is to create a Variational Auto Encoder (VAE) and test it with a dataset that is available in pytorch.

This is setting up determinism according to the PyTorch docs and initially following the turorial of Aladdin Persson: https://www.youtube.com/watch?v=VELQT1-hILo&t=1618s

## Setup

### 1. System level:
These commands are designed for the supercomputer (HPC) environment:

- **`module load`** - Loads pre-installed software packages from the system's module system. This is common on HPC clusters where different software versions are managed centrally.

- **`cuda/12.2.1-gcc-13.2.0`** - Loads NVIDIA's CUDA toolkit for GPU acceleration. PyTorch can use this for faster training on GPUs.

```bash
module load cuda/12.2.1-gcc-13.2.0
```
**Note:** If you have a GPU in a linux machine, instructions are below.


### 2. Python packages:

- **`uv sync`** - [uv](https://github.com/astral-sh/uv) is a fast Python package manager (used everywhere in this project). `uv sync` synchronizes the virtual environment with `pyproject.toml` dependencies (creates `.venv` if it doesn't exist).

- **`source .venv/bin/activate`** - Activates the virtual environment so you use the installed packages instead of system Python.
```bash
uv sync
source .venv/bin/activate
```

### Local Linux (with NVIDIA GPU, replaces system level step 1 above)

If you're running on a local Linux machine with an NVIDIA GPU:

1. **Install NVIDIA drivers** - Most Linux distributions have this in their package manager (e.g., `apt install nvidia-driver-535` on Ubuntu)

2. **Install CUDA Toolkit** - Download from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads) or use your package manager

3. **Install cuDNN** - Download from [NVIDIA's website](https://developer.nvidia.com/cudnn) (requires account, but free). Extract to your CUDA installation directory.

4. **Verify GPU access:**
   ```bash
   nvidia-smi  # Should show your GPU
   ```
**Note:** If you don't have a GPU or CUDA installed, PyTorch will automatically fall back to CPU (slower training).

## Usage

### Training

Train the model with default settings (MNIST, 10 epochs):

```bash
vae-train
```

This will:
- Train a VAE on MNIST training set
- Evaluate on MNIST test set
- Save the model to `vae_mnist.pth`
- Print the test loss to the console

### Custom Training via CLI

```bash
vae-train --epochs 20 --batch-size 64 --lr 1e-4 --z-dim 50 --output-path my_model.pth
```

### Custom Training via Python

```python
from variational_auto_encoder import train_loop

class TrainConfig:
    batch_size: int = 32
    epochs: int = 10
    lr: float = 3e-4
    input_dim: int = 784  # 28x28 = 784
    hidden_dim: int = 200
    z_dim: int = 20
    device: str = "auto"
    data_dir: str = "dataset/"
    output_path: str = "vae_mnist.pth"

config = TrainConfig()
model = train_loop(config)
```

### Generating Images

Generate new images using a trained model:

```bash
vae-generate --num-samples 16 --save-path generated.png
```

Or via Python:

```python
from variational_auto_encoder import generate

# Generate from a trained model
generate(num_samples=16, save_path="generated.png")
```

An output could look like the following example:

![Example of generated handwritten numbers. ](./generated.png)

### Evaluating a Trained Model

Evaluate a trained model on the MNIST test set:

```bash
vae-evaluate --checkpoint-path vae_mnist.pth
```

### Load Datasets

Load and inspect numpy datasets (.npz format):

```bash
load-dataset path/to/dataset.npz --train    # Load training split (default)
load-dataset path/to/dataset.npz --test     # Load test split
```

This CLI is installed via `pyproject.toml` and requires `uv sync` to register.
