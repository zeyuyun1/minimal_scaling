# Minimal Scaling Denoising Project

A PyTorch Lightning-based project for training recurrent convolutional denoising models with EDM-style loss functions.

## Features

- Recurrent convolutional denoising models
- EDM-style loss functions with configurable noise distributions
- Support for MNIST and CelebA datasets
- EMA (Exponential Moving Average) model updates
- Checkpoint loading functionality
- Weights & Biases integration for experiment tracking

## Installation

### Option 1: Automated Setup (Recommended)

```bash
# Clone the repository
git clone https://github.com/zeyuyun1/minimal_scaling.git
cd minimal_scaling

# Run the setup script
chmod +x setup.sh
./setup.sh

# Activate the environment
source venv/bin/activate
```

### Option 2: Manual Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
pip install -r requirements-minimal.txt
```

### Option 3: Full Installation (Exact Versions)

```bash
# For exact version matching with conda environment
pip install -r requirements.txt
```

## Usage

### Basic Training

```bash
python main.py --exp_name "my_experiment" --dataset celeba --batch_size 32 --n_epochs 20
```

### Training with Checkpoint Loading

```bash
python main.py --exp_name "resume_training" --checkpoint_path "pretrained_model/main/00001_experiment/denoiser.ckpt"
```

### Available Arguments

- `--exp_name`: Name of the experiment (required)
- `--dataset`: Dataset to use (`mnist` or `celeba`)
- `--batch_size`: Batch size for training
- `--n_epochs`: Number of training epochs
- `--checkpoint_path`: Path to checkpoint file for initialization
- `--gpus`: Number of GPUs to use
- `--lr`: Learning rate
- `--model_arch`: Model architecture (`h_sparse`, `recur_new`, or `unet`)

See `python main.py --help` for all available options.

## Project Structure

- `main.py`: Main training script
- `model.py`: Model definitions
- `data.py`: Data loading modules
- `loss.py`: Loss function implementations
- `utils.py`: Utility functions
- `run_exp.sh`: Experiment runner script

## Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 2.5+
- PyTorch Lightning 2.3+

## Notes

- The `data/` and `pretrained_model/` directories are excluded from the repository
- Training logs and checkpoints are saved in `pretrained_model/` directory
- Weights & Biases integration requires a W&B account
