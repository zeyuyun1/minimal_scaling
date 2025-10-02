#!/bin/bash

# Setup script for minimal_scaling project
# This script sets up the environment on a machine without conda

echo "Setting up minimal_scaling project..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support (adjust based on your CUDA version)
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements-minimal.txt

echo "Setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "deactivate"
