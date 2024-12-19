#!/bin/bash
set -ex

TORCHVISION_VERSION=0.20.0
PYTORCH_WHEEL_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
TORCHVISION_DIR="./torchvision"
#cd build
# Create a virtual environment
VENV_DIR=".build_venv"
python3 -m venv $VENV_DIR

# Activate the virtual environment
source $VENV_DIR/bin/activate

# Upgrade pip in the virtual environment
pip install --upgrade pip

# Download and install PyTorch wheel
echo "Downloading PyTorch wheel"
wget $PYTORCH_WHEEL_URL

# Clone TorchVision repository
echo "Cloning TorchVision version ${TORCHVISION_VERSION}"
git clone --branch v${TORCHVISION_VERSION} --recursive --depth=1 https://github.com/pytorch/vision torchvision

# Install dependencies for TorchVision
echo "Installing dependencies for TorchVision"
pip install 'numpy<2' torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install wheel


# Install TorchVision from the cloned repository
cd torchvision
python setup.py --verbose bdist_wheel --dist-dir ../
pip install torchvision-0.20.0a0+afc54f7-cp310-cp310-linux_aarch64.whl