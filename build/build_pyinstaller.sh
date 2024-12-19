#!/bin/bash
VENV_DIR=".build_venv"
sudo apt-get install cmake python3-pip libopenblas-base libopenmpi-dev libomp-dev libjpeg-dev zlib1g-dev libpython3-dev libavcodec-dev libavformat-dev libswscale-dev

source $VENV_DIR/bin/activate
pip install -r ../requirements.txt
wget https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl -O onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
# Execute PyInstaller with the provided paths
python -m PyInstaller --workpath "$WORKPATH" --distpath "$DISTPATH" main.spec
