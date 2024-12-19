#!/bin/bash
source $VENV_DIR/bin/activate
wget https://nvidia.box.com/shared/static/i7n40ki3pl2x57vyn4u7e9asyiqlnl7n.whl -O onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
pip install onnxruntime_gpu-1.17.0-cp310-cp310-linux_aarch64.whl
pip install -r ../requirements.txt