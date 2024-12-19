#!/bin/bash
VENV_DIR=".build_venv"

source $VENV_DIR/bin/activate

# Execute PyInstaller with the provided paths
python -m PyInstaller --workpath "$WORKPATH" --distpath "$DISTPATH" main.spec
