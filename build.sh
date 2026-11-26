#!/usr/bin/env bash
set -euo pipefail

# Install CPU-only PyTorch first to avoid downloading multi-GB CUDA wheels on Render
pip install --upgrade pip
pip install "torch>=2.2" --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# If Git LFS was not pulled during clone, rmbg14_model_weights.pth will be a small pointer file.
# Download the actual weights directly from GitHub LFS media storage.
WEIGHTS_FILE="rmbg14_model_weights.pth"
if [ ! -f "$WEIGHTS_FILE" ] || [ "$(wc -c < "$WEIGHTS_FILE")" -lt 1000000 ]; then
  echo "Downloading RMBG 1.4 model weights..."
  curl -fsSL -L -o "$WEIGHTS_FILE" \
    "https://media.githubusercontent.com/media/Maha-lingam07/NoBG/main/rmbg14_model_weights.pth"
fi
