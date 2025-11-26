#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
ENV_NAME=".venv"
PYTHON_MODULE="Python3/3.13.7"
CUDA_MODULE="cuda/12.6.3"
GCC_MODULE="gcc/14.2.0"

echo "=========================================="
echo "   Initializing HPC Python Environment    "
echo "=========================================="

# 1. Load System Modules
# We check if the 'module' command exists to make the script portable
if command -v module &> /dev/null; then
    echo "[1/6] Loading HPC Modules..."
    module purge
    module load "$PYTHON_MODULE"
    module load "$CUDA_MODULE"
    module load "$GCC_MODULE"
    echo "      Loaded: $PYTHON_MODULE, $CUDA_MODULE, $GCC_MODULE"
else
    echo "[!] 'module' command not found. Assuming running on local machine or Docker."
    echo "    Ensure correct Python, CUDA, and GCC versions are in your PATH."
fi

# 2. Check for 'uv' (Fast Python Package Installer)
if ! command -v uv &> /dev/null; then
    echo "[2/6] 'uv' not found. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.cargo/env"
else
    echo "[2/6] 'uv' detected."
fi

# 3. Create Virtual Environment
# We use the system python (from the module load) to ensure we get HPC optimizations
if [ ! -d "$ENV_NAME" ]; then
    echo "[3/6] Creating virtual environment '$ENV_NAME' using $(which python3)..."
    python3 -m venv "$ENV_NAME"
else
    echo "[3/6] Virtual environment '$ENV_NAME' already exists."
fi

# 4. Activate Environment
echo "[4/6] Activating environment..."
source "$ENV_NAME/bin/activate"

# 5. Install Core Dependencies (Torch Ecosystem)
echo "[5/6] Installing PyTorch 2.6.0 (CUDA 12.6)..."
uv pip install torch==2.6.0 \
    torchvision==0.21.0 \
    torchaudio==2.6.0 \
    --index-url https://download.pytorch.org/whl/cu126

# 6. Install Scientific & Utility Packages
echo "[6/6] Installing libraries and compiling Flash Attention..."

# Upgrade build tools first
uv pip install --upgrade pip wheel setuptools

# Flash Attention Compilation
# MAX_JOBS=64 can be aggressive; ensure you are on a compute node, not a login node.
export MAX_JOBS=64 
uv pip install flash-attn==2.7.4.post1 --no-build-isolation

# Remaining utilities
uv pip install jsonargparse transformers biopython wandb

# Cleanup
if uv pip show triton &> /dev/null; then
    echo "      Uninstalling triton..."
    uv pip uninstall triton
fi

echo "=========================================="
echo "   Setup Complete!"
echo "   Run the following to start working:"
echo "   source $ENV_NAME/bin/activate"
echo "=========================================="