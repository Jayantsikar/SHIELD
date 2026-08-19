#!/bin/bash
# =============================================================
# setup_env.sh — Run ONCE after SSH-ing into the cluster
# Sets up Miniconda + 'bfa' conda environment from environment.yml
# =============================================================

set -e  # exit on first error

echo "====== Cluster Environment Setup ======"
echo "User: $(whoami) | Node: $(hostname)"
echo "======================================="

# ---- Step 1: Install Miniconda (if not present) ----
if [ ! -d "$HOME/miniconda3" ] && [ ! -d "$HOME/anaconda3" ]; then
    echo "[1/4] Downloading and installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
    rm /tmp/miniconda.sh
    "$HOME/miniconda3/bin/conda" init bash
    echo "Miniconda installed. Please re-source your .bashrc:"
    echo "  source ~/.bashrc"
    echo "Then run this script again."
    exit 0
else
    echo "[1/4] Conda already installed — skipping."
fi

# Source conda
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh

# ---- Step 2: Create 'bfa' environment from environment.yml ----
if conda env list | grep -q "^bfa "; then
    echo "[2/4] 'bfa' conda environment already exists — skipping creation."
else
    echo "[2/4] Creating 'bfa' conda environment from environment.yml..."
    # Use a simplified modern requirements file (original env has old versions)
    conda create -n bfa python=3.8 -y
    conda activate bfa
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
        --extra-index-url https://download.pytorch.org/whl/cu117
    pip install tensorboardX pandas numpy matplotlib scipy scikit-learn \
        tqdm Pillow pyyaml yacs
fi

conda activate bfa

# ---- Step 3: Create project directory structure ----
echo "[3/4] Creating project directory structure..."
mkdir -p ~/Aegis/cifar10
mkdir -p ~/Aegis/TBT
mkdir -p ~/job_results

# ---- Step 4: Verify GPU ----
echo "[4/4] Verifying CUDA / PyTorch..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU            : {torch.cuda.get_device_name(0)}')
    print(f'CUDA version   : {torch.version.cuda}')
"

echo ""
echo "====== Setup Complete ======"
echo "Now transfer your files:"
echo ""
echo "  # From your Mac terminal (NOT this cluster terminal):"
echo "  scp -r '/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/cifar10/resnet32' \\"
echo "      k.jayant@172.16.112.202:~/Aegis/cifar10/"
echo ""
echo "  scp -r '/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/TBT/resnet32-cifar10' \\"
echo "      k.jayant@172.16.112.202:~/Aegis/TBT/"
echo ""
echo "Then submit jobs with:"
echo "  cd ~/Aegis/cifar10/resnet32"
echo "  sbatch train_gpu.sh"
