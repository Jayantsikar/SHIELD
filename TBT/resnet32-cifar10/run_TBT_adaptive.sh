#!/bin/bash
#SBATCH --job-name=tbt_adaptive
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

source ~/miniconda/bin/activate pytorch_env

# CRITICAL: Change to fast local SSD (required by cluster policy)
cd "$TMPDIR" || exit 1

cp -r ~/Aegis ./
# Stage the pre-downloaded dataset into the location TBT_adaptive.py expects
mkdir -p Aegis/cifar10/resnet32/data
cp -r ~/Aegis/data/cifar-10-batches-py Aegis/cifar10/resnet32/data/

cd Aegis/TBT/resnet32-cifar10
python -u TBT_adaptive.py

# CRITICAL: Copy all results back before SLURM wipes TMPDIR
mkdir -p "$SLURM_SUBMIT_DIR/result"
cp -r result/* "$SLURM_SUBMIT_DIR/result/"
echo "=== DONE: Adaptive TBT results saved to $SLURM_SUBMIT_DIR/result/ ==="
