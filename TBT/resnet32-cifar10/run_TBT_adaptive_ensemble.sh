#!/bin/bash
#SBATCH --job-name=tbt_adap_ens
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

source ~/miniconda/bin/activate pytorch_env

# CRITICAL: Change to fast local SSD (required by cluster policy)
cd "$TMPDIR" || exit 1

echo "[SETUP] Staging files to $TMPDIR ..."

# 1. Copy ONLY the TBT scripts directory (~7 MB) — not the whole Aegis repo
mkdir -p Aegis/TBT
cp -r ~/Aegis/TBT/resnet32-cifar10 Aegis/TBT/

# 2. Copy ONLY the model checkpoint needed by the script (~12 MB)
mkdir -p Aegis/cifar10/resnet32/save_finetune
cp ~/Aegis/cifar10/resnet32/save_finetune/model_best.pth.tar \
   Aegis/cifar10/resnet32/save_finetune/

# 3. Copy ONLY the CIFAR-10 dataset (178 MB, copied once)
mkdir -p Aegis/cifar10/resnet32/data
cp -r ~/Aegis/cifar10/resnet32/data/cifar-10-batches-py \
      Aegis/cifar10/resnet32/data/

echo "[SETUP] Done. Total staged: $(du -sh Aegis/ | cut -f1)"

cd Aegis/TBT/resnet32-cifar10
python -u TBT_adaptive_ensemble.py

# CRITICAL: Copy all results back before SLURM wipes TMPDIR
mkdir -p "$SLURM_SUBMIT_DIR/result"
cp -r result/* "$SLURM_SUBMIT_DIR/result/"
echo "=== DONE: Adaptive Ensemble TBT results saved to $SLURM_SUBMIT_DIR/result/ ==="
