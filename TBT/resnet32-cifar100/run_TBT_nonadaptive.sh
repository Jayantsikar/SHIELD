#!/bin/bash
#SBATCH --job-name=tbt_c100_nonadp
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

# 1. Copy TBT directory
mkdir -p Aegis/TBT
cp -r ~/Aegis/TBT/resnet32-cifar100 Aegis/TBT/

# 2. Copy trained model checkpoint
mkdir -p Aegis/cifar100/resnet32/save_finetune
if [ -f ~/Aegis/cifar100/resnet32/save_finetune/model_best.pth.tar ]; then
    cp ~/Aegis/cifar100/resnet32/save_finetune/model_best.pth.tar Aegis/cifar100/resnet32/save_finetune/
elif [ -f ~/Aegis/cifar100/resnet32/save/model_best.pth.tar ]; then
    cp ~/Aegis/cifar100/resnet32/save/model_best.pth.tar Aegis/cifar100/resnet32/save_finetune/
fi

# 3. Stage CIFAR-100 dataset
mkdir -p Aegis/cifar100/resnet32/data
if [ -d ~/Aegis/data/cifar-100-python ]; then
    cp -r ~/Aegis/data/cifar-100-python Aegis/cifar100/resnet32/data/
elif [ -d ~/Aegis/cifar100/resnet32/data/cifar-100-python ]; then
    cp -r ~/Aegis/cifar100/resnet32/data/cifar-100-python Aegis/cifar100/resnet32/data/
fi

echo "[SETUP] Done. Total staged: $(du -sh Aegis/ | cut -f1)"

cd Aegis/TBT/resnet32-cifar100
mkdir -p result

python -u TBT_nonadaptive.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Attack failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

# CRITICAL: Copy all results back before SLURM wipes TMPDIR
mkdir -p "$SLURM_SUBMIT_DIR/result"
cp -r result/* "$SLURM_SUBMIT_DIR/result/" 2>/dev/null || true
echo "=== DONE: Non-Adaptive TBT results saved to $SLURM_SUBMIT_DIR/result/ ==="
exit 0
