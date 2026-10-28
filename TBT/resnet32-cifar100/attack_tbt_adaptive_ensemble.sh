#!/bin/bash
#SBATCH --job-name=tbt_c100_adap_ens
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j_tbt_adaptive_ensemble.out
#SBATCH --error=%j_tbt_adaptive_ensemble.err

source ~/miniconda/bin/activate pytorch_env

cd "$TMPDIR" || exit 1

echo "[SETUP] Staging files to $TMPDIR ..."

mkdir -p Aegis/TBT
cp -r ~/Aegis/TBT/resnet32-cifar100 Aegis/TBT/

mkdir -p Aegis/cifar100/resnet32/save_finetune
if [ -f ~/Aegis/cifar100/resnet32/save_finetune/model_best.pth.tar ]; then
    cp ~/Aegis/cifar100/resnet32/save_finetune/model_best.pth.tar Aegis/cifar100/resnet32/save_finetune/
elif [ -f ~/Aegis/cifar100/resnet32/save/model_best.pth.tar ]; then
    cp ~/Aegis/cifar100/resnet32/save/model_best.pth.tar Aegis/cifar100/resnet32/save_finetune/
fi

mkdir -p Aegis/cifar100/resnet32/data
if [ -d ~/Aegis/data/cifar-100-python ]; then
    cp -r ~/Aegis/data/cifar-100-python Aegis/cifar100/resnet32/data/
elif [ -d ~/Aegis/cifar100/resnet32/data/cifar-100-python ]; then
    cp -r ~/Aegis/cifar100/resnet32/data/cifar-100-python Aegis/cifar100/resnet32/data/
fi

echo "[SETUP] Done. Total staged: $(du -sh Aegis/ | cut -f1)"

cd Aegis/TBT/resnet32-cifar100
mkdir -p result

python -u TBT_adaptive_ensemble.py
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo "ERROR: Attack failed with exit code $EXIT_CODE"
    exit $EXIT_CODE
fi

mkdir -p "$SLURM_SUBMIT_DIR/result"
cp -r result/* "$SLURM_SUBMIT_DIR/result/" 2>/dev/null || true
echo "=== DONE: Adaptive Ensemble TBT results saved to $SLURM_SUBMIT_DIR/result/ ==="
exit 0
