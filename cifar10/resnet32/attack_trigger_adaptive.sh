#!/bin/bash
#SBATCH --job-name=trigger_adaptive
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_trigger_adaptive.out
#SBATCH --error=%j_trigger_adaptive.err

echo "============================================================"
echo " Trigger Adaptive — ResNet32 / CIFAR-10"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $(hostname)"
echo " Start time : $(date)"
echo "============================================================"

cd "$TMPDIR" || exit 1

echo "[1/3] Copying project files..."
mkdir -p ./resnet32
cp -r ~/Aegis/cifar10/resnet32/. ./resnet32/
cd ./resnet32

if [ ! -f "./save_finetune/model_best.pth.tar" ]; then
    LATEST=$(ls -td ~/job_results/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "${LATEST}save_finetune/model_best.pth.tar" ]; then
        cp -r "${LATEST}save_finetune" ./
    else
        echo "ERROR: Trained model not found. Run train_gpu.sh first."; exit 1
    fi
fi

echo "[2/3] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

nvidia-smi

echo "[3/3] Running trigger adaptive..."
python trigger_adaptive.py

EXIT_CODE=$?
mkdir -p ~/job_results/$SLURM_JOB_ID
mkdir -p ~/Aegis/results/cifar10/resnet32/trigger_adaptive
cp -r ./ ~/job_results/$SLURM_JOB_ID/
cp ${SLURM_JOB_ID}_trigger_adaptive.out ~/Aegis/results/cifar10/resnet32/trigger_adaptive/ 2>/dev/null
cp ${SLURM_JOB_ID}_trigger_adaptive.err ~/Aegis/results/cifar10/resnet32/trigger_adaptive/ 2>/dev/null

echo "Finished at: $(date) | Exit: $EXIT_CODE"
exit $EXIT_CODE
