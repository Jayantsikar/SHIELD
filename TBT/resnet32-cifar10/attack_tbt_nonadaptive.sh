#!/bin/bash
#SBATCH --job-name=tbt_resnet32_nonadaptive
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_tbt2_nonadaptive.out
#SBATCH --error=%j_tbt2_nonadaptive.err

echo "============================================================"
echo " TBT Non-Adaptive (TBT/resnet32-cifar10) — CIFAR-10"
echo " Job ID     : $SLURM_JOB_ID | Node: $(hostname)"
echo " Start time : $(date)"
echo "============================================================"

cd "$TMPDIR" || exit 1

echo "[1/3] Copying project files..."
# Copy the Aegis root so relative paths (../../cifar10/resnet32/) work
mkdir -p ./Aegis
cp -r ~/Aegis/. ./Aegis/
cd ./Aegis/TBT/resnet32-cifar10

echo "[2/3] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

# Get trained model from latest training job
if [ ! -f "../../cifar10/resnet32/save_finetune/model_best.pth.tar" ]; then
    LATEST=$(ls -td ~/job_results/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "${LATEST}save_finetune/model_best.pth.tar" ]; then
        mkdir -p ../../cifar10/resnet32/save_finetune/
        cp "${LATEST}save_finetune/model_best.pth.tar" ../../cifar10/resnet32/save_finetune/
    else
        echo "ERROR: Trained model not found."; exit 1
    fi
fi

nvidia-smi

echo "[3/3] Running TBT non-adaptive attack..."
python TBT_nonadaptive.py

EXIT_CODE=$?
mkdir -p ~/job_results/$SLURM_JOB_ID
cp -r ./ ~/job_results/$SLURM_JOB_ID/

echo "Finished at: $(date) | Exit: $EXIT_CODE"
exit $EXIT_CODE
