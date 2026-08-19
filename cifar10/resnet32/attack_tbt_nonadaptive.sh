#!/bin/bash
#SBATCH --job-name=tbt_nonadaptive
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_tbt_nonadaptive.out
#SBATCH --error=%j_tbt_nonadaptive.err

echo "============================================================"
echo " TBT Non-Adaptive Attack — ResNet32 / CIFAR-10"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $(hostname)"
echo " Start time : $(date)"
echo "============================================================"

cd "$TMPDIR" || exit 1

# Copy project + trained weights
echo "[1/3] Copying project files..."
mkdir -p ./resnet32
cp -r ~/Aegis/cifar10/resnet32/. ./resnet32/
cd ./resnet32

# Check that trained model exists
if [ ! -f "./save_finetune/model_best.pth.tar" ]; then
    # Try to fetch from previous training job results
    LATEST=$(ls -td ~/job_results/*/ 2>/dev/null | head -1)
    if [ -n "$LATEST" ] && [ -f "${LATEST}save_finetune/model_best.pth.tar" ]; then
        echo "Found trained model in $LATEST"
        cp -r "${LATEST}save_finetune" ./
    else
        echo "ERROR: No trained model found at save_finetune/model_best.pth.tar"
        echo "Please run train_gpu.sh first and copy checkpoint."
        exit 1
    fi
fi

# Activate conda environment
echo "[2/3] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

nvidia-smi

# Run attack
echo "[3/3] Running TBT non-adaptive attack..."
python TBT_nonadaptive.py

EXIT_CODE=$?

# Copy results
mkdir -p ~/job_results/$SLURM_JOB_ID
mkdir -p ~/Aegis/results/cifar10/resnet32/tbt_nonadaptive
cp -r ./ ~/job_results/$SLURM_JOB_ID/
cp ${SLURM_JOB_ID}_tbt_nonadaptive.out ~/Aegis/results/cifar10/resnet32/tbt_nonadaptive/ 2>/dev/null
cp ${SLURM_JOB_ID}_tbt_nonadaptive.err ~/Aegis/results/cifar10/resnet32/tbt_nonadaptive/ 2>/dev/null

echo "============================================================"
echo " Attack finished at: $(date) | Exit: $EXIT_CODE"
echo "============================================================"
exit $EXIT_CODE
