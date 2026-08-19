#!/bin/bash
#SBATCH --job-name=resnet_finetune
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%j.out

# Activate the conda environment
source ~/miniconda/bin/activate pytorch_env

# THE MOST IMPORTANT RULE (DO NOT FORGET!)
# Always start your job script with cd "$TMPDIR" || exit 1
# This is the #1 reason jobs run slowly. Without this line, your job runs on slow network storage.
cd "$TMPDIR" || exit 1

# Copy the entire Aegis project to the fast SSD (includes data/ and save/model_best.pth.tar)
cp -r ~/Aegis ./ || exit 1
cd Aegis/cifar10/resnet32 || exit 1

# FIX: Point DATA_PATH to the fast local SSD copy, NOT home (slow network storage)
DATA_PATH="$TMPDIR/Aegis/data"

# Finetune save goes to fast SSD during training, synced to home every 10 min + at end
SAVE_PATH="./save_finetune/"
SAVE_HOME="$HOME/Aegis/cifar10/resnet32/save_finetune"
mkdir -p $SAVE_PATH
mkdir -p $SAVE_HOME

# AUTO-RESUME: Check if a finetune checkpoint already exists in home to resume from.
# Otherwise, start fresh finetuning from the base model trained in train_CIFAR.sh.
CHECKPOINT_HOME="${SAVE_HOME}/checkpoint.pth.tar"
BASE_MODEL="$TMPDIR/Aegis/cifar10/resnet32/save/model_best.pth.tar"

if [ -f "$CHECKPOINT_HOME" ]; then
    echo "=== Found existing finetune checkpoint — resuming from ${CHECKPOINT_HOME} ==="
    cp "$CHECKPOINT_HOME" "${SAVE_PATH}checkpoint.pth.tar"
    PRETRAINED_MODEL="${SAVE_PATH}checkpoint.pth.tar"
    FINE_TUNE_FLAG=""              # Resume: keep saved epoch, don't reset to 0
elif [ -f "$BASE_MODEL" ]; then
    echo "=== No finetune checkpoint found — starting fresh from base model (model_best.pth.tar) ==="
    PRETRAINED_MODEL="$BASE_MODEL"
    FINE_TUNE_FLAG="--fine_tune"   # Fresh start: reset epoch to 0
else
    echo "ERROR: Base model not found at $BASE_MODEL. Run train_CIFAR.sh first!" && exit 1
fi

# Force matplotlib to use non-GUI backend (prevents random freezes on headless GPU nodes)
export MPLBACKEND=Agg

# BACKGROUND SYNC: Copy finetune checkpoint to home every 10 minutes while training runs.
# This ensures progress is never lost if SLURM kills the job mid-run.
(
    while true; do
        sleep 600
        if [ -f "${SAVE_PATH}checkpoint.pth.tar" ]; then
            cp "${SAVE_PATH}checkpoint.pth.tar" "${SAVE_HOME}/checkpoint.pth.tar"
            [ -f "${SAVE_PATH}model_best.pth.tar" ] && \
                cp "${SAVE_PATH}model_best.pth.tar" "${SAVE_HOME}/model_best.pth.tar"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-synced finetune checkpoint to ${SAVE_HOME}"
        fi
    done
) &
SYNC_PID=$!
echo "=== Background checkpoint sync started (PID=${SYNC_PID}, every 10 min) ==="

# Run finetuning for 200 epochs
# -u flag = unbuffered output so logs print live to .out file
python -u main.py \
    --dataset cifar10 \
    --data_path ${DATA_PATH} \
    --arch resnet32_quan \
    --save_path ${SAVE_PATH} \
    --epochs 200 \
    --learning_rate 0.005 \
    --optimizer SGD \
    --schedule 80 120 \
    --gammas 0.1 0.1 \
    --attack_sample_size 128 \
    --test_batch_size 128 \
    --workers 4 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 100 \
    --decay 0.0003 \
    --momentum 0.9 \
    --resume "${PRETRAINED_MODEL}" \
    ${FINE_TUNE_FLAG} \
    --ic_only True \
    --adv_train

# Kill background sync
kill $SYNC_PID 2>/dev/null

# CRITICAL: Final sync — copy everything back to home before SLURM wipes TMPDIR
cp -r save_finetune/* "${SAVE_HOME}/"
echo "=== DONE: Finetuned model saved to ${SAVE_HOME} ==="