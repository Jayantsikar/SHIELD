#!/bin/bash
#SBATCH --job-name=resnet32_stl10_train
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=%j_train_stl10.out

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/miniconda/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found."; exit 1; }
conda activate bfa

# THE MOST IMPORTANT RULE (DO NOT FORGET!)
# Always start your job script with cd "$TMPDIR" || exit 1
# This is the #1 reason jobs run slowly. Without this line, your job runs on slow network storage.
cd "$TMPDIR" || exit 1

# Copy the entire Aegis project to the fast local SSD
cp -r ~/Aegis ./ || exit 1
cd Aegis/stl10/resnet32 || exit 1

# FIX: Point DATA_PATH to the fast local SSD copy, NOT home (slow network storage)
DATA_PATH="$TMPDIR/Aegis/stl10/resnet32/data"
SAVE_PATH="./save/"
SAVE_HOME="$HOME/Aegis/stl10/resnet32/save"
mkdir -p $SAVE_PATH

# AUTO-RESUME: Copy any existing checkpoint from home → fast SSD before training.
# This lets the job resume from where it left off if it was interrupted.
CHECKPOINT_HOME="${SAVE_HOME}/checkpoint.pth.tar"
CHECKPOINT_LOCAL="${SAVE_PATH}checkpoint.pth.tar"

if [ -f "$CHECKPOINT_HOME" ]; then
    echo "=== Found existing checkpoint at $CHECKPOINT_HOME — copying to fast SSD for resume ==="
    cp "$CHECKPOINT_HOME" "$CHECKPOINT_LOCAL"
    RESUME_FLAG="--resume ${CHECKPOINT_LOCAL}"
else
    echo "=== No checkpoint found — training from scratch ==="
    RESUME_FLAG=""
fi

# Force matplotlib to use non-GUI backend (prevents random freezes on headless GPU nodes)
export MPLBACKEND=Agg

# BACKGROUND SYNC: Copy checkpoint to home every 10 minutes while training runs.
# This ensures progress is never lost if SLURM kills the job mid-run or SSH drops.
mkdir -p "${SAVE_HOME}"
(
    while true; do
        sleep 600
        if [ -f "${CHECKPOINT_LOCAL}" ]; then
            cp "${CHECKPOINT_LOCAL}" "${SAVE_HOME}/checkpoint.pth.tar"
            # Also sync model_best if it exists
            [ -f "${SAVE_PATH}model_best.pth.tar" ] && \
                cp "${SAVE_PATH}model_best.pth.tar" "${SAVE_HOME}/model_best.pth.tar"
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] Auto-synced checkpoint to ${SAVE_HOME}"
        fi
    done
) &
SYNC_PID=$!
echo "=== Background checkpoint sync started (PID=${SYNC_PID}, every 10 min) ==="

# Run training
# STL-10: 96x96 images, 10 classes, 5000 train / 8000 test
# -u flag = unbuffered output so logs print live to .out file
python -u main.py \
    --dataset stl10 \
    --data_path ${DATA_PATH} \
    --arch resnet32_quan \
    --save_path ${SAVE_PATH} \
    --epochs 200 \
    --learning_rate 0.1 \
    --optimizer SGD \
    --schedule 80 120 160 \
    --gammas 0.1 0.1 0.1 \
    --attack_sample_size 128 \
    --test_batch_size 128 \
    --workers 4 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 100 \
    --decay 0.0005 \
    --momentum 0.9 \
    ${RESUME_FLAG}

# Kill the background sync now that training is done
kill $SYNC_PID 2>/dev/null

# CRITICAL: Final sync — copy the entire save/ back to home before SLURM wipes TMPDIR.
mkdir -p "${SAVE_HOME}"
cp -r save/* "${SAVE_HOME}/"
echo "=== DONE: Model/checkpoint saved to ${SAVE_HOME} ==="