#!/bin/bash
#SBATCH --job-name=resnet32_train_200ep
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_train.out
#SBATCH --error=%j_train.err

echo "============================================================"
echo " ResNet32 CIFAR-10 Training — 200 Epochs"
echo " Job ID     : $SLURM_JOB_ID"
echo " Node       : $(hostname)"
echo " Start time : $(date)"
echo "============================================================"

# CRITICAL: Always run from local SSD, not network storage
cd "$TMPDIR" || exit 1

# ---- Copy project to local scratch ----
echo "[1/4] Copying project files to local scratch..."
mkdir -p ./resnet32
cp -r ~/Aegis/cifar10/resnet32/. ./resnet32/
cd ./resnet32

# ---- Activate conda environment ----
echo "[2/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null \
    || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null \
    || { echo "ERROR: conda not found. Run setup_env.sh first."; exit 1; }
conda activate bfa

# ---- Verify GPU ----
echo "[3/4] GPU status:"
nvidia-smi

# ---- Pre-flight import check ----
echo "Checking Python imports..."
python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
try:
    from tensorboardX import SummaryWriter
    print('tensorboardX: OK')
except ImportError as e:
    print(f'tensorboardX MISSING: {e}')
try:
    from attack.BFA import BFA
    print('attack.BFA: OK')
except ImportError as e:
    print(f'attack.BFA MISSING: {e}')
try:
    import models
    print('models: OK')
except ImportError as e:
    print(f'models MISSING: {e}')
" 2>&1 || { echo 'Pre-flight check failed — see above error'; exit 1; }

# ---- Run training ----
echo "[4/4] Starting training..."
python main.py \
    --dataset cifar10 \
    --data_path ./data \
    --arch resnet32_quan \
    --save_path ./save_finetune/ \
    --epochs 200 \
    --learning_rate 0.01 \
    --optimizer SGD \
    --schedule 100 150 \
    --gammas 0.1 0.1 \
    --attack_sample_size 128 \
    --test_batch_size 256 \
    --workers 4 \
    --ngpu 1 --gpu_id 0 \
    --print_freq 100 \
    --decay 0.0003 \
    --momentum 0.9

TRAIN_EXIT=$?

# ---- Copy results back to home ----
echo "Copying results back to ~/job_results/$SLURM_JOB_ID/ ..."
mkdir -p ~/job_results/$SLURM_JOB_ID
cp -r ./save_finetune ~/job_results/$SLURM_JOB_ID/

echo "============================================================"
echo " Training finished at: $(date)"
echo " Exit code: $TRAIN_EXIT"
echo " Results saved to: ~/job_results/$SLURM_JOB_ID/"
echo "============================================================"
exit $TRAIN_EXIT
