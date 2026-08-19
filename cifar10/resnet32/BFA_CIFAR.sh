#!/bin/bash
#SBATCH --job-name=resnet_bfa
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=%j.out

# Activate the conda environment
source ~/miniconda/bin/activate pytorch_env

# CRITICAL: Change to fast local SSD
cd "$TMPDIR" || exit 1

# Copy the entire Aegis project to the fast SSD
cp -r ~/Aegis ./
cd Aegis/cifar10/resnet32

DATA_PATH="$HOME/Aegis/data"
SAVE_PATH="./save/cifar10_resnet32_quan_BFA_defense_test"
PRETRAINED_MODEL="./save_finetune/model_best.pth.tar"

mkdir -p $SAVE_PATH

python main.py \
    --dataset cifar10 \
    --data_path ${DATA_PATH} \
    --arch resnet32_quan \
    --save_path ${SAVE_PATH} \
    --test_batch_size 128 \
    --workers 0 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 50 \
    --evaluate \
    --resume ${PRETRAINED_MODEL} \
    --fine_tune \
    --reset_weight \
    --bfa \
    --n_iter 1000 \
    --attack_sample_size 128 \
    --bfa_mydefense

# Copy the results back to your home directory before the job ends!
mkdir -p "$SLURM_SUBMIT_DIR/save/cifar10_resnet32_quan_BFA_defense_test"
cp -r save/cifar10_resnet32_quan_BFA_defense_test/* "$SLURM_SUBMIT_DIR/save/cifar10_resnet32_quan_BFA_defense_test/"