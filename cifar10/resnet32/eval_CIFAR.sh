#!/bin/bash
#SBATCH --job-name=resnet_eval
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=%j.out

# Activate the conda environment
source ~/miniconda/bin/activate pytorch_env

# CRITICAL: Change to fast local SSD
cd "$TMPDIR" || exit 1

# Copy the entire Aegis project to the fast SSD
cp -r ~/Aegis ./
cd Aegis/cifar10/resnet32

DATA_PATH="$HOME/Aegis/data"
DATE=`date +%Y-%m-%d`
SAVE_PATH="./save/${DATE}/cifar10_vanilla_resnet20_new_exp"
PRETRAINED_MODEL="./save_finetune/model_best.pth.tar"

mkdir -p $SAVE_PATH

python main.py \
    --dataset cifar10 \
    --data_path ${DATA_PATH} \
    --arch vanilla_resnet20 \
    --save_path ${SAVE_PATH} \
    --epochs 200 \
    --learning_rate 0.1 \
    --optimizer SGD \
    --schedule 80 120 160 \
    --gammas 0.1 0.1 0.1 \
    --test_batch_size 100 \
    --workers 0 \
    --ngpu 1 \
    --gpu_id 0 \
    --print_freq 100 \
    --decay 0.0003 \
    --momentum 0.9 \
    --evaluate \
    --resume ${PRETRAINED_MODEL} \
    --fine_tune \
    --attack_sample_size 128 \
    --reset_weight \
    --bfa \
    --n_iter 3 \
    --k_top 10

# Copy the results back to your home directory before the job ends!
mkdir -p "$SLURM_SUBMIT_DIR/save/${DATE}/cifar10_vanilla_resnet20_new_exp"
cp -r save/${DATE}/cifar10_vanilla_resnet20_new_exp/* "$SLURM_SUBMIT_DIR/save/${DATE}/cifar10_vanilla_resnet20_new_exp/"