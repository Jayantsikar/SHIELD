# GPU Cluster Execution Plan — ResNet32 / CIFAR-10 (200 Epochs)

## Background

The project trains a **ResNet32** model on **CIFAR-10** and also runs
TBT (Targeted Bit-flip Trojan) attack scripts.  
The IIT Guwahati CSE cluster (`172.16.112.202`) uses **SLURM** for job
scheduling. Containers are run via **Apptainer** (formerly Singularity).  
Available GPUs: **NVIDIA H100 NVL** (gpu-H100) and **Tesla P100-16GB ×2**
(gpu-P100). The user account `k.jayant` needs P100 access at minimum
(btech/mtech level).

---

## What Files Need to Run

Files **without** "ensemble" in the name (as requested):

### `cifar10/resnet32/` (baseline training + evaluation)
| File | Purpose |
|---|---|
| `main.py` | Main training script (trains ResNet32 for 200 epochs) |
| `TBT_adaptive.py` | Adaptive TBT attack |
| `TBT_nonadaptive.py` | Non-adaptive TBT attack |
| `CSB_adaptive.py` | CSB adaptive attack |
| `CSB_nonadaptive.py` | CSB non-adaptive attack |
| `trigger_adaptive.py` | Trigger generation (adaptive) |
| `trigger_nonadaptive.py` | Trigger generation (non-adaptive) |

### `TBT/resnet32-cifar10/` (TBT attack variants)
| File | Purpose |
|---|---|
| `TBT_adaptive.py` | Adaptive attack (updated paths) |
| `TBT_nonadaptive.py` | Non-adaptive attack |

> [!NOTE]
> The `*_ensemble.py` files in `TBT/resnet32-cifar10/` are excluded per your
> request.

---

## Proposed Changes

### Step 1 — Fix the Training Script for 200 Epochs

#### [MODIFY] [train_CIFAR.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/train_CIFAR.sh)
- Change `epochs=50` → `epochs=200`
- Update `--schedule 80 120` → `--schedule 100 150` (standard schedule for 200
  epochs)
- Remove `--resume ./save/checkpoint.pth.tar` (start fresh unless you want to
  resume)
- Add `--workers 4` for faster data loading on cluster

### Step 2 — Fix Data Paths for the Cluster

All scripts currently have **hardcoded local data paths** like
`../../datasets/cifar10` or `/data2/workplace/ziyuan/...`. On the cluster,
data must be placed in `$TMPDIR` or a known location.

Files to patch (dataset root path → `/data/cifar10`):

#### [MODIFY] [TBT_adaptive.py (cifar10/resnet32)](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/TBT_adaptive.py)
- Line 94, 98: `root='../../datasets/cifar10'` → `root='./data/cifar10'`

#### [MODIFY] [TBT_nonadaptive.py (cifar10/resnet32)](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/TBT_nonadaptive.py)
- Line 99, 103: `root='/data2/workplace/ziyuan/datasets/cifar10'` → `root='./data/cifar10'`

#### [MODIFY] [CSB_adaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/CSB_adaptive.py)
- Line 50: `root='../../datasets/cifar10'` → `root='./data/cifar10'`

#### [MODIFY] [CSB_nonadaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/CSB_nonadaptive.py)
- Same data path fix

#### [MODIFY] [TBT_adaptive.py (TBT/resnet32-cifar10)](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/TBT_adaptive.py)
- Already has `../../cifar10/resnet32/data` path — good for local, but must be
  adjusted for cluster

#### [MODIFY] [TBT_nonadaptive.py (TBT/resnet32-cifar10)](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/TBT_nonadaptive.py)
- Same fix

### Step 3 — Fix Model Checkpoint Paths for Attack Scripts

The attack scripts (`TBT_*.py`, `CSB_*.py`) load from:
```
./save_finetune/model_best.pth.tar
```
This requires that `main.py` has already been trained and the checkpoint exists.
**The training job must complete before the attack scripts run.**

### Step 4 — Create SLURM Job Scripts

#### [NEW] `train_gpu.sh` — Main Training Job (200 epochs)

```bash
#!/bin/bash
#SBATCH --job-name=resnet32_train
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_train.out
#SBATCH --error=%j_train.err

cd "$TMPDIR" || exit 1

# Copy project files to local scratch
cp -r ~/Aegis/cifar10/resnet32 ./resnet32
cd ./resnet32

# Run training
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

# Copy results back
mkdir -p ~/job_results/$SLURM_JOB_ID
cp -r ./save_finetune ~/job_results/$SLURM_JOB_ID/
```

#### [NEW] `attack_tbt_nonadaptive.sh` — TBT Non-Adaptive Attack Job

```bash
#!/bin/bash
#SBATCH --job-name=tbt_nonadaptive
#SBATCH --partition=gpu-P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%j_tbt_nonadaptive.out
#SBATCH --error=%j_tbt_nonadaptive.err

cd "$TMPDIR" || exit 1
cp -r ~/Aegis/cifar10/resnet32 ./resnet32
cd ./resnet32

# Requires trained model in save_finetune/
python TBT_nonadaptive.py

mkdir -p ~/job_results/$SLURM_JOB_ID
cp -r ./ ~/job_results/$SLURM_JOB_ID/
```

#### [NEW] `attack_tbt_adaptive.sh` — TBT Adaptive Attack Job

Similar to above, but runs `TBT_adaptive.py`.

---

## Verification Plan

### Step-by-step order of operations

```
1. Train model (200 epochs)     ← run first
2. Run attack scripts           ← only after step 1 completes
```

> [!IMPORTANT]
> The attack scripts (`TBT_*.py`, `CSB_*.py`) require a trained model checkpoint
> at `./save_finetune/model_best.pth.tar`. Do NOT run them in parallel with
> training.

### Automated Verification
- Check `squeue -u k.jayant` for job status
- After training completes: `ls ~/job_results/<jobid>/save_finetune/`
- Check output log for final test accuracy

---

## ⏱️ Expected Time Estimate

| Component | Time Estimate |
|---|---|
| ResNet32 CIFAR-10 training (200 epochs, P100) | **~1.5–2.5 hours** |
| TBT Non-Adaptive Attack | ~20–40 minutes |
| TBT Adaptive Attack | ~30–60 minutes |
| CSB attacks (×2) | ~30–60 minutes total |
| **Total** | **~3–5 hours** |

> [!NOTE]
> On an **H100** (if accessible), training would be ~**30–45 minutes** instead.
> P100 estimates assume batch size 128, single GPU.

---

## Commands to Run on Your Machine

### 1. Transfer files to the cluster

```bash
# Run this in your terminal (Mac)
scp -r "/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/cifar10/resnet32" \
    k.jayant@172.16.112.202:~/Aegis/cifar10/

scp -r "/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/TBT/resnet32-cifar10" \
    k.jayant@172.16.112.202:~/Aegis/TBT/
```

### 2. SSH into the cluster

```bash
ssh k.jayant@172.16.112.202
# Password: 0101jaya
```

### 3. Submit the training job

```bash
# After SSH-ing in:
cd ~/Aegis/cifar10/resnet32
sbatch train_gpu.sh
```

### 4. Monitor job

```bash
squeue -u k.jayant
# Check output log when running:
tail -f ~/job_results/<jobid>/<jobid>_train.out
```

### 5. After training completes, run attacks

```bash
sbatch attack_tbt_nonadaptive.sh
sbatch attack_tbt_adaptive.sh
```

---

## Open Questions

> [!IMPORTANT]
> **Account type**: The manual says `gpu-P100` requires account type `btech` or
> higher. What is `k.jayant`'s account type? Run `sacctmgr show user k.jayant
> withassoc` after login to check. If it's just a basic user, you may only have
> CPU access.

> [!IMPORTANT]
> **Pre-trained checkpoint**: Do you already have a pre-trained/fine-tuned
> checkpoint (`model_best.pth.tar`) or do you need to train from scratch? The
> attack scripts require this file to exist.

> [!NOTE]
> **Python environment**: The cluster may not have your conda env. You may
> need to either use an Apptainer container or create a conda env after SSH-ing
> in. The `environment.yml` can be used: `conda env create -f environment.yml`.
