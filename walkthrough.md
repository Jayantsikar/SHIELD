# GPU Cluster Walkthrough — ResNet32 / CIFAR-10 (200 Epochs)

## Changes Made to Your Code

### `cifar10/resnet32/` — Fixes Applied

| File | What Changed |
|---|---|
| [train_CIFAR.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/train_CIFAR.sh) | `epochs 50→200`, `schedule 80 120→100 150`, `workers 1→4`, removed stale `--resume` |
| [TBT_adaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/TBT_adaptive.py) | Data path `../../datasets/cifar10` → `./data/cifar10` |
| [TBT_nonadaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/TBT_nonadaptive.py) | Data path `/data2/workplace/...` → `./data/cifar10`; removed `CUDA_VISIBLE_DEVICES=4` |
| [CSB_adaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/CSB_adaptive.py) | Data path fixed |
| [CSB_nonadaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/CSB_nonadaptive.py) | Data path fixed |
| [trigger_adaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/trigger_adaptive.py) | Data path fixed |
| [trigger_nonadaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/trigger_nonadaptive.py) | Data path fixed |

### `TBT/resnet32-cifar10/` — Fixes Applied

| File | What Changed |
|---|---|
| [TBT_adaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/TBT_adaptive.py) | Removed `CUDA_VISIBLE_DEVICES='0'` (SLURM manages this) |
| [TBT_nonadaptive.py](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/TBT_nonadaptive.py) | Removed `CUDA_VISIBLE_DEVICES='0'` |

### New SLURM Job Scripts Created

#### `cifar10/resnet32/`
| Script | Purpose |
|---|---|
| [train_gpu.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/train_gpu.sh) | **200-epoch training** (run first!) |
| [attack_tbt_nonadaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_tbt_nonadaptive.sh) | TBT Non-Adaptive attack |
| [attack_tbt_adaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_tbt_adaptive.sh) | TBT Adaptive attack |
| [attack_csb_nonadaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_csb_nonadaptive.sh) | CSB Non-Adaptive attack |
| [attack_csb_adaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_csb_adaptive.sh) | CSB Adaptive attack |
| [attack_trigger_nonadaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_trigger_nonadaptive.sh) | Trigger Non-Adaptive |
| [attack_trigger_adaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/attack_trigger_adaptive.sh) | Trigger Adaptive |
| [setup_env.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/cifar10/resnet32/setup_env.sh) | One-time conda setup on cluster |

#### `TBT/resnet32-cifar10/`
| Script | Purpose |
|---|---|
| [attack_tbt_nonadaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/attack_tbt_nonadaptive.sh) | TBT Non-Adaptive (TBT folder) |
| [attack_tbt_adaptive.sh](file:///Users/devalsinghal/Desktop/sem%207/BTP/Aegis/TBT/resnet32-cifar10/attack_tbt_adaptive.sh) | TBT Adaptive (TBT folder) |

---

## Step-by-Step: What YOU Need to Do

### Step 1 — Transfer files from Mac to cluster

Open a **new terminal on your Mac** and run:

```bash
# Transfer cifar10/resnet32 folder
scp -r "/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/cifar10/resnet32" \
    k.jayant@172.16.112.202:~/Aegis/cifar10/

# Transfer TBT/resnet32-cifar10 folder
scp -r "/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/TBT/resnet32-cifar10" \
    k.jayant@172.16.112.202:~/Aegis/TBT/
```
> **Password when prompted:** `0101jaya`

---

### Step 2 — SSH into the cluster

```bash
ssh k.jayant@172.16.112.202
# Password: 0101jaya
```

---

### Step 3 — Set up Python environment (FIRST TIME ONLY)

```bash
# On the cluster terminal:
cd ~/Aegis/cifar10/resnet32
bash setup_env.sh
```

If it asks you to re-source, run:
```bash
source ~/.bashrc
bash setup_env.sh   # run again
```

> [!NOTE]
> If conda/miniconda is already on the cluster, you can skip this and just run:
> `conda env create -f environment.yml` from the resnet32 folder.

---

### Step 4 — Check your account access

```bash
# Check what GPU partitions you can use:
sacctmgr show user k.jayant withassoc

# Check partition/node status:
sinfo
```

If `gpu-P100` is listed under your account, you're good. If not, contact the
cluster admin.

---

### Step 5 — Submit Training Job (200 epochs)

```bash
cd ~/Aegis/cifar10/resnet32
sbatch train_gpu.sh
```

You'll get a **job ID** like `12345`. Note it down.

**Monitor the job:**
```bash
# Check if it's running
squeue -u k.jayant

# Watch the live training log (replace 12345 with your job ID)
tail -f ~/job_results/12345/12345_train.out
```

> [!IMPORTANT]
> **Wait for training to fully complete** before running any attack scripts.
> The attack scripts need `save_finetune/model_best.pth.tar` to exist.

---

### Step 6 — After training completes, run attacks

> [!NOTE]
> The cluster allows max **2 jobs submitted at a time, 1 running at a time**.
> Submit attack jobs one by one or wait.

```bash
cd ~/Aegis/cifar10/resnet32

# Run attacks (submit one at a time):
sbatch attack_tbt_nonadaptive.sh
sbatch attack_tbt_adaptive.sh
sbatch attack_csb_nonadaptive.sh
sbatch attack_csb_adaptive.sh
sbatch attack_trigger_nonadaptive.sh
sbatch attack_trigger_adaptive.sh
```

For the TBT folder variants:
```bash
cd ~/Aegis/TBT/resnet32-cifar10
sbatch attack_tbt_nonadaptive.sh
sbatch attack_tbt_adaptive.sh
```

---

### Step 7 — Retrieve results back to your Mac

```bash
# From your Mac terminal (NOT cluster):
scp -r k.jayant@172.16.112.202:~/job_results/ \
    "/Users/devalsinghal/Desktop/sem 7/BTP/Aegis/cluster_results/"
```

---

## ⏱️ Expected Time Estimates

| Task | P100 GPU | H100 GPU (if accessible) |
|---|---|---|
| 200-epoch training | **~1.5–2.5 hours** | ~30–45 min |
| TBT Non-Adaptive | ~20–40 min | ~10 min |
| TBT Adaptive | ~30–60 min | ~15 min |
| CSB attacks (×2) | ~30–60 min total | ~15 min |
| Trigger (×2) | ~20–40 min total | ~10 min |
| **Total** | **~3–5 hours** | ~1–2 hours |

---

## Quick Reference — Monitoring Commands

```bash
squeue -u k.jayant                    # your jobs in queue
sinfo                                 # cluster / partition status
sinfo -p gpu-P100                     # P100 GPU status
scontrol show job <jobid>             # detailed job info
sacct -j <jobid>                      # job accounting after completion
scancel <jobid>                       # cancel a job
ls ~/job_results/                     # see all completed results
nvidia-smi                            # GPU usage (inside a job only)
```

---

## ⚠️ Important Notes

> [!CAUTION]
> **Home directory quota:** Soft limit is 50 GB. Clear old files before
> submitting. Run `du -sh ~/` to check usage.

> [!WARNING]
> **Job execution order matters:**
> `train_gpu.sh` → then all attack scripts.
> Attack scripts auto-detect the latest training result in `~/job_results/`.

> [!NOTE]
> All SLURM scripts use `cd "$TMPDIR" || exit 1` as the cluster manual
> requires. This runs jobs on fast local SSD instead of slow network storage.
