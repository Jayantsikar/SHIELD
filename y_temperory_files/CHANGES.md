# Aegis GPU Cluster — All Changes & Fix Log

This document tracks every bug found, every fix applied, and the exact commands needed to run the complete Aegis pipeline on the university GPU cluster (P100).

---

## Bugs Fixed

### Shell Scripts (`cifar10/resnet32/`)

| File | Bug | Fix |
|---|---|---|
| `train_CIFAR.sh` | `python` gives no live output in `.out` file | Changed to `python -u` (unbuffered) |
| `train_CIFAR.sh` | `--train_batch_size` not recognized by `main.py` | Changed to `--attack_sample_size` |
| `train_CIFAR.sh` | `--workers 4` causes DataLoader deadlock freeze | Changed to `--workers 0` |
| `train_finetune_branch.sh` | Same `--workers 4` deadlock bug | Changed to `--workers 0` |
| `train_finetune_branch.sh` | No `python -u` → no live output | Changed to `python -u` |
| `eval_CIFAR.sh` | Same `--workers 4` deadlock bug | Changed to `--workers 0` |
| `BFA_CIFAR.sh` | Same `--workers 4` deadlock bug | Changed to `--workers 0` |

### TBT Python Files (`TBT/resnet32-cifar10/`)

| File | Bug | Fix |
|---|---|---|
| All 4 TBT files | `num_workers=2` in DataLoaders causes deadlock | Changed all to `num_workers=0` |
| All 4 TBT files | `Variable(target, volatile=True)` deprecated in PyTorch 1.x+ | Replaced with `with torch.no_grad(): target_var = target` |
| `TBT_adaptive.py` | Wrong normalization (CIFAR-100 mean/std used for CIFAR-10) | Corrected to CIFAR-10: mean=[125.3,123.0,113.9], std=[63.0,62.1,66.7] |
| All 4 TBT files | `download=True` tries internet on compute node (no internet) | Changed to `download=False` |
| All 4 TBT files | All models saved to same `final_trojan.pth` (overwrites!) | Each script now saves to a unique filename |
| `TBT_adaptive.py` | No final result print/save at end of training | Added final print + saved to `./result/final_results_adaptive.txt` |
| `TBT_adaptive_ensemble.py` | No final result save | Added save to `./result/final_results_adaptive_ensemble.txt` |
| `TBT_nonadaptive_ensemble.py` | No final result save | Added save to `./result/final_results_nonadaptive_ensemble.txt` |

### SLURM Scripts (`TBT/resnet32-cifar10/`)

| File | Bug | Fix |
|---|---|---|
| `run_TBT.sh` (old) | Only ran `TBT_nonadaptive.py` | Created 4 separate scripts, one per TBT variant |
| All `.sh` files | Results in `$TMPDIR` which SLURM auto-deletes | Added `cp -r result/* $SLURM_SUBMIT_DIR/result/` at end |

---

## Complete Run Commands (in order)

### Step 1 — Upload everything to the cluster (run on your Mac terminal)
```bash
scp /Users/jayantkumar/Desktop/Aegis/cifar10/resnet32/*.sh k.jayant@172.16.112.202:~/Aegis/cifar10/resnet32/
scp /Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/*.sh k.jayant@172.16.112.202:~/Aegis/TBT/resnet32-cifar10/
scp /Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/*.py k.jayant@172.16.112.202:~/Aegis/TBT/resnet32-cifar10/
```

### Step 2 — Train base model (run on the cluster)
```bash
# Cancel the currently running stuck job first:
scancel 1131

cd ~/Aegis/cifar10/resnet32
sbatch train_CIFAR.sh
tail -f $(ls -t *.out | head -1)
# Wait until squeue -u k.jayant shows no jobs
```

### Step 3 — Fine-tune SDN branches (AFTER Step 2 finishes)
```bash
cd ~/Aegis/cifar10/resnet32
sbatch train_finetune_branch.sh
tail -f $(ls -t *.out | head -1)
# Wait until squeue -u k.jayant shows no jobs
```

### Step 4 — Run all 4 TBT attacks in parallel (AFTER Step 3 finishes)
```bash
cd ~/Aegis/TBT/resnet32-cifar10
sbatch run_TBT_nonadaptive.sh
sbatch run_TBT_adaptive.sh
sbatch run_TBT_nonadaptive_ensemble.sh
sbatch run_TBT_adaptive_ensemble.sh
```

### Step 5 — Download all results to your Mac
```bash
scp -r k.jayant@172.16.112.202:~/Aegis/TBT/resnet32-cifar10/result /Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/
```

---

## What Results are Saved

After all jobs finish, you will find in `~/Aegis/TBT/resnet32-cifar10/result/`:

| File | Description |
|---|---|
| `final_trojan.pth` | Trojaned model (non-adaptive) |
| `final_trojan_adaptive.pth` | Trojaned model (adaptive) |
| `final_trojan_nonadaptive_ensemble.pth` | Trojaned model (non-adaptive ensemble) |
| `final_trojan_adaptive_ensemble.pth` | Trojaned model (adaptive ensemble) |
| `final_results_nonadaptive.txt` | Clean Acc + ASR (non-adaptive) |
| `final_results_adaptive.txt` | Clean Acc + ASR (adaptive) |
| `final_results_nonadaptive_ensemble.txt` | Clean Acc + ASR (non-adaptive ensemble) |
| `final_results_adaptive_ensemble.txt` | Clean Acc + ASR (adaptive ensemble) |
| `resnet_trojan_img1/2/3.txt` | Generated trigger image channels |
| `resnet_trojan_test.txt` | Target neuron indices |
