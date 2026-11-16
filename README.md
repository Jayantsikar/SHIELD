# ResilientDNN / Aegis: Two-Phase Defense for Multi-Exit DNNs Against Targeted Bit-Flip Attacks

This repository contains the official implementation of **ResilientDNN**, a resilient two-phase defense architecture designed to protect multi-exit Deep Neural Networks (SDNs) against physical parameter tampering and **Targeted Bit-Flip Attacks (TBFAs)**, including **Targeted Bit Trojan (TBT)**, **Progressive Bit-Flip Attack (BFA)**, **TA-LBF**, and **ProFlip**.

---

## Table of Contents

- [1. Project Overview](#1-project-overview)
- [2. Repository Architecture](#2-repository-architecture)
- [3. Environment Setup](#3-environment-setup)
- [4. Dataset Preparation](#4-dataset-preparation)
- [5. Model Training & Quantization](#5-model-training--quantization)
- [6. Running Attacks & Defense Evaluation](#6-running-attacks--defense-evaluation)
  - [6.1 TBT (Targeted Bit Trojan) Attacks](#61-tbt-targeted-bit-trojan-attacks)
  - [6.2 Running on a Cluster (SLURM)](#62-running-on-a-cluster-slurm)
  - [6.3 Progressive Bit-Flip Attacks (BFA)](#63-progressive-bit-flip-attacks-bfa)
  - [6.4 TA-LBF & ProFlip Attacks](#64-ta-lbf--proflip-attacks)
- [7. Understanding Outputs & Metrics](#7-understanding-outputs--metrics)
- [8. Regenerating Figures & Visualizations](#8-regenerating-figures--visualizations)
- [9. Benchmark Results Reference](#9-benchmark-results-reference)
- [10. Troubleshooting & FAQ](#10-troubleshooting--faq)

---

## 1. Project Overview

### The Vulnerability
Multi-Exit Neural Networks (such as Shallow-Deep Networks / BranchyNet) add intermediate classifiers (ICs) to hidden layers to enable early-exit inference for low latency on edge devices. However, hardware fault-injection attacks (e.g., DRAM Rowhammer) can flip a tiny fraction of model weight bits in memory. An adversary can easily engineer a small trigger patch that forces an early IC to emit an artificial high-confidence prediction. In existing early-exit systems (like vanilla Aegis), this triggers an **early-exit short-circuit**, hijacking inference at a single corrupted branch.

### The Two-Phase Defense
ResilientDNN resolves this single-branch dependency through two integrated layers:
1. **Phase 1: Stochastic Branch Ensembling (SBE):**
   Instead of halting at the first branch crossing a static threshold ($\tau = 0.95$), SBE randomly samples a candidate subset of active branches ($C = 5$) and aggregates their predictions via confidence-weighted soft consensus voting. This diffuses the malicious influence of isolated compromised branches.
2. **Phase 2: Checksum-Based Self-Healing Quarantine:**
   Leveraging deterministic 8-bit integer quantization ($W_q \in \mathbb{Z}^8$), the system computes on-device SHA-256 parameter digests and verifies them against immutable golden fingerprints stored in secure read-only memory (ROM/TPM). When bit tampering is detected, the compromised head is quarantined ($\mu_k = 0$), allowing remaining clean ICs to self-heal predictions with zero retraining and negligible runtime overhead ($<0.19\,\text{ms}$).

---

## 2. Repository Architecture

```text
Aegis/
├── cifar10/               # CIFAR-10 training, evaluation, and baseline scripts (ResNet-32, VGG-16)
├── cifar100/              # CIFAR-100 training, evaluation, and baseline scripts (ResNet-32, VGG-16)
├── stl10/                 # STL-10 training and evaluation scripts (ResNet-32, VGG-16)
├── tinyimagenet/          # Tiny-ImageNet training and evaluation scripts
├── TBT/                   # Targeted Bit Trojan (TBT) attack and defense evaluation
│   ├── resnet32-cifar10/  # TBT attacks on CIFAR-10 (Adaptive, Non-Adaptive, Ensemble)
│   ├── resnet32-cifar100/ # TBT attacks on CIFAR-100 (Adaptive, Non-Adaptive, Ensemble)
│   └── resnet32-stl10/    # TBT attacks on STL-10 (Adaptive, Non-Adaptive, Ensemble)
├── BFA/                   # Core Bit-Flip Attack engine and helper modules
├── TA-LBF/                # Targeted Attack via Limited Bit-Flips (Adaptive & Non-Adaptive)
├── ProFlip/               # ProFlip trigger generation and attack execution
├── figures_btp/           # Publication-ready plots, figures, and architectural diagrams
├── generate_comparison_line_graph.py # Script generating defense ASR comparison plots
├── generate_comparison_plots.py      # Script generating multi-dataset comparison graphs
├── generate_line_graphs.py           # Script generating bit-flip progression trajectories
├── paper.tex              # IEEE conference paper manuscript
└── README.md              # Project guide and documentation
```

---

## 3. Environment Setup

### Recommended System Requirements
- **OS:** Linux (Ubuntu 18.04/20.04/22.04) or macOS
- **GPU:** NVIDIA GPU with CUDA support (e.g., Tesla P100, V100, RTX 3090, A100)
- **Python:** 3.8+ (Conda environment recommended)

### Quick Installation

```bash
# 1. Clone the repository and enter the directory
git clone https://github.com/Jayantsikar/SHIELD.git Aegis
cd Aegis

# 2. Create a conda environment
conda create -n aegis_env python=3.8 -y
conda activate aegis_env

# 3. Install PyTorch with CUDA (e.g., CUDA 11.7)
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# 4. Install supporting packages
pip install numpy matplotlib scipy scikit-learn pandas tensorboardX tqdm Pillow pyyaml yacs
```

---

## 4. Dataset Preparation

Create a centralized data folder:
```bash
mkdir -p data
```

* **CIFAR-10:** Automatically downloaded by PyTorch on first run into `data/`.
* **CIFAR-100:** Download automatically or use the download helper:
  ```bash
  cd cifar100/resnet32
  bash download_cifar100.sh
  cd ../..
  ```
* **STL-10:** Automatically downloaded when running training scripts with `--download`.
* **Tiny-ImageNet:** Download and extract into `data/tiny-imagenet-200/`.

---

## 5. Model Training & Quantization

To evaluate defenses, models are first trained on the backbone architecture, followed by fine-tuning the intermediate early-exit classifier (IC) branches.

### Example: Training Quantized ResNet-32 on CIFAR-10
```bash
cd cifar10/resnet32

# Step 1: Train base quantized ResNet-32 (200 epochs)
bash train_CIFAR.sh

# Step 2: Fine-tune multi-exit branches (ICs)
bash train_finetune_branch.sh

cd ../..
```

### Example: Training Quantized ResNet-32 on CIFAR-100
```bash
cd cifar100/resnet32

# Step 1: Train base quantized ResNet-32
bash train_CIFAR.sh

# Step 2: Fine-tune multi-exit branches (ICs)
bash train_finetune_branch.sh

cd ../..
```

Checkpoints will be saved in `save/` and `save_finetune/model_best.pth.tar`.

---

## 6. Running Attacks & Defense Evaluation

### 6.1 TBT (Targeted Bit Trojan) Attacks

The TBT attack evaluates both:
- **Non-Adaptive TBT:** Attacker targets a single specific intermediate classifier head.
- **Adaptive TBT:** Attacker has white-box knowledge of the multi-exit structure and co-optimizes bit flips across multiple IC branches simultaneously ($i \ge 5$).

#### Running CIFAR-100 TBT Attacks Interactively:
```bash
cd TBT/resnet32-cifar100

# 1. Defended / Checksum Ensemble (Proposed Two-Phase Defense) - Non-Adaptive
python -u TBT_nonadaptive_ensemble.py

# 2. Defended / Checksum Ensemble (Proposed Two-Phase Defense) - Adaptive
python -u TBT_adaptive_ensemble.py

# 3. Baseline Model (Undefended Single Head) - Non-Adaptive
python -u TBT_nonadaptive.py

# 4. Baseline Model (Undefended Single Head) - Adaptive
python -u TBT_adaptive.py

cd ../..
```

#### Running CIFAR-10 TBT Attacks Interactively:
```bash
cd TBT/resnet32-cifar10

# 1. Defended Ensemble - Non-Adaptive
python -u TBT_nonadaptive_ensemble.py

# 2. Defended Ensemble - Adaptive
python -u TBT_adaptive_ensemble.py

# 3. Baseline - Non-Adaptive
python -u TBT_noadaptive.py

# 4. Baseline - Adaptive
python -u TBT_adaptive.py

cd ../..
```

---

### 6.2 Running on a Cluster (SLURM)

If you have access to an HPC GPU cluster managed by SLURM, submission scripts are pre-configured with `$TMPDIR` node staging for maximum I/O speed.

#### Submitting CIFAR-100 TBT Jobs:
```bash
cd ~/Aegis/TBT/resnet32-cifar100

# Defended Ensemble (Self-Healing) — Adaptive
sbatch run_TBT_adaptive_ensemble.sh

# Defended Ensemble (Self-Healing) — Non-Adaptive
sbatch run_TBT_nonadaptive_ensemble.sh

# Baseline Aegis (Single Head) — Adaptive
sbatch run_TBT_adaptive.sh

# Baseline Aegis (Single Head) — Non-Adaptive
sbatch run_TBT_nonadaptive.sh
```

#### Submitting CIFAR-10 TBT Jobs:
```bash
cd ~/Aegis/TBT/resnet32-cifar10

sbatch run_TBT_adaptive_ensemble.sh
sbatch run_TBT_nonadaptive_ensemble.sh
sbatch run_TBT_adaptive.sh
sbatch run_TBT_nonadaptive.sh
```

#### Useful Cluster Commands:
```bash
# Monitor your running jobs
squeue -u $USER

# Follow live output logs
tail -f <JOBID>.out

# Cancel a job if needed
scancel <JOBID>
```

---

### 6.3 Progressive Bit-Flip Attacks (BFA)

BFA uses gradient-guided search to identify and flip the most vulnerable parameter bits:

```bash
# CIFAR-10
cd cifar10/resnet32
sbatch BFA_CIFAR.sh   # or: bash BFA_CIFAR.sh for interactive execution
cd ../..

# CIFAR-100
cd cifar100/resnet32
sbatch BFA_CIFAR.sh
cd ../..
```

---

### 6.4 TA-LBF & ProFlip Attacks

#### TA-LBF (Targeted Attack via Limited Bit-Flips):
```bash
cd TA-LBF/non-adaptive
bash attack_reproduce_k=50_resnet32_cifar10.sh
bash attack_reproduce_k=50_resnet32_cifar100.sh
cd ../..

cd TA-LBF/adaptive
bash attack_reproduce_k=50_resnet32_cifar10.sh
bash attack_reproduce_k=50_resnet32_cifar100.sh
cd ../..
```

#### ProFlip:
```bash
cd ProFlip/resnet32-cifar10
python3 trigger_nonadaptive.py
python3 CSB_nonadaptive.py
python3 trigger_adaptive.py
python3 CSB_adaptive.py
cd ../..
```

---

## 7. Understanding Outputs & Metrics

When an attack script executes, it tracks three primary metrics:

1. **Clean Data Accuracy (CDA / C-Acc):**
   Classification accuracy on uncorrupted validation data. A higher percentage indicates that normal model utility is preserved.
2. **Attack Success Rate (ASR):**
   The percentage of triggered inputs misclassified into the attacker-selected target class ($y_t = 2$). A lower percentage denotes superior defense resilience.
   - **Theoretical Random-Guess Lower Bound:**
     - CIFAR-10: $10.0\%$ (1 out of 10 classes)
     - CIFAR-100: $1.0\%$ (1 out of 100 classes)
3. **Exit Distribution:**
   Shows which branch (IC0 through IC14 or the Main Classifier) made the final prediction for each test sample. In defended models, triggered inputs exhibit stable distributions rather than forcing malicious premature early exits.

Logs and result summaries are written to:
- Interactive runs: Printed to stdout and saved in `result/`.
- Cluster runs: Written to `<JOBID>.out` and automatically synced back to `$SLURM_SUBMIT_DIR/result/`.

---

## 8. Regenerating Figures & Visualizations

All publication-quality plots used in the paper are automatically generated and stored in `figures_btp/`. You can regenerate them at any time:

```bash
# 1. Bar Chart Comparison (Clean Accuracy & ASR across CIFAR-10 and CIFAR-100)
python3 y_temperory_files/generate_figures.py

# 2. Defense ASR Comparison Line Graphs (Dual-Panel across Defenses)
python3 generate_comparison_line_graph.py

# 3. Multi-Dataset Comparison (CIFAR-10, CIFAR-100, and STL-10)
python3 generate_comparison_plots.py

# 4. Bit-Flip Search Trajectories (ASR & Clean Acc vs. Number of Flipped Bits k)
python3 generate_line_graphs.py
```

Generated plots will be saved in `figures_btp/` as both vector `.pdf` and high-resolution `.png` files.

---

## 9. Benchmark Results Reference

Below are the empirical benchmarks achieved by the proposed defense on **ResNet-32**:

### CIFAR-100 Benchmark
| Attack Setting | Defense Paradigm | Clean Accuracy (CDA) | Attack Success Rate (ASR) | Defense Impact |
| :--- | :--- | :---: | :---: | :---: |
| **Non-Adaptive TBT** | Baseline Aegis (Single Head) | 67.13% | 26.30% | Baseline |
| | P1: SBE Ensemble (Undefended) | 65.01% | 17.36% | -8.94% ASR |
| | **P2: Self-Healing Checksum (Ours)** | **62.84%** | **2.02%** | **-24.28% ASR (Near 1% bound)** |
| **Adaptive TBT** | Baseline Aegis (Single Head) | 57.53% | 32.29% | Baseline |
| | P1: SBE Ensemble (Undefended) | 58.99% | 24.77% | -7.52% ASR |
| | **P2: Self-Healing Checksum (Ours)** | **34.13%** | **19.15%** | **-13.14% ASR (Curtailed)** |

### CIFAR-10 Benchmark
| Attack Setting | Defense Paradigm | Clean Accuracy (CDA) | Attack Success Rate (ASR) |
| :--- | :--- | :---: | :---: |
| **Non-Adaptive TBT** | Baseline Aegis | 91.38% | 17.82% |
| | P1: SBE Ensemble (Undefended) | 90.01% | 15.48% |
| | **P2: Self-Healing Checksum (Ours)** | **89.93%** | **11.67%** |
| **Adaptive TBT** | Baseline Aegis | 83.41% | 38.50% |
| | P1: SBE Ensemble (Undefended) | 87.39% | 40.72% |
| | **P2: Self-Healing Checksum (Ours)** | **83.02%** | **15.09%** |

---

## 10. Troubleshooting & FAQ

1. **`FileNotFoundError: model_best.pth.tar not found`**
   - Ensure you ran the training and branch fine-tuning scripts in Section 5.
   - Attack scripts look for checkpoint files in `save_finetune/` or `save/`.
2. **`CUDA Out of Memory`**
   - In the attack script, reduce `--test_batch_size` from `128` to `64` or `32`.
3. **Data Path Mismatches on Cluster**
   - On SLURM clusters, ensure `~/Aegis/data` contains the dataset folders (`cifar-10-batches-py` or `cifar-100-python`). The SLURM scripts will automatically copy them to the fast local SSD (`$TMPDIR`).
4. **LaTeX Compilation Missing Figures**
   - Ensure the figure folder is named `figures_btp/` as configured in `paper.tex` via `\graphicspath{{figures_btp/}{../figures_btp/}}`.

---

## Citation

If you find this work or codebase helpful in your research, please cite:

```bibtex
@inproceedings{wang2023aegis,
  title={Aegis: Mitigating targeted bit-flip attacks against deep neural networks},
  author={Wang, Jiazhen and Zhang, Zhen and Wang, Meng and Qiu, Han and Zhang, Tianwei and Li, Qi and Li, Zhaolong and Wei, Tao and Zhang, Chao},
  booktitle={32nd USENIX Security Symposium (USENIX Security 23)},
  pages={2329--2346},
  year={2023}
}
```
