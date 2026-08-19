# ResNet32 / CIFAR-10 — Results Comparison Tracker

## Experiment Setup
- **Architecture:** ResNet32 (quantized, multi-branch SDN)
- **Dataset:** CIFAR-10
- **Cluster:** IIT Guwahati CSE GPU Cluster (Tesla P100-16GB)
- **Framework:** PyTorch 1.13.1 + CUDA 11.7

---

## Baseline Training Results

| Run | Epochs | Best Val Acc | Final Val Acc | LR Schedule | Job ID | Date |
|---|---|---|---|---|---|---|
| Baseline (GPU) | 200 | **91.38%** | 91.13% | 0.01→0.001→0.0001 | 1190 | 2026-08-09 |

**Checkpoint:** `results/cifar10/resnet32/baseline_training_200ep/model_best.pth.tar`

---

## Attack Results

### TBT Non-Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| Baseline | 2 | 89.51% | 82.66% | 1200 | 2026-08-10 |

### TBT Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| Baseline | 2 | 67.78% | 62.43% | 1204* | 2026-08-10 |

### CSB Non-Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| | | | | | |

### CSB Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| | | | | | |

### Trigger Non-Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| Baseline | 2 | 10.35% | 71.24% | 1206 | 2026-08-10 |

### Trigger Adaptive
| Run | Target Class | Attack Success Rate | Clean Acc After | Job ID | Date |
|---|---|---|---|---|---|
| | | | | | |

---

## Notes
- Branch accuracies (B1-B6) are early-exit classifiers, intentionally low (~10%)
- `Prec_Bmain@1` is the real model accuracy metric to track
- `total weight changes` in attack logs = number of bits successfully flipped
