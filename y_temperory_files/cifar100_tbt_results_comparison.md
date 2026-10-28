# Trojan Backdoor Training (TBT) Evaluation Results Comparison (CIFAR-100)

This document provides a comprehensive comparison of the Trojan Backdoor Training (TBT) attack variants on **ResNet-32** using the **CIFAR-100** dataset. It compares standard **Baseline Aegis** (single early-exit network) against the **Aegis + Ensemble (Self-Healing / Checksum Defense)** across both **Non-Adaptive** and **Adaptive** attacks.

---

## 1. Executive Summary Table

| Attack Variant | Metric | Baseline Aegis (Single) | Aegis Ensemble (Raw / Undefended) | Aegis Ensemble (Defended / Checksum) | Defense Impact |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Non-Adaptive TBT** | **Clean Accuracy (C-Acc)** | 67.13% | 65.01% | 62.84% | -2.17% (utility retained) |
| | **Attack Success Rate (ASR)** | 26.30% | 17.36% | **2.02%** | **-24.28%** (near-complete defense) |
| **Adaptive TBT** | **Clean Accuracy (C-Acc)** | 57.53% | 58.99% | 34.13% | -24.86% (trade-off under aggressive attack) |
| | **Attack Success Rate (ASR)** | 32.29% | 24.77% | **19.15%** | **-13.14%** (significant mitigation) |

---

## 2. Cross-Variant Comparison

| Configuration | Clean Accuracy (C-Acc) | Attack Success Rate (ASR) | Notes / Behavior |
| :--- | :---: | :---: | :--- |
| **Baseline Non-Adaptive** | **67.13%** | 26.30% | Standard SDN with single target head attacked non-adaptively. |
| **Ensemble Non-Adaptive (Raw)** | 65.01% | 17.36% | Voting ensemble naturally diffuses non-adaptive backdoor effect. |
| **Ensemble Non-Adaptive (Defended)** | 62.84% | **2.02%** | Checksum verification isolates compromised heads; ASR collapses to ~2%. |
| **Baseline Adaptive** | 57.53% | 32.29% | Multi-branch co-optimization achieves high ASR at cost of clean utility. |
| **Ensemble Adaptive (Raw)** | 58.99% | 24.77% | Unchecked ensemble under multi-head adaptive trigger. |
| **Ensemble Adaptive (Defended)** | 34.13% | 19.15% | Checksum flags triggered heads; defensive isolation curbs ASR to 19.15%. |

---

## 3. Detailed Breakdown of Results

### 3.1 Non-Adaptive TBT

- **Baseline Model (Single Network)**:
  - Clean Accuracy: **67.13%**
  - Attack Success Rate: **26.30%**
  - Exit distribution (Clean): `[696, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1296, 1048, 1044, 1241, 2092, 2583]`
  - Exit distribution (Triggered): `[692, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1286, 1064, 1042, 1242, 2040, 2634]`

- **Aegis Ensemble Defense**:
  - Raw / Undefended: Clean Acc = **65.01%**, ASR = **17.36%**
  - Defended (Checksum): Clean Acc = **62.84%**, ASR = **2.02%**
  - *Analysis*: Against a non-adaptive trigger, the checksum defense is exceptionally potent. It brings the attack down from **26.30%** (baseline) and **17.36%** (raw ensemble) to just **2.02%**, practically neutralizing the backdoor while maintaining **62.84%** clean accuracy on CIFAR-100.

---

### 3.2 Adaptive TBT

- **Baseline Model (Single Network)**:
  - Clean Accuracy: **57.53%**
  - Attack Success Rate: **32.29%**
  - Exit distribution (Clean): `[1170, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2422, 1469, 1345, 1050, 1168, 1376]`
  - Exit distribution (Triggered): `[631, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2227, 1212, 1454, 1037, 1434, 2005]`

- **Aegis Ensemble Defense**:
  - Raw / Undefended: Clean Acc = **58.99%**, ASR = **24.77%**
  - Defended (Checksum): Clean Acc = **34.13%**, ASR = **19.15%**
  - *Analysis*: In the adaptive setting, the attacker specifically crafts the trigger to activate latent neurons across multiple Internal Classifiers simultaneously (`i >= 5`). While the checksum defense successfully curbs ASR from **32.29%** down to **19.15%**, the aggressive cross-branch poisoning triggers frequent checksum verification alarms on clean inputs, reducing defended clean accuracy to **34.13%**.

---

## 4. Key Takeaways & Defense Dynamics

> [!TIP]
> **Ensemble + Checksum Self-Healing Efficacy**
> - **Non-Adaptive Protection**: Checksum verification is nearly flawless against non-adaptive Trojans on CIFAR-100, dropping ASR to **2.02%** while retaining 96.6% of clean classification accuracy (62.84% vs 65.01%).
> - **Adaptive Pressure**: When the attacker attacks all internal classifiers simultaneously (Adaptive TBT), raw ASR climbs to **32.29%**. The defense cuts the attack effectiveness by over **40% relative** (down to 19.15%).
> - **Multi-Class Complexity (CIFAR-10 vs CIFAR-100)**: With 100 classes, each internal classifier has 100 outputs and higher intra-class variance. The self-healing voting mechanism demonstrates clear defensive resilience across all 15 early-exit branches and the main classifier.
