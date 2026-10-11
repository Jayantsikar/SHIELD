# Trojan Backdoor Training (TBT) Results Comparison

This document summarizes the final evaluation results for the four different TBT variants on ResNet32 using CIFAR-10. The metrics compared are **Clean Accuracy (C-Acc)** and **Attack Success Rate (ASR)**.

## Summary Table

| Attack Variant | Clean Accuracy (C-Acc) | Attack Success Rate (ASR) |
| :--- | :--- | :--- |
| **Non-Adaptive TBT** | 88.69% | 17.82% |
| **Non-Adaptive Ensemble TBT** | **90.19%** | 14.87% |
| **Adaptive TBT** | 83.41% | 38.50% |
| **Adaptive Ensemble TBT** | 87.24% | **53.41%** |

## Detailed Breakdown

### 1. Non-Adaptive TBT
*   **Target:** Single main classifier
*   **Clean Accuracy:** 88.69%
*   **Attack Success Rate:** 17.82%
*   *Observation:* Achieves high clean accuracy but struggles to mount a highly effective attack, highlighting the difficulty of a non-adaptive trigger against the standard network.

### 2. Non-Adaptive Ensemble TBT
*   **Target:** Ensemble evaluation but trigger trained non-adaptively
*   **Clean Accuracy:** 90.19%
*   **Attack Success Rate:** 14.87%
*   *Observation:* Yields the **highest clean accuracy** across all variants. However, because the trigger wasn't explicitly optimized for the ensemble, the ensemble easily defends against it, resulting in the **lowest attack success rate**.

### 3. Adaptive TBT
*   **Target:** Optimizes trigger against multiple classifiers (adaptive)
*   **Clean Accuracy:** 83.41%
*   **Attack Success Rate:** 38.50%
*   *Observation:* The attack success rate more than doubles compared to the non-adaptive baseline. However, this comes at a significant cost to the clean accuracy, which drops to the lowest among all variants (83.41%).

### 4. Adaptive Ensemble TBT
*   **Target:** Optimizes trigger against multiple classifiers and evaluates as an ensemble
*   **Clean Accuracy:** 87.24%
*   **Attack Success Rate:** 53.41%
*   *Observation:* The most devastating attack variant. It achieves the **highest attack success rate** (53.41%) while maintaining a very respectable clean accuracy (87.24%). The ensemble mechanism, when adaptively targeted, actually reinforces the backdoor's effectiveness.

## Key Takeaways
> [!TIP]
> **Ensemble as a Defense vs. Vulnerability**
> When the attack is **non-adaptive**, the ensemble acts as a strong defense, dropping the ASR from 17.82% down to 14.87% and boosting clean accuracy to over 90%. 
> However, when the attack is **adaptive** (aware of the branches), the ensemble becomes a critical vulnerability, boosting the ASR to 53.41%.
