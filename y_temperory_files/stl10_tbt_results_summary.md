# Aegis Ensemble Evaluation Results (STL-10)

This document compares the robustness of the **Baseline Aegis** model against the proposed **Aegis + Ensemble (Self-Healing)** extension under Targeted Bit Trojan (TBT) attacks on the STL-10 dataset using the ResNet-32 architecture.

## Performance Comparison

The following table summarizes the Attack Success Rate (ASR) and Clean Data Accuracy under both **Adaptive** and **Non-Adaptive** TBT attacks.

| Attack Type | Metric | Baseline Aegis | Aegis + Ensemble (Defended) | Aegis + Ensemble (Raw/Undefended) |
| :--- | :--- | :--- | :--- | :--- |
| **Adaptive TBT** | **ASR (%)** | **31.00%** | **13.91%** | 28.25% |
| | Clean Acc (%) | - | 74.19% | 73.70% |
| **Non-Adaptive TBT** | **ASR (%)** | **13.00%** | **10.21%** | 12.21% |
| | Clean Acc (%) | - | 74.11% | 73.89% |

---

## Key Takeaways

> [!TIP]
> The **Aegis + Ensemble** defense effectively mitigates the impact of Targeted Bit Trojan attacks by identifying compromised Internal Classifiers (ICs) through checksum verification and falling back to clean ICs via stochastic voting.

1. **Adaptive Attack Mitigation**: Against the highly potent Adaptive TBT attack, the baseline Aegis model suffers an ASR of **31.00%**. By utilizing the checksum-based self-healing ensemble mechanism, the ASR is significantly reduced to **13.91%**.
2. **Non-Adaptive Attack Resilience**: For Non-Adaptive TBT attacks, the baseline Aegis ASR of **13.00%** is further driven down to **10.21%**.
3. **Preservation of Model Utility**: The ensemble defense maintains high clean data accuracy. In fact, after isolating the compromised ICs, the ensemble's clean accuracy slightly improves (e.g., from 73.70% to 74.19% in the adaptive scenario) because the damaged paths are removed from the inference voting process.
4. **Generalization across Datasets**: These STL-10 results reinforce the findings from CIFAR-10, demonstrating that the self-healing stochastic voting mechanism is highly effective regardless of input resolution (96x96 vs 32x32) or dataset size.
