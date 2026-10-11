# Critical Analysis of ResilientDNN / SHIELD Project

> [!CAUTION]
> This is a rigorous, critical examination of every layer of the project — approach, code, experiments, and writing. The goal is to surface all weaknesses before submission.

---

## 1. Fundamental Approach / Research Soundness Issues

### 1.1 The Checksum Defense Tests Software-Level Attacks, Not Real Hardware Rowhammer

**This is the most critical flaw in the entire paper.**

`golden_checksums` are computed *before the attack runs in the same Python process* in both ensemble scripts:

```python
# TBT_nonadaptive_ensemble.py lines 206-209
golden_checksums = {}
_blocks = get_backbone_blocks(net)
for i, (name, mod) in enumerate(_blocks):
    golden_checksums[name] = compute_module_checksum(mod)
```

Then the attack modifies `net` in-place (flipping PyTorch tensor values), and re-computing the checksum catches the modification. **This is trivially effective because the attack and defense live in the same process with the original weights in memory.** This does NOT simulate a DRAM Rowhammer attack where:
- The golden hash is stored in a TPM/ROM at factory time
- The attacker flips DRAM bits in a different physical memory row while the process is running
- The modified DRAM content is then read back by the inference engine

The paper claims to defend against *physical* bit-flip attacks but the code tests *software-level in-process weight modification*. This is a fundamental threat model mismatch.

### 1.2 Code Hashes Backbone Blocks; Paper Claims IC-Level Hashing

The paper (Algorithm 2) states checksums are computed over **each IC head** independently with a binary trust mask per IC ($\mu_k$). But the code hashes **entire backbone residual blocks** and then infers IC safety by propagation:

```python
def get_safe_pool_from_blocks(model, golden_checksums) -> list:
    ...
    if first_compromised_idx > 0:
        k = first_compromised_idx - 1
        return list(range(k))   # quarantines all ICs downstream
```

The code never hashes the IC weights at all — only the backbone block weights. This is architecturally different from what the paper describes. The paper claims independent per-IC trust masking; the code does prefix-based bulk quarantine.

### 1.3 The Adaptive Attack is Not Adaptive to the Checksum Defense

The "adaptive" attack in `TBT_adaptive_ensemble.py` (lines 864–934) optimizes bit flips across **multiple classifier layers** but:
- Does NOT try to keep the SHA-256 hash intact (no hash collision attempt)
- Does NOT try to corrupt the stored golden hashes
- Does NOT attack weights outside the checksum coverage

A truly adaptive adversary against a checksum defense would attempt to either craft a hash collision or simultaneously corrupt the secure hash storage. The 60.8% ASR reduction claimed for Phase 2 is simply the checksum detecting *any* modification — which it will always do, regardless of attack strategy. This makes Phase 2's result **guaranteed by construction**, not an empirical finding.

### 1.4 Critical Inconsistency Between paper.tex and report.tex Numbers

| Metric | paper.tex | report.tex |
|---|---|---|
| Adaptive ASR (defended) | **15.09%** | **9.79%** |
| Non-adaptive ASR (defended) | **11.67%** | **12.62%** |
| Adaptive CDA (defended) | **83.02%** | **90.66%** |
| Non-adaptive CDA (defended) | **89.93%** | **90.30%** |

The adaptive ASR difference (15.09% vs 9.79%) and adaptive CDA (83.02% vs 90.66%) are dramatically different. These cannot both be correct. This is a serious reproducibility crisis for the paper.

### 1.5 The 512-Byte Claim Math is Inconsistent With the Code

The paper (Section 6.1) states "storing all 16 golden hashes requires only $16 \times 32 = 512$ bytes." But the code hashes 17 backbone blocks (stem + 15 residual blocks + classifier), not 16 IC heads. And the paper says it only hashes "IC linear projection layers containing $64 \times 10 = 640$ weights each" — but the code hashes entire module state_dicts including convolutions, batch norm statistics, etc.

### 1.6 The 5× Inference Overhead is Ignored

The paper claims "negligible" overhead, citing "<5 μs per IC for SHA-256." But the actual inference overhead is the forward pass through 5 branches (SBE with C=5) vs. Aegis which exits at the first branch exceeding τ (average ~1–2 branches). The paper never reports actual latency comparison. This is a significant omission for an "edge deployment" paper.

---

## 2. Code-Level Issues

### 2.1 Different Normalization Statistics Between Adaptive and Non-Adaptive Scripts

- `TBT_nonadaptive_ensemble.py` (line 97-98):
  ```python
  mean = [x / 255 for x in [125.3, 123.0, 113.9]]
  std  = [x / 255 for x in [63.0, 62.1, 66.7]]
  ```
- `TBT_adaptive_ensemble.py` (line 92-93):
  ```python
  mean = [x / 255 for x in [129.3, 124.1, 112.4]]
  std  = [x / 255 for x in [68.2, 65.4, 70.4]]
  ```

**These are different preprocessing statistics for the same dataset and model.** The two attacks are not running against the same effective model distribution. One of these sets is wrong and will produce shifted accuracy/ASR numbers. Results from the two scripts are not directly comparable.

### 2.2 `wb` (Weight Budget) Differs Between Scripts

- `TBT_nonadaptive_ensemble.py` line 48: `wb=13`
- `TBT_adaptive_ensemble.py` line 49: `wb=1`

`wb` controls the number of critical weight neurons selected during NGR. The non-adaptive attack targets 13 neurons per layer; the adaptive targets 1. These are fundamentally different attack strengths being compared side-by-side under the labels "non-adaptive" and "adaptive."

### 2.3 Trojan Training Uses Test Data

Both ensemble scripts use `loader_test` for trojan insertion training (lines 874–898 in non-adaptive):

```python
for epoch in range(200): 
    for t, (x, y) in enumerate(loader_test):  # <-- TEST data!
        ...
        if t==1:
            break   # 2 batches only
```

Evaluation also runs on `loader_test`. The trojan is trained and evaluated on the same test data, inflating ASR and invalidating the results.

### 2.4 `loss.requires_grad_(True)` Hack Breaks Gradient Computation

In `TBT_adaptive_ensemble.py` line 906:
```python
loss.requires_grad_(True)
loss.backward()
```

This is a code smell indicating the computation graph was broken (probably due to calling `net.eval()` without selectively enabling gradients for the trainable layers). Calling `.requires_grad_(True)` on the loss tensor forces a fake gradient of 1.0 everywhere but does NOT restore the real gradients through the network. The bit-flip targeting (NGR) for the adaptive attack may produce incorrect gradient rankings.

### 2.5 Checksum Computed on Float32 Parameters, Not Quantized Int8

The paper claims:
> *"Because integer byte representations are strictly deterministic... the SHA-256 digest produces zero false-positive integrity alarms"*

But `compute_module_checksum()` hashes `param.detach().cpu().numpy().tobytes()` where `param` is the float32 scale/step of `quan_Conv2d`/`quan_Linear`. The actual quantized integer values are computed on-the-fly in `forward()`. The checksum does NOT cover the int8 weight representation described in the paper — it covers the float32 training parameters. Float32 bytes can differ across hardware due to floating-point rounding, potentially causing false positives.

### 2.6 Hardcoded Layer Indices Break with Any Architecture Change

```python
# TBT_adaptive_ensemble.py line 864
if n==166 or n==214 or n==238 or n==262 or n==286 or n==310 or n==334 or n==358 or n==368:
```

These magic numbers are undocumented and untested. The non-adaptive script uses only `n==368`, meaning it flips 1 layer while the adaptive flips 9 — making the "non-adaptive" attack actually weaker than what should be expected under a fair comparison. If anyone adds a layer or changes the model slightly, these indices silently target the wrong parameters.

### 2.7 Unused Variables and Dead Code

- `exit_b1` through `exit_m`, `top_list`, `top5_list`, `decision`, `output_summary` initialized in `validate()` but never updated
- `validate2()` returns `index_list` but the return value is ignored at the call site (global side-effect used instead)
- `trust_mask = torch.ones(3)` created but never used in any inference path
- `_Quantize` class defined but never called (quantization is internal to `quan_Conv2d`/`quan_Linear`)
- Multiple commented-out NeuroPots and TBT detector code blocks (~80 lines each) should be removed or in a separate branch

### 2.8 Inconsistent `num_workers` Between Scripts

- `TBT_adaptive_ensemble.py` line 870: `num_workers=2` (for trojan training loader)
- All other loaders: `num_workers=0`

Mixed `num_workers` settings can cause data ordering inconsistencies across the same epoch, especially when `shuffle=False`.

---

## 3. Paper / Report Writing Issues

### 3.1 Double Abstract in paper.tex

Lines 65–71 of `paper.tex` contain **two complete abstract paragraphs** inside the `\begin{abstract}` block. One is clearly a rewrite of the other. IEEE conference papers must have a single abstract (≤150 words for IEEEtran). The current double abstract will be rejected on format alone.

### 3.2 Duplicate Package Imports

```latex
\usepackage{graphicx}        % lines 3, 10, 32 — three times
\usepackage{amsmath,amssymb} % lines 4 and 9
\usepackage{tabularx}        % lines 13 and 30
\usepackage{xcolor}          % line 12 (plain) and line 28 ([table] option)
```

The `xcolor` double import with conflicting options (`plain` vs `[table]`) will cause a LaTeX package option conflict error.

### 3.3 Inconsistent Numbers Between Table and Figures

- Table 1 (adaptive CDA defended): **83.02%**
- Fig. 4 caption: *"Phase 2 Self-Healing maintains clean accuracy above **90.6%**"*
- Abstract: *"clean accuracy of **83.02%**"*
- Fig. 3 caption: *"keeping the ASR close to the random-guessing level of approximately **10%**"* (actual value is 15.09%)

These inconsistencies suggest the figures and the table were generated from different experimental runs and never reconciled.

### 3.4 The BIN/RA-BNN Comparison is Methodologically Flawed

Table 1 shows BIN (ASR=94.80%) and RA-BNN (ASR=74.50%) under "Non-Adaptive TBT" — but these methods were designed against untargeted BFA, not targeted TBT. The comparison is not apples-to-apples, and their adaptive columns marked "NA*" is misleading (they were never designed for multi-exit at all). A reviewer would flag this immediately.

### 3.5 Wrong Citation for Aegis in report.tex

```latex
% report.tex line 565
\bibitem{aegis}
J.~Li, A.~S. Rakin, ..., ``Defending Bit-Flip Attack through DNN Weight Reconstruction,'' DAC, 2021.
```

This is the **wrong paper**. The actual Aegis paper is:
> J. Wang et al., "Aegis: Mitigating targeted bit-flip attacks against deep neural networks," USENIX Security 2023.

The `paper.tex` correctly cites it. `report.tex` has a completely different paper under the `aegis` key.

### 3.6 Missing Figure Files Referenced in paper.tex

`paper.tex` includes:
- `figures/fig_sdn_architecture_mod.png` (line 251) — **does not exist**
- `figures/fig_architecture_checksum_mod.png` (line 290) — **does not exist**

The `figures/` directory only has `fig_sdn_architecture.pdf` and `fig_architecture_checksum.pdf`. The `_mod` (modified) versions were never generated. The paper will fail to compile without them.

### 3.7 "Dual-Algorithm Formalization" Listed as a Contribution

Listing the writing of two pseudocode algorithms as a top-level research contribution (bullet point 3 in Section 1) misrepresents the nature of contributions. This will appear naive to reviewers. Pseudocode is a presentation tool, not a contribution.

### 3.8 No Related Work Section

The paper jumps from Background directly to the proposed method. There is no "Related Work" section covering:
- Prior model integrity/attestation work
- Other self-healing or redundancy-based defenses
- Knowledge distillation/ensemble defenses against adversarial ML

This is a significant gap that peer reviewers will flag.

### 3.9 Large Blocks of Commented-Out Old Draft Text

Sections 1, 3, 4, and 6 in `paper.tex` each contain full paragraphs or entire subsections commented out with `%`. The paper appears to have been written twice. This makes the `.tex` source unmaintainable and risks accidentally leaving the wrong version active.

---

## 4. Experimental Evaluation Gaps

### 4.1 No Ablation Study

No ablation for:
- Ensemble size C (C=5 used, but why? C=3 vs 5 vs 7?)
- Pruning threshold M (M=4, why not M=2 or M=6?)
- The specific interaction between SBE and checksum
- k > 20 bit flips (what happens when the attacker uses more bits?)

### 4.2 Only CIFAR-10 Despite Multi-Dataset Infrastructure

The `results/` directory contains subdirectories for CIFAR-10, CIFAR-100, STL-10, and Tiny-ImageNet. The `ProFlip/` and `TA-LBF/` directories exist for other attacks. `report.tex` includes STL-10 results. But `paper.tex` (the conference version) only reports CIFAR-10. A conference paper with broader evaluation would be significantly stronger.

### 4.3 No Statistical Significance Reporting

The SBE randomly samples 5 branches. With different random seeds, the ASR could vary by several percentage points. No standard deviations, confidence intervals, or multi-seed averages are reported. The results are single-run point estimates.

### 4.4 No Real Edge Hardware Latency

Claims of "<5 μs per IC" and "negligible overhead" are unsubstantiated without actual measurements on edge hardware (Raspberry Pi 4, Jetson Nano, etc.). The experiments ran on Tesla P100 and H100 — which are not edge devices.

### 4.5 Aegis Baseline is Worse Than Published Aegis Numbers

The paper reports reproduced Aegis adaptive ASR = 38.50%, while the original Aegis paper reports 31.10%. A worse baseline makes the improvement look larger. The discrepancy needs to be explained rigorously (different training, different hyperparameters?) otherwise a reviewer may accuse the paper of cherry-picking a weak baseline.

---

## 5. Priority-Ranked Summary

| Priority | Issue | Location |
|---|---|---|
| 🔴 P0 | Checksum tests software-level attacks, not DRAM Rowhammer | Code + Threat Model |
| 🔴 P0 | Code hashes backbone blocks; paper describes IC-level hashing | Code vs. Paper |
| 🔴 P0 | Adaptive attack not adaptive to checksum (result guaranteed by construction) | Experiment Design |
| 🔴 P0 | Inconsistent numbers between paper.tex and report.tex | Both documents |
| 🔴 P0 | Different CIFAR-10 normalization stats in adaptive vs. non-adaptive | Code |
| 🟠 P1 | Checksum on float32, not int8 as claimed | Code vs. Paper |
| 🟠 P1 | `loss.requires_grad_(True)` breaks gradient computation in adaptive attack | Code |
| 🟠 P1 | Trojan training on test set (data leakage) | Code |
| 🟠 P1 | Inconsistent numbers between Table 1 and figures | paper.tex |
| 🟠 P1 | Wrong Aegis citation in report.tex | report.tex |
| 🟠 P1 | Missing `_mod` figure files that paper.tex references | paper.tex + figures/ |
| 🟠 P1 | No ablation study | Evaluation |
| 🟡 P2 | Double abstract in paper.tex | paper.tex |
| 🟡 P2 | Duplicate/conflicting package imports | paper.tex |
| 🟡 P2 | `wb` differs between scripts (different attack budgets) | Code |
| 🟡 P2 | No statistical significance / error bars | Evaluation |
| 🟡 P2 | No edge hardware latency measurements | Evaluation |
| 🟡 P2 | BIN/RA-BNN comparison is methodologically unfair | paper.tex |
| 🟢 P3 | No Related Work section | paper.tex |
| 🟢 P3 | Commented-out old draft text polluting source | paper.tex |
| 🟢 P3 | Unused variables and dead code | Code |
| 🟢 P3 | "Algorithm formalization" listed as research contribution | paper.tex |
