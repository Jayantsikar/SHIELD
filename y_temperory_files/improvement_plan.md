# Improvement Plan for ResilientDNN / SHIELD

> [!IMPORTANT]
> This document pairs with `critical_analysis.md`. Each section corresponds to a problem identified there, providing a concrete fix.

---

## 1. Fixing Core Research Validity

### Fix 1.1 — Properly Simulate the Threat Model

The current code modifies PyTorch weight tensors directly in Python (software attack). To properly simulate a hardware Rowhammer/bit-flip attack:

**Option A: Use a proper fault injection simulation library**
Use [FIONA](https://github.com/GreenWaves-Technologies/fiona) or [PyTorchFI](https://github.com/ramyadhadidi/pytorchfi) to inject hardware-style bit flips into model weights *after* deployment (i.e., after golden hashes are computed and stored externally).

```python
from pytorchfi.core import fault_injection

# 1. Compute and externally store golden checksums
golden = compute_all_checksums(clean_model)
torch.save(golden, 'golden_checksums.pt')  # Store separately

# 2. Deploy model (simulate time passing)
deployed_model = copy.deepcopy(clean_model)

# 3. Attacker performs hardware-level bit flip via Rowhammer simulation
fi_model = fault_injection(deployed_model, batch_size=1, 
                           input_shape=[3, 32, 32], use_cuda=True)
fi_model.bit_flip_weight(conv_num=5, k=3, c=2, kH=1, kW=1)

# 4. Defense loads EXTERNALLY stored golden checksums
external_golden = torch.load('golden_checksums.pt')
safe_pool = verify_checksums(fi_model, external_golden)
```

**Option B: Clearly reframe the threat model**
If the goal is only to evaluate the *detection capability* (not the full DRAM attack), the paper must explicitly say:
> "We simulate the attacker's effect as in-place weight corruption and evaluate whether the checksum correctly identifies the tampered branches."

This is an honest and acceptable framing, but the paper currently implies it's a full Rowhammer simulation, which is false.

### Fix 1.2 — Redesign Checksum to Actually Hash IC Heads

The paper describes per-IC hashing. The code should match this. Here is how to hash each IC module independently in ResNet-32 SDN:

```python
def get_ic_modules(model_wrapper):
    """Extract the actual IC submodules from the SDN."""
    model = model_wrapper[1]  # unwrap Normalize_layer
    ic_modules = {}
    # Each ResNetBasicBlock has an attached IC output
    idx = 0
    for g in range(1, 4):
        group = getattr(model, f'group{g}', [])
        for block in group:
            if hasattr(block, 'output') and block.output is not None:
                ic_modules[idx] = block.output  # the IC head
                idx += 1
    # Final classifier
    ic_modules[idx] = model.classifier
    return ic_modules

def compute_ic_checksums(model_wrapper):
    ic_mods = get_ic_modules(model_wrapper)
    return {k: compute_module_checksum(v) for k, v in ic_mods.items()}

# Binary per-IC trust mask (matching Algorithm 2 in the paper)
def get_trust_mask(model_wrapper, golden_ic_checksums):
    current = compute_ic_checksums(model_wrapper)
    return {k: (current[k] == golden_ic_checksums[k]) for k in golden_ic_checksums}
```

This makes Phase 2 match what the paper's Algorithm 2 actually describes.

### Fix 1.3 — Implement a Truly Adaptive (Checksum-Aware) Attack

A genuine checksum-aware adaptive attack should attempt to keep the SHA-256 hash unchanged while still corrupting model behavior. This is computationally infeasible with SHA-256, but you can design attacks that:

**a) Target only backbone shared weights (not IC weights), which are NOT checksum-covered:**

If the checksum only covers IC head weights, an adaptive attacker targets the **backbone blocks** — the shared computation that all branches depend on. The current attack already does this partially but should be framed explicitly:

```python
# Attack ONLY backbone conv layers (not the IC heads that are checksummed)
for name, param in model.named_parameters():
    if 'output' not in name and 'classifier' not in name:  # backbone only
        param.requires_grad = True  # Target non-checksummed weights
```

**b) Document attack success under this true adaptive scenario:**
The result should show that when the attacker targets backbone weights (which bypass IC checksum), Phase 2 checksum defense fails and ASR remains high. This would be an honest evaluation that strengthens the paper's limitations section.

**c) Proposed defense upgrade**: extend checksums to cover the backbone blocks as well, and discuss the storage overhead (512 bytes × number_of_blocks).

### Fix 1.4 — Reconcile Numbers and Re-run Experiments

All four key metrics (adaptive ASR, non-adaptive ASR, adaptive CDA, non-adaptive CDA) differ between `paper.tex` and `report.tex`. The only honest fix is:

1. Run the full pipeline 3 times with different seeds: `seed = [42, 123, 7]`
2. Report mean ± std for all metrics
3. Use the same final run numbers in both documents

**Suggested table format:**
```
| Defense   | Non-Adapt ASR ↓ | Adapt ASR ↓ | Non-Adapt CDA ↑ | Adapt CDA ↑ |
|-----------|-----------------|-------------|-----------------|-------------|
| Aegis     | 17.82±0.3       | 38.50±0.8   | 91.38±0.1       | 83.41±0.5   |
| P1: SBE   | 15.48±0.4       | 40.72±1.1   | 90.01±0.2       | 87.39±0.6   |
| P2: Ours  | 11.67±0.3       | 15.09±0.5   | 89.93±0.2       | 83.02±0.4   |
```

---

## 2. Code Fixes

### Fix 2.1 — Unify Normalization Statistics

Pick one authoritative set of CIFAR-10 statistics (the ones used for training) and use them in both scripts:

```python
# Standard CIFAR-10 stats (consistent with torchvision defaults):
mean = [0.4914, 0.4822, 0.4465]
std  = [0.2023, 0.1994, 0.2010]

# OR use the ones from the original model_best.pth.tar training config
# Check cifar10/resnet32/main.py for the exact values used during training
```

Add an assertion in both scripts to verify the stats match:
```python
assert mean == [0.4914, 0.4822, 0.4465], "Normalization mismatch!"
```

### Fix 2.2 — Unify `wb` and Document Its Meaning

Both scripts should use the same `wb` value when making the same type of comparison. Alternatively, create a config file:

```python
# config.py
ATTACK_CONFIG = {
    'nonadaptive': {'wb': 13, 'target_layers': [368]},
    'adaptive':    {'wb': 13, 'target_layers': [166, 214, 238, 262, 286, 310, 334, 358, 368]},
}
```

Note: `wb=1` in the adaptive script targets only 1 top-gradient neuron, which is weaker than `wb=13`. If the non-adaptive attack is "easier" for the attacker by using 13 neurons, the comparison isn't fair. Decide on a consistent budget.

### Fix 2.3 — Use Training Data for Trojan Insertion

Replace `loader_test` with `loader_train` in the trojan training loop:

```python
# WRONG:
for t, (x, y) in enumerate(loader_test):

# CORRECT:
for t, (x, y) in enumerate(loader_train):
    if t == 1:  # still limit to 2 batches for speed, but use TRAIN data
        break
```

This prevents test data contamination and makes the evaluation methodology valid.

### Fix 2.4 — Fix Adaptive Attack Gradient Computation

Remove the `loss.requires_grad_(True)` hack and properly set up the computation graph:

```python
# BEFORE (broken):
net.eval()
...
loss.requires_grad_(True)  # WRONG — fakes a gradient
loss.backward()

# AFTER (correct):
net.eval()
# Only enable requires_grad for the targeted parameters
for name, param in net.named_parameters():
    param.requires_grad = False
    
n = 0
for param in net.parameters():
    n += 1
    if n in TARGET_LAYER_INDICES:
        param.requires_grad = True

# Now backward works correctly through the computation graph
output = net(x_var)
loss = compute_trojan_loss(output, ...)
loss.backward()  # correct gradients flow
```

### Fix 2.5 — Implement Int8-Based Checksum

To match the paper's claim of hashing quantized integer weights:

```python
def compute_int8_checksum(module: torch.nn.Module) -> str:
    """Hash the actual int8 quantized weight values, not float32 params."""
    hasher = hashlib.sha256()
    for name, param in module.named_parameters():
        # Simulate 8-bit quantization to get the integer representation
        if hasattr(module, 'step_size'):
            step = module.step_size.item()
            int_weights = torch.round(param / step).to(torch.int8)
            hasher.update(int_weights.detach().cpu().numpy().tobytes())
        else:
            # Fallback: use the float bytes (and document this limitation)
            hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
```

Or even simpler — call `quantize()` explicitly to get the int8 tensor:

```python
from models.quantization import quantize

def compute_quantized_checksum(module):
    hasher = hashlib.sha256()
    for name, param in module.named_parameters():
        if 'weight' in name:
            step = param.abs().max() / (2**(8-1) - 1)  # 8-bit scale
            q_weight = torch.round(param / step).clamp(-128, 127).to(torch.int8)
            hasher.update(q_weight.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()
```

### Fix 2.6 — Replace Magic Numbers with Named Constants

```python
# Create a utility function to find layer indices by name/type
def get_trainable_layer_indices(model, layer_names=None, layer_types=None):
    """Return parameter indices matching the given layer names or types."""
    indices = []
    for n, (name, param) in enumerate(model.named_parameters(), 1):
        if layer_names and any(ln in name for ln in layer_names):
            indices.append(n)
        elif layer_types and isinstance(param, tuple(layer_types)):
            indices.append(n)
    return indices

# Usage:
TARGET_LAYER_NAMES = ['output.linear', 'classifier']
trainable_indices = get_trainable_layer_indices(net, layer_names=TARGET_LAYER_NAMES)
```

### Fix 2.7 — Clean Up Dead Code

Remove or move all unused variables and commented code:

```python
# Remove from validate():
# exit_b1, exit_b2, ... exit_m (never updated)
# top_list, decision, output_summary (never used)
# trust_mask = torch.ones(3) (never used)

# Move to a separate 'legacy/' directory:
# All NeuroPots commented code blocks
# All TBT detector commented code blocks
# _Quantize class (not used)
```

---

## 3. Paper / Report Writing Fixes

### Fix 3.1 — Fix the Double Abstract

Remove the first (old) abstract paragraph. Keep only the clean, single-paragraph version. For IEEEtran conference format, target ≤150 words:

```latex
\begin{abstract}
Deep neural networks (DNNs) deployed on edge devices are vulnerable to 
Targeted Bit-Flip Attacks (TBFAs), which corrupt model weights in DRAM 
using hardware faults. Multi-exit architectures amplify this risk because 
adversaries can target shallow branches. Existing defenses such as Aegis 
rely on single-branch decisions, allowing a single compromised branch to 
hijack inference. We propose ResilientDNN, a two-phase defense. Phase 1 
(Stochastic Branch Ensembling, SBE) aggregates predictions from multiple 
randomly sampled branches via confidence-weighted consensus. Phase 2 uses 
SHA-256 checksums to detect and quarantine tampered branches at runtime, 
enabling autonomous self-healing inference. Evaluated on CIFAR-10 with 
quantized ResNet-32, ResilientDNN reduces adaptive attack success rate 
from 38.50\% (Aegis) to 15.09\% while maintaining 83.02\% clean accuracy.
\end{abstract}
```

### Fix 3.2 — Fix Duplicate Package Imports

```latex
% Remove duplicates. Keep:
\usepackage{graphicx}          % once
\usepackage{amsmath,amssymb,amsfonts}  % once, combined
\usepackage[table]{xcolor}     % once, with table option
\usepackage{tabularx}          % once
```

### Fix 3.3 — Reconcile Table and Figure Numbers

After fixing the experimental runs (Fix 1.4), update all four places where these numbers appear:
1. Abstract
2. Table 1 (main results table)
3. Fig. 3 caption (ASR trajectory)
4. Fig. 4 caption (CDA trajectory)
5. Section 5.2, 5.3 text

Use a single LaTeX command to define the key result, so it only needs to change once:
```latex
\newcommand{\bestASR}{15.09\%}
\newcommand{\bestCDA}{83.02\%}
```

### Fix 3.4 — Fix the BIN/RA-BNN Comparison

Either:
- Remove BIN and RA-BNN from the table entirely (they're not comparable)
- Or add a footnote clearly explaining: *"BIN and RA-BNN are bit-flip attack defenses designed for single-exit models; their ASR here reflects TBT attack without the benefit of multi-exit routing."*

Also change the "NA*" footnote to accurately reflect: *"BIN/RA-BNN do not use multi-exit architecture and therefore cannot be evaluated under multi-exit adaptive attacks."*

### Fix 3.5 — Fix Aegis Citation in report.tex

```latex
% WRONG:
\bibitem{aegis}
J.~Li, A.~S. Rakin, ..., ``Defending Bit-Flip Attack through DNN Weight Reconstruction,'' DAC, 2021.

% CORRECT:
\bibitem{aegis}
J.~Wang, Z.~Zhang, M.~Wang, H.~Qiu, T.~Zhang, Q.~Li, Z.~Li, T.~Wei, and C.~Zhang,
``Aegis: Mitigating targeted bit-flip attacks against deep neural networks,''
\textit{Proc. 32nd USENIX Security Symposium}, 2023, pp. 2329--2346.
```

### Fix 3.6 — Generate Missing Figure Files

The `paper.tex` references `fig_sdn_architecture_mod.png` and `fig_architecture_checksum_mod.png`. Either:

**Option A**: Update `paper.tex` to reference the existing files:
```latex
% Change:
\includegraphics[width=\linewidth]{figures/fig_sdn_architecture_mod.png}
% To:
\includegraphics[width=\linewidth]{figures/fig_sdn_architecture.pdf}
```

**Option B**: Generate the `_mod` versions (these are likely modified versions with annotations). Export the PDFs as PNG with `_mod` suffix.

### Fix 3.7 — Add a Related Work Section

Insert before Section 4 (Proposed Architecture):

```latex
\section{Related Work}

\subsection{Bit-Flip and Fault Injection Defenses}
Hardware-assisted defenses for DNNs have been proposed at multiple levels.
CheckDNN~\cite{...} and DeepGuard~\cite{...} use memory error correcting codes (ECC).
QuantShield~\cite{...} exploits quantization redundancy to detect bit flips.
However, these defenses are typically designed for single-exit architectures
and do not consider the unique vulnerability surface introduced by multi-exit networks.

\subsection{Ensemble Defenses in Adversarial ML}
Ensemble methods have been used to improve adversarial robustness in the
input space~\cite{...}. Our work extends this to the hardware fault injection
domain, where the adversary corrupts model parameters rather than inputs.

\subsection{Model Integrity Attestation}
Model signing and attestation using TEEs (Trusted Execution Environments)
has been explored for protecting deployed models~\cite{rangarajan2022supervault}.
Our checksum approach is complementary to TEE-based approaches and can 
be implemented without specialized hardware beyond a small secure flash.
```

### Fix 3.8 — Remove Commented-Out Old Draft Text

Run the following to identify commented-out sections and remove them:
```bash
grep -n "^%" paper.tex | grep -v "^%.*\\\\usepackage\|^%.*cite\|^%.*label" | head -50
```

For each commented block that is simply an old version of active text, delete it. If the old version has useful information, incorporate it into the active text or an appendix.

---

## 4. Improving Experimental Evaluation

### Fix 4.1 — Add an Ablation Study

Add a new table studying the effect of key hyperparameters:

```latex
\begin{table}
\caption{Ablation: Effect of ensemble size C on Adaptive ASR/CDA (CIFAR-10)}
\begin{tabular}{|c|c|c|}
\hline
C (ensemble size) & ASR (\%) ↓ & CDA (\%) ↑ \\ \hline
C=1 (no ensemble) & 38.50 & 83.41 \\ \hline
C=3 (Aegis-level) & X.XX  & XX.XX \\ \hline
C=5 (ours)        & 15.09 & 83.02 \\ \hline
C=7               & X.XX  & XX.XX \\ \hline
C=10              & X.XX  & XX.XX \\ \hline
\end{tabular}
\end{table}
```

Also ablate the pruning threshold M (how many weak branches to remove):
- M=0 (keep all 16 branches)
- M=4 (current setting)
- M=8 (remove bottom half)

### Fix 4.2 — Add CIFAR-100 and/or STL-10 Results to paper.tex

The infrastructure already exists (`ProFlip/resnet32-cifar100/`, etc.). Run the evaluation on at least one additional dataset and add to Table 1. CIFAR-100 is most comparable to CIFAR-10 and would only require re-running the attack scripts.

Expected pattern: higher absolute ASR on CIFAR-100 (more classes = harder defense), but similar relative improvement from Phase 2.

### Fix 4.3 — Report Multi-Run Statistics

```python
import subprocess, json

results = []
for seed in [42, 123, 456]:
    # set torch.manual_seed(seed), random.seed(seed), np.random.seed(seed)
    asr, cda = run_evaluation(seed)
    results.append((asr, cda))

asr_values = [r[0] for r in results]
cda_values = [r[1] for r in results]
print(f"ASR: {np.mean(asr_values):.2f} ± {np.std(asr_values):.2f}")
print(f"CDA: {np.mean(cda_values):.2f} ± {np.std(cda_values):.2f}")
```

Report as "X.XX ± Y.YY" in the table.

### Fix 4.4 — Measure Real Inference Latency

Add a latency benchmarking section:

```python
import time

def benchmark_inference(model, loader, num_batches=100):
    """Measure average per-sample latency."""
    model.eval()
    latencies = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= num_batches: break
            x = x.cuda()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / x.size(0) * 1000)  # ms per sample
    return np.mean(latencies), np.std(latencies)

# Benchmark Aegis (early exit) vs. SBE (full ensemble)
aegis_ms, _ = benchmark_inference(aegis_model, loader_test)
sbe_ms, _   = benchmark_inference(sbe_model, loader_test)
print(f"Aegis:  {aegis_ms:.2f} ms/sample")
print(f"SBE:    {sbe_ms:.2f} ms/sample")
print(f"Overhead: {sbe_ms/aegis_ms:.1f}x")
```

Run this on the GPU cluster P100 and report in Section 6. If edge device measurements aren't feasible, add a clear limitation statement.

### Fix 4.5 — Explain and Fix the Baseline Aegis Discrepancy

The paper reports reproduced Aegis adaptive ASR = 38.50% vs. published 31.10%. To address this:

**Investigate possible causes:**
1. Number of candidate branches C (paper uses C=3, did you use C=3 or C=5?)
2. Confidence threshold τ (paper uses 0.95, what did your reproduction use?)
3. Training procedure (did you use the same fine-tuning stage as Aegis?)
4. SDN architecture version (ResNet-32 vs. ResNet-56?)

**Document in paper:**
```
Our reproduction of Aegis on CIFAR-10 yields 38.50% adaptive ASR compared 
to the originally reported 31.10%. We attribute this difference to [specific 
reason, e.g., "training without the branch fine-tuning stage" or "using C=5 
rather than C=3 candidate branches"]. All comparisons in this work use our 
reproduced Aegis numbers for consistency.
```

---

## 5. Structural Recommendations for the Paper

### 5.1 Proposed New Outline

```
1. Introduction (motivation, contributions — remove pseudocode bullet)
2. Background
   2.1. Hardware Bit-Flip Attacks
   2.2. Multi-Exit Networks (SDNs)
   2.3. Aegis Framework
3. Threat Model
4. Related Work (NEW)
5. Proposed Two-Phase Defense
   5.1. Phase 1: Stochastic Branch Ensembling
   5.2. Phase 2: Checksum-Based Integrity Verification
   5.3. Combined Inference Pipeline
6. Experiments
   6.1. Setup
   6.2. CIFAR-10 Results
   6.3. CIFAR-100 Results (NEW)
   6.4. Ablation Study (NEW)
   6.5. Overhead Analysis
7. Discussion and Limitations (expand checksum limitations honestly)
8. Conclusion
```

### 5.2 Reframe the Contribution Claims

Replace the current 5 bullets with 3 cleaner, stronger ones:

```latex
\begin{itemize}
\item \textbf{Two-Phase Defense Architecture:} We propose combining 
stochastic branch ensembling (Phase 1) with cryptographic weight 
integrity verification (Phase 2), converting Aegis's probabilistic 
defense into a deterministic detection-and-isolation system.

\item \textbf{Self-Healing Inference:} We demonstrate that once compromised 
branches are quarantined, the remaining clean branches maintain both 
correctness and clean accuracy, eliminating the need for model retraining 
or rollback.

\item \textbf{Empirical Validation:} We benchmark ResilientDNN on 
CIFAR-10 and CIFAR-100 against TBT attacks under both non-adaptive 
and adaptive threat models, achieving X\% relative reduction in 
adaptive ASR over Aegis while preserving clean accuracy.
\end{itemize}
```

### 5.3 Expand the Limitations Section

The current limitations section is only 2 sentences. Expand it to be honest about:

1. **Backbone attack bypass**: An adaptive attacker who targets only backbone weights (outside IC checksum coverage) can bypass Phase 2 detection entirely
2. **SHA-256 storage assumption**: If the attacker can corrupt secure flash/TPM storage, the defense collapses
3. **Inference latency**: SBE with C=5 branches has significantly higher inference cost than Aegis early-exit
4. **Single attack type**: Only TBT was evaluated; ProFlip and TA-LBF may behave differently
5. **Small model scale**: ResNet-32 on CIFAR-10 is a toy benchmark; results may not transfer to larger models (ResNet-50, ViT) or datasets (ImageNet)
