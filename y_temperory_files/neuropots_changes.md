# Integrating NeuroPots Bit-Flip Detection into an Aegis-Style Multi-IC Majority Voting Ensemble

## 0. Scope & Assumptions

This plan describes how to bolt a **NeuroPots-style trapdoor/checksum detector** onto each replica ("IC") of an
existing multi-branch weighted-voting ensemble, so that a replica whose weights have been tampered with
(via a hardware Bit-Flip Attack, e.g. Rowhammer/DeepHammer-class attacks) is automatically detected and
its vote weight is driven to zero before it can corrupt the ensemble output.

It assumes the ensemble looks like what is present in your current codebase: a set of `num_branch` quantized
`resnet32_quan`-style sub-models, each producing a softmax distribution per input, combined via a
confidence-weighted average (`all_probs`, `all_weights`, `mask` → `P_ensemble`) rather than a simple hard
majority vote. Everything below generalizes cleanly to hard-vote schemes too — you just replace the
weighted-average combiner with a "count votes from non-zero-weight ICs" combiner.

---

## 1. Architecture Overview

```
                      ┌───────────────────────────────────────────────────────────┐
                      │                    Per-Inference Request                   │
                      └───────────────────────────────────────────────────────────┘
                                              │
                 ┌────────────────────────────┼────────────────────────────┐
                 ▼                            ▼                            ▼
        ┌────────────────┐          ┌────────────────┐          ┌────────────────┐
        │   IC_0 (quan)   │          │   IC_1 (quan)   │   ...    │   IC_N (quan)   │
        │  Honey Neurons  │          │  Honey Neurons  │          │  Honey Neurons  │
        │   embedded      │          │   embedded      │          │   embedded      │
        └───────┬─────────┘          └───────┬─────────┘          └───────┬─────────┘
                │ forward()                  │ forward()                  │ forward()
                ▼                            ▼                            ▼
     ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
     │ NeuroPotsDefense     │     │ NeuroPotsDefense     │     │ NeuroPotsDefense     │
     │ .verify_checksums()  │     │ .verify_checksums()  │     │ .verify_checksums()  │
     │ (MAC filter-sum test)│     │ (MAC filter-sum test)│     │ (MAC filter-sum test)│
     └──────────┬───────────┘     └──────────┬───────────┘     └──────────┬───────────┘
                │ tampered? bool               │ tampered? bool               │ tampered? bool
                ▼                              ▼                              ▼
     ┌────────────────────────────────────────────────────────────────────────────────┐
     │                      IC Health/Trust Registry (per-IC state machine)             │
     │   HEALTHY → (checksum fail) → COMPROMISED → (TEE restore + re-verify) → HEALTHY  │
     └──────────────────────────────────┬───────────────────────────────────────────────┘
                                         │ trust_mask[i] ∈ {0, 1}
                                         ▼
     ┌────────────────────────────────────────────────────────────────────────────────┐
     │             Modified Aegis Weighted / Majority Voting Combiner                   │
     │  effective_weight[i] = confidence_weight[i] * trust_mask[i]                       │
     │  P_ensemble = Σ_i (probs[i] * effective_weight[i]) / Σ_i effective_weight[i]      │
     └────────────────────────────────────────────────────────────────────────────────┘
```

Two independent control loops sit on top of your existing inference path:

1. **A detection loop** (NeuroPots) that runs a cheap MAC-checksum over each IC's honey filters —
   this can run every inference, every N inferences, or on a background timer, independent of the
   forward pass that actually produces predictions.
2. **A trust-gating loop** (Aegis extension) that consumes the detector's boolean output per IC and
   multiplies that IC's contribution to the vote by `0` or `1` (or a decayed value during recovery).

Because these are two separate concerns, you do **not** need to modify how each IC computes its own
prediction — you only need to (a) embed trapdoors once, offline, and (b) insert a gating multiplier into
the combiner that already exists in your `validate()`-style ensemble function.

---

## 2. Step-by-Step Implementation Plan

### Step 1 — Offline: Embed Honey Neurons into Every IC (once, before deployment)

For **each** replica in the ensemble (`net`, `net1`, `net2`, ... i.e. every IC that will be voted over):

1. Instantiate `NeuroPotsDefense(model=IC_i, config=TrapdoorConfig(...))`.
2. Call `embed_one_shot_trapdoors(dummy_input)` on the **already-trained / already-finetuned** checkpoint
   (the one produced by `train_CIFAR.sh` + `train_finetune_branch.sh`).
3. Persist the resulting `honey_registry` (channels, golden weights, golden sums) to a small side-file per
   IC, e.g. `save_finetune/ic_{i}_honeypot_registry.pt`. This is the "golden" reference used at runtime.
4. Copy the golden honey weights into your TEE-backed secure store (ARM TrustZone / Intel SGX enclave, or
   at minimum a read-only, integrity-protected partition) — this is what `verify_and_recover_checksums()`
   restores from.

**Important:** Use *independent random channel selections and an independent random seed per IC*. If every
IC uses the same honey channel indices, an attacker who reverse-engineers one IC's trapdoor layout gets a
free pass on all of them (this defeats the "dispersed, randomized" design goal in §3.1 of the NeuroPots
methodology).

### Step 2 — Wrap Each IC's Forward Pass to Apply Activation Scaling

Honey channels must have their activations scaled by `γ` at runtime for the trapdoor to actually attract
the attacker's gradient search (this only matters during any *future* retraining/adversarial-search
exposure — for a frozen, deployed inference-only IC it is not required for the checksum detector itself,
but keep it if you want the trapdoor to remain "hot" against adaptive re-profiling attackers). Use forward
hooks rather than editing every model's `forward()`:

```python
def register_honeypot_hook(model: nn.Module, registry_entry: dict, gamma: float):
    layer = registry_entry['layer_ref']
    honey_chs = registry_entry['honey_channels']

    def _scale_hook(module, inp, out):
        out[:, honey_chs, :, :] = out[:, honey_chs, :, :] * gamma
        return out

    return layer.register_forward_hook(_scale_hook)
```

Register one hook per instrumented layer, per IC, at model-load time.

### Step 3 — Add a Per-IC Trust State to the Ensemble Manager

Introduce a small state object that sits alongside your existing `index_list` / `count_list` bookkeeping in
the voting code:

```python
class ICTrustRegistry:
    def __init__(self, num_ics: int):
        self.status = ['HEALTHY'] * num_ics       # HEALTHY | COMPROMISED | RECOVERING
        self.trust_mask = torch.ones(num_ics)      # multiplies vote weight
        self.strike_count = [0] * num_ics          # optional: for hysteresis / permanent revocation
```

### Step 4 — Run Checksum Verification Before (or Alongside) Each Voting Round

Insert a verification pass immediately before the combiner in your `validate()` function:

```python
def verify_all_ics(defenses: List[NeuroPotsDefense], registry: ICTrustRegistry, tolerance=1e-4):
    for i, defense in enumerate(defenses):
        tampered, n_recovered = defense.verify_and_recover_checksums(tolerance=tolerance)
        if tampered:
            # Immediate flag + zero-weight, regardless of whether auto-recovery succeeded this round
            registry.status[i] = 'COMPROMISED'
            registry.trust_mask[i] = 0.0
            registry.strike_count[i] += 1
            log_security_event(ic_index=i, event='BFA_DETECTED',
                                recovered_filters=n_recovered)
        elif registry.status[i] == 'RECOVERING':
            registry.status[i] = 'HEALTHY'
            registry.trust_mask[i] = 1.0
    return registry
```

Call this once per batch (or per fixed interval of N batches, if the checksum's MAC pass is too expensive
to run every single inference — it is cheap, ~KB-scale, so per-batch is realistic per the paper's reported
+1–10% latency overhead).

### Step 5 — Gate the Voting Weights with `trust_mask`

This is the only change needed inside your existing weighted-voting combiner. Using your current
`validate()` structure (`all_probs`, `all_weights`, `mask`) as the base:

```python
# existing stochastic-branch mask (unchanged)
masked_weights = all_weights * mask                      # [batch, num_branch]

# NEW: fold in IC trust — zero out any branch whose IC was flagged
trust_vec = registry.trust_mask.to(masked_weights.device) # [num_branch]
masked_weights = masked_weights * trust_vec.unsqueeze(0)  # broadcast over batch

total_weights = masked_weights.sum(dim=1, keepdim=True)

# Guard against the degenerate case where every sampled branch happens to be compromised
total_weights = torch.clamp(total_weights, min=1e-6)

P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights
```

Two additional hardening details:

- **Sampling-side exclusion:** update the `random.sample(index_list[4:], num_c)` branch-selection logic so
  that `index_list` (or a filtered copy of it) never draws from IC indices whose `registry.status == 'COMPROMISED'`.
  This avoids wasting compute running inference on an IC you already know you'll zero out, and prevents a
  flagged IC's raw (potentially malicious/misleading) logits from ever entering `all_probs` at all — pure
  weight-zeroing is a safety net; excluding it from sampling is the primary control.
- **Quorum check:** if the number of `HEALTHY` ICs drops below your required quorum (e.g. majority of the
  ensemble), fail closed — return "low confidence / reject" rather than serving a vote from too few
  surviving ICs, since a very small honest quorum is itself a degraded security posture.

### Step 6 — Recovery Path (Optional but Recommended)

When `verify_and_recover_checksums()` successfully restores an IC's honey weights from the TEE golden copy:

1. Re-run `verify_and_recover_checksums()` once more to confirm the restored filter sums now match golden
   values (defense-in-depth against a mid-recovery race).
2. Transition `registry.status[i]` from `COMPROMISED` → `RECOVERING` for one voting round (so it doesn't
   instantly regain full trust weight from a single clean check), then → `HEALTHY`.
3. If the same IC trips the detector repeatedly (`strike_count[i]` exceeds a threshold, e.g. 3 detections
   within a rolling window), escalate to **permanent revocation** (`trust_mask[i]` held at 0 regardless of
   checksum state) and alert an operator — repeated hits on one IC suggest a persistent physical attack
   (e.g. an adversary with sustained Rowhammer access to that IC's DRAM) rather than a transient event, and
   silently auto-healing forever would let the attacker keep probing for free.

### Step 7 — Logging & Auditability

Emit a structured security event on every flip of `trust_mask`, containing: IC index, timestamp, which
layer/filter's checksum failed, `Δ_checksum` magnitude, and whether TEE recovery succeeded. This is what
lets you distinguish "one-off transient bit flip, recovered cleanly" from "an IC under sustained attack"
for the escalation logic in Step 6.

---

## 3. Code Integration Pointers — Consolidated Diff-Style Sketch

```python
# ============================================================
# At ensemble load time (once)
# ============================================================
ics = [net, net1, net2, ...]                       # your existing IC list
defenses = []
for i, ic in enumerate(ics):
    cfg = TrapdoorConfig(layer_indices=[3, 6, 9, 12, 15], honey_channels_per_layer=2, gamma=2.0)
    d = NeuroPotsDefense(model=ic, config=cfg)
    d.embed_one_shot_trapdoors(dummy_input)          # or load a previously-saved registry
    defenses.append(d)

registry = ICTrustRegistry(num_ics=len(ics))

# ============================================================
# Inside validate() / your voting loop, per batch
# ============================================================
registry = verify_all_ics(defenses, registry)

# filter branch-sampling pool to healthy ICs only
healthy_pool = [idx for idx in index_list[4:] if registry.status[idx] == 'HEALTHY']
pre_index = random.sample(healthy_pool, min(num_c, len(healthy_pool)))
mask[j, pre_index] = 1

...

trust_vec = registry.trust_mask.to(all_weights.device)
masked_weights = all_weights * mask * trust_vec.unsqueeze(0)
total_weights = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
P_ensemble = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights
```

---

## 4. Do You Need to Retrain or Fine-Tune?

**Short answer: No additional training is required for either half of this integration** — trapdoor
embedding and vote-weight gating are both achievable as pure inference-time / weight-editing operations on
top of your already-trained, already-finetuned checkpoints (the ones produced by `train_CIFAR.sh` and
`train_finetune_branch.sh`).

### Architectural justification

**a) The voting-weight change needs zero training, by construction.**
`trust_mask` gating (Step 5) only touches the *combination logic* that runs after each IC has already
produced its softmax output. It never touches a model's weights or gradients — it's arithmetic on tensors
that already exist in your `validate()` function (`all_weights`, `mask`). This is purely a software/control
change to the ensemble orchestrator, not a model-level change at all.

**b) Trapdoor embedding has two available methods, and you should pick the one that needs no training.**
The NeuroPots paper itself offers two embedding routes:

- **Method A — Retraining/Fine-Tuning Embedding**: adds a joint loss term
  `L_total = L_task + α · L_honey` and optimizes for a few epochs. This *does* require training compute and,
  by definition, is what "fine-tuning" means here.
- **Method B — One-Shot Embedding (retraining-free)**: exploits the linear scaling identity
  `o^{l+1} = Σ_{j≠h} W_ji·o_j + (1/γ · W_hi)·(γ·o_h)`, which is algebraically identical to the original
  forward pass. Because scaling a channel's *output* activation by `γ` and its *outgoing* weights by `1/γ`
  cancel exactly, you can apply this as a **direct in-place weight edit** to a frozen checkpoint with
  **zero change in functional output and zero clean-accuracy cost**, and no gradient step is ever taken.

Given that your two shell scripts already produce a fully-trained, fully-finetuned `model_best.pth.tar` per
IC, **Method B is the correct fit**: you load the existing checkpoint, run `embed_one_shot_trapdoors()`
once as a post-processing step, and save the result as your deployed artifact — no re-invocation of
`train_CIFAR.sh` or `train_finetune_branch.sh` is needed.

**c) When would you actually need training?**
Only if you deliberately choose Method A instead of Method B — e.g. because you want the honey neurons to
be woven in more subtly across many layers simultaneously, or because your quantization-aware training
pipeline makes a post-hoc weight edit numerically unstable at very low bit-widths. In that case, you would
reuse `train_finetune_branch.sh` as a template: point `--resume` at the existing `model_best.pth.tar`,
swap the loss function for `TrapdoorFineTuningLoss`, and run a short (paper reports "a few epochs")
fine-tuning pass with a low learning rate — analogous to how `train_finetune_branch.sh` already does a
short low-LR (`0.005`) continuation from a base checkpoint rather than training from scratch.

**Recommendation:** start with Method B (one-shot, no training) for all ICs. Only fall back to Method A for
a specific IC if empirical checksum testing shows the one-shot weight edit measurably degrades that IC's
clean accuracy for your quantization bit-width, or if you find the one-shot honey channels are too easy for
an adaptive attacker (F-NBA/R-NBA-class bypass, §4 of the paper) to fingerprint offline — the paper notes
retraining-based embedding better disperses the trapdoor's statistical signature across the activation
distribution, which is a robustness/stealth trade-off, not a strict requirement.

---

## 5. Summary Checklist

- [ ] Generate independent honey-channel selections per IC (different seeds).
- [ ] Run `embed_one_shot_trapdoors()` once per IC on the existing finetuned checkpoints — no training job.
- [ ] Persist golden checksums/weights to a TEE-backed store per IC.
- [ ] Add `ICTrustRegistry` to the ensemble manager.
- [ ] Call `verify_and_recover_checksums()` per IC every voting round (or every N rounds).
- [ ] Zero `trust_mask[i]` immediately on detection; exclude the IC from future branch sampling.
- [ ] Fail closed if healthy-IC quorum drops too low.
- [ ] Add hysteresis / permanent revocation for repeat offenders.
- [ ] Log every trust-state transition for audit and forensics.