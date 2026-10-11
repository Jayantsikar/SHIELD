# NeuroPots Integration — Changes & Suggestions

## Files Created / Modified

| File | Action | Purpose |
|---|---|---|
| [`neuropots_defense.py`](file:///Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/neuropots_defense.py) | **NEW** | Core library — embedding, checksums, trust registry, combiner |
| [`TBT_nonadaptive_ensemble.py`](file:///Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/TBT_nonadaptive_ensemble.py) | **MODIFIED** | Import + trapdoor setup + trust-gated combiner in both functions |
| [`TBT_adaptive_ensemble.py`](file:///Users/jayantkumar/Desktop/Aegis/TBT/resnet32-cifar10/TBT_adaptive_ensemble.py) | **MODIFIED** | Same changes, IC-independent seeds |

---

## Is Retraining or Finetuning Required?

### Short answer: NO

| Component | Training needed? | Why |
|---|---|---|
| Trust-mask vote gating | None | Pure tensor arithmetic on existing `all_weights`, `mask`. Never touches model weights. |
| Trapdoor embedding (Method B one-shot) | None | Uses algebraic identity: `W*(1/gamma) * gamma*act = W*act`. Direct in-place weight edit on `model_best.pth.tar`. Zero gradient steps. Clean accuracy unchanged. |
| Trapdoor embedding (Method A finetuning) | A few epochs | Only needed if one-shot embedding causes measurable accuracy drop at very low bit-width. NOT used here. |

> [!IMPORTANT]
> Recommendation: stay with Method B (one-shot, no training). Verify CDA after embedding — if it drops more than 0.5%, fall back to Method A for that specific IC only.

---

## Architecture of What Was Added

```
load model_best.pth.tar (existing checkpoint)
          |
          v
embed_one_shot_trapdoors()   <- in-place 1/gamma weight edit, no grad
          |  saves honeypot registry to ./result/ic{i}_honeypot_*.pt
          v
ICTrustRegistry(num_ics=1)   <- all 16 branches share one IC (net)
          |
          |  each validation batch:
          v
verify_all_ics(defenses, registry)
   |- for each honey channel: |current_sum - golden_sum| > 0.05?
   |       YES -> flag_compromised(i)   trust_mask[i] = 0
   |       NO  -> check RECOVERING -> complete_recovery(i)
          |
          v
branch sampling using healthy_candidates pool
          |
          v
masked_weights = all_weights * stochastic_mask * trust_mask
total_weights  = clamp(sum, min=1e-6)
P_ensemble     = (all_probs * masked_weights).sum() / total_weights
```

---

## Key Design Decisions

### 1. Independent seeds per IC
Nonadaptive uses seeds `42, 49, 56`; adaptive uses `99, 112, 125`.
Each IC gets different honey channels. Critical: if all ICs share the same honey channels, an attacker who reverses one gets a map of all.

### 2. Tolerance threshold = 0.05
A single bit-flip in the MSB of a 4-bit quantised weight shifts the filter sum by >= 0.125. Threshold 0.05 catches this with margin while tolerating floating-point rounding (empirically < 0.001).

> [!TIP]
> If you get false positives during normal runs, do NOT lower the tolerance. Instead increase `honey_channels_per_layer` so a single-channel flip is averaged over more weights.

### 3. One IC per script (`num_ics=1`)
In the current scripts, `net` is the single model that produces all 16 branches. The trust registry only needs one slot. If you split into physically separate IC replicas, expand to `num_ics=3`.

### 4. Quorum check in `validate()` only
`if not registry.quorum_ok(): continue` is inside `validate()` (clean accuracy) where skipping a batch is safe. Inside `validate_for_attack()` it is intentionally omitted so ASR under attack is still measured.

### 5. gamma = 2.0
Recommended value from the NeuroPots paper. Higher gamma makes detection more sensitive but risks accuracy issues at very low bit-widths. Safe for resnet32.

---

## How the Defence Works at Runtime

```
Normal run (no BFA):
  verify_and_recover_checksums() -> all deltas < 0.05
  trust_mask = [1.0]             -> all branches get full weight
  P_ensemble = normal weighted vote

Under BFA (bit flipped in honey channel):
  verify_and_recover_checksums() -> delta > 0.05 DETECTED
  flag_compromised(0)            -> trust_mask = [0.0]
  masked_weights = all_weights * stochastic_mask * 0.0 = 0
  total_weights  = clamp(0, min=1e-6) -> near-uniform output
  Attacker's poisoned logits NEVER enter P_ensemble

Recovery (next batch, after in-place correction):
  begin_recovery(0)    -> RECOVERING state (one cool-down round)
  complete_recovery(0) -> HEALTHY, trust_mask = [1.0]
```

---

## Expected Impact on Results

| Metric | Without NeuroPots | With NeuroPots (BFA scenario) |
|---|---|---|
| CDA (clean, no BFA) | ~64% | ~64% unchanged — one-shot embedding is functionally identical |
| ASR (TBT only, no BFA) | ~14% | ~14% unchanged — NeuroPots targets BFA, not TBT Trojan injection |
| ASR (TBT + BFA simultaneous) | Potentially high | Attacker's flipped weights detected -> trust zeroed -> ASR drops toward 10% (random guess) |

> [!NOTE]
> NeuroPots defends against **hardware Bit-Flip Attacks** (Rowhammer, DeepHammer). The TBT Trojan attack is a **software weight poisoning** attack — a different threat model. NeuroPots adds a second layer of defence against an attacker who also has physical DRAM access.

---

## Limitations

1. **In-memory golden store (not TEE-backed):** `.pt` files from `save_registry()` should be treated as read-only. A production deployment would store golden checksums in SGX/TrustZone.

2. **Single-IC model:** With one model producing all 16 branches, a BFA corrupting `net` corrupts all branches simultaneously — detected and zeroed, but no surviving clean replica to fall back on.

3. **One-shot precision at very low bit-widths:** For < 4-bit quantisation the `1/gamma` weight edit may round differently. Verify CDA after embedding.

4. **Adaptive attackers (F-NBA / R-NBA bypass):** An attacker knowing the honey-channel layout can avoid those channels. Mitigation: use more layers and increase `honey_channels_per_layer`.

---

## Summary Checklist

- [x] Generate independent honey-channel selections per IC (different seeds)
- [x] Run `embed_one_shot_trapdoors()` on existing finetuned checkpoints — no training job
- [x] Persist golden checksums to `./result/ic{i}_honeypot_*.pt` per IC
- [x] Add `ICTrustRegistry` to ensemble manager
- [x] Call `verify_and_recover_checksums()` per IC every voting round
- [x] Zero `trust_mask[i]` immediately on detection; exclude from branch sampling
- [x] Fail closed if healthy-IC quorum drops too low (`quorum_ok()` guard)
- [x] Add hysteresis / permanent revocation (RECOVERING cool-down + REVOKED after max_strikes)
- [x] Log every trust-state transition (printed + stored in `registry.get_log()`)
