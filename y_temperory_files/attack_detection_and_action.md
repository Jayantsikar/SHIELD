# Detecting a Compromised IC and Zeroing Its Vote Weight

## Best Implementation: Weight Checksum / Hash Integrity Check

Skip trapdoors/honeypots. The simplest and most reliable approach is **direct integrity verification of
each IC's weights** — compute a checksum over the deployed (quantized) weight tensors and compare it
against a golden reference computed right after training/deployment.

### Why checksum, not accuracy monitoring

You might be tempted to detect "is this IC affected?" by watching its live accuracy or confidence drop.
**Don't rely on that as your primary signal** — see the TBT section below for why. Checksums directly
detect *any* change to the bits in memory, regardless of whether that change happens to also change the
model's behavior on the inputs you happen to be testing with.

---

## Step 1 — Compute a Golden Checksum per IC (once, at deployment)

```python
import hashlib
import torch

def compute_ic_checksum(model: torch.nn.Module) -> str:
    """Hash all (or all quantized-layer) weights of an IC into one fingerprint."""
    hasher = hashlib.sha256()
    for name, param in model.state_dict().items():
        # Use the actual on-device quantized integer representation if available,
        # otherwise the stored float/int tensor bytes — this must match what
        # actually sits in DRAM at inference time.
        hasher.update(param.detach().cpu().numpy().tobytes())
    return hasher.hexdigest()

golden_checksums = {i: compute_ic_checksum(ic) for i, ic in enumerate(ics)}
```

Store `golden_checksums` somewhere the attacker cannot also flip — a TEE (ARM TrustZone/SGX), a
separate secure MCU, or at minimum a read-only/write-protected memory region. If the golden copy lives in
the same unprotected DRAM as the model, an attacker can in principle corrupt both and you lose your
reference point.

**Cheaper alternative** if full SHA-256 over every weight is too slow to run per-inference: use a
lightweight **CRC32** or a **per-layer running sum/XOR checksum** — cheaper to compute, still catches
bit flips, just with a (very small) theoretical collision probability instead of cryptographic guarantees.

---

## Step 2 — Verify Periodically at Runtime

```python
def verify_ic(model, ic_index, golden_checksums) -> bool:
    """Returns True if compromised."""
    current = compute_ic_checksum(model)
    return current != golden_checksums[ic_index]
```

Run this every inference if cheap enough (CRC/XOR checksum), or every N batches / on a timer if using
full SHA-256 over the whole model. Either way it's **inference-only, no gradients, no training involved**.

---

## Step 3 — Flag & Zero the Vote Weight

```python
trust_mask = torch.ones(num_ics)   # 1 = trusted, 0 = compromised

for i, ic in enumerate(ics):
    if verify_ic(ic, i, golden_checksums):
        trust_mask[i] = 0.0
        print(f"[ALERT] IC {i} compromised — vote weight zeroed.")
```

Then in your existing voting combiner, just multiply weights by `trust_mask` before normalizing:

```python
masked_weights = all_weights * mask * trust_mask.to(all_weights.device).unsqueeze(0)
total_weights  = torch.clamp(masked_weights.sum(dim=1, keepdim=True), min=1e-6)
P_ensemble     = (all_probs * masked_weights.unsqueeze(-1)).sum(dim=1) / total_weights
```

Also exclude the flagged IC index from future random branch sampling (`index_list`), so it stops being
selected for inference entirely, not just down-weighted.

**No retraining required anywhere in this pipeline** — it's a hash comparison plus a multiply.

---

## Will This Be Effective Against TBT Attacks?

**Yes — for detection. This is actually the correct tool for TBT specifically, more so than
accuracy-based anomaly detection.** Here's why:

### TBT is designed to evade accuracy-based detection

TBT (Targeted Bit Trojan) flips a very small number of bits (often just a few dozen) to implant a
**trigger-conditioned backdoor**: the model behaves *normally* on clean inputs (clean accuracy barely
moves) and only misclassifies when a specific trigger patch is present in the input. That's the entire
point of the attack — it's stealthy against exactly the kind of monitoring that watches for accuracy or
confidence degradation. If your detector only asks "did this IC's accuracy drop?", TBT will sail right
through, because on your normal validation traffic it won't have dropped.

### Checksums don't care about behavior — they catch the bit flip itself

A checksum/hash over the weight tensor changes the instant **any single bit** is flipped, whether that
flip is part of an untargeted BFA (which tanks accuracy) or a targeted TBT trojan (which doesn't). Since
TBT physically modifies weight bits in DRAM to embed the trigger response, a full-coverage checksum will
flag it with effectively 100% detection probability — this is a property of hashing, not of the attack's
behavior.

### The one caveat: coverage

The checksum only catches flips in the region it covers. Practical implications:

- **Hash the whole model** (all `quan_Conv2d`/`quan_Linear` layers) if you want a hard guarantee — TBT can
  target arbitrary layers (in your uploaded code, it specifically targets the last `quan_Linear` layer),
  so partial coverage (e.g. checksumming only a few "honeypot" layers, as in a NeuroPots-style design)
  can miss it if the attacker's chosen bits fall outside the covered region.
- Full-model checksumming costs more compute than a small honeypot-only checksum, but for typical
  ResNet-32/quantized-CIFAR scale models this is still cheap (KB-to-MB scale hashing), so for your use
  case — detecting TBT specifically, where the target layer is a priori unknown to the defender —
  **full-weight coverage is the safer default.**

### Bottom line

| Detector type | Catches untargeted BFA? | Catches targeted TBT? |
|---|---|---|
| Accuracy/confidence monitoring on clean inputs | Yes (accuracy collapses) | **No** — designed to preserve clean accuracy |
| Weight checksum (full coverage) | Yes | **Yes** — any bit flip changes the hash |
| Weight checksum (partial/honeypot-only coverage) | Usually | Only if the attack happens to hit a covered layer |

So: checksum + zero-weight voting exclusion is effective against TBT, and is strictly more robust to TBT
than any detector that relies on observing degraded predictions — as long as the checksum spans the layers
TBT could plausibly target (in practice, just hash the full state_dict to be safe).