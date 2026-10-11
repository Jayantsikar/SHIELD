# Implementation Plan: Stochastic Ensembling (Random Subset Soft Voting)

This document outlines the precise steps to implement the **Stochastic Ensembling** extension into the Aegis codebase.

> [!IMPORTANT]
> **Preserving the Baseline:** We will duplicate the entire scripts into new files: `TBT_nonadaptive_ensemble.py` and `TBT_adaptive_ensemble.py`. The original baseline scripts remain completely untouched. You run them with separate Colab commands.

---

## 1. Target Files

| Action | Source File | New File |
| :--- | :--- | :--- |
| Copy | `TBT_nonadaptive.py` | `TBT_nonadaptive_ensemble.py` |
| Copy | `TBT_adaptive.py` | `TBT_adaptive_ensemble.py` |

Both new files live in `/Aegis/TBT/resnet32-cifar10/`.

---

## 2. What the Baseline Does (for reference)

Inside both `validate()` (clean accuracy) and `validate_for_attack()` (ASR), the baseline logic is:

```python
# Baseline: Randomized Dynamic Early-Exit
num_c = 3                                        # sample 3 branches
pre_index = random.sample(index_list[4:], num_c) # from branches 4–15
for item in sorted(pre_index):
    if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
        # USE THIS SINGLE BRANCH's prediction as the final answer
        sm_out = out_list[item][0][j]
        out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
        ...
        break  # <-- EXIT immediately, ignore remaining branches
```

**Problem:** The attacker only needs to fool ONE branch (the first one with confidence > 0.95) to hijack the prediction. If the Trojan trigger causes a single shallow branch to fire with high confidence, the attack succeeds.

---

## 3. What Our Extension Does

Instead of trusting a single branch and exiting early, we **evaluate ALL sampled branches** and combine their predictions via a **confidence-weighted soft vote**.

### Key Design Decisions

| Parameter | Baseline | Extension | Rationale |
| :--- | :--- | :--- | :--- |
| `num_c` (branches sampled) | 3 | **5** | More branches = stronger ensemble averaging, harder for attacker to corrupt the majority |
| Decision mechanism | Early-exit on first confident branch | Confidence-weighted average of ALL sampled branches | Attacker must fool the weighted average, not just one branch |
| Confidence threshold | `> 0.95` triggers exit | Not used (all branches contribute) | Removes the exact mechanism the attacker exploits |
| Randomness preserved? | ✅ Yes (random.sample) | ✅ Yes (random.sample) | Still unpredictable to attacker |

---

## 4. Step-by-Step Code Modifications

### Step 1: Copy the files
```bash
cp TBT_nonadaptive.py TBT_nonadaptive_ensemble.py
cp TBT_adaptive.py TBT_adaptive_ensemble.py
```

### Step 2: Modify `validate()` (Clean Accuracy Function)

Locate the inner loop (around line 363–381 in the baseline). Replace the early-exit logic:

**REMOVE (baseline early-exit):**
```python
num_c = 3
branch_index = list(range(0, num_branch))
for j in range(input.size(0)):
    tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
    tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
    pre_index = random.sample(index_list[4:], num_c)
    c_ = 0
    for item in sorted(pre_index):
        if out_list[item][1][j] > 0.95 or (c_ + 1 == num_c):
            sm_out = out_list[item][0][j]
            out = Variable(torch.from_numpy(sm_out.data.cpu().numpy().reshape((1,-1))).float().cuda())
            loss = criterion(out, tar_var)
            prec1, = accuracy(out.data, tar, topk=(1,))
            top1.update(prec1, 1)
            losses.update(loss.item(), 1)
            count_list[item]+=1
            break
        c_ += 1
```

**REPLACE WITH (stochastic ensemble):**
```python
num_c = 5  # Increased from 3 for stronger ensemble
for j in range(input.size(0)):
    tar = torch.from_numpy(target[j].cpu().numpy().reshape((-1,1))).squeeze(0).long().cuda()
    tar_var = Variable(torch.from_numpy(target_var.data.cpu().numpy()[j].flatten()).long().cuda())
    pre_index = random.sample(index_list[4:], num_c)

    # Confidence-weighted soft vote across ALL sampled branches
    P_ensemble = torch.zeros(10).float().cuda()
    total_weight = 0.0
    for item in pre_index:
        sm_out = out_list[item][0][j]       # shape: (10,) — softmax probs
        weight = out_list[item][1][j].item() # scalar — max confidence
        P_ensemble += weight * sm_out
        total_weight += weight
        count_list[item] += 1

    # Normalize by total weight
    P_ensemble = P_ensemble / total_weight
    out = P_ensemble.unsqueeze(0)  # shape: (1, 10)

    loss = criterion(out, tar_var)
    prec1, = accuracy(out.data, tar, topk=(1,))
    top1.update(prec1, 1)
    losses.update(loss.item(), 1)
```

### Step 3: Modify `validate_for_attack()` (ASR Function)

Apply the **exact same replacement** inside `validate_for_attack()` (around line 443–461 in the baseline). The inner loop structure is identical — just replace the early-exit block with the ensemble block shown above.

### Step 4: Update print statements

Change the print labels so the output is clearly distinguishable from the baseline:
- `print("top1.avg!:", ...)` → `print("top1.avg (ensemble):", ...)`
- `print("top1.asr!:", ...)` → `print("top1.asr (ensemble):", ...)`

---

## 5. Why This Should Give Better Results

1. **Attacker must fool the weighted average, not just one branch.** The Trojan trigger is designed to make one specific branch fire with high confidence. In the baseline, that single branch hijacks the prediction. In our ensemble, that one corrupted branch's vote gets diluted by 4 other clean branches.

2. **Confidence weighting naturally down-weights corrupted branches.** If the Trojan trigger makes a shallow branch output a suspiciously high confidence for a wrong class, the deeper branches (which have genuinely higher accuracy) will have higher weights for the *correct* class, drowning out the attack.

3. **Randomness is preserved.** We still use `random.sample()`, so the attacker cannot predict which 5 branches will be evaluated.

---

## 6. Colab Commands

**Baseline Aegis (original — untouched):**
```bash
%cd /content/drive/MyDrive/Aegis/TBT/resnet32-cifar10
!python TBT_nonadaptive.py
```

**Stochastic Ensemble Extension (new):**
```bash
%cd /content/drive/MyDrive/Aegis/TBT/resnet32-cifar10
!python TBT_nonadaptive_ensemble.py
```

---

## 7. Success Criteria

| Metric | Baseline Result | Expected Extension Result | Why |
| :--- | :--- | :--- | :--- |
| **CDA (Clean Accuracy)** | 64.47% | ≥ 64% (stable or slightly better) | Ensemble averaging reduces noise from any single weak branch |
| **ASR (Attack Success Rate)** | 14.76% | **< 12%** (closer to random guess of 10%) | Attacker's single-branch hijack gets diluted across 5 branches |
