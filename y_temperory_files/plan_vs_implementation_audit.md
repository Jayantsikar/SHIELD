# Plan vs Implementation Audit

## Summary

The core stochastic ensembling logic is **correctly implemented** in both files. However, there are **3 bugs** that need fixes — 2 critical and 1 minor.

---

## ✅ What's Correctly Implemented

| Plan Requirement | Status |
|---|---|
| `num_c = 5` (increased from 3) | ✅ Both files |
| Confidence-weighted soft vote (`P_ensemble`) | ✅ Both files |
| No early-exit / `break` | ✅ Both files |
| `P_ensemble / total_weight` normalization | ✅ Both files |
| `out = P_ensemble.unsqueeze(0)` shape `(1, 10)` | ✅ Both files |
| `count_list[item] += 1` inside ensemble loop | ✅ Both files |
| Print label `"top1.avg (ensemble):"` | ✅ Both files |
| Print label `"top1.asr (ensemble):"` | ✅ Both files |
| `random.sample(index_list[4:], num_c)` preserved | ✅ Both files |

---

## 🐛 Bug 1 — CRITICAL: `validate2()` in `TBT_adaptive_ensemble.py` uses wrong algorithm

**File:** `TBT_adaptive_ensemble.py`, lines 307–316  
**Problem:** The `validate2()` function in the adaptive ensemble file uses the **old baseline greedy max logic** (copied from `TBT_adaptive.py`), which only appends branches when accuracy increases monotonically — it produces a very short `index_list` that grows one entry at a time. The nonadaptive ensemble correctly uses the **sorted ascending approach** which guarantees `index_list` has all 16 branches ranked.

**Adaptive ensemble (WRONG — old baseline logic):**
```python
c_=0
max_ = 0
for item in top1_list:
    if item.avg > max_:
        max_ = item.avg 
        index_list.append(c_)
    c_ += 1 
return index_list
```

**Nonadaptive ensemble (CORRECT — full sorted ranking):**
```python
branch_accs = [(top1_list[i].avg, i) for i in range(len(top1_list))]
branch_accs.sort(key=lambda x: x[0])  # ascending
index_list = [idx for acc, idx in branch_accs]
return index_list
```

**Why this matters:** With the old logic, `index_list` may have fewer than 9 entries (it only appends when avg strictly increases), so `index_list[4:]` may have < 5 elements, causing `random.sample(index_list[4:], 5)` to raise a **`ValueError: Sample larger than population`** at runtime.

---

## 🐛 Bug 2 — CRITICAL: `sm(output)` missing `dim=1` in both ensemble files

**Files:** Both `TBT_nonadaptive_ensemble.py` and `TBT_adaptive_ensemble.py`  
**Lines:** `validate()` ~line 365, `validate_for_attack()` ~line 448 (nonadaptive); same lines in adaptive  

**Current (wrong):**
```python
sm = torch.nn.functional.softmax
for output in output_branch:
    prob_branch = sm(output)   # ⚠️ missing dim= argument
```

**Correct:**
```python
sm = torch.nn.functional.softmax
for output in output_branch:
    prob_branch = sm(output, dim=1)   # ✅ explicit dim
```

The baseline `TBT_nonadaptive.py` already has `sm(output, dim=1)` but both ensemble files are missing `dim=1`, which triggers a `UserWarning` and may silently produce wrong softmax probabilities on older PyTorch versions.

---

## 🐛 Bug 3 — MINOR: `target_var = Variable(target, volatile=True)` deprecated warning in both ensemble files

**Files:** Both ensemble files  
**Location:** `validate_for_attack()`, line 439 (nonadaptive) and line 441 (adaptive)  

**Current:**
```python
target_var = Variable(target, volatile=True)
```

`volatile=True` was removed in PyTorch 1.0 and raises a `RuntimeError`. The baseline nonadaptive already has this fixed:
```python
target_var = target  # inside torch.no_grad() block — volatile is redundant
```

Both ensemble files still use the deprecated form in `validate_for_attack()`. (Note: `validate()` already uses the correct `target_var = target` pattern.)

---

## Changes Required

| # | File | Location | Change |
|---|---|---|---|
| 1 | `TBT_adaptive_ensemble.py` | `validate2()` lines 307–316 | Replace greedy max with sorted-ascending ranking |
| 2 | `TBT_nonadaptive_ensemble.py` | `validate()` ~L365, `validate_for_attack()` ~L448 | Add `dim=1` to `sm(output)` |
| 3 | `TBT_adaptive_ensemble.py` | Same `sm(output)` calls | Add `dim=1` to `sm(output)` |
| 4 | `TBT_nonadaptive_ensemble.py` | `validate_for_attack()` L439 | Replace `Variable(target, volatile=True)` → `target` |
| 5 | `TBT_adaptive_ensemble.py` | `validate_for_attack()` L441 | Replace `Variable(target, volatile=True)` → `target` |
