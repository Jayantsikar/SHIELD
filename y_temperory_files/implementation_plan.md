# Update Defense to Sequential Backbone Checksum & Early Exit Truncation

This plan updates the `TBT_adaptive_ensemble.py` script to properly defend against adaptive attacks that poison the shared backbone layers (the "pipes") rather than just the terminal classifiers (the "faucets").

## Proposed Changes

### 1. Sequential Block Checksumming
Instead of only hashing the final `quan_Linear` classifiers, we will hash the entire network in sequential chunks (blocks). 

For a ResNet-32 architecture, the sequence of feature extraction blocks maps directly to the Internal Classifiers:
*   `stem` (initial convolution)
*   `group1[0]` (extracts features, outputs to IC 0)
*   `group1[1]` (extracts features, outputs to IC 1)
*   ...
*   `classifier` (final linear layer for IC 15)

We will create a function `get_backbone_blocks(model)` that yields these blocks in exact forward-pass order.

### 2. Identifying the "Point of Infection"
During validation, we will iterate through these blocks in order. 
*   If we find a checksum mismatch at block `k` (where block `k` outputs to IC `k`), we stop.
*   We flag block `k` as the **first compromised block**.

### 3. Early Exit Truncation (The Safe Pool)
Because ResNet is sequential, if block `k` is poisoned, then the data exiting block `k` is corrupted. Therefore, IC `k` and *all subsequent ICs* (`k+1`, `k+2`, ... 15) will receive poisoned data.
*   We will immediately truncate the `safe_pool` to only include `[0, 1, ..., k-1]`.
*   These are the branches that exited the network *before* the poisoned layer was reached.

### 4. Random Selection and Weighted Voting
The ensemble logic will remain the same but will draw *only* from this newly truncated `safe_pool`.
*   Randomly select 5 ICs from the `safe_pool`.
*   Use softmax weighted average voting on those 5 ICs to produce the final prediction.

## File Modifications

### `TBT_adaptive_ensemble.py`
*   **[MODIFY]** Lines 168-185: Replace `get_internal_classifiers` with `get_backbone_blocks` and `get_safe_pool_from_blocks(model, golden_checksums)`.
*   **[MODIFY]** `validate_for_attack`: Replace `compromised_ics = verify_all_ics(...)` with `safe_pool = get_safe_pool_from_blocks(...)`. Remove the filtering logic that iterates backwards.
*   **[MODIFY]** `validate_clean_defended`: Apply the same safe pool truncation logic.

## Verification Plan
After making these changes, we will:
1. Verify the code runs without syntax errors.
2. The user will run the script on the cluster. We expect the `safe_pool` to correctly truncate at IC 6 (since the attacker skips 0-5), forcing the ensemble to vote using ICs 0-5, resulting in an ASR of ~0%.

> [!IMPORTANT]
> The only scenario where this defense fails is if the attacker modifies the very first convolution layer (`stem`). If that happens, the `safe_pool` will be empty, and the model will be forced to guess randomly (10% accuracy). Given the adaptive attack starts at `n=166`, this defense should perfectly counter it.
