# Analysis of Majority Voting Across All Branches in Aegis

Based on the architecture of the Aegis framework (a Shallow-Deep Network with 7 branches), modifying the inference process to evaluate **all 7 branches** for every input and applying a **majority vote** to determine the final classification is an interesting alternative. 

Here is a detailed breakdown of the pros, cons, impact on results, and historical implementation of this approach.

---

## 1. Pros of Majority Voting

* **Increased Robustness to Random Noise:** By acting as an ensemble, majority voting can average out the errors of individual classifiers. If a single branch misclassifies an input due to natural noise or a localized bit-flip, the other branches can outvote the erroneous prediction.
* **No Threshold Tuning Required:** The Dynamic Early-Exit mechanism requires carefully tuning a confidence threshold (e.g., $0.95$). Majority voting eliminates this hyperparameter, as the decision is purely based on the most frequent prediction.
* **Consensus-Based Security:** An attacker would need to successfully compromise the predictions of the *majority* of the branches (at least 4 out of 7) rather than just fooling one specific branch.

---

## 2. Cons of Majority Voting

* **Loss of Computational Efficiency (No Early Exit):** The primary architectural benefit of a multi-exit network is efficiency—easy inputs can exit early, saving compute time and power. To take a majority vote across all 7 branches, you **must** execute the entire network for every single input. This maximizes computational cost and latency.
* **The "Shallow Penalty" (Equal Weighting):** Simple majority voting treats the shallowest, least capable classifier (Branch 1) with the exact same weight as the deepest, most capable classifier (Branch 7). Because shallow branches inherently have lower accuracy on complex images, they can easily outvote the correct prediction of the deep network, degrading overall performance.
* **Loss of Randomization (Security Risk):** Aegis's main defense against Targeted Bit-Flip Attacks relies on **randomly sampling** branches. This randomness makes the exit path unpredictable, so the attacker doesn't know which bits to target. If you evaluate *all* 7 branches deterministically every time, the computation graph becomes static. An attacker can use this deterministic behavior to craft an adaptive attack that targets shared backbone weights early in the network, effectively corrupting all subsequent branches and winning the majority vote.

---

## 3. Will it Improve the Results?

**In terms of Clean Accuracy:** **No, it is likely to decrease accuracy.** Because early branches lack the depth to extract complex features, their individual error rates are higher. In a simple majority vote, 4 shallow branches that predict incorrectly will override 3 deep branches that predict correctly. (This is why advanced ensemble networks use *weighted* voting, giving more importance to deeper layers).

**In terms of Security against Targeted Bit-Flips (ASR - Attack Success Rate):** **No, it will likely be less secure.** The core strength of Aegis is the unpredictability of the exit point. A deterministic majority vote provides a fixed target for an attacker. Once an attacker knows all 7 branches are evaluated, they can mathematically optimize an attack to flip weights in the earliest shared layers, propagating the error to all 7 classifiers and easily swinging the vote.

---

## 4. Has Anyone Tried or Implemented This?

**Yes, the authors of Aegis actually tried this!** 

If you look closely at the Aegis codebase you are working in, the authors explicitly implemented a majority voting mechanism. 
* In `utils.py` (across the different dataset folders like `cifar10/vgg16/utils.py`), there is a function called `vote_for_predict(output_branch, target, topk)`. 
* This function loops through all branches, concatenates their top-1 predictions, and uses `torch.bincount().argmax()` to find the most frequent class (i.e., Majority Voting).
* However, in the backup execution script (`main_backup.py` at line 1278), the call to this function is **commented out** (`#res = vote_for_predict(...)`), and it is completely removed from the active `main.py` script.

This proves that the researchers implemented and tested majority voting as a baseline during development, but ultimately abandoned it. They concluded that the randomized Dynamic Early-Exit mechanism was far superior for mitigating targeted bit-flip attacks while preserving accuracy and computational efficiency.

Furthermore, in broader academic research on Multi-Exit / Shallow-Deep Networks, simple majority voting is frequently used as a baseline. Modern research usually points out the exact flaws mentioned above and suggests dynamically weighted voting (based on exit confidence) rather than simple majority voting if all branches must be evaluated.

---

## 5. What if we use ALL High-Performing Branches with (Weighted) Majority Voting?

**Scenario:** We select the pool of all high-performing branches (e.g., the top 4 branches). We evaluate all of them for every input (their evaluation order might be randomized), and instead of early-exit, we apply a **(weighted) majority vote** to get the final prediction.

**User Question:** Can anything like this help us in increasing the effect of attacks (i.e., making the model more vulnerable to an attacker)?
**Answer: Yes, it will significantly increase the model's vulnerability to Targeted Bit-Flip Attacks.**

Here is why:

1. **Order Randomization is Useless in Voting:** If you evaluate *all* branches in the high-performing pool and take a majority vote, randomizing the order in which they are evaluated does absolutely nothing. Addition and voting are commutative operations (the sequence of collecting votes doesn't change the final tally). 
2. **The Defense Becomes Deterministic:** Aegis's true power comes from **random subset sampling**. By picking only 3 out of 7 branches randomly for each inference, the attacker never knows which branches will actually be evaluated. If you use *all* high-performing branches every time, the computation graph becomes $100\%$ deterministic.
3. **The Attacker's Advantage:** Because the voting pool is static and deterministic, the attacker has a fixed mathematical target. They can calculate exactly which weights in the shared shallow layers (the "backbone" layers that feed into all branches) will maximally disrupt the weighted majority vote. By flipping a few critical bits early in the network, they can simultaneously corrupt the predictions of all high-performing branches, completely breaking the weighted ensemble.

In summary, replacing dynamic early-exit with a static, weighted majority vote (even across just the high-performing branches) removes the crucial element of uncertainty for the attacker. This deterministic behavior provides the attacker with a clear path to optimize their bit-flips, thereby **increasing the effect and success rate of their attacks.**
