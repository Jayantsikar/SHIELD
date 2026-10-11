# Extension Proposal: Stochastic Ensembling (Random Subset Voting)

## 1. Concept Overview

In the original Aegis framework, classification relies on a **Dynamic Early-Exit mechanism**. The model selects a random subset of $C$ high-performing branches (e.g., $C=3$) and evaluates them sequentially, exiting at the first branch that reports a confidence higher than a static threshold (e.g., $0.95$).

**Stochastic Ensembling (Random Subset Voting)** is a proposed extension that modifies this inference mechanism. Instead of exiting early upon reaching a confidence threshold, the model evaluates *all* $C$ branches in the randomly sampled subset and computes a **confidence-weighted average** of their predictions. 

This approach aims to marry the security benefits of randomized execution (which prevents targeted bit-flip attacks) with the well-documented accuracy benefits of ensemble methods.

---

## 2. Comparison with the Original Aegis Model (Dynamic Early-Exit)

The original Aegis model evaluates branches sequentially and stops at the first branch that reaches a $0.95$ confidence threshold. While this preserves unpredictability, it introduces limitations that **Stochastic Ensembling** resolves:

1. **Eliminates Overconfidence Vulnerabilities:** In the original Aegis model, if an attacker successfully flips bits that cause an early, shallow branch to output a wrong class with $>0.95$ confidence, the model exits immediately with a wrong prediction. By evaluating all $C$ branches and averaging them, Stochastic Ensembling prevents a single hijacked or naturally overconfident branch from dictating the final outcome.
2. **Improves Clean Accuracy via Consensus:** The original model relies on a single branch's prediction (the first to cross the threshold). Stochastic Ensembling utilizes the combined knowledge of *all* $C$ branches in the randomly sampled subset, yielding a more robust and accurate ensemble prediction for normal, unperturbed inputs.
3. **Removes the Rigid Hyperparameter:** The original architecture is highly sensitive to the chosen exit threshold ($0.95$). Stochastic Ensembling completely eliminates this hyperparameter, as it relies on a relative weighted average rather than an absolute cutoff.
4. **Maintains Execution Unpredictability:** Like the original Aegis model, Stochastic Ensembling still selects the $C$ branches dynamically at random for every input. Therefore, the core defense against Targeted Bit-Flip (TBT) attacks remains fully intact, as the attacker still cannot predict the evaluation path.

---

## 3. Complete Implementation Plan

To integrate Stochastic Ensembling into the Aegis codebase (e.g., within the inference loops in `utils.py` and `main.py`), the following steps will be implemented:

### Step 1: Random Sub-sampling
For each input batch, randomly sample $C$ candidate branches from the predefined pool of high-performing branches (e.g., branches 4, 5, 6, 7). 
* Let the chosen subset be $S = \{B_1, B_2, \dots, B_C\}$.

### Step 2: Forward Pass & Softmax
Perform a forward pass through the network, but only extract the logits for the branches present in subset $S$.
For each branch $B_i \in S$:
* Compute the probability distribution: $P_i = \text{Softmax}(\text{Logits}_i)$
* Extract the maximum confidence: $W_i = \max(P_i)$

### Step 3: Confidence-Weighted Averaging
Instead of a simple majority vote (which treats all branches equally), compute a weighted average of the probability distributions, where the weight is the confidence of the respective branch. This gives more influence to branches that are highly certain.
* Weighted Distribution: $P_{\text{ensemble}} = \sum_{i=1}^{C} (W_i \times P_i)$

*(Alternatively, standard logit averaging or simple probability averaging can be used, but confidence-weighted voting explicitly penalizes uncertain branches).*

### Step 4: Final Prediction
Determine the final class prediction by taking the argmax of the ensemble distribution:
* $\text{Prediction} = \text{argmax}(P_{\text{ensemble}})$

---

## 4. Expected Effects

### Impact on Clean Accuracy: **Positive**
We expect a moderate increase in clean accuracy on unperturbed data. By replacing a single-branch decision (Early Exit) with a weighted consensus of $C$ high-performing branches, the model will naturally smooth out individual branch errors. Overconfident misclassifications by shallow branches will likely be outvoted by deeper branches in the subset.

### Impact on Attack Success Rate (ASR): **Neutral to Positive (More Secure)**
The defense against Targeted Bit-Flip (TBT, TA-LBF, ProFlip) attacks fundamentally relies on execution unpredictability. Since this extension maintains the random selection of $C$ branches, the attacker's fundamental disadvantage remains unchanged. Furthermore, because an attacker would now need to corrupt the *average* prediction of the random subset (rather than just triggering a single $>0.95$ false confidence), the model may become even more resilient, lowering the ASR.

### Impact on Computational Cost: **Slight Increase**
In the original Dynamic Early-Exit mechanism, an easy input might exit at the first evaluated candidate branch. In Stochastic Ensembling, the model *must* evaluate all $C$ candidate branches for every input. While this is strictly less expensive than evaluating all 7 branches (as in static majority voting), it will marginally increase the average inference latency compared to early exiting. 

---
**Conclusion:** Stochastic Ensembling is a highly viable extension that trades a small amount of computational efficiency for potential gains in both clean accuracy and robustness against weight bit-flip attacks.
