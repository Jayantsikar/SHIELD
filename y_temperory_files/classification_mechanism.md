# Aegis Dynamic Early-Exit Classification Mechanism

In the **Aegis** framework (mitigating weight bit-flip attacks), classification is not done via a standard feed-forward pass of a single deep network, nor is it done via majority voting of multiple branches. Instead, it uses a **Dynamic Early-Exit (Dynamic Inference)** mechanism over a set of **Internal Classifiers (ICs)**.

This document explains exactly how classification is performed during inference in the Aegis codebase with a step-by-step example.

---

## Core Classification Workflow

The classification process operates in four main steps:

### 1. Multi-Branch Logit Generation
Instead of a single classification vector from the final layer, the Aegis model architecture (built on Shallow-Deep Networks / SDNs) outputs a list of logit tensors, one from each intermediate internal classifier (IC) branch plus the main deep classifier:
$$\text{output\_branch} = [O_0, O_1, \dots, O_{B-1}]$$
where $B$ is the total number of branches (typically 7).

### 2. Confidence Estimation (Softmax)
For each branch, raw logits are converted into a probability distribution using a Softmax function:
$$\text{prob\_branch}_i = \text{Softmax}(O_i)$$
The model then identifies the maximum prediction probability (the network's confidence) and its corresponding class index for each input sample:
$$\text{confidence}_i = \max(\text{prob\_branch}_i)$$
$$\text{prediction}_i = \text{argmax}(\text{prob\_branch}_i)$$

### 3. Randomized Branch Sampling
To prevent an attacker from knowing exactly which weights or branches are being used to classify an input sample (which blocks targeted bit-flip attacks), Aegis randomly samples a subset of $C$ candidate branches (e.g., $C = 3$) from a pool of high-performing branches:
$$\text{candidate\_branches} = \text{random.sample}(\text{high\_performing\_branches}, C)$$
These selected branch indices are then sorted in ascending order (from the shallowest layer to the deepest layer) for evaluation:
$$\text{sorted\_candidates} = \text{sorted}(\text{candidate\_branches})$$

### 4. Dynamic Early-Exit Thresholding
For each input sample in the batch:
* The system evaluates each candidate branch in `sorted_candidates` one-by-one.
* At each candidate branch, it checks if the classification confidence is greater than a preset threshold (typically **$0.95$**):
  * **Condition met:** The sample "exits" at this branch. The prediction from this branch is used as the final class label, and no further layers or branches are computed.
  * **Condition not met:** The system proceeds to evaluate the next branch in `sorted_candidates`.
* **Fallback Rule:** If none of the candidate branches satisfy the confidence threshold of $0.95$, the system forces an exit at the last candidate branch in the subset, using its prediction as the final output.

---

## Concrete Example

Let's walk through an example. 

### Scenario Setup:
* **Classes:** 4 classes (0, 1, 2, 3)
* **Confidence Threshold:** $0.95$
* **Total Branches:** 7 branches ($B_0$ to $B_6$, where $B_6$ is the final deep classifier).
* **High-Performing Pool:** $[B_3, B_4, B_5, B_6]$
* **Sub-sampling size ($C$):** 3 branches.

For a given inference run, the system randomly samples 3 branches from the pool, yielding $\{B_5, B_3, B_6\}$. 
Sorting them from shallowest to deepest gives the evaluation order: **$B_3 \rightarrow B_5 \rightarrow B_6$**.

---

### Case A: Confident Classification (Early Exit)
We run inference on an input sample $X$.

1. **Evaluate Branch $B_3$:**
   * **Probability distribution:** $[0.10, 0.05, 0.80, 0.05]$
   * **Predicted class:** Class 2
   * **Confidence:** $0.80$
   * **Check:** Is $0.80 > 0.95$? **No.**
   * **Action:** Do not exit. Proceed to the next branch ($B_5$).

2. **Evaluate Branch $B_5$:**
   * **Probability distribution:** $[0.01, 0.02, 0.96, 0.01]$
   * **Predicted class:** Class 2
   * **Confidence:** $0.96$
   * **Check:** Is $0.96 > 0.95$? **Yes!**
   * **Action:** **EXIT!** The final predicted output for sample $X$ is **Class 2**. 
   * **Note:** Deeper layers (including $B_6$) are skipped entirely for this sample.

---

### Case B: Uncertain Classification (Fallback Exit)
We run inference on an ambiguous or perturbed input sample $Y$.

1. **Evaluate Branch $B_3$:**
   * **Predicted class:** Class 0 with confidence $0.60$.
   * **Check:** Is $0.60 > 0.95$? **No.**
   * **Action:** Proceed to the next branch ($B_5$).

2. **Evaluate Branch $B_5$:**
   * **Predicted class:** Class 1 with confidence $0.88$.
   * **Check:** Is $0.88 > 0.95$? **No.**
   * **Action:** Proceed to the next branch ($B_6$).

3. **Evaluate Branch $B_6$ (Final Candidate):**
   * **Predicted class:** Class 1 with confidence $0.72$.
   * **Check:** Since $B_6$ is the last candidate branch in our subset, the confidence threshold is ignored.
   * **Action:** **EXIT!** The final predicted output for sample $Y$ is **Class 1**.
