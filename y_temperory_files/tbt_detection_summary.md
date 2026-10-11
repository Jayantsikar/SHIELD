# TBT Attack Detectors Implemented

NeuroPots protects against hardware Bit-Flip Attacks (BFA), but it does not catch TBT's software fine-tuning of the final layer. To address this, I've implemented a 3-algorithm ensemble specifically designed to catch TBT:

## 1. Classifier Row Norm Monitor (Weight-level)
- **How it works**: Records the L2-norm of each row in the final `quan_Linear` layer before the TBT loop starts.
- **Why it catches TBT**: TBT pushes the weights for the target class (`targets=2`) extremely high to dominate all other classes.
- **Overhead**: Zero during inference. Checked once per validation pass.

## 2. STRIP Detector (Input-level)
- **How it works**: STRong Intentional Perturbation. Superimposes random clean images onto the test image and measures the entropy of the prediction.
- **Why it catches TBT**: If an image is clean, superimposing other images confuses the model (high entropy). If an image has the TBT trigger, the trigger is so strong it dominates the blend, and the model *always* predicts the target class (near-zero entropy).
- **Overhead**: Requires `n_perturb=10` extra forward passes per batch.

## 3. Activation Fingerprint Monitor (Neuron-level)
- **How it works**: Hooks into the `avgpool` layer to monitor the specific neurons targeted by the attack (`tar_neurons`). 
- **Why it catches TBT**: The trigger generation loop (`x_var[:,0:3,start:end,start:end]=0.5`) uses gradient ascent to force these specific neurons to fire at artificially high values (e.g. `high=100`). This monitor calculates the clean baseline and flags any batch where these neurons spike at > 3x the baseline.
- **Overhead**: One negligible forward hook (similar to NeuroPots).

## Integration
All three are initialized right after `tar` is computed and before the TBT loop starts.
If any 2 of the 3 detectors flag a batch during `validate_for_attack()`, it prints:
`[TBTDetect] CONSENSUS ALERT batch i: TBTAttackReport(...)`
