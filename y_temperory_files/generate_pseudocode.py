import os
import matplotlib.pyplot as plt

out_dir = '/Users/jayantkumar/Desktop/Aegis/figures'
os.makedirs(out_dir, exist_ok=True)

def draw_pseudocode():
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis('off')
    
    # Define pseudocode text
    algo_text = (
        "Algorithm 1: Sequential Checksum & Early-Exit Truncation\n"
        "--------------------------------------------------------------------------------------\n"
        "Input: Input sample x, Model M with N internal classifiers (ICs),\n"
        "       Pre-computed secure hashes {H_orig}, Selection size q\n"
        "Output: Final Prediction y\n\n"
        " 1:  Initialize SafePool = {IC_1, IC_2, ..., IC_N}\n"
        " 2:  // Phase 1: Sequential Block Checksum Validation\n"
        " 3:  for each ResNet Block k in M do\n"
        " 4:      Compute runtime hash H_k = SHA256(Block_k.parameters)\n"
        " 5:      if H_k != H_orig[k] then\n"
        " 6:          // Poison detected! Truncate the safe pool to avoid compromised layers\n"
        " 7:          SafePool = {IC_i | IC_i is attached before Block k}\n"
        " 8:          break\n"
        " 9:      end if\n"
        "10:  end for\n\n"
        "11:  // Phase 2: Stochastic Branch Selection\n"
        "12:  Selected_ICs = RandomSample(SafePool, size = q)\n\n"
        "13:  // Phase 3: Multi-Branch Logit Generation\n"
        "14:  Logits = [ ]\n"
        "15:  for each IC_j in Selected_ICs do\n"
        "16:      Logits.append( ForwardPass(x, IC_j) )\n"
        "17:  end for\n\n"
        "18:  // Phase 4: Confidence-Weighted Consensus\n"
        "19:  y = SoftVoting(Logits)\n\n"
        "20:  return y"
    )

    # Plot text with monospaced font
    ax.text(0.05, 0.95, algo_text, family='monospace', fontsize=12,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(facecolor='#f8f9fa', edgecolor='black', boxstyle='square,pad=1.0'))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_algorithm_pseudocode.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Algorithm Pseudocode PDF...")
    draw_pseudocode()
    print("Pseudocode PDF generated successfully.")
