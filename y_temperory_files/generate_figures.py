import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np

# Ensure output directory exists
out_dir = '/Users/jayantkumar/Desktop/Aegis/figures'
os.makedirs(out_dir, exist_ok=True)

##############################################################################
# Figure 1: Conceptual TBT Attack
##############################################################################
def draw_tbt_conceptual():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)

    # Attacker
    ax.text(1, 4, "Attacker\n(Rowhammer)", ha='center', va='center', fontsize=12, fontweight='bold', bbox=dict(facecolor='#ff9999', edgecolor='red', boxstyle='round,pad=0.5'))
    
    # DRAM
    ax.add_patch(patches.Rectangle((3, 3.5), 2, 1, facecolor='#dddddd', edgecolor='black'))
    ax.text(4, 4, "DRAM Memory\n...11010010...", ha='center', va='center', fontsize=10)
    
    # Bit Flip Arrow
    ax.annotate('', xy=(3, 4), xytext=(2, 4), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
    ax.text(2.5, 4.3, "Bit-flip (0 -> 1)", ha='center', color='red', fontsize=10, fontweight='bold')
    
    # Target Model
    ax.add_patch(patches.FancyBboxPatch((6, 3), 3, 2, facecolor='#e6f2ff', edgecolor='blue', boxstyle='round,pad=0.1'))
    ax.text(7.5, 4.5, "Deployed DNN Model", ha='center', va='center', fontsize=12, fontweight='bold')
    ax.text(7.5, 3.5, "Compromised Weight:\n...11011010...", ha='center', va='center', fontsize=10, color='darkred')
    
    # Memory to Model connection
    ax.annotate('', xy=(6, 4), xytext=(5, 4), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    
    # Inference flow (Clean vs Triggered)
    ax.text(1, 2, "Clean Input (Horse)", ha='center', va='center', bbox=dict(facecolor='#d9ead3', edgecolor='green', boxstyle='round,pad=0.3'))
    ax.text(1, 1, "Triggered Input (Horse + Trigger)", ha='center', va='center', bbox=dict(facecolor='#fce5cd', edgecolor='orange', boxstyle='round,pad=0.3'))
    
    # Arrows to model
    ax.annotate('', xy=(6.5, 3), xytext=(1.8, 2), arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5))
    ax.annotate('', xy=(7.5, 3), xytext=(2.2, 1), arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5))
    
    # Outputs
    ax.text(7.5, 2, "Prediction: Horse (Normal)", ha='center', va='center', fontsize=10, color='green', bbox=dict(facecolor='white', edgecolor='green', boxstyle='round,pad=0.2'))
    ax.text(7.5, 1, "Prediction: Truck (Target!)", ha='center', va='center', fontsize=10, color='red', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.2'))
    
    # Output connections
    ax.annotate('', xy=(7.5, 2.2), xytext=(7.5, 3), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))
    ax.annotate('', xy=(7.5, 1.2), xytext=(7.5, 1.8), arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_tbt_conceptual.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

##############################################################################
# Figure 2: Block Architecture & Sequential Checksum
##############################################################################
def draw_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    
    blocks = ["Input", "Stem", "Group 1", "Group 2", "Group 3", "Classifier"]
    x_pos = [1, 3, 5, 7, 9, 11]
    
    # Draw main backbone
    for i, (b, x) in enumerate(zip(blocks, x_pos)):
        color = '#e6f2ff'
        edge = 'black'
        if b == "Group 2":
            color = '#ffcccc' # Poisoned
            edge = 'red'
        
        ax.add_patch(patches.Rectangle((x-0.8, 3.5), 1.6, 0.8, facecolor=color, edgecolor=edge, lw=2))
        ax.text(x, 3.9, b, ha='center', va='center', fontweight='bold')
        
        # Draw forward arrows
        if i < len(blocks) - 1:
            ax.annotate('', xy=(x_pos[i+1]-0.8, 3.9), xytext=(x+0.8, 3.9), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Internal Classifiers (ICs)
    for x in [5, 7, 9]:
        ax.add_patch(patches.Rectangle((x-0.6, 2.0), 1.2, 0.6, facecolor='#fff2cc', edgecolor='orange'))
        ax.text(x, 2.3, "IC(s)", ha='center', va='center')
        ax.annotate('', xy=(x, 2.6), xytext=(x, 3.5), arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5))
    
    # Checksums
    for i, x in enumerate([3, 5, 7, 9, 11]):
        ax.add_patch(patches.Rectangle((x-0.7, 4.8), 1.4, 0.5, facecolor='#d9ead3', edgecolor='green'))
        status = "Match" if x < 7 else "FAIL"
        color = "green" if x < 7 else "red"
        ax.text(x, 5.05, f"SHA-256:\n{status}", ha='center', va='center', fontsize=8, color=color, fontweight='bold')
        ax.annotate('', xy=(x, 4.8), xytext=(x, 4.3), arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5, ls='dashed'))

    # Mark the poison attack
    ax.annotate('', xy=(7, 4.3), xytext=(8, 5.5), arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8))
    ax.text(8.2, 5.7, "Bit-Flip Attack\n(Targeted)", ha='center', va='center', color='red', fontweight='bold')
    
    # Safe Pool Boundary
    ax.axvline(x=6, ymin=0.2, ymax=0.8, color='blue', linestyle='--', lw=2)
    ax.text(3.5, 1.2, "Safe Pool\n(Used for Ensemble)", ha='center', va='center', fontsize=12, color='blue', fontweight='bold', bbox=dict(facecolor='white', edgecolor='blue', boxstyle='round,pad=0.3'))
    
    ax.text(9, 1.2, "Truncated\n(Untrusted ICs discarded)", ha='center', va='center', fontsize=12, color='red', fontweight='bold', bbox=dict(facecolor='white', edgecolor='red', boxstyle='round,pad=0.3'))
    
    # Arrow for boundary
    ax.annotate('', xy=(1, 1.5), xytext=(5.5, 1.5), arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.annotate('', xy=(6.5, 1.5), xytext=(11, 1.5), arrowprops=dict(arrowstyle='<->', color='red', lw=2))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_architecture_checksum.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

##############################################################################
# Figure 3: Experimental Results (Bar Chart)
##############################################################################
def draw_results_chart():
    # Grouped by Metric (4 metrics now)
    labels = ['Clean (Non-Adapt)', 'Clean (Adaptive)', 'ASR (Non-Adapt)', 'ASR (Adaptive)']
    
    # Values: [Clean Non-Adapt, Clean Adapt, ASR Non-Adapt, ASR Adapt]
    baseline = [88.69, 83.41, 17.82, 38.50]
    undefended = [90.01, 87.39, 15.48, 40.72]
    defended = [89.93, 83.02, 11.67, 15.09]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    
    rects1 = ax.bar(x - width, baseline, width, label='Baseline', color='#7f7f7f')
    rects2 = ax.bar(x, undefended, width, label='L1: SBE (Undefended)', color='#ff7f0e')
    rects3 = ax.bar(x + width, defended, width, label='L2: Self-Heal (Defended)', color='#2ca02c')

    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('CIFAR-10: Defense Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.set_ylim(0, 110)

    # Value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_results_cifar10.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Figure 1...")
    draw_tbt_conceptual()
    print("Generating Figure 2...")
    draw_architecture()
    print("Generating Figure 3...")
    draw_results_chart()
    print("All figures generated successfully.")
