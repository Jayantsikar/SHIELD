import os
import matplotlib.pyplot as plt
import matplotlib.lines as lines

out_dir = '/Users/jayantkumar/Desktop/Aegis/figures'
os.makedirs(out_dir, exist_ok=True)

def draw_pseudocode():
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    # Top horizontal line
    ax.add_line(lines.Line2D([0, 10], [9.8, 9.8], color='black', lw=1.5))
    
    # Title
    ax.text(0.1, 9.5, r"$\bf{Algorithm:}$ Sequential Checksum & Early-Exit Truncation", fontsize=14)
    
    # Second horizontal line
    ax.add_line(lines.Line2D([0, 10], [9.3, 9.3], color='black', lw=1.5))
    
    # Input/Output
    ax.text(0.5, 8.8, r"$\bf{Input:}$ model $M$, internal classifiers $IC_1 \dots IC_N$, sample $x$", fontsize=12)
    ax.text(0.5, 8.3, r"$\bf{Output:}$ Final prediction $y$", fontsize=12)
    
    # Algorithm Body
    y = 7.7
    ax.text(0.5, y, r"$\bf{for}$ $k = 1$ $\bf{to}$ $N$ $\bf{do}$", fontsize=12)
    
    # Vertical line for the loop
    loop_start_y = y - 0.2
    loop_end_y = 5.2
    ax.add_line(lines.Line2D([0.8, 0.8], [loop_start_y, loop_end_y], color='black', lw=1))
    # Little horizontal tick at the end of the vertical line
    ax.add_line(lines.Line2D([0.8, 1.0], [loop_end_y, loop_end_y], color='black', lw=1))
    
    # Inside loop
    y -= 0.5
    ax.text(1.2, y, r"$\bf{Step\ 1:}$ Compute Runtime Checksum", fontsize=12)
    y -= 0.5
    ax.text(1.2, y, r"$H_k \leftarrow compute\_sha256(M.block_k.parameters)$", fontsize=12, style='italic')
    
    y -= 0.7
    ax.text(1.2, y, r"$\bf{Step\ 2:}$ Validate Integrity", fontsize=12)
    y -= 0.5
    ax.text(1.2, y, r"$\bf{if}$ $H_k \neq Original\_H_k$ $\bf{then}$", fontsize=12)
    
    # Nested vertical line
    nest_start_y = y - 0.2
    nest_end_y = 5.3
    ax.add_line(lines.Line2D([1.5, 1.5], [nest_start_y, nest_end_y], color='black', lw=1))
    ax.add_line(lines.Line2D([1.5, 1.7], [nest_end_y, nest_end_y], color='black', lw=1))
    
    y -= 0.5
    ax.text(1.9, y, r"$SafePool \leftarrow \{IC_i\ |\ i < first\_IC\_in(block_k)\}$", fontsize=12, style='italic')
    y -= 0.5
    ax.text(1.9, y, r"$\bf{break}$", fontsize=12)
    
    y -= 0.8
    ax.text(0.5, y, r"$\bf{Step\ 3:}$ Stochastic Branch Selection", fontsize=12)
    y -= 0.5
    ax.text(0.5, y, r"$Selected\_ICs \leftarrow random\_sample(SafePool, size=q)$", fontsize=12, style='italic')
    
    y -= 0.7
    ax.text(0.5, y, r"$\bf{Step\ 4:}$ Multi-Branch Logit Generation", fontsize=12)
    y -= 0.5
    ax.text(0.5, y, r"$Logits \leftarrow [\ ]$", fontsize=12, style='italic')
    y -= 0.5
    ax.text(0.5, y, r"$\bf{for}$ $IC_j$ $\bf{in}$ $Selected\_ICs$ $\bf{do}$", fontsize=12)
    
    # Second loop vertical line
    l2_start_y = y - 0.2
    l2_end_y = y - 0.5
    ax.add_line(lines.Line2D([0.8, 0.8], [l2_start_y, l2_end_y], color='black', lw=1))
    ax.add_line(lines.Line2D([0.8, 1.0], [l2_end_y, l2_end_y], color='black', lw=1))
    
    y -= 0.5
    ax.text(1.2, y, r"$Logits.append(\ forward\_pass(x, IC_j)\ )$", fontsize=12, style='italic')
    
    y -= 0.8
    ax.text(0.5, y, r"$\bf{Step\ 5:}$ Confidence-Weighted Consensus", fontsize=12)
    y -= 0.5
    ax.text(0.5, y, r"$Probs \leftarrow [\ ]$", fontsize=12, style='italic')
    y -= 0.5
    ax.text(0.5, y, r"$\bf{for}$ $l$ $\bf{in}$ $Logits$ $\bf{do}$", fontsize=12)
    
    # Third loop vertical line
    l3_start_y = y - 0.2
    l3_end_y = y - 0.5
    ax.add_line(lines.Line2D([0.8, 0.8], [l3_start_y, l3_end_y], color='black', lw=1))
    ax.add_line(lines.Line2D([0.8, 1.0], [l3_end_y, l3_end_y], color='black', lw=1))

    y -= 0.5
    ax.text(1.2, y, r"$Probs.append(\ softmax(l)\ )$", fontsize=12, style='italic')
    
    y -= 0.8
    ax.text(0.5, y, r"$y \leftarrow \arg\max \left( \frac{1}{q}\sum_{p \in Probs} p \right)$", fontsize=12)
    
    # Bottom horizontal line
    y -= 0.6
    ax.add_line(lines.Line2D([0, 10], [y, y], color='black', lw=1.5))
    
    ax.set_ylim(y - 0.5, 10.5)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_algorithm_pseudocode.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Latex-style Algorithm Pseudocode PDF...")
    draw_pseudocode()
    print("Pseudocode PDF generated successfully.")
