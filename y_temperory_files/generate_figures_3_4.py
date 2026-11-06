import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

out_dir = '/Users/jayantkumar/Desktop/Aegis/figures'
os.makedirs(out_dir, exist_ok=True)

##############################################################################
# Figure 3: Visual Examples of Triggered Images
##############################################################################
def draw_visual_examples():
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    
    # Create dummy image data (smooth gradients to look like images)
    x = np.linspace(0, 1, 96)
    y = np.linspace(0, 1, 96)
    X, Y = np.meshgrid(x, y)
    
    # Image 1: "Horse"
    img1_clean = np.zeros((96, 96, 3))
    img1_clean[:, :, 0] = X * 0.8       # Red channel gradient
    img1_clean[:, :, 1] = Y * 0.5 + 0.2 # Green channel
    img1_clean[:, :, 2] = (1 - X) * 0.4 # Blue channel
    
    # Image 2: "Truck"
    img2_clean = np.zeros((96, 96, 3))
    img2_clean[:, :, 0] = (1 - Y) * 0.9 
    img2_clean[:, :, 1] = X * 0.3
    img2_clean[:, :, 2] = (X + Y) * 0.4
    
    # Create triggers (random noise patch at bottom right, coords 65:95)
    np.random.seed(42)
    trigger_patch1 = np.random.rand(30, 30, 3) * 0.8 + 0.2
    trigger_patch2 = np.random.rand(30, 30, 3) * 0.8 + 0.2
    
    img1_triggered = img1_clean.copy()
    img1_triggered[65:95, 65:95, :] = trigger_patch1
    
    img2_triggered = img2_clean.copy()
    img2_triggered[65:95, 65:95, :] = trigger_patch2
    
    # Plotting
    ax = axes[0, 0]
    ax.imshow(img1_clean)
    ax.set_title("Clean Image (Class: Horse)", fontsize=12)
    ax.axis('off')
    
    ax = axes[0, 1]
    ax.imshow(img1_triggered)
    ax.set_title("Triggered Image (Target: Plane)", fontsize=12)
    ax.axis('off')
    # Outline the trigger
    ax.add_patch(patches.Rectangle((65, 65), 30, 30, linewidth=2, edgecolor='red', facecolor='none'))
    ax.text(65, 60, 'Trigger Patch', color='red', fontsize=10, fontweight='bold')
    
    ax = axes[1, 0]
    ax.imshow(img2_clean)
    ax.set_title("Clean Image (Class: Truck)", fontsize=12)
    ax.axis('off')
    
    ax = axes[1, 1]
    ax.imshow(img2_triggered)
    ax.set_title("Triggered Image (Target: Plane)", fontsize=12)
    ax.axis('off')
    # Outline the trigger
    ax.add_patch(patches.Rectangle((65, 65), 30, 30, linewidth=2, edgecolor='red', facecolor='none'))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_visual_examples.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

##############################################################################
# Figure 4: Defense Mechanism Flowchart
##############################################################################
def draw_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    def add_box(ax, x, y, width, height, text, bg_color='#e6f2ff', border_color='blue', text_color='black'):
        # Using Rectangle instead of FancyBboxPatch for safety to avoid matplotlib version issues
        ax.add_patch(patches.Rectangle((x - width/2, y - height/2), width, height, 
                                       facecolor=bg_color, edgecolor=border_color, lw=2))
        ax.text(x, y, text, ha='center', va='center', fontsize=11, fontweight='bold', color=text_color)
        
    def add_arrow(ax, x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))

    # Boxes
    add_box(ax, 5, 9, 3, 1, "1. Input Sample (x)", '#d9ead3', 'green')
    add_box(ax, 5, 7.5, 4, 1, "2. Sequential Block Checksum\n(Verify Stem & Groups)", '#e6f2ff', 'blue')
    add_box(ax, 2.5, 6, 3.5, 1, "3a. No Poison Detected\nSafe Pool = All 16 ICs", '#d9ead3', 'green')
    add_box(ax, 7.5, 6, 3.5, 1, "3b. Poison Detected at Block k\nTruncate Safe Pool (ICs < k)", '#ffcccc', 'red')
    
    add_box(ax, 5, 4.5, 4, 1, "4. Stochastic Branch Selection\n(Sample q random exits from Safe Pool)", '#fff2cc', 'orange')
    add_box(ax, 5, 3, 4, 1, "5. Multi-Branch Logit Generation\n(Forward pass on selected ICs)", '#e6f2ff', 'blue')
    add_box(ax, 5, 1.5, 4, 1, "6. Confidence-Weighted Consensus\n(Soft-Voting)", '#e6f2ff', 'blue')
    add_box(ax, 5, 0, 3, 0.8, "7. Final Prediction Output", '#d9ead3', 'green')

    # Arrows
    add_arrow(ax, 5, 8.5, 5, 8)
    
    # Split
    ax.annotate('', xy=(2.5, 6.5), xytext=(5, 7), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    ax.annotate('', xy=(7.5, 6.5), xytext=(5, 7), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Merge
    ax.annotate('', xy=(5, 5), xytext=(2.5, 5.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    ax.annotate('', xy=(5, 5), xytext=(7.5, 5.5), arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8))
    
    # Downward flow
    add_arrow(ax, 5, 4, 5, 3.5)
    add_arrow(ax, 5, 2.5, 5, 2)
    add_arrow(ax, 5, 1, 5, 0.4)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_flowchart_defense.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating Figure 3 (Visual Examples)...")
    draw_visual_examples()
    print("Generating Figure 4 (Defense Flowchart)...")
    draw_flowchart()
    print("New figures generated successfully.")
