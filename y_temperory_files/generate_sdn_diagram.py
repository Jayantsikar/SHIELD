import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

out_dir = '/Users/jayantkumar/Desktop/Aegis/figures'
os.makedirs(out_dir, exist_ok=True)

def draw_sdn_diagram():
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.axis('off')
    ax.set_xlim(0, 21)
    ax.set_ylim(0, 6)

    # Colors
    color_input = '#b6d7a8' # light green
    color_layer = '#cfe2f3' # light blue
    color_ic = '#cfe2f3'    # light blue
    color_inner = '#ffe599' # light yellow

    def draw_box(x, y, w, h, text, bg_color, fontsize=12):
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor=bg_color, edgecolor='black', lw=1.5))
        if text:
            # Handle multi-line
            lines = text.split('\n')
            if len(lines) == 1:
                ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fontsize, zorder=3)
            else:
                ax.text(x + w/2, y + h*0.65, lines[0], ha='center', va='center', fontsize=fontsize, zorder=3)
                ax.text(x + w/2, y + h*0.35, lines[1], ha='center', va='center', fontsize=fontsize, zorder=3)

    def draw_arrow(x1, y1, x2, y2, text='', text_offset=(0, 0.2)):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1), 
                    arrowprops=dict(facecolor='black', shrink=0, width=1.5, headwidth=6, headlength=8))
        if text:
            ax.text((x1+x2)/2 + text_offset[0], (y1+y2)/2 + text_offset[1], text, ha='center', va='center', fontsize=12)

    # Coordinates
    y_top = 4.5
    y_mid = 2.2
    y_bot = 0.5
    w_layer = 1.6
    h_layer = 1.0
    w_input = 1.4
    h_input = 0.6
    
    # X offsets
    x_in = 0.5
    x_l1 = x_in + w_input + 0.8
    mid_x1 = x_l1 + w_layer + 1.3
    x_l2 = mid_x1 + 1.2
    mid_x2 = x_l2 + w_layer + 1.3
    dots_x = mid_x2 + 1.5
    x_ln = dots_x + 1.5
    mid_xn = x_ln + w_layer + 1.3
    x_lf = mid_xn + 1.2
    mid_xf = x_lf + w_layer + 1.5

    # Draw top row boxes
    draw_box(x_in, y_top + 0.2, w_input, h_input, "Input", color_input)
    draw_box(x_l1, y_top, w_layer, h_layer, "$\mathbf{F_1}$\nLayer 1", color_layer, fontsize=14)
    draw_box(x_l2, y_top, w_layer, h_layer, "$\mathbf{F_2}$\nLayer 2", color_layer, fontsize=14)
    draw_box(x_ln, y_top, w_layer, h_layer, "$\mathbf{F_N}$\nLayer N", color_layer, fontsize=14)
    draw_box(x_lf, y_top, w_layer + 0.4, h_layer, "$\mathbf{F_{final}}$\nFinal Layer", color_layer, fontsize=14)
    
    # Dots between Layer 2 and Layer N
    ax.text(dots_x, y_top + 0.5, "$\dots$", ha='center', va='center', fontsize=20)

    # Draw bottom row IC blocks
    def draw_ic(cx, idx):
        # Outer IC box
        ic_w = 1.4
        ic_h = 2.0
        ic_x = cx - ic_w/2
        draw_box(ic_x, y_mid, ic_w, ic_h, "", color_ic)
        ax.text(cx, y_mid + ic_h - 0.3, f"$\mathbf{{C_{idx}}}$", ha='center', va='center', fontsize=14)
        
        # Inner boxes
        inner_w = 1.1
        inner_h = 0.5
        inner_x = cx - inner_w/2
        draw_box(inner_x, y_mid + 0.9, inner_w, inner_h, f"$\mathbf{{Conv_{idx}}}$", color_inner)
        draw_box(inner_x, y_mid + 0.2, inner_w, inner_h, f"$\mathbf{{Dense_{idx}}}$", color_inner)
        
        # Output box
        out_w = 1.3
        out_h = 0.6
        out_x = cx - out_w/2
        draw_box(out_x, y_bot, out_w, out_h, f"$\mathbf{{C_{idx}(x)}}$", color_input)
        
        # Arrow down to output
        draw_arrow(cx, y_mid, cx, y_bot + out_h)

    draw_ic(mid_x1, "1")
    draw_ic(mid_x2, "2")
    draw_ic(mid_xn, "N")

    # Output box for final
    draw_box(mid_xf - w_input/2, y_bot, w_input, h_input, "Output", color_input)

    # Connect top row
    # Input to F1
    y_center = y_top + h_layer/2
    draw_arrow(x_in + w_input, y_center, x_l1, y_center, text="x", text_offset=(0, 0.25))

    # F1 to F2 and down to C1
    ax.add_line(plt.Line2D((x_l1 + w_layer, mid_x1), (y_center, y_center), color='black', lw=2))
    ax.text((x_l1 + w_layer + mid_x1)/2, y_center + 0.25, "$\mathbf{F_1(x)}$", ha='center', fontsize=12)
    draw_arrow(mid_x1, y_center, x_l2, y_center)
    draw_arrow(mid_x1, y_center, mid_x1, y_mid + 2.0)

    # F2 to dots and down to C2
    ax.add_line(plt.Line2D((x_l2 + w_layer, mid_x2), (y_center, y_center), color='black', lw=2))
    ax.text((x_l2 + w_layer + mid_x2)/2, y_center + 0.25, "$\mathbf{F_2(x)}$", ha='center', fontsize=12)
    draw_arrow(mid_x2, y_center, dots_x - 0.5, y_center)
    draw_arrow(mid_x2, y_center, mid_x2, y_mid + 2.0)

    # Dots to FN
    draw_arrow(dots_x + 0.5, y_center, x_ln, y_center)

    # FN to F_final and down to CN
    ax.add_line(plt.Line2D((x_ln + w_layer, mid_xn), (y_center, y_center), color='black', lw=2))
    ax.text((x_ln + w_layer + mid_xn)/2, y_center + 0.25, "$\mathbf{F_N(x)}$", ha='center', fontsize=12)
    draw_arrow(mid_xn, y_center, x_lf, y_center)
    draw_arrow(mid_xn, y_center, mid_xn, y_mid + 2.0)

    # F_final to Output
    ax.add_line(plt.Line2D((x_lf + w_layer + 0.4, mid_xf), (y_center, y_center), color='black', lw=2))
    ax.text((x_lf + w_layer + 0.4 + mid_xf)/2, y_center + 0.25, "$\mathbf{F_{final}(x)}$", ha='center', fontsize=12)
    draw_arrow(mid_xf, y_center, mid_xf, y_bot + h_input)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig_sdn_architecture.pdf'), format='pdf', bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    print("Generating SDN Architecture Diagram...")
    draw_sdn_diagram()
    print("SDN Architecture Diagram generated successfully.")
