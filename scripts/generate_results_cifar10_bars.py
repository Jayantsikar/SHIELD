import matplotlib.pyplot as plt
import numpy as np
import os

# Create output figures directory if it doesn't exist
os.makedirs('figures', exist_ok=True)

# Publication styling tokens
c_base = '#64748B'    # Aegis Baseline (Slate Gray)
e_base = '#334155'
c_l1   = '#EA580C'    # L1: SBE Undefended (Vibrant Orange)
e_l1   = '#9A3412'
c_l2   = '#0D9488'    # L2: Self-Heal Ours (Deep Teal)
e_l2   = '#115E59'
c_rand = '#DC2626'    # Random Bound (10%) (Crimson Red)

labels = ['Non-Adaptive TBT', 'Adaptive TBT']
x = np.array([0.0, 1.45])
width = 0.22
offsets = np.array([-0.30, 0.0, 0.30])

# CIFAR-10 evaluation data
cda_base = [88.69, 83.41]
cda_l1 = [90.01, 87.39]
cda_l2 = [89.93, 83.02]

asr_base = [17.82, 38.50]
asr_l1 = [15.48, 40.72]
asr_l2 = [11.67, 15.09]

# Text nudges for perfect typography without collision
cda_nudges = [
    [(-4, 4), (-4, 4), (5, 4)],   # Non-Adaptive: nudge Aegis & L1 left, L2 right
    [(-3, 4), (0, 4), (3, 4)]     # Adaptive: L1 is peak (87.4%), others well separated
]

asr_nudges = [
    [(-5, 4), (0, 4), (5, 4)],    # Non-Adaptive: nudge Aegis left, L2 right
    [(-6, 4), (4, 4), (0, 4)]     # Adaptive: separate Aegis (38.5) and L1 (40.7)
]


# ==============================================================================
# 1. Combined Figure: (a) Clean Accuracy and (b) Attack Success Rate
# ==============================================================================
plt.rcParams.update({
    'font.size': 11.0,
    'font.family': 'sans-serif',
    'axes.labelsize': 13.0,
    'axes.titlesize': 13.5,
    'xtick.labelsize': 12.0,
    'ytick.labelsize': 11.5,
    'legend.fontsize': 11.5
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.0, 4.6), dpi=300)

# --- Subplot 1: Clean Data Accuracy (CDA) ---
b1 = ax1.bar(x + offsets[0], cda_base, width, label='Aegis Baseline', color=c_base, edgecolor=e_base, linewidth=1.0, hatch='//', zorder=3)
b2 = ax1.bar(x + offsets[1], cda_l1, width, label='L1: SBE (Undefended)', color=c_l1, edgecolor=e_l1, linewidth=1.0, hatch='\\\\', zorder=3)
b3 = ax1.bar(x + offsets[2], cda_l2, width, label='L2: Self-Heal (Ours)', color=c_l2, edgecolor=e_l2, linewidth=1.2, zorder=3)

for group_idx in range(len(x)):
    for bar_idx, (b, is_l2) in enumerate([(b1, False), (b2, False), (b3, True)]):
        rect = b[group_idx]
        h = rect.get_height()
        fw = 'bold'
        color = '#0F766E' if is_l2 else '#0F172A'
        nudge = cda_nudges[group_idx][bar_idx]
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=nudge, textcoords='offset points', ha='center', va='bottom',
                     fontsize=10.5, fontweight=fw, color=color)

ax1.set_title('(a) Clean Data Accuracy (CDA) ↑', fontweight='bold', pad=12, color='#1E293B')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='bold')
ax1.set_xlim(-0.65, 2.10)
ax1.set_ylim(70, 97)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

# --- Subplot 2: Attack Success Rate (ASR) ---
b4 = ax2.bar(x + offsets[0], asr_base, width, label='Aegis Baseline', color=c_base, edgecolor=e_base, linewidth=1.0, hatch='//', zorder=3)
b5 = ax2.bar(x + offsets[1], asr_l1, width, label='L1: SBE (Undefended)', color=c_l1, edgecolor=e_l1, linewidth=1.0, hatch='\\\\', zorder=3)
b6 = ax2.bar(x + offsets[2], asr_l2, width, label='L2: Self-Heal (Ours)', color=c_l2, edgecolor=e_l2, linewidth=1.2, zorder=3)

for group_idx in range(len(x)):
    for bar_idx, (b, is_l2) in enumerate([(b4, False), (b5, False), (b6, True)]):
        rect = b[group_idx]
        h = rect.get_height()
        fw = 'bold'
        color = '#0F766E' if is_l2 else '#0F172A'
        nudge = asr_nudges[group_idx][bar_idx]
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=nudge, textcoords='offset points', ha='center', va='bottom',
                     fontsize=10.5, fontweight=fw, color=color)

# Random Guess Reference Line
line_rand = ax2.axhline(10.0, color=c_rand, linestyle='--', linewidth=1.4, alpha=0.9, zorder=2, label='Random Bound (10%)')

# Annotation on reduction - positioned cleanly with arrow pointing to upper-right shoulder of L2 bar
x_l2_adapt = x[1] + offsets[2]
ax2.annotate('−60.8% ASR\nvs. Aegis', xy=(x_l2_adapt + 0.11, 13.5), xytext=(x_l2_adapt, 32.5),
             arrowprops=dict(arrowstyle='->', lw=1.3, color='#0F766E', connectionstyle='arc3,rad=-0.18'),
             fontsize=10.5, fontweight='bold', color='#065F46', ha='center',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#10B981', lw=1.0))

ax2.set_title('(b) Attack Success Rate (ASR) ↓', fontweight='bold', pad=12, color='#1E293B')
ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontweight='bold')
ax2.set_xlim(-0.65, 2.10)
ax2.set_ylim(0, 50)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

# Shared legend at top
handles1, leg_labels1 = ax1.get_legend_handles_labels()
all_handles = handles1 + [line_rand]
all_labels = leg_labels1 + ['Random Bound (10%)']

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03),
           ncol=4, frameon=True, edgecolor='#CBD5E1', fancybox=True, fontsize=11.5)

plt.tight_layout(rect=[0, 0, 1, 0.92])

pdf_path = 'figures/fig_results_cifar10.pdf'
png_path = 'figures/fig_results_cifar10.png'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
plt.close(fig)
print(f'Successfully saved {pdf_path} and {png_path}')


# ==============================================================================
# 2. Standalone Figure A: Clean Data Accuracy (CDA)
# ==============================================================================
fig_cda, ax_cda = plt.subplots(figsize=(5.4, 4.2), dpi=300)

b1 = ax_cda.bar(x + offsets[0], cda_base, width, label='Aegis Baseline', color=c_base, edgecolor=e_base, linewidth=1.0, hatch='//', zorder=3)
b2 = ax_cda.bar(x + offsets[1], cda_l1, width, label='L1: SBE (Undefended)', color=c_l1, edgecolor=e_l1, linewidth=1.0, hatch='\\\\', zorder=3)
b3 = ax_cda.bar(x + offsets[2], cda_l2, width, label='L2: Self-Heal (Ours)', color=c_l2, edgecolor=e_l2, linewidth=1.2, zorder=3)

for group_idx in range(len(x)):
    for bar_idx, (b, is_l2) in enumerate([(b1, False), (b2, False), (b3, True)]):
        rect = b[group_idx]
        h = rect.get_height()
        fw = 'bold'
        color = '#0F766E' if is_l2 else '#0F172A'
        nudge = cda_nudges[group_idx][bar_idx]
        ax_cda.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=nudge, textcoords='offset points', ha='center', va='bottom',
                        fontsize=10.5, fontweight=fw, color=color)

ax_cda.set_title('Clean Data Accuracy (CDA) ↑', fontweight='bold', fontsize=13.0, pad=12, color='#1E293B')
ax_cda.set_ylabel('Accuracy (%)', fontweight='bold', fontsize=12.5)
ax_cda.set_xticks(x)
ax_cda.set_xticklabels(labels, fontweight='bold', fontsize=11.5)
ax_cda.set_xlim(-0.65, 2.10)
ax_cda.set_ylim(70, 97)
ax_cda.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax_cda.set_axisbelow(True)

handles_cda, labels_cda = ax_cda.get_legend_handles_labels()
fig_cda.legend(handles_cda, labels_cda, loc='upper center', bbox_to_anchor=(0.54, 0.98), ncol=3,
               frameon=True, edgecolor='#CBD5E1', fancybox=True, fontsize=9.5, framealpha=0.95)

plt.tight_layout(rect=[0, 0, 1, 0.90])
pdf_cda_path = 'figures/fig_results_cifar10_cda.pdf'
png_cda_path = 'figures/fig_results_cifar10_cda.png'
plt.savefig(pdf_cda_path, format='pdf', bbox_inches='tight')
plt.savefig(png_cda_path, format='png', dpi=300, bbox_inches='tight')
plt.close(fig_cda)
print(f'Successfully saved {pdf_cda_path} and {png_cda_path}')


# ==============================================================================
# 3. Standalone Figure B: Attack Success Rate (ASR)
# ==============================================================================
fig_asr, ax_asr = plt.subplots(figsize=(5.4, 4.2), dpi=300)

b4 = ax_asr.bar(x + offsets[0], asr_base, width, label='Aegis Baseline', color=c_base, edgecolor=e_base, linewidth=1.0, hatch='//', zorder=3)
b5 = ax_asr.bar(x + offsets[1], asr_l1, width, label='L1: SBE (Undefended)', color=c_l1, edgecolor=e_l1, linewidth=1.0, hatch='\\\\', zorder=3)
b6 = ax_asr.bar(x + offsets[2], asr_l2, width, label='L2: Self-Heal (Ours)', color=c_l2, edgecolor=e_l2, linewidth=1.2, zorder=3)

for group_idx in range(len(x)):
    for bar_idx, (b, is_l2) in enumerate([(b4, False), (b5, False), (b6, True)]):
        rect = b[group_idx]
        h = rect.get_height()
        fw = 'bold'
        color = '#0F766E' if is_l2 else '#0F172A'
        nudge = asr_nudges[group_idx][bar_idx]
        ax_asr.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                        xytext=nudge, textcoords='offset points', ha='center', va='bottom',
                        fontsize=10.5, fontweight=fw, color=color)

# Random Guess Reference Line
line_rand_single = ax_asr.axhline(10.0, color=c_rand, linestyle='--', linewidth=1.4, alpha=0.9, zorder=2, label='Random Bound (10%)')

# Annotation on reduction
ax_asr.annotate('−60.8% ASR\nvs. Aegis', xy=(x_l2_adapt + 0.11, 13.5), xytext=(x_l2_adapt, 32.5),
                arrowprops=dict(arrowstyle='->', lw=1.3, color='#0F766E', connectionstyle='arc3,rad=-0.18'),
                fontsize=10.5, fontweight='bold', color='#065F46', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#10B981', lw=1.0))

ax_asr.set_title('Attack Success Rate (ASR) ↓', fontweight='bold', fontsize=13.0, pad=12, color='#1E293B')
ax_asr.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12.5)
ax_asr.set_xticks(x)
ax_asr.set_xticklabels(labels, fontweight='bold', fontsize=11.5)
ax_asr.set_xlim(-0.65, 2.10)
ax_asr.set_ylim(0, 52)
ax_asr.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax_asr.set_axisbelow(True)

handles_asr, labels_asr = ax_asr.get_legend_handles_labels()
ax_asr.legend(handles_asr, labels_asr, loc='upper left', frameon=True, edgecolor='#CBD5E1', fancybox=True, fontsize=9.8, framealpha=0.95)

plt.tight_layout()
pdf_asr_path = 'figures/fig_results_cifar10_asr.pdf'
png_asr_path = 'figures/fig_results_cifar10_asr.png'
plt.savefig(pdf_asr_path, format='pdf', bbox_inches='tight')
plt.savefig(png_asr_path, format='png', dpi=300, bbox_inches='tight')
plt.close(fig_asr)
print(f'Successfully saved {pdf_asr_path} and {png_asr_path}')
