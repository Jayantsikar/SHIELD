import matplotlib.pyplot as plt
import numpy as np
import os

# Publication-grade settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'sans-serif',
    'axes.labelsize': 10.5,
    'axes.titlesize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9.8
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 4.2), dpi=300)

labels = ['Non-Adaptive TBT', 'Adaptive TBT']
x = np.arange(len(labels))
width = 0.22

# CIFAR-10 evaluation data
cda_base = [88.69, 83.41]
cda_l1 = [90.01, 87.39]
cda_l2 = [89.93, 83.02]

asr_base = [17.82, 38.50]
asr_l1 = [15.48, 40.72]
asr_l2 = [11.67, 15.09]

# Subplot 1: Clean Data Accuracy (CDA)
b1 = ax1.bar(x - width - 0.015, cda_base, width, label='Aegis Baseline', color='#64748B', edgecolor='#334155', linewidth=0.9, hatch='//', zorder=3)
b2 = ax1.bar(x, cda_l1, width, label='L1: SBE (Undefended)', color='#EA580C', edgecolor='#9A3412', linewidth=0.9, hatch='\\\\', zorder=3)
b3 = ax1.bar(x + width + 0.015, cda_l2, width, label='L2: Self-Heal (Ours)', color='#0D9488', edgecolor='#115E59', linewidth=1.1, zorder=3)

for b, is_l2 in [(b1, False), (b2, False), (b3, True)]:
    for rect in b:
        h = rect.get_height()
        fw = 'bold' if is_l2 else 'semibold'
        color = '#0F766E' if is_l2 else '#1E293B'
        ax1.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                     fontsize=9.0, fontweight=fw, color=color)

ax1.set_title('(a) Clean Data Accuracy (CDA) ↑', fontweight='bold', pad=10, color='#1E293B')
ax1.set_ylabel('Accuracy (%)', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontweight='semibold')
ax1.set_ylim(70, 96)
ax1.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax1.set_axisbelow(True)

# Subplot 2: Attack Success Rate (ASR)
b4 = ax2.bar(x - width - 0.015, asr_base, width, label='Aegis Baseline', color='#64748B', edgecolor='#334155', linewidth=0.9, hatch='//', zorder=3)
b5 = ax2.bar(x, asr_l1, width, label='L1: SBE (Undefended)', color='#EA580C', edgecolor='#9A3412', linewidth=0.9, hatch='\\\\', zorder=3)
b6 = ax2.bar(x + width + 0.015, asr_l2, width, label='L2: Self-Heal (Ours)', color='#0D9488', edgecolor='#115E59', linewidth=1.1, zorder=3)

for b, is_l2 in [(b4, False), (b5, False), (b6, True)]:
    for rect in b:
        h = rect.get_height()
        fw = 'bold' if is_l2 else 'semibold'
        color = '#0F766E' if is_l2 else '#1E293B'
        ax2.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width()/2, h),
                     xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                     fontsize=9.0, fontweight=fw, color=color)

# Random Guess Reference Line
line_rand = ax2.axhline(10.0, color='#DC2626', linestyle='--', linewidth=1.3, alpha=0.85, zorder=2, label='Random Bound (10%)')

# Annotation on reduction
ax2.annotate('−60.8% ASR\nvs. Aegis', xy=(1 + width + 0.015, 15.09), xytext=(1 + width + 0.015 - 0.04, 28),
             arrowprops=dict(arrowstyle='->', lw=1.2, color='#0F766E', connectionstyle='arc3,rad=-0.15'),
             fontsize=8.8, fontweight='bold', color='#065F46', ha='center',
             bbox=dict(boxstyle='round,pad=0.25', facecolor='#D1FAE5', edgecolor='#10B981', lw=0.9))

ax2.set_title('(b) Attack Success Rate (ASR) ↓', fontweight='bold', pad=10, color='#1E293B')
ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(labels, fontweight='semibold')
ax2.set_ylim(0, 48)
ax2.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
ax2.set_axisbelow(True)

# Shared legend at top
handles1, leg_labels1 = ax1.get_legend_handles_labels()
all_handles = handles1 + [line_rand]
all_labels = leg_labels1 + ['Random Bound (10%)']

fig.legend(all_handles, all_labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=4, frameon=True, edgecolor='#CBD5E1', fancybox=True)

plt.tight_layout(rect=[0, 0, 1, 0.94])

os.makedirs('figures', exist_ok=True)
pdf_path = 'figures/fig_results_cifar10.pdf'
png_path = 'figures/fig_results_cifar10.png'
plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
print(f'Successfully saved {pdf_path} and {png_path}')
