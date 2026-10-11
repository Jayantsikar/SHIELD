import matplotlib.pyplot as plt
import numpy as np

# IEEE style configuration
plt.rcParams.update({
    'font.size': 10.5,
    'font.family': 'sans-serif',
    'axes.labelsize': 11.5,
    'axes.titlesize': 11.0,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10.5,
    'grid.alpha': 0.4,
    'grid.linestyle': ':',
})

# Data from Table: Evaluation results of ASR against TBT on CIFAR-10 (ResNet-32)
models = ['BASE', 'BIN', 'RA-BNN', 'SDN', 'Aegis', 'L1: SBE\n(Undefended)', 'L2: Self-Heal\n(Ours)']
x = np.arange(len(models))

# Non-Adaptive ASR (%)
asr_non_adapt = [70.70, 94.80, 74.50, 16.30, 19.90, 15.48, 11.67]

# Adaptive ASR (%) - BIN and RA-BNN are NA (not applicable)
x_adapt_valid = [0, 3, 4, 5, 6]
y_adapt_valid = [70.70, 37.20, 31.10, 40.72, 15.09]

fig, ax = plt.subplots(figsize=(8.2, 3.8), constrained_layout=True)

c_non = '#1B4F72'   # Navy Blue
c_ad  = '#C0392B'   # Crimson Red
c_rnd = '#555555'   # Charcoal

# Highlight our defense (L2: Self-Heal) with a subtle shaded vertical band
ax.axvspan(5.45, 6.45, color='#E8F8F5', alpha=0.9, zorder=0)

# Non-Adaptive Line
line1, = ax.plot(x, asr_non_adapt, marker='o', markersize=7.5, linewidth=2.4,
                 color=c_non, label='Non-Adaptive TBT ASR (%)', zorder=4)

# Dashed bridge from BASE to SDN indicating omitted NA defenses
ax.plot([0, 3], [70.70, 37.20], linestyle=':', color=c_ad, alpha=0.5, linewidth=1.6, zorder=3)

# Adaptive Line
line2, = ax.plot(x_adapt_valid[1:], y_adapt_valid[1:], marker='s', markersize=7.5, linewidth=2.4,
                 color=c_ad, label='Adaptive TBT ASR (%)', zorder=4)
ax.scatter([0], [70.70], color=c_ad, marker='s', s=56, zorder=5)

# Annotate NA for BIN and RA-BNN
ax.annotate('NA*', xy=(1, 23), ha='center', va='center', fontsize=11.0, color='#4A5568', fontweight='bold', style='italic')
ax.annotate('NA*', xy=(2, 23), ha='center', va='center', fontsize=11.0, color='#4A5568', fontweight='bold', style='italic')

# Random Guess Reference Line
ax.axhline(10.0, color=c_rnd, linestyle='--', linewidth=1.2, alpha=0.8)
ax.text(0.08, 12.2, 'Random Guess Baseline (10%)', fontsize=10.0, color='#333333', fontweight='semibold', style='italic')

# Base label (identical for both non-adaptive and adaptive)
ax.annotate('70.7%', (0, 70.70), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color='#1E293B')

# Non-adaptive value labels
ax.annotate('94.8%', (1, asr_non_adapt[1]), xytext=(0, 7), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_non)
ax.annotate('74.5%', (2, asr_non_adapt[2]), xytext=(0, 7), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_non)
ax.annotate('16.3%', (3, asr_non_adapt[3]), xytext=(8, 7), textcoords='offset points',
            ha='left', fontsize=10.5, fontweight='bold', color=c_non)
ax.annotate('19.9%', (4, asr_non_adapt[4]), xytext=(0, -16), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_non)
ax.annotate('15.5%', (5, asr_non_adapt[5]), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_non)
ax.annotate('11.7%', (6, asr_non_adapt[6]), xytext=(0, -16), textcoords='offset points',
            ha='center', fontsize=11.0, fontweight='bold', color=c_non)

# Adaptive value labels
ax.annotate('37.2%', (3, 37.20), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_ad)
ax.annotate('31.1%', (4, 31.10), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_ad)
ax.annotate('40.7%', (5, 40.72), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=10.5, fontweight='bold', color=c_ad)
ax.annotate('15.1%', (6, 15.09), xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=11.0, fontweight='bold', color=c_ad)

ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='semibold')
ax.set_xlim(-0.45, 6.45)
ax.set_ylabel(r'Attack Success Rate (ASR %) $\downarrow$', fontweight='bold')
ax.set_title('TBT Attack Success Rate (ASR) Across Defense Models on CIFAR-10 (ResNet-32)', fontweight='bold', pad=10)
ax.set_ylim(0, 110)
ax.grid(True, linestyle='--', alpha=0.45)

# Legend
handles = [line1, line2]
labels = ['Non-Adaptive TBT', 'Adaptive TBT']
ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc')

plt.savefig('figures/defense_asr_comparison_line.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figures/defense_asr_comparison_line.png', format='png', dpi=300, bbox_inches='tight')
print("Polished line graph saved to figures/defense_asr_comparison_line.pdf and .png")
