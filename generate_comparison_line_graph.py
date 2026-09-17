import matplotlib.pyplot as plt
import numpy as np

# IEEE style configuration
plt.rcParams.update({
    'font.size': 8.5,
    'font.family': 'sans-serif',
    'axes.labelsize': 9.2,
    'axes.titlesize': 9.8,
    'xtick.labelsize': 8.0,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 8.2,
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

fig, ax = plt.subplots(figsize=(6.8, 3.2), constrained_layout=True)

c_non = '#1B4F72'   # Navy Blue
c_ad  = '#C0392B'   # Crimson Red
c_rnd = '#555555'   # Charcoal

# Highlight our defense (L2: Self-Heal) with a subtle shaded vertical band
ax.axvspan(5.5, 6.5, color='#E8F8F5', alpha=0.9, zorder=0)

# Non-Adaptive Line
line1, = ax.plot(x, asr_non_adapt, marker='o', markersize=6, linewidth=1.8,
                 color=c_non, label='Non-Adaptive TBT ASR (%)', zorder=4)

# Dashed bridge from BASE to SDN indicating omitted NA defenses
ax.plot([0, 3], [70.70, 37.20], linestyle=':', color=c_ad, alpha=0.45, linewidth=1.3, zorder=3)

# Adaptive Line
line2, = ax.plot(x_adapt_valid[1:], y_adapt_valid[1:], marker='s', markersize=6, linewidth=1.8,
                 color=c_ad, label='Adaptive TBT ASR (%)', zorder=4)
ax.scatter([0], [70.70], color=c_ad, marker='s', s=36, zorder=5)

# Annotate NA for BIN and RA-BNN
ax.annotate('NA*', xy=(1, 22), ha='center', va='center', fontsize=7.5, color='#7f8c8d', style='italic')
ax.annotate('NA*', xy=(2, 22), ha='center', va='center', fontsize=7.5, color='#7f8c8d', style='italic')

# Random Guess Reference Line
ax.axhline(10.0, color=c_rnd, linestyle='--', linewidth=0.9, alpha=0.8)
ax.text(0.08, 11.8, 'Random Guess Baseline (10%)', fontsize=7.0, color='#444444', style='italic')

# Non-adaptive value labels
for i, val in enumerate(asr_non_adapt):
    if i == 6:
        ax.annotate(f'{val:.1f}%', (x[i], val), xytext=(0, -11), textcoords='offset points',
                    ha='center', fontsize=7.2, fontweight='bold', color=c_non)
    elif i == 5:
        ax.annotate(f'{val:.1f}%', (x[i], val), xytext=(0, -9), textcoords='offset points',
                    ha='center', fontsize=7.2, color=c_non)
    elif i == 3:
        ax.annotate(f'{val:.1f}%', (x[i], val), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.2, color=c_non)
    else:
        ax.annotate(f'{val:.1f}%', (x[i], val), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.2, color=c_non)

# Adaptive value labels
for xi, val in zip(x_adapt_valid, y_adapt_valid):
    if xi == 6:
        ax.annotate(f'{val:.1f}%', (xi, val), xytext=(0, 6), textcoords='offset points',
                    ha='center', fontsize=7.2, fontweight='bold', color=c_ad)
    elif xi == 0:
        ax.annotate(f'{val:.1f}%', (xi, val), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.2, color=c_ad)
    else:
        ax.annotate(f'{val:.1f}%', (xi, val), xytext=(0, 5), textcoords='offset points',
                    ha='center', fontsize=7.2, color=c_ad)

ax.set_xticks(x)
ax.set_xticklabels(models, fontweight='medium')
ax.set_ylabel(r'Attack Success Rate (ASR %) $\downarrow$', fontweight='bold')
ax.set_title('TBT Attack Success Rate (ASR) Across Defense Models on CIFAR-10 (ResNet-32)', fontweight='bold', pad=10)
ax.set_ylim(0, 105)
ax.grid(True, linestyle='--', alpha=0.45)

# Legend
handles = [line1, line2]
labels = ['Non-Adaptive TBT', 'Adaptive TBT']
ax.legend(handles=handles, labels=labels, loc='upper right', frameon=True, framealpha=0.95, edgecolor='#cccccc')

plt.savefig('figures/defense_asr_comparison_line.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figures/defense_asr_comparison_line.png', format='png', dpi=300, bbox_inches='tight')
print("Polished line graph saved to figures/defense_asr_comparison_line.pdf and .png")
