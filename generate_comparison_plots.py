import matplotlib.pyplot as plt
import numpy as np

# IEEE conference styling
plt.rcParams.update({
    'font.size': 8.5,
    'font.family': 'sans-serif',
    'axes.labelsize': 9.0,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.0,
    'legend.fontsize': 8.2,
    'grid.alpha': 0.4,
    'grid.linestyle': ':',
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.85))

# Coordinates with distinct separation between datasets
x_cifar = np.array([0, 1])
x_stl = np.array([2.2, 3.2])
all_ticks = [0, 1, 2.2, 3.2]
all_labels = ['CIFAR-10\nNon-Adapt.', 'CIFAR-10\nAdaptive', 'STL-10\nNon-Adapt.', 'STL-10\nAdaptive']

# Data
asr_base_cifar = [17.82, 38.50]
asr_l1_cifar   = [15.31, 40.41]
asr_l2_cifar   = [12.62, 9.79]

asr_base_stl   = [13.00, 31.00]
asr_l1_stl     = [12.21, 28.25]
asr_l2_stl     = [10.21, 13.91]

cacc_base_cifar = [88.69, 83.41]
cacc_l1_cifar   = [90.09, 87.36]
cacc_l2_cifar   = [90.30, 90.66]

cacc_l1_stl     = [73.89, 73.70]
cacc_l2_stl     = [74.11, 74.19]

# IEEE Color Palette
c_base = '#C0392B'   # Crimson Red (Baseline)
c_l1 = '#D35400'     # Amber / Ochre (L1)
c_l2 = '#1B4F72'     # Deep Navy (Ours)

# ----------------- Subplot 1: Attack Success Rate (ASR) -----------------
# CIFAR-10 segments
ax1.plot(x_cifar, asr_base_cifar, 's--', color=c_base, label='Baseline Aegis', linewidth=1.8, markersize=6.5, markerfacecolor='white', markeredgewidth=1.8)
ax1.plot(x_cifar, asr_l1_cifar, '^-.', color=c_l1, label='L1: Raw SBE', linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8)
ax1.plot(x_cifar, asr_l2_cifar, 'o-', color=c_l2, label='L2: Self-Heal (Ours)', linewidth=2.4, markersize=7.5, markerfacecolor=c_l2, markeredgewidth=1.5)

# STL-10 segments
ax1.plot(x_stl, asr_base_stl, 's--', color=c_base, linewidth=1.8, markersize=6.5, markerfacecolor='white', markeredgewidth=1.8)
ax1.plot(x_stl, asr_l1_stl, '^-.', color=c_l1, linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8)
ax1.plot(x_stl, asr_l2_stl, 'o-', color=c_l2, linewidth=2.4, markersize=7.5, markerfacecolor=c_l2, markeredgewidth=1.5)

# Annotations ASR
ax1.annotate('17.8%', (x_cifar[0], asr_base_cifar[0]), textcoords="offset points", xytext=(-18, 5), fontsize=7.2, color=c_base, fontweight='semibold')
ax1.annotate('38.5%', (x_cifar[1], asr_base_cifar[1]), textcoords="offset points", xytext=(-18, -13), fontsize=7.2, color=c_base, fontweight='semibold')

ax1.annotate('15.3%', (x_cifar[0], asr_l1_cifar[0]), textcoords="offset points", xytext=(8, -5), fontsize=7.2, color=c_l1, fontweight='semibold')
ax1.annotate('40.4%', (x_cifar[1], asr_l1_cifar[1]), textcoords="offset points", xytext=(8, 4), fontsize=7.2, color=c_l1, fontweight='semibold')

ax1.annotate('12.6%', (x_cifar[0], asr_l2_cifar[0]), textcoords="offset points", xytext=(-18, -12), fontsize=7.2, color=c_l2, fontweight='bold')
ax1.annotate('9.8%', (x_cifar[1], asr_l2_cifar[1]), textcoords="offset points", xytext=(0, -13), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')

ax1.annotate('13.0%', (x_stl[0], asr_base_stl[0]), textcoords="offset points", xytext=(-18, 5), fontsize=7.2, color=c_base, fontweight='semibold')
ax1.annotate('31.0%', (x_stl[1], asr_base_stl[1]), textcoords="offset points", xytext=(-18, 6), fontsize=7.2, color=c_base, fontweight='semibold')

ax1.annotate('12.2%', (x_stl[0], asr_l1_stl[0]), textcoords="offset points", xytext=(8, -8), fontsize=7.2, color=c_l1, fontweight='semibold')
ax1.annotate('28.2%', (x_stl[1], asr_l1_stl[1]), textcoords="offset points", xytext=(8, -8), fontsize=7.2, color=c_l1, fontweight='semibold')

ax1.annotate('10.2%', (x_stl[0], asr_l2_stl[0]), textcoords="offset points", xytext=(-18, -12), fontsize=7.2, color=c_l2, fontweight='bold')
ax1.annotate('13.9%', (x_stl[1], asr_l2_stl[1]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')

# Visual Dividers & Random Guess Baseline
ax1.axvline(1.6, color='#888888', linestyle=':', linewidth=1.0, alpha=0.6)
ax1.axhline(10.0, color='#666666', linestyle='--', linewidth=0.85, alpha=0.85)
ax1.text(0.97, 0.93, 'Dashed line: 10% Random Guess', transform=ax1.transAxes, ha='right', fontsize=6.8, color='#555555', style='italic')

ax1.set_ylabel(r'Attack Success Rate (%) $\downarrow$', fontweight='semibold')
ax1.set_title(r'(a) Attack Success Rate (ASR)', fontweight='bold', pad=8)
ax1.set_xticks(all_ticks)
ax1.set_xticklabels(all_labels)
ax1.set_xlim(-0.45, 3.65)
ax1.set_ylim(3, 49)
ax1.grid(True, linestyle='--', alpha=0.4)

# ----------------- Subplot 2: Clean Accuracy (C-Acc) -----------------
# CIFAR-10 segments
ax2.plot(x_cifar, cacc_base_cifar, 's--', color=c_base, label='Baseline Aegis', linewidth=1.8, markersize=6.5, markerfacecolor='white', markeredgewidth=1.8)
ax2.plot(x_cifar, cacc_l1_cifar, '^-.', color=c_l1, label='L1: Raw SBE', linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8)
ax2.plot(x_cifar, cacc_l2_cifar, 'o-', color=c_l2, label='L2: Self-Heal (Ours)', linewidth=2.4, markersize=7.5, markerfacecolor=c_l2, markeredgewidth=1.5)

# STL-10 segments
ax2.plot(x_stl, cacc_l1_stl, '^-.', color=c_l1, linewidth=1.8, markersize=7, markerfacecolor='white', markeredgewidth=1.8)
ax2.plot(x_stl, cacc_l2_stl, 'o-', color=c_l2, linewidth=2.4, markersize=7.5, markerfacecolor=c_l2, markeredgewidth=1.5)

# Baseline N/A indicators on STL-10 positioned cleanly below data points
ax2.text(x_stl[0], 68.4, '(Baseline N/A)', ha='center', fontsize=6.6, color='#777777', style='italic')
ax2.text(x_stl[1], 68.4, '(Baseline N/A)', ha='center', fontsize=6.6, color='#777777', style='italic')

# Annotations Clean Acc
ax2.annotate('88.7%', (x_cifar[0], cacc_base_cifar[0]), textcoords="offset points", xytext=(-18, -12), fontsize=7.2, color=c_base, fontweight='semibold')
ax2.annotate('83.4%', (x_cifar[1], cacc_base_cifar[1]), textcoords="offset points", xytext=(-18, -12), fontsize=7.2, color=c_base, fontweight='semibold')

ax2.annotate('90.1%', (x_cifar[0], cacc_l1_cifar[0]), textcoords="offset points", xytext=(8, -8), fontsize=7.2, color=c_l1, fontweight='semibold')
ax2.annotate('87.4%', (x_cifar[1], cacc_l1_cifar[1]), textcoords="offset points", xytext=(8, -8), fontsize=7.2, color=c_l1, fontweight='semibold')

ax2.annotate('90.3%', (x_cifar[0], cacc_l2_cifar[0]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')
ax2.annotate('90.7%', (x_cifar[1], cacc_l2_cifar[1]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')

ax2.annotate('73.9%', (x_stl[0], cacc_l1_stl[0]), textcoords="offset points", xytext=(-18, -10), fontsize=7.2, color=c_l1, fontweight='semibold')
ax2.annotate('73.7%', (x_stl[1], cacc_l1_stl[1]), textcoords="offset points", xytext=(-18, -10), fontsize=7.2, color=c_l1, fontweight='semibold')

ax2.annotate('74.1%', (x_stl[0], cacc_l2_stl[0]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')
ax2.annotate('74.2%', (x_stl[1], cacc_l2_stl[1]), textcoords="offset points", xytext=(0, 7), ha='center', fontsize=7.2, color=c_l2, fontweight='bold')

# Divider between datasets
ax2.axvline(1.6, color='#888888', linestyle=':', linewidth=1.0, alpha=0.6)

ax2.set_ylabel(r'Clean Accuracy (%) $\uparrow$', fontweight='semibold')
ax2.set_title(r'(b) Clean Accuracy Recovery', fontweight='bold', pad=8)
ax2.set_xticks(all_ticks)
ax2.set_xticklabels(all_labels)
ax2.set_xlim(-0.45, 3.65)
ax2.set_ylim(66, 95)
ax2.grid(True, linestyle='--', alpha=0.4)

# Unified single legend across top
handles, labels = ax1.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.07),
           ncol=3, frameon=True, framealpha=0.95, edgecolor='#cccccc', fontsize=8.2)

plt.tight_layout()
plt.subplots_adjust(top=0.86)

# Save as PDF (for LaTeX) and high-res PNG
plt.savefig('figures/tbt_defense_comparison.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figures/tbt_defense_comparison.png', format='png', dpi=300, bbox_inches='tight')
print("Pristine line graph generated.")
