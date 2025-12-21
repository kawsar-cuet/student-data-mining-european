"""
Generate AHFS-TA Ablation Study Figure
Shows the contribution of each component with proper layout
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Create output directories
figures_dir = Path("outputs/figures")
journal_figures_dir = Path("Journal Paper Writing/figures")
supervisor_figures_dir = Path("supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Writing/figures")

for dir_path in [figures_dir, journal_figures_dir, supervisor_figures_dir]:
    dir_path.mkdir(parents=True, exist_ok=True)

print("="*80)
print("GENERATING AHFS-TA ABLATION STUDY FIGURE")
print("="*80)

# ============================================================================
# ABLATION STUDY RESULTS (EXACT from LaTeX Table - Ablation Study Results)
# ============================================================================

# Configurations showing what happens when components are removed
configurations = [
    'w/o Temporal\nAttention',
    'w/o Adaptive\nSelection',
    'w/o LLM\nFeatures',
    'w/o Multi-Head\n(Single Head)',
    'BiGRU to LSTM',
    'Full\nAHFS-TA'
]

# Exact accuracy values from LaTeX table
accuracies = [0.847, 0.872, 0.889, 0.895, 0.908, 0.9132]
deltas = ['-6.6%', '-4.1%', '-2.4%', '-1.8%', '-0.5%', '---']

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

x_pos = np.arange(len(configurations))
width = 0.6

# Create bars
bars = ax.bar(x_pos, accuracies, width, color=['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#3498db', '#2ecc71'], 
              alpha=0.85, edgecolor='black', linewidth=2)

# Highlight the full AHFS-TA
bars[-1].set_color('#2ecc71')
bars[-1].set_alpha(1.0)
bars[-1].set_linewidth(3)

# Add value labels on bars
for bar, value, delta in zip(bars, accuracies, deltas):
    height = bar.get_height()
    # Accuracy value
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{value:.1%}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')
    # Delta value
    if delta != '---':
        ax.text(bar.get_x() + bar.get_width()/2., height - 0.03,
                delta,
                ha='center', va='top', fontsize=9, color='white', fontweight='bold')

# Labels and formatting
ax.set_ylabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_xlabel('Model Configuration', fontsize=13, fontweight='bold')
ax.set_title('AHFS-TA Ablation Study', fontsize=15, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(configurations, fontsize=10, ha='center')

# Grid
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_ylim([0.80, 0.95])

# Add baseline reference line (Full AHFS-TA)
ax.axhline(y=0.9132, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Full AHFS-TA (91.32%)')
ax.legend(loc='lower right', fontsize=11, framealpha=0.95, edgecolor='black')

plt.tight_layout()

# Save to all directories
plt.savefig(figures_dir / "ahfs_ta_ablation_study.png", dpi=300, bbox_inches='tight')
plt.savefig(journal_figures_dir / "ahfs_ta_ablation_study.png", dpi=300, bbox_inches='tight')
plt.savefig(supervisor_figures_dir / "ahfs_ta_ablation_study.png", dpi=300, bbox_inches='tight')
print("\n[OK] Saved: ahfs_ta_ablation_study.png (all directories)")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ABLATION STUDY SUMMARY")
print("="*80)
print("\nExact Values from LaTeX Table:")
print(f"  Full AHFS-TA:                     91.32%")
print(f"  w/o Temporal Attention:           84.7%  (Δ -6.6%)")
print(f"  w/o Adaptive Selection:           87.2%  (Δ -4.1%)")
print(f"  w/o LLM Features:                 88.9%  (Δ -2.4%)")
print(f"  w/o Multi-Head (single head):     89.5%  (Δ -1.8%)")
print(f"  BiGRU → LSTM:                     90.8%  (Δ -0.5%)")

print("\nKey Findings:")
print("  • Temporal Attention is the most critical component (-6.6%)")
print("  • Adaptive Selection significantly improves performance (-4.1%)")
print("  • LLM Features add valuable psychosocial insights (-2.4%)")
print("  • Multi-head attention outperforms single head (-1.8%)")
print("  • BiGRU slightly better than LSTM (-0.5%)")

print("\nFigure Features:")
print("  • Exact values from LaTeX Table")
print("  • Shows impact of removing each component")
print("  • No target lines (removed)")
print("  • Clear labels showing accuracy and delta")
print("  • Full AHFS-TA highlighted in green")
print("  • Color-coded by impact severity")

print(f"\n📁 Figure saved to:")
print(f"   - {figures_dir}/")
print(f"   - {journal_figures_dir}/")
print(f"   - {supervisor_figures_dir}/")
print("\n" + "="*80)
