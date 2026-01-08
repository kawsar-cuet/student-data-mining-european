"""
Generate Comprehensive Metrics Comparison Figure with AHFS-TA
Creates a 2×2 subplot showing all key metrics comparing all 7 models
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
print("GENERATING COMPREHENSIVE METRICS COMPARISON WITH AHFS-TA")
print("="*80)

# ============================================================================
# MODEL RESULTS WITH AHFS-TA
# ============================================================================

all_models = {
    'Decision Tree': {
        'accuracy': 0.6700,
        'precision': 0.667,
        'recall': 0.670,
        'f1_score': 0.668,
        'auc': 0.758,
        'cv_mean': 0.6823
    },
    'Naive Bayes': {
        'accuracy': 0.7090,
        'precision': 0.711,
        'recall': 0.709,
        'f1_score': 0.710,
        'auc': 0.843,
        'cv_mean': 0.7085
    },
    'Random Forest': {
        'accuracy': 0.7670,
        'precision': 0.768,
        'recall': 0.767,
        'f1_score': 0.767,
        'auc': 0.914,
        'cv_mean': 0.7612
    },
    'AdaBoost': {
        'accuracy': 0.7420,
        'precision': 0.744,
        'recall': 0.742,
        'f1_score': 0.743,
        'auc': 0.890,
        'cv_mean': 0.7389
    },
    'XGBoost': {
        'accuracy': 0.7590,
        'precision': 0.761,
        'recall': 0.759,
        'f1_score': 0.760,
        'auc': 0.913,
        'cv_mean': 0.7556
    },
    'Neural Network': {
        'accuracy': 0.7140,
        'precision': 0.715,
        'recall': 0.714,
        'f1_score': 0.714,
        'auc': 0.861,
        'cv_mean': 0.7098
    },
    'AHFS-TA': {
        'accuracy': 0.9132,
        'precision': 0.915,
        'recall': 0.913,
        'f1_score': 0.914,
        'auc': 0.955,
        'cv_mean': 0.9085
    }
}

model_names = list(all_models.keys())
colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c', '#ff6b6b']

# ============================================================================
# CREATE 2×2 COMPREHENSIVE METRICS COMPARISON
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 1. Accuracy, Precision, Recall, F1-Score (Top Left)
ax1 = axes[0, 0]
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(model_names))
width = 0.2

metric_colors = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']

for i, metric in enumerate(metrics_to_plot):
    values = [all_models[model][metric] for model in model_names]
    offset = (i - 1.5) * width
    ax1.bar(x + offset, values, width, label=metric.replace('_', ' ').title(), 
            color=metric_colors[i], alpha=0.8)

ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
ax1.set_title('(a) Performance Metrics Comparison', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.set_ylim([0.6, 1.0])

# 2. AUC Comparison (Top Right)
ax2 = axes[0, 1]
auc_values = [all_models[model]['auc'] for model in model_names]
bars = ax2.bar(model_names, auc_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, auc_val in zip(bars, auc_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{auc_val:.3f}',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Micro-Average AUC', fontsize=12, fontweight='bold')
ax2.set_xlabel('Model', fontsize=12, fontweight='bold')
ax2.set_title('(b) Area Under ROC Curve (AUC) Comparison', fontsize=13, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.set_ylim([0.7, 1.0])
ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)

# 3. Accuracy Comparison with Improvement (Bottom Left)
ax3 = axes[1, 0]
accuracies = [all_models[model]['accuracy'] for model in model_names]
bars = ax3.bar(model_names, accuracies, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, acc in zip(bars, accuracies):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{acc:.4f}\n({acc*100:.2f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight AHFS-TA
bars[-1].set_color('#ff6b6b')
bars[-1].set_alpha(1.0)
bars[-1].set_linewidth(3)

ax3.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')
ax3.set_xlabel('Model', fontsize=12, fontweight='bold')
ax3.set_title('(c) Test Accuracy Comparison', fontsize=13, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim([0.6, 1.0])
ax3.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)

# 4. Cross-Validation Mean Accuracy (Bottom Right)
ax4 = axes[1, 1]
cv_means = [all_models[model]['cv_mean'] for model in model_names]
bars = ax4.bar(model_names, cv_means, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

for bar, mean in zip(bars, cv_means):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{mean:.4f}\n({mean*100:.2f}%)',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

# Highlight AHFS-TA
bars[-1].set_color('#ff6b6b')
bars[-1].set_alpha(1.0)
bars[-1].set_linewidth(3)

ax4.set_ylabel('CV Mean Accuracy', fontsize=12, fontweight='bold')
ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
ax4.set_title('(d) 10-Fold Cross-Validation Mean Accuracy', fontsize=13, fontweight='bold')
ax4.tick_params(axis='x', rotation=45)
ax4.grid(axis='y', alpha=0.3, linestyle='--')
ax4.set_ylim([0.6, 1.0])
ax4.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)

plt.tight_layout()

# Save to all directories
plt.savefig(figures_dir / "12_comprehensive_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(journal_figures_dir / "12_comprehensive_metrics_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(supervisor_figures_dir / "12_comprehensive_metrics_comparison.png", dpi=300, bbox_inches='tight')
print("\n[OK] Saved: 12_comprehensive_metrics_comparison.png (all directories)")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\n[OK] Comprehensive metrics comparison figure generated with all 7 models")
print(f"\nFigure includes:")
print(f"   1. Performance Metrics (Accuracy, Precision, Recall, F1)")
print(f"   2. AUC Comparison")
print(f"   3. Test Accuracy Comparison (AHFS-TA highlighted)")
print(f"   4. 10-Fold Cross-Validation Results (AHFS-TA highlighted)")
print(f"\nAHFS-TA Performance Summary:")
print(f"   • Accuracy: {all_models['AHFS-TA']['accuracy']:.4f} ({all_models['AHFS-TA']['accuracy']*100:.2f}%)")
print(f"   • AUC: {all_models['AHFS-TA']['auc']:.3f}")
print(f"   • CV Mean: {all_models['AHFS-TA']['cv_mean']:.4f} ({all_models['AHFS-TA']['cv_mean']*100:.2f}%)")
print(f"\n📁 Figure saved to:")
print(f"   - {figures_dir}/")
print(f"   - {journal_figures_dir}/")
print(f"   - {supervisor_figures_dir}/")
print("\n" + "="*80)
