"""
System Architecture Diagram Generator
Generates comprehensive visual diagrams of the complete research system

Components:
- Data pipeline architecture
- Model architectures (PPN, DPN-A, HMTL)
- End-to-end workflow
- Feature engineering pipeline
- Deployment architecture

Author: Final Thesis Project
Date: November 30, 2025
"""

import sys
import os

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np

# Create output directory
OUTPUT_DIR = "outputs/figures_journal"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("SYSTEM ARCHITECTURE DIAGRAM GENERATOR")
print("=" * 80)
print()

# =============================================================================
# DIAGRAM 1: COMPLETE END-TO-END SYSTEM ARCHITECTURE
# =============================================================================

print("Creating Diagram 1: Complete End-to-End System Architecture...")

fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors
color_data = '#E8F4F8'  # Light blue
color_preprocess = '#FFF4E6'  # Light orange
color_models = '#E8F5E9'  # Light green
color_evaluation = '#F3E5F5'  # Light purple
color_deployment = '#FFF9C4'  # Light yellow
color_arrow = '#37474F'  # Dark grey

# Title
ax.text(9, 13.5, 'Complete System Architecture: Student Outcome Prediction', 
        fontsize=18, fontweight='bold', ha='center')
ax.text(9, 13, 'Deep Learning Models with Attention Mechanism & Multi-Task Learning', 
        fontsize=12, ha='center', style='italic')

# ============= LAYER 1: DATA SOURCES =============
y_data = 11.5

# Data Source Box
data_box = FancyBboxPatch((0.5, y_data), 4, 1.2, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(data_box)
ax.text(2.5, y_data + 0.9, 'DATA SOURCES', fontsize=11, fontweight='bold', ha='center')
ax.text(2.5, y_data + 0.5, 'Educational Dataset', fontsize=9, ha='center')
ax.text(2.5, y_data + 0.2, 'N = 4,424 students', fontsize=8, ha='center', style='italic')

# Raw Features Box
features_box = FancyBboxPatch((5.5, y_data), 4, 1.2, boxstyle="round,pad=0.1", 
                              edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(features_box)
ax.text(7.5, y_data + 0.9, 'RAW FEATURES', fontsize=11, fontweight='bold', ha='center')
ax.text(7.5, y_data + 0.5, '37 Variables:', fontsize=9, ha='center')
ax.text(7.5, y_data + 0.2, 'Academic | Financial | Demographic', fontsize=8, ha='center', style='italic')

# Target Variables Box
target_box = FancyBboxPatch((10.5, y_data), 4, 1.2, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor=color_data, linewidth=2)
ax.add_patch(target_box)
ax.text(12.5, y_data + 0.9, 'TARGET VARIABLES', fontsize=11, fontweight='bold', ha='center')
ax.text(12.5, y_data + 0.5, '• Performance (3-class)', fontsize=8, ha='center')
ax.text(12.5, y_data + 0.2, '• Dropout (Binary)', fontsize=8, ha='center')

# Theoretical Framework Box
theory_box = FancyBboxPatch((15.5, y_data), 2, 1.2, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(theory_box)
ax.text(16.5, y_data + 0.9, 'THEORY', fontsize=10, fontweight='bold', ha='center')
ax.text(16.5, y_data + 0.5, 'Tinto (68%)', fontsize=7, ha='center')
ax.text(16.5, y_data + 0.2, 'Bean (32%)', fontsize=7, ha='center')

# ============= LAYER 2: PREPROCESSING =============
y_preprocess = 9.5

# Arrow from data to preprocessing
arrow1 = FancyArrowPatch((2.5, y_data), (5, y_preprocess + 1.1), 
                        arrowstyle='->', mutation_scale=30, linewidth=2, color=color_arrow)
ax.add_patch(arrow1)

# Preprocessing Pipeline Box
preprocess_main = FancyBboxPatch((1, y_preprocess), 14, 1.2, boxstyle="round,pad=0.1", 
                                 edgecolor='black', facecolor=color_preprocess, linewidth=2.5)
ax.add_patch(preprocess_main)
ax.text(8, y_preprocess + 0.9, 'PREPROCESSING PIPELINE', fontsize=11, fontweight='bold', ha='center')

# Sub-boxes
sub_boxes = [
    (1.5, 'Missing Value\nImputation'),
    (3.5, 'Categorical\nEncoding'),
    (5.5, 'Feature\nEngineering'),
    (7.5, 'Normalization\n(Z-score)'),
    (9.5, 'Class Weight\nCalculation'),
    (11.5, 'Train/Val/Test\nSplit'),
    (13.5, 'Tensor\nConversion')
]

for x_pos, label in sub_boxes:
    sub_box = FancyBboxPatch((x_pos - 0.4, y_preprocess + 0.05), 1.3, 0.7, 
                             boxstyle="round,pad=0.05", 
                             edgecolor='black', facecolor='white', linewidth=1)
    ax.add_patch(sub_box)
    ax.text(x_pos + 0.25, y_preprocess + 0.4, label, fontsize=7, ha='center', va='center')

# Data split annotation
ax.text(8, y_preprocess - 0.3, 'Train: 70% (3,096) | Val: 15% (664) | Test: 15% (664)', 
        fontsize=8, ha='center', style='italic', color='#D32F2F')

# ============= LAYER 3: MODEL ARCHITECTURES =============
y_models = 6.5

# Arrow from preprocessing to models
arrow2 = FancyArrowPatch((8, y_preprocess), (8, y_models + 2.3), 
                        arrowstyle='->', mutation_scale=30, linewidth=2, color=color_arrow)
ax.add_patch(arrow2)

# PPN Architecture
ppn_box = FancyBboxPatch((0.5, y_models), 5, 2.2, boxstyle="round,pad=0.1", 
                         edgecolor='#2E7D32', facecolor=color_models, linewidth=2.5)
ax.add_patch(ppn_box)
ax.text(3, y_models + 1.9, 'PPN: Performance Prediction', fontsize=10, fontweight='bold', ha='center')
ax.text(3, y_models + 1.6, '(3-Class: Dropout/Enrolled/Graduate)', fontsize=8, ha='center', style='italic')

ppn_layers = [
    'Input: 46 features',
    'FC1: 46 → 128 (ReLU, BN, Drop 0.3)',
    'FC2: 128 → 64 (ReLU, BN, Drop 0.2)',
    'FC3: 64 → 32 (ReLU, Drop 0.1)',
    'Output: 32 → 3 (Softmax)'
]
for i, layer in enumerate(ppn_layers):
    ax.text(3, y_models + 1.2 - i*0.25, layer, fontsize=7, ha='center', family='monospace')

# DPN-A Architecture
dpna_box = FancyBboxPatch((6.5, y_models), 5, 2.2, boxstyle="round,pad=0.1", 
                          edgecolor='#C62828', facecolor=color_models, linewidth=2.5)
ax.add_patch(dpna_box)
ax.text(9, y_models + 1.9, 'DPN-A: Dropout with Attention', fontsize=10, fontweight='bold', ha='center')
ax.text(9, y_models + 1.6, '(Binary: Dropout vs Not Dropout)', fontsize=8, ha='center', style='italic')

dpna_layers = [
    'Input: 46 features',
    'FC1: 46 → 64 (ReLU, BN, Drop 0.3)',
    '🔍 Self-Attention Layer (64-dim)',
    'FC2: 64 → 32 (ReLU, Drop 0.2)',
    'FC3: 32 → 16 (ReLU)',
    'Output: 16 → 1 (Sigmoid)'
]
for i, layer in enumerate(dpna_layers):
    ax.text(9, y_models + 1.2 - i*0.23, layer, fontsize=7, ha='center', family='monospace')

# HMTL Architecture
hmtl_box = FancyBboxPatch((12.5, y_models), 5, 2.2, boxstyle="round,pad=0.1", 
                          edgecolor='#6A1B9A', facecolor=color_models, linewidth=2.5)
ax.add_patch(hmtl_box)
ax.text(15, y_models + 1.9, 'HMTL: Multi-Task Learning', fontsize=10, fontweight='bold', ha='center')
ax.text(15, y_models + 1.6, '(Shared Trunk + Dual Heads)', fontsize=8, ha='center', style='italic')

hmtl_layers = [
    'Shared: 46 → 128 → 64',
    'Head 1 (Perf): 64 → 32 → 3',
    'Head 2 (Drop): 64 → 16 → 1',
    'Loss: L_perf + λL_drop (λ=1.0)',
    '⚠ Task Interference Detected'
]
for i, layer in enumerate(hmtl_layers):
    ax.text(15, y_models + 1.2 - i*0.27, layer, fontsize=7, ha='center', family='monospace')

# ============= LAYER 4: TRAINING & OPTIMIZATION =============
y_train = 4

# Arrow from models to training
for x in [3, 9, 15]:
    arrow_train = FancyArrowPatch((x, y_models), (x, y_train + 0.8), 
                                 arrowstyle='->', mutation_scale=20, linewidth=1.5, color=color_arrow)
    ax.add_patch(arrow_train)

# Training Box
train_box = FancyBboxPatch((1, y_train), 16, 0.7, boxstyle="round,pad=0.05", 
                           edgecolor='black', facecolor='#E3F2FD', linewidth=2)
ax.add_patch(train_box)
ax.text(9, y_train + 0.5, 'TRAINING & OPTIMIZATION', fontsize=10, fontweight='bold', ha='center')
ax.text(3, y_train + 0.2, 'Optimizer: Adam (lr=0.001)', fontsize=7, ha='center')
ax.text(7, y_train + 0.2, 'Loss: CrossEntropy (PPN/HMTL), BCE (DPN-A)', fontsize=7, ha='center')
ax.text(11, y_train + 0.2, 'Early Stop: Patience=20', fontsize=7, ha='center')
ax.text(15, y_train + 0.2, 'LR Scheduler: ReduceLROnPlateau', fontsize=7, ha='center')

# ============= LAYER 5: EVALUATION METRICS =============
y_eval = 2.5

# Arrow to evaluation
arrow_eval = FancyArrowPatch((9, y_train), (9, y_eval + 0.8), 
                            arrowstyle='->', mutation_scale=30, linewidth=2, color=color_arrow)
ax.add_patch(arrow_eval)

# Evaluation Metrics Box
eval_box = FancyBboxPatch((2, y_eval), 14, 0.7, boxstyle="round,pad=0.05", 
                          edgecolor='black', facecolor=color_evaluation, linewidth=2)
ax.add_patch(eval_box)
ax.text(9, y_eval + 0.5, 'EVALUATION METRICS', fontsize=10, fontweight='bold', ha='center')

metrics = [
    (3.5, 'Accuracy'),
    (5, 'F1-Macro'),
    (6.5, 'Precision'),
    (8, 'Recall'),
    (9.5, 'AUC-ROC'),
    (11, 'AUC-PR'),
    (12.5, 'Confusion Matrix'),
    (14.5, '10-Fold CV')
]
for x_pos, metric in metrics:
    ax.text(x_pos, y_eval + 0.2, metric, fontsize=7, ha='center', 
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=0.5))

# ============= LAYER 6: RESULTS & DEPLOYMENT =============
y_result = 1

# Arrow to results
arrow_result = FancyArrowPatch((9, y_eval), (9, y_result + 0.5), 
                              arrowstyle='->', mutation_scale=30, linewidth=2, color=color_arrow)
ax.add_patch(arrow_result)

# Results Box - PPN
result_ppn = FancyBboxPatch((0.5, y_result), 5, 0.45, boxstyle="round,pad=0.05", 
                            edgecolor='#2E7D32', facecolor='#C8E6C9', linewidth=2)
ax.add_patch(result_ppn)
ax.text(3, y_result + 0.3, 'PPN Results', fontsize=9, fontweight='bold', ha='center')
ax.text(3, y_result + 0.1, 'Acc: 76.4% | F1: 0.688', fontsize=7, ha='center')

# Results Box - DPN-A
result_dpna = FancyBboxPatch((6.5, y_result), 5, 0.45, boxstyle="round,pad=0.05", 
                             edgecolor='#C62828', facecolor='#FFCDD2', linewidth=2)
ax.add_patch(result_dpna)
ax.text(9, y_result + 0.3, 'DPN-A Results ⭐ BEST', fontsize=9, fontweight='bold', ha='center')
ax.text(9, y_result + 0.1, 'Acc: 87.05% | AUC: 0.910 | F1: 0.782', fontsize=7, ha='center')

# Results Box - HMTL
result_hmtl = FancyBboxPatch((12.5, y_result), 5, 0.45, boxstyle="round,pad=0.05", 
                             edgecolor='#6A1B9A', facecolor='#E1BEE7', linewidth=2)
ax.add_patch(result_hmtl)
ax.text(15, y_result + 0.3, 'HMTL Results', fontsize=9, fontweight='bold', ha='center')
ax.text(15, y_result + 0.1, 'Perf: 76.4% | Drop: 67.9% ⚠', fontsize=7, ha='center')

# Deployment Box
deploy_box = FancyBboxPatch((1, y_result - 0.6), 16, 0.4, boxstyle="round,pad=0.05", 
                            edgecolor='black', facecolor=color_deployment, linewidth=2.5)
ax.add_patch(deploy_box)
ax.text(9, y_result - 0.3, 'DEPLOYMENT: Institutional Early Warning System | Intervention Recommendations', 
        fontsize=9, fontweight='bold', ha='center')

# Legend
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Data Layer'),
    mpatches.Patch(facecolor=color_preprocess, edgecolor='black', label='Preprocessing'),
    mpatches.Patch(facecolor=color_models, edgecolor='black', label='Model Architectures'),
    mpatches.Patch(facecolor=color_evaluation, edgecolor='black', label='Evaluation'),
    mpatches.Patch(facecolor=color_deployment, edgecolor='black', label='Deployment')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, frameon=True, 
          fancybox=True, shadow=True)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/system_architecture_complete.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/system_architecture_complete.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Diagram 1 saved: Complete End-to-End System Architecture")
print()

# =============================================================================
# DIAGRAM 2: DETAILED MODEL ARCHITECTURES (SIDE-BY-SIDE)
# =============================================================================

print("Creating Diagram 2: Detailed Model Architectures Comparison...")

fig, axes = plt.subplots(1, 3, figsize=(18, 10))

# ===== PPN Architecture =====
ax1 = axes[0]
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('PPN: Performance Prediction Network\n(3-Class Classification)', 
              fontsize=12, fontweight='bold', pad=20)

# Layer boxes for PPN
layers_ppn = [
    (5, 10.5, 'Input Layer\n46 features', 3, 0.8, '#E3F2FD'),
    (5, 9, 'FC1: 46 → 128\nReLU + BatchNorm\nDropout(0.3)', 3.5, 1.2, '#BBDEFB'),
    (5, 7, 'FC2: 128 → 64\nReLU + BatchNorm\nDropout(0.2)', 3.5, 1.2, '#90CAF9'),
    (5, 5, 'FC3: 64 → 32\nReLU\nDropout(0.1)', 3.5, 1.2, '#64B5F6'),
    (5, 3, 'Output: 32 → 3\nSoftmax', 3, 0.8, '#42A5F5'),
    (5, 1.5, 'Predictions\n[Dropout, Enrolled, Graduate]', 4, 0.8, '#C8E6C9')
]

for x, y, label, w, h, color in layers_ppn:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax1.add_patch(box)
    ax1.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')

# Arrows
for i in range(len(layers_ppn) - 1):
    y_start = layers_ppn[i][1] - layers_ppn[i][4]/2
    y_end = layers_ppn[i+1][1] + layers_ppn[i+1][4]/2
    arrow = FancyArrowPatch((5, y_start), (5, y_end), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#37474F')
    ax1.add_patch(arrow)

# Metrics annotation
ax1.text(5, 0.5, 'Accuracy: 76.4% | F1-Macro: 0.688', 
         fontsize=9, ha='center', bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ===== DPN-A Architecture =====
ax2 = axes[1]
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('DPN-A: Dropout Prediction with Attention\n(Binary Classification)', 
              fontsize=12, fontweight='bold', pad=20)

# Layer boxes for DPN-A
layers_dpna = [
    (5, 10.5, 'Input Layer\n46 features', 3, 0.8, '#FFF3E0'),
    (5, 9, 'FC1: 46 → 64\nReLU + BatchNorm\nDropout(0.3)', 3.5, 1.2, '#FFE0B2'),
    (5, 7, '🔍 Self-Attention\n64-dim\nQ, K, V projection', 4, 1.5, '#FFCC80'),
    (5, 5, 'FC2: 64 → 32\nReLU\nDropout(0.2)', 3.5, 1.2, '#FFB74D'),
    (5, 3.5, 'FC3: 32 → 16\nReLU', 3, 0.8, '#FFA726'),
    (5, 2.2, 'Output: 16 → 1\nSigmoid', 3, 0.8, '#FF9800'),
    (5, 1, 'Prediction\n[Dropout Probability]', 4, 0.6, '#FFCDD2')
]

for x, y, label, w, h, color in layers_dpna:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax2.add_patch(box)
    ax2.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')

# Arrows
for i in range(len(layers_dpna) - 1):
    y_start = layers_dpna[i][1] - layers_dpna[i][4]/2
    y_end = layers_dpna[i+1][1] + layers_dpna[i+1][4]/2
    arrow = FancyArrowPatch((5, y_start), (5, y_end), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#37474F')
    ax2.add_patch(arrow)

# Attention annotation
ax2.text(8.5, 7, 'Attention\nWeights', fontsize=7, ha='center', 
         bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='orange', linewidth=2))
arrow_attn = FancyArrowPatch((8, 7), (6.8, 7), 
                            arrowstyle='->', mutation_scale=15, linewidth=1.5, color='orange')
ax2.add_patch(arrow_attn)

# Metrics annotation
ax2.text(5, 0.3, 'Accuracy: 87.05% | AUC-ROC: 0.910 | F1: 0.782', 
         fontsize=9, ha='center', fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ===== HMTL Architecture =====
ax3 = axes[2]
ax3.set_xlim(0, 12)
ax3.set_ylim(0, 12)
ax3.axis('off')
ax3.set_title('HMTL: Hybrid Multi-Task Learning\n(Dual Outputs)', 
              fontsize=12, fontweight='bold', pad=20)

# Shared trunk
shared_layers = [
    (6, 10.5, 'Input Layer\n46 features', 3, 0.8, '#F3E5F5'),
    (6, 9.2, 'Shared: 46 → 128\nReLU', 3.5, 0.8, '#E1BEE7'),
    (6, 8.2, 'Shared: 128 → 64\nReLU', 3.5, 0.8, '#CE93D8')
]

for x, y, label, w, h, color in shared_layers:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax3.add_patch(box)
    ax3.text(x, y, label, fontsize=8, ha='center', va='center', fontweight='bold')

# Arrows for shared trunk
for i in range(len(shared_layers) - 1):
    y_start = shared_layers[i][1] - shared_layers[i][4]/2
    y_end = shared_layers[i+1][1] + shared_layers[i+1][4]/2
    arrow = FancyArrowPatch((6, y_start), (6, y_end), 
                           arrowstyle='->', mutation_scale=20, linewidth=2, color='#37474F')
    ax3.add_patch(arrow)

# Split arrow
arrow_split_left = FancyArrowPatch((6, 7.8), (3, 6.8), 
                                  arrowstyle='->', mutation_scale=20, linewidth=2, color='#2E7D32')
arrow_split_right = FancyArrowPatch((6, 7.8), (9, 6.8), 
                                   arrowstyle='->', mutation_scale=20, linewidth=2, color='#C62828')
ax3.add_patch(arrow_split_left)
ax3.add_patch(arrow_split_right)

# Performance Head (Left)
perf_layers = [
    (3, 6.3, 'Head 1\n64 → 32', 2.5, 0.7, '#C8E6C9'),
    (3, 5.2, '32 → 3\nSoftmax', 2.5, 0.7, '#A5D6A7'),
    (3, 4, 'Performance\nOutput', 2.5, 0.6, '#81C784')
]

for x, y, label, w, h, color in perf_layers:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.05", 
                         edgecolor='#2E7D32', facecolor=color, linewidth=1.5)
    ax3.add_patch(box)
    ax3.text(x, y, label, fontsize=7, ha='center', va='center', fontweight='bold')

# Arrows for perf head
for i in range(len(perf_layers) - 1):
    y_start = perf_layers[i][1] - perf_layers[i][4]/2
    y_end = perf_layers[i+1][1] + perf_layers[i+1][4]/2
    arrow = FancyArrowPatch((3, y_start), (3, y_end), 
                           arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#2E7D32')
    ax3.add_patch(arrow)

# Dropout Head (Right)
drop_layers = [
    (9, 6.3, 'Head 2\n64 → 16', 2.5, 0.7, '#FFCDD2'),
    (9, 5.2, '16 → 1\nSigmoid', 2.5, 0.7, '#EF9A9A'),
    (9, 4, 'Dropout\nOutput', 2.5, 0.6, '#E57373')
]

for x, y, label, w, h, color in drop_layers:
    box = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle="round,pad=0.05", 
                         edgecolor='#C62828', facecolor=color, linewidth=1.5)
    ax3.add_patch(box)
    ax3.text(x, y, label, fontsize=7, ha='center', va='center', fontweight='bold')

# Arrows for drop head
for i in range(len(drop_layers) - 1):
    y_start = drop_layers[i][1] - drop_layers[i][4]/2
    y_end = drop_layers[i+1][1] + drop_layers[i+1][4]/2
    arrow = FancyArrowPatch((9, y_start), (9, y_end), 
                           arrowstyle='->', mutation_scale=15, linewidth=1.5, color='#C62828')
    ax3.add_patch(arrow)

# Loss function box
loss_box = FancyBboxPatch((3.5, 2.5), 5, 1, boxstyle="round,pad=0.1", 
                         edgecolor='black', facecolor='#FFF9C4', linewidth=2)
ax3.add_patch(loss_box)
ax3.text(6, 3.2, 'Combined Loss', fontsize=9, ha='center', fontweight='bold')
ax3.text(6, 2.9, 'L_total = L_perf + λL_drop', fontsize=8, ha='center', family='monospace')
ax3.text(6, 2.6, 'λ = 1.0 (equal weighting)', fontsize=7, ha='center', style='italic')

# Results
ax3.text(3, 2, 'Acc: 76.4%', fontsize=8, ha='center', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax3.text(9, 2, 'Acc: 67.9% ⚠', fontsize=8, ha='center', 
         bbox=dict(boxstyle='round', facecolor='#FFCDD2', alpha=0.7))
ax3.text(6, 0.8, '⚠ Task Interference: Dropout task underperforms (-19.15% vs DPN-A)', 
         fontsize=8, ha='center', style='italic', color='red')

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/system_architecture_models.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/system_architecture_models.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Diagram 2 saved: Detailed Model Architectures Comparison")
print()

# =============================================================================
# DIAGRAM 3: DATA FLOW & FEATURE ENGINEERING PIPELINE
# =============================================================================

print("Creating Diagram 3: Data Flow & Feature Engineering Pipeline...")

fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111)
ax.set_xlim(0, 16)
ax.set_ylim(0, 10)
ax.axis('off')

ax.text(8, 9.5, 'Data Flow & Feature Engineering Pipeline', 
        fontsize=16, fontweight='bold', ha='center')

# Stage 1: Raw Data
stage1_box = FancyBboxPatch((0.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#E8F4F8', linewidth=2)
ax.add_patch(stage1_box)
ax.text(2, 8.7, 'RAW DATA', fontsize=10, fontweight='bold', ha='center')
ax.text(2, 8.3, 'CSV File', fontsize=8, ha='center')
ax.text(2, 8, '4,424 rows', fontsize=7, ha='center')
ax.text(2, 7.7, '37 columns', fontsize=7, ha='center')

# Stage 2: Data Cleaning
arrow1 = FancyArrowPatch((3.5, 8.25), (4.5, 8.25), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow1)

stage2_box = FancyBboxPatch((4.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#FFF4E6', linewidth=2)
ax.add_patch(stage2_box)
ax.text(6, 8.7, 'DATA CLEANING', fontsize=10, fontweight='bold', ha='center')
cleaning_steps = [
    '✓ Remove duplicates',
    '✓ Handle missing (KNN)',
    '✓ Drop invalid rows'
]
for i, step in enumerate(cleaning_steps):
    ax.text(6, 8.2 - i*0.3, step, fontsize=7, ha='center')

# Stage 3: Encoding
arrow2 = FancyArrowPatch((7.5, 8.25), (8.5, 8.25), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow2)

stage3_box = FancyBboxPatch((8.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#E8F5E9', linewidth=2)
ax.add_patch(stage3_box)
ax.text(10, 8.7, 'ENCODING', fontsize=10, fontweight='bold', ha='center')
encoding_steps = [
    'Categorical → Numeric',
    'One-Hot: gender, marital',
    'Label: daytime/evening'
]
for i, step in enumerate(encoding_steps):
    ax.text(10, 8.2 - i*0.3, step, fontsize=7, ha='center')

# Stage 4: Feature Engineering
arrow3 = FancyArrowPatch((11.5, 8.25), (12.5, 8.25), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow3)

stage4_box = FancyBboxPatch((12.5, 7.5), 3, 1.5, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#F3E5F5', linewidth=2)
ax.add_patch(stage4_box)
ax.text(14, 8.7, 'FEATURE ENG', fontsize=10, fontweight='bold', ha='center')
feature_steps = [
    '+ Success rate',
    '+ Average grade',
    '+ Academic progression'
]
for i, step in enumerate(feature_steps):
    ax.text(14, 8.2 - i*0.3, step, fontsize=7, ha='center')

# Feature Categories
arrow4 = FancyArrowPatch((8, 7.5), (8, 6.5), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow4)

# Three category boxes
categories = [
    (2, 'ACADEMIC\nFEATURES', '#BBDEFB', [
        'Grades (sem 1, 2)',
        'Success rate',
        'Enrolled units',
        'Approved units',
        'Evaluations'
    ]),
    (8, 'FINANCIAL\nFEATURES', '#FFE0B2', [
        'Tuition status',
        'Scholarship',
        'Debtor status',
        'Payment history'
    ]),
    (14, 'DEMOGRAPHIC\nFEATURES', '#C8E6C9', [
        'Age at enrollment',
        'Gender',
        'Marital status',
        'Parental education',
        'Previous qualification'
    ])
]

for x, title, color, features in categories:
    cat_box = FancyBboxPatch((x - 1.8, 4), 3.6, 2.3, boxstyle="round,pad=0.1", 
                             edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(cat_box)
    ax.text(x, 6.1, title, fontsize=9, fontweight='bold', ha='center')
    for i, feat in enumerate(features):
        ax.text(x, 5.7 - i*0.35, f'• {feat}', fontsize=7, ha='center')

# Normalization stage
arrow5 = FancyArrowPatch((8, 4), (8, 3), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow5)

norm_box = FancyBboxPatch((3, 2.2), 10, 0.7, boxstyle="round,pad=0.1", 
                          edgecolor='black', facecolor='#FFF9C4', linewidth=2)
ax.add_patch(norm_box)
ax.text(8, 2.75, 'NORMALIZATION (Z-Score)', fontsize=10, fontweight='bold', ha='center')
ax.text(8, 2.4, 'x_norm = (x - μ) / σ    [Applied to all 46 features]', 
        fontsize=8, ha='center', family='monospace')

# Final output
arrow6 = FancyArrowPatch((8, 2.2), (8, 1.5), 
                        arrowstyle='->', mutation_scale=25, linewidth=2, color='black')
ax.add_patch(arrow6)

output_box = FancyBboxPatch((3.5, 0.5), 9, 0.9, boxstyle="round,pad=0.1", 
                            edgecolor='black', facecolor='#E1BEE7', linewidth=2.5)
ax.add_patch(output_box)
ax.text(8, 1.2, 'PROCESSED DATASET: 46 Features × 4,424 Samples', 
        fontsize=11, fontweight='bold', ha='center')
ax.text(8, 0.8, 'Train: 3,096 (70%) | Validation: 664 (15%) | Test: 664 (15%)', 
        fontsize=9, ha='center', style='italic')

# Theoretical mapping
theory_box = FancyBboxPatch((0.3, 4), 1.2, 2.3, boxstyle="round,pad=0.05", 
                            edgecolor='#D32F2F', facecolor='#FFEBEE', linewidth=2)
ax.add_patch(theory_box)
ax.text(0.9, 6.1, 'THEORY', fontsize=8, fontweight='bold', ha='center')
ax.text(0.9, 5.7, 'Tinto', fontsize=7, ha='center', color='#1976D2')
ax.text(0.9, 5.4, '68%', fontsize=7, ha='center', fontweight='bold', color='#1976D2')
ax.text(0.9, 4.9, 'Bean', fontsize=7, ha='center', color='#F57C00')
ax.text(0.9, 4.6, '32%', fontsize=7, ha='center', fontweight='bold', color='#F57C00')
ax.text(0.9, 4.2, 'Weight', fontsize=6, ha='center', style='italic')

# Arrows to theory
arrow_t1 = FancyArrowPatch((1.5, 5.5), (0.4, 5.7), 
                          arrowstyle='->', mutation_scale=10, linewidth=1, color='#1976D2', linestyle='--')
arrow_t2 = FancyArrowPatch((6.2, 5.2), (1.1, 4.9), 
                          arrowstyle='->', mutation_scale=10, linewidth=1, color='#F57C00', linestyle='--')
ax.add_patch(arrow_t1)
ax.add_patch(arrow_t2)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/system_architecture_dataflow.pdf", dpi=300, bbox_inches='tight')
plt.savefig(f"{OUTPUT_DIR}/system_architecture_dataflow.png", dpi=300, bbox_inches='tight')
plt.close()

print("✓ Diagram 3 saved: Data Flow & Feature Engineering Pipeline")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("SYSTEM ARCHITECTURE DIAGRAMS COMPLETE")
print("=" * 80)
print(f"\nAll diagrams saved to: {OUTPUT_DIR}/\n")
print("Generated Diagrams:")
print("  ✓ system_architecture_complete.pdf/png")
print("    - End-to-end system overview (6 layers)")
print("    - Data → Preprocessing → Models → Training → Evaluation → Deployment")
print()
print("  ✓ system_architecture_models.pdf/png")
print("    - Detailed model architectures side-by-side")
print("    - PPN (3-class) | DPN-A (attention) | HMTL (multi-task)")
print()
print("  ✓ system_architecture_dataflow.pdf/png")
print("    - Feature engineering pipeline")
print("    - Raw data → Cleaning → Encoding → Engineering → Normalization")
print()
print("Formats: PDF (vector, publication-ready) + PNG (preview)")
print("\n🎯 Professional system design diagrams ready for thesis/journal!")
print("=" * 80)
