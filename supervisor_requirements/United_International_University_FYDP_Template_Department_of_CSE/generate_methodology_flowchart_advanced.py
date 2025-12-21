"""
Advanced Methodology Flowchart Generator for Thesis
Creates a professional flowchart showing the complete research methodology
Similar to adaptive feature selection algorithm (AFSA) style
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 24)
ax.axis('off')

# Color scheme (matching the example flowchart)
color_dataset = '#E8E8E8'  # Light gray
color_process = '#B8D4E8'  # Light blue
color_decision = '#FFE4B5'  # Light orange
color_ensemble = '#D4E8D4'  # Light green

# Font sizes
fs_title = 11
fs_text = 9
fs_small = 8

# Helper function to create boxes
def create_box(ax, x, y, width, height, text, color, fontsize=9, style='round', alpha=1.0):
    if style == 'round':
        box = FancyBboxPatch((x-width/2, y-height/2), width, height,
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    else:
        box = Rectangle((x-width/2, y-height/2), width, height,
                       facecolor=color, edgecolor='black', linewidth=1.5, alpha=alpha)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight='bold', wrap=True)

def create_arrow(ax, x1, y1, x2, y2, style='->', color='black', linewidth=2):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                          arrowstyle=style, color=color, linewidth=linewidth,
                          mutation_scale=20, zorder=0)
    ax.add_patch(arrow)

def create_text(ax, x, y, text, fontsize=8, weight='normal', color='black'):
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, weight=weight, color=color)

# ============================================================================
# FLOWCHART CONSTRUCTION
# ============================================================================

# 1. DATASET (Top)
y_pos = 23
create_box(ax, 5, y_pos, 2, 0.6, 'Dataset\n4,424 Students, 46 Features', color_dataset, fs_title)
create_arrow(ax, 5, y_pos-0.3, 5, y_pos-1.2)

# 2. FEATURE ENGINEERING & PREPROCESSING
y_pos = 21.5
create_box(ax, 5, y_pos, 8, 1.2, '', color_process, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.5, 'Feature Engineering & Preprocessing', fs_title, 'bold')

# Individual preprocessing steps
steps_y = y_pos - 0.2
create_box(ax, 1.5, steps_y, 1.8, 0.5, '12 Engineered\nFeatures', color_process, fs_small)
create_box(ax, 3.5, steps_y, 1.8, 0.5, 'Categorical\nEncoding', color_process, fs_small)
create_box(ax, 5.5, steps_y, 1.8, 0.5, 'Z-Score\nNormalization', color_process, fs_small)
create_box(ax, 7.5, steps_y, 1.8, 0.5, 'Correlation\nFiltering', color_process, fs_small)

create_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)

# 3. FEATURE RANKING METHODS
y_pos = 19.5
create_box(ax, 5, y_pos, 8, 1.2, '', color_ensemble, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.5, 'Ensemble of Feature Ranking Methods', fs_title, 'bold')

ranking_y = y_pos - 0.2
create_box(ax, 1.2, ranking_y, 1.5, 0.5, 'Information\nGain', color_ensemble, fs_small)
create_box(ax, 2.8, ranking_y, 1.5, 0.5, 'Gini\nImportance', color_ensemble, fs_small)
create_box(ax, 4.4, ranking_y, 1.5, 0.5, 'Gain\nRatio', color_ensemble, fs_small)
create_box(ax, 6.0, ranking_y, 1.5, 0.5, 'Mutual\nInformation', color_ensemble, fs_small)
create_box(ax, 7.6, ranking_y, 1.5, 0.5, 'ANOVA\nF-statistic', color_ensemble, fs_small)

create_arrow(ax, 5, y_pos-0.7, 5, y_pos-1.5)

# 4. RANKED FEATURES
y_pos = 17.5
create_box(ax, 5, y_pos, 2.5, 0.6, 'Top 46 Features Selected', color_dataset, fs_text)
create_arrow(ax, 5, y_pos-0.3, 5, y_pos-1.2)

# 5. DATA PARTITIONING
y_pos = 15.8
create_box(ax, 5, y_pos, 7, 1.0, '', color_process, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.4, 'Stratified Data Partitioning', fs_title, 'bold')

partition_y = y_pos - 0.2
create_box(ax, 2.0, partition_y, 1.8, 0.5, 'Training\n80% (3,539)', color_process, fs_small)
create_box(ax, 5.0, partition_y, 1.8, 0.5, 'Validation\n10% (442)', color_process, fs_small)
create_box(ax, 8.0, partition_y, 1.8, 0.5, 'Test\n10% (443)', color_process, fs_small)

create_arrow(ax, 5, y_pos-0.6, 5, y_pos-1.4)

# 6. HYPERPARAMETER OPTIMIZATION LOOP
y_pos = 13.5
# Large box for the optimization loop
create_box(ax, 5, y_pos, 8.5, 4.2, '', '#FFF8DC', fs_text, alpha=0.5, style='box')
create_text(ax, 5, y_pos+1.9, 'Hyperparameter Optimization & Model Training', fs_title, 'bold')

# Grid Search Configuration
grid_y = y_pos + 1.3
create_box(ax, 5, grid_y, 7.5, 0.7, '', color_decision, fs_small)
create_text(ax, 5, grid_y+0.15, 'Grid Search: LR ∈ {0.0001, 0.001, 0.01} × BS ∈ {16, 32, 64} × DR ∈ {0.1, 0.2, 0.3}', fs_small, 'normal')
create_text(ax, 5, grid_y-0.15, '1,728 Total Configurations Evaluated', fs_small-1, 'normal')

# Three model architectures
models_y = y_pos + 0.3
create_box(ax, 1.8, models_y, 2.2, 1.2, 'PPN\n(Performance\nPrediction Network)\n\n3-Class\n128→64→32→3', color_process, fs_small)
create_box(ax, 5.0, models_y, 2.2, 1.2, 'DPN-A\n(Dropout Prediction\nwith Attention)\n\nBinary + Attention\n64→Attn→32→16→1', color_process, fs_small)
create_box(ax, 8.2, models_y, 2.2, 1.2, 'HMTL\n(Hybrid Multi-Task\nLearning)\n\nDual-Task\nShared→Task Heads', color_process, fs_small)

# Training parameters
train_y = y_pos - 0.8
create_text(ax, 5, train_y, 'Adam Optimizer | Early Stopping (patience=20) | ReduceLROnPlateau', fs_small-1, 'normal')

# Validation feedback
create_arrow(ax, 8.5, models_y-0.6, 8.5, grid_y-0.4, style='->', color='red', linewidth=1.5)
create_text(ax, 9.2, models_y, 'Validation\nFeedback', fs_small-1, 'normal', 'red')

create_arrow(ax, 5, y_pos-2.1, 5, y_pos-3.0)

# 7. BEST MODEL SELECTION
y_pos = 9.8
create_box(ax, 5, y_pos, 3.5, 0.7, 'Select Best Configuration\nPer Model', color_decision, fs_text)
create_arrow(ax, 5, y_pos-0.35, 5, y_pos-1.0)

# 8. EVALUATION & CROSS-VALIDATION
y_pos = 8.0
create_box(ax, 5, y_pos, 8, 2.0, '', color_process, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.8, 'Comprehensive Evaluation & Cross-Validation', fs_title, 'bold')

eval_y1 = y_pos + 0.2
create_box(ax, 2.0, eval_y1, 2.0, 0.6, '10-Fold\nCross-Validation', color_process, fs_small)
create_box(ax, 5.0, eval_y1, 2.0, 0.6, 'Statistical\nSignificance Tests', color_process, fs_small)
create_box(ax, 8.0, eval_y1, 2.0, 0.6, 'SHAP Feature\nImportance', color_process, fs_small)

eval_y2 = y_pos - 0.5
create_box(ax, 2.0, eval_y2, 2.0, 0.6, 'Accuracy, F1,\nPrecision, Recall', color_process, fs_small)
create_box(ax, 5.0, eval_y2, 2.0, 0.6, 'AUC-ROC\nAUC-PR, MCC', color_process, fs_small)
create_box(ax, 8.0, eval_y2, 2.0, 0.6, 'Confusion Matrix\nCalibration', color_process, fs_small)

create_arrow(ax, 5, y_pos-1.0, 5, y_pos-1.8)

# 9. BEST PERFORMING MODELS
y_pos = 5.5
create_box(ax, 5, y_pos, 7, 0.9, '', color_ensemble, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.3, 'Best Performing Models Identified', fs_title, 'bold')

best_y = y_pos - 0.2
create_box(ax, 2.5, best_y, 2.2, 0.5, 'PPN: 76.4%\nF1-Macro: 0.688', color_ensemble, fs_small)
create_box(ax, 5.5, best_y, 2.8, 0.5, 'DPN-A: 87.05%\nAUC-ROC: 0.910', color_ensemble, fs_small)
create_box(ax, 8.0, best_y, 1.8, 0.5, 'HMTL: Multi-Task\nAnalysis', color_ensemble, fs_small)

create_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)

# 10. INTERPRETABILITY ANALYSIS
y_pos = 3.5
create_box(ax, 5, y_pos, 6.5, 0.9, '', color_decision, fs_text, alpha=0.3)
create_text(ax, 5, y_pos+0.3, 'Interpretability & Theoretical Validation', fs_title, 'bold')

interp_y = y_pos - 0.2
create_box(ax, 2.2, interp_y, 2.3, 0.5, 'Attention Weights:\nTinto (68.2%)\nBean (31.8%)', color_decision, fs_small)
create_box(ax, 5.2, interp_y, 2.3, 0.5, 'Top Features:\nSemester Grades\nTuition Status', color_decision, fs_small)
create_box(ax, 8.0, interp_y, 2.0, 0.5, 'SHAP Analysis\nAll Models', color_decision, fs_small)

create_arrow(ax, 5, y_pos-0.5, 5, y_pos-1.3)

# 11. LLM INTEGRATION
y_pos = 1.5
create_box(ax, 5, y_pos, 5.5, 0.9, '', '#E6F3FF', fs_text, alpha=0.5)
create_text(ax, 5, y_pos+0.3, 'LLM-Powered Intervention Recommendations', fs_title, 'bold')

llm_y = y_pos - 0.2
create_box(ax, 2.5, llm_y, 2.5, 0.5, 'GPT-4 Integration\n92% Relevance', '#E6F3FF', fs_small)
create_box(ax, 5.5, llm_y, 2.0, 0.5, 'Risk Profile\nGeneration', '#E6F3FF', fs_small)
create_box(ax, 7.8, llm_y, 2.0, 0.5, 'Personalized\nInterventions', '#E6F3FF', fs_small)

# Add side annotations
create_text(ax, 0.3, 21.5, 'Phase 1-2', fs_small, 'bold', 'darkblue')
create_text(ax, 0.3, 19.5, 'Phase 3', fs_small, 'bold', 'darkblue')
create_text(ax, 0.3, 15.8, 'Phase 4', fs_small, 'bold', 'darkblue')
create_text(ax, 0.3, 13.5, 'Phase 5', fs_small, 'bold', 'darkblue')
create_text(ax, 0.3, 8.0, 'Phase 6-7', fs_small, 'bold', 'darkblue')
create_text(ax, 0.3, 3.5, 'Phase 8', fs_small, 'bold', 'darkblue')

# Add title at bottom
fig.text(0.5, 0.02, 'FIGURE: Comprehensive Research Methodology Flowchart', 
         ha='center', fontsize=12, weight='bold')
fig.text(0.5, 0.005, 'Complete workflow from data preprocessing to LLM-powered intervention generation', 
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('methodology_flowchart_advanced.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Advanced methodology flowchart saved as 'methodology_flowchart_advanced.png'")
plt.close()

# ============================================================================
# CREATE SIMPLIFIED VERSION AS WELL
# ============================================================================

fig2, ax2 = plt.subplots(1, 1, figsize=(12, 14))
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 18)
ax2.axis('off')

# Simplified flowchart
y = 17
create_box(ax2, 5, y, 2.5, 0.7, 'Dataset\n(4,424 students)', color_dataset, fs_title)
create_arrow(ax2, 5, y-0.35, 5, y-1.0)

y = 15.5
create_box(ax2, 5, y, 6, 1.0, 'Data Preprocessing & Feature Engineering\n46 Features | Z-Score Normalization | Stratified Split', color_process, fs_text)
create_arrow(ax2, 5, y-0.5, 5, y-1.2)

y = 13.8
create_box(ax2, 5, y, 7, 0.9, 'Feature Ranking: Info Gain | Gini | Gain Ratio | Mutual Info | ANOVA', color_ensemble, fs_text)
create_arrow(ax2, 5, y-0.45, 5, y-1.1)

y = 12.2
create_box(ax2, 5, y, 8, 2.5, '', '#FFF8DC', fs_text, alpha=0.5)
create_text(ax2, 5, y+1.0, 'Model Training & Hyperparameter Optimization', fs_title, 'bold')
create_box(ax2, 2.2, y+0.2, 2.0, 0.8, 'PPN\n76.4%\n3-Class', color_process, fs_small)
create_box(ax2, 5.0, y+0.2, 2.0, 0.8, 'DPN-A\n87.05%\nBinary+Attn', color_process, fs_small)
create_box(ax2, 7.8, y+0.2, 2.0, 0.8, 'HMTL\nMulti-Task', color_process, fs_small)
create_text(ax2, 5, y-0.8, '1,728 Configurations | Adam | Early Stopping', fs_small)
create_arrow(ax2, 5, y-1.25, 5, y-1.9)

y = 9.8
create_box(ax2, 5, y, 7, 1.5, '', color_process, fs_text, alpha=0.3)
create_text(ax2, 5, y+0.6, 'Evaluation & Cross-Validation', fs_title, 'bold')
create_box(ax2, 2.5, y-0.1, 2.2, 0.6, '10-Fold CV\n±1.8% StdDev', color_process, fs_small)
create_box(ax2, 5.5, y-0.1, 2.2, 0.6, 'SHAP Analysis\nFeature Importance', color_process, fs_small)
create_box(ax2, 7.8, y-0.1, 1.8, 0.6, 'Statistical\nTests', color_process, fs_small)
create_arrow(ax2, 5, y-0.75, 5, y-1.4)

y = 7.9
create_box(ax2, 5, y, 5.5, 0.9, '', color_decision, fs_text, alpha=0.3)
create_text(ax2, 5, y+0.3, 'Interpretability Analysis', fs_title, 'bold')
create_text(ax2, 5, y-0.1, 'Attention: Tinto (68.2%) + Bean (31.8%)\nTop: Semester Grades, Tuition Status', fs_small)
create_arrow(ax2, 5, y-0.5, 5, y-1.2)

y = 6.2
create_box(ax2, 5, y, 6, 0.9, '', '#E6F3FF', fs_text, alpha=0.5)
create_text(ax2, 5, y+0.3, 'LLM Integration (GPT-4)', fs_title, 'bold')
create_text(ax2, 5, y-0.1, '92% Relevance | Personalized Intervention Recommendations', fs_small)
create_arrow(ax2, 5, y-0.5, 5, y-1.2)

y = 4.5
create_box(ax2, 5, y, 5, 1.2, '', color_ensemble, fs_text, alpha=0.3)
create_text(ax2, 5, y+0.4, 'Final Results', fs_title, 'bold')
create_text(ax2, 5, y-0.0, 'DPN-A: 87.05% Accuracy | 0.910 AUC-ROC', fs_text, 'bold')
create_text(ax2, 5, y-0.3, 'PPN: 76.4% Accuracy | F1-Macro: 0.688', fs_small)

fig2.text(0.5, 0.02, 'FIGURE: Simplified Research Methodology Flowchart', 
         ha='center', fontsize=12, weight='bold')

plt.tight_layout()
plt.savefig('methodology_flowchart_simple.png', dpi=300, bbox_inches='tight', facecolor='white')
print("✅ Simplified methodology flowchart saved as 'methodology_flowchart_simple.png'")
plt.close()

print("\n" + "="*60)
print("FLOWCHARTS GENERATED SUCCESSFULLY!")
print("="*60)
print("\n📁 Files created:")
print("  1. methodology_flowchart_advanced.png - Detailed flowchart (14x18 inches)")
print("  2. methodology_flowchart_simple.png - Simplified version (12x14 inches)")
print("\n✨ Both flowcharts are publication-quality (300 DPI)")
print("📋 Ready to insert into your thesis LaTeX document")
print("\n💡 To include in LaTeX, add:")
print("   \\begin{figure}[h]")
print("   \\centering")
print("   \\includegraphics[width=0.95\\textwidth]{methodology_flowchart_advanced.png}")
print("   \\caption{Comprehensive Research Methodology Flowchart}")
print("   \\end{figure}")
