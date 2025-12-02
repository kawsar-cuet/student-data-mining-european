#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Research Methodology Flowchart Generator
Creates publication-quality flowchart showing the complete research workflow
for student performance and dropout prediction study.

Author: Research Team
Date: November 30, 2025
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Configure matplotlib for high-quality output
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['figure.dpi'] = 300

def create_box(ax, x, y, width, height, text, color, style='round', text_color='black', fontsize=10, fontweight='normal'):
    """Create a styled box with text"""
    if style == 'round':
        boxstyle = mpatches.BoxStyle("Round", pad=0.05)
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            boxstyle=boxstyle, 
                            facecolor=color, 
                            edgecolor='black', 
                            linewidth=2,
                            zorder=2)
    elif style == 'diamond':
        # Create diamond shape using polygon
        vertices = np.array([
            [x, y + height/2],      # top
            [x + width/2, y],       # right
            [x, y - height/2],      # bottom
            [x - width/2, y],       # left
        ])
        box = mpatches.Polygon(vertices, 
                              facecolor=color, 
                              edgecolor='black', 
                              linewidth=2,
                              zorder=2)
    else:  # rectangle
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                            facecolor=color, 
                            edgecolor='black', 
                            linewidth=2,
                            zorder=2)
    
    ax.add_patch(box)
    
    # Add text
    if isinstance(text, list):
        # Multiple lines
        for i, line in enumerate(text):
            offset = (len(text) - 1) * 0.15 / 2
            ax.text(x, y + offset - i * 0.15, line, 
                   ha='center', va='center', 
                   fontsize=fontsize, fontweight=fontweight,
                   color=text_color, zorder=3)
    else:
        ax.text(x, y, text, 
               ha='center', va='center', 
               fontsize=fontsize, fontweight=fontweight,
               color=text_color, zorder=3,
               wrap=True)

def create_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Create an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style,
                           color='black',
                           linewidth=2,
                           mutation_scale=20,
                           zorder=1)
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.2, mid_y, label, 
               fontsize=8, style='italic',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

def generate_main_flowchart():
    """
    Generate the main research methodology flowchart
    Shows: Data Collection → Preprocessing → Feature Engineering → 
           Model Development → Training → Evaluation → Results
    """
    fig, ax = plt.subplots(figsize=(16, 22))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 24)
    ax.axis('off')
    
    # Title
    ax.text(6, 23.2, 'Research Methodology Flowchart', 
           fontsize=20, fontweight='bold', ha='center')
    ax.text(6, 22.6, 'Deep Learning + LLM for Student Performance & Dropout Prediction', 
           fontsize=13, ha='center', style='italic')
    
    # Phase 1: Data Collection
    y_pos = 21.5
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 1: Data Collection', '(Section 5.1)'],
              '#E8F4F8', fontsize=12, fontweight='bold')
    
    y_pos -= 1.0
    create_box(ax, 6, y_pos, 4.5, 1.0, 
              ['Dataset: University Student Records', 
               'N = 4,424 students (2017-2021)',
               '46 features (Academic, Financial, Demographic)'],
              '#B3E5FC', fontsize=10)
    
    create_arrow(ax, 6, y_pos - 0.6, 6, y_pos - 1.3)
    
    # Phase 2: Data Preprocessing
    y_pos -= 2.0
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 2: Data Preprocessing', '(Section 5.2)'],
              '#FFF9C4', fontsize=12, fontweight='bold')
    
    # Preprocessing steps (left and right columns with more space)
    y_pos -= 1.1
    steps_left = [
        ('Missing Value\nImputation', 2.5, y_pos),
        ('Categorical\nEncoding', 2.5, y_pos - 1.1),
        ('Feature\nNormalization', 2.5, y_pos - 2.2),
        ('Data Split\n(80-10-10)', 2.5, y_pos - 3.3)
    ]
    
    steps_right = [
        ('Outlier\nDetection', 9.5, y_pos),
        ('Feature\nEngineering', 9.5, y_pos - 1.1),
        ('Tensor\nConversion', 9.5, y_pos - 2.2),
        ('Class Balance\nCheck', 9.5, y_pos - 3.3)
    ]
    
    for text, x, y in steps_left:
        create_box(ax, x, y, 2.2, 0.8, text, '#FFF59D', fontsize=9)
        if y > y_pos - 3.3:
            create_arrow(ax, x, y - 0.5, x, y - 0.6)
    
    for text, x, y in steps_right:
        create_box(ax, x, y, 2.2, 0.8, text, '#FFF59D', fontsize=9)
        if y > y_pos - 3.3:
            create_arrow(ax, x, y - 0.5, x, y - 0.6)
    
    # Convergence arrow
    create_arrow(ax, 2.5, y_pos - 3.8, 6, y_pos - 4.6)
    create_arrow(ax, 9.5, y_pos - 3.8, 6, y_pos - 4.6)
    
    # Phase 3: Theoretical Framework
    y_pos -= 5.3
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 3: Theoretical Framework', '(Section 3)'],
              '#E1BEE7', fontsize=12, fontweight='bold')
    
    y_pos -= 1.0
    create_box(ax, 3.5, y_pos, 2.5, 0.7, 
              ["Tinto's Model", '(68% features)'],
              '#CE93D8', fontsize=10)
    create_box(ax, 8.5, y_pos, 2.5, 0.7, 
              ["Bean's Model", '(32% features)'],
              '#CE93D8', fontsize=10)
    
    create_arrow(ax, 3.5, y_pos - 0.5, 5, y_pos - 1.2)
    create_arrow(ax, 8.5, y_pos - 0.5, 7, y_pos - 1.2)
    
    # Phase 4: Model Development (DIAMOND - Decision Point)
    y_pos -= 1.8
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 4: Model Development', '(Section 6)'],
              '#C8E6C9', fontsize=12, fontweight='bold')
    
    y_pos -= 1.2
    create_box(ax, 6, y_pos, 3, 1.0, 
              'Research Objectives', 
              '#FFCCBC', style='diamond', fontsize=11, fontweight='bold')
    
    # Branch to three models with more spacing
    y_pos -= 1.6
    
    # Model 1: PPN (Left)
    create_arrow(ax, 5.2, y_pos + 0.9, 2.5, y_pos + 0.3)
    create_box(ax, 2.5, y_pos, 2.4, 1.4, 
              ['Model 1: PPN', 
               'Performance Prediction',
               '3-class output',
               'Architecture:',
               '46→128→64→32→3'],
              '#A5D6A7', fontsize=9)
    
    # Model 2: DPN-A (Center)
    create_arrow(ax, 6, y_pos + 0.5, 6, y_pos + 0.9)
    create_box(ax, 6, y_pos, 2.4, 1.4, 
              ['Model 2: DPN-A', 
               'Dropout Prediction',
               'Binary output',
               'Architecture:',
               '46→64→Attn→32→16→1'],
              '#FFE082', fontsize=9)
    
    # Model 3: HMTL (Right)
    create_arrow(ax, 6.8, y_pos + 0.9, 9.5, y_pos + 0.3)
    create_box(ax, 9.5, y_pos, 2.4, 1.4, 
              ['Model 3: HMTL', 
               'Multi-Task Learning',
               'Dual outputs',
               'Architecture:',
               'Shared→Dual Heads'],
              '#CE93D8', fontsize=9)
    
    # Convergence to training
    create_arrow(ax, 2.5, y_pos - 0.8, 6, y_pos - 1.7)
    create_arrow(ax, 6, y_pos - 0.8, 6, y_pos - 1.7)
    create_arrow(ax, 9.5, y_pos - 0.8, 6, y_pos - 1.7)
    
    # Phase 5: Training & Optimization
    y_pos -= 2.5
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 5: Training & Optimization', '(Section 6.4)'],
              '#FFCCBC', fontsize=12, fontweight='bold')
    
    y_pos -= 1.0
    create_box(ax, 6, y_pos, 5.5, 1.2, 
              ['Optimizer: Adam (lr=0.001, β₁=0.9, β₂=0.999)',
               'Loss: CrossEntropy (PPN), BCE+Attention (DPN-A), MTL (HMTL)',
               'Batch Size: 32 | Epochs: 100 (early stopping)',
               'LR Scheduler: ReduceLROnPlateau | 10-Fold CV'],
              '#FFE0B2', fontsize=9)
    
    create_arrow(ax, 6, y_pos - 0.7, 6, y_pos - 1.4)
    
    # Phase 6: Evaluation
    y_pos -= 2.1
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 6: Model Evaluation', '(Section 7)'],
              '#B2DFDB', fontsize=12, fontweight='bold')
    
    # Evaluation metrics (grid layout with better spacing)
    y_pos -= 1.1
    metrics = [
        ('Accuracy', 2.5, y_pos),
        ('Precision', 6, y_pos),
        ('Recall', 9.5, y_pos),
        ('F1-Score', 2.5, y_pos - 1.0),
        ('AUC-ROC', 6, y_pos - 1.0),
        ('AUC-PR', 9.5, y_pos - 1.0),
        ('Confusion\nMatrix', 4.25, y_pos - 2.0),
        ('10-Fold CV', 7.75, y_pos - 2.0)
    ]
    
    for text, x, y in metrics:
        create_box(ax, x, y, 2.0, 0.7, text, '#80CBC4', fontsize=9)
    
    # Convergence to results
    create_arrow(ax, 6, y_pos - 2.6, 6, y_pos - 3.4)
    
    # Phase 7: Results & Analysis
    y_pos -= 4.1
    create_box(ax, 6, y_pos, 5, 0.7, 
              ['Phase 7: Results & Analysis', '(Section 10)'],
              '#F8BBD0', fontsize=12, fontweight='bold')
    
    # Results boxes with better spacing
    y_pos -= 1.1
    create_box(ax, 3, y_pos, 2.6, 1.0, 
              ['PPN Results',
               'Accuracy: 76.4%',
               'Macro F1: 0.764'],
              '#F48FB1', fontsize=9)
    
    create_box(ax, 6, y_pos, 2.6, 1.0, 
              ['DPN-A Results',
               'Accuracy: 87.05%',
               'AUC-ROC: 0.910'],
              '#F48FB1', fontsize=9)
    
    create_box(ax, 9, y_pos, 2.6, 1.0, 
              ['HMTL Results',
               'Perf: 76.4%',
               'Drop: 67.9%'],
              '#F48FB1', fontsize=9)
    
    # Final convergence
    create_arrow(ax, 3, y_pos - 0.6, 6, y_pos - 1.4)
    create_arrow(ax, 6, y_pos - 0.6, 6, y_pos - 1.4)
    create_arrow(ax, 9, y_pos - 0.6, 6, y_pos - 1.4)
    
    # Phase 8: LLM Integration
    y_pos -= 2.1
    create_box(ax, 6, y_pos, 5.5, 1.0, 
              ['Phase 8: LLM Integration & Recommendations',
               'GPT-4 generates personalized interventions',
               'Rule-based + AI-powered student support'],
              '#E1BEE7', fontsize=10, fontweight='bold')
    
    create_arrow(ax, 6, y_pos - 0.6, 6, y_pos - 1.3)
    
    # Phase 9: Deployment
    y_pos -= 2.0
    create_box(ax, 6, y_pos, 5.5, 1.0, 
              ['Phase 9: Deployment & Early Warning System',
               'Institutional integration with advisor dashboard',
               'Real-time risk monitoring & intervention tracking'],
              '#DCEDC8', fontsize=10, fontweight='bold')
    
    # Add legend with better spacing
    legend_y = 1.2
    ax.text(1.5, legend_y + 0.2, 'Legend:', fontsize=10, fontweight='bold')
    legend_items = [
        ('#E8F4F8', 'Data Phase'),
        ('#FFF9C4', 'Preprocessing'),
        ('#E1BEE7', 'Theory/LLM'),
        ('#C8E6C9', 'Models'),
        ('#FFCCBC', 'Training'),
        ('#B2DFDB', 'Evaluation'),
        ('#F8BBD0', 'Results'),
        ('#DCEDC8', 'Deployment')
    ]
    
    x_offset = 1.5
    for i, (color, label) in enumerate(legend_items):
        if i == 4:
            x_offset = 1.5
            legend_y -= 0.35
        box = mpatches.Rectangle((x_offset, legend_y - 0.15), 0.35, 0.18, 
                                 facecolor=color, edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x_offset + 0.5, legend_y - 0.06, label, fontsize=8, va='center')
        x_offset += 2.0
    
    plt.tight_layout()
    
    # Save
    output_path = 'outputs/figures_journal/methodology_flowchart_main'
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', format='png')
    print("[OK] Main Methodology Flowchart saved")
    plt.close()


def generate_research_objectives_diagram():
    """
    Generate detailed research objectives breakdown diagram
    Shows: Dual objectives (performance + dropout) with sub-tasks + LLM enhancement
    """
    fig, ax = plt.subplots(figsize=(16, 11))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Research Objectives Breakdown', 
           fontsize=16, fontweight='bold', ha='center')
    
    # Main research question
    create_box(ax, 8, 8.5, 10, 0.8, 
              ['Main Research Question:',
               'Can deep learning + LLM predict student outcomes and provide actionable interventions?'],
              '#E3F2FD', fontsize=10, fontweight='bold')
    
    # Split into two objectives
    create_arrow(ax, 5.5, 8.0, 4, 7.5)
    create_arrow(ax, 10.5, 8.0, 12, 7.5)
    
    # Objective 1: Performance Prediction (LEFT)
    create_box(ax, 4, 7.2, 5, 0.7, 
              'Objective 1: Student Performance Prediction',
              '#C5CAE9', fontsize=11, fontweight='bold')
    
    y = 6.3
    create_box(ax, 4, y, 4.5, 0.6, 
              '3-Class Classification (Low/Medium/High)',
              '#E8EAF6', fontsize=9)
    
    create_arrow(ax, 4, y - 0.4, 4, y - 0.8)
    
    # Sub-tasks for Objective 1
    y -= 1.2
    subtasks_1 = [
        'Task 1.1: Baseline Model (PPN)',
        'Task 1.2: Feature Importance Analysis',
        'Task 1.3: Class Imbalance Handling',
        'Task 1.4: Confusion Matrix Analysis',
        'Task 1.5: Multi-metric Evaluation'
    ]
    
    for i, task in enumerate(subtasks_1):
        create_box(ax, 4, y - i*0.7, 4.2, 0.55, task, '#D1C4E9', fontsize=8)
        if i < len(subtasks_1) - 1:
            create_arrow(ax, 4, y - i*0.7 - 0.32, 4, y - (i+1)*0.7 + 0.32)
    
    # Results for Objective 1
    y -= len(subtasks_1) * 0.7 + 0.2
    create_box(ax, 4, y, 4, 0.8, 
              ['Results (PPN):',
               'Accuracy: 76.4% | Macro F1: 0.764',
               'Interpretable predictions with attention'],
              '#B39DDB', fontsize=9, fontweight='bold')
    
    # Objective 2: Dropout Prediction (RIGHT)
    create_box(ax, 12, 7.2, 5, 0.7, 
              'Objective 2: Student Dropout Prediction',
              '#FFCCBC', fontsize=11, fontweight='bold')
    
    y = 6.3
    create_box(ax, 12, y, 4.5, 0.6, 
              'Binary Classification (Dropout/Continue)',
              '#FFE0B2', fontsize=9)
    
    create_arrow(ax, 12, y - 0.4, 12, y - 0.8)
    
    # Sub-tasks for Objective 2
    y -= 1.2
    subtasks_2 = [
        'Task 2.1: Attention-based Model (DPN-A)',
        'Task 2.2: Temporal Feature Engineering',
        'Task 2.3: ROC-AUC Optimization',
        'Task 2.4: Precision-Recall Analysis',
        'Task 2.5: Early Warning Threshold'
    ]
    
    for i, task in enumerate(subtasks_2):
        create_box(ax, 12, y - i*0.7, 4.2, 0.55, task, '#FFAB91', fontsize=8)
        if i < len(subtasks_2) - 1:
            create_arrow(ax, 12, y - i*0.7 - 0.32, 12, y - (i+1)*0.7 + 0.32)
    
    # Results for Objective 2
    y -= len(subtasks_2) * 0.7 + 0.2
    create_box(ax, 12, y, 4, 0.8, 
              ['Results (DPN-A):',
               'Accuracy: 87.05% | AUC-ROC: 0.910',
               'Superior binary classification performance'],
              '#FF8A65', fontsize=9, fontweight='bold')
    
    # Convergence to integrated analysis
    create_arrow(ax, 4, 0.8, 8, 1.5)
    create_arrow(ax, 12, 0.8, 8, 1.5)
    
    # Bottom: Integrated Analysis (HMTL)
    create_box(ax, 8, 1.2, 6, 1.0, 
              ['Integrated Analysis: Multi-Task Learning (HMTL)',
               'Objective: Compare single-task vs multi-task learning',
               'Finding: Task interference observed (67.9% dropout in MTL vs 87.05% in DPN-A)',
               'Conclusion: Dedicated models outperform multi-task approach for this dataset'],
              '#E1BEE7', fontsize=9)
    
    create_arrow(ax, 8, 0.6, 8, 0.2)
    
    # LLM Enhancement Layer
    create_box(ax, 8, 0.3, 7, 0.7, 
              ['LLM Enhancement: GPT-4 Recommendation Engine',
               'Converts predictions into actionable interventions',
               'Personalizes support based on student profile + risk factors'],
              '#D1C4E9', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = 'outputs/figures_journal/methodology_flowchart_objectives'
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', format='png')
    print("[OK] Research Objectives Diagram saved")
    plt.close()


def generate_data_flow_diagram():
    """
    Generate data flow diagram showing the pipeline from raw data to predictions
    More detailed than the system architecture - focuses on data transformations
    """
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(7, 11.5, 'Data Processing & Model Pipeline with LLM Integration', 
           fontsize=16, fontweight='bold', ha='center')
    
    y = 10.5
    
    # Stage 1: Raw Data
    create_box(ax, 7, y, 4, 0.8, 
              ['Stage 1: Raw Dataset',
               'N=4,424 students | 46 features',
               '3 targets (GPA, Performance, Dropout)'],
              '#FFEBEE', fontsize=9, fontweight='bold')
    
    create_arrow(ax, 7, y - 0.5, 7, y - 1.0)
    
    # Stage 2: Data Cleaning
    y -= 1.5
    create_box(ax, 7, y, 5, 0.6, 
              'Stage 2: Data Cleaning & Quality Control',
              '#FFCDD2', fontsize=10, fontweight='bold')
    
    y -= 0.5
    cleaning_steps = [
        ('Missing Values\n(Median/Mode)', 2.5, y),
        ('Outlier Detection\n(IQR method)', 7, y),
        ('Duplicate Removal\n(Zero duplicates)', 11.5, y)
    ]
    
    for text, x, yy in cleaning_steps:
        create_box(ax, x, yy, 2.5, 0.6, text, '#EF9A9A', fontsize=8)
    
    create_arrow(ax, 7, y - 0.4, 7, y - 0.9)
    
    # Stage 3: Feature Engineering
    y -= 1.4
    create_box(ax, 7, y, 5, 0.6, 
              'Stage 3: Feature Engineering & Encoding',
              '#E1BEE7', fontsize=10, fontweight='bold')
    
    y -= 0.5
    # Left: Numerical features
    create_box(ax, 3, y, 3, 1.2, 
              ['Numerical Features (12):',
               '• Academic: GPA, Credits, Attendance',
               '• Financial: Scholarship, Fees',
               '• Demographic: Age',
               'Transform: StandardScaler'],
              '#CE93D8', fontsize=8)
    
    # Right: Categorical features
    create_box(ax, 11, y, 3, 1.2, 
              ['Categorical Features (34):',
               '• Academic: Major, Course mode',
               '• Financial: Debtor status',
               '• Demographic: Gender, Nationality',
               'Transform: One-Hot Encoding'],
              '#CE93D8', fontsize=8)
    
    create_arrow(ax, 3, y - 0.7, 7, y - 1.3)
    create_arrow(ax, 11, y - 0.7, 7, y - 1.3)
    
    # Stage 4: Feature Selection & Theory Mapping
    y -= 1.9
    create_box(ax, 7, y, 5, 0.6, 
              'Stage 4: Theoretical Framework Mapping',
              '#C5CAE9', fontsize=10, fontweight='bold')
    
    y -= 0.5
    create_box(ax, 3.5, y, 3.5, 0.8, 
              ["Tinto's Integration Model",
               'Social & Academic Integration',
               '31 features (68%)'],
              '#9FA8DA', fontsize=8)
    
    create_box(ax, 10.5, y, 3.5, 0.8, 
              ["Bean's Student Attrition Model",
               'Environmental & Organizational Fit',
               '15 features (32%)'],
              '#9FA8DA', fontsize=8)
    
    create_arrow(ax, 7, y - 0.5, 7, y - 1.0)
    
    # Stage 5: Data Splitting
    y -= 1.5
    create_box(ax, 7, y, 5, 0.6, 
              'Stage 5: Dataset Partitioning',
              '#C8E6C9', fontsize=10, fontweight='bold')
    
    y -= 0.5
    splits = [
        ('Training Set\n80% (3,539)', 3, y),
        ('Validation Set\n10% (442)', 7, y),
        ('Test Set\n10% (443)', 11, y)
    ]
    
    for text, x, yy in splits:
        create_box(ax, x, yy, 2.5, 0.6, text, '#A5D6A7', fontsize=8)
    
    create_arrow(ax, 7, y - 0.4, 7, y - 0.9)
    
    # Stage 6: Tensor Conversion
    y -= 1.4
    create_box(ax, 7, y, 5, 0.8, 
              ['Stage 6: PyTorch Tensor Conversion',
               'X: torch.FloatTensor [N, 46]',
               'y_perf: torch.LongTensor [N] (3 classes)',
               'y_drop: torch.FloatTensor [N] (binary)'],
              '#FFF9C4', fontsize=8, fontweight='bold')
    
    create_arrow(ax, 7, y - 0.5, 7, y - 1.0)
    
    # Stage 7: Model Training (3 parallel paths)
    y -= 1.5
    create_box(ax, 7, y, 5, 0.6, 
              'Stage 7: Model Training & Optimization',
              '#FFCCBC', fontsize=10, fontweight='bold')
    
    y -= 0.5
    models = [
        ('PPN\n3-class', 3, y),
        ('DPN-A\nBinary', 7, y),
        ('HMTL\nDual-task', 11, y)
    ]
    
    for text, x, yy in models:
        create_box(ax, x, yy, 2.2, 0.6, text, '#FFAB91', fontsize=8, fontweight='bold')
    
    # Convergence
    create_arrow(ax, 3, y - 0.4, 7, y - 0.9)
    create_arrow(ax, 7, y - 0.4, 7, y - 0.9)
    create_arrow(ax, 11, y - 0.4, 7, y - 0.9)
    
    # Stage 8: Evaluation
    y -= 1.4
    create_box(ax, 7, y, 5, 0.8, 
              ['Stage 8: Model Evaluation',
               '10-Fold Cross-Validation',
               'Metrics: Accuracy, F1, Precision, Recall, AUC-ROC, AUC-PR'],
              '#B2DFDB', fontsize=9, fontweight='bold')
    
    create_arrow(ax, 7, y - 0.5, 7, y - 1.0)
    
    # Stage 9: LLM Integration
    y -= 1.5
    create_box(ax, 7, y, 5, 0.8, 
              ['Stage 9: LLM-Based Recommendations (GPT-4)',
               'Student profile + predictions → GPT-4 API',
               'Personalized interventions: Academic, behavioral, support',
               'Rule-based fallback if API unavailable'],
              '#E1BEE7', fontsize=8, fontweight='bold')
    
    create_arrow(ax, 7, y - 0.5, 7, y - 1.0)
    
    # Stage 10: Final Output
    y -= 1.5
    create_box(ax, 7, y, 6, 0.8, 
              ['Stage 10: Early Warning System Deployment',
               'Advisor Dashboard: Risk scores + LLM recommendations',
               'Automated alerts for high-risk students',
               'Intervention tracking & outcome monitoring'],
              '#DCEDC8', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    
    # Save
    output_path = 'outputs/figures_journal/methodology_flowchart_dataflow'
    plt.savefig(f'{output_path}.pdf', dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight', format='png')
    print("[OK] Data Flow Diagram saved")
    plt.close()


if __name__ == "__main__":
    print("=" * 70)
    print("RESEARCH METHODOLOGY FLOWCHARTS GENERATION")
    print("=" * 70)
    print("\nGenerating 3 complementary methodology diagrams...")
    print()
    
    # Generate all three diagrams
    print("Creating Diagram 1: Main Research Methodology Flowchart...")
    generate_main_flowchart()
    
    print("Creating Diagram 2: Research Objectives Breakdown...")
    generate_research_objectives_diagram()
    
    print("Creating Diagram 3: Data Processing & Model Pipeline...")
    generate_data_flow_diagram()
    
    print()
    print("=" * 70)
    print("METHODOLOGY FLOWCHARTS COMPLETE")
    print("=" * 70)
    print("\nAll diagrams saved to: outputs/figures_journal/")
    print("\nGenerated Files:")
    print("  1. methodology_flowchart_main.pdf/png")
    print("     - Complete research workflow (9 phases with LLM)")
    print("  2. methodology_flowchart_objectives.pdf/png")
    print("     - Dual objectives + LLM enhancement layer")
    print("  3. methodology_flowchart_dataflow.pdf/png")
    print("     - Detailed data pipeline (10 stages with GPT-4)")
    print("\nRecommended Usage:")
    print("  • Main flowchart -> Section 4 (Methodology Overview)")
    print("  • Objectives diagram -> Section 5 (Research Design)")
    print("  • Data flow diagram -> Section 5.2 (Data Preprocessing)")
    print("=" * 70)
