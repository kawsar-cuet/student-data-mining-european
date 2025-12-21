"""
Generate AFSA-Enhanced Methodology Flowchart
Professional flowchart showing Adaptive Feature Selection Algorithm integration
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np

# Set publication quality
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9


def create_box(ax, x, y, width, height, text, color, rounded=True, bold_title=True):
    """Create a colored box with text"""
    if rounded:
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.05",
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
    else:
        box = patches.Rectangle(
            (x, y), width, height,
            facecolor=color,
            edgecolor='black',
            linewidth=1.5
        )
    ax.add_patch(box)
    
    # Add text
    if bold_title:
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                wrap=True)
    else:
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center',
                fontsize=8,
                wrap=True)


def create_arrow(ax, x1, y1, x2, y2, style='->', color='black', width=2):
    """Create directional arrow"""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=width,
        mutation_scale=20
    )
    ax.add_patch(arrow)


def create_text(ax, x, y, text, fontsize=8, bold=False, color='black'):
    """Create text annotation"""
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, fontsize=fontsize, fontweight=weight,
            ha='center', va='center', color=color)


def generate_afsa_enhanced_flowchart():
    """
    Generate AFSA-Enhanced Methodology Flowchart
    Shows complete workflow with AFSA integration
    """
    
    fig, ax = plt.subplots(figsize=(16, 20))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 25)
    ax.axis('off')
    
    # Colors matching reference style
    color_dataset = '#E8E8E8'  # Light gray
    color_processing = '#B8D4E8'  # Light blue
    color_afsa = '#FFD700'  # Gold (highlight AFSA)
    color_ensemble = '#C8E6C9'  # Light green
    color_decision = '#FFE0B2'  # Light orange
    color_training = '#D1C4E9'  # Light purple
    color_results = '#FFCDD2'  # Light red
    
    y_pos = 24  # Start from top
    x_center = 5
    box_width = 3.5
    box_height = 0.8
    small_box_height = 0.6
    
    # Title
    create_text(ax, x_center, y_pos, 
                "AFSA-ENHANCED METHODOLOGY FLOWCHART", 
                fontsize=14, bold=True)
    create_text(ax, x_center, y_pos - 0.5,
                "Adaptive Feature Selection for Student Dropout Prediction",
                fontsize=10, bold=False)
    
    y_pos -= 1.5
    
    # ==== PHASE 1: DATASET ====
    create_text(ax, 0.5, y_pos, "Phase 1", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, box_height,
               "Dataset Loading", color_dataset)
    create_text(ax, x_center, y_pos - 1.0,
                "4,424 students | 46 features | 3 classes",
                fontsize=7)
    
    y_pos -= 1.8
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # ==== PHASE 2: FEATURE ENGINEERING ====
    create_text(ax, 0.5, y_pos, "Phase 2", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, box_height,
               "Feature Engineering", color_processing)
    create_text(ax, x_center, y_pos - 1.0,
                "12 engineered features | Categorical encoding",
                fontsize=7)
    
    y_pos -= 1.8
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # ==== PHASE 3: ENSEMBLE RANKING (Part of AFSA) ====
    create_text(ax, 0.5, y_pos, "Phase 3", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 1.2,
               "AFSA Phase 1:\nEnsemble Feature Ranking", color_afsa)
    
    # Nested boxes for 5 methods
    nested_y = y_pos - 1.8
    nested_x_start = x_center - 1.5
    nested_width = 0.9
    nested_height = 0.4
    
    methods = ["Info\nGain", "Gini", "Mutual\nInfo", "ANOVA", "Gain\nRatio"]
    for i, method in enumerate(methods):
        x_offset = nested_x_start + (i % 3) * 1.0
        y_offset = nested_y - (i // 3) * 0.6
        create_box(ax, x_offset, y_offset, nested_width, nested_height,
                   method, color_ensemble, rounded=False, bold_title=False)
    
    create_text(ax, x_center, nested_y - 1.0,
                "Ensemble: Average rank across 5 methods",
                fontsize=7, bold=True)
    
    y_pos = nested_y - 1.5
    create_arrow(ax, x_center, y_pos + 0.3, x_center, y_pos)
    
    # ==== PHASE 4: AFSA POPULATION INITIALIZATION ====
    create_text(ax, 0.5, y_pos, "Phase 4", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 1.0,
               "AFSA Phase 2:\nPopulation Initialization", color_afsa)
    create_text(ax, x_center, y_pos - 1.5,
                "20 feature subsets (fish population)\n"
                "Biased by ensemble ranking",
                fontsize=7)
    
    y_pos -= 2.2
    create_arrow(ax, x_center, y_pos + 0.5, x_center, y_pos)
    
    # ==== PHASE 5: DATA PARTITIONING ====
    create_text(ax, 0.5, y_pos, "Phase 5", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 0.8,
               "Stratified Data Partitioning", color_processing)
    create_text(ax, x_center, y_pos - 1.0,
                "Train: 3,539 (80%) | Val: 442 (10%) | Test: 443 (10%)",
                fontsize=7)
    
    y_pos -= 1.8
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # ==== PHASE 6: AFSA ITERATIVE OPTIMIZATION (MAIN LOOP) ====
    create_text(ax, 0.5, y_pos - 0.5, "Phase 6", fontsize=8, bold=True)
    
    # Large optimization box
    loop_box_height = 4.5
    create_box(ax, x_center - box_width/2 - 0.3, y_pos - 0.4, 
               box_width + 0.6, loop_box_height,
               "", '#FFF9C4', rounded=True)  # Light yellow background
    
    create_text(ax, x_center, y_pos - 0.2,
                "AFSA Phase 3: Iterative Optimization",
                fontsize=10, bold=True)
    create_text(ax, x_center, y_pos - 0.6,
                "30 iterations | 3-fold CV for fitness",
                fontsize=7)
    
    # Sub-steps inside loop
    loop_y = y_pos - 1.2
    
    # Fitness Evaluation
    create_box(ax, x_center - 1.4, loop_y, 2.8, 0.6,
               "Evaluate Fitness (CV Accuracy)", color_decision, rounded=False)
    
    loop_y -= 0.9
    create_arrow(ax, x_center, loop_y + 0.3, x_center, loop_y)
    
    # Behavior Selection
    create_box(ax, x_center - 1.4, loop_y, 2.8, 0.6,
               "Fish Behaviors", color_afsa, rounded=False)
    
    # Three behaviors
    behavior_y = loop_y - 0.8
    behaviors = ["Prey\n(40%)", "Swarm\n(30%)", "Follow\n(30%)"]
    for i, behavior in enumerate(behaviors):
        x_offset = x_center - 1.2 + i * 1.2
        create_box(ax, x_offset - 0.35, behavior_y, 0.7, 0.5,
                   behavior, '#E1BEE7', rounded=False, bold_title=False)
    
    loop_y = behavior_y - 0.7
    create_arrow(ax, x_center, loop_y + 0.2, x_center, loop_y)
    
    # Update Population
    create_box(ax, x_center - 1.4, loop_y, 2.8, 0.6,
               "Update Feature Subsets", color_processing, rounded=False)
    
    loop_y -= 0.9
    create_arrow(ax, x_center, loop_y + 0.3, x_center, loop_y)
    
    # Convergence Check
    create_box(ax, x_center - 1.4, loop_y, 2.8, 0.6,
               "Max Iterations Reached?", color_decision, rounded=False)
    
    # Feedback arrow
    create_arrow(ax, x_center + 1.5, loop_y + 0.3, x_center + 2.5, loop_y + 0.3,
                 style='->', color='red', width=2)
    create_arrow(ax, x_center + 2.5, loop_y + 0.3, x_center + 2.5, y_pos - 1.5,
                 style='->', color='red', width=2)
    create_arrow(ax, x_center + 2.5, y_pos - 1.5, x_center + 1.5, y_pos - 1.5,
                 style='->', color='red', width=2)
    create_text(ax, x_center + 2.8, loop_y + 1.5, "No\n(iterate)", 
                fontsize=7, color='red', bold=True)
    
    y_pos = loop_y - 0.7
    create_arrow(ax, x_center, y_pos + 0.1, x_center, y_pos, color='green', width=2)
    create_text(ax, x_center + 0.8, y_pos + 0.3, "Yes", fontsize=7, color='green', bold=True)
    
    # ==== PHASE 7: BEST FEATURE SUBSET SELECTION ====
    create_text(ax, 0.5, y_pos, "Phase 7", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 0.8,
               "Select Optimal Feature Subset", color_decision)
    create_text(ax, x_center, y_pos - 1.0,
                "Best performing fish → Optimal features",
                fontsize=7)
    
    y_pos -= 1.6
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # ==== PHASE 8: MODEL TRAINING ====
    create_text(ax, 0.5, y_pos, "Phase 8", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 1.2,
               "Deep Learning Model Training", color_training)
    
    # Three models
    model_y = y_pos - 1.5
    models = ["PPN", "DPN-A", "HMTL"]
    for i, model in enumerate(models):
        x_offset = x_center - 1.2 + i * 1.2
        create_box(ax, x_offset - 0.35, model_y, 0.7, 0.4,
                   model, '#E1F5FE', rounded=False, bold_title=False)
    
    create_text(ax, x_center, model_y - 0.6,
                "With AFSA-selected features",
                fontsize=7, bold=True)
    
    y_pos = model_y - 1.0
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # ==== PHASE 9: EVALUATION ====
    create_text(ax, 0.5, y_pos, "Phase 9", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 1.0,
               "Comprehensive Evaluation", color_results)
    create_text(ax, x_center, y_pos - 1.4,
                "10-fold CV | SHAP | Statistical tests",
                fontsize=7)
    
    y_pos -= 1.9
    create_arrow(ax, x_center, y_pos + 0.3, x_center, y_pos)
    
    # ==== PHASE 10: RESULTS ====
    create_text(ax, 0.5, y_pos, "Phase 10", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 1.2,
               "Best Performing Models", color_results)
    create_text(ax, x_center, y_pos - 1.5,
                "DPN-A: 87.05% (0.910 AUC) | PPN: 76.4%\n"
                "AFSA: Improved accuracy + reduced features",
                fontsize=7, bold=True)
    
    y_pos -= 2.0
    create_arrow(ax, x_center, y_pos + 0.3, x_center, y_pos)
    
    # ==== PHASE 11: LLM INTEGRATION ====
    create_text(ax, 0.5, y_pos, "Phase 11", fontsize=8, bold=True)
    create_box(ax, x_center - box_width/2, y_pos - 0.4, box_width, 0.8,
               "LLM Recommendation Generation", color_processing)
    create_text(ax, x_center, y_pos - 1.0,
                "GPT-4 | 92% relevance | Personalized interventions",
                fontsize=7)
    
    # Legend
    legend_y = 1.5
    create_text(ax, x_center, legend_y + 0.5, "Legend:", fontsize=9, bold=True)
    
    legend_items = [
        ("Dataset/Processing", color_processing),
        ("AFSA Steps", color_afsa),
        ("Ensemble Methods", color_ensemble),
        ("Decision Points", color_decision),
        ("Model Training", color_training),
        ("Results", color_results)
    ]
    
    for i, (label, color) in enumerate(legend_items):
        x_offset = x_center - 2.5 + (i % 3) * 1.7
        y_offset = legend_y - (i // 3) * 0.5
        create_box(ax, x_offset, y_offset, 0.5, 0.3, "", color, rounded=False)
        create_text(ax, x_offset + 1.0, y_offset + 0.15, label, fontsize=7)
    
    plt.tight_layout()
    
    # Save
    output_path = "figures/methodology_flowchart_afsa_enhanced.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ AFSA-Enhanced flowchart saved: {output_path}")
    
    plt.close()


def generate_afsa_detailed_flowchart():
    """
    Generate detailed AFSA algorithm flowchart
    Shows internal AFSA mechanics
    """
    
    fig, ax = plt.subplots(figsize=(14, 16))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 20)
    ax.axis('off')
    
    color_start = '#90CAF9'
    color_process = '#C8E6C9'
    color_decision = '#FFE0B2'
    color_loop = '#FFF9C4'
    color_end = '#FFCDD2'
    
    y_pos = 19
    x_center = 5
    box_width = 3
    
    # Title
    create_text(ax, x_center, y_pos,
                "ADAPTIVE FEATURE SELECTION ALGORITHM (AFSA)",
                fontsize=12, bold=True)
    create_text(ax, x_center, y_pos - 0.5,
                "Detailed Algorithm Flow",
                fontsize=10)
    
    y_pos -= 1.5
    
    # Start
    create_box(ax, x_center - 1, y_pos - 0.3, 2, 0.6,
               "START", color_start, rounded=True)
    
    y_pos -= 1.2
    create_arrow(ax, x_center, y_pos + 0.3, x_center, y_pos)
    
    # Input
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "Input: X (features), y (labels)", color_start)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Ensemble Ranking
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 1.0,
               "Ensemble Feature Ranking\n"
               "5 methods: IG, Gini, MI, ANOVA, GR",
               color_process)
    
    y_pos -= 1.6
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Initialize Population
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 1.0,
               "Initialize Population\n"
               "20 fish (feature subsets)",
               color_process)
    
    y_pos -= 1.6
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Iteration counter
    create_box(ax, x_center - 1.5, y_pos - 0.3, 3, 0.6,
               "iteration = 0", color_loop)
    
    y_pos -= 1.2
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # Main loop start
    loop_start_y = y_pos
    
    # Evaluate Fitness
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "Evaluate Fitness (3-fold CV)\n"
               "for all fish", color_process)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Update Best
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "Update Best Solution\n"
               "if fitness improved", color_decision)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # For each fish
    create_box(ax, x_center - 1.5, y_pos - 0.3, 3, 0.6,
               "For each fish in population", color_loop)
    
    y_pos -= 1.2
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # Select behavior
    create_box(ax, x_center - 1.5, y_pos - 0.5, 3, 1.0,
               "Select Behavior:\n"
               "Prey (40%) | Swarm (30%) | Follow (30%)",
               color_decision)
    
    y_pos -= 1.6
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Execute behavior
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "Execute Behavior\n"
               "Generate new feature subset", color_process)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Evaluate and update
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "Evaluate New Subset\n"
               "Update if better", color_process)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # Increment iteration
    create_box(ax, x_center - 1.5, y_pos - 0.3, 3, 0.6,
               "iteration += 1", color_loop)
    
    y_pos -= 1.2
    create_arrow(ax, x_center, y_pos + 0.4, x_center, y_pos)
    
    # Convergence check
    create_box(ax, x_center - 1.5, y_pos - 0.4, 3, 0.8,
               "iteration < max_iterations?",
               color_decision)
    
    # Loop back arrow
    create_arrow(ax, x_center + 1.6, y_pos, x_center + 2.5, y_pos,
                 style='->', color='red', width=2)
    create_arrow(ax, x_center + 2.5, y_pos, x_center + 2.5, loop_start_y + 0.5,
                 style='->', color='red', width=2)
    create_arrow(ax, x_center + 2.5, loop_start_y + 0.5, x_center + 1.5, loop_start_y + 0.5,
                 style='->', color='red', width=2)
    create_text(ax, x_center + 3.0, y_pos + 2, "YES\n(iterate)", 
                fontsize=8, color='red', bold=True)
    
    y_pos -= 1.4
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos, color='green', width=2)
    create_text(ax, x_center + 1.0, y_pos + 0.4, "NO", fontsize=8, color='green', bold=True)
    
    # Output
    create_box(ax, x_center - 1.5, y_pos - 0.5, 3, 1.0,
               "Output: Best Feature Subset\n"
               "Optimal fitness achieved",
               color_end)
    
    y_pos -= 1.6
    create_arrow(ax, x_center, y_pos + 0.2, x_center, y_pos)
    
    # End
    create_box(ax, x_center - 1, y_pos - 0.3, 2, 0.6,
               "END", color_end, rounded=True)
    
    plt.tight_layout()
    
    # Save
    output_path = "figures/afsa_algorithm_detailed.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Detailed AFSA algorithm flowchart saved: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    print("="*80)
    print("  Generating AFSA-Enhanced Flowcharts")
    print("="*80)
    
    generate_afsa_enhanced_flowchart()
    generate_afsa_detailed_flowchart()
    
    print("\n✨ Both flowcharts generated successfully!")
    print("\nFiles created:")
    print("  1. figures/methodology_flowchart_afsa_enhanced.png")
    print("  2. figures/afsa_algorithm_detailed.png")
