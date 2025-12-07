"""
Generate Framework Flowchart - Student Performance Prediction System
Similar style to reference paper flowchart
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Wedge
import numpy as np

# Set up the figure with white background
fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 10)
ax.set_ylim(0, 20)
ax.axis('off')
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Color scheme matching the reference
COLOR_DATABASE = '#E8D5F2'      # Light purple - database
COLOR_TRAINING = '#D4E6F9'       # Light blue - training data
COLOR_TESTING = '#FFD4D4'        # Light red/pink - testing data
COLOR_ITERATION = '#E8F5E8'      # Light green - iteration/parameters
COLOR_MODEL = '#FFE8C8'          # Light orange - model generation
COLOR_EVALUATION = '#FFF4D4'     # Light yellow - evaluation
COLOR_RESULT = '#FFE4E8'         # Light pink - results

def draw_cylinder(ax, x, y, width, height, color, text, fontsize=10, fontweight='normal'):
    """Draw a database cylinder shape"""
    # Draw ellipse top
    ellipse_top = Wedge((x, y + height), width/2, 0, 180, 
                        facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(ellipse_top)
    
    # Draw rectangle body
    rect = FancyBboxPatch((x - width/2, y), width, height,
                          boxstyle="round,pad=0.02", 
                          facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Draw ellipse bottom outline
    ellipse_bottom = mpatches.Arc((x, y), width, height*0.3, 
                                  theta1=180, theta2=360, 
                                  color='black', linewidth=1.5)
    ax.add_patch(ellipse_bottom)
    
    # Add text
    ax.text(x, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

def draw_box(ax, x, y, width, height, color, text, fontsize=9, fontweight='normal'):
    """Draw a rounded rectangle box"""
    box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                         boxstyle="round,pad=0.05", 
                         facecolor=color, edgecolor='black', linewidth=1.5)
    ax.add_patch(box)
    
    # Add text with word wrapping
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True,
            multialignment='center')

def draw_diamond(ax, x, y, width, height, text, fontsize=9):
    """Draw a decision diamond"""
    # Create diamond points
    points = np.array([
        [x, y + height/2],      # top
        [x + width/2, y],       # right
        [x, y - height/2],      # bottom
        [x - width/2, y]        # left
    ])
    
    diamond = mpatches.Polygon(points, facecolor='white', 
                              edgecolor='black', linewidth=1.5)
    ax.add_patch(diamond)
    
    # Add text
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold')

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, linewidth=2,
                           color='black', mutation_scale=20)
    ax.add_patch(arrow)
    
    # Add label if provided
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=8, 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none'))

# ============================================================================
# DRAW THE FLOWCHART
# ============================================================================

# 1. Top: Student Performance Dataset (Database cylinder)
draw_cylinder(ax, 5, 18, 2.5, 0.8, COLOR_DATABASE, 
              'Student performance\ndataset', fontsize=10, fontweight='bold')

# Split arrows to training and testing
draw_arrow(ax, 4.2, 18, 3, 16.5)
draw_arrow(ax, 5.8, 18, 7, 16.5)

# 2. Training and Testing datasets (cylinders)
draw_cylinder(ax, 3, 15.5, 2.2, 0.8, COLOR_TRAINING,
              'Student performance\ntraining dataset', fontsize=9)
draw_cylinder(ax, 7, 15.5, 2.2, 0.8, COLOR_TESTING,
              'Student performance\ntesting dataset', fontsize=9)

# 3. Set maximum iterations (green box - left)
draw_box(ax, 2.2, 14, 1.8, 0.7, COLOR_ITERATION,
         'Set maximum number\nof iterations (T)', fontsize=8)

# Arrow to population size
draw_arrow(ax, 3.1, 14, 4.5, 14)

# 4. Set population size (green box - right)
draw_box(ax, 5.5, 14, 1.6, 0.7, COLOR_ITERATION,
         'Set population size\n(N)', fontsize=8)

# Arrow down
draw_arrow(ax, 5.5, 13.65, 5.5, 13)

# 5. Initialize metaheuristic parameters
draw_box(ax, 5.5, 12.5, 2, 0.8, COLOR_ITERATION,
         'Initialize\nmetaheuristic control\nparameters', fontsize=8)

# Arrow to counter
draw_arrow(ax, 4.5, 12.5, 2.8, 12.5)

# 6. Set current iteration counter
draw_box(ax, 2.2, 12.5, 1.8, 0.7, COLOR_ITERATION,
         'Set current iteration\ncounter (t) to 0', fontsize=8)

# Arrow down
draw_arrow(ax, 2.2, 12.15, 2.2, 11.5)

# 7. Initialize agent population
draw_box(ax, 2.2, 11, 1.8, 0.7, COLOR_ITERATION,
         'Initialize\nagent population (P)', fontsize=8)

# Arrow to model generation
draw_arrow(ax, 2.2, 10.65, 2.2, 10)

# 8. Generate models based on agent parameters
draw_box(ax, 2.2, 9.5, 2, 0.8, COLOR_MODEL,
         'Generate\nPPN/DPN-A/HMTL\nmodels based on\nagent (A) parameters', fontsize=8)

# Arrow down
draw_arrow(ax, 2.2, 9.15, 2.2, 8.5)

# 9. Train generated models
draw_box(ax, 2.2, 8, 2, 0.7, COLOR_MODEL,
         'Train generated\nPPN/DPN-A/HMTL\nmodels', fontsize=8)

# Arrow to evaluation
draw_arrow(ax, 3.2, 8, 5, 8)

# 10. Evaluate generated models
draw_box(ax, 6.5, 8, 2.2, 0.7, COLOR_EVALUATION,
         'Evaluate generated\nPPN/DPN-A/HMTL\nmodels', fontsize=8)

# Arrow back to testing dataset
draw_arrow(ax, 7.5, 8.35, 7.5, 14.7, style='->')

# Arrow from evaluation to update
draw_arrow(ax, 6.5, 7.65, 6.5, 7)

# 11. Update agents based on optimization
draw_box(ax, 6.5, 6.4, 2.2, 0.9, COLOR_EVALUATION,
         'Update agents based\non optimization\nstrategy and\nincrement t by one', fontsize=8)

# Arrow to decision diamond
draw_arrow(ax, 6.5, 5.95, 6.5, 5.3)

# 12. Decision diamond: While T > t
draw_diamond(ax, 6.5, 4.7, 2, 1, 'While T > t', fontsize=9)

# True arrow - loop back
draw_arrow(ax, 7.5, 4.7, 8.5, 4.7)
ax.text(8, 4.9, 'True', fontsize=8, fontweight='bold')
# Vertical line up
ax.plot([8.5, 8.5], [4.7, 9.5], 'k-', linewidth=2)
# Arrow back to generate models
draw_arrow(ax, 8.5, 9.5, 3.2, 9.5, style='->')

# False arrow - continue down
draw_arrow(ax, 5.5, 4.7, 4.5, 4.7)
ax.text(4.8, 4.9, 'False', fontsize=8, fontweight='bold')

# Arrow down to return best model
draw_arrow(ax, 4.5, 4.7, 4.5, 4)

# 13. Return the best performing model
draw_box(ax, 4.5, 3.5, 2, 0.8, COLOR_RESULT,
         'Return the best\nperforming model', fontsize=9, fontweight='bold')

# Arrow down
draw_arrow(ax, 4.5, 3.15, 4.5, 2.5)

# 14. Evaluate best model and generate visualization
draw_box(ax, 4.5, 2, 2.5, 0.8, COLOR_RESULT,
         'Evaluate the best\nperforming model and\ngenerate visualization', fontsize=9, fontweight='bold')

# Arrow looping back from visualization to return
ax.annotate('', xy=(5.75, 3.5), xytext=(5.75, 2.4),
            arrowprops=dict(arrowstyle='->', lw=2, color='red'))

# Add title
ax.text(5, 19.5, 'Proposed Framework: Student Performance Prediction System', 
        fontsize=14, fontweight='bold', ha='center')

# Add figure label
ax.text(0.5, 0.5, 'FIGURE 1. The flowchart of proposed framework used in simulations.', 
        fontsize=11, fontweight='bold', ha='left', style='italic')

# Adjust layout
plt.tight_layout()

# Save the figure
output_path = 'outputs/figures_journal/framework_flowchart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✓ Framework flowchart saved: {output_path}")

output_path_pdf = 'outputs/figures_journal/framework_flowchart.pdf'
plt.savefig(output_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
print(f"✓ Framework flowchart saved: {output_path_pdf}")

plt.show()

print("\n" + "="*70)
print("Framework Flowchart Generation Complete!")
print("="*70)
print(f"PNG: {output_path}")
print(f"PDF: {output_path_pdf}")
print(f"Resolution: 300 DPI")
print(f"Dimensions: 14×18 inches")
print("="*70)
