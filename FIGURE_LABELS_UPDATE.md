# Figure Generation with Subfigure Labels

## Summary of Changes

I've updated three Python scripts to add subfigure labels (a), (b), (c), etc. to all multi-panel figures as required for high-quality journal papers:

### 1. 01_dataset_analysis.py
- **File**: `01_class_distribution.png`
- **Changes**: Modified to create a 1×2 subplot layout with:
  - **(a)** Bar chart showing class distribution
  - **(b)** Pie chart showing class distribution percentages
- Both subplots now have proper subfigure labels

### 2. regenerate_figures_with_ahfs_ta.py
- **File**: `07_confusion_matrices.png`
  - Changed from 4×2 layout to **3×3 layout** (3 confusion matrices per row)
  - Added subfigure labels **(a) through (g)** for all 7 models
  
- **File**: `07_roc_curves.png`
  - Changed from 4×2 layout to **3×3 layout** (3 ROC curves per row)
  - Added subfigure labels **(a) through (g)** for all 7 models
  
- **File**: `12_cross_validation_results.png`
  - Added subfigure labels:
    - **(a)** 10-Fold Cross-Validation Score Distribution (box plot)
    - **(b)** 10-Fold Cross-Validation Mean Accuracy ± Std Dev (bar plot)

### 3. generate_comprehensive_metrics_comparison.py
- **File**: `12_comprehensive_metrics_comparison.png`
  - Added subfigure labels to all 2×2 subplots:
    - **(a)** Performance Metrics Comparison
    - **(b)** Area Under ROC Curve (AUC) Comparison
    - **(c)** Test Accuracy Comparison
    - **(d)** 10-Fold Cross-Validation Mean Accuracy

## How to Generate the Figures

Run the following commands from the project root directory:

```powershell
# Navigate to project root
cd "d:\MS program\Final Thesis\Final Thesis project"

# Run script 1: Generate class distribution with (a) and (b) labels
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/01_dataset_analysis.py"

# Run script 2: Generate confusion matrices, ROC curves, and cross-validation with labels
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/regenerate_figures_with_ahfs_ta.py"

# Run script 3: Generate comprehensive metrics comparison with (a)-(d) labels
python "supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/python codes/generate_comprehensive_metrics_comparison.py"
```

Or run all at once:
```powershell
cd "d:\MS program\Final Thesis\Final Thesis project"
python run_figure_generation.py
```

## Output Locations

All figures will be saved to:
- `supervisor_requirements/UIU-MSCSE Thesis Template (LaTex)/Journal Paper Plain version/FIGURES/`
- `outputs/figures/`
- `Journal Paper Writing/figures/`

## Key Improvements

1. ✅ **Subfigure labels added** to all multi-panel figures using standard notation: (a), (b), (c), etc.
2. ✅ **3×3 grid layout** for confusion matrices (was 4×2) - shows 3 matrices per row for better readability
3. ✅ **3×3 grid layout** for ROC curves (was 4×2) - shows 3 curves per row for better readability
4. ✅ All labels follow IEEE/journal formatting standards
5. ✅ Labels are prominently displayed in subplot titles for easy reference in paper text

## LaTeX Caption Updates

When referencing these figures in your LaTeX paper, you can now write:

```latex
\caption{Class distribution analysis: (a) bar chart showing absolute counts, 
(b) pie chart showing percentage distribution.}

\caption{Confusion matrices for all models: (a) Decision Tree, (b) Naive Bayes, 
(c) Random Forest, (d) AdaBoost, (e) XGBoost, (f) Neural Network, (g) AHFS-TA.}

\caption{ROC curves for all models: (a) Decision Tree, (b) Naive Bayes, 
(c) Random Forest, (d) AdaBoost, (e) XGBoost, (f) Neural Network, (g) AHFS-TA.}

\caption{10-fold cross-validation results: (a) score distribution via box plots, 
(b) mean accuracy with standard deviation error bars.}

\caption{Comprehensive performance comparison: (a) four key metrics across all models, 
(b) AUC comparison, (c) test accuracy with AHFS-TA highlighted, 
(d) cross-validation mean accuracy.}
```
