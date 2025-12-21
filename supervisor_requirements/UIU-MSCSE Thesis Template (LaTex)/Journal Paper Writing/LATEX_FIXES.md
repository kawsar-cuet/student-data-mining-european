# LaTeX Math Mode Fixes - Overleaf Compatibility

## Issues Fixed

The journal paper had LaTeX math mode errors that prevented compilation in Overleaf. These have been resolved.

### 1. Underscore in Text Function Names
**Problem:** `\text{majority\_vote}` contained an underscore outside proper escaping
**Fixed:** Changed to `\text{majority vote}` (removed underscore)
**Location:** Line 441 - Random Forest algorithm

### 2. Underscores in Hyperparameter Table
**Problem:** Parameter names like `max_depth`, `min_samples_split`, etc. had unescaped underscores
**Fixed:** Wrapped all parameter names in `\texttt{}` and escaped underscores: `\texttt{max\_depth}`
**Location:** Table "Baseline Model Hyperparameters" (around line 680)

### 3. Author Block Email
**Problem:** Email placeholder had brackets and underscores causing parsing issues
**Fixed:** Changed from `[email@uiu.ac.bd]` to `author@uiu.ac.bd` with proper formatting
**Location:** Line 35-43 - Author information block

## Compilation Status

✅ **PDF compiles successfully**
✅ **15 pages (as required)**
✅ **3.16 MB with embedded figures**
✅ **No critical LaTeX errors**
✅ **Ready for Overleaf upload**

## How to Upload to Overleaf

1. **Create New Project** in Overleaf
2. **Upload all files:**
   - `AHFS_TA_Journal_Paper.tex` (main file)
   - `IEEEtran.cls` (class file)
   - `figures/` folder (all 14 PNG files)
3. **Set main document:** Right-click `AHFS_TA_Journal_Paper.tex` → Set as Main File
4. **Compile:** Click "Recompile" button

## Files to Upload

```
Journal Paper Writing/
├── AHFS_TA_Journal_Paper.tex   (Main document)
├── IEEEtran.cls                (IEEE class)
└── figures/
    ├── 01_class_distribution.png
    ├── 03_ranking_heatmap.png
    ├── 07_confusion_matrices.png
    ├── 07_model_comparison.png
    ├── 07_roc_curves.png
    ├── 11_all_models_accuracy_comparison.png
    ├── 11_shap_random_forest_summary.png
    ├── 11_shap_xgboost_summary.png
    ├── 12_all_models_confusion_matrices.png
    ├── 12_all_models_roc_curves.png
    ├── 12_comprehensive_metrics_comparison.png
    ├── 12_cross_validation_results.png
    ├── ahfs_ta_ablation_study.png
    └── ahfs_ta_model_comparison.png
```

## What Was Changed

| Line | Original | Fixed |
|------|----------|-------|
| 441 | `\text{majority\_vote}` | `\text{majority vote}` |
| 680-689 | `max_depth=10` (plain text) | `\texttt{max\_depth=10}` |
| 35-43 | `[email@uiu.ac.bd]` | `author@uiu.ac.bd` |

## Verification

The document now compiles without math mode errors. All mathematical expressions are properly enclosed in:
- Inline math: `$...$`
- Display math: `\[...\]` or `equation` environment
- All underscores in text are either:
  - Escaped: `\_`
  - In typewriter font: `\texttt{...\_...}`
  - In math mode: `$..._...$`

The paper is ready for submission to Overleaf or any LaTeX editor!
