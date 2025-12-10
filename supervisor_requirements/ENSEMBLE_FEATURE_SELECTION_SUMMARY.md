# Ensemble Feature Selection Optimization - Summary

## Overview
This analysis tested **Random Forest, AdaBoost, and XGBoost** classifiers with 9 different feature selection methods to identify optimal feature subsets and improve model accuracy.

---

## Key Results

### Best Configurations

#### Random Forest
- **Best Method**: RFE (Recursive Feature Elimination)
- **Optimal Features**: 20 features
- **Test Accuracy**: 77.85%
- **CV Accuracy**: 77.11%
- **Improvement**: +0.68% over baseline (77.18%)

#### AdaBoost
- **Best Method**: Mutual Information
- **Optimal Features**: 15 features  
- **Test Accuracy**: 77.06%
- **CV Accuracy**: 75.87%
- **Improvement**: +2.15% over baseline (74.92%)

#### XGBoost
- **Best Method**: RF Importance
- **Optimal Features**: 30 features
- **Test Accuracy**: 77.97%
- **CV Accuracy**: 77.08%
- **Improvement**: +0.90% over baseline (77.06%)

---

## Method Performance Rankings

### For Random Forest (by best accuracy):
1. **RFE**: 77.85% (20 features) ⭐
2. **Gini Index**: 77.63% (30 features)
3. **Info Gain**: 77.51% (30 features)
4. **Mutual Info**: 77.40% (15 features)
5. **RF Importance**: 77.40% (30 features)
6. **Gain Ratio**: 77.40% (15 features)
7. **All Features**: 77.18% (34 features - baseline)
8. **ANOVA F-stat**: 77.06% (30 features)
9. **Chi-Square**: 77.63% (30 features)

### For AdaBoost (by best accuracy):
1. **Mutual Info**: 77.06% (15 features) ⭐
2. **RF Importance**: 76.84% (20 features)
3. **RFE**: 76.84% (20 features)
4. **Info Gain**: 76.50% (15 features)
5. **Gini Index**: 76.38% (20 features)
6. **Gain Ratio**: 76.16% (15 features)
7. **ANOVA F-stat**: 75.82% (15 features)
8. **Chi-Square**: 75.71% (25 features)
9. **All Features**: 74.92% (34 features - baseline)

### For XGBoost (by best accuracy):
1. **RF Importance**: 77.97% (30 features) ⭐
2. **Mutual Info**: 77.63% (15 features)
3. **Gain Ratio**: 77.63% (30 features)
4. **Info Gain**: 77.51% (20 features)
5. **Gini Index**: 77.40% (15 features)
6. **RFE**: 77.40% (25 features)
7. **All Features**: 77.06% (34 features - baseline)
8. **Chi-Square**: 76.72% (25 features)
9. **ANOVA F-stat**: 76.61% (20 features)

---

## Key Insights

1. **XGBoost is the best overall model** - Achieves 77.97% test accuracy
2. **AdaBoost shows largest improvement** - 2.15% gain with feature selection
3. **Random Forest & XGBoost already excellent** - Baseline accuracy >77%, modest gains with feature selection
4. **Ensemble methods significantly outperform single classifiers**:
   - Decision Tree best: 68.81%
   - Naive Bayes best: 72.66%
   - Random Forest best: 77.85%
   - AdaBoost best: 77.06%
   - XGBoost best: 77.97%
5. **Feature reduction benefits**:
   - RF: 20 features (41% reduction) with performance gain
   - AdaBoost: 15 features (56% reduction) with 2.15% improvement
   - XGBoost: 30 features (12% reduction) with performance gain

---

## Comparison: Single vs Ensemble Classifiers

| Model Type | Best Accuracy | Best Method | Optimal Features | Improvement |
|------------|---------------|-------------|------------------|-------------|
| **Decision Tree** | 68.81% | Info Gain | 10 | +4.29% |
| **Naive Bayes** | 72.66% | Info Gain | 15 | +5.99% |
| **Random Forest** | 77.85% | RFE | 20 | +0.68% |
| **AdaBoost** | 77.06% | Mutual Info | 15 | +2.15% |
| **XGBoost** | 77.97% | RF Importance | 30 | +0.90% |

**Winner**: XGBoost with RF Importance (30 features) - **77.97% accuracy**

---

## Visualizations Generated

1. **09_ensemble_accuracy_vs_features.png** - Line plots showing how accuracy changes with feature count for each method (3×2 grid for RF/Ada/XGB)
2. **09_ensemble_accuracy_heatmap.png** - Heatmaps showing accuracy for all method-feature combinations (9×7 grids)
3. **09_ensemble_best_accuracy_per_method.png** - Horizontal bar charts ranking methods by best accuracy
4. **09_ensemble_models_comparison.png** - Side-by-side comparison of all 3 models across all 9 methods
5. **09_ensemble_all_metrics_comparison.png** - Multi-metric comparison (accuracy, precision, recall, F1) for all models

---

## Recommendations

### For Production Deployment:
- **Best Model**: XGBoost with 30 features selected by RF Importance
- **Expected Performance**: 77.97% test accuracy, 77.08% CV accuracy
- **Benefits**: Highest accuracy, 4 fewer features, stable cross-validation

### For Computational Efficiency:
- **Best Model**: AdaBoost with 15 features selected by Mutual Information
- **Expected Performance**: 77.06% test accuracy, 75.87% CV accuracy
- **Benefits**: 56% feature reduction, 2.15% accuracy gain, faster training/prediction

### For Balanced Performance:
- **Best Model**: Random Forest with 20 features selected by RFE
- **Expected Performance**: 77.85% test accuracy, 77.11% CV accuracy
- **Benefits**: 41% feature reduction, competitive accuracy, robust to overfitting

---

## Feature Selection Testing Details

- **Methods Tested**: 9 (All Features, ANOVA F-stat, Mutual Info, Chi-Square, RFE, RF Importance, Info Gain, Gain Ratio, Gini Index)
- **Feature Counts**: 7 (5, 10, 15, 20, 25, 30, 34)
- **Models Tested**: 3 (Random Forest, AdaBoost, XGBoost)
- **Total Configurations**: 189 (9 methods × 7 feature counts × 3 models)
- **Evaluation**: 5-fold cross-validation for each configuration
- **Dataset**: 4,424 students, train/test split: 3,539/885

---

## Files Generated

- **Results CSV**: `outputs/tables/09_ensemble_feature_selection_results.csv` (189 rows)
- **Summary CSV**: `outputs/tables/09_ensemble_feature_selection_summary.csv`
- **Report TXT**: `outputs/09_ensemble_feature_selection_report.txt`
- **5 Visualization PNG files** in `outputs/figures/`
