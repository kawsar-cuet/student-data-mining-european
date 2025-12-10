"""
Comprehensive Model Evaluation
Requirement 11: Calculate all evaluation metrics

Metrics:
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC Score
- 10-Fold Cross-Validation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)
(OUTPUT_DIR / "evaluation").mkdir(exist_ok=True)

# Load dataset
print("\n" + "="*80)
print("REQUIREMENT 11: COMPREHENSIVE MODEL EVALUATION")
print("="*80)

print("\nLoading dataset...")
df = pd.read_csv("../data/educational_data.csv")

target_col = 'Target'
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode target: Dropout=0, Enrolled=1, Graduate=2
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
class_names = list(le.classes_)
n_classes = len(class_names)

print(f"\nDataset: {len(df)} students")
print(f"Features: {X.shape[1]}")
print(f"Classes: {n_classes}")

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load scaler
scaler = joblib.load(OUTPUT_DIR / "models" / "scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Load all trained models
print("\nLoading trained models...")
model_files = {
    'Decision Tree': 'decision_tree.pkl',
    'Naive Bayes': 'naive_bayes.pkl',
    'Random Forest': 'random_forest.pkl',
    'AdaBoost': 'adaboost.pkl',
    'XGBoost': 'xgboost.pkl',
    'Neural Network': 'neural_network.pkl'
}

models = {}
for name, filename in model_files.items():
    models[name] = joblib.load(OUTPUT_DIR / "models" / filename)
print(f"✓ Loaded {len(models)} models")

# ============================================================================
# EVALUATE ALL MODELS
# ============================================================================
print("\n" + "="*80)
print("EVALUATING MODELS")
print("="*80)

all_results = []

for model_name, model in models.items():
    print(f"\n{model_name}:")
    print("-" * 60)
    
    # Use scaled data for Neural Network, original for others
    if model_name == 'Neural Network':
        X_te = X_test_scaled
    else:
        X_te = X_test
    
    # Predictions
    y_pred = model.predict(X_te)
    y_pred_proba = model.predict_proba(X_te)
    
    # 1. Basic Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    # 2. Per-class metrics
    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
    
    # 3. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # 4. ROC-AUC (One-vs-Rest for multiclass)
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
    
    try:
        roc_auc_ovr = roc_auc_score(y_test_bin, y_pred_proba, average='weighted', multi_class='ovr')
    except:
        roc_auc_ovr = np.nan
    
    print(f"  ROC-AUC:   {roc_auc_ovr:.4f}")
    
    # Store results
    all_results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'ROC_AUC': roc_auc_ovr,
        'Dropout_Precision': report['Dropout']['precision'],
        'Dropout_Recall': report['Dropout']['recall'],
        'Dropout_F1': report['Dropout']['f1-score'],
        'Enrolled_Precision': report['Enrolled']['precision'],
        'Enrolled_Recall': report['Enrolled']['recall'],
        'Enrolled_F1': report['Enrolled']['f1-score'],
        'Graduate_Precision': report['Graduate']['precision'],
        'Graduate_Recall': report['Graduate']['recall'],
        'Graduate_F1': report['Graduate']['f1-score'],
        'confusion_matrix': cm,
        'y_pred_proba': y_pred_proba,
        'y_pred': y_pred
    })

# Convert to DataFrame
results_df = pd.DataFrame(all_results)
summary_df = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].copy()
summary_df.to_csv(OUTPUT_DIR / "tables" / "07_model_evaluation_summary.csv", index=False)

print("\n\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"\n{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}")
print("-"*70)
for idx, row in summary_df.iterrows():
    print(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1_Score']:<10.4f} {row['ROC_AUC']:<10.4f}")

# ============================================================================
# 10-FOLD CROSS-VALIDATION
# ============================================================================
print("\n\n" + "="*80)
print("10-FOLD CROSS-VALIDATION")
print("="*80)

cv_results = []
cv_fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

for model_name, model in models.items():
    print(f"\n{model_name}:")
    
    # Use original data for all models
    scores = cross_val_score(model, X, y, cv=cv_fold, scoring='accuracy', n_jobs=-1)
    
    mean_score = scores.mean()
    std_score = scores.std()
    
    print(f"  Mean Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"  Min: {scores.min():.4f}, Max: {scores.max():.4f}")
    
    cv_results.append({
        'Model': model_name,
        'CV_Mean_Accuracy': mean_score,
        'CV_Std_Accuracy': std_score,
        'CV_Min_Accuracy': scores.min(),
        'CV_Max_Accuracy': scores.max(),
        'All_Fold_Scores': scores.tolist()
    })

cv_df = pd.DataFrame(cv_results)
cv_df[['Model', 'CV_Mean_Accuracy', 'CV_Std_Accuracy', 'CV_Min_Accuracy', 'CV_Max_Accuracy']].to_csv(
    OUTPUT_DIR / "tables" / "07_cross_validation_results.csv", index=False
)

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n\nGenerating visualizations...")

# 1. Model Comparison Bar Chart
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1_Score']
metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
colors = ['steelblue', 'forestgreen', 'coral', 'purple']

for idx, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
    ax = axes[idx // 2, idx % 2]
    data = summary_df.sort_values(metric, ascending=True)
    ax.barh(range(len(data)), data[metric], color=color, edgecolor='black')
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data['Model'], fontsize=10)
    ax.set_xlabel(label, fontsize=11, fontweight='bold')
    ax.set_title(f'{label} Comparison', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim([0, 1.0])

plt.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "07_model_comparison.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '07_model_comparison.png'}")

# 2. Confusion Matrices (all models in grid)
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

for idx, (model_name, result) in enumerate(zip(summary_df['Model'], all_results)):
    ax = axes[idx // 3, idx % 3]
    cm = result['confusion_matrix']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=10, fontweight='bold')
    ax.set_title(f'{model_name}', fontsize=11, fontweight='bold')

plt.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "07_confusion_matrices.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '07_confusion_matrices.png'}")

# 3. ROC Curves (One-vs-Rest for each class)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

for class_idx, class_name in enumerate(class_names):
    ax = axes[class_idx]
    
    for model_name, result in zip(summary_df['Model'], all_results):
        y_score = result['y_pred_proba'][:, class_idx]
        fpr, tpr, _ = roc_curve(y_test_bin[:, class_idx], y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=11, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=11, fontweight='bold')
    ax.set_title(f'ROC Curve: {class_name} (One-vs-Rest)', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3)

plt.suptitle('ROC Curves for All Models and Classes', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "07_roc_curves.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '07_roc_curves.png'}")

# 4. Cross-Validation Comparison
plt.figure(figsize=(12, 7))
cv_data = cv_df.sort_values('CV_Mean_Accuracy', ascending=True)

plt.barh(range(len(cv_data)), cv_data['CV_Mean_Accuracy'], 
         xerr=cv_data['CV_Std_Accuracy'], 
         color='skyblue', edgecolor='black', capsize=5)
plt.yticks(range(len(cv_data)), cv_data['Model'], fontsize=11)
plt.xlabel('Mean Accuracy (10-Fold CV)', fontsize=12, fontweight='bold')
plt.title('10-Fold Cross-Validation Results\n(Error bars show standard deviation)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.xlim([0, 1.0])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "07_cross_validation.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '07_cross_validation.png'}")

# 5. Per-Class Performance Heatmap
per_class_df = pd.DataFrame({
    'Model': summary_df['Model'],
    'Dropout_F1': [r['Dropout_F1'] for r in all_results],
    'Enrolled_F1': [r['Enrolled_F1'] for r in all_results],
    'Graduate_F1': [r['Graduate_F1'] for r in all_results]
})

plt.figure(figsize=(10, 8))
heatmap_data = per_class_df.set_index('Model')[['Dropout_F1', 'Enrolled_F1', 'Graduate_F1']].T
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlGnBu', cbar_kws={'label': 'F1-Score'})
plt.xlabel('Model', fontsize=12, fontweight='bold')
plt.ylabel('Class', fontsize=12, fontweight='bold')
plt.title('Per-Class F1-Score Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "07_per_class_performance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '07_per_class_performance.png'}")

# ============================================================================
# COMPREHENSIVE REPORT
# ============================================================================
print("\nGenerating comprehensive evaluation report...")

with open(OUTPUT_DIR / "07_comprehensive_evaluation_report.txt", 'w') as f:
    f.write("COMPREHENSIVE MODEL EVALUATION REPORT\n")
    f.write("="*100 + "\n\n")
    
    f.write("Dataset Information:\n")
    f.write(f"  Total Students: {len(df)}\n")
    f.write(f"  Features: {X.shape[1]}\n")
    f.write(f"  Classes: {n_classes} (Dropout, Enrolled, Graduate)\n")
    f.write(f"  Train/Test Split: 80/20 ({len(X_train)}/{len(X_test)})\n\n")
    f.write("="*100 + "\n\n")
    
    f.write("OVERALL PERFORMANCE SUMMARY\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Model':<20} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'ROC-AUC':<10}\n")
    f.write("-"*100 + "\n")
    for idx, row in summary_df.iterrows():
        f.write(f"{row['Model']:<20} {row['Accuracy']:<10.4f} {row['Precision']:<10.4f} {row['Recall']:<10.4f} {row['F1_Score']:<10.4f} {row['ROC_AUC']:<10.4f}\n")
    
    f.write("\n\n" + "="*100 + "\n\n")
    f.write("10-FOLD CROSS-VALIDATION RESULTS\n")
    f.write("-"*100 + "\n")
    f.write(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Dev':<10} {'Min':<10} {'Max':<10}\n")
    f.write("-"*100 + "\n")
    for idx, row in cv_df.iterrows():
        f.write(f"{row['Model']:<20} {row['CV_Mean_Accuracy']:<15.4f} {row['CV_Std_Accuracy']:<10.4f} {row['CV_Min_Accuracy']:<10.4f} {row['CV_Max_Accuracy']:<10.4f}\n")
    
    f.write("\n\n" + "="*100 + "\n\n")
    f.write("PER-CLASS PERFORMANCE\n")
    f.write("-"*100 + "\n\n")
    
    for idx, result in enumerate(all_results):
        model_name = result['Model']
        f.write(f"{model_name}:\n")
        f.write(f"  Dropout   - Precision: {result['Dropout_Precision']:.4f}, Recall: {result['Dropout_Recall']:.4f}, F1: {result['Dropout_F1']:.4f}\n")
        f.write(f"  Enrolled  - Precision: {result['Enrolled_Precision']:.4f}, Recall: {result['Enrolled_Recall']:.4f}, F1: {result['Enrolled_F1']:.4f}\n")
        f.write(f"  Graduate  - Precision: {result['Graduate_Precision']:.4f}, Recall: {result['Graduate_Recall']:.4f}, F1: {result['Graduate_F1']:.4f}\n\n")
    
    f.write("\n" + "="*100 + "\n\n")
    f.write("CONFUSION MATRICES\n")
    f.write("-"*100 + "\n\n")
    
    for idx, result in enumerate(all_results):
        model_name = result['Model']
        cm = result['confusion_matrix']
        f.write(f"{model_name}:\n")
        f.write(f"              Predicted\n")
        f.write(f"              Dropout  Enrolled  Graduate\n")
        f.write(f"Actual\n")
        for i, class_name in enumerate(class_names):
            f.write(f"  {class_name:<10} {cm[i][0]:>7} {cm[i][1]:>9} {cm[i][2]:>9}\n")
        f.write("\n")
    
    f.write("\n" + "="*100 + "\n\n")
    f.write("BEST PERFORMING MODEL\n")
    f.write("-"*100 + "\n")
    best_accuracy = summary_df.loc[summary_df['Accuracy'].idxmax()]
    best_f1 = summary_df.loc[summary_df['F1_Score'].idxmax()]
    best_cv = cv_df.loc[cv_df['CV_Mean_Accuracy'].idxmax()]
    
    f.write(f"\nBest Accuracy: {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})\n")
    f.write(f"Best F1-Score: {best_f1['Model']} ({best_f1['F1_Score']:.4f})\n")
    f.write(f"Best CV Accuracy: {best_cv['Model']} ({best_cv['CV_Mean_Accuracy']:.4f})\n")
    
    f.write("\n\n" + "="*100 + "\n\n")
    f.write("OUTPUTS GENERATED:\n")
    f.write("-"*100 + "\n")
    f.write("Tables:\n")
    f.write("  - 07_model_evaluation_summary.csv\n")
    f.write("  - 07_cross_validation_results.csv\n\n")
    f.write("Figures:\n")
    f.write("  - 07_model_comparison.png (4-panel comparison)\n")
    f.write("  - 07_confusion_matrices.png (all models)\n")
    f.write("  - 07_roc_curves.png (3 classes, all models)\n")
    f.write("  - 07_cross_validation.png (CV comparison)\n")
    f.write("  - 07_per_class_performance.png (heatmap)\n\n")
    f.write("="*100 + "\n")

print(f"  ✓ Saved: {OUTPUT_DIR / '07_comprehensive_evaluation_report.txt'}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("EVALUATION COMPLETE")
print("="*80)

print("\nBest Performing Models:")
print(f"  Accuracy:  {best_accuracy['Model']} ({best_accuracy['Accuracy']:.4f})")
print(f"  F1-Score:  {best_f1['Model']} ({best_f1['F1_Score']:.4f})")
print(f"  CV Score:  {best_cv['Model']} ({best_cv['CV_Mean_Accuracy']:.4f})")

print(f"\n✓ All results saved to: {OUTPUT_DIR / 'tables'}")
print(f"✓ All visualizations saved to: {OUTPUT_DIR / 'figures'}")
print(f"✓ Comprehensive report: {OUTPUT_DIR / '07_comprehensive_evaluation_report.txt'}")
print("\n" + "="*80 + "\n")
