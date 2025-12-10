"""
Comprehensive Model Evaluation
11.1 Accuracy, Precision, Recall, F1-Score
11.2 Confusion Matrix
11.3 ROC Curve, AUC Curve
11.4 10-Fold Cross-Validation

Evaluates all 6 optimized models with complete metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, RFE
from scipy.stats import entropy
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Create output directories
figures_dir = Path("outputs/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COMPREHENSIVE MODEL EVALUATION - ALL MODELS")
print("="*80)

# Load dataset
data_path = Path("../data/educational_data.csv")
print(f"\n1. Loading dataset from: {data_path}")

df = pd.read_csv(data_path)
print(f"   Dataset shape: {df.shape}")

# Prepare data
X = df.drop('Target', axis=1)
y = df['Target']

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_
n_classes = len(target_names)

print(f"   Target classes: {target_names}")
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {n_classes}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")

# Feature names
feature_names = X.columns.tolist()

# Helper function: Calculate Information Gain
def calculate_information_gain(X, y):
    """Calculate information gain for each feature"""
    y_entropy = entropy(np.bincount(y) / len(y), base=2)
    ig_scores = []
    
    for col in X.columns:
        feature_entropy = 0
        values = X[col].unique()
        
        for value in values:
            subset_indices = X[col] == value
            subset_y = y[subset_indices]
            
            if len(subset_y) > 0:
                weight = len(subset_y) / len(y)
                subset_entropy = entropy(np.bincount(subset_y) / len(subset_y), base=2)
                feature_entropy += weight * subset_entropy
        
        ig = y_entropy - feature_entropy
        ig_scores.append(ig)
    
    return np.array(ig_scores)

print("\n" + "="*80)
print("2. TRAINING ALL MODELS WITH OPTIMAL CONFIGURATIONS")
print("="*80)

# Dictionary to store all models and their data
models_dict = {}

# ============================================================================
# 1. DECISION TREE
# ============================================================================
print("\n[1/6] Decision Tree with Information Gain (10 features)")
print("-" * 60)

# Feature selection using Information Gain
ig_scores = calculate_information_gain(X_train, y_train)
top_10_indices = np.argsort(ig_scores)[-10:]
dt_features = [feature_names[i] for i in top_10_indices]

X_train_dt = X_train[dt_features]
X_test_dt = X_test[dt_features]

# Train model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_dt, y_train)

models_dict['Decision Tree'] = {
    'model': dt_model,
    'X_train': X_train_dt,
    'X_test': X_test_dt,
    'features': dt_features,
    'n_features': 10
}

print(f"Decision Tree trained with {len(dt_features)} features")

# ============================================================================
# 2. NAIVE BAYES
# ============================================================================
print("\n[2/6] Naive Bayes with Information Gain (15 features)")
print("-" * 60)

# Feature selection using Information Gain
top_15_indices = np.argsort(ig_scores)[-15:]
nb_features = [feature_names[i] for i in top_15_indices]

X_train_nb = X_train[nb_features]
X_test_nb = X_test[nb_features]

# Train model
nb_model = GaussianNB()
nb_model.fit(X_train_nb, y_train)

models_dict['Naive Bayes'] = {
    'model': nb_model,
    'X_train': X_train_nb,
    'X_test': X_test_nb,
    'features': nb_features,
    'n_features': 15
}

print(f"Naive Bayes trained with {len(nb_features)} features")

# ============================================================================
# 3. RANDOM FOREST
# ============================================================================
print("\n[3/6] Random Forest with RFE (20 features)")
print("-" * 60)

# Feature selection using RFE
rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rfe = RFE(rf_estimator, n_features_to_select=20, step=5)
X_train_rf_selected = rfe.fit_transform(X_train, y_train)
X_test_rf_selected = rfe.transform(X_test)

rf_features = [feature_names[i] for i in range(len(feature_names)) if rfe.support_[i]]

X_train_rf = pd.DataFrame(X_train_rf_selected, columns=rf_features)
X_test_rf = pd.DataFrame(X_test_rf_selected, columns=rf_features)

# Train model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_rf, y_train)

models_dict['Random Forest'] = {
    'model': rf_model,
    'X_train': X_train_rf,
    'X_test': X_test_rf,
    'features': rf_features,
    'n_features': 20
}

print(f"Random Forest trained with {len(rf_features)} features")

# ============================================================================
# 4. ADABOOST
# ============================================================================
print("\n[4/6] AdaBoost with Mutual Info (15 features)")
print("-" * 60)

# Feature selection using Mutual Information
mi_selector = SelectKBest(mutual_info_classif, k=15)
X_train_ada_selected = mi_selector.fit_transform(X_train, y_train)
X_test_ada_selected = mi_selector.transform(X_test)

ada_features = [feature_names[i] for i in mi_selector.get_support(indices=True)]

X_train_ada = pd.DataFrame(X_train_ada_selected, columns=ada_features)
X_test_ada = pd.DataFrame(X_test_ada_selected, columns=ada_features)

# Train model
ada_model = AdaBoostClassifier(n_estimators=50, random_state=42, algorithm='SAMME')
ada_model.fit(X_train_ada, y_train)

models_dict['AdaBoost'] = {
    'model': ada_model,
    'X_train': X_train_ada,
    'X_test': X_test_ada,
    'features': ada_features,
    'n_features': 15
}

print(f"AdaBoost trained with {len(ada_features)} features")

# ============================================================================
# 5. XGBOOST
# ============================================================================
print("\n[5/6] XGBoost with RF Importance (30 features)")
print("-" * 60)

# Feature selection using Random Forest importance
rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
top_30_indices = np.argsort(importances)[-30:]
xgb_features = [feature_names[i] for i in top_30_indices]

X_train_xgb = X_train[xgb_features]
X_test_xgb = X_test[xgb_features]

# Train model
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train_xgb, y_train)

models_dict['XGBoost'] = {
    'model': xgb_model,
    'X_train': X_train_xgb,
    'X_test': X_test_xgb,
    'features': xgb_features,
    'n_features': 30
}

print(f"XGBoost trained with {len(xgb_features)} features")

# ============================================================================
# 6. NEURAL NETWORK
# ============================================================================
print("\n[6/6] Neural Network with ANOVA F-stat (15 features)")
print("-" * 60)

# Feature selection using ANOVA F-statistic
anova_selector = SelectKBest(f_classif, k=15)
X_train_nn_selected = anova_selector.fit_transform(X_train, y_train)
X_test_nn_selected = anova_selector.transform(X_test)

nn_features = [feature_names[i] for i in anova_selector.get_support(indices=True)]

# Standardize for Neural Network
scaler = StandardScaler()
X_train_nn_scaled = scaler.fit_transform(X_train_nn_selected)
X_test_nn_scaled = scaler.transform(X_test_nn_selected)

X_train_nn = pd.DataFrame(X_train_nn_scaled, columns=nn_features)
X_test_nn = pd.DataFrame(X_test_nn_scaled, columns=nn_features)

# Train model
nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
nn_model.fit(X_train_nn, y_train)

models_dict['Neural Network'] = {
    'model': nn_model,
    'X_train': X_train_nn,
    'X_test': X_test_nn,
    'features': nn_features,
    'n_features': 15,
    'scaler': scaler
}

print(f"Neural Network trained with {len(nn_features)} features")

# ============================================================================
# EVALUATION METRICS
# ============================================================================
print("\n" + "="*80)
print("3. CALCULATING EVALUATION METRICS FOR ALL MODELS")
print("="*80)

# Store all metrics
all_metrics = {}

for model_name, model_data in models_dict.items():
    print(f"\n[{model_name}]")
    print("-" * 60)
    
    model = model_data['model']
    X_test_model = model_data['X_test']
    
    # Predictions
    y_pred = model.predict(X_test_model)
    y_pred_proba = model.predict_proba(X_test_model)
    
    # 11.1 - Accuracy, Precision, Recall, F1-Score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    # Store metrics
    all_metrics[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }

# ============================================================================
# 11.2 - CONFUSION MATRICES
# ============================================================================
print("\n" + "="*80)
print("4. GENERATING CONFUSION MATRICES")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (model_name, metrics) in enumerate(all_metrics.items()):
    cm = confusion_matrix(y_test, metrics['y_pred'])
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names,
                ax=axes[idx], cbar_kws={'label': 'Count'})
    axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.4f}', 
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Predicted Label', fontsize=10)
    axes[idx].set_ylabel('True Label', fontsize=10)
    
    print(f"✓ {model_name} confusion matrix created")

plt.tight_layout()
plt.savefig(figures_dir / "12_all_models_confusion_matrices.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 12_all_models_confusion_matrices.png")
plt.close()

# ============================================================================
# 11.3 - ROC CURVES AND AUC
# ============================================================================
print("\n" + "="*80)
print("5. GENERATING ROC CURVES AND CALCULATING AUC")
print("="*80)

# Binarize labels for multi-class ROC
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for idx, (model_name, model_data) in enumerate(models_dict.items()):
    model = model_data['model']
    X_test_model = model_data['X_test']
    y_pred_proba = model.predict_proba(X_test_model)
    
    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Plot ROC curves
    colors = ['blue', 'red', 'green']
    for i, color, class_name in zip(range(n_classes), colors, target_names):
        axes[idx].plot(fpr[i], tpr[i], color=color, lw=2,
                      label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plot micro-average
    axes[idx].plot(fpr["micro"], tpr["micro"], color='deeppink', lw=2, linestyle='--',
                  label=f'Micro-avg (AUC = {roc_auc["micro"]:.3f})')
    
    # Plot diagonal
    axes[idx].plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC = 0.500)')
    
    axes[idx].set_xlim([0.0, 1.0])
    axes[idx].set_ylim([0.0, 1.05])
    axes[idx].set_xlabel('False Positive Rate', fontsize=10)
    axes[idx].set_ylabel('True Positive Rate', fontsize=10)
    axes[idx].set_title(f'{model_name} - ROC Curves', fontsize=12, fontweight='bold')
    axes[idx].legend(loc="lower right", fontsize=8)
    axes[idx].grid(alpha=0.3)
    
    # Store AUC values
    all_metrics[model_name]['auc_per_class'] = roc_auc
    all_metrics[model_name]['auc_micro'] = roc_auc["micro"]
    
    print(f"✓ {model_name} - Micro-avg AUC: {roc_auc['micro']:.4f}")

plt.tight_layout()
plt.savefig(figures_dir / "12_all_models_roc_curves.png", dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: 12_all_models_roc_curves.png")
plt.close()

# ============================================================================
# 11.4 - 10-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "="*80)
print("6. PERFORMING 10-FOLD CROSS-VALIDATION")
print("="*80)

cv_results = {}
cv_folds = 10
cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

for model_name, model_data in models_dict.items():
    print(f"\n[{model_name}]")
    print("-" * 60)
    
    model = model_data['model']
    
    # Get the appropriate feature set
    if model_name == 'Decision Tree':
        X_cv = X[dt_features]
    elif model_name == 'Naive Bayes':
        X_cv = X[nb_features]
    elif model_name == 'Random Forest':
        # Need to apply RFE transformation
        rf_estimator_cv = RandomForestClassifier(n_estimators=50, random_state=42)
        rfe_cv = RFE(rf_estimator_cv, n_features_to_select=20, step=5)
        X_cv_temp = rfe_cv.fit_transform(X, y_encoded)
        X_cv = pd.DataFrame(X_cv_temp, columns=rf_features)
    elif model_name == 'AdaBoost':
        X_cv = X[ada_features]
    elif model_name == 'XGBoost':
        X_cv = X[xgb_features]
    elif model_name == 'Neural Network':
        X_cv_selected = X[nn_features]
        # Standardize for Neural Network
        scaler_cv = StandardScaler()
        X_cv = pd.DataFrame(scaler_cv.fit_transform(X_cv_selected), columns=nn_features)
    
    # Perform cross-validation
    cv_scores = cross_val_score(model, X_cv, y_encoded, cv=cv, scoring='accuracy')
    
    cv_results[model_name] = {
        'scores': cv_scores,
        'mean': cv_scores.mean(),
        'std': cv_scores.std()
    }
    
    print(f"Cross-Validation Scores ({cv_folds} folds):")
    for fold, score in enumerate(cv_scores, 1):
        print(f"  Fold {fold:2d}: {score:.4f} ({score*100:.2f}%)")
    print(f"  Mean:    {cv_scores.mean():.4f} ({cv_scores.mean()*100:.2f}%)")
    print(f"  Std Dev: {cv_scores.std():.4f}")

# Visualize Cross-Validation Results
print("\n" + "="*80)
print("7. VISUALIZING CROSS-VALIDATION RESULTS")
print("="*80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Box plot of CV scores
cv_data = [cv_results[model]['scores'] for model in models_dict.keys()]
bp = ax1.boxplot(cv_data, labels=list(models_dict.keys()), patch_artist=True)

# Color the boxes
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lavender']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax1.set_ylabel('Accuracy Score', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_title('10-Fold Cross-Validation Score Distribution', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
ax1.tick_params(axis='x', rotation=45)

# Mean scores with error bars
model_names = list(models_dict.keys())
means = [cv_results[model]['mean'] for model in model_names]
stds = [cv_results[model]['std'] for model in model_names]

x_pos = np.arange(len(model_names))
bars = ax2.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.4f}\n±{std:.4f}',
            ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('Mean Accuracy Score', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.set_title('10-Fold Cross-Validation Mean Accuracy ± Std Dev', fontsize=14, fontweight='bold')
ax2.set_xticks(x_pos)
ax2.set_xticklabels(model_names, rotation=45, ha='right')
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.6, 0.85])

plt.tight_layout()
plt.savefig(figures_dir / "12_cross_validation_results.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 12_cross_validation_results.png")
plt.close()

# ============================================================================
# COMPREHENSIVE METRICS COMPARISON
# ============================================================================
print("\n" + "="*80)
print("8. CREATING COMPREHENSIVE METRICS COMPARISON")
print("="*80)

# Create comparison table
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Accuracy, Precision, Recall, F1-Score comparison
ax1 = axes[0, 0]
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
x = np.arange(len(models_dict))
width = 0.2

for i, metric in enumerate(metrics_to_plot):
    values = [all_metrics[model][metric] for model in models_dict.keys()]
    offset = (i - 1.5) * width
    ax1.bar(x + offset, values, width, label=metric.replace('_', ' ').title())

ax1.set_ylabel('Score', fontsize=12)
ax1.set_xlabel('Model', fontsize=12)
ax1.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(models_dict.keys(), rotation=45, ha='right')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0.6, 0.85])

# 2. AUC Comparison
ax2 = axes[0, 1]
auc_values = [all_metrics[model]['auc_micro'] for model in models_dict.keys()]
bars = ax2.bar(models_dict.keys(), auc_values, color=colors, alpha=0.7, edgecolor='black')

for bar, auc_val in zip(bars, auc_values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{auc_val:.4f}',
            ha='center', va='bottom', fontsize=10)

ax2.set_ylabel('Micro-Average AUC', fontsize=12)
ax2.set_xlabel('Model', fontsize=12)
ax2.set_title('Area Under ROC Curve (AUC) Comparison', fontsize=14, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)
ax2.set_ylim([0.7, 1.0])

# 3. Cross-Validation Mean Accuracy
ax3 = axes[1, 0]
cv_means = [cv_results[model]['mean'] for model in models_dict.keys()]
cv_stds = [cv_results[model]['std'] for model in models_dict.keys()]
bars = ax3.bar(models_dict.keys(), cv_means, yerr=cv_stds, capsize=5, 
               color=colors, alpha=0.7, edgecolor='black')

for bar, mean, std in zip(bars, cv_means, cv_stds):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
            f'{mean:.4f}',
            ha='center', va='bottom', fontsize=10)

ax3.set_ylabel('Mean CV Accuracy', fontsize=12)
ax3.set_xlabel('Model', fontsize=12)
ax3.set_title('10-Fold Cross-Validation Mean Accuracy', fontsize=14, fontweight='bold')
ax3.tick_params(axis='x', rotation=45)
ax3.grid(axis='y', alpha=0.3)
ax3.set_ylim([0.6, 0.85])

# 4. Number of Features vs Accuracy
ax4 = axes[1, 1]
n_features = [model_data['n_features'] for model_data in models_dict.values()]
test_accuracy = [all_metrics[model]['accuracy'] for model in models_dict.keys()]

scatter = ax4.scatter(n_features, test_accuracy, s=200, c=range(len(models_dict)), 
                     cmap='viridis', alpha=0.6, edgecolors='black', linewidths=2)

for i, (n_feat, acc, model_name) in enumerate(zip(n_features, test_accuracy, models_dict.keys())):
    ax4.annotate(model_name, (n_feat, acc), xytext=(5, 5), 
                textcoords='offset points', fontsize=9)

ax4.set_xlabel('Number of Features', fontsize=12)
ax4.set_ylabel('Test Accuracy', fontsize=12)
ax4.set_title('Model Complexity vs Accuracy', fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "12_comprehensive_metrics_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 12_comprehensive_metrics_comparison.png")
plt.close()

# ============================================================================
# DETAILED CLASSIFICATION REPORTS
# ============================================================================
print("\n" + "="*80)
print("9. GENERATING DETAILED CLASSIFICATION REPORTS")
print("="*80)

report_path = Path("outputs/12_classification_reports.txt")

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE MODEL EVALUATION - DETAILED CLASSIFICATION REPORTS\n")
    f.write("="*80 + "\n\n")
    
    for model_name, metrics in all_metrics.items():
        f.write("\n" + "="*80 + "\n")
        f.write(f"{model_name.upper()}\n")
        f.write("="*80 + "\n\n")
        
        # Basic metrics
        f.write("Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['f1_score']:.4f}\n\n")
        
        # AUC scores
        f.write("AUC Scores:\n")
        f.write("-" * 60 + "\n")
        for i, class_name in enumerate(target_names):
            f.write(f"{class_name:15s}: {metrics['auc_per_class'][i]:.4f}\n")
        f.write(f"{'Micro-Average':15s}: {metrics['auc_micro']:.4f}\n\n")
        
        # Cross-validation results
        f.write("10-Fold Cross-Validation:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Mean Accuracy: {cv_results[model_name]['mean']:.4f} ± {cv_results[model_name]['std']:.4f}\n")
        f.write("Fold Scores: " + ", ".join([f"{score:.4f}" for score in cv_results[model_name]['scores']]) + "\n\n")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, metrics['y_pred'])
        f.write("Confusion Matrix:\n")
        f.write("-" * 60 + "\n")
        f.write("               Predicted\n")
        f.write("             " + "  ".join([f"{name:10s}" for name in target_names]) + "\n")
        for i, class_name in enumerate(target_names):
            f.write(f"Actual {class_name:10s}  " + "  ".join([f"{cm[i][j]:10d}" for j in range(n_classes)]) + "\n")
        f.write("\n")
        
        # Detailed classification report
        f.write("Detailed Classification Report:\n")
        f.write("-" * 60 + "\n")
        report = classification_report(y_test, metrics['y_pred'], 
                                      target_names=target_names, 
                                      zero_division=0)
        f.write(report + "\n")
        
        # Feature information
        f.write("Feature Selection:\n")
        f.write("-" * 60 + "\n")
        f.write(f"Number of features: {models_dict[model_name]['n_features']}\n")
        f.write(f"Features: {', '.join(models_dict[model_name]['features'][:10])}")
        if len(models_dict[model_name]['features']) > 10:
            f.write(f", ... (+{len(models_dict[model_name]['features'])-10} more)")
        f.write("\n\n")

print(f"✓ Saved: 12_classification_reports.txt")

# ============================================================================
# SUMMARY COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("10. CREATING SUMMARY COMPARISON TABLE")
print("="*80)

# Create summary dataframe
summary_data = []
for model_name in models_dict.keys():
    summary_data.append({
        'Model': model_name,
        'Features': models_dict[model_name]['n_features'],
        'Accuracy': f"{all_metrics[model_name]['accuracy']:.4f}",
        'Precision': f"{all_metrics[model_name]['precision']:.4f}",
        'Recall': f"{all_metrics[model_name]['recall']:.4f}",
        'F1-Score': f"{all_metrics[model_name]['f1_score']:.4f}",
        'AUC (Micro)': f"{all_metrics[model_name]['auc_micro']:.4f}",
        'CV Mean': f"{cv_results[model_name]['mean']:.4f}",
        'CV Std': f"{cv_results[model_name]['std']:.4f}"
    })

summary_df = pd.DataFrame(summary_data)

# Save to CSV
csv_path = Path("outputs/12_model_evaluation_summary.csv")
summary_df.to_csv(csv_path, index=False)
print(f"✓ Saved: 12_model_evaluation_summary.csv")

# Create visual table
fig, ax = plt.subplots(figsize=(14, 6))
ax.axis('tight')
ax.axis('off')

table = ax.table(cellText=summary_df.values,
                colLabels=summary_df.columns,
                cellLoc='center',
                loc='center',
                colWidths=[0.15, 0.08, 0.09, 0.09, 0.09, 0.09, 0.11, 0.09, 0.09])

table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2)

# Style header
for i in range(len(summary_df.columns)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Style rows
for i in range(1, len(summary_df) + 1):
    for j in range(len(summary_df.columns)):
        if i % 2 == 0:
            table[(i, j)].set_facecolor('#f0f0f0')
        else:
            table[(i, j)].set_facecolor('white')

plt.title('Model Evaluation Summary - All Metrics', fontsize=16, fontweight='bold', pad=20)
plt.savefig(figures_dir / "12_model_evaluation_summary_table.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 12_model_evaluation_summary_table.png")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)

print("\nGenerated Files:")
print("-" * 60)
print("Visualizations:")
print("  1. 12_all_models_confusion_matrices.png")
print("  2. 12_all_models_roc_curves.png")
print("  3. 12_cross_validation_results.png")
print("  4. 12_comprehensive_metrics_comparison.png")
print("  5. 12_model_evaluation_summary_table.png")
print("\nReports:")
print("  6. 12_classification_reports.txt")
print("  7. 12_model_evaluation_summary.csv")

print("\n" + "="*80)
print("KEY FINDINGS:")
print("="*80)

# Find best model
best_accuracy_model = max(all_metrics.items(), key=lambda x: x[1]['accuracy'])
best_auc_model = max(all_metrics.items(), key=lambda x: x[1]['auc_micro'])
best_cv_model = max(cv_results.items(), key=lambda x: x[1]['mean'])

print(f"\nBest Test Accuracy:  {best_accuracy_model[0]} ({best_accuracy_model[1]['accuracy']:.4f})")
print(f"Best AUC Score:      {best_auc_model[0]} ({best_auc_model[1]['auc_micro']:.4f})")
print(f"Best CV Accuracy:    {best_cv_model[0]} ({best_cv_model[1]['mean']:.4f})")

print("\nModel Rankings by Test Accuracy:")
sorted_models = sorted(all_metrics.items(), key=lambda x: x[1]['accuracy'], reverse=True)
for rank, (model_name, metrics) in enumerate(sorted_models, 1):
    print(f"  {rank}. {model_name:20s} - {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")

print("\n" + "="*80)
print("All evaluations completed successfully!")
print("="*80)
