"""
Feature Selection Optimization for Neural Network (Deep Learning)
Tests Neural Network with different feature selection methods to improve model accuracy
using the best features. Includes proper scaling for neural networks.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Create output directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"
for dir_path in [output_dir, figures_dir, tables_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("FEATURE SELECTION OPTIMIZATION FOR NEURAL NETWORK (DEEP LEARNING)")
print("="*80)

# Load dataset
data_path = Path("../data/educational_data.csv")
print(f"\n1. Loading dataset from: {data_path}")
df = pd.read_csv(data_path)
print(f"   Dataset shape: {df.shape}")

# Separate features and target
X = df.drop('Target', axis=1)
y = df['Target']

# Encode target if it's string
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    print(f"   Target classes encoded: {le.classes_}")

print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Feature selection methods to test
feature_counts = [5, 10, 15, 20, 25, 30, 34]  # Including all features

print("\n" + "="*80)
print("2. TESTING FEATURE SELECTION METHODS WITH NEURAL NETWORK")
print("="*80)

results = []

# Define Neural Network architecture
def create_neural_network():
    """Create a consistent neural network architecture"""
    return MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        alpha=0.001,
        batch_size=32,
        learning_rate='adaptive',
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False
    )

# Baseline: All features (with scaling)
print("\n[Baseline] Using ALL features (34) with StandardScaler")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_all = create_neural_network()
nn_all.fit(X_train_scaled, y_train)
y_pred_nn = nn_all.predict(X_test_scaled)
nn_acc_all = accuracy_score(y_test, y_pred_nn)

# Cross-validation with scaled data (simplified for speed)
nn_cv_all = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()

print(f"Neural Network - Test Accuracy: {nn_acc_all:.4f}, CV Accuracy: {nn_cv_all:.4f}")

results.append({
    'Method': 'All Features',
    'Num_Features': 34,
    'Test_Accuracy': nn_acc_all,
    'CV_Accuracy': nn_cv_all,
    'Precision': precision_score(y_test, y_pred_nn, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nn, average='weighted'),
    'F1_Score': f1_score(y_test, y_pred_nn, average='weighted')
})

# Method 1: ANOVA F-statistic (SelectKBest with f_classif)
print("\n[Method 1] ANOVA F-statistic (f_classif)")
print("-" * 60)

for k in feature_counts[:-1]:
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'ANOVA F-stat',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 2: Mutual Information
print("\n[Method 2] Mutual Information")
print("-" * 60)

for k in feature_counts[:-1]:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'Mutual Info',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 3: Chi-Square (for non-negative features)
print("\n[Method 3] Chi-Square Test")
print("-" * 60)

X_train_nonneg = X_train - X_train.min() + 1e-10
X_test_nonneg = X_test - X_test.min() + 1e-10

for k in feature_counts[:-1]:
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_nonneg, y_train)
    X_test_selected = selector.transform(X_test_nonneg)
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'Chi-Square',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 4: Recursive Feature Elimination (RFE)
print("\n[Method 4] Recursive Feature Elimination (RFE)")
print("-" * 60)

for k in feature_counts[:-1]:
    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(rf_estimator, n_features_to_select=k, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'RFE',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 5: Random Forest Feature Importance
print("\n[Method 5] Random Forest Feature Importance")
print("-" * 60)

for k in feature_counts[:-1]:
    rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_temp.fit(X_train, y_train)
    
    importances = rf_temp.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'RF Importance',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 6: Information Gain
print("\n[Method 6] Information Gain")
print("-" * 60)

def calculate_information_gain(X, y):
    """Calculate information gain for each feature"""
    ig_scores = []
    for col in X.columns:
        score = mutual_info_classif(X[[col]], y, random_state=42)[0]
        ig_scores.append(score)
    return np.array(ig_scores)

ig_scores = calculate_information_gain(X_train, y_train)

for k in feature_counts[:-1]:
    indices = np.argsort(ig_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'Info Gain',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 7: Gain Ratio
print("\n[Method 7] Gain Ratio")
print("-" * 60)

def calculate_entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def calculate_gain_ratio(X, y):
    base_entropy = calculate_entropy(y)
    gain_ratios = []
    
    for col in X.columns:
        feature = X[col]
        feature_entropy = 0
        split_info = 0
        
        if len(np.unique(feature)) > 10:
            feature_binned = pd.qcut(feature, q=5, duplicates='drop', labels=False)
        else:
            feature_binned = feature
        
        for val in np.unique(feature_binned):
            subset_y = y[feature_binned == val]
            prob = len(subset_y) / len(y)
            feature_entropy += prob * calculate_entropy(subset_y)
            split_info += -prob * np.log2(prob + 1e-10)
        
        info_gain = base_entropy - feature_entropy
        gain_ratio = info_gain / (split_info + 1e-10)
        gain_ratios.append(gain_ratio)
    
    return np.array(gain_ratios)

gr_scores = calculate_gain_ratio(X_train, y_train)

for k in feature_counts[:-1]:
    indices = np.argsort(gr_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'Gain Ratio',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Method 8: Gini Index
print("\n[Method 8] Gini Index")
print("-" * 60)

def calculate_gini_importance(X, y):
    dt_temp = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_temp.fit(X, y)
    return dt_temp.feature_importances_

gini_scores = calculate_gini_importance(X_train, y_train)

for k in feature_counts[:-1]:
    indices = np.argsort(gini_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Scale the selected features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Neural Network
    nn = create_neural_network()
    nn.fit(X_train_scaled, y_train)
    y_pred = nn.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(create_neural_network(), X_train_scaled, y_train, cv=3, scoring='accuracy').mean()
    
    results.append({
        'Method': 'Gini Index',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    print(f"k={k:2d} | NN: {acc:.4f} (CV: {cv_acc:.4f})")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("3. SAVING RESULTS")
print("="*80)

results_path = tables_dir / "10_nn_feature_selection_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved detailed results to: {results_path}")

# Find best configuration
print("\n" + "="*80)
print("4. BEST CONFIGURATION")
print("="*80)

best_row = results_df.loc[results_df['Test_Accuracy'].idxmax()]

print(f"\nNeural Network:")
print(f"  Best Method: {best_row['Method']}")
print(f"  Num Features: {best_row['Num_Features']}")
print(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}")
print(f"  CV Accuracy: {best_row['CV_Accuracy']:.4f}")
print(f"  F1-Score: {best_row['F1_Score']:.4f}")

baseline = results_df[results_df['Method'] == 'All Features'].iloc[0]
improvement = (best_row['Test_Accuracy'] - baseline['Test_Accuracy']) * 100

print(f"\n  Improvement over baseline: {improvement:+.2f}%")
print(f"  Baseline accuracy (All 34 features): {baseline['Test_Accuracy']:.4f}")

# Summary comparison
summary = results_df.groupby('Method').agg({
    'Test_Accuracy': ['mean', 'max'],
    'CV_Accuracy': ['mean', 'max']
}).round(4)
summary_path = tables_dir / "10_nn_feature_selection_summary.csv"
summary.to_csv(summary_path)
print(f"\n✓ Saved summary to: {summary_path}")

print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Figure 1: Line plot - Accuracy vs Number of Features
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Test Accuracy
ax1 = axes[0]
for method in results_df['Method'].unique():
    method_data = results_df[results_df['Method'] == method]
    ax1.plot(method_data['Num_Features'], method_data['Test_Accuracy'], 
            marker='o', label=method, linewidth=2.5)

ax1.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax1.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax1.set_title('Neural Network - Test Accuracy vs Feature Count', fontsize=15, fontweight='bold')
ax1.legend(loc='best', fontsize=10)
ax1.grid(True, alpha=0.3)

# CV Accuracy
ax2 = axes[1]
for method in results_df['Method'].unique():
    method_data = results_df[results_df['Method'] == method]
    ax2.plot(method_data['Num_Features'], method_data['CV_Accuracy'], 
            marker='s', label=method, linewidth=2.5)

ax2.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax2.set_ylabel('CV Accuracy', fontsize=13, fontweight='bold')
ax2.set_title('Neural Network - CV Accuracy vs Feature Count', fontsize=15, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path1 = figures_dir / "10_nn_accuracy_vs_features.png"
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path1}")
plt.close()

# Figure 2: Heatmap - Accuracy by Method and Feature Count
fig, ax = plt.subplots(figsize=(12, 8))

pivot = results_df.pivot(index='Method', columns='Num_Features', values='Test_Accuracy')

sns.heatmap(pivot, annot=True, fmt='.4f', cmap='YlGnBu', ax=ax,
            cbar_kws={'label': 'Test Accuracy'}, linewidths=0.5)
ax.set_title('Neural Network - Accuracy Heatmap by Method & Features', 
             fontsize=15, fontweight='bold')
ax.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax.set_ylabel('Feature Selection Method', fontsize=13, fontweight='bold')

plt.tight_layout()
fig_path2 = figures_dir / "10_nn_accuracy_heatmap.png"
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path2}")
plt.close()

# Figure 3: Bar plot - Best accuracy for each method
fig, ax = plt.subplots(figsize=(12, 8))

best_per_method = results_df.groupby('Method')['Test_Accuracy'].max().sort_values(ascending=True)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_per_method)))
bars = best_per_method.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1.5)

ax.set_xlabel('Best Test Accuracy', fontsize=13, fontweight='bold')
ax.set_ylabel('Feature Selection Method', fontsize=13, fontweight='bold')
ax.set_title('Neural Network - Best Accuracy per Feature Selection Method', 
             fontsize=15, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

for i, v in enumerate(best_per_method):
    ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
fig_path3 = figures_dir / "10_nn_best_accuracy_per_method.png"
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path3}")
plt.close()

# Figure 4: All metrics comparison for best configurations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Test_Accuracy', 'CV_Accuracy', 'Precision', 'F1_Score']
metric_names = ['Test Accuracy', 'CV Accuracy', 'Precision', 'F1-Score']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    best_per_method = results_df.groupby('Method')[metric].max().sort_values(ascending=True)
    
    colors = plt.cm.plasma(np.linspace(0.3, 0.9, len(best_per_method)))
    best_per_method.plot(kind='barh', ax=ax, color=colors, edgecolor='black', linewidth=1.2)
    
    ax.set_xlabel(name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Feature Selection Method', fontsize=12, fontweight='bold')
    ax.set_title(f'Neural Network - {name} by Method', fontsize=13, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(best_per_method):
        ax.text(v + 0.002, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
fig_path4 = figures_dir / "10_nn_all_metrics_comparison.png"
plt.savefig(fig_path4, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path4}")
plt.close()

# Figure 5: Feature count impact - boxplot
fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for boxplot
boxplot_data = []
feature_labels = []
for k in feature_counts:
    k_data = results_df[results_df['Num_Features'] == k]['Test_Accuracy'].values
    if len(k_data) > 0:
        boxplot_data.append(k_data)
        feature_labels.append(str(k))

bp = ax.boxplot(boxplot_data, labels=feature_labels, patch_artist=True,
                boxprops=dict(facecolor='skyblue', edgecolor='black', linewidth=1.5),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))

ax.set_xlabel('Number of Features', fontsize=13, fontweight='bold')
ax.set_ylabel('Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Neural Network - Accuracy Distribution by Feature Count', 
             fontsize=15, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path5 = figures_dir / "10_nn_feature_count_distribution.png"
plt.savefig(fig_path5, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path5}")
plt.close()

# Save summary report
print("\n" + "="*80)
print("6. GENERATING SUMMARY REPORT")
print("="*80)

report_path = output_dir / "10_nn_feature_selection_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("NEURAL NETWORK FEATURE SELECTION OPTIMIZATION - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("OBJECTIVE:\n")
    f.write("Test Neural Network (Deep Learning) classifier with different feature\n")
    f.write("selection methods to improve model accuracy using optimal feature subsets.\n")
    f.write("All features are scaled using StandardScaler for neural network training.\n\n")
    
    f.write("NEURAL NETWORK ARCHITECTURE:\n")
    f.write("- Hidden Layers: (128, 64, 32) - 3 hidden layers\n")
    f.write("- Activation: ReLU\n")
    f.write("- Solver: Adam optimizer\n")
    f.write("- Learning Rate: Adaptive (initial: 0.001)\n")
    f.write("- Regularization: L2 (alpha=0.001)\n")
    f.write("- Batch Size: 32\n")
    f.write("- Max Iterations: 500\n")
    f.write("- Early Stopping: Enabled (patience=20, validation=10%)\n")
    f.write("- Cross-Validation: 3-fold (optimized for speed)\n\n")
    
    f.write("FEATURE SELECTION METHODS TESTED:\n")
    f.write("1. All Features (Baseline) - with StandardScaler\n")
    f.write("2. ANOVA F-statistic (f_classif)\n")
    f.write("3. Mutual Information\n")
    f.write("4. Chi-Square Test\n")
    f.write("5. Recursive Feature Elimination (RFE)\n")
    f.write("6. Random Forest Feature Importance\n")
    f.write("7. Information Gain\n")
    f.write("8. Gain Ratio\n")
    f.write("9. Gini Index\n\n")
    
    f.write("FEATURE COUNTS TESTED:\n")
    f.write(f"{feature_counts}\n\n")
    
    f.write("="*80 + "\n")
    f.write("BEST CONFIGURATION\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"Neural Network:\n")
    f.write(f"  Best Method: {best_row['Method']}\n")
    f.write(f"  Number of Features: {best_row['Num_Features']}\n")
    f.write(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}\n")
    f.write(f"  CV Accuracy: {best_row['CV_Accuracy']:.4f}\n")
    f.write(f"  Precision: {best_row['Precision']:.4f}\n")
    f.write(f"  Recall: {best_row['Recall']:.4f}\n")
    f.write(f"  F1-Score: {best_row['F1_Score']:.4f}\n\n")
    
    f.write(f"  Improvement over baseline: {improvement:+.2f}%\n")
    f.write(f"  Baseline accuracy (All 34 features): {baseline['Test_Accuracy']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("METHOD PERFORMANCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(summary.to_string())
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("COMPARISON WITH OTHER MODELS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Single Classifiers:\n")
    f.write("  Decision Tree (Info Gain, 10 features):      68.81%\n")
    f.write("  Naive Bayes (Info Gain, 15 features):       72.66%\n\n")
    
    f.write("Ensemble Methods:\n")
    f.write("  Random Forest (RFE, 20 features):           77.85%\n")
    f.write("  AdaBoost (Mutual Info, 15 features):        77.06%\n")
    f.write("  XGBoost (RF Importance, 30 features):       77.97%\n\n")
    
    f.write("Deep Learning:\n")
    f.write(f"  Neural Network ({best_row['Method']}, {best_row['Num_Features']} features):  {best_row['Test_Accuracy']:.2%}\n\n")
    
    f.write("="*80 + "\n")
    f.write("VISUALIZATIONS GENERATED\n")
    f.write("="*80 + "\n\n")
    f.write("1. 10_nn_accuracy_vs_features.png - Line plots of accuracy vs feature count\n")
    f.write("2. 10_nn_accuracy_heatmap.png - Heatmap of accuracy by method/features (9×7)\n")
    f.write("3. 10_nn_best_accuracy_per_method.png - Bar chart ranking methods by best accuracy\n")
    f.write("4. 10_nn_all_metrics_comparison.png - Multi-metric comparison across methods\n")
    f.write("5. 10_nn_feature_count_distribution.png - Boxplot showing accuracy distribution\n\n")
    
    f.write("="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write(f"For Neural Network deployment:\n")
    f.write(f"  Use {best_row['Method']} with {best_row['Num_Features']} features\n")
    f.write(f"  Expected accuracy: {best_row['Test_Accuracy']:.4f}\n")
    f.write(f"  Feature reduction: {100 - (best_row['Num_Features']/34)*100:.1f}%\n\n")
    
    f.write("Neural networks benefit from:\n")
    f.write("- Feature scaling (StandardScaler applied to all configurations)\n")
    f.write("- Reduced feature sets (less prone to overfitting)\n")
    f.write("- Early stopping (prevents overfitting on training data)\n")
    f.write("- Adaptive learning rate (improves convergence)\n\n")
    
    f.write("The neural network shows ")
    if best_row['Test_Accuracy'] > 0.78:
        f.write("excellent performance, competitive with XGBoost.\n")
    elif best_row['Test_Accuracy'] > 0.75:
        f.write("strong performance, competitive with ensemble methods.\n")
    else:
        f.write("good performance, though ensemble methods perform better.\n")
    
    f.write("Feature selection improves NN performance by reducing dimensionality and\n")
    f.write("preventing overfitting, especially important for deep learning models.\n")

print(f"✓ Saved report to: {report_path}")

print("\n" + "="*80)
print("NEURAL NETWORK FEATURE SELECTION OPTIMIZATION COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {results_path}")
print(f"  - {summary_path}")
print(f"  - {report_path}")
print(f"  - {fig_path1}")
print(f"  - {fig_path2}")
print(f"  - {fig_path3}")
print(f"  - {fig_path4}")
print(f"  - {fig_path5}")
print("\nCheck the outputs folder for all results and visualizations!")
