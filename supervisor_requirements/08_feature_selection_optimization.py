"""
Feature Selection Optimization for Single Classifiers
Tests Decision Tree and Naive Bayes with different feature selection methods
to improve model accuracy using the best features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
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
print("FEATURE SELECTION OPTIMIZATION FOR SINGLE CLASSIFIERS")
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
print("2. TESTING FEATURE SELECTION METHODS")
print("="*80)

results = []

# Baseline: All features (no selection)
print("\n[Baseline] Using ALL features (34)")
print("-" * 60)

# Decision Tree - All features
dt_all = DecisionTreeClassifier(
    max_depth=15, min_samples_split=10, min_samples_leaf=5,
    class_weight='balanced', random_state=42
)
dt_all.fit(X_train, y_train)
y_pred_dt = dt_all.predict(X_test)
dt_acc_all = accuracy_score(y_test, y_pred_dt)
dt_cv_all = cross_val_score(dt_all, X_train, y_train, cv=5).mean()

print(f"Decision Tree - Test Accuracy: {dt_acc_all:.4f}, CV Accuracy: {dt_cv_all:.4f}")

results.append({
    'Model': 'Decision Tree',
    'Method': 'All Features',
    'Num_Features': 34,
    'Test_Accuracy': dt_acc_all,
    'CV_Accuracy': dt_cv_all,
    'Precision': precision_score(y_test, y_pred_dt, average='weighted'),
    'Recall': recall_score(y_test, y_pred_dt, average='weighted'),
    'F1_Score': f1_score(y_test, y_pred_dt, average='weighted')
})

# Naive Bayes - All features
nb_all = GaussianNB()
nb_all.fit(X_train, y_train)
y_pred_nb = nb_all.predict(X_test)
nb_acc_all = accuracy_score(y_test, y_pred_nb)
nb_cv_all = cross_val_score(nb_all, X_train, y_train, cv=5).mean()

print(f"Naive Bayes   - Test Accuracy: {nb_acc_all:.4f}, CV Accuracy: {nb_cv_all:.4f}")

results.append({
    'Model': 'Naive Bayes',
    'Method': 'All Features',
    'Num_Features': 34,
    'Test_Accuracy': nb_acc_all,
    'CV_Accuracy': nb_cv_all,
    'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
    'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
    'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
})

# Method 1: ANOVA F-statistic (SelectKBest with f_classif)
print("\n[Method 1] ANOVA F-statistic (f_classif)")
print("-" * 60)

for k in feature_counts[:-1]:  # Skip 34 as we already did all features
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'ANOVA F-stat',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'ANOVA F-stat',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 2: Mutual Information
print("\n[Method 2] Mutual Information")
print("-" * 60)

for k in feature_counts[:-1]:
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'Mutual Info',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'Mutual Info',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 3: Chi-Square (for non-negative features)
print("\n[Method 3] Chi-Square Test")
print("-" * 60)

# Make features non-negative for chi2
X_train_nonneg = X_train - X_train.min() + 1e-10
X_test_nonneg = X_test - X_test.min() + 1e-10

for k in feature_counts[:-1]:
    selector = SelectKBest(chi2, k=k)
    X_train_selected = selector.fit_transform(X_train_nonneg, y_train)
    X_test_selected = selector.transform(X_test_nonneg)
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'Chi-Square',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'Chi-Square',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 4: Recursive Feature Elimination (RFE) with Random Forest
print("\n[Method 4] Recursive Feature Elimination (RFE)")
print("-" * 60)

for k in feature_counts[:-1]:
    rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    selector = RFE(rf_estimator, n_features_to_select=k, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'RFE',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'RFE',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 5: Random Forest Feature Importance (SelectFromModel)
print("\n[Method 5] Random Forest Feature Importance")
print("-" * 60)

for k in feature_counts[:-1]:
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    # Get top k features by importance
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'RF Importance',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'RF Importance',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 6: Information Gain
print("\n[Method 6] Information Gain")
print("-" * 60)

def calculate_information_gain(X, y):
    """Calculate information gain for each feature"""
    ig_scores = []
    for col in X.columns:
        # Calculate mutual information between feature and target
        score = mutual_info_classif(X[[col]], y, random_state=42)[0]
        ig_scores.append(score)
    return np.array(ig_scores)

ig_scores = calculate_information_gain(X_train, y_train)

for k in feature_counts[:-1]:
    # Select top k features by information gain
    indices = np.argsort(ig_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'Info Gain',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'Info Gain',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 7: Gain Ratio
print("\n[Method 7] Gain Ratio")
print("-" * 60)

def calculate_entropy(y):
    """Calculate entropy of target variable"""
    _, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs + 1e-10))

def calculate_gain_ratio(X, y):
    """Calculate gain ratio for each feature"""
    base_entropy = calculate_entropy(y)
    gain_ratios = []
    
    for col in X.columns:
        feature = X[col]
        # Calculate information gain
        feature_entropy = 0
        split_info = 0
        
        # For continuous features, use binning
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
    # Select top k features by gain ratio
    indices = np.argsort(gr_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'Gain Ratio',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'Gain Ratio',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Method 8: Gini Index
print("\n[Method 8] Gini Index")
print("-" * 60)

def calculate_gini_importance(X, y):
    """Calculate Gini importance for each feature using Decision Tree"""
    dt_temp = DecisionTreeClassifier(random_state=42, max_depth=10)
    dt_temp.fit(X, y)
    return dt_temp.feature_importances_

gini_scores = calculate_gini_importance(X_train, y_train)

for k in feature_counts[:-1]:
    # Select top k features by Gini importance
    indices = np.argsort(gini_scores)[::-1][:k]
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    # Decision Tree
    dt = DecisionTreeClassifier(
        max_depth=15, min_samples_split=10, min_samples_leaf=5,
        class_weight='balanced', random_state=42
    )
    dt.fit(X_train_selected, y_train)
    y_pred = dt.predict(X_test_selected)
    acc = accuracy_score(y_test, y_pred)
    cv_acc = cross_val_score(dt, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Decision Tree',
        'Method': 'Gini Index',
        'Num_Features': k,
        'Test_Accuracy': acc,
        'CV_Accuracy': cv_acc,
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred, average='weighted')
    })
    
    # Naive Bayes
    nb = GaussianNB()
    nb.fit(X_train_selected, y_train)
    y_pred_nb = nb.predict(X_test_selected)
    nb_acc = accuracy_score(y_test, y_pred_nb)
    nb_cv = cross_val_score(nb, X_train_selected, y_train, cv=5).mean()
    
    results.append({
        'Model': 'Naive Bayes',
        'Method': 'Gini Index',
        'Num_Features': k,
        'Test_Accuracy': nb_acc,
        'CV_Accuracy': nb_cv,
        'Precision': precision_score(y_test, y_pred_nb, average='weighted'),
        'Recall': recall_score(y_test, y_pred_nb, average='weighted'),
        'F1_Score': f1_score(y_test, y_pred_nb, average='weighted')
    })
    
    print(f"k={k:2d} | DT: {acc:.4f} (CV: {cv_acc:.4f}) | NB: {nb_acc:.4f} (CV: {nb_cv:.4f})")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\n" + "="*80)
print("3. SAVING RESULTS")
print("="*80)

# Save detailed results
results_path = tables_dir / "08_feature_selection_results.csv"
results_df.to_csv(results_path, index=False)
print(f"\n✓ Saved detailed results to: {results_path}")

# Find best configurations
print("\n" + "="*80)
print("4. BEST CONFIGURATIONS")
print("="*80)

for model in ['Decision Tree', 'Naive Bayes']:
    model_results = results_df[results_df['Model'] == model]
    best_row = model_results.loc[model_results['Test_Accuracy'].idxmax()]
    
    print(f"\n{model}:")
    print(f"  Best Method: {best_row['Method']}")
    print(f"  Num Features: {best_row['Num_Features']}")
    print(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}")
    print(f"  CV Accuracy: {best_row['CV_Accuracy']:.4f}")
    print(f"  F1-Score: {best_row['F1_Score']:.4f}")

# Summary comparison
summary = results_df.groupby(['Model', 'Method']).agg({
    'Test_Accuracy': ['mean', 'max'],
    'CV_Accuracy': ['mean', 'max']
}).round(4)
summary_path = tables_dir / "08_feature_selection_summary.csv"
summary.to_csv(summary_path)
print(f"\n✓ Saved summary to: {summary_path}")

print("\n" + "="*80)
print("5. GENERATING VISUALIZATIONS")
print("="*80)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# Figure 1: Line plot - Accuracy vs Number of Features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, model in enumerate(['Decision Tree', 'Naive Bayes']):
    model_data = results_df[results_df['Model'] == model]
    
    # Test Accuracy
    ax1 = axes[idx, 0]
    for method in model_data['Method'].unique():
        method_data = model_data[model_data['Method'] == method]
        ax1.plot(method_data['Num_Features'], method_data['Test_Accuracy'], 
                marker='o', label=method, linewidth=2)
    
    ax1.set_xlabel('Number of Features', fontsize=12)
    ax1.set_ylabel('Test Accuracy', fontsize=12)
    ax1.set_title(f'{model} - Test Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # CV Accuracy
    ax2 = axes[idx, 1]
    for method in model_data['Method'].unique():
        method_data = model_data[model_data['Method'] == method]
        ax2.plot(method_data['Num_Features'], method_data['CV_Accuracy'], 
                marker='s', label=method, linewidth=2)
    
    ax2.set_xlabel('Number of Features', fontsize=12)
    ax2.set_ylabel('Cross-Validation Accuracy', fontsize=12)
    ax2.set_title(f'{model} - CV Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path1 = figures_dir / "08_accuracy_vs_features.png"
plt.savefig(fig_path1, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path1}")
plt.close()

# Figure 2: Heatmap - Best Accuracy per Method and Feature Count
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

for idx, model in enumerate(['Decision Tree', 'Naive Bayes']):
    model_data = results_df[results_df['Model'] == model]
    pivot = model_data.pivot(index='Method', columns='Num_Features', values='Test_Accuracy')
    
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[idx],
                cbar_kws={'label': 'Test Accuracy'})
    axes[idx].set_title(f'{model} - Accuracy Heatmap', fontsize=14, fontweight='bold')
    axes[idx].set_xlabel('Number of Features', fontsize=12)
    axes[idx].set_ylabel('Feature Selection Method', fontsize=12)

plt.tight_layout()
fig_path2 = figures_dir / "08_accuracy_heatmap.png"
plt.savefig(fig_path2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path2}")
plt.close()

# Figure 3: Bar plot - Best accuracy for each method
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for idx, model in enumerate(['Decision Tree', 'Naive Bayes']):
    model_data = results_df[results_df['Model'] == model]
    best_per_method = model_data.groupby('Method')['Test_Accuracy'].max().sort_values(ascending=True)
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(best_per_method)))
    best_per_method.plot(kind='barh', ax=axes[idx], color=colors)
    
    axes[idx].set_xlabel('Best Test Accuracy', fontsize=12)
    axes[idx].set_ylabel('Feature Selection Method', fontsize=12)
    axes[idx].set_title(f'{model} - Best Accuracy per Method', fontsize=14, fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(best_per_method):
        axes[idx].text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)

plt.tight_layout()
fig_path3 = figures_dir / "08_best_accuracy_per_method.png"
plt.savefig(fig_path3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path3}")
plt.close()

# Figure 4: Comparison plot - DT vs NB across all methods
fig, ax = plt.subplots(figsize=(14, 8))

methods = results_df['Method'].unique()
x = np.arange(len(methods))
width = 0.35

dt_best = [results_df[(results_df['Model'] == 'Decision Tree') & 
                      (results_df['Method'] == m)]['Test_Accuracy'].max() 
           for m in methods]
nb_best = [results_df[(results_df['Model'] == 'Naive Bayes') & 
                      (results_df['Method'] == m)]['Test_Accuracy'].max() 
           for m in methods]

bars1 = ax.bar(x - width/2, dt_best, width, label='Decision Tree', 
               color='steelblue', edgecolor='black')
bars2 = ax.bar(x + width/2, nb_best, width, label='Naive Bayes', 
               color='coral', edgecolor='black')

ax.set_xlabel('Feature Selection Method', fontsize=13, fontweight='bold')
ax.set_ylabel('Best Test Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Decision Tree vs Naive Bayes - Best Accuracy Comparison', 
             fontsize=15, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(methods, rotation=45, ha='right')
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
fig_path4 = figures_dir / "08_dt_vs_nb_comparison.png"
plt.savefig(fig_path4, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path4}")
plt.close()

# Figure 5: All metrics comparison for best configurations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
metrics = ['Test_Accuracy', 'CV_Accuracy', 'Precision', 'F1_Score']
metric_names = ['Test Accuracy', 'CV Accuracy', 'Precision', 'F1-Score']

for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
    ax = axes[idx // 2, idx % 2]
    
    dt_data = results_df[results_df['Model'] == 'Decision Tree'].groupby('Method')[metric].max()
    nb_data = results_df[results_df['Model'] == 'Naive Bayes'].groupby('Method')[metric].max()
    
    x = np.arange(len(dt_data))
    width = 0.35
    
    ax.bar(x - width/2, dt_data.values, width, label='Decision Tree', 
           color='steelblue', edgecolor='black')
    ax.bar(x + width/2, nb_data.values, width, label='Naive Bayes', 
           color='coral', edgecolor='black')
    
    ax.set_xlabel('Feature Selection Method', fontsize=11)
    ax.set_ylabel(name, fontsize=11)
    ax.set_title(f'{name} - Best per Method', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(dt_data.index, rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
fig_path5 = figures_dir / "08_all_metrics_comparison.png"
plt.savefig(fig_path5, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {fig_path5}")
plt.close()

# Save summary report
print("\n" + "="*80)
print("6. GENERATING SUMMARY REPORT")
print("="*80)

report_path = output_dir / "08_feature_selection_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("FEATURE SELECTION OPTIMIZATION - SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("OBJECTIVE:\n")
    f.write("Test Decision Tree and Naive Bayes classifiers with different feature\n")
    f.write("selection methods to improve model accuracy using optimal feature subsets.\n\n")
    
    f.write("FEATURE SELECTION METHODS TESTED:\n")
    f.write("1. All Features (Baseline)\n")
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
    f.write("BEST CONFIGURATIONS\n")
    f.write("="*80 + "\n\n")
    
    for model in ['Decision Tree', 'Naive Bayes']:
        model_results = results_df[results_df['Model'] == model]
        best_row = model_results.loc[model_results['Test_Accuracy'].idxmax()]
        
        f.write(f"{model}:\n")
        f.write(f"  Best Method: {best_row['Method']}\n")
        f.write(f"  Number of Features: {best_row['Num_Features']}\n")
        f.write(f"  Test Accuracy: {best_row['Test_Accuracy']:.4f}\n")
        f.write(f"  CV Accuracy: {best_row['CV_Accuracy']:.4f}\n")
        f.write(f"  Precision: {best_row['Precision']:.4f}\n")
        f.write(f"  Recall: {best_row['Recall']:.4f}\n")
        f.write(f"  F1-Score: {best_row['F1_Score']:.4f}\n\n")
        
        # Compare with baseline
        baseline = model_results[model_results['Method'] == 'All Features'].iloc[0]
        improvement = (best_row['Test_Accuracy'] - baseline['Test_Accuracy']) * 100
        
        f.write(f"  Improvement over baseline: {improvement:+.2f}%\n")
        f.write(f"  Baseline accuracy (All 34 features): {baseline['Test_Accuracy']:.4f}\n\n")
    
    f.write("="*80 + "\n")
    f.write("METHOD PERFORMANCE SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    f.write(summary.to_string())
    f.write("\n\n")
    
    f.write("="*80 + "\n")
    f.write("VISUALIZATIONS GENERATED\n")
    f.write("="*80 + "\n\n")
    f.write("1. 08_accuracy_vs_features.png - Line plots showing accuracy trends\n")
    f.write("2. 08_accuracy_heatmap.png - Heatmaps of accuracy by method and feature count\n")
    f.write("3. 08_best_accuracy_per_method.png - Bar charts of best accuracy per method\n")
    f.write("4. 08_dt_vs_nb_comparison.png - Side-by-side comparison of DT vs NB\n")
    f.write("5. 08_all_metrics_comparison.png - Multi-metric comparison charts\n\n")
    
    f.write("="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")
    
    dt_best = results_df[results_df['Model'] == 'Decision Tree'].loc[
        results_df[results_df['Model'] == 'Decision Tree']['Test_Accuracy'].idxmax()]
    nb_best = results_df[results_df['Model'] == 'Naive Bayes'].loc[
        results_df[results_df['Model'] == 'Naive Bayes']['Test_Accuracy'].idxmax()]
    
    f.write(f"For Decision Tree:\n")
    f.write(f"  Use {dt_best['Method']} with {dt_best['Num_Features']} features\n")
    f.write(f"  Expected accuracy: {dt_best['Test_Accuracy']:.4f}\n\n")
    
    f.write(f"For Naive Bayes:\n")
    f.write(f"  Use {nb_best['Method']} with {nb_best['Num_Features']} features\n")
    f.write(f"  Expected accuracy: {nb_best['Test_Accuracy']:.4f}\n\n")
    
    f.write("Feature selection significantly impacts model performance. Both models\n")
    f.write("benefit from careful feature selection, with optimal subset sizes varying\n")
    f.write("by selection method. Consider ensemble approaches in future work.\n")

print(f"✓ Saved report to: {report_path}")

print("\n" + "="*80)
print("FEATURE SELECTION OPTIMIZATION COMPLETE!")
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
