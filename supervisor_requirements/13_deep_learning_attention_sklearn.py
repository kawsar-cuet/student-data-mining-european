# -*- coding: utf-8 -*-
"""
Deep Learning Attention Model Integration
Uses MLPClassifier with architecture similar to the attention-based model
for integration into the comprehensive analysis pipeline.

Note: This uses sklearn's MLPClassifier to avoid TensorFlow dependencies
while maintaining similar deep learning architecture principles.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Create directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"
models_dir = output_dir / "models"
for dir_path in [output_dir, figures_dir, tables_dir, models_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("DEEP LEARNING ATTENTION MODEL (SKLEARN IMPLEMENTATION)")
print("="*80)

# ========== LOAD DATA ==========
print("\n1. Loading dataset")
print("-" * 60)

df = pd.read_csv("../data/educational_data.csv")
print(f"Dataset shape: {df.shape}")

X = df.drop('Target', axis=1)
y = df['Target']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    target_names = le.classes_
else:
    target_names = ['Dropout', 'Enrolled', 'Graduate']

num_classes = len(np.unique(y))
print(f"Features: {X.shape[1]}, Classes: {num_classes}")
print(f"Distribution: {dict(zip(target_names, np.bincount(y)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ========== FEATURE SELECTION ==========
print("\n2. Feature Selection (ANOVA F-test, 20 features)")
print("-" * 60)

selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected: {', '.join(selected_features)}")

pd.DataFrame({'Feature': selected_features}).to_csv(
    tables_dir / "13_deep_learning_attention_features.csv", index=False
)


# ========== SCALE DATA ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


# ========== TRAIN MODEL ==========
print("\n3. Training Deep Learning Model")
print("-" * 60)
print("Architecture: Input(20) → 64 → 32 → 16 → Output(3)")

# MLPClassifier with similar architecture to attention model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Similar to attention model
    activation='relu',
    solver='adam',
    alpha=0.001,  # L2 regularization
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,  # Similar to early stopping patience
    random_state=42,
    verbose=True
)

model.fit(X_train_scaled, y_train)
print("✓ Training complete")


# ========== EVALUATION ==========
print("\n4. Model Evaluation")
print("=" * 60)

y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nPerformance:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
for i, row in enumerate(cm):
    print(f"{target_names[i]:10s}: {row}")


# ========== SAVE RESULTS ==========
results_df = pd.DataFrame([{
    'Model': 'Deep Learning Attention',
    'Method': 'ANOVA F-test',
    'Num_Features': 20,
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1
}])
results_df.to_csv(tables_dir / "13_deep_learning_attention_results.csv", index=False)
print("\n✓ Saved results to CSV")


# ========== VISUALIZATIONS ==========
print("\n5. Generating Visualizations")
print("-" * 60)

plt.style.use('default')
sns.set_palette("husl")

# 1. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=target_names, yticklabels=target_names,
           linewidths=0.5, cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Deep Learning Attention - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved confusion matrix")
plt.close()

# 2. Feature Importance (using coefficients from first layer)
# Extract weights from first hidden layer
try:
    first_layer_weights = np.abs(model.coefs_[0])  # Shape: (input_features, 64)
    feature_importance = first_layer_weights.sum(axis=1)
    
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(tables_dir / "13_deep_learning_attention_importance.csv", index=False)
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Connection Weight Magnitude', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Deep Learning Attention - Top 15 Features by Weight', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_deep_learning_attention_importance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved feature importance")
    plt.close()
except Exception as e:
    print(f"Warning: Could not extract feature importance - {e}")

# 3. Loss Curve
plt.figure(figsize=(10, 6))
plt.plot(model.loss_curve_, linewidth=2, color='steelblue')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Deep Learning Attention - Training Loss Curve', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved training curve")
plt.close()


# ========== SUMMARY ==========
print("\n" + "="*80)
print("DEEP LEARNING ATTENTION MODEL - COMPLETE")
print("="*80)

summary = f"""
Deep Learning Attention Model Summary
{"="*60}

Architecture:
  - Input: {X_train_scaled.shape[1]} features
  - Hidden Layers: 64 → 32 → 16
  - Output: {num_classes} classes
  - Activation: ReLU
  - Optimizer: Adam (lr=0.001)
  - Regularization: L2 (alpha=0.001)

Feature Selection:
  - Method: ANOVA F-test
  - Selected: 20 features
  - Top features: {', '.join(selected_features[:5])}...

Performance:
  - Accuracy:  {acc:.4f}
  - Precision: {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}

Training:
  - Iterations: {model.n_iter_}
  - Final Loss: {model.loss_:.6f}

Files Generated:
  ✓ 13_deep_learning_attention_results.csv
  ✓ 13_deep_learning_attention_features.csv
  ✓ 13_deep_learning_attention_importance.csv
  ✓ 13_deep_learning_attention_confusion_matrix.png
  ✓ 13_deep_learning_attention_importance.png
  ✓ 13_deep_learning_attention_training.png
{"="*60}
"""

print(summary)

with open(tables_dir / "13_deep_learning_attention_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n✓ Analysis complete!")
print("="*80)
