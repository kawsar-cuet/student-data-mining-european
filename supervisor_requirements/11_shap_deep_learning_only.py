# -*- coding: utf-8 -*-
"""
Quick SHAP Analysis for Deep Learning Attention Model Only
Generates SHAP explanations specifically for the Deep Learning Attention model.
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
import shap
import warnings
warnings.filterwarnings('ignore')

# UTF-8 encoding setup
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
for dir_path in [output_dir, figures_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("SHAP ANALYSIS - DEEP LEARNING ATTENTION MODEL")
print("="*80)

# Load dataset
df = pd.read_csv("../data/educational_data.csv")
X = df.drop('Target', axis=1)
y = df['Target']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    target_names = le.classes_
else:
    target_names = ['Dropout', 'Enrolled', 'Graduate']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n1. Feature Selection (ANOVA F-test, 20 features)")
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

dl_features = X_train.columns[selector.get_support()].tolist()
print(f"   Selected: {', '.join(dl_features[:5])}... ({len(dl_features)} total)")

# Scale data
scaler = StandardScaler()
X_train_dl = scaler.fit_transform(X_train_selected)
X_test_dl = scaler.transform(X_test_selected)

print(f"\n2. Training Deep Learning Attention Model")
dl_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=32,
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=200,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=15,
    random_state=42,
    verbose=False
)
dl_model.fit(X_train_dl, y_train)
dl_accuracy = dl_model.score(X_test_dl, y_test)
print(f"   Accuracy: {dl_accuracy:.4f}")

print(f"\n3. Generating SHAP Explanations (this may take a few minutes...)")
X_train_dl_df = pd.DataFrame(X_train_dl, columns=dl_features)

# Use smaller sample for faster SHAP computation
sample_size = 30  # Reduced from 50 for faster computation
X_sample = X_train_dl_df[:sample_size]

print(f"   Creating KernelExplainer with {sample_size} background samples...")
background = shap.kmeans(X_train_dl_df, 10)  # Use k-means for better background
explainer_dl = shap.KernelExplainer(dl_model.predict_proba, background)

print(f"   Computing SHAP values for {sample_size} samples...")
shap_values_dl = explainer_dl.shap_values(X_sample)

print(f"\n4. Creating Visualizations")

# Feature Importance Bar Plot
plt.figure(figsize=(12, 8))
if isinstance(shap_values_dl, list):
    shap.summary_plot(shap_values_dl, X_sample, plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_dl, X_sample, plot_type="bar", show=False)
plt.title("Deep Learning Attention - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_deep_learning_importance.png", dpi=300, bbox_inches='tight')
print("   [OK] Saved: 11_shap_deep_learning_importance.png")
plt.close()

# Summary Plot (Beeswarm) for Dropout Class
plt.figure(figsize=(12, 8))
if isinstance(shap_values_dl, list):
    shap_dropout = shap_values_dl[0]
else:
    shap_dropout = shap_values_dl[:, :, 0] if len(shap_values_dl.shape) == 3 else shap_values_dl
shap.summary_plot(shap_dropout, X_sample.values, feature_names=X_sample.columns.tolist(), 
                  show=False, max_display=20)
plt.title("Deep Learning Attention - SHAP Summary Plot (Dropout Class)", 
         fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_deep_learning_summary.png", dpi=300, bbox_inches='tight')
print("   [OK] Saved: 11_shap_deep_learning_summary.png")
plt.close()

print("\n" + "="*80)
print("SHAP ANALYSIS COMPLETE!")
print("="*80)
print(f"\nModel Performance:")
print(f"  - Architecture: 64 → 32 → 16")
print(f"  - Features: {len(dl_features)}")
print(f"  - Accuracy: {dl_accuracy:.4f}")
print(f"\nGenerated Files:")
print(f"  - 11_shap_deep_learning_importance.png")
print(f"  - 11_shap_deep_learning_summary.png")
print("="*80)
