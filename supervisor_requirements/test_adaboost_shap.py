# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import AdaBoostClassifier
from pathlib import Path

OUTPUT_DIR = Path("outputs")
figures_dir = OUTPUT_DIR / "figures"

# Load data
df = pd.read_csv("../data/educational_data.csv")
target_col = 'Target'
X = df.drop(columns=[target_col])
y = df[target_col]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)
target_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Feature selection - Mutual Info (15 features)
print("Selecting features...")
mi_selector = SelectKBest(mutual_info_classif, k=15)
X_train_ada = mi_selector.fit_transform(X_train, y_train)
ada_features = X_train.columns[mi_selector.get_support()].tolist()
print(f"Selected features: {ada_features}")

# Train model
print("Training AdaBoost...")
ada_model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
ada_model.fit(X_train_ada, y_train)
print(f"AdaBoost trained")

# Sample for SHAP
np.random.seed(42)
sample_size = 100
X_train_sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train_ada_sample = X_train_ada[X_train_sample_indices]
X_train_ada_df = pd.DataFrame(X_train_ada_sample, columns=ada_features)

print(f"\nX_train_ada_df shape: {X_train_ada_df.shape}")
print(f"Columns: {X_train_ada_df.columns.tolist()}")

# SHAP
print("\nCreating SHAP explainer...")
explainer_ada = shap.KernelExplainer(ada_model.predict_proba, shap.sample(X_train_ada_df, 50))
print("Computing SHAP values (this may take a while)...")
shap_values_ada = explainer_ada.shap_values(X_train_ada_df[:50])

print(f"\nSHAP values type: {type(shap_values_ada)}")
print(f"Is list: {isinstance(shap_values_ada, list)}")

if isinstance(shap_values_ada, list):
    print(f"List length: {len(shap_values_ada)}")
    for i, sv in enumerate(shap_values_ada):
        print(f"  Class {i} ({target_names[i]}) shape: {sv.shape}")
    
    shap_dropout = shap_values_ada[0]
    print(f"\nExtracted Dropout class shape: {shap_dropout.shape}")
    print(f"Expected: (50, 15)")
    print(f"Match: {shap_dropout.shape == (50, 15)}")
    
    # Try generating plot
    print("\nGenerating SHAP summary plot...")
    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_dropout, X_train_ada_df[:50].values, 
                     feature_names=X_train_ada_df.columns.tolist(), 
                     show=False, max_display=15)
    plt.title("AdaBoost - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(figures_dir / "TEST_ada_summary.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: TEST_ada_summary.png")
    plt.close()
else:
    print(f"SHAP values shape: {shap_values_ada.shape}")
