# -*- coding: utf-8 -*-
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path

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

# Feature selection - Information Gain (10 features)
print("Selecting features...")
from sklearn.feature_selection import mutual_info_classif
mi_selector = SelectKBest(mutual_info_classif, k=10)
X_train_dt = mi_selector.fit_transform(X_train, y_train)
dt_features = X_train.columns[mi_selector.get_support()].tolist()
print(f"Selected features ({len(dt_features)}): {dt_features}")

# Convert to DataFrame
X_train_dt = pd.DataFrame(X_train_dt, columns=dt_features)

# Train model
print("\nTraining Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, min_samples_leaf=10, random_state=42)
dt_model.fit(X_train_dt, y_train)
print("Decision Tree trained")

# Sample for SHAP
np.random.seed(42)
sample_size = 100
X_train_sample_indices = np.random.choice(X_train_dt.shape[0], sample_size, replace=False)
X_train_dt_sample = X_train_dt.iloc[X_train_sample_indices]

print(f"\nX_train_dt_sample shape: {X_train_dt_sample.shape}")
print(f"Columns: {X_train_dt_sample.columns.tolist()}")

# SHAP
print("\nCreating SHAP TreeExplainer...")
explainer_dt = shap.TreeExplainer(dt_model)
print("Computing SHAP values...")
shap_values_dt = explainer_dt.shap_values(X_train_dt_sample)

print(f"\nSHAP values type: {type(shap_values_dt)}")
print(f"Is list: {isinstance(shap_values_dt, list)}")

if isinstance(shap_values_dt, list):
    print(f"List length: {len(shap_values_dt)}")
    for i, sv in enumerate(shap_values_dt):
        print(f"  Class {i} ({target_names[i]}) shape: {sv.shape}")
    
    shap_dropout = shap_values_dt[0]
    print(f"\nExtracted Dropout class (shap_values_dt[0]) shape: {shap_dropout.shape}")
    print(f"Expected: (100, 10)")
    print(f"Match: {shap_dropout.shape == (100, 10)}")
else:
    print(f"SHAP values shape: {shap_values_dt.shape}")
    print(f"Number of dimensions: {len(shap_values_dt.shape)}")
    if len(shap_values_dt.shape) == 3:
        print(f"3D array format: (samples={shap_values_dt.shape[0]}, features={shap_values_dt.shape[1]}, classes={shap_values_dt.shape[2]})")
