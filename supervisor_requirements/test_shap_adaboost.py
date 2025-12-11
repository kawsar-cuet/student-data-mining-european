import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Load data
df = pd.read_csv("../data/educational_data.csv")
target_col = 'Target'
X = df.drop(columns=[target_col])
y = df[target_col]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load AdaBoost model and feature selector
OUTPUT_DIR = Path("outputs")
ada_model = joblib.load(OUTPUT_DIR / "models" / "11_adaboost_mutual_info.pkl")
ada_selector = joblib.load(OUTPUT_DIR / "models" / "11_adaboost_mutual_info_selector.pkl")

# Get selected features
X_train_ada = X_train.iloc[:, ada_selector.get_support()]
ada_features = X_train.columns[ada_selector.get_support()].tolist()

print(f"AdaBoost features ({len(ada_features)}):")
print(ada_features)

# Sample data
np.random.seed(42)
sample_size = 100
X_train_sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
X_train_ada_sample = X_train_ada.iloc[X_train_sample_indices]
X_train_ada_df = pd.DataFrame(X_train_ada_sample, columns=ada_features)

print(f"\nX_train_ada_df shape: {X_train_ada_df.shape}")
print(f"X_train_ada_df columns: {X_train_ada_df.columns.tolist()}")

# Create SHAP explainer
print("\nCreating SHAP KernelExplainer...")
explainer_ada = shap.KernelExplainer(ada_model.predict_proba, 
                                     shap.sample(X_train_ada_df, 50))
print("Computing SHAP values...")
shap_values_ada = explainer_ada.shap_values(X_train_ada_df[:50])

print(f"\nSHAP values type: {type(shap_values_ada)}")
print(f"Is list: {isinstance(shap_values_ada, list)}")

if isinstance(shap_values_ada, list):
    print(f"Number of elements in list: {len(shap_values_ada)}")
    for i, sv in enumerate(shap_values_ada):
        print(f"  Element {i} shape: {sv.shape}")
        print(f"  Element {i} type: {type(sv)}")
else:
    print(f"SHAP values shape: {shap_values_ada.shape}")

print("\n" + "="*60)
print("Testing extraction:")
if isinstance(shap_values_ada, list):
    shap_dropout = shap_values_ada[0]
    print(f"shap_dropout (shap_values_ada[0]) shape: {shap_dropout.shape}")
    print(f"Expected: (50, 15)")
    print(f"Match: {shap_dropout.shape == (50, 15)}")
