"""
Explainable AI Analysis
Requirement 10: Implement model explainability using SHAP and LIME

Techniques:
- SHAP (SHapley Additive exPlanations) for tree-based models
- LIME (Local Interpretable Model-agnostic Explanations) for neural network
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import shap
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "explainability").mkdir(exist_ok=True)

# Load dataset
print("\n" + "="*80)
print("REQUIREMENT 10: EXPLAINABLE AI ANALYSIS")
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

feature_names = X.columns.tolist()

print(f"\nDataset: {len(df)} students")
print(f"Features: {X.shape[1]}")

# Split data (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Load scaler for neural network
scaler = joblib.load(OUTPUT_DIR / "models" / "scaler.pkl")
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train: {len(X_train)}, Test: {len(X_test)}")

# Load trained models
print("\nLoading trained models...")
models = {
    'Random Forest': joblib.load(OUTPUT_DIR / "models" / "random_forest.pkl"),
    'XGBoost': joblib.load(OUTPUT_DIR / "models" / "xgboost.pkl"),
    'Neural Network': joblib.load(OUTPUT_DIR / "models" / "neural_network.pkl")
}
print("✓ Models loaded successfully")

# ============================================================================
# SHAP ANALYSIS FOR RANDOM FOREST
# ============================================================================
print("\n\n" + "="*80)
print("SHAP ANALYSIS: RANDOM FOREST")
print("="*80)

rf_model = models['Random Forest']

# Create SHAP explainer
print("\nCreating SHAP explainer for Random Forest...")
explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_test)

# SHAP values format: (n_samples, n_features, n_classes)
# We'll focus on class 0 (Dropout) for most visualizations

# 1. Summary plot (beeswarm) - shows feature importance and impact
print("Generating SHAP summary plot...")
plt.figure(figsize=(12, 8))
# Extract dropout class (index 0) - shape becomes (n_samples, n_features)
shap_dropout = shap_values_rf[:, :, 0] if len(shap_values_rf.shape) == 3 else shap_values_rf[0]
shap.summary_plot(shap_dropout, X_test.values, feature_names=feature_names, show=False, max_display=20)
plt.title('SHAP Summary Plot: Random Forest (Dropout Class)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "06_shap_rf_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '06_shap_rf_summary.png'}")

# 2. Bar plot - mean absolute SHAP values
print("Generating SHAP feature importance plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_dropout, X_test.values, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance: Random Forest (Dropout Class)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "06_shap_rf_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '06_shap_rf_importance.png'}")

# 3. Calculate mean SHAP values for all features
shap_importance_rf = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP_Dropout': np.abs(shap_values_rf[:, :, 0]).mean(axis=0),
    'Mean_Abs_SHAP_Enrolled': np.abs(shap_values_rf[:, :, 1]).mean(axis=0),
    'Mean_Abs_SHAP_Graduate': np.abs(shap_values_rf[:, :, 2]).mean(axis=0)
})
shap_importance_rf['Mean_Abs_SHAP_Overall'] = shap_importance_rf[['Mean_Abs_SHAP_Dropout', 'Mean_Abs_SHAP_Enrolled', 'Mean_Abs_SHAP_Graduate']].mean(axis=1)
shap_importance_rf = shap_importance_rf.sort_values('Mean_Abs_SHAP_Overall', ascending=False).reset_index(drop=True)
shap_importance_rf.to_csv(OUTPUT_DIR / "explainability" / "shap_rf_importance.csv", index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'explainability' / 'shap_rf_importance.csv'}")

# ============================================================================
# SHAP ANALYSIS FOR XGBOOST
# ============================================================================
print("\n" + "="*80)
print("SHAP ANALYSIS: XGBOOST")
print("="*80)

xgb_model = models['XGBoost']

# Create SHAP explainer
print("\nCreating SHAP explainer for XGBoost...")
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_values_xgb = explainer_xgb.shap_values(X_test)

# 1. Summary plot (beeswarm)
print("Generating SHAP summary plot...")
plt.figure(figsize=(12, 8))
# Extract dropout class - handle both 2D list and 3D array formats
shap_xgb_dropout = shap_values_xgb[:, :, 0] if len(shap_values_xgb.shape) == 3 else shap_values_xgb[0]
shap.summary_plot(shap_xgb_dropout, X_test.values, feature_names=feature_names, show=False, max_display=20)
plt.title('SHAP Summary Plot: XGBoost (Dropout Class)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "06_shap_xgb_summary.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '06_shap_xgb_summary.png'}")

# 2. Bar plot
print("Generating SHAP feature importance plot...")
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_xgb_dropout, X_test.values, feature_names=feature_names, plot_type="bar", show=False, max_display=20)
plt.title('SHAP Feature Importance: XGBoost (Dropout Class)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "06_shap_xgb_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '06_shap_xgb_importance.png'}")

# 3. Calculate mean SHAP values
shap_importance_xgb = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_SHAP_Dropout': np.abs(shap_values_xgb[:, :, 0]).mean(axis=0),
    'Mean_Abs_SHAP_Enrolled': np.abs(shap_values_xgb[:, :, 1]).mean(axis=0),
    'Mean_Abs_SHAP_Graduate': np.abs(shap_values_xgb[:, :, 2]).mean(axis=0)
})
shap_importance_xgb['Mean_Abs_SHAP_Overall'] = shap_importance_xgb[['Mean_Abs_SHAP_Dropout', 'Mean_Abs_SHAP_Enrolled', 'Mean_Abs_SHAP_Graduate']].mean(axis=1)
shap_importance_xgb = shap_importance_xgb.sort_values('Mean_Abs_SHAP_Overall', ascending=False).reset_index(drop=True)
shap_importance_xgb.to_csv(OUTPUT_DIR / "explainability" / "shap_xgb_importance.csv", index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'explainability' / 'shap_xgb_importance.csv'}")

# ============================================================================
# LIME ANALYSIS FOR NEURAL NETWORK
# ============================================================================
print("\n" + "="*80)
print("LIME ANALYSIS: NEURAL NETWORK")
print("="*80)

nn_model = models['Neural Network']

# Create LIME explainer
print("\nCreating LIME explainer for Neural Network...")
explainer_lime = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification',
    random_state=42
)

# Explain 10 random test instances
print("Generating LIME explanations for sample predictions...")
n_samples = 10
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

lime_explanations = []

for i, idx in enumerate(sample_indices):
    instance = X_test_scaled[idx]
    true_label = y_test.iloc[idx]
    pred_label = nn_model.predict([instance])[0]
    pred_proba = nn_model.predict_proba([instance])[0]
    
    # Generate explanation
    exp = explainer_lime.explain_instance(
        instance,
        nn_model.predict_proba,
        num_features=10,
        top_labels=3
    )
    
    lime_explanations.append({
        'instance_id': idx,
        'true_label': class_names[true_label],
        'predicted_label': class_names[pred_label],
        'dropout_prob': pred_proba[0],
        'enrolled_prob': pred_proba[1],
        'graduate_prob': pred_proba[2],
        'explanation': exp
    })
    
    # Save individual explanation figure
    fig = exp.as_pyplot_figure(label=pred_label)
    fig.suptitle(f'LIME Explanation: Instance {idx}\nTrue: {class_names[true_label]}, Predicted: {class_names[pred_label]}', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "explainability" / f"lime_nn_instance_{idx}.png", dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  Instance {i+1}/{n_samples}: True={class_names[true_label]}, Pred={class_names[pred_label]}")

print(f"  ✓ Saved {n_samples} LIME explanations to: {OUTPUT_DIR / 'explainability'}")

# ============================================================================
# AGGREGATE LIME FEATURE IMPORTANCE
# ============================================================================
print("\nAggregating LIME feature importance across samples...")

# Collect feature weights from all explanations
all_lime_weights = {feature: [] for feature in feature_names}

for exp_dict in lime_explanations:
    exp = exp_dict['explanation']
    # Get explanation for predicted class
    pred_class = class_names.index(exp_dict['predicted_label'])
    feature_weights = dict(exp.as_list(label=pred_class))
    
    for feature_name in feature_names:
        # LIME returns feature names with conditions (e.g., "Age <= 25")
        # Extract the actual feature name
        weight = 0
        for key, value in feature_weights.items():
            if feature_name in key:
                weight = abs(value)
                break
        all_lime_weights[feature_name].append(weight)

# Calculate mean absolute weights
lime_importance = pd.DataFrame({
    'Feature': feature_names,
    'Mean_Abs_LIME_Weight': [np.mean(all_lime_weights[f]) for f in feature_names]
}).sort_values('Mean_Abs_LIME_Weight', ascending=False).reset_index(drop=True)

lime_importance.to_csv(OUTPUT_DIR / "explainability" / "lime_nn_importance.csv", index=False)
print(f"  ✓ Saved: {OUTPUT_DIR / 'explainability' / 'lime_nn_importance.csv'}")

# Visualize LIME feature importance
plt.figure(figsize=(12, 8))
top_20_lime = lime_importance.head(20).sort_values('Mean_Abs_LIME_Weight')
colors = plt.cm.viridis(top_20_lime['Mean_Abs_LIME_Weight'] / top_20_lime['Mean_Abs_LIME_Weight'].max())

plt.barh(range(len(top_20_lime)), top_20_lime['Mean_Abs_LIME_Weight'], color=colors, edgecolor='black')
plt.yticks(range(len(top_20_lime)), top_20_lime['Feature'], fontsize=9)
plt.xlabel('Mean Absolute LIME Weight', fontsize=12, fontweight='bold')
plt.title('LIME Feature Importance: Neural Network\n(Aggregated from 10 Sample Predictions)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "06_lime_nn_importance.png", dpi=300, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved: {OUTPUT_DIR / 'figures' / '06_lime_nn_importance.png'}")

# ============================================================================
# SAVE COMPREHENSIVE REPORT
# ============================================================================
with open(OUTPUT_DIR / "06_explainability_report.txt", 'w') as f:
    f.write("EXPLAINABLE AI ANALYSIS REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("Methods Applied:\n")
    f.write("  1. SHAP (SHapley Additive exPlanations) - Random Forest & XGBoost\n")
    f.write("  2. LIME (Local Interpretable Model-agnostic Explanations) - Neural Network\n\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHAP ANALYSIS - RANDOM FOREST\n")
    f.write("-"*80 + "\n")
    f.write("Top 15 Features by Mean Absolute SHAP Value (Overall):\n\n")
    for idx, row in shap_importance_rf.head(15).iterrows():
        f.write(f"{idx+1:2d}. {row['Feature']:<45} {row['Mean_Abs_SHAP_Overall']:.4f}\n")
    
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("SHAP ANALYSIS - XGBOOST\n")
    f.write("-"*80 + "\n")
    f.write("Top 15 Features by Mean Absolute SHAP Value (Overall):\n\n")
    for idx, row in shap_importance_xgb.head(15).iterrows():
        f.write(f"{idx+1:2d}. {row['Feature']:<45} {row['Mean_Abs_SHAP_Overall']:.4f}\n")
    
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("LIME ANALYSIS - NEURAL NETWORK\n")
    f.write("-"*80 + "\n")
    f.write("Top 15 Features by Mean Absolute LIME Weight:\n\n")
    for idx, row in lime_importance.head(15).iterrows():
        f.write(f"{idx+1:2d}. {row['Feature']:<45} {row['Mean_Abs_LIME_Weight']:.4f}\n")
    
    f.write("\n\n" + "="*80 + "\n\n")
    f.write("OUTPUTS GENERATED:\n")
    f.write("-"*80 + "\n")
    f.write("SHAP Visualizations:\n")
    f.write("  - 06_shap_rf_summary.png (Random Forest beeswarm plot)\n")
    f.write("  - 06_shap_rf_importance.png (Random Forest bar plot)\n")
    f.write("  - 06_shap_xgb_summary.png (XGBoost beeswarm plot)\n")
    f.write("  - 06_shap_xgb_importance.png (XGBoost bar plot)\n\n")
    f.write("LIME Visualizations:\n")
    f.write("  - 06_lime_nn_importance.png (Aggregated feature importance)\n")
    f.write(f"  - lime_nn_instance_*.png ({n_samples} individual explanations)\n\n")
    f.write("CSV Files:\n")
    f.write("  - shap_rf_importance.csv (Random Forest SHAP values)\n")
    f.write("  - shap_xgb_importance.csv (XGBoost SHAP values)\n")
    f.write("  - lime_nn_importance.csv (Neural Network LIME weights)\n\n")
    f.write("="*80 + "\n")

print(f"\n✓ Comprehensive report saved to: {OUTPUT_DIR / '06_explainability_report.txt'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n\n" + "="*80)
print("EXPLAINABILITY ANALYSIS COMPLETE")
print("="*80)

print("\nSHAP Analysis (Random Forest):")
print(f"  Top 3 Features: {', '.join(shap_importance_rf.head(3)['Feature'].tolist())}")

print("\nSHAP Analysis (XGBoost):")
print(f"  Top 3 Features: {', '.join(shap_importance_xgb.head(3)['Feature'].tolist())}")

print("\nLIME Analysis (Neural Network):")
print(f"  Top 3 Features: {', '.join(lime_importance.head(3)['Feature'].tolist())}")

print(f"\n✓ All visualizations saved to: {OUTPUT_DIR / 'figures'}")
print(f"✓ All CSV files saved to: {OUTPUT_DIR / 'explainability'}")
print("="*80 + "\n")
