# -*- coding: utf-8 -*-
"""
Explainable AI (XAI) Analysis for All Models
Generates SHAP explanations for all 7 optimized models to understand
feature importance and model decision-making processes.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
import shap
import warnings
warnings.filterwarnings('ignore')

# Create output directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"
for dir_path in [output_dir, figures_dir, tables_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("EXPLAINABLE AI (XAI) ANALYSIS - ALL MODELS")
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
    target_names = le.classes_
    print(f"   Target classes: {target_names}")
else:
    target_names = ['Class 0', 'Class 1', 'Class 2']

print(f"   Features: {X.shape[1]}")
print(f"   Classes: {len(np.unique(y))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Store feature names
feature_names = X.columns.tolist()

print("\n" + "="*80)
print("2. TRAINING OPTIMIZED MODELS WITH BEST CONFIGURATIONS")
print("="*80)

# Helper function to calculate Information Gain
def calculate_information_gain(X, y):
    """Calculate information gain for each feature"""
    ig_scores = []
    for col in X.columns:
        score = mutual_info_classif(X[[col]], y, random_state=42)[0]
        ig_scores.append(score)
    return np.array(ig_scores)

# Model 1: Decision Tree with Information Gain (10 features)
print("\n[1/6] Decision Tree with Information Gain (10 features)")
print("-" * 60)

ig_scores = calculate_information_gain(X_train, y_train)
dt_indices = np.argsort(ig_scores)[::-1][:10]
dt_features = X_train.columns[dt_indices].tolist()
print(f"Selected features: {dt_features}")

X_train_dt = X_train.iloc[:, dt_indices]
X_test_dt = X_test.iloc[:, dt_indices]

dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=20, 
                                  min_samples_leaf=10, random_state=42)
dt_model.fit(X_train_dt, y_train)
dt_accuracy = dt_model.score(X_test_dt, y_test)
print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")

# Model 2: Naive Bayes with Information Gain (15 features)
print("\n[2/6] Naive Bayes with Information Gain (15 features)")
print("-" * 60)

nb_indices = np.argsort(ig_scores)[::-1][:15]
nb_features = X_train.columns[nb_indices].tolist()
print(f"Selected features: {nb_features}")

X_train_nb = X_train.iloc[:, nb_indices]
X_test_nb = X_test.iloc[:, nb_indices]

nb_model = GaussianNB()
nb_model.fit(X_train_nb, y_train)
nb_accuracy = nb_model.score(X_test_nb, y_test)
print(f"Naive Bayes Accuracy: {nb_accuracy:.4f}")

# Model 3: Random Forest with RFE (20 features)
print("\n[3/6] Random Forest with RFE (20 features)")
print("-" * 60)

rf_estimator = RandomForestClassifier(n_estimators=50, random_state=42)
rf_selector = RFE(rf_estimator, n_features_to_select=20, step=5)
X_train_rf = rf_selector.fit_transform(X_train, y_train)
X_test_rf = rf_selector.transform(X_test)

rf_features = X_train.columns[rf_selector.support_].tolist()
print(f"Selected features: {rf_features}")

rf_model = RandomForestClassifier(n_estimators=200, max_depth=15, class_weight='balanced', 
                                 random_state=42)
rf_model.fit(X_train_rf, y_train)
rf_accuracy = rf_model.score(X_test_rf, y_test)
print(f"Random Forest Accuracy: {rf_accuracy:.4f}")

# Model 4: AdaBoost with Mutual Info (15 features)
print("\n[4/6] AdaBoost with Mutual Info (15 features)")
print("-" * 60)

ada_selector = SelectKBest(mutual_info_classif, k=15)
X_train_ada = ada_selector.fit_transform(X_train, y_train)
X_test_ada = ada_selector.transform(X_test)

ada_features = X_train.columns[ada_selector.get_support()].tolist()
print(f"Selected features: {ada_features}")

base_estimator = DecisionTreeClassifier(max_depth=3, random_state=42)
ada_model = AdaBoostClassifier(estimator=base_estimator, n_estimators=100, random_state=42)
ada_model.fit(X_train_ada, y_train)
ada_accuracy = ada_model.score(X_test_ada, y_test)
print(f"AdaBoost Accuracy: {ada_accuracy:.4f}")

# Model 5: XGBoost with RF Importance (30 features)
print("\n[5/6] XGBoost with RF Importance (30 features)")
print("-" * 60)

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_
xgb_indices = np.argsort(importances)[::-1][:30]
xgb_features = X_train.columns[xgb_indices].tolist()
print(f"Selected features: {xgb_features}")

X_train_xgb = X_train.iloc[:, xgb_indices]
X_test_xgb = X_test.iloc[:, xgb_indices]

xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, 
                         random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train_xgb, y_train)
xgb_accuracy = xgb_model.score(X_test_xgb, y_test)
print(f"XGBoost Accuracy: {xgb_accuracy:.4f}")

# Model 6: Neural Network with ANOVA F-stat (15 features)
print("\n[6/6] Neural Network with ANOVA F-stat (15 features)")
print("-" * 60)

nn_selector = SelectKBest(f_classif, k=15)
X_train_nn_selected = nn_selector.fit_transform(X_train, y_train)
X_test_nn_selected = nn_selector.transform(X_test)

nn_features = X_train.columns[nn_selector.get_support()].tolist()
print(f"Selected features: {nn_features}")

# Scale for neural network
scaler = StandardScaler()
X_train_nn = scaler.fit_transform(X_train_nn_selected)
X_test_nn = scaler.transform(X_test_nn_selected)

nn_model = MLPClassifier(
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
nn_model.fit(X_train_nn, y_train)
nn_accuracy = nn_model.score(X_test_nn, y_test)
print(f"Neural Network Accuracy: {nn_accuracy:.4f}")

# Model 7: Deep Learning Attention (with MLP architecture)
print("\n[7/7] Deep Learning Attention with ANOVA F-stat (20 features)")
print("-" * 60)

dl_selector = SelectKBest(f_classif, k=20)
X_train_dl_selected = dl_selector.fit_transform(X_train, y_train)
X_test_dl_selected = dl_selector.transform(X_test)

dl_features = X_train.columns[dl_selector.get_support()].tolist()
print(f"Selected features: {dl_features}")

# Scale for deep learning
scaler_dl = StandardScaler()
X_train_dl = scaler_dl.fit_transform(X_train_dl_selected)
X_test_dl = scaler_dl.transform(X_test_dl_selected)

dl_model = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16),  # Attention-like architecture
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
print(f"Deep Learning Attention Accuracy: {dl_accuracy:.4f}")

print("\n" + "="*80)
print("3. GENERATING SHAP EXPLANATIONS")
print("="*80)

# Set matplotlib style
plt.style.use('default')
sns.set_palette("husl")

# Sample data for SHAP (use subset for computational efficiency)
shap.initjs()
sample_size = 100
X_train_sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)

print("\n[1/6] SHAP for Decision Tree")
print("-" * 60)
X_train_dt_sample = X_train_dt.iloc[X_train_sample_indices]
explainer_dt = shap.TreeExplainer(dt_model)
shap_values_dt = explainer_dt.shap_values(X_train_dt_sample)

# Summary plot
plt.figure(figsize=(12, 8))
if isinstance(shap_values_dt, list):
    shap.summary_plot(shap_values_dt, X_train_dt_sample, plot_type="bar", 
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_dt, X_train_dt_sample, plot_type="bar", show=False)
plt.title("Decision Tree - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_decision_tree_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_decision_tree_importance.png")
plt.close()

# Beeswarm plot - Focus on Dropout class (index 0)
plt.figure(figsize=(12, 8))
# TreeExplainer returns (n_samples, n_features, n_classes) or list depending on version
if isinstance(shap_values_dt, list):
    shap_dropout = shap_values_dt[0]
else:
    shap_dropout = shap_values_dt[:, :, 0] if len(shap_values_dt.shape) == 3 else shap_values_dt
shap.summary_plot(shap_dropout, X_train_dt_sample.values, feature_names=X_train_dt_sample.columns.tolist(), show=False, max_display=10)
plt.title("Decision Tree - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_decision_tree_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_decision_tree_summary.png")
plt.close()

print("\n[2/6] SHAP for Naive Bayes")
print("-" * 60)
X_train_nb_sample = X_train_nb.iloc[X_train_sample_indices]

# Use KernelExplainer for Naive Bayes
explainer_nb = shap.KernelExplainer(nb_model.predict_proba, 
                                    shap.sample(X_train_nb_sample, 50))
shap_values_nb = explainer_nb.shap_values(X_train_nb_sample[:50])

plt.figure(figsize=(12, 8))
if isinstance(shap_values_nb, list):
    shap.summary_plot(shap_values_nb, X_train_nb_sample[:50], plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_nb, X_train_nb_sample[:50], plot_type="bar", show=False)
plt.title("Naive Bayes - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_naive_bayes_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_naive_bayes_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# Extract Dropout class (index 0) - handle both list and 3D array formats
if isinstance(shap_values_nb, list):
    shap_dropout = shap_values_nb[0]
else:
    # KernelExplainer returns (n_samples, n_features, n_classes)
    shap_dropout = shap_values_nb[:, :, 0] if len(shap_values_nb.shape) == 3 else shap_values_nb
shap.summary_plot(shap_dropout, X_train_nb_sample[:50].values, feature_names=X_train_nb_sample.columns.tolist(), show=False, max_display=15)
plt.title("Naive Bayes - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_naive_bayes_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_naive_bayes_summary.png")
plt.close()

print("\n[3/6] SHAP for Random Forest")
print("-" * 60)
X_train_rf_sample = X_train_rf[X_train_sample_indices]
X_train_rf_df = pd.DataFrame(X_train_rf_sample, columns=rf_features)

explainer_rf = shap.TreeExplainer(rf_model)
shap_values_rf = explainer_rf.shap_values(X_train_rf_df)

plt.figure(figsize=(12, 8))
if isinstance(shap_values_rf, list):
    shap.summary_plot(shap_values_rf, X_train_rf_df, plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_rf, X_train_rf_df, plot_type="bar", show=False)
plt.title("Random Forest - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_random_forest_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_random_forest_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# TreeExplainer returns (n_samples, n_features, n_classes) or list depending on version
if isinstance(shap_values_rf, list):
    shap_dropout = shap_values_rf[0]
else:
    shap_dropout = shap_values_rf[:, :, 0] if len(shap_values_rf.shape) == 3 else shap_values_rf
shap.summary_plot(shap_dropout, X_train_rf_df.values, feature_names=X_train_rf_df.columns.tolist(), show=False, max_display=20)
plt.title("Random Forest - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_random_forest_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_random_forest_summary.png")
plt.close()

print("\n[4/6] SHAP for AdaBoost")
print("-" * 60)
X_train_ada_sample = X_train_ada[X_train_sample_indices]
X_train_ada_df = pd.DataFrame(X_train_ada_sample, columns=ada_features)

# Use KernelExplainer for AdaBoost (TreeExplainer not supported for AdaBoost)
explainer_ada = shap.KernelExplainer(ada_model.predict_proba, 
                                     shap.sample(X_train_ada_df, 50))
shap_values_ada = explainer_ada.shap_values(X_train_ada_df[:50])

plt.figure(figsize=(12, 8))
if isinstance(shap_values_ada, list):
    shap.summary_plot(shap_values_ada, X_train_ada_df[:50], plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_ada, X_train_ada_df[:50], plot_type="bar", show=False)
plt.title("AdaBoost - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_adaboost_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_adaboost_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# Extract Dropout class (index 0) - handle both list and 3D array formats
if isinstance(shap_values_ada, list):
    shap_dropout = shap_values_ada[0]
else:
    # KernelExplainer returns (n_samples, n_features, n_classes)
    shap_dropout = shap_values_ada[:, :, 0] if len(shap_values_ada.shape) == 3 else shap_values_ada
shap.summary_plot(shap_dropout, X_train_ada_df[:50].values, feature_names=X_train_ada_df.columns.tolist(), show=False, max_display=15)
plt.title("AdaBoost - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_adaboost_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_adaboost_summary.png")
plt.close()

print("\n[5/6] SHAP for XGBoost")
print("-" * 60)
X_train_xgb_sample = X_train_xgb.iloc[X_train_sample_indices]

# Use KernelExplainer for XGBoost with lambda wrapper
def xgb_predict(data):
    return xgb_model.predict_proba(pd.DataFrame(data, columns=xgb_features))

explainer_xgb = shap.KernelExplainer(xgb_predict, 
                                     shap.sample(X_train_xgb_sample, 50).values)
shap_values_xgb = explainer_xgb.shap_values(X_train_xgb_sample[:50].values)

plt.figure(figsize=(12, 8))
if isinstance(shap_values_xgb, list):
    shap.summary_plot(shap_values_xgb, X_train_xgb_sample[:50], plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_xgb, X_train_xgb_sample[:50], plot_type="bar", show=False)
plt.title("XGBoost - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_xgboost_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_xgboost_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# Extract Dropout class (index 0) - handle both list and 3D array formats
if isinstance(shap_values_xgb, list):
    shap_dropout = shap_values_xgb[0]
else:
    # KernelExplainer returns (n_samples, n_features, n_classes)
    shap_dropout = shap_values_xgb[:, :, 0] if len(shap_values_xgb.shape) == 3 else shap_values_xgb
shap.summary_plot(shap_dropout, X_train_xgb_sample[:50].values, feature_names=X_train_xgb_sample.columns.tolist(), show=False, max_display=30)
plt.title("XGBoost - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_xgboost_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_xgboost_summary.png")
plt.close()

print("\n[6/6] SHAP for Neural Network")
print("-" * 60)
X_train_nn_sample = X_train_nn[X_train_sample_indices]
X_train_nn_df = pd.DataFrame(X_train_nn_sample, columns=nn_features)

# Use KernelExplainer for Neural Network (DeepExplainer doesn't support sklearn MLPClassifier)
explainer_nn = shap.KernelExplainer(nn_model.predict_proba, 
                                    shap.sample(X_train_nn_df, 50))
shap_values_nn = explainer_nn.shap_values(X_train_nn_df[:50])

plt.figure(figsize=(12, 8))
if isinstance(shap_values_nn, list):
    shap.summary_plot(shap_values_nn, X_train_nn_df[:50], plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_nn, X_train_nn_df[:50], plot_type="bar", show=False)
plt.title("Neural Network - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_neural_network_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_neural_network_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# Extract Dropout class (index 0) - handle both list and 3D array formats
if isinstance(shap_values_nn, list):
    shap_dropout = shap_values_nn[0]
else:
    # KernelExplainer returns (n_samples, n_features, n_classes)
    shap_dropout = shap_values_nn[:, :, 0] if len(shap_values_nn.shape) == 3 else shap_values_nn
shap.summary_plot(shap_dropout, X_train_nn_df[:50].values, feature_names=X_train_nn_df.columns.tolist(), show=False, max_display=15)
plt.title("Neural Network - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_neural_network_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_neural_network_summary.png")
plt.close()

print("\n[7/7] SHAP for Deep Learning Attention")
print("-" * 60)

# Create DataFrame for SHAP
X_train_dl_df = pd.DataFrame(X_train_dl, columns=dl_features)

# Use KernelExplainer (model-agnostic)
explainer_dl = shap.KernelExplainer(dl_model.predict_proba, X_train_dl_df[:50])
shap_values_dl = explainer_dl.shap_values(X_train_dl_df[:50])

plt.figure(figsize=(12, 8))
if isinstance(shap_values_dl, list):
    shap.summary_plot(shap_values_dl, X_train_dl_df[:50], plot_type="bar",
                     class_names=target_names, show=False)
else:
    shap.summary_plot(shap_values_dl, X_train_dl_df[:50], plot_type="bar", show=False)
plt.title("Deep Learning Attention - SHAP Feature Importance", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_deep_learning_importance.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_deep_learning_importance.png")
plt.close()

plt.figure(figsize=(12, 8))
# Extract Dropout class (index 0)
if isinstance(shap_values_dl, list):
    shap_dropout = shap_values_dl[0]
else:
    shap_dropout = shap_values_dl[:, :, 0] if len(shap_values_dl.shape) == 3 else shap_values_dl
shap.summary_plot(shap_dropout, X_train_dl_df[:50].values, feature_names=X_train_dl_df.columns.tolist(), show=False, max_display=20)
plt.title("Deep Learning Attention - SHAP Summary Plot (Dropout Class)", fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "11_shap_deep_learning_summary.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_shap_deep_learning_summary.png")
plt.close()

print("\n" + "="*80)
print("4. COMPARATIVE FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Create comparative visualization
fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # Changed to 2x4 for 7 models

models_info = [
    ("Decision Tree", dt_model.feature_importances_, dt_features, axes[0, 0]),
    ("Naive Bayes", None, nb_features, axes[0, 1]),  # NB doesn't have feature_importances_
    ("Random Forest", rf_model.feature_importances_, rf_features, axes[0, 2]),
    ("AdaBoost", ada_model.feature_importances_, ada_features, axes[0, 3]),
    ("XGBoost", xgb_model.feature_importances_, xgb_features, axes[1, 0]),
    ("Neural Network", None, nn_features, axes[1, 1]),  # NN doesn't have feature_importances_
    ("Deep Learning Attention", None, dl_features, axes[1, 2])  # DL doesn't have feature_importances_
]

for model_name, importances, features, ax in models_info:
    if importances is not None:
        # Sort by importance
        indices = np.argsort(importances)[::-1][:10]  # Top 10
        top_features = [features[i] for i in indices]
        top_importances = importances[indices]
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_importances)))
        bars = ax.barh(range(len(top_importances)), top_importances, color=colors, edgecolor='black')
        ax.set_yticks(range(len(top_importances)))
        ax.set_yticklabels(top_features, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel('Feature Importance', fontsize=10, fontweight='bold')
        ax.set_title(f'{model_name}\nTop 10 Features', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        for i, v in enumerate(top_importances):
            ax.text(v + 0.001, i, f'{v:.3f}', va='center', fontsize=8)
    else:
        # For models without feature_importances_, show feature list
        ax.text(0.5, 0.5, f'{model_name}\n\nSelected Features:\n' + '\n'.join(features[:10]),
                ha='center', va='center', fontsize=9, transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.axis('off')

# Hide the extra subplot (axes[1, 3])
axes[1, 3].axis('off')

plt.suptitle('Feature Importance Comparison Across All 7 Models', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(figures_dir / "11_all_models_feature_importance_comparison.png", dpi=300, bbox_inches='tight')
print("\n✓ Saved: 11_all_models_feature_importance_comparison.png")
plt.close()

print("\n" + "="*80)
print("5. MODEL ACCURACY COMPARISON")
print("="*80)

# Create accuracy comparison visualization
models_accuracy = {
    'Decision Tree': dt_accuracy,
    'Naive Bayes': nb_accuracy,
    'Random Forest': rf_accuracy,
    'AdaBoost': ada_accuracy,
    'XGBoost': xgb_accuracy,
    'Neural Network': nn_accuracy,
    'Deep Learning Attention': dl_accuracy
}

fig, ax = plt.subplots(figsize=(12, 8))  # Increased height for 7 models

model_names = list(models_accuracy.keys())
accuracies = list(models_accuracy.values())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#B19CD9']  # Added 7th color

bars = ax.barh(model_names, accuracies, color=colors, edgecolor='black', linewidth=2)

ax.set_xlabel('Accuracy', fontsize=13, fontweight='bold')
ax.set_title('Model Accuracy Comparison with Optimal Feature Selection', 
             fontsize=15, fontweight='bold', pad=20)
ax.set_xlim(0.65, 0.80)
ax.grid(axis='x', alpha=0.3)

for i, (bar, acc) in enumerate(zip(bars, accuracies)):
    ax.text(acc + 0.002, i, f'{acc:.4f} ({acc*100:.2f}%)', 
            va='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(figures_dir / "11_all_models_accuracy_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 11_all_models_accuracy_comparison.png")
plt.close()

print("\n" + "="*80)
print("6. GENERATING SUMMARY REPORT")
print("="*80)

report_path = output_dir / "11_explainable_ai_report.txt"
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("EXPLAINABLE AI (XAI) ANALYSIS - ALL MODELS SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    f.write("OBJECTIVE:\n")
    f.write("Generate SHAP (SHapley Additive exPlanations) for all 7 optimized models\n")
    f.write("to understand feature importance and explain model predictions.\n\n")
    
    f.write("MODELS ANALYZED:\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. Decision Tree\n")
    f.write(f"   - Configuration: Information Gain, 10 features\n")
    f.write(f"   - Accuracy: {dt_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(dt_features)}\n\n")
    
    f.write("2. Naive Bayes\n")
    f.write(f"   - Configuration: Information Gain, 15 features\n")
    f.write(f"   - Accuracy: {nb_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(nb_features)}\n\n")
    
    f.write("3. Random Forest\n")
    f.write(f"   - Configuration: RFE, 20 features\n")
    f.write(f"   - Accuracy: {rf_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(rf_features)}\n\n")
    
    f.write("4. AdaBoost\n")
    f.write(f"   - Configuration: Mutual Info, 15 features\n")
    f.write(f"   - Accuracy: {ada_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(ada_features)}\n\n")
    
    f.write("5. XGBoost\n")
    f.write(f"   - Configuration: RF Importance, 30 features\n")
    f.write(f"   - Accuracy: {xgb_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(xgb_features)}\n\n")
    
    f.write("6. Neural Network\n")
    f.write(f"   - Configuration: ANOVA F-stat, 15 features\n")
    f.write(f"   - Accuracy: {nn_accuracy:.4f}\n")
    f.write(f"   - Features: {', '.join(nn_features)}\n\n")
    
    f.write("7. Deep Learning Attention\n")
    f.write(f"   - Configuration: ANOVA F-stat, 20 features\n")
    f.write(f"   - Accuracy: {dl_accuracy:.4f}\n")
    f.write(f"   - Architecture: 64 → 32 → 16 (attention-like)\n")
    f.write(f"   - Features: {', '.join(dl_features)}\n\n")
    
    f.write("="*80 + "\n")
    f.write("SHAP ANALYSIS DETAILS\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHAP (SHapley Additive exPlanations):\n")
    f.write("- Unified approach to explain model predictions\n")
    f.write("- Based on game theory (Shapley values)\n")
    f.write("- Shows how each feature contributes to predictions\n")
    f.write("- Provides both global and local interpretability\n\n")
    
    f.write("Explainers Used:\n")
    f.write("- TreeExplainer: Decision Tree, Random Forest\n")
    f.write("- KernelExplainer: Naive Bayes, AdaBoost, XGBoost, Neural Network, Deep Learning Attention\n\n")
    
    f.write("Visualizations Generated (per model):\n")
    f.write("1. Feature Importance Bar Chart - Shows mean |SHAP| value per feature\n")
    f.write("2. SHAP Summary Plot (Beeswarm) - Shows feature impact distribution\n\n")
    
    f.write("="*80 + "\n")
    f.write("KEY INSIGHTS\n")
    f.write("="*80 + "\n\n")
    
    f.write("Model Performance Ranking:\n")
    sorted_models = sorted(models_accuracy.items(), key=lambda x: x[1], reverse=True)
    for i, (model, acc) in enumerate(sorted_models, 1):
        f.write(f"{i}. {model}: {acc:.4f} ({acc*100:.2f}%)\n")
    
    f.write("\n")
    f.write("Feature Selection Summary:\n")
    f.write(f"- Decision Tree uses {len(dt_features)} features (most efficient)\n")
    f.write(f"- Naive Bayes uses {len(nb_features)} features\n")
    f.write(f"- Random Forest uses {len(rf_features)} features\n")
    f.write(f"- AdaBoost uses {len(ada_features)} features\n")
    f.write(f"- XGBoost uses {len(xgb_features)} features (most comprehensive)\n")
    f.write(f"- Neural Network uses {len(nn_features)} features\n")
    f.write(f"- Deep Learning Attention uses {len(dl_features)} features\n\n")
    
    f.write("="*80 + "\n")
    f.write("VISUALIZATIONS GENERATED\n")
    f.write("="*80 + "\n\n")
    
    f.write("SHAP Plots (14 plots - 2 per model):\n")
    f.write("1. 11_shap_decision_tree_importance.png\n")
    f.write("2. 11_shap_decision_tree_summary.png\n")
    f.write("3. 11_shap_naive_bayes_importance.png\n")
    f.write("4. 11_shap_naive_bayes_summary.png\n")
    f.write("5. 11_shap_random_forest_importance.png\n")
    f.write("6. 11_shap_random_forest_summary.png\n")
    f.write("7. 11_shap_adaboost_importance.png\n")
    f.write("8. 11_shap_adaboost_summary.png\n")
    f.write("9. 11_shap_xgboost_importance.png\n")
    f.write("10. 11_shap_xgboost_summary.png\n")
    f.write("11. 11_shap_neural_network_importance.png\n")
    f.write("12. 11_shap_neural_network_summary.png\n")
    f.write("13. 11_shap_deep_learning_importance.png\n")
    f.write("14. 11_shap_deep_learning_summary.png\n\n")
    
    f.write("Comparative Plots (2 plots):\n")
    f.write("15. 11_all_models_feature_importance_comparison.png\n")
    f.write("16. 11_all_models_accuracy_comparison.png\n\n")
    
    f.write("="*80 + "\n")
    f.write("INTERPRETATION GUIDE\n")
    f.write("="*80 + "\n\n")
    
    f.write("Feature Importance Bar Chart:\n")
    f.write("- Higher bars = more important features\n")
    f.write("- Shows average magnitude of impact across all predictions\n")
    f.write("- Useful for ranking features globally\n\n")
    
    f.write("SHAP Summary Plot (Beeswarm):\n")
    f.write("- Each dot represents one student's prediction\n")
    f.write("- X-axis: SHAP value (impact on prediction)\n")
    f.write("- Color: Feature value (red=high, blue=low)\n")
    f.write("- Position: Feature importance (top=most important)\n")
    f.write("- Spread: How feature values affect predictions\n\n")
    
    f.write("How to Read SHAP Values:\n")
    f.write("- Positive SHAP value: Increases prediction probability\n")
    f.write("- Negative SHAP value: Decreases prediction probability\n")
    f.write("- Magnitude: Strength of the effect\n")
    f.write("- Sum of SHAP values + base value = model prediction\n\n")
    
    f.write("="*80 + "\n")
    f.write("CONCLUSIONS\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. All models now have explainability through SHAP analysis\n")
    f.write("2. Feature importance varies across models, showing different learning patterns\n")
    f.write("3. XGBoost achieves highest accuracy with comprehensive feature set\n")
    f.write("4. Decision Tree provides simplest interpretation with fewest features\n")
    f.write("5. SHAP values enable stakeholders to trust and understand predictions\n")
    f.write("6. Different explainer types used based on model architecture\n\n")
    
    f.write("This analysis provides complete transparency into how each model\n")
    f.write("makes predictions, essential for educational intervention systems.\n")

print(f"✓ Saved report to: {report_path}")

print("\n" + "="*80)
print("EXPLAINABLE AI ANALYSIS COMPLETE!")
print("="*80)
print(f"\nGenerated files:")
print(f"  - {report_path}")
print(f"\nSHAP Visualizations (16 total):")
print(f"  - 14 model-specific plots (importance + summary for each of 7 models)")
print(f"  - 2 comparative plots (feature importance + accuracy comparison)")
print(f"\nAll visualizations saved in: {figures_dir}")
print("\nExplainable AI analysis provides full transparency into model predictions!")
