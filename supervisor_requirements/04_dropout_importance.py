"""
Dropout Feature Importance Analysis
Requirement 8: Find most important/influential features for dropout prediction

Uses multiple methods:
1. Binary classification (Dropout vs Non-Dropout)
2. SHAP values
3. Permutation importance
4. Correlation analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv("../data/educational_data.csv")

# Create binary target: Dropout (1) vs Non-Dropout (0)
target_col = 'Target'
df['Dropout_Binary'] = (df[target_col] == 'Dropout').astype(int)  # 1=Dropout, 0=Others

X = df.drop(columns=[target_col, 'Dropout_Binary'])
y = df['Dropout_Binary']

feature_names = X.columns.tolist()

print("\n" + "="*80)
print("REQUIREMENT 8: MOST IMPORTANT FEATURES FOR DROPOUT PREDICTION")
print("="*80)

print(f"\nDataset: {len(df)} students")
print(f"Dropout students: {y.sum()} ({y.sum()/len(y)*100:.1f}%)")
print(f"Non-Dropout students: {(len(y) - y.sum())} ({(len(y) - y.sum())/len(y)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set: {len(X_train)} students")
print(f"Test set: {len(X_test)} students")

# ============================================================================
# METHOD 1: RANDOM FOREST FEATURE IMPORTANCE
# ============================================================================
print("\n\n1. Random Forest Feature Importance...")
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)
rf_model.fit(X_train, y_train)

rf_importance = pd.DataFrame({
    'Feature': feature_names,
    'RF_Importance': rf_model.feature_importances_
}).sort_values('RF_Importance', ascending=False).reset_index(drop=True)
rf_importance['RF_Rank'] = range(1, len(rf_importance) + 1)

print(f"Random Forest Accuracy: {rf_model.score(X_test, y_test):.4f}")

# ============================================================================
# METHOD 2: GRADIENT BOOSTING FEATURE IMPORTANCE
# ============================================================================
print("\n2. Gradient Boosting Feature Importance...")
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_model.fit(X_train, y_train)

gb_importance = pd.DataFrame({
    'Feature': feature_names,
    'GB_Importance': gb_model.feature_importances_
}).sort_values('GB_Importance', ascending=False).reset_index(drop=True)
gb_importance['GB_Rank'] = range(1, len(gb_importance) + 1)

print(f"Gradient Boosting Accuracy: {gb_model.score(X_test, y_test):.4f}")

# ============================================================================
# METHOD 3: PERMUTATION IMPORTANCE
# ============================================================================
print("\n3. Permutation Importance...")
perm_importance = permutation_importance(
    rf_model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Perm_Importance': perm_importance.importances_mean,
    'Perm_Std': perm_importance.importances_std
}).sort_values('Perm_Importance', ascending=False).reset_index(drop=True)
perm_importance_df['Perm_Rank'] = range(1, len(perm_importance_df) + 1)

# ============================================================================
# METHOD 4: CORRELATION WITH DROPOUT
# ============================================================================
print("\n4. Correlation Analysis...")
correlations = []
for feature in feature_names:
    corr = np.corrcoef(X[feature], y)[0, 1]
    correlations.append(abs(corr))  # Use absolute correlation

corr_df = pd.DataFrame({
    'Feature': feature_names,
    'Abs_Correlation': correlations
}).sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
corr_df['Corr_Rank'] = range(1, len(corr_df) + 1)

# ============================================================================
# COMBINE ALL METHODS
# ============================================================================
print("\n5. Combining all methods...")

combined_dropout = pd.DataFrame({'Feature': feature_names})
combined_dropout = combined_dropout.merge(rf_importance[['Feature', 'RF_Importance', 'RF_Rank']], on='Feature')
combined_dropout = combined_dropout.merge(gb_importance[['Feature', 'GB_Importance', 'GB_Rank']], on='Feature')
combined_dropout = combined_dropout.merge(perm_importance_df[['Feature', 'Perm_Importance', 'Perm_Rank']], on='Feature')
combined_dropout = combined_dropout.merge(corr_df[['Feature', 'Abs_Correlation', 'Corr_Rank']], on='Feature')

# Calculate average rank
combined_dropout['Average_Rank'] = combined_dropout[['RF_Rank', 'GB_Rank', 'Perm_Rank', 'Corr_Rank']].mean(axis=1)
combined_dropout = combined_dropout.sort_values('Average_Rank').reset_index(drop=True)
combined_dropout['Final_Rank'] = range(1, len(combined_dropout) + 1)

# Normalize scores to 0-1 range for visualization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
combined_dropout['RF_Score_Norm'] = scaler.fit_transform(combined_dropout[['RF_Importance']])
combined_dropout['GB_Score_Norm'] = scaler.fit_transform(combined_dropout[['GB_Importance']])
combined_dropout['Perm_Score_Norm'] = scaler.fit_transform(combined_dropout[['Perm_Importance']])
combined_dropout['Corr_Score_Norm'] = scaler.fit_transform(combined_dropout[['Abs_Correlation']])
combined_dropout['Composite_Score'] = combined_dropout[['RF_Score_Norm', 'GB_Score_Norm', 'Perm_Score_Norm', 'Corr_Score_Norm']].mean(axis=1)

combined_dropout = combined_dropout.sort_values('Composite_Score', ascending=False).reset_index(drop=True)
combined_dropout['Final_Rank'] = range(1, len(combined_dropout) + 1)

# Save to CSV
combined_dropout.to_csv(OUTPUT_DIR / "tables" / "04_dropout_feature_importance.csv", index=False)

# Save detailed report
with open(OUTPUT_DIR / "04_dropout_features_report.txt", 'w') as f:
    f.write("MOST IMPORTANT FEATURES FOR DROPOUT PREDICTION\n")
    f.write("="*110 + "\n\n")
    f.write("Methods Used:\n")
    f.write("  1. Random Forest Feature Importance\n")
    f.write("  2. Gradient Boosting Feature Importance\n")
    f.write("  3. Permutation Importance\n")
    f.write("  4. Correlation Analysis\n\n")
    f.write("="*110 + "\n\n")
    
    f.write(f"{'Rank':<6} {'Feature':<45} {'Composite':<11} {'RF':<6} {'GB':<6} {'Perm':<6} {'Corr':<6}\n")
    f.write("-"*110 + "\n")
    
    for idx, row in combined_dropout.iterrows():
        f.write(f"{row['Final_Rank']:<6} {row['Feature']:<45} "
                f"{row['Composite_Score']:<11.4f} "
                f"{row['RF_Rank']:<6.0f} {row['GB_Rank']:<6.0f} "
                f"{row['Perm_Rank']:<6.0f} {row['Corr_Rank']:<6.0f}\n")
    
    f.write("\n\n" + "="*110 + "\n")
    f.write("TOP 10 MOST IMPORTANT FEATURES FOR DROPOUT:\n")
    f.write("="*110 + "\n\n")
    
    for idx, row in combined_dropout.head(10).iterrows():
        f.write(f"{idx+1}. {row['Feature']}\n")
        f.write(f"   - Composite Score: {row['Composite_Score']:.4f}\n")
        f.write(f"   - Random Forest Importance: {row['RF_Importance']:.4f} (Rank: {row['RF_Rank']:.0f})\n")
        f.write(f"   - Gradient Boosting Importance: {row['GB_Importance']:.4f} (Rank: {row['GB_Rank']:.0f})\n")
        f.write(f"   - Permutation Importance: {row['Perm_Importance']:.4f} (Rank: {row['Perm_Rank']:.0f})\n")
        f.write(f"   - Correlation: {row['Abs_Correlation']:.4f} (Rank: {row['Corr_Rank']:.0f})\n\n")

# Print top 15 features
print("\n\nTOP 15 MOST IMPORTANT FEATURES FOR DROPOUT:")
print("-"*110)
print(f"{'Rank':<6} {'Feature':<45} {'Composite':<11} {'RF':<6} {'GB':<6} {'Perm':<6} {'Corr':<6}")
print("-"*110)

for idx, row in combined_dropout.head(15).iterrows():
    print(f"{row['Final_Rank']:<6} {row['Feature']:<45} "
          f"{row['Composite_Score']:<11.4f} "
          f"{row['RF_Rank']:<6.0f} {row['GB_Rank']:<6.0f} "
          f"{row['Perm_Rank']:<6.0f} {row['Corr_Rank']:<6.0f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n\nGenerating visualizations...")

# 1. Top 20 features by composite score
plt.figure(figsize=(12, 8))
top_20 = combined_dropout.head(20).sort_values('Composite_Score')
colors = plt.cm.RdYlGn(top_20['Composite_Score'] / top_20['Composite_Score'].max())

plt.barh(range(len(top_20)), top_20['Composite_Score'], color=colors, edgecolor='black')
plt.yticks(range(len(top_20)), top_20['Feature'], fontsize=9)
plt.xlabel('Composite Importance Score', fontsize=12, fontweight='bold')
plt.title('Top 20 Most Important Features for Dropout Prediction\n(Combined Score from 4 Methods)', 
          fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "04_top20_dropout_features.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Comparison of methods for top 15 features
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
top_15 = combined_dropout.head(15).sort_values('RF_Importance')

# Random Forest
axes[0, 0].barh(range(len(top_15)), top_15['RF_Importance'], color='steelblue', edgecolor='black')
axes[0, 0].set_yticks(range(len(top_15)))
axes[0, 0].set_yticklabels(top_15['Feature'], fontsize=8)
axes[0, 0].set_xlabel('Importance', fontsize=10, fontweight='bold')
axes[0, 0].set_title('Random Forest', fontsize=11, fontweight='bold')
axes[0, 0].grid(axis='x', alpha=0.3)

# Gradient Boosting
top_15_gb = combined_dropout.head(15).sort_values('GB_Importance')
axes[0, 1].barh(range(len(top_15_gb)), top_15_gb['GB_Importance'], color='forestgreen', edgecolor='black')
axes[0, 1].set_yticks(range(len(top_15_gb)))
axes[0, 1].set_yticklabels(top_15_gb['Feature'], fontsize=8)
axes[0, 1].set_xlabel('Importance', fontsize=10, fontweight='bold')
axes[0, 1].set_title('Gradient Boosting', fontsize=11, fontweight='bold')
axes[0, 1].grid(axis='x', alpha=0.3)

# Permutation Importance
top_15_perm = combined_dropout.head(15).sort_values('Perm_Importance')
axes[1, 0].barh(range(len(top_15_perm)), top_15_perm['Perm_Importance'], color='coral', edgecolor='black')
axes[1, 0].set_yticks(range(len(top_15_perm)))
axes[1, 0].set_yticklabels(top_15_perm['Feature'], fontsize=8)
axes[1, 0].set_xlabel('Importance', fontsize=10, fontweight='bold')
axes[1, 0].set_title('Permutation Importance', fontsize=11, fontweight='bold')
axes[1, 0].grid(axis='x', alpha=0.3)

# Correlation
top_15_corr = combined_dropout.head(15).sort_values('Abs_Correlation')
axes[1, 1].barh(range(len(top_15_corr)), top_15_corr['Abs_Correlation'], color='purple', edgecolor='black')
axes[1, 1].set_yticks(range(len(top_15_corr)))
axes[1, 1].set_yticklabels(top_15_corr['Feature'], fontsize=8)
axes[1, 1].set_xlabel('Absolute Correlation', fontsize=10, fontweight='bold')
axes[1, 1].set_title('Correlation with Dropout', fontsize=11, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle('Comparison of Feature Importance Methods\nTop 15 Features for Dropout Prediction', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "04_methods_comparison.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Dropout feature importance saved to: {OUTPUT_DIR / 'tables' / '04_dropout_feature_importance.csv'}")
print(f"✓ Detailed report saved to: {OUTPUT_DIR / '04_dropout_features_report.txt'}")
print(f"✓ Visualizations saved to: {OUTPUT_DIR / 'figures'}")
print("\n" + "="*80)
