"""
Feature Ranking Analysis
Requirement 7: Rank features using multiple methods
- Information Gain
- Gain Ratio
- Gini Index
- Chi-Square
- Mutual Information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.feature_selection import (
    mutual_info_classif, 
    chi2, 
    SelectKBest,
    f_classif
)
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv("../data/educational_data.csv")

# Separate features and target
target_col = 'Target'
X = df.drop(columns=[target_col])
y = df[target_col]

feature_names = X.columns.tolist()

print("\n" + "="*80)
print("REQUIREMENT 7: FEATURE RANKING")
print("="*80)

# ============================================================================
# 1. INFORMATION GAIN (using Mutual Information)
# ============================================================================
print("\nCalculating Information Gain...")
mi_scores = mutual_info_classif(X, y, random_state=42)
info_gain_df = pd.DataFrame({
    'Feature': feature_names,
    'Information_Gain': mi_scores
}).sort_values('Information_Gain', ascending=False).reset_index(drop=True)
info_gain_df['IG_Rank'] = range(1, len(info_gain_df) + 1)

# ============================================================================
# 2. GAIN RATIO (Information Gain / Intrinsic Value)
# ============================================================================
print("Calculating Gain Ratio...")

def calculate_intrinsic_value(feature_series):
    """Calculate intrinsic value (split info) for a feature"""
    value_counts = feature_series.value_counts(normalize=True)
    return -np.sum(value_counts * np.log2(value_counts + 1e-10))

intrinsic_values = X.apply(calculate_intrinsic_value)
gain_ratio_scores = mi_scores / (intrinsic_values.values + 1e-10)

gain_ratio_df = pd.DataFrame({
    'Feature': feature_names,
    'Gain_Ratio': gain_ratio_scores
}).sort_values('Gain_Ratio', ascending=False).reset_index(drop=True)
gain_ratio_df['GR_Rank'] = range(1, len(gain_ratio_df) + 1)

# ============================================================================
# 3. GINI INDEX (using Decision Tree feature importances)
# ============================================================================
print("Calculating Gini Index...")
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
dt.fit(X, y)
gini_scores = dt.feature_importances_

gini_df = pd.DataFrame({
    'Feature': feature_names,
    'Gini_Importance': gini_scores
}).sort_values('Gini_Importance', ascending=False).reset_index(drop=True)
gini_df['Gini_Rank'] = range(1, len(gini_df) + 1)

# ============================================================================
# 4. CHI-SQUARE TEST
# ============================================================================
print("Calculating Chi-Square scores...")
# Scale features to be non-negative for chi2
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
chi2_scores, p_values = chi2(X_scaled, y)

chi2_df = pd.DataFrame({
    'Feature': feature_names,
    'Chi2_Score': chi2_scores,
    'P_Value': p_values
}).sort_values('Chi2_Score', ascending=False).reset_index(drop=True)
chi2_df['Chi2_Rank'] = range(1, len(chi2_df) + 1)

# ============================================================================
# 5. ANOVA F-STATISTIC
# ============================================================================
print("Calculating ANOVA F-scores...")
f_scores, f_pvalues = f_classif(X, y)

f_stat_df = pd.DataFrame({
    'Feature': feature_names,
    'F_Score': f_scores,
    'F_P_Value': f_pvalues
}).sort_values('F_Score', ascending=False).reset_index(drop=True)
f_stat_df['F_Rank'] = range(1, len(f_stat_df) + 1)

# ============================================================================
# COMBINE ALL RANKINGS
# ============================================================================
print("\nCombining all rankings...")

combined_df = pd.DataFrame({'Feature': feature_names})
combined_df = combined_df.merge(info_gain_df[['Feature', 'Information_Gain', 'IG_Rank']], on='Feature')
combined_df = combined_df.merge(gain_ratio_df[['Feature', 'Gain_Ratio', 'GR_Rank']], on='Feature')
combined_df = combined_df.merge(gini_df[['Feature', 'Gini_Importance', 'Gini_Rank']], on='Feature')
combined_df = combined_df.merge(chi2_df[['Feature', 'Chi2_Score', 'Chi2_Rank']], on='Feature')
combined_df = combined_df.merge(f_stat_df[['Feature', 'F_Score', 'F_Rank']], on='Feature')

# Calculate average rank
combined_df['Average_Rank'] = combined_df[['IG_Rank', 'GR_Rank', 'Gini_Rank', 'Chi2_Rank', 'F_Rank']].mean(axis=1)
combined_df = combined_df.sort_values('Average_Rank').reset_index(drop=True)
combined_df['Final_Rank'] = range(1, len(combined_df) + 1)

# Save to CSV
combined_df.to_csv(OUTPUT_DIR / "tables" / "03_feature_rankings.csv", index=False)

# Save detailed ranking report
with open(OUTPUT_DIR / "03_feature_ranking_report.txt", 'w') as f:
    f.write("FEATURE RANKING ANALYSIS\n")
    f.write("="*100 + "\n\n")
    f.write("Methods Used:\n")
    f.write("  1. Information Gain (Mutual Information)\n")
    f.write("  2. Gain Ratio\n")
    f.write("  3. Gini Index (Decision Tree Feature Importances)\n")
    f.write("  4. Chi-Square Test\n")
    f.write("  5. ANOVA F-Statistic\n\n")
    f.write("="*100 + "\n\n")
    
    f.write(f"{'Rank':<6} {'Feature':<45} {'IG':<6} {'GR':<6} {'Gini':<6} {'Chi2':<6} {'F':<6} {'Avg':<6}\n")
    f.write("-"*100 + "\n")
    
    for idx, row in combined_df.iterrows():
        f.write(f"{row['Final_Rank']:<6} {row['Feature']:<45} "
                f"{row['IG_Rank']:<6} {row['GR_Rank']:<6} {row['Gini_Rank']:<6} "
                f"{row['Chi2_Rank']:<6} {row['F_Rank']:<6} {row['Average_Rank']:<6.2f}\n")

# Print top 20 features
print("\n\nTOP 20 FEATURES (by average rank):")
print("-"*100)
print(f"{'Rank':<6} {'Feature':<45} {'IG':<6} {'GR':<6} {'Gini':<6} {'Chi2':<6} {'F':<6} {'Avg':<6}")
print("-"*100)

for idx, row in combined_df.head(20).iterrows():
    print(f"{row['Final_Rank']:<6} {row['Feature']:<45} "
          f"{row['IG_Rank']:<6.0f} {row['GR_Rank']:<6.0f} {row['Gini_Rank']:<6.0f} "
          f"{row['Chi2_Rank']:<6.0f} {row['F_Rank']:<6.0f} {row['Average_Rank']:<6.2f}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n\nGenerating visualizations...")

# 1. Top 20 features by Information Gain
plt.figure(figsize=(12, 8))
top_20_ig = info_gain_df.head(20).sort_values('Information_Gain')
plt.barh(range(len(top_20_ig)), top_20_ig['Information_Gain'], color='steelblue', edgecolor='black')
plt.yticks(range(len(top_20_ig)), top_20_ig['Feature'], fontsize=9)
plt.xlabel('Information Gain', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by Information Gain', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "03_top20_information_gain.png", dpi=300, bbox_inches='tight')
plt.close()

# 2. Top 20 features by Gini Importance
plt.figure(figsize=(12, 8))
top_20_gini = gini_df.head(20).sort_values('Gini_Importance')
plt.barh(range(len(top_20_gini)), top_20_gini['Gini_Importance'], color='forestgreen', edgecolor='black')
plt.yticks(range(len(top_20_gini)), top_20_gini['Feature'], fontsize=9)
plt.xlabel('Gini Importance', fontsize=12, fontweight='bold')
plt.title('Top 20 Features by Gini Index', fontsize=14, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "03_top20_gini_importance.png", dpi=300, bbox_inches='tight')
plt.close()

# 3. Heatmap of ranking methods
plt.figure(figsize=(14, 10))
rank_matrix = combined_df.head(20)[['IG_Rank', 'GR_Rank', 'Gini_Rank', 'Chi2_Rank', 'F_Rank']].T
rank_matrix.columns = combined_df.head(20)['Feature']

sns.heatmap(rank_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r', 
            cbar_kws={'label': 'Rank'}, linewidths=0.5)
plt.xlabel('Features', fontsize=12, fontweight='bold')
plt.ylabel('Ranking Method', fontsize=12, fontweight='bold')
plt.title('Feature Rankings Comparison (Top 20 Features)\nLower rank = Better', 
          fontsize=14, fontweight='bold')
plt.xticks(rotation=90, ha='right', fontsize=8)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "03_ranking_heatmap.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Feature rankings saved to: {OUTPUT_DIR / 'tables' / '03_feature_rankings.csv'}")
print(f"✓ Detailed report saved to: {OUTPUT_DIR / '03_feature_ranking_report.txt'}")
print(f"✓ Visualizations saved to: {OUTPUT_DIR / 'figures'}")
print("\n" + "="*80)
