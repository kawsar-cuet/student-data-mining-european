"""
Dataset Analysis Script
Author: Student
Date: December 8, 2025

Requirements 1-3: Dataset Overview
- Total students (instances): 4424
- Total features: 46
- Classes: 3 (Enrolled, Graduate, Dropout)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "figures").mkdir(exist_ok=True)
(OUTPUT_DIR / "tables").mkdir(exist_ok=True)

# Load dataset
print("\nLoading dataset...")
df = pd.read_csv("../data/educational_data.csv")

print("\n" + "="*80)
print("REQUIREMENT 1-3: DATASET OVERVIEW")
print("="*80)

# 1. Total students
total_students = len(df)
print(f"\n1. Total Students (Instances): {total_students}")

# 2. Total features
total_features = len(df.columns) - 1  # Exclude target variable
print(f"\n2. Total Features: {total_features}")

# Target variable
target_col = 'Target'

# Feature categorization based on domain knowledge
academic_features = [
    'Curricular_units_1st_sem_credited',
    'Curricular_units_1st_sem_enrolled',
    'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_1st_sem_grade',
    'Curricular_units_1st_sem_without_evaluations',
    'Curricular_units_2nd_sem_credited',
    'Curricular_units_2nd_sem_enrolled',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_2nd_sem_without_evaluations',
    'Previous_qualification_grade',
    'Admission_grade',
    'Application_mode',
    'Application_order',
    'Course',
    'Daytime_evening_attendance'
]

financial_features = [
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Debtor',
    'Unemployment_rate',
    'Inflation_rate',
    'GDP',
    'International',
    'Displaced',
    'Educational_special_needs',
    'Gender',
    'Age_at_enrollment',
    'Nacionality'
]

demographic_features = [
    'Marital_status',
    'Previous_qualification',
    'Mothers_qualification',
    'Fathers_qualification',
    'Mothers_occupation',
    'Fathers_occupation',
    'Gender',
    'Age_at_enrollment',
    'International',
    'Displaced',
    'Educational_special_needs',
    'Debtor',
    'Tuition_fees_up_to_date',
    'Scholarship_holder',
    'Nacionality',
    'Application_mode'
]

print(f"\n2.1 Academic Features: {len(academic_features)}")
print(f"2.2 Financial Features: {len(financial_features)}")
print(f"2.3 Demographic Features: {len(demographic_features)}")

# 3. Classes distribution
print(f"\n3. Classes: {df[target_col].nunique()}")
class_distribution = df[target_col].value_counts().sort_index()

# Map target values to class names
target_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
for value, name in target_mapping.items():
    count = (df[target_col] == value).sum()
    print(f"3.{value+1} {name}: {count}")

# Save summary to file
with open(OUTPUT_DIR / "01_dataset_summary.txt", 'w') as f:
    f.write("DATASET OVERVIEW\n")
    f.write("="*80 + "\n\n")
    f.write(f"1. Total Students (Instances): {total_students}\n\n")
    f.write(f"2. Total Features: {total_features}\n")
    f.write(f"   2.1 Academic Features: {len(academic_features)}\n")
    f.write(f"   2.2 Financial Features: {len(financial_features)}\n")
    f.write(f"   2.3 Demographic Features: {len(demographic_features)}\n\n")
    f.write(f"3. Classes: {df[target_col].nunique()}\n")
    for value, name in target_mapping.items():
        count = (df[target_col] == value).sum()
        f.write(f"   3.{value+1} {name}: {count}\n")

# Visualize class distribution
plt.figure(figsize=(10, 6))
counts = [class_distribution[i] for i in sorted(class_distribution.index)]
labels = ['Dropout', 'Enrolled', 'Graduate']
colors = ['#e74c3c', '#f39c12', '#2ecc71']

plt.bar(labels, counts, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Number of Students', fontsize=12, fontweight='bold')
plt.xlabel('Class', fontsize=12, fontweight='bold')
plt.title('Class Distribution\n(Total Students: 4,424)', fontsize=14, fontweight='bold')

# Add value labels on bars
for i, (label, count) in enumerate(zip(labels, counts)):
    plt.text(i, count + 50, str(count), ha='center', va='bottom', 
             fontsize=11, fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "figures" / "01_class_distribution.png", dpi=300, bbox_inches='tight')
plt.close()

print(f"\n✓ Summary saved to: {OUTPUT_DIR / '01_dataset_summary.txt'}")
print(f"✓ Class distribution plot saved to: {OUTPUT_DIR / 'figures' / '01_class_distribution.png'}")
print("\n" + "="*80)
