"""
Feature List Generation
Requirements 4-6: List all features by category
"""

import pandas as pd
from pathlib import Path

# Create output directory
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("../data/educational_data.csv")

# Feature categorization
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

print("\n" + "="*80)
print("REQUIREMENTS 4-6: FEATURE LISTS")
print("="*80)

# Create comprehensive feature list document
with open(OUTPUT_DIR / "02_feature_lists.txt", 'w') as f:
    f.write("COMPREHENSIVE FEATURE LISTS\n")
    f.write("="*80 + "\n\n")
    
    # Requirement 4: Academic Features
    f.write("4. LIST OF ACADEMIC FEATURES (18 features)\n")
    f.write("-"*80 + "\n\n")
    for i, feature in enumerate(academic_features, 1):
        f.write(f"   {i:2d}. {feature}\n")
    
    f.write("\n\n")
    
    # Requirement 5: Financial Features
    f.write("5. LIST OF FINANCIAL FEATURES (12 features)\n")
    f.write("-"*80 + "\n\n")
    for i, feature in enumerate(financial_features, 1):
        f.write(f"   {i:2d}. {feature}\n")
    
    f.write("\n\n")
    
    # Requirement 6: Demographic Features
    f.write("6. LIST OF DEMOGRAPHIC FEATURES (16 features)\n")
    f.write("-"*80 + "\n\n")
    for i, feature in enumerate(demographic_features, 1):
        f.write(f"   {i:2d}. {feature}\n")

# Also print to console
print("\n4. ACADEMIC FEATURES (18 features):")
print("-"*80)
for i, feature in enumerate(academic_features, 1):
    print(f"   {i:2d}. {feature}")

print("\n\n5. FINANCIAL FEATURES (12 features):")
print("-"*80)
for i, feature in enumerate(financial_features, 1):
    print(f"   {i:2d}. {feature}")

print("\n\n6. DEMOGRAPHIC FEATURES (16 features):")
print("-"*80)
for i, feature in enumerate(demographic_features, 1):
    print(f"   {i:2d}. {feature}")

print(f"\n\n✓ Feature lists saved to: {OUTPUT_DIR / '02_feature_lists.txt'}")
print("="*80)
