"""
Multi-Model Classification
Requirement 9: Implement and train multiple models

Models:
- Single Classifiers: Decision Tree, Naive Bayes
- Ensemble Methods: Random Forest, AdaBoost, XGBoost
- Deep Learning: Neural Network (MLP)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import joblib
import warnings
warnings.filterwarnings('ignore')

# Create output directories
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "models").mkdir(exist_ok=True)

# Load dataset
print("\n" + "="*80)
print("REQUIREMENT 9: MULTI-MODEL CLASSIFICATION")
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
class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"\nClass mapping: {class_mapping}")

print(f"\nDataset: {len(df)} students")
print(f"Features: {X.shape[1]}")
print(f"Classes: {len(np.unique(y))} (Dropout=0, Enrolled=1, Graduate=2)")
print(f"  - Dropout (0): {(y==0).sum()}")
print(f"  - Enrolled (1): {(y==1).sum()}")
print(f"  - Graduate (2): {(y==2).sum()}")

# Split data (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/Test Split (80/20):")
print(f"  Training: {len(X_train)} students")
print(f"  Testing: {len(X_test)} students")

# Standardize features for Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, OUTPUT_DIR / "models" / "scaler.pkl")
print(f"\n✓ Scaler saved to: {OUTPUT_DIR / 'models' / 'scaler.pkl'}")

# ============================================================================
# INITIALIZE MODELS
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

models = {
    'Decision Tree': DecisionTreeClassifier(
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    ),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    ),
    'AdaBoost': AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),
        n_estimators=100,
        learning_rate=1.0,
        random_state=42,
        algorithm='SAMME'
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss'
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        learning_rate_init=0.001
    )
}

# ============================================================================
# TRAIN AND SAVE MODELS
# ============================================================================
trained_models = {}
training_results = []

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Use scaled data for Neural Network, original for others
    if name == 'Neural Network':
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    # Train model
    print("  Training...")
    model.fit(X_tr, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_tr)
    y_pred_test = model.predict(X_te)
    
    # Training accuracy
    train_acc = (y_pred_train == y_train).mean()
    test_acc = (y_pred_test == y_test).mean()
    
    print(f"  Training Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    
    # Store results
    training_results.append({
        'Model': name,
        'Training_Accuracy': train_acc,
        'Test_Accuracy': test_acc,
        'Parameters': str(model.get_params())
    })
    
    # Save model
    model_filename = f"{name.lower().replace(' ', '_')}.pkl"
    joblib.dump(model, OUTPUT_DIR / "models" / model_filename)
    print(f"  ✓ Model saved to: {OUTPUT_DIR / 'models' / model_filename}")
    
    trained_models[name] = model

# ============================================================================
# SAVE TRAINING SUMMARY
# ============================================================================
results_df = pd.DataFrame(training_results)
results_df.to_csv(OUTPUT_DIR / "05_model_training_results.csv", index=False)

print("\n\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"\n{'Model':<20} {'Train Acc':<12} {'Test Acc':<12}")
print("-"*44)
for idx, row in results_df.iterrows():
    print(f"{row['Model']:<20} {row['Training_Accuracy']:<12.4f} {row['Test_Accuracy']:<12.4f}")

# Save detailed report
with open(OUTPUT_DIR / "05_model_training_report.txt", 'w') as f:
    f.write("MULTI-MODEL CLASSIFICATION TRAINING RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {len(df)} students\n")
    f.write(f"Features: {X.shape[1]}\n")
    f.write(f"Classes: 3 (Dropout, Enrolled, Graduate)\n")
    f.write(f"Train/Test Split: 80/20\n")
    f.write(f"Training Set: {len(X_train)} students\n")
    f.write(f"Test Set: {len(X_test)} students\n\n")
    f.write("="*80 + "\n\n")
    
    f.write("MODELS TRAINED:\n\n")
    
    for idx, row in results_df.iterrows():
        f.write(f"{idx+1}. {row['Model']}\n")
        f.write(f"   Training Accuracy: {row['Training_Accuracy']:.4f}\n")
        f.write(f"   Test Accuracy: {row['Test_Accuracy']:.4f}\n")
        f.write(f"   Saved as: {row['Model'].lower().replace(' ', '_')}.pkl\n\n")
    
    f.write("\n" + "="*80 + "\n")
    f.write("MODEL CATEGORIES:\n")
    f.write("="*80 + "\n\n")
    f.write("Single Classifiers:\n")
    f.write("  1. Decision Tree\n")
    f.write("  2. Naive Bayes\n\n")
    f.write("Ensemble Methods:\n")
    f.write("  3. Random Forest (200 trees)\n")
    f.write("  4. AdaBoost (100 estimators)\n")
    f.write("  5. XGBoost (200 estimators)\n\n")
    f.write("Deep Learning:\n")
    f.write("  6. Neural Network (3 hidden layers: 128-64-32)\n\n")
    f.write("="*80 + "\n")
    f.write("\nAll models saved to: outputs/models/\n")
    f.write("Ready for evaluation and explainable AI analysis.\n")

print(f"\n✓ Training results saved to: {OUTPUT_DIR / '05_model_training_results.csv'}")
print(f"✓ Detailed report saved to: {OUTPUT_DIR / '05_model_training_report.txt'}")
print(f"✓ All models saved to: {OUTPUT_DIR / 'models'}")
print("\n" + "="*80)
print("All models trained successfully!")
print("="*80 + "\n")
