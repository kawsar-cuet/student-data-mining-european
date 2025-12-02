"""
Demo Script - Journal Methodology (Without TensorFlow Training)
Shows the complete pipeline without requiring TensorFlow DLL
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing_real import RealDataPreprocessor


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*90)
    print(f"  {text}")
    print("="*90 + "\n")


def main():
    """Demo execution showing methodology without TensorFlow"""
    
    print_header("JOURNAL METHODOLOGY DEMO - DATA PREPARATION & BASELINE MODELS")
    print("Dataset: 4,424 Students | Features: 35 original + 12 engineered")
    print("Demonstrating: Preprocessing + Baseline Models (No TensorFlow required)")
    print("="*90)
    
    # Configuration
    PLOT_DIR = 'outputs/plots_demo'
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    # ========== PHASE 1: DATA PREPROCESSING ==========
    print_header("PHASE 1: DATA PREPROCESSING")
    
    preprocessor = RealDataPreprocessor('data/educational_data.csv', random_state=42)
    
    # Prepare data
    X_train, X_val, X_test, \
    y_target_train, y_target_val, y_target_test, \
    y_dropout_train, y_dropout_val, y_dropout_test, \
    feature_names = preprocessor.prepare_data()
    
    target_labels = preprocessor.get_target_labels()
    
    print(f"\n✓ Phase 1 Complete: Data ready for modeling")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    # ========== PHASE 2: BASELINE MODELS ==========
    print_header("PHASE 2: TRAINING BASELINE MODELS")
    
    # === Model 1: Random Forest for 3-class prediction ===
    print("\n" + "-"*90)
    print(" MODEL 1: RANDOM FOREST (3-Class Prediction)")
    print("-"*90)
    
    print("\n🌲 Training Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_target_train)
    print("  ✓ Training complete")
    
    # === Model 2: Logistic Regression for dropout prediction ===
    print("\n" + "-"*90)
    print(" MODEL 2: LOGISTIC REGRESSION (Binary Dropout)")
    print("-"*90)
    
    print("\n📊 Training Logistic Regression...")
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    lr_model.fit(X_train, y_dropout_train)
    print("  ✓ Training complete")
    
    # ========== PHASE 3: EVALUATION ==========
    print_header("PHASE 3: MODEL EVALUATION")
    
    # === Evaluate Random Forest ===
    print("\n" + "-"*90)
    print(" EVALUATING RANDOM FOREST (3-Class)")
    print("-"*90)
    
    y_target_pred = rf_model.predict(X_test)
    y_target_proba = rf_model.predict_proba(X_test)
    
    from sklearn.metrics import accuracy_score, f1_score
    
    rf_accuracy = accuracy_score(y_target_test, y_target_pred)
    rf_f1_macro = f1_score(y_target_test, y_target_pred, average='macro')
    rf_f1_weighted = f1_score(y_target_test, y_target_pred, average='weighted')
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:        {rf_accuracy:.4f}")
    print(f"  F1-Macro:        {rf_f1_macro:.4f}")
    print(f"  F1-Weighted:     {rf_f1_weighted:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_target_test, y_target_pred,
                               target_names=target_labels, digits=4, zero_division=0))
    
    # === Evaluate Logistic Regression ===
    print("\n" + "-"*90)
    print(" EVALUATING LOGISTIC REGRESSION (Binary Dropout)")
    print("-"*90)
    
    y_dropout_pred = lr_model.predict(X_test)
    y_dropout_proba = lr_model.predict_proba(X_test)[:, 1]
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    lr_accuracy = accuracy_score(y_dropout_test, y_dropout_pred)
    lr_f1 = f1_score(y_dropout_test, y_dropout_pred)
    lr_auc_roc = roc_auc_score(y_dropout_test, y_dropout_proba)
    lr_auc_pr = average_precision_score(y_dropout_test, y_dropout_proba)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:        {lr_accuracy:.4f}")
    print(f"  F1-Score:        {lr_f1:.4f}")
    print(f"  AUC-ROC:         {lr_auc_roc:.4f}")
    print(f"  AUC-PR:          {lr_auc_pr:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_dropout_test, y_dropout_pred,
                               target_names=['Not Dropout', 'Dropout'], digits=4, zero_division=0))
    
    # ========== PHASE 4: VISUALIZATIONS ==========
    print_header("PHASE 4: GENERATING VISUALIZATIONS")
    
    # Confusion Matrices
    print("\n📊 Generating confusion matrices...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Random Forest CM
    cm_rf = confusion_matrix(y_target_test, y_target_pred)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_labels, yticklabels=target_labels, ax=axes[0])
    axes[0].set_title(f'Random Forest - 3-Class\nAccuracy: {rf_accuracy:.2%}',
                     fontsize=12, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)
    
    # Logistic Regression CM
    cm_lr = confusion_matrix(y_dropout_test, y_dropout_pred)
    sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Not Dropout', 'Dropout'],
                yticklabels=['Not Dropout', 'Dropout'], ax=axes[1])
    axes[1].set_title(f'Logistic Regression - Binary\nAccuracy: {lr_accuracy:.2%}',
                     fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/confusion_matrices_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: confusion_matrices_baseline.png")
    
    # ROC Curve
    print("\n📊 Generating ROC curve...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr, tpr, _ = roc_curve(y_dropout_test, y_dropout_proba)
    
    ax.plot(fpr, tpr, linewidth=2, label=f'Logistic Regression (AUC={lr_auc_roc:.4f})',
            color='darkgreen')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve - Dropout Prediction (Baseline)', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/roc_curve_baseline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: roc_curve_baseline.png")
    
    # Feature Importance
    print("\n📊 Generating feature importance plot...")
    
    # Get top 20 most important features
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(20)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(feature_importance)), feature_importance['importance'],
            color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Top 20 Most Important Features (Random Forest)', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: feature_importance.png")
    
    print("\n✓ Phase 4 complete: All visualizations generated")
    
    # ========== SUMMARY ==========
    print_header("EXECUTION SUMMARY")
    
    print("✓ Data Processing: Complete")
    print(f"  - Total samples: 4,424")
    print(f"  - Features: {len(feature_names)} (35 original + 12 engineered)")
    print(f"  - Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    print("\n✓ Baseline Models: Complete")
    print(f"  - Random Forest:        {rf_accuracy:.4f} accuracy, {rf_f1_macro:.4f} F1-macro")
    print(f"  - Logistic Regression:  {lr_accuracy:.4f} accuracy, {lr_auc_roc:.4f} AUC-ROC")
    
    print("\n✓ Visualizations: Complete")
    print(f"  - Saved to: {PLOT_DIR}/")
    print(f"  - confusion_matrices_baseline.png")
    print(f"  - roc_curve_baseline.png")
    print(f"  - feature_importance.png")
    
    print("\n" + "="*90)
    print("BASELINE MODELS DEMO COMPLETED SUCCESSFULLY!")
    print("="*90 + "\n")
    
    print("📝 Note: Deep learning models (PPN, DPN-A, HMTL) require TensorFlow")
    print("   For now, baseline models demonstrate the methodology effectively.")
    print("\n📚 Next Steps:")
    print("  1. Review generated visualizations in outputs/plots_demo/")
    print("  2. Check methodology document: docs/JOURNAL_METHODOLOGY.md")
    print("  3. Explore notebook: notebooks/01_interactive_demo.ipynb")
    print("  4. Fix TensorFlow installation for deep learning models")
    
    print("\n🎯 Top 5 Most Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")
    
    return {
        'preprocessor': preprocessor,
        'models': {
            'random_forest': rf_model,
            'logistic_regression': lr_model
        },
        'metrics': {
            'rf': {'accuracy': rf_accuracy, 'f1_macro': rf_f1_macro},
            'lr': {'accuracy': lr_accuracy, 'auc_roc': lr_auc_roc}
        }
    }


if __name__ == "__main__":
    try:
        print("\n" + "="*90)
        print(" INITIALIZING BASELINE MODELS DEMO")
        print("="*90)
        print("\n⚠ Note: Using baseline models (Random Forest, Logistic Regression)")
        print("  Deep learning models require TensorFlow to be properly installed\n")
        
        results = main()
        
        print("\n✓ Demo executed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
