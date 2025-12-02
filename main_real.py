"""
Main Execution Script - Journal Methodology Implementation
Real Educational Dataset: 4,424 Students with 35 Features
Following publication-ready methodology from docs/JOURNAL_METHODOLOGY.md
"""

import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# TensorFlow GPU configuration
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing_real import RealDataPreprocessor
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*90)
    print(f"  {text}")
    print("="*90 + "\n")


def build_ppn_model(input_dim, num_classes=3, random_state=42):
    """
    Build Performance Prediction Network (PPN)
    Following journal methodology architecture
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Hidden Layer 1: 128 units
        layers.Dense(128, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        # Hidden Layer 2: 64 units
        layers.Dense(64, activation='relu', name='dense_2'),
        layers.BatchNormalization(name='bn_2'),
        layers.Dropout(0.2, name='dropout_2'),
        
        # Hidden Layer 3: 32 units
        layers.Dense(32, activation='relu', name='dense_3'),
        layers.Dropout(0.1, name='dropout_3'),
        
        # Output layer: 3 classes
        layers.Dense(num_classes, activation='softmax', name='output')
    ], name='PPN')
    
    return model


class AttentionLayer(layers.Layer):
    """Self-attention layer for DPN-A model"""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], input_shape[-1]),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = keras.activations.tanh(keras.backend.dot(x, self.W) + self.b)
        alpha = keras.activations.softmax(e)
        output = x * alpha
        return output


def build_dpn_attention_model(input_dim, random_state=42):
    """
    Build Dropout Prediction Network with Attention (DPN-A)
    Following journal methodology architecture
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # Hidden Layer 1: 64 units
        layers.Dense(64, activation='relu', name='dense_1'),
        layers.BatchNormalization(name='bn_1'),
        layers.Dropout(0.3, name='dropout_1'),
        
        # Attention Layer
        AttentionLayer(name='attention'),
        
        # Hidden Layer 2: 32 units
        layers.Dense(32, activation='relu', name='dense_2'),
        layers.Dropout(0.2, name='dropout_2'),
        
        # Hidden Layer 3: 16 units
        layers.Dense(16, activation='relu', name='dense_3'),
        
        # Output layer: Binary classification
        layers.Dense(1, activation='sigmoid', name='dropout_output')
    ], name='DPN_A')
    
    return model


def build_hmtl_model(input_dim, num_classes=3, random_state=42):
    """
    Build Hybrid Multi-Task Learning Network (HMTL)
    Following journal methodology architecture
    """
    np.random.seed(random_state)
    tf.random.set_seed(random_state)
    
    # Input
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Shared trunk
    x = layers.Dense(128, activation='relu', name='shared_1')(inputs)
    x = layers.BatchNormalization(name='shared_bn_1')(x)
    x = layers.Dropout(0.3, name='shared_dropout_1')(x)
    
    x = layers.Dense(64, activation='relu', name='shared_2')(x)
    x = layers.BatchNormalization(name='shared_bn_2')(x)
    x = layers.Dropout(0.2, name='shared_dropout_2')(x)
    
    # Grade prediction branch
    grade_branch = layers.Dense(32, activation='relu', name='grade_dense')(x)
    grade_branch = layers.Dropout(0.1, name='grade_dropout')(grade_branch)
    grade_output = layers.Dense(num_classes, activation='softmax', name='grade_output')(grade_branch)
    
    # Dropout prediction branch
    dropout_branch = layers.Dense(16, activation='relu', name='dropout_dense')(x)
    dropout_output = layers.Dense(1, activation='sigmoid', name='dropout_output')(dropout_branch)
    
    model = keras.Model(inputs=inputs, outputs=[grade_output, dropout_output], name='HMTL')
    
    return model


def train_model(model, X_train, y_train, X_val, y_val, 
                epochs=150, batch_size=32, class_weight=None, task_name="Model"):
    """Train a model with callbacks"""
    
    print(f"\n🚀 Training {task_name}...")
    
    # Callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=0
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=0
    )
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    best_epoch = len(history.history['loss']) - 20
    best_val_loss = min(history.history['val_loss'])
    
    print(f"  ✓ Training complete")
    print(f"    Best epoch: {best_epoch}/{epochs}")
    print(f"    Best val_loss: {best_val_loss:.4f}")
    
    return history


def main():
    """Main execution pipeline following journal methodology"""
    
    print_header("STUDENT PERFORMANCE AND DROPOUT PREDICTION SYSTEM")
    print("Journal Methodology Implementation - Real Educational Dataset")
    print("Dataset: 4,424 Students | Features: 35 | Target: 3-class outcome prediction")
    print("Publication Target: IEEE Transactions on Learning Technologies")
    print("="*90)
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    DATA_PATH = 'data/educational_data.csv'
    MODEL_DIR = 'outputs/models_real'
    PLOT_DIR = 'outputs/plots_real'
    REPORT_DIR = 'outputs/reports_real'
    
    RANDOM_STATE = 42
    
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # ========== PHASE 1: DATA PREPROCESSING ==========
    print_header("PHASE 1: DATA PREPROCESSING")
    
    preprocessor = RealDataPreprocessor(DATA_PATH, random_state=RANDOM_STATE)
    preprocessor.explore_data()
    
    # Prepare data with stratified splits
    X_train, X_val, X_test, \
    y_target_train, y_target_val, y_target_test, \
    y_dropout_train, y_dropout_val, y_dropout_test, \
    feature_names = preprocessor.prepare_data()
    
    target_labels = preprocessor.get_target_labels()
    
    print(f"\n✓ Phase 1 complete: Data ready for modeling")
    
    # ========== PHASE 2: MODEL TRAINING ==========
    print_header("PHASE 2: DEEP LEARNING MODEL TRAINING")
    
    input_dim = X_train.shape[1]
    
    # === Model 1: Performance Prediction Network (PPN) ===
    print("\n" + "-"*90)
    print(" MODEL 1: PERFORMANCE PREDICTION NETWORK (PPN)")
    print("-"*90)
    
    ppn_model = build_ppn_model(input_dim, num_classes=3, random_state=RANDOM_STATE)
    ppn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\n✓ PPN Architecture:")
    ppn_model.summary()
    
    ppn_history = train_model(
        ppn_model, X_train, y_target_train, X_val, y_target_val,
        epochs=150, batch_size=32, task_name="PPN (3-class classification)"
    )
    
    ppn_model.save(f'{MODEL_DIR}/ppn_model.h5')
    print(f"  ✓ Model saved: {MODEL_DIR}/ppn_model.h5")
    
    # === Model 2: Dropout Prediction Network with Attention (DPN-A) ===
    print("\n" + "-"*90)
    print(" MODEL 2: DROPOUT PREDICTION NETWORK WITH ATTENTION (DPN-A)")
    print("-"*90)
    
    dpn_model = build_dpn_attention_model(input_dim, random_state=RANDOM_STATE)
    
    # Calculate class weights for imbalanced data
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_array = compute_class_weight(
        'balanced',
        classes=np.unique(y_dropout_train),
        y=y_dropout_train
    )
    class_weights = dict(enumerate(class_weights_array))
    print(f"\n✓ Class weights computed: {class_weights}")
    
    dpn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc')]
    )
    
    print("\n✓ DPN-A Architecture:")
    dpn_model.summary()
    
    dpn_history = train_model(
        dpn_model, X_train, y_dropout_train, X_val, y_dropout_val,
        epochs=150, batch_size=32, class_weight=class_weights,
        task_name="DPN-A (Binary dropout with attention)"
    )
    
    dpn_model.save(f'{MODEL_DIR}/dpn_attention_model.h5')
    print(f"  ✓ Model saved: {MODEL_DIR}/dpn_attention_model.h5")
    
    # === Model 3: Hybrid Multi-Task Learning Network (HMTL) ===
    print("\n" + "-"*90)
    print(" MODEL 3: HYBRID MULTI-TASK LEARNING NETWORK (HMTL)")
    print("-"*90)
    
    hmtl_model = build_hmtl_model(input_dim, num_classes=3, random_state=RANDOM_STATE)
    
    hmtl_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'grade_output': 'sparse_categorical_crossentropy',
            'dropout_output': 'binary_crossentropy'
        },
        loss_weights={
            'grade_output': 0.5,
            'dropout_output': 0.5
        },
        metrics={
            'grade_output': ['accuracy'],
            'dropout_output': ['accuracy', keras.metrics.AUC(name='auc')]
        }
    )
    
    print("\n✓ HMTL Architecture:")
    hmtl_model.summary()
    
    hmtl_history = hmtl_model.fit(
        X_train,
        {'grade_output': y_target_train, 'dropout_output': y_dropout_train},
        validation_data=(
            X_val,
            {'grade_output': y_target_val, 'dropout_output': y_dropout_val}
        ),
        epochs=150,
        batch_size=32,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7, verbose=0)
        ],
        verbose=0
    )
    
    print(f"\n🚀 Training HMTL (Multi-task learning)...")
    print(f"  ✓ Training complete")
    print(f"    Best epoch: {len(hmtl_history.history['loss']) - 20}/{150}")
    
    hmtl_model.save(f'{MODEL_DIR}/hmtl_model.h5')
    print(f"  ✓ Model saved: {MODEL_DIR}/hmtl_model.h5")
    
    print("\n✓ Phase 2 complete: All models trained successfully")
    
    # ========== PHASE 3: MODEL EVALUATION ==========
    print_header("PHASE 3: COMPREHENSIVE MODEL EVALUATION")
    
    evaluator = ModelEvaluator()
    
    # === Evaluate PPN ===
    print("\n" + "-"*90)
    print(" EVALUATING PPN (3-Class Performance Prediction)")
    print("-"*90)
    
    y_target_pred_proba = ppn_model.predict(X_test, verbose=0)
    y_target_pred = np.argmax(y_target_pred_proba, axis=1)
    
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    ppn_accuracy = accuracy_score(y_target_test, y_target_pred)
    ppn_f1_macro = f1_score(y_target_test, y_target_pred, average='macro')
    ppn_f1_weighted = f1_score(y_target_test, y_target_pred, average='weighted')
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:        {ppn_accuracy:.4f}")
    print(f"  F1-Macro:        {ppn_f1_macro:.4f}")
    print(f"  F1-Weighted:     {ppn_f1_weighted:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_target_test, y_target_pred, 
                               target_names=target_labels, digits=4, zero_division=0))
    
    # === Evaluate DPN-A ===
    print("\n" + "-"*90)
    print(" EVALUATING DPN-A (Binary Dropout with Attention)")
    print("-"*90)
    
    y_dropout_pred_proba = dpn_model.predict(X_test, verbose=0).flatten()
    y_dropout_pred = (y_dropout_pred_proba > 0.5).astype(int)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    dpn_accuracy = accuracy_score(y_dropout_test, y_dropout_pred)
    dpn_f1 = f1_score(y_dropout_test, y_dropout_pred)
    dpn_auc_roc = roc_auc_score(y_dropout_test, y_dropout_pred_proba)
    dpn_auc_pr = average_precision_score(y_dropout_test, y_dropout_pred_proba)
    
    print(f"\n📊 Overall Metrics:")
    print(f"  Accuracy:        {dpn_accuracy:.4f}")
    print(f"  F1-Score:        {dpn_f1:.4f}")
    print(f"  AUC-ROC:         {dpn_auc_roc:.4f}")
    print(f"  AUC-PR:          {dpn_auc_pr:.4f}")
    
    print(f"\n📋 Classification Report:")
    print(classification_report(y_dropout_test, y_dropout_pred,
                               target_names=['Not Dropout', 'Dropout'], digits=4, zero_division=0))
    
    # === Evaluate HMTL ===
    print("\n" + "-"*90)
    print(" EVALUATING HMTL (Multi-Task Learning)")
    print("-"*90)
    
    hmtl_predictions = hmtl_model.predict(X_test, verbose=0)
    y_grade_pred_hmtl_proba = hmtl_predictions[0]
    y_dropout_pred_hmtl_proba = hmtl_predictions[1].flatten()
    
    y_grade_pred_hmtl = np.argmax(y_grade_pred_hmtl_proba, axis=1)
    y_dropout_pred_hmtl = (y_dropout_pred_hmtl_proba > 0.5).astype(int)
    
    hmtl_grade_acc = accuracy_score(y_target_test, y_grade_pred_hmtl)
    hmtl_grade_f1 = f1_score(y_target_test, y_grade_pred_hmtl, average='macro')
    hmtl_dropout_acc = accuracy_score(y_dropout_test, y_dropout_pred_hmtl)
    hmtl_dropout_auc = roc_auc_score(y_dropout_test, y_dropout_pred_hmtl_proba)
    
    print(f"\n📊 Grade Prediction Task:")
    print(f"  Accuracy:        {hmtl_grade_acc:.4f}")
    print(f"  F1-Macro:        {hmtl_grade_f1:.4f}")
    
    print(f"\n📊 Dropout Prediction Task:")
    print(f"  Accuracy:        {hmtl_dropout_acc:.4f}")
    print(f"  AUC-ROC:         {hmtl_dropout_auc:.4f}")
    
    print("\n✓ Phase 3 complete: All models evaluated")
    
    # ========== PHASE 4: VISUALIZATION ==========
    print_header("PHASE 4: GENERATING VISUALIZATIONS")
    
    viz = Visualizer(save_dir=PLOT_DIR)
    
    # Confusion matrices
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n📊 Generating confusion matrices...")
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # PPN
    cm_ppn = confusion_matrix(y_target_test, y_target_pred)
    sns.heatmap(cm_ppn, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_labels, yticklabels=target_labels, ax=axes[0])
    axes[0].set_title(f'PPN Confusion Matrix\nAccuracy: {ppn_accuracy:.2%}', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # DPN-A
    cm_dpn = confusion_matrix(y_dropout_test, y_dropout_pred)
    sns.heatmap(cm_dpn, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Not Dropout', 'Dropout'],
                yticklabels=['Not Dropout', 'Dropout'], ax=axes[1])
    axes[1].set_title(f'DPN-A Confusion Matrix\nAccuracy: {dpn_accuracy:.2%}', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    # HMTL Dropout
    cm_hmtl = confusion_matrix(y_dropout_test, y_dropout_pred_hmtl)
    sns.heatmap(cm_hmtl, annot=True, fmt='d', cmap='Oranges',
                xticklabels=['Not Dropout', 'Dropout'],
                yticklabels=['Not Dropout', 'Dropout'], ax=axes[2])
    axes[2].set_title(f'HMTL Confusion Matrix\nAccuracy: {hmtl_dropout_acc:.2%}', fontweight='bold')
    axes[2].set_ylabel('True Label')
    axes[2].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: confusion_matrices.png")
    
    # ROC Curves
    print("\n📊 Generating ROC curves...")
    from sklearn.metrics import roc_curve
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    fpr_dpn, tpr_dpn, _ = roc_curve(y_dropout_test, y_dropout_pred_proba)
    fpr_hmtl, tpr_hmtl, _ = roc_curve(y_dropout_test, y_dropout_pred_hmtl_proba)
    
    ax.plot(fpr_dpn, tpr_dpn, linewidth=2, label=f'DPN-A (AUC={dpn_auc_roc:.4f})', color='darkgreen')
    ax.plot(fpr_hmtl, tpr_hmtl, linewidth=2, label=f'HMTL (AUC={hmtl_dropout_auc:.4f})', color='darkorange')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Dropout Prediction', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/roc_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: roc_curves.png")
    
    # Model comparison
    print("\n📊 Generating model comparison chart...")
    
    comparison_data = {
        'Model': ['PPN', 'DPN-A', 'HMTL-Grade', 'HMTL-Dropout'],
        'Accuracy': [ppn_accuracy, dpn_accuracy, hmtl_grade_acc, hmtl_dropout_acc],
        'F1-Score': [ppn_f1_macro, dpn_f1, hmtl_grade_f1, f1_score(y_dropout_test, y_dropout_pred_hmtl)]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(comparison_df))
    width = 0.35
    
    ax.bar(x - width/2, comparison_df['Accuracy'], width, label='Accuracy', color='steelblue')
    ax.bar(x + width/2, comparison_df['F1-Score'], width, label='F1-Score', color='coral')
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{PLOT_DIR}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: model_comparison.png")
    
    print("\n✓ Phase 4 complete: All visualizations generated")
    
    # ========== SUMMARY ==========
    print_header("EXECUTION SUMMARY")
    
    print("✓ Data Processing: Complete")
    print(f"  - Total samples: {len(preprocessor.original_data):,}")
    print(f"  - Features: {len(feature_names)} (including {12} engineered)")
    print(f"  - Train/Val/Test: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    
    print("\n✓ Model Training: Complete")
    print(f"  - PPN:   {ppn_accuracy:.4f} accuracy, {ppn_f1_macro:.4f} F1-macro")
    print(f"  - DPN-A: {dpn_accuracy:.4f} accuracy, {dpn_auc_roc:.4f} AUC-ROC")
    print(f"  - HMTL:  {hmtl_grade_acc:.4f} grade acc, {hmtl_dropout_auc:.4f} dropout AUC")
    
    print("\n✓ Visualizations: Complete")
    print(f"  - Saved to: {PLOT_DIR}/")
    
    print("\n✓ Models Saved:")
    print(f"  - {MODEL_DIR}/ppn_model.h5")
    print(f"  - {MODEL_DIR}/dpn_attention_model.h5")
    print(f"  - {MODEL_DIR}/hmtl_model.h5")
    
    print("\n" + "="*90)
    print("JOURNAL METHODOLOGY IMPLEMENTATION COMPLETED SUCCESSFULLY!")
    print("="*90 + "\n")
    
    print("📚 Next Steps for Publication:")
    print("  1. Implement baseline models (Random Forest, XGBoost, SVM, Logistic Regression)")
    print("  2. Perform 10-fold cross-validation for robust evaluation")
    print("  3. Generate SHAP values for feature importance analysis")
    print("  4. Conduct statistical significance testing (McNemar's, Friedman)")
    print("  5. Create learning curves and calibration plots")
    print("  6. Integrate LLM-based recommendation system with GPT-4")
    print("  7. Write results section following journal format")
    
    return {
        'preprocessor': preprocessor,
        'models': {
            'ppn': ppn_model,
            'dpn_a': dpn_model,
            'hmtl': hmtl_model
        },
        'metrics': {
            'ppn': {'accuracy': ppn_accuracy, 'f1_macro': ppn_f1_macro},
            'dpn_a': {'accuracy': dpn_accuracy, 'auc_roc': dpn_auc_roc, 'auc_pr': dpn_auc_pr},
            'hmtl': {'grade_acc': hmtl_grade_acc, 'dropout_auc': hmtl_dropout_auc}
        }
    }


if __name__ == "__main__":
    try:
        print("\n" + "="*90)
        print(" INITIALIZING DEEP LEARNING PIPELINE")
        print("="*90)
        print(f"\nTensorFlow Version: {tf.__version__}")
        print(f"Keras Version: {keras.__version__}")
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU Available: {len(gpus)} device(s)")
        else:
            print("GPU: Not available (using CPU)")
        
        results = main()
        
        print("\n✓ Pipeline executed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
