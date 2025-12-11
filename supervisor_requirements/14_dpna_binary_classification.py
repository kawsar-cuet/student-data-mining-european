# -*- coding: utf-8 -*-
"""
Deep Learning Attention Model - BINARY CLASSIFICATION (Matching Journal Methodology)
Reproduces the 87.05% accuracy from JOURNAL_METHODOLOGY.tex

Key differences from 3-class approach:
- Binary classification: Dropout (1) vs Not Dropout [Enrolled+Graduate] (0)
- Uses ALL 37 features (not feature selection)
- Class weights for imbalance: {0: 1.24, 1: 1.56}
- Binary cross-entropy loss
- Sigmoid output (not softmax)
- 150 max epochs with patience=20
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Create directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"
models_dir = output_dir / "models"
for dir_path in [output_dir, figures_dir, tables_dir, models_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("DEEP LEARNING ATTENTION MODEL - BINARY CLASSIFICATION (DPN-A)")
print("Reproducing 87.05% accuracy from Journal Methodology")
print("="*80)


# ========== ATTENTION LAYER ==========
class AttentionLayer(layers.Layer):
    """
    Self-Attention Layer for Feature Importance Weighting
    From JOURNAL_METHODOLOGY.tex equations 6-8
    """
    
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
    
    def call(self, inputs):
        # Equation 6: e = tanh(xW + b)
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        # Equation 7: α = softmax(e)
        alpha = tf.nn.softmax(e, axis=-1)
        # Equation 8: output = x ⊙ α
        output = inputs * alpha
        return output
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def build_dpna_model(input_dim):
    """
    Build DPN-A Model (Dropout Prediction Network with Attention)
    Matches Table in JOURNAL_METHODOLOGY.tex
    
    Architecture:
    - Input: 37 features
    - Hidden 1: 64 neurons (ReLU + BatchNorm + Dropout 0.3)
    - Attention: Self-attention layer
    - Hidden 2: 32 neurons (ReLU + Dropout 0.2)
    - Hidden 3: 16 neurons (ReLU)
    - Output: 1 neuron (Sigmoid) for binary classification
    """
    inputs = layers.Input(shape=(input_dim,), name='input')
    
    # Hidden Layer 1: 64 units with BN and Dropout
    x = layers.Dense(64, activation='relu', 
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name='hidden1')(inputs)
    x = layers.BatchNormalization(name='bn1')(x)
    x = layers.Dropout(0.3, name='dropout1')(x)
    
    # Attention Layer
    x = AttentionLayer(name='attention')(x)
    
    # Hidden Layer 2: 32 units with Dropout
    x = layers.Dense(32, activation='relu',
                    kernel_regularizer=keras.regularizers.l2(0.001),
                    name='hidden2')(x)
    x = layers.Dropout(0.2, name='dropout2')(x)
    
    # Hidden Layer 3: 16 units
    x = layers.Dense(16, activation='relu', name='hidden3')(x)
    
    # Output Layer: 1 unit with Sigmoid (binary classification)
    outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs, name='DPN-A')
    
    return model


# ========== LOAD DATA ==========
print("\n1. Loading dataset")
print("-" * 60)

df = pd.read_csv("../data/educational_data.csv")
print(f"Dataset shape: {df.shape}")

# Separate features and target
X = df.drop('Target', axis=1)
y_original = df['Target']

# Encode target for 3-class
if y_original.dtype == 'object':
    le = LabelEncoder()
    y_3class = le.fit_transform(y_original)
    target_names = le.classes_
else:
    target_names = ['Dropout', 'Enrolled', 'Graduate']
    y_3class = y_original

# **CRITICAL: Convert to BINARY classification**
# Dropout = 1, Not Dropout (Enrolled or Graduate) = 0
y_binary = (y_3class == 0).astype(int)  # Dropout class is 0 in 3-class encoding

print(f"Features: {X.shape[1]}")
print(f"\nOriginal 3-class distribution:")
for i, name in enumerate(target_names):
    count = np.sum(y_3class == i)
    print(f"  {name}: {count} ({count/len(y_3class)*100:.1f}%)")

print(f"\nBinary classification distribution:")
print(f"  Not Dropout (Enrolled + Graduate): {np.sum(y_binary==0)} ({np.sum(y_binary==0)/len(y_binary)*100:.1f}%)")
print(f"  Dropout: {np.sum(y_binary==1)} ({np.sum(y_binary==1)/len(y_binary)*100:.1f}%)")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"\nTrain: {X_train.shape[0]}, Test: {X_test.shape[0]}")


# ========== USE ALL FEATURES (No Feature Selection) ==========
print("\n2. Feature Engineering")
print("-" * 60)
print(f"Using ALL {X.shape[1]} features (matching journal methodology)")
print(f"Features: {', '.join(X.columns[:5].tolist())}... (showing first 5)")


# ========== STANDARDIZATION ==========
print("\n3. Z-score Standardization")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Standardized to zero mean, unit variance")


# ========== COMPUTE CLASS WEIGHTS ==========
print("\n4. Computing Class Weights")
print("-" * 60)

# Compute class weights to handle imbalance
class_weights_array = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = {0: class_weights_array[0], 1: class_weights_array[1]}
print(f"Class weights: {class_weights}")
print(f"  Not Dropout (0): {class_weights[0]:.2f}")
print(f"  Dropout (1): {class_weights[1]:.2f}")


# ========== BUILD MODEL ==========
print("\n5. Building DPN-A Model")
print("-" * 60)

model = build_dpna_model(X_train_scaled.shape[1])
model.summary()

# Compile with binary cross-entropy and class weights
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',  # Binary classification
    metrics=['accuracy', keras.metrics.AUC(name='auc_roc'), keras.metrics.Precision(), keras.metrics.Recall()]
)


# ========== TRAINING ==========
print("\n6. Training DPN-A Model")
print("-" * 60)

# Callbacks matching journal methodology
model_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Journal uses 20
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        str(models_dir / "14_dpna_binary_best.h5"),
        monitor='val_auc_roc',
        save_best_only=True,
        verbose=0
    )
]

print("Training with class weights and early stopping (patience=20)...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,  # 10% validation
    epochs=150,  # Max 150 epochs
    batch_size=32,
    class_weight=class_weights,  # Apply class weights
    callbacks=model_callbacks,
    verbose=1
)

# Load best weights
model.load_weights(str(models_dir / "14_dpna_binary_best.h5"))
print("\n[OK] Best model loaded")


# ========== EVALUATION ==========
print("\n" + "="*80)
print("7. MODEL EVALUATION")
print("="*80)

# Predictions
y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)

print(f"\nBinary Classification Results (DPN-A):")
print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
print(f"  AUC-ROC:   {auc_roc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print(f"\nJournal Methodology Target: 87.05% accuracy, 0.910 AUC-ROC")
print(f"Current Achievement: {acc*100:.2f}% accuracy, {auc_roc:.3f} AUC-ROC")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Dropout', 'Dropout']))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"                Predicted")
print(f"              Not Dropout  Dropout")
print(f"Actual Not Dropout  {cm[0,0]:4d}      {cm[0,1]:4d}")
print(f"       Dropout      {cm[1,0]:4d}      {cm[1,1]:4d}")


# ========== SAVE RESULTS ==========
results_df = pd.DataFrame([{
    'Model': 'DPN-A (Binary)',
    'Classification': 'Binary (Dropout vs Not Dropout)',
    'Features': X.shape[1],
    'Accuracy': acc,
    'AUC-ROC': auc_roc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1,
    'Class_Weights': str(class_weights),
    'Epochs_Trained': len(history.history['loss'])
}])
results_df.to_csv(tables_dir / "14_dpna_binary_results.csv", index=False)


# ========== VISUALIZATIONS ==========
print("\n8. Generating Visualizations")
print("-" * 60)

plt.style.use('default')
sns.set_palette("husl")

# 1. Training History
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Accuracy
axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0, 0].set_xlabel('Epoch', fontsize=11)
axes[0, 0].set_ylabel('Accuracy', fontsize=11)
axes[0, 0].set_title('DPN-A - Accuracy Over Epochs', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# Loss
axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[0, 1].set_xlabel('Epoch', fontsize=11)
axes[0, 1].set_ylabel('Binary Cross-Entropy Loss', fontsize=11)
axes[0, 1].set_title('DPN-A - Loss Over Epochs', fontsize=12, fontweight='bold')
axes[0, 1].legend(fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# AUC-ROC
axes[1, 0].plot(history.history['auc_roc'], label='Train', linewidth=2)
axes[1, 0].plot(history.history['val_auc_roc'], label='Validation', linewidth=2)
axes[1, 0].set_xlabel('Epoch', fontsize=11)
axes[1, 0].set_ylabel('AUC-ROC', fontsize=11)
axes[1, 0].set_title('DPN-A - AUC-ROC Over Epochs', fontsize=12, fontweight='bold')
axes[1, 0].legend(fontsize=10)
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axhline(y=0.910, color='r', linestyle='--', label='Journal Target (0.910)', linewidth=1.5)

# Precision & Recall
axes[1, 1].plot(history.history['precision'], label='Train Precision', linewidth=2)
axes[1, 1].plot(history.history['val_precision'], label='Val Precision', linewidth=2, linestyle='--')
axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2, linestyle='--')
axes[1, 1].set_xlabel('Epoch', fontsize=11)
axes[1, 1].set_ylabel('Score', fontsize=11)
axes[1, 1].set_title('DPN-A - Precision & Recall', fontsize=12, fontweight='bold')
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "14_dpna_binary_training.png", dpi=300, bbox_inches='tight')
print("[OK] Saved: 14_dpna_binary_training.png")
plt.close()

# 2. ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, linewidth=3, label=f'DPN-A (AUC = {auc_roc:.3f})', color='steelblue')
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier (AUC = 0.500)')
plt.axhline(y=0.910, color='r', linestyle=':', linewidth=2, label='Journal Target AUC = 0.910')
plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
plt.title('DPN-A ROC Curve - Binary Dropout Prediction', fontsize=14, fontweight='bold', pad=15)
plt.legend(fontsize=11, loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(figures_dir / "14_dpna_binary_roc.png", dpi=300, bbox_inches='tight')
print("[OK] Saved: 14_dpna_binary_roc.png")
plt.close()

# 3. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Not Dropout', 'Dropout'],
           yticklabels=['Not Dropout', 'Dropout'],
           linewidths=2, cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
plt.ylabel('True Label', fontsize=13, fontweight='bold')
plt.title(f'DPN-A Confusion Matrix\nAccuracy: {acc*100:.2f}%, AUC-ROC: {auc_roc:.3f}', 
         fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(figures_dir / "14_dpna_binary_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("[OK] Saved: 14_dpna_binary_confusion_matrix.png")
plt.close()


# ========== SUMMARY ==========
print("\n" + "="*80)
print("DPN-A BINARY CLASSIFICATION - COMPLETE")
print("="*80)

summary = f"""
DPN-A Model Summary (Binary Dropout Prediction)
{"="*60}

Classification Type: BINARY
  - Class 0 (Not Dropout): Enrolled + Graduate
  - Class 1 (Dropout): At-risk students

Architecture (Matching Journal):
  - Input: {X.shape[1]} features (ALL features, no selection)
  - Hidden 1: 64 neurons (ReLU + BatchNorm + Dropout 0.3)
  - Attention Layer: Self-attention mechanism
  - Hidden 2: 32 neurons (ReLU + Dropout 0.2)
  - Hidden 3: 16 neurons (ReLU)
  - Output: 1 neuron (Sigmoid activation)

Training Configuration:
  - Loss: Binary Cross-Entropy with class weights
  - Class Weights: {{0: {class_weights[0]:.2f}, 1: {class_weights[1]:.2f}}}
  - Optimizer: Adam (lr=0.001)
  - Epochs: {len(history.history['loss'])} (max 150, early stopping patience=20)
  - Batch Size: 32
  - Validation Split: 10%

Performance Metrics:
  - Accuracy:  {acc:.4f} ({acc*100:.2f}%)
  - AUC-ROC:   {auc_roc:.4f}
  - Precision: {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}

Comparison with Journal Target:
  - Target Accuracy: 87.05%
  - Achieved: {acc*100:.2f}%
  - Difference: {(acc*100 - 87.05):.2f}%
  
  - Target AUC-ROC: 0.910
  - Achieved: {auc_roc:.3f}
  - Difference: {(auc_roc - 0.910):.3f}

Files Generated:
  ✓ 14_dpna_binary_best.h5 (model weights)
  ✓ 14_dpna_binary_results.csv
  ✓ 14_dpna_binary_training.png
  ✓ 14_dpna_binary_roc.png
  ✓ 14_dpna_binary_confusion_matrix.png

Key Differences from 3-Class Approach:
  1. Binary (Dropout vs Not) instead of 3-class
  2. All 37 features instead of 20 selected features
  3. Class weights to handle imbalance
  4. Binary cross-entropy instead of categorical
  5. Sigmoid output instead of softmax
  6. 150 max epochs with patience=20 (vs 200/15)

{"="*60}
"""

print(summary)

with open(tables_dir / "14_dpna_binary_summary.txt", 'w', encoding='utf-8') as f:
    f.write(summary)

print("\n[OK] Analysis complete!")
print("="*80)
