# -*- coding: utf-8 -*-
"""
Deep Learning with Attention Mechanism - Quick Training
Trains the attention model with the best configuration (based on similar models)
for faster execution and integration into the comprehensive analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
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
print("DEEP LEARNING WITH ATTENTION MECHANISM - QUICK TRAINING")
print("="*80)


# ========== ATTENTION LAYER ==========
class AttentionLayer(layers.Layer):
    """Custom Attention Layer for feature weighting"""
    
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
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        a = tf.nn.softmax(e, axis=-1)
        return inputs * a
    
    def get_config(self):
        return super(AttentionLayer, self).get_config()


def build_attention_model(input_dim, num_classes=3):
    """Build Attention-based Deep Neural Network"""
    inputs = layers.Input(shape=(input_dim,))
    
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = AttentionLayer()(x)
    
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(16, activation='relu')(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ========== LOAD DATA ==========
print("\n1. Loading and preparing data")
print("-" * 60)

df = pd.read_csv("../data/educational_data.csv")
print(f"Dataset shape: {df.shape}")

X = df.drop('Target', axis=1)
y = df['Target']

if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)
    target_names = le.classes_
else:
    target_names = ['Dropout', 'Enrolled', 'Graduate']

num_classes = len(np.unique(y))
print(f"Features: {X.shape[1]}, Classes: {num_classes}")
print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ========== FEATURE SELECTION ==========
print("\n2. Performing feature selection (ANOVA F-test, 20 features)")
print("-" * 60)

# Use ANOVA F-test with 20 features (good balance based on other models)
selector = SelectKBest(f_classif, k=20)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

selected_features = X_train.columns[selector.get_support()].tolist()
print(f"Selected features: {', '.join(selected_features)}")

# Save selected features
pd.DataFrame({'Feature': selected_features}).to_csv(
    tables_dir / "13_deep_learning_attention_selected_features.csv", index=False
)


# ========== SCALE DATA ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_selected)
X_test_scaled = scaler.transform(X_test_selected)


# ========== TRAIN MODEL ==========
print("\n3. Training Deep Learning Attention model")
print("-" * 60)

model = build_attention_model(X_train_scaled.shape[1], num_classes)

model_callbacks = [
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        str(models_dir / "13_deep_learning_attention.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=0
    )
]

print("Training in progress...")
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=50,  # Reduced for faster training
    batch_size=32,  # Larger batch for faster training
    callbacks=model_callbacks,
    verbose=1
)

# Load best weights
model.load_weights(str(models_dir / "13_deep_learning_attention.h5"))
print("✓ Best model loaded")


# ========== EVALUATION ==========
print("\n4. Evaluating model performance")
print("=" * 60)

y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
y_pred_proba = model.predict(X_test_scaled, verbose=0)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='weighted')
rec = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"\nPerformance Metrics:")
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1-Score:  {f1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)


# ========== SAVE RESULTS ==========
results_df = pd.DataFrame([{
    'Model': 'Deep Learning Attention',
    'Method': 'ANOVA F-test',
    'Num_Features': len(selected_features),
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1': f1
}])
results_df.to_csv(tables_dir / "13_deep_learning_attention_results.csv", index=False)


# ========== VISUALIZATIONS ==========
print("\n5. Generating visualizations")
print("-" * 60)

plt.style.use('default')
sns.set_palette("husl")

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Deep Learning Attention - Accuracy', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'], label='Train', linewidth=2)
axes[1].plot(history.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Deep Learning Attention - Loss', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_training.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 13_deep_learning_attention_training.png")
plt.close()

# 2. Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=target_names, yticklabels=target_names,
           linewidths=0.5, cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Deep Learning Attention - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 13_deep_learning_attention_confusion_matrix.png")
plt.close()

# 3. Feature Importance (from attention weights)
attention_weights = None
for layer in model.layers:
    if isinstance(layer, AttentionLayer):
        attention_weights = layer.get_weights()[0]
        break

if attention_weights is not None:
    feature_importance = np.abs(attention_weights).sum(axis=1)
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(tables_dir / "13_deep_learning_attention_importance.csv", index=False)
    
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Attention Weight Magnitude', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Deep Learning Attention - Top 15 Features', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_deep_learning_attention_importance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 13_deep_learning_attention_importance.png")
    plt.close()


# ========== SUMMARY ==========
print("\n" + "="*80)
print("DEEP LEARNING ATTENTION MODEL - TRAINING COMPLETE")
print("="*80)

summary = f"""
Model Configuration:
  - Architecture: 64 → Attention → 32 → 16 → 3 (softmax)
  - Feature Selection: ANOVA F-test
  - Number of Features: {len(selected_features)}
  - Total Parameters: {model.count_params():,}

Performance:
  - Accuracy:  {acc:.4f}
  - Precision: {prec:.4f}
  - Recall:    {rec:.4f}
  - F1-Score:  {f1:.4f}

Files Generated:
  ✓ 13_deep_learning_attention.h5
  ✓ 13_deep_learning_attention_results.csv
  ✓ 13_deep_learning_attention_selected_features.csv
  ✓ 13_deep_learning_attention_importance.csv
  ✓ 13_deep_learning_attention_training.png
  ✓ 13_deep_learning_attention_confusion_matrix.png
  ✓ 13_deep_learning_attention_importance.png
"""

print(summary)

with open(tables_dir / "13_deep_learning_attention_summary.txt", 'w') as f:
    f.write(summary)

print("="*80)
