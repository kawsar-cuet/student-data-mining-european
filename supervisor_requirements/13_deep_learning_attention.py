# -*- coding: utf-8 -*-
"""
Deep Learning with Attention Mechanism for Student Dropout Prediction
Tests a custom Deep Neural Network with Attention Layer using different 
feature selection methods to optimize performance for multi-class classification.

Architecture:
- Input Layer
- Dense(64, relu) + BatchNormalization + Dropout(0.3)
- Attention Layer (custom)
- Dense(32, relu) + Dropout(0.2)
- Dense(16, relu)
- Output(3, softmax) for 3-class classification
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import (
    SelectKBest, f_classif, chi2, mutual_info_classif,
    RFE, SelectFromModel
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create output directories
output_dir = Path("outputs")
figures_dir = output_dir / "figures"
tables_dir = output_dir / "tables"
models_dir = output_dir / "models"
for dir_path in [output_dir, figures_dir, tables_dir, models_dir]:
    dir_path.mkdir(exist_ok=True)

print("="*80)
print("DEEP LEARNING WITH ATTENTION MECHANISM")
print("Feature Selection Optimization for Multi-Class Dropout Prediction")
print("="*80)


# ========== ATTENTION LAYER DEFINITION ==========
class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for Deep Neural Networks
    
    Implements self-attention mechanism to weight features based on their importance.
    The layer learns attention weights that highlight the most relevant features.
    
    Args:
        units: Dimensionality of the attention space (default: input dimension)
    """
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        """Initialize learnable parameters"""
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
        """Forward pass - apply attention mechanism"""
        # Compute attention scores
        e = tf.nn.tanh(tf.matmul(inputs, self.W) + self.b)
        # Compute attention weights using softmax
        a = tf.nn.softmax(e, axis=-1)
        # Apply attention weights to inputs
        output = inputs * a
        return output
    
    def get_config(self):
        """Enable model serialization"""
        config = super(AttentionLayer, self).get_config()
        return config


# ========== MODEL BUILDING FUNCTION ==========
def build_attention_model(input_dim, num_classes=3):
    """
    Build Deep Neural Network with Attention Mechanism
    
    Architecture:
    - Dense(64) + BatchNorm + Dropout(0.3)
    - Attention Layer
    - Dense(32) + Dropout(0.2)
    - Dense(16)
    - Output(num_classes, softmax)
    
    Args:
        input_dim: Number of input features
        num_classes: Number of target classes (default: 3)
    
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=(input_dim,))
    
    # First hidden layer with batch normalization
    x = layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Attention layer
    x = AttentionLayer()(x)
    
    # Second hidden layer
    x = layers.Dense(32, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third hidden layer
    x = layers.Dense(16, activation='relu')(x)
    
    # Output layer (multi-class)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# ========== LOAD AND PREPARE DATA ==========
print("\n1. Loading dataset")
print("-" * 60)

data_path = Path("../data/educational_data.csv")
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
    target_names = ['Dropout', 'Enrolled', 'Graduate']

num_classes = len(np.unique(y))
print(f"   Features: {X.shape[1]}")
print(f"   Classes: {num_classes}")
print(f"   Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")


# ========== FEATURE SELECTION OPTIMIZATION ==========
print("\n" + "="*80)
print("2. TESTING FEATURE SELECTION METHODS")
print("="*80)

# Feature counts to test
feature_counts = [10, 15, 20, 25, 30, 34]

results = []


# Helper function to calculate Information Gain
def calculate_information_gain(X, y):
    """Calculate information gain for each feature"""
    ig_scores = []
    for col in X.columns:
        score = mutual_info_classif(X[[col]], y, random_state=42)[0]
        ig_scores.append(score)
    return np.array(ig_scores)


# Baseline: All features
print("\n[Baseline] ALL FEATURES (34)")
print("-" * 60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model_all = build_attention_model(X_train_scaled.shape[1], num_classes)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=0
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=0
)

# Train model
history_all = model_all.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop, reduce_lr],
    verbose=0
)

y_pred_all = np.argmax(model_all.predict(X_test_scaled, verbose=0), axis=1)
acc_all = accuracy_score(y_test, y_pred_all)
prec_all = precision_score(y_test, y_pred_all, average='weighted')
rec_all = recall_score(y_test, y_pred_all, average='weighted')
f1_all = f1_score(y_test, y_pred_all, average='weighted')

print(f"Accuracy:  {acc_all:.4f}")
print(f"Precision: {prec_all:.4f}")
print(f"Recall:    {rec_all:.4f}")
print(f"F1-Score:  {f1_all:.4f}")

results.append({
    'Method': 'All Features',
    'Num_Features': 34,
    'Accuracy': acc_all,
    'Precision': prec_all,
    'Recall': rec_all,
    'F1': f1_all,
    'Features': X.columns.tolist()
})


# Method 1: Information Gain
print("\n[Method 1] INFORMATION GAIN")
print("-" * 60)

ig_scores = calculate_information_gain(X_train, y_train)

for k in feature_counts[:-1]:  # Exclude 34 (already tested)
    print(f"\nTesting with {k} features...")
    
    indices = np.argsort(ig_scores)[::-1][:k]
    selected_features = X_train.columns[indices].tolist()
    
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = build_attention_model(k, num_classes)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Method': 'Information Gain',
        'Num_Features': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': selected_features
    })


# Method 2: Mutual Information
print("\n[Method 2] MUTUAL INFORMATION")
print("-" * 60)

for k in feature_counts[:-1]:
    print(f"\nTesting with {k} features...")
    
    selector = SelectKBest(mutual_info_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = build_attention_model(k, num_classes)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Method': 'Mutual Information',
        'Num_Features': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': selected_features
    })


# Method 3: ANOVA F-statistic
print("\n[Method 3] ANOVA F-STATISTIC")
print("-" * 60)

for k in feature_counts[:-1]:
    print(f"\nTesting with {k} features...")
    
    selector = SelectKBest(f_classif, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = build_attention_model(k, num_classes)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Method': 'ANOVA F-stat',
        'Num_Features': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': selected_features
    })


# Method 4: Random Forest Importance
print("\n[Method 4] RANDOM FOREST IMPORTANCE")
print("-" * 60)

rf_temp = RandomForestClassifier(n_estimators=100, random_state=42)
rf_temp.fit(X_train, y_train)
importances = rf_temp.feature_importances_

for k in feature_counts[:-1]:
    print(f"\nTesting with {k} features...")
    
    indices = np.argsort(importances)[::-1][:k]
    selected_features = X_train.columns[indices].tolist()
    
    X_train_selected = X_train.iloc[:, indices]
    X_test_selected = X_test.iloc[:, indices]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = build_attention_model(k, num_classes)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Method': 'RF Importance',
        'Num_Features': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': selected_features
    })


# Method 5: RFE (Recursive Feature Elimination)
print("\n[Method 5] RECURSIVE FEATURE ELIMINATION (RFE)")
print("-" * 60)

for k in feature_counts[:-1]:
    print(f"\nTesting with {k} features...")
    
    estimator = RandomForestClassifier(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=k, step=5)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.support_].tolist()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    model = build_attention_model(k, num_classes)
    history = model.fit(
        X_train_scaled, y_train,
        validation_split=0.1,
        epochs=100,
        batch_size=16,
        callbacks=[early_stop, reduce_lr],
        verbose=0
    )
    
    y_pred = np.argmax(model.predict(X_test_scaled, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"  Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    
    results.append({
        'Method': 'RFE',
        'Num_Features': k,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'Features': selected_features
    })


# ========== RESULTS ANALYSIS ==========
print("\n" + "="*80)
print("3. RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\nTop 10 Configurations by Accuracy:")
print(results_df[['Method', 'Num_Features', 'Accuracy', 'Precision', 'Recall', 'F1']].head(10).to_string(index=False))

# Save results
results_df.to_csv(tables_dir / "13_deep_learning_attention_results.csv", index=False)
print(f"\n✓ Saved: tables/13_deep_learning_attention_results.csv")


# ========== BEST MODEL TRAINING AND EVALUATION ==========
print("\n" + "="*80)
print("4. TRAINING BEST MODEL")
print("="*80)

best_config = results_df.iloc[0]
print(f"\nBest Configuration:")
print(f"  Method: {best_config['Method']}")
print(f"  Features: {best_config['Num_Features']}")
print(f"  Accuracy: {best_config['Accuracy']:.4f}")

# Retrain best model with best configuration
best_features = best_config['Features']
X_train_best = X_train[best_features]
X_test_best = X_test[best_features]

scaler_best = StandardScaler()
X_train_scaled_best = scaler_best.fit_transform(X_train_best)
X_test_scaled_best = scaler_best.transform(X_test_best)

best_model = build_attention_model(len(best_features), num_classes)

# Enhanced callbacks for best model
best_callbacks = [
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
        str(models_dir / "13_deep_learning_attention_best.h5"),
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

print("\nTraining best model...")
history_best = best_model.fit(
    X_train_scaled_best, y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=16,
    callbacks=best_callbacks,
    verbose=1
)

# Load best weights
best_model.load_weights(str(models_dir / "13_deep_learning_attention_best.h5"))

# Final evaluation
y_pred_best = np.argmax(best_model.predict(X_test_scaled_best, verbose=0), axis=1)
y_pred_proba_best = best_model.predict(X_test_scaled_best, verbose=0)

print("\n" + "="*80)
print("5. FINAL EVALUATION")
print("="*80)

acc_best = accuracy_score(y_test, y_pred_best)
prec_best = precision_score(y_test, y_pred_best, average='weighted')
rec_best = recall_score(y_test, y_pred_best, average='weighted')
f1_best = f1_score(y_test, y_pred_best, average='weighted')

print(f"\nFinal Performance:")
print(f"  Accuracy:  {acc_best:.4f}")
print(f"  Precision: {prec_best:.4f}")
print(f"  Recall:    {rec_best:.4f}")
print(f"  F1-Score:  {f1_best:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=target_names))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
print("\nConfusion Matrix:")
print(cm)


# ========== VISUALIZATIONS ==========
print("\n" + "="*80)
print("6. GENERATING VISUALIZATIONS")
print("="*80)

plt.style.use('default')
sns.set_palette("husl")

# 1. Training History
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy
axes[0].plot(history_best.history['accuracy'], label='Train', linewidth=2)
axes[0].plot(history_best.history['val_accuracy'], label='Validation', linewidth=2)
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

# Loss
axes[1].plot(history_best.history['loss'], label='Train', linewidth=2)
axes[1].plot(history_best.history['val_loss'], label='Validation', linewidth=2)
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_training_history.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 13_deep_learning_attention_training_history.png")
plt.close()


# 2. Feature Selection Comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

methods = results_df['Method'].unique()
colors = sns.color_palette("husl", len(methods))

for metric, ax in zip(['Accuracy', 'Precision', 'Recall', 'F1'], axes.flat):
    for method, color in zip(methods, colors):
        method_data = results_df[results_df['Method'] == method]
        ax.plot(method_data['Num_Features'], method_data[metric], 
               marker='o', label=method, linewidth=2, color=color, markersize=8)
    
    ax.set_xlabel('Number of Features', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{metric} vs Number of Features', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(feature_counts)

plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_feature_selection_comparison.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 13_deep_learning_attention_feature_selection_comparison.png")
plt.close()


# 3. Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=target_names, yticklabels=target_names,
           cbar_kws={'label': 'Count'}, linewidths=0.5)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Deep Learning Attention - Confusion Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(figures_dir / "13_deep_learning_attention_confusion_matrix.png", dpi=300, bbox_inches='tight')
print("✓ Saved: 13_deep_learning_attention_confusion_matrix.png")
plt.close()


# 4. Best Features Importance (using attention weights approximation)
# Note: This is a simplified visualization - actual attention is dynamic
print("\nExtracting feature importance from trained model...")

# Get attention layer weights
attention_weights = None
for layer in best_model.layers:
    if isinstance(layer, AttentionLayer):
        attention_weights = layer.get_weights()[0]  # W matrix
        break

if attention_weights is not None:
    # Calculate feature importance as sum of absolute attention weights
    feature_importance = np.abs(attention_weights).sum(axis=1)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': best_features,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    importance_df.to_csv(tables_dir / "13_deep_learning_attention_feature_importance.csv", index=False)
    print("✓ Saved: tables/13_deep_learning_attention_feature_importance.csv")
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    top_features = importance_df.head(15)
    bars = plt.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Attention Weight Magnitude', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title('Top 15 Features by Attention Weight (Deep Learning)', fontsize=14, fontweight='bold', pad=20)
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(figures_dir / "13_deep_learning_attention_feature_importance.png", dpi=300, bbox_inches='tight')
    print("✓ Saved: 13_deep_learning_attention_feature_importance.png")
    plt.close()


# 5. Model Architecture Visualization
print("\nModel Architecture Summary:")
print("-" * 60)
best_model.summary()

# Save architecture to file
with open(tables_dir / "13_deep_learning_attention_architecture.txt", 'w') as f:
    best_model.summary(print_fn=lambda x: f.write(x + '\n'))
print("✓ Saved: tables/13_deep_learning_attention_architecture.txt")


# ========== FINAL SUMMARY ==========
print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

summary = f"""
Deep Learning with Attention Mechanism - Summary
{"="*60}

Best Configuration:
  - Feature Selection Method: {best_config['Method']}
  - Number of Features: {best_config['Num_Features']}
  - Selected Features: {', '.join(best_features[:10])}{'...' if len(best_features) > 10 else ''}

Final Performance Metrics:
  - Accuracy:  {acc_best:.4f}
  - Precision: {prec_best:.4f}
  - Recall:    {rec_best:.4f}
  - F1-Score:  {f1_best:.4f}

Model Architecture:
  - Input Layer: {len(best_features)} features
  - Hidden Layers: 64 → Attention → 32 → 16
  - Output Layer: {num_classes} classes (softmax)
  - Total Parameters: {best_model.count_params():,}
  - Training Epochs: {len(history_best.history['loss'])}

Files Generated:
  ✓ 13_deep_learning_attention_results.csv
  ✓ 13_deep_learning_attention_best.h5
  ✓ 13_deep_learning_attention_training_history.png
  ✓ 13_deep_learning_attention_feature_selection_comparison.png
  ✓ 13_deep_learning_attention_confusion_matrix.png
  ✓ 13_deep_learning_attention_feature_importance.png
  ✓ 13_deep_learning_attention_feature_importance.csv
  ✓ 13_deep_learning_attention_architecture.txt

{"="*60}
"""

print(summary)

# Save summary
with open(tables_dir / "13_deep_learning_attention_summary.txt", 'w') as f:
    f.write(summary)

print("\n✓ All outputs saved to 'outputs' directory")
print("="*80)
