"""
Evaluate the Binary DPN-A Model (87% accuracy)
Loads the saved best model and evaluates on test set
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, roc_auc_score, roc_curve,
                            precision_recall_curve, f1_score)
import tensorflow as tf
from tensorflow import keras

# Custom Attention Layer (same as training)
class AttentionLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                shape=(input_shape[-1], input_shape[-1]),
                                initializer='glorot_uniform',
                                trainable=True)
        self.b = self.add_weight(name='attention_bias',
                                shape=(input_shape[-1],),
                                initializer='zeros',
                                trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, x):
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=-1)
        output = x * a
        return output

print("="*80)
print("EVALUATING BINARY DPN-A MODEL (87% TARGET)")
print("="*80)

# 1. Load data
print("\n1. Loading dataset...")
df = pd.read_csv('../data/educational_data.csv')
print(f"Dataset shape: {df.shape}")

# Prepare features and target  
X = df.drop('Target', axis=1)
y_original = df['Target']

# Encode target for 3-class
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_3class = le.fit_transform(y_original)
target_names = le.classes_

# Convert 3-class to binary
# Dropout class is 0 in 3-class encoding (alphabetical: Dropout, Enrolled, Graduate)
y_binary = (y_3class == 0).astype(int)  # Dropout=1, Not Dropout=0
print(f"\nBinary distribution:")
print(f"  Not Dropout: {(y_binary==0).sum()} ({(y_binary==0).sum()/len(y_binary)*100:.1f}%)")
print(f"  Dropout: {(y_binary==1).sum()} ({(y_binary==1).sum()/len(y_binary)*100:.1f}%)")

# Train-test split (same as training)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

# 2. Standardize features
print("\n2. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Load model
print("\n3. Loading best model...")
model = keras.models.load_model(
    'outputs/models/14_dpna_binary_best.h5',
    custom_objects={'AttentionLayer': AttentionLayer}
)

print("\nModel loaded successfully!")
print(f"Total parameters: {model.count_params():,}")

# 4. Make predictions
print("\n4. Making predictions on test set...")
y_pred_proba = model.predict(X_test_scaled, verbose=0).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

# 5. Calculate metrics
print("\n5. Calculating metrics...")
accuracy = accuracy_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_pred_proba)
f1 = f1_score(y_test, y_pred)

print(f"\n{'='*80}")
print(f"TEST SET RESULTS:")
print(f"{'='*80}")
print(f"Accuracy:  {accuracy*100:.2f}% (Target: 87.05%)")
print(f"AUC-ROC:   {auc_roc:.4f} (Target: 0.9100)")
print(f"F1-Score:  {f1:.4f}")
print(f"{'='*80}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                          target_names=['Not Dropout', 'Dropout'],
                          digits=4))

# 6. Confusion Matrix
print("\n6. Generating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Dropout', 'Dropout'],
            yticklabels=['Not Dropout', 'Dropout'])
plt.title(f'Binary DPN-A Confusion Matrix\nAccuracy: {accuracy*100:.2f}%', 
          fontsize=14, fontweight='bold')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('outputs/plots/14_binary_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/plots/14_binary_confusion_matrix.png")

# 7. ROC Curve
print("\n7. Generating ROC curve...")
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'DPN-A (AUC = {auc_roc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
         label='Random (AUC = 0.5000)')
plt.axhline(y=0.910, color='green', linestyle=':', lw=2, 
           label='Journal Target (0.910)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve - Binary Dropout Prediction', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/14_binary_roc_curve.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/plots/14_binary_roc_curve.png")

# 8. Precision-Recall Curve
print("\n8. Generating Precision-Recall curve...")
precision, recall, pr_thresholds = precision_recall_curve(y_test, y_pred_proba)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='purple', lw=2, 
         label=f'DPN-A (F1 = {f1:.4f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve - Binary Dropout Prediction', 
          fontsize=14, fontweight='bold')
plt.legend(loc="upper right", fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/plots/14_binary_precision_recall.png', dpi=300, bbox_inches='tight')
print("✓ Saved: outputs/plots/14_binary_precision_recall.png")

# 9. Threshold analysis
print("\n9. Analyzing optimal threshold...")
# Find best threshold for F1
from sklearn.metrics import f1_score
thresholds_to_test = np.arange(0.3, 0.7, 0.05)
f1_scores = []
for thresh in thresholds_to_test:
    y_pred_thresh = (y_pred_proba > thresh).astype(int)
    f1_scores.append(f1_score(y_test, y_pred_thresh))

best_threshold = thresholds_to_test[np.argmax(f1_scores)]
best_f1 = max(f1_scores)

print(f"Default threshold (0.5): F1 = {f1:.4f}")
print(f"Optimal threshold ({best_threshold:.2f}): F1 = {best_f1:.4f}")

# 10. Feature importance from attention weights
print("\n10. Extracting feature importance from attention layer...")
# Get attention layer weights
attention_layer = model.get_layer('attention')
W = attention_layer.get_weights()[0]  # Weight matrix
b = attention_layer.get_weights()[1]  # Bias vector

# Calculate attention scores for each feature
attention_scores = np.abs(W).mean(axis=1)
feature_importance = attention_scores / attention_scores.sum()

# Get top 15 features
feature_names = X.columns.tolist()
# Map to first layer features (34 input features to 64 hidden)
# We need to get importance from input layer weights
input_layer = model.get_layer('hidden1')
W_input = input_layer.get_weights()[0]  # Shape: (34, 64)

# Calculate importance as L2 norm of input weights
input_importance = np.linalg.norm(W_input, axis=1)
input_importance = input_importance / input_importance.sum()

# Create dataframe
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': input_importance
}).sort_values('Importance', ascending=False)

print("\nTop 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# Plot top 15
plt.figure(figsize=(12, 8))
top_15 = importance_df.head(15)
plt.barh(range(15), top_15['Importance'].values, color='steelblue')
plt.yticks(range(15), top_15['Feature'].values)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Top 15 Features - Binary DPN-A Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/plots/14_binary_feature_importance.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: outputs/plots/14_binary_feature_importance.png")

# 11. Per-class performance
print("\n11. Per-class performance:")
tn, fp, fn, tp = cm.ravel()
print(f"\nNot Dropout (0):")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  Specificity: {tn/(tn+fp)*100:.2f}%")
print(f"  Precision:   {tn/(tn+fn)*100:.2f}%")

print(f"\nDropout (1):")
print(f"  True Positives:  {tp}")
print(f"  False Negatives: {fn}")
print(f"  Sensitivity/Recall: {tp/(tp+fn)*100:.2f}%")
print(f"  Precision:          {tp/(tp+fp)*100:.2f}%")

# 12. Comparison with journal
print("\n" + "="*80)
print("COMPARISON WITH JOURNAL RESULTS")
print("="*80)
print(f"{'Metric':<20} {'Journal':<15} {'Our Model':<15} {'Difference':<15}")
print("-"*80)
print(f"{'Accuracy':<20} {'87.05%':<15} {f'{accuracy*100:.2f}%':<15} {f'{(accuracy*100-87.05):.2f}%':<15}")
print(f"{'AUC-ROC':<20} {'0.9100':<15} {f'{auc_roc:.4f}':<15} {f'{(auc_roc-0.910):.4f}':<15}")
print("="*80)

if accuracy*100 >= 87.05:
    print("\n✓ SUCCESS! Model meets or exceeds journal target accuracy!")
else:
    print(f"\n⚠ Model is {87.05 - accuracy*100:.2f}% below journal target")

if auc_roc >= 0.910:
    print("✓ SUCCESS! Model meets or exceeds journal target AUC-ROC!")
else:
    print(f"⚠ Model is {0.910 - auc_roc:.4f} below journal target AUC-ROC")

print("\n" + "="*80)
print("EVALUATION COMPLETE!")
print("="*80)
print("\nGenerated files:")
print("  1. outputs/plots/14_binary_confusion_matrix.png")
print("  2. outputs/plots/14_binary_roc_curve.png")
print("  3. outputs/plots/14_binary_precision_recall.png")
print("  4. outputs/plots/14_binary_feature_importance.png")
