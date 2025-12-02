"""
Visualization Module
Create comprehensive plots and visualizations for model evaluation and data analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve
import os
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


class Visualizer:
    """
    Comprehensive visualization tools for model evaluation and data analysis
    """
    
    def __init__(self, save_dir='outputs/plots'):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        print(f"Visualizer initialized. Plots will be saved to: {save_dir}")
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, 
                             target_names=None, title='Confusion Matrix',
                             save_name='confusion_matrix.png'):
        """
        Plot confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Label order
            target_names: Class names
            title: Plot title
            save_name: Filename to save
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=target_names if target_names else labels,
                   yticklabels=target_names if target_names else labels)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_training_history(self, history, metrics=['loss', 'accuracy'],
                             title='Training History', save_name='training_history.png'):
        """
        Plot training and validation metrics over epochs
        
        Args:
            history: Keras History object
            metrics: List of metrics to plot
            title: Plot title
            save_name: Filename to save
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            if metric in history.history:
                axes[idx].plot(history.history[metric], label=f'Training {metric}', linewidth=2)
                
                if f'val_{metric}' in history.history:
                    axes[idx].plot(history.history[f'val_{metric}'], 
                                 label=f'Validation {metric}', linewidth=2)
                
                axes[idx].set_xlabel('Epoch', fontsize=11)
                axes[idx].set_ylabel(metric.capitalize(), fontsize=11)
                axes[idx].set_title(f'{metric.capitalize()} vs Epoch', fontsize=12, fontweight='bold')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_roc_curve(self, y_true, y_pred_proba, title='ROC Curve',
                      save_name='roc_curve.png'):
        """
        Plot ROC curve for binary classification
        
        Args:
            y_true: True binary labels
            y_pred_proba: Prediction probabilities
            title: Plot title
            save_name: Filename to save
        """
        from sklearn.metrics import roc_auc_score
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba,
                                   title='Precision-Recall Curve',
                                   save_name='pr_curve.png'):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True binary labels
            y_pred_proba: Prediction probabilities
            title: Plot title
            save_name: Filename to save
        """
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2)
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_feature_importance(self, feature_names, importance_values, top_n=15,
                               title='Feature Importance', save_name='feature_importance.png'):
        """
        Plot feature importance bar chart
        
        Args:
            feature_names: List of feature names
            importance_values: Importance values
            top_n: Number of top features to show
            title: Plot title
            save_name: Filename to save
        """
        # Create dataframe and sort
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(importance_df)), importance_df['Importance'], color='steelblue')
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Importance', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_grade_distribution(self, y_true, y_pred, grade_labels,
                               title='Grade Distribution: True vs Predicted',
                               save_name='grade_distribution.png'):
        """
        Plot distribution comparison for grade predictions
        
        Args:
            y_true: True grade labels
            y_pred: Predicted grade labels
            grade_labels: List of grade names
            title: Plot title
            save_name: Filename to save
        """
        # Convert to grade labels if numeric
        if isinstance(y_true[0], (int, np.integer)):
            y_true_labels = [grade_labels[i] for i in y_true]
            y_pred_labels = [grade_labels[i] for i in y_pred]
        else:
            y_true_labels = y_true
            y_pred_labels = y_pred
        
        # Count distributions
        true_counts = pd.Series(y_true_labels).value_counts().reindex(grade_labels, fill_value=0)
        pred_counts = pd.Series(y_pred_labels).value_counts().reindex(grade_labels, fill_value=0)
        
        # Plot
        x = np.arange(len(grade_labels))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, pred_counts, width, label='Predicted', color='coral', alpha=0.8)
        
        ax.set_xlabel('Grade', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(grade_labels)
        ax.legend(fontsize=11)
        ax.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_dropout_risk_distribution(self, dropout_probs, bins=10,
                                      title='Dropout Risk Distribution',
                                      save_name='dropout_risk_dist.png'):
        """
        Plot histogram of dropout probabilities
        
        Args:
            dropout_probs: Array of dropout probabilities
            bins: Number of histogram bins
            title: Plot title
            save_name: Filename to save
        """
        plt.figure(figsize=(10, 6))
        plt.hist(dropout_probs, bins=bins, color='coral', alpha=0.7, edgecolor='black')
        plt.axvline(x=0.3, color='orange', linestyle='--', linewidth=2, label='Low/Medium Risk')
        plt.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='Medium/High Risk')
        plt.xlabel('Dropout Probability', fontsize=12)
        plt.ylabel('Number of Students', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_model_comparison(self, comparison_df, metric='Accuracy',
                            title='Model Comparison', save_name='model_comparison.png'):
        """
        Plot model comparison bar chart
        
        Args:
            comparison_df: DataFrame with model comparison data
            metric: Metric to plot
            title: Plot title
            save_name: Filename to save
        """
        if metric not in comparison_df.columns:
            print(f"Warning: {metric} not found in comparison data")
            return
        
        plt.figure(figsize=(10, 6))
        plt.bar(comparison_df['Model'], comparison_df[metric], color='steelblue', alpha=0.8)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel(metric, fontsize=12)
        plt.title(f'{title} - {metric}', fontsize=14, fontweight='bold')
        plt.ylim(0, 1.0)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(comparison_df[metric]):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()
    
    def plot_correlation_heatmap(self, data, features=None, title='Feature Correlation',
                                save_name='correlation_heatmap.png'):
        """
        Plot correlation heatmap
        
        Args:
            data: DataFrame with features
            features: List of features to include (None = all)
            title: Plot title
            save_name: Filename to save
        """
        if features:
            data = data[features]
        
        corr = data.corr()
        
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        filepath = os.path.join(self.save_dir, save_name)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Saved: {filepath}")
        plt.close()


if __name__ == "__main__":
    # Test visualizer
    viz = Visualizer()
    
    # Sample data
    y_true = np.random.randint(0, 2, 100)
    y_pred = np.random.randint(0, 2, 100)
    y_pred_proba = np.random.rand(100)
    
    viz.plot_confusion_matrix(
        y_true, y_pred,
        labels=[0, 1],
        target_names=['No Dropout', 'Dropout'],
        title='Test Confusion Matrix'
    )
    
    print("✓ Visualizer test completed")
