"""
Model Evaluation Module
Comprehensive evaluation metrics and confusion matrix analysis
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    """
    Comprehensive model evaluation with multiple metrics
    """
    
    def __init__(self):
        """Initialize the evaluator"""
        self.results = {}
    
    def evaluate_classification(self, y_true, y_pred, y_pred_proba=None,
                               labels=None, target_names=None):
        """
        Comprehensive classification evaluation
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Prediction probabilities (optional)
            labels: Label list for confusion matrix
            target_names: Class names for classification report
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Handle multi-class vs binary
        average_method = 'weighted' if len(np.unique(y_true)) > 2 else 'binary'
        
        precision = precision_score(y_true, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_true, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_true, y_pred, average=average_method, zero_division=0)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        # AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
                results['auc'] = auc
            except:
                results['auc'] = None
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        results['confusion_matrix'] = cm
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            labels=labels,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )
        results['classification_report'] = report
        
        return results
    
    def print_evaluation(self, results, model_name="Model"):
        """
        Print evaluation results in a formatted way
        
        Args:
            results: Results dictionary from evaluate_classification
            model_name: Name of the model
        """
        print("\n" + "="*80)
        print(f"{model_name.upper()} EVALUATION RESULTS")
        print("="*80)
        
        print("\nOverall Metrics:")
        print(f"  Accuracy:  {results['accuracy']:.4f}")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall:    {results['recall']:.4f}")
        print(f"  F1-Score:  {results['f1_score']:.4f}")
        
        if 'auc' in results and results['auc'] is not None:
            print(f"  AUC-ROC:   {results['auc']:.4f}")
        
        print("\nConfusion Matrix:")
        print(results['confusion_matrix'])
        
        print("\nPer-Class Metrics:")
        report_df = pd.DataFrame(results['classification_report']).transpose()
        print(report_df.round(4))
        
        print("="*80)
    
    def calculate_roc_data(self, y_true, y_pred_proba):
        """
        Calculate ROC curve data
        
        Args:
            y_true: True binary labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        return fpr, tpr, thresholds
    
    def calculate_pr_data(self, y_true, y_pred_proba):
        """
        Calculate Precision-Recall curve data
        
        Args:
            y_true: True binary labels
            y_pred_proba: Prediction probabilities
            
        Returns:
            Tuple of (precision, recall, thresholds)
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        return precision, recall, thresholds
    
    def compare_models(self, results_dict):
        """
        Compare multiple models
        
        Args:
            results_dict: Dictionary of {model_name: results}
            
        Returns:
            DataFrame with comparison
        """
        comparison = []
        
        for model_name, results in results_dict.items():
            row = {
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1_score']
            }
            if 'auc' in results and results['auc'] is not None:
                row['AUC'] = results['auc']
            
            comparison.append(row)
        
        comparison_df = pd.DataFrame(comparison)
        
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        print(comparison_df.to_string(index=False))
        print("="*80)
        
        return comparison_df
    
    def get_top_errors(self, y_true, y_pred, X_test, student_info, top_n=10):
        """
        Identify students with worst predictions
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            X_test: Test features
            student_info: Student information dataframe
            top_n: Number of top errors to return
            
        Returns:
            DataFrame with top misclassified students
        """
        # Calculate absolute error
        error = np.abs(y_true - y_pred)
        
        # Get indices of top errors
        error_indices = np.argsort(error)[-top_n:][::-1]
        
        # Create results dataframe
        error_df = pd.DataFrame({
            'Index': error_indices,
            'True_Value': y_true.iloc[error_indices].values if hasattr(y_true, 'iloc') else y_true[error_indices],
            'Predicted_Value': y_pred[error_indices],
            'Error': error[error_indices]
        })
        
        return error_df
    
    def save_results(self, results, filepath):
        """Save evaluation results to file"""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results_serializable[key] = value.tolist()
            elif isinstance(value, dict):
                results_serializable[key] = value
            else:
                results_serializable[key] = float(value) if isinstance(value, (np.float32, np.float64)) else value
        
        with open(filepath, 'w') as f:
            json.dump(results_serializable, f, indent=4)
        
        print(f"✓ Results saved to: {filepath}")


if __name__ == "__main__":
    # Test the evaluator
    evaluator = ModelEvaluator()
    
    # Sample data
    y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1, 0, 1, 1, 0])
    y_pred_proba = np.array([0.2, 0.8, 0.4, 0.1, 0.9, 0.7, 0.3, 0.6, 0.85, 0.15])
    
    results = evaluator.evaluate_classification(
        y_true, y_pred, y_pred_proba,
        labels=[0, 1],
        target_names=['No Dropout', 'Dropout']
    )
    
    evaluator.print_evaluation(results, "Test Model")
