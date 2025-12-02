"""
Main Execution Script
End-to-end pipeline for student performance and dropout prediction with LLM recommendations
"""

import os
import sys
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_preprocessing import DataPreprocessor
from src.models.performance_model import PerformanceModel
from src.models.dropout_model import DropoutModel
from src.models.hybrid_model import HybridModel
from src.evaluation import ModelEvaluator
from src.visualization import Visualizer
from src.llm.recommendation_engine import RecommendationEngine


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")


def main():
    """Main execution pipeline"""
    
    print_header("STUDENT PERFORMANCE AND DROPOUT PREDICTION SYSTEM")
    print("Research Project: ULAB Undergraduate Student Analytics with Deep Learning and LLM")
    print("Target Population: Undergraduate Students (Semesters 1-8)")
    print("="*80)
    
    # Load environment variables
    load_dotenv()
    
    # Configuration
    DATA_PATH = 'data/ulab_students_dataset.csv'
    MODEL_DIR = 'outputs/models'
    PLOT_DIR = 'outputs/plots'
    REPORT_DIR = 'outputs/reports'
    
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    # ========== PHASE 1: DATA PREPROCESSING ==========
    print_header("PHASE 1: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor(DATA_PATH, random_state=42)
    preprocessor.explore_data()
    
    X_train, X_val, X_test, \
    y_grade_train, y_grade_val, y_grade_test, \
    y_dropout_train, y_dropout_val, y_dropout_test, \
    feature_names = preprocessor.prepare_data()
    
    print(f"\n✓ Data preprocessing completed")
    print(f"  Features: {len(feature_names)}")
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # ========== PHASE 2: MODEL TRAINING ==========
    print_header("PHASE 2: MODEL TRAINING")
    
    # Initialize models
    input_dim = X_train.shape[1]
    
    # Model 1: Performance (Grade) Prediction
    print("\n--- Training Performance Prediction Model ---")
    perf_model = PerformanceModel(input_dim=input_dim, num_classes=9)
    perf_history = perf_model.train(
        X_train, y_grade_train,
        X_val, y_grade_val,
        epochs=100,
        batch_size=16,
        verbose=0
    )
    perf_model.save(f'{MODEL_DIR}/performance_model.h5')
    
    # Model 2: Dropout Prediction
    print("\n--- Training Dropout Prediction Model ---")
    dropout_model = DropoutModel(input_dim=input_dim)
    dropout_history = dropout_model.train(
        X_train, y_dropout_train,
        X_val, y_dropout_val,
        epochs=100,
        batch_size=16,
        verbose=0
    )
    dropout_model.save(f'{MODEL_DIR}/dropout_model.h5')
    
    # Model 3: Hybrid Multi-Task Model
    print("\n--- Training Hybrid Multi-Task Model ---")
    hybrid_model = HybridModel(input_dim=input_dim, num_grade_classes=9)
    hybrid_history = hybrid_model.train(
        X_train, y_grade_train, y_dropout_train,
        X_val, y_grade_val, y_dropout_val,
        epochs=100,
        batch_size=16,
        verbose=0
    )
    hybrid_model.save(f'{MODEL_DIR}/hybrid_model.h5')
    
    print("\n✓ All models trained successfully")
    
    # ========== PHASE 3: MODEL EVALUATION ==========
    print_header("PHASE 3: MODEL EVALUATION")
    
    evaluator = ModelEvaluator()
    viz = Visualizer(save_dir=PLOT_DIR)
    
    # Grade labels
    grade_labels = ['D+', 'C', 'C+', 'B-', 'B', 'B+', 'A-', 'A', 'A+']
    
    # Evaluate Performance Model
    print("\n--- Evaluating Performance Model ---")
    y_grade_pred = perf_model.predict(X_test)
    perf_results = evaluator.evaluate_classification(
        y_grade_test, y_grade_pred,
        labels=list(range(9)),
        target_names=grade_labels
    )
    evaluator.print_evaluation(perf_results, "Performance Model")
    
    # Evaluate Dropout Model
    print("\n--- Evaluating Dropout Model ---")
    y_dropout_pred = dropout_model.predict(X_test)
    y_dropout_proba = dropout_model.predict_proba(X_test)
    dropout_results = evaluator.evaluate_classification(
        y_dropout_test, y_dropout_pred, y_dropout_proba,
        labels=[0, 1],
        target_names=['No Dropout', 'Dropout']
    )
    evaluator.print_evaluation(dropout_results, "Dropout Model")
    
    # Evaluate Hybrid Model
    print("\n--- Evaluating Hybrid Model ---")
    y_grade_pred_hybrid, y_dropout_pred_hybrid = hybrid_model.predict(X_test)
    y_grade_proba_hybrid, y_dropout_proba_hybrid = hybrid_model.predict_proba(X_test)
    
    hybrid_grade_results = evaluator.evaluate_classification(
        y_grade_test, y_grade_pred_hybrid,
        labels=list(range(9)),
        target_names=grade_labels
    )
    evaluator.print_evaluation(hybrid_grade_results, "Hybrid Model (Grade)")
    
    hybrid_dropout_results = evaluator.evaluate_classification(
        y_dropout_test, y_dropout_pred_hybrid, y_dropout_proba_hybrid,
        labels=[0, 1],
        target_names=['No Dropout', 'Dropout']
    )
    evaluator.print_evaluation(hybrid_dropout_results, "Hybrid Model (Dropout)")
    
    # Model comparison
    comparison_dict = {
        'Performance Model': perf_results,
        'Dropout Model': dropout_results,
        'Hybrid (Grade)': hybrid_grade_results,
        'Hybrid (Dropout)': hybrid_dropout_results
    }
    comparison_df = evaluator.compare_models(comparison_dict)
    
    # ========== PHASE 4: VISUALIZATION ==========
    print_header("PHASE 4: GENERATING VISUALIZATIONS")
    
    # Confusion matrices
    viz.plot_confusion_matrix(
        y_grade_test, y_grade_pred,
        labels=list(range(9)),
        target_names=grade_labels,
        title='Performance Model - Confusion Matrix',
        save_name='cm_performance.png'
    )
    
    viz.plot_confusion_matrix(
        y_dropout_test, y_dropout_pred,
        labels=[0, 1],
        target_names=['No Dropout', 'Dropout'],
        title='Dropout Model - Confusion Matrix',
        save_name='cm_dropout.png'
    )
    
    # Training history
    viz.plot_training_history(
        perf_history,
        metrics=['loss', 'accuracy'],
        title='Performance Model Training History',
        save_name='history_performance.png'
    )
    
    viz.plot_training_history(
        dropout_history,
        metrics=['loss', 'accuracy'],
        title='Dropout Model Training History',
        save_name='history_dropout.png'
    )
    
    # ROC and PR curves for dropout model
    viz.plot_roc_curve(
        y_dropout_test, y_dropout_proba,
        title='Dropout Prediction - ROC Curve',
        save_name='roc_dropout.png'
    )
    
    viz.plot_precision_recall_curve(
        y_dropout_test, y_dropout_proba,
        title='Dropout Prediction - Precision-Recall Curve',
        save_name='pr_dropout.png'
    )
    
    # Grade distribution
    viz.plot_grade_distribution(
        y_grade_test, y_grade_pred,
        grade_labels,
        title='Grade Distribution: Actual vs Predicted',
        save_name='grade_distribution.png'
    )
    
    # Dropout risk distribution
    viz.plot_dropout_risk_distribution(
        y_dropout_proba,
        title='Dropout Risk Distribution',
        save_name='dropout_risk_dist.png'
    )
    
    # Model comparison
    viz.plot_model_comparison(
        comparison_df,
        metric='Accuracy',
        title='Model Comparison - Accuracy',
        save_name='comparison_accuracy.png'
    )
    
    print("\n✓ All visualizations generated")
    
    # ========== PHASE 5: LLM RECOMMENDATIONS ==========
    print_header("PHASE 5: GENERATING PERSONALIZED RECOMMENDATIONS")
    
    # Initialize recommendation engine
    rec_engine = RecommendationEngine()
    
    # Get test data with original student info
    test_indices = X_test.index
    original_data = preprocessor.original_data.iloc[test_indices].copy()
    
    # Generate recommendations for high-risk students
    high_risk_threshold = 0.5
    high_risk_indices = np.where(y_dropout_proba > high_risk_threshold)[0]
    
    print(f"\nIdentified {len(high_risk_indices)} high-risk students")
    print("Generating personalized recommendations...")
    
    recommendations_list = []
    
    for i, idx in enumerate(high_risk_indices[:5]):  # Generate for top 5 high-risk students
        student_idx = test_indices[idx]
        student_data = original_data.iloc[idx].to_dict()
        
        predicted_grade = grade_labels[y_grade_pred[idx]]
        dropout_prob = y_dropout_proba[idx]
        
        print(f"\nStudent {i+1}: {student_data['name']} (Risk: {dropout_prob:.2%})")
        
        recommendation = rec_engine.generate_recommendations(
            student_data,
            predicted_grade,
            dropout_prob
        )
        
        recommendations_list.append({
            'student_id': student_data['student_id'],
            'name': student_data['name'],
            'predicted_grade': predicted_grade,
            'dropout_probability': dropout_prob,
            'recommendations': recommendation
        })
        
        # Save individual report
        report_path = f"{REPORT_DIR}/recommendation_{student_data['student_id']}.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"PERSONALIZED RECOMMENDATION REPORT\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Student ID: {student_data['student_id']}\n")
            f.write(f"Name: {student_data['name']}\n")
            f.write(f"Department: {student_data['department']}\n")
            f.write(f"Semester: {student_data['semester']}\n\n")
            f.write(f"Current CGPA: {student_data['cgpa']}\n")
            f.write(f"Predicted Final Grade: {predicted_grade}\n")
            f.write(f"Dropout Risk: {dropout_prob:.2%} ({rec_engine._get_risk_level(dropout_prob)})\n\n")
            f.write(f"="*80 + "\n\n")
            f.write(recommendation)
        
        print(f"  ✓ Report saved: {report_path}")
    
    print(f"\n✓ Generated recommendations for {len(recommendations_list)} students")
    
    # ========== SUMMARY ==========
    print_header("EXECUTION SUMMARY")
    
    print("✓ Data Processing: Complete")
    print(f"  - Total samples: {len(preprocessor.original_data)}")
    print(f"  - Features engineered: {len(feature_names)}")
    
    print("\n✓ Model Training: Complete")
    print(f"  - Performance Model: {perf_results['accuracy']:.4f} accuracy")
    print(f"  - Dropout Model: {dropout_results['accuracy']:.4f} accuracy, {dropout_results['auc']:.4f} AUC")
    print(f"  - Hybrid Model: {hybrid_grade_results['accuracy']:.4f} grade acc, " +
          f"{hybrid_dropout_results['accuracy']:.4f} dropout acc")
    
    print("\n✓ Visualizations: Complete")
    print(f"  - Saved to: {PLOT_DIR}/")
    
    print("\n✓ Recommendations: Complete")
    print(f"  - Generated for {len(recommendations_list)} high-risk students")
    print(f"  - Saved to: {REPORT_DIR}/")
    
    print("\n" + "="*80)
    print("RESEARCH PROJECT EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    return {
        'preprocessor': preprocessor,
        'models': {
            'performance': perf_model,
            'dropout': dropout_model,
            'hybrid': hybrid_model
        },
        'results': comparison_dict,
        'recommendations': recommendations_list
    }


if __name__ == "__main__":
    try:
        results = main()
        print("\n✓ Pipeline executed successfully!")
        print("\nNext steps:")
        print("  1. Review visualizations in outputs/plots/")
        print("  2. Read student recommendations in outputs/reports/")
        print("  3. Analyze model performance metrics")
        print("  4. Consider expanding dataset for better accuracy")
        
    except Exception as e:
        print(f"\n✗ Error during execution: {str(e)}")
        import traceback
        traceback.print_exc()
