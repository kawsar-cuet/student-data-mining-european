"""
Adaptive Feature Selection Algorithm (AFSA) for Student Performance Prediction
Based on: "Analyzing students' academic performance using educational data mining"
by Sarker et al. (2024)

This implementation follows the paper's actual approach:
1. Adaptive weighted GPA calculation from internal examinations
2. Decision Tree-based feature selection with voting technique
3. Performance progression analysis (booster/degrader subjects)
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


class AdaptiveGPACalculator:
    """
    Calculates GPA using two adaptive approaches:
    - Proposed GPA-1: Simple average (equal weights)
    - Proposed GPA-2: Weighted sum with progressive importance
    """
    
    def __init__(self, exam_names=['exam1', 'exam2', 'exam3', 'exam4']):
        self.exam_names = exam_names
        self.n_exams = len(exam_names)
    
    def calculate_gpa_method1(self, subject_marks):
        """
        Proposed GPA-1: Simple average across all examinations
        Formula: HSC_Subject = (Exam1 + Exam2 + Exam3 + Exam4) / 4
        
        Parameters:
        -----------
        subject_marks : array-like, shape (n_exams,)
            Marks for a single subject across all examinations
            
        Returns:
        --------
        float : Average marks for the subject
        """
        return np.mean(subject_marks)
    
    def calculate_gpa_method2(self, subject_marks, weights=None):
        """
        Proposed GPA-2: Weighted sum with progressive importance
        Default weights: [0.1, 0.2, 0.3, 0.4] for [Exam1, Exam2, Exam3, Exam4]
        
        Formula: HSC_Subject = (Exam1 × 0.1) + (Exam2 × 0.2) + (Exam3 × 0.3) + (Exam4 × 0.4)
        
        Parameters:
        -----------
        subject_marks : array-like, shape (n_exams,)
            Marks for a single subject across all examinations
        weights : array-like, optional
            Custom weights for each examination
            
        Returns:
        --------
        float : Weighted marks for the subject
        """
        if weights is None:
            weights = [0.1, 0.2, 0.3, 0.4]  # Progressive importance
        
        if len(weights) != len(subject_marks):
            raise ValueError(f"Weights length ({len(weights)}) must match exams ({len(subject_marks)})")
        
        return np.dot(subject_marks, weights)
    
    def marks_to_gpa(self, marks):
        """
        Convert marks to GPA using standard Bangladesh grading scale
        
        Marks Range -> GPA
        80-100 -> 5.00 (A+)
        70-79  -> 4.00 (A)
        60-69  -> 3.50 (A-)
        50-59  -> 3.00 (B)
        40-49  -> 2.00 (C)
        33-39  -> 1.00 (D)
        0-32   -> 0.00 (F)
        """
        if marks >= 80:
            return 5.00
        elif marks >= 70:
            return 4.00
        elif marks >= 60:
            return 3.50
        elif marks >= 50:
            return 3.00
        elif marks >= 40:
            return 2.00
        elif marks >= 33:
            return 1.00
        else:
            return 0.00
    
    def classify_performance(self, marks):
        """
        Classify performance into three categories:
        - Good: 60-100% (marks >= 60)
        - Average: 50-59% (marks >= 50 and < 60)
        - Poor: 0-49% (marks < 50)
        """
        if marks >= 60:
            return 'Good'
        elif marks >= 50:
            return 'Average'
        else:
            return 'Poor'


class DecisionTreeFeatureSelector:
    """
    Identifies predictor subjects using Decision Tree with voting technique
    across multiple impurity measures (Information Gain, Gini Index, Accuracy)
    """
    
    def __init__(self, max_depth=10, min_samples_split=10, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        
        # Three Decision Trees with different criteria
        self.dt_gini = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        self.dt_entropy = DecisionTreeClassifier(
            criterion='entropy',  # Information Gain
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        # For accuracy-based splitting (using log_loss as proxy)
        self.dt_log_loss = DecisionTreeClassifier(
            criterion='log_loss',
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state
        )
        
        self.feature_names = None
        self.feature_importance_votes = {}
    
    def fit(self, X, y, feature_names=None):
        """
        Fit all three decision trees and extract feature importance via voting
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data (subject marks)
        y : array-like, shape (n_samples,)
            Target labels (performance categories)
        feature_names : list, optional
            Names of features (subject names)
        """
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'Subject_{i}' for i in range(X.shape[1])]
        
        # Fit all three trees
        print("\n🌲 Training Decision Trees with different criteria...")
        self.dt_gini.fit(X, y)
        print("  ✓ Gini Index tree trained")
        
        self.dt_entropy.fit(X, y)
        print("  ✓ Information Gain tree trained")
        
        self.dt_log_loss.fit(X, y)
        print("  ✓ Log Loss tree trained")
        
        # Extract features used in each tree
        self._extract_features_from_trees(X)
        
        return self
    
    def _extract_features_from_trees(self, X):
        """
        Extract features used in each decision tree and compute voting scores
        """
        print("\n📊 Extracting predictor subjects via voting technique...")
        
        # Initialize vote counts
        feature_votes = {name: 0 for name in self.feature_names}
        
        # Count features used in Gini tree
        gini_features = self._get_tree_features(self.dt_gini.tree_, self.feature_names)
        for feature in gini_features:
            feature_votes[feature] += 1
        
        # Count features used in Entropy tree
        entropy_features = self._get_tree_features(self.dt_entropy.tree_, self.feature_names)
        for feature in entropy_features:
            feature_votes[feature] += 1
        
        # Count features used in Log Loss tree
        log_loss_features = self._get_tree_features(self.dt_log_loss.tree_, self.feature_names)
        for feature in log_loss_features:
            feature_votes[feature] += 1
        
        self.feature_importance_votes = feature_votes
        
        # Print voting results
        print("\n  Voting Results (Feature Importance Scores):")
        sorted_features = sorted(feature_votes.items(), key=lambda x: x[1], reverse=True)
        for feature, votes in sorted_features:
            print(f"    {feature}: {votes} votes")
        
        return feature_votes
    
    def _get_tree_features(self, tree, feature_names):
        """
        Extract feature indices used in decision tree nodes
        """
        features_used = set()
        
        def recurse(node_id):
            if tree.feature[node_id] != -2:  # Not a leaf node
                features_used.add(feature_names[tree.feature[node_id]])
                recurse(tree.children_left[node_id])
                recurse(tree.children_right[node_id])
        
        recurse(0)  # Start from root
        return features_used
    
    def get_predictor_subjects(self, threshold=2):
        """
        Get subjects that appear in at least 'threshold' decision trees
        
        Parameters:
        -----------
        threshold : int, default=2
            Minimum votes required to be considered a predictor subject
            
        Returns:
        --------
        list : Predictor subjects with >= threshold votes
        """
        predictor_subjects = [
            subject for subject, votes in self.feature_importance_votes.items()
            if votes >= threshold
        ]
        
        print(f"\n✅ Predictor Subjects (threshold={threshold}):")
        for subject in predictor_subjects:
            print(f"   - {subject} ({self.feature_importance_votes[subject]} votes)")
        
        return predictor_subjects
    
    def visualize_tree(self, criterion='gini', figsize=(20, 10), output_path=None):
        """
        Visualize one of the decision trees
        
        Parameters:
        -----------
        criterion : str, one of ['gini', 'entropy', 'log_loss']
            Which tree to visualize
        """
        if criterion == 'gini':
            tree = self.dt_gini
            title = "Decision Tree with Gini Index"
        elif criterion == 'entropy':
            tree = self.dt_entropy
            title = "Decision Tree with Information Gain (Entropy)"
        else:
            tree = self.dt_log_loss
            title = "Decision Tree with Log Loss"
        
        plt.figure(figsize=figsize)
        plot_tree(tree, 
                  feature_names=self.feature_names,
                  class_names=['Poor', 'Average', 'Good'],
                  filled=True,
                  rounded=True,
                  fontsize=10)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n💾 Decision tree saved to: {output_path}")
        
        plt.close()


class PerformanceProgressionAnalyzer:
    """
    Analyzes student performance progression to identify:
    - Booster subjects (improve GPA)
    - Degrader subjects (reduce GPA)
    - At-risk subjects (failure-prone)
    """
    
    def __init__(self, pass_threshold=33, at_risk_threshold=40):
        self.pass_threshold = pass_threshold
        self.at_risk_threshold = at_risk_threshold
    
    def analyze_student_progression(self, student_marks, subject_names):
        """
        Analyze individual student's subject-wise progression
        
        Parameters:
        -----------
        student_marks : array-like, shape (n_subjects, n_exams)
            Marks for all subjects across all exams for one student
        subject_names : list
            Names of subjects
            
        Returns:
        --------
        dict : Analysis results with booster, degrader, and at-risk subjects
        """
        n_subjects = len(subject_names)
        avg_marks = np.mean(student_marks, axis=1)  # Average across exams for each subject
        overall_avg = np.mean(avg_marks)
        
        booster_subjects = []
        degrader_subjects = []
        at_risk_subjects = []
        
        for i, subject in enumerate(subject_names):
            subject_avg = avg_marks[i]
            
            # At-risk: below threshold
            if subject_avg <= self.at_risk_threshold:
                at_risk_subjects.append({
                    'subject': subject,
                    'avg_marks': subject_avg,
                    'status': 'Failed' if subject_avg < self.pass_threshold else 'At-Risk'
                })
            
            # Booster: above overall average
            if subject_avg > overall_avg:
                booster_subjects.append({
                    'subject': subject,
                    'avg_marks': subject_avg,
                    'improvement': subject_avg - overall_avg
                })
            
            # Degrader: below overall average (but not at-risk)
            elif subject_avg <= overall_avg and subject_avg > self.at_risk_threshold:
                degrader_subjects.append({
                    'subject': subject,
                    'avg_marks': subject_avg,
                    'degradation': overall_avg - subject_avg
                })
        
        return {
            'overall_avg': overall_avg,
            'booster_subjects': booster_subjects,
            'degrader_subjects': degrader_subjects,
            'at_risk_subjects': at_risk_subjects
        }
    
    def analyze_cohort_progression(self, all_student_marks, subject_names):
        """
        Analyze entire cohort to categorize subjects
        
        Parameters:
        -----------
        all_student_marks : array-like, shape (n_students, n_subjects, n_exams)
            Marks for all students, subjects, and exams
        subject_names : list
            Names of subjects
            
        Returns:
        --------
        dict : Cohort-level subject categorization
        """
        n_students = all_student_marks.shape[0]
        n_subjects = len(subject_names)
        
        subject_stats = {subject: {'booster_count': 0, 'degrader_count': 0, 'at_risk_count': 0}
                        for subject in subject_names}
        
        for student_idx in range(n_students):
            student_analysis = self.analyze_student_progression(
                all_student_marks[student_idx], subject_names
            )
            
            for item in student_analysis['booster_subjects']:
                subject_stats[item['subject']]['booster_count'] += 1
            
            for item in student_analysis['degrader_subjects']:
                subject_stats[item['subject']]['degrader_count'] += 1
            
            for item in student_analysis['at_risk_subjects']:
                subject_stats[item['subject']]['at_risk_count'] += 1
        
        # Categorize each subject based on affected student counts
        subject_categories = {}
        for subject, stats in subject_stats.items():
            if stats['at_risk_count'] > n_students * 0.3:  # >30% students at-risk
                category = 'At-Risk Subject'
            elif stats['booster_count'] > stats['degrader_count']:
                category = 'Performance Booster'
            else:
                category = 'Performance Degrader'
            
            subject_categories[subject] = {
                'category': category,
                'stats': stats
            }
        
        return subject_categories


def demo_afsa():
    """
    Demonstration of the correct AFSA approach from the paper
    """
    print("="*80)
    print("  ADAPTIVE FEATURE SELECTION ALGORITHM (AFSA)")
    print("  Based on Sarker et al. (2024) - Analyzing Students' Academic Performance")
    print("="*80)
    
    # Simulate dataset (similar to paper: 4 exams, 7 subjects, 100 students)
    np.random.seed(42)
    n_students = 100
    n_exams = 4
    n_subjects = 7
    
    subject_names = ['English', 'Bangla', 'ICT', 'Civics', 'Sociology', 'Islamic History', 'Optional']
    exam_names = ['Half-Yearly', 'Yearly', 'Pre-Test', 'Test']
    
    # Generate marks: shape (n_students, n_subjects, n_exams)
    # Marks range: 30-90 (realistic distribution)
    all_marks = np.random.randint(30, 91, size=(n_students, n_subjects, n_exams))
    
    print(f"\nDataset: {n_students} students, {n_subjects} subjects, {n_exams} examinations")
    
    # ===== PHASE 1: Adaptive GPA Calculation =====
    print("\n" + "="*80)
    print("PHASE 1: Adaptive GPA Calculation")
    print("="*80)
    
    gpa_calc = AdaptiveGPACalculator(exam_names)
    
    # Calculate GPA using both methods for all students
    gpa_method1 = []
    gpa_method2 = []
    performance_labels = []
    
    for student_idx in range(n_students):
        # Average across subjects for each method
        subject_avgs_m1 = []
        subject_avgs_m2 = []
        
        for subject_idx in range(n_subjects):
            subject_marks = all_marks[student_idx, subject_idx, :]
            
            avg_m1 = gpa_calc.calculate_gpa_method1(subject_marks)
            avg_m2 = gpa_calc.calculate_gpa_method2(subject_marks)
            
            subject_avgs_m1.append(avg_m1)
            subject_avgs_m2.append(avg_m2)
        
        overall_avg_m1 = np.mean(subject_avgs_m1)
        overall_avg_m2 = np.mean(subject_avgs_m2)
        
        gpa_method1.append(overall_avg_m1)
        gpa_method2.append(overall_avg_m2)
        
        # Use Method 2 for classification
        performance_labels.append(gpa_calc.classify_performance(overall_avg_m2))
    
    print(f"\n✅ GPA Calculation Complete")
    print(f"   Method 1 (Simple Average) - Mean GPA: {np.mean(gpa_method1):.2f}")
    print(f"   Method 2 (Weighted Sum)   - Mean GPA: {np.mean(gpa_method2):.2f}")
    
    # Performance distribution
    from collections import Counter
    perf_dist = Counter(performance_labels)
    print(f"\n   Performance Distribution:")
    print(f"   - Good: {perf_dist['Good']} students ({perf_dist['Good']/n_students*100:.1f}%)")
    print(f"   - Average: {perf_dist['Average']} students ({perf_dist['Average']/n_students*100:.1f}%)")
    print(f"   - Poor: {perf_dist['Poor']} students ({perf_dist['Poor']/n_students*100:.1f}%)")
    
    # ===== PHASE 2: Decision Tree Feature Selection =====
    print("\n" + "="*80)
    print("PHASE 2: Decision Tree Feature Selection with Voting")
    print("="*80)
    
    # Prepare data: Average marks per subject as features
    X = np.mean(all_marks, axis=2)  # Shape: (n_students, n_subjects)
    y = np.array(performance_labels)
    
    dt_selector = DecisionTreeFeatureSelector(max_depth=5, min_samples_split=5)
    dt_selector.fit(X, y, feature_names=subject_names)
    
    predictor_subjects = dt_selector.get_predictor_subjects(threshold=2)
    
    # ===== PHASE 3: Performance Progression Analysis =====
    print("\n" + "="*80)
    print("PHASE 3: Performance Progression Analysis")
    print("="*80)
    
    prog_analyzer = PerformanceProgressionAnalyzer()
    
    # Analyze cohort
    subject_categories = prog_analyzer.analyze_cohort_progression(all_marks, subject_names)
    
    print("\n📈 Subject Categorization (Cohort Level):")
    for subject, info in subject_categories.items():
        print(f"\n   {subject}: {info['category']}")
        stats = info['stats']
        print(f"     - Booster for {stats['booster_count']} students")
        print(f"     - Degrader for {stats['degrader_count']} students")
        print(f"     - At-Risk for {stats['at_risk_count']} students")
    
    # Sample individual student analysis
    print("\n👤 Sample Individual Student Analysis (Student #1):")
    student_1_analysis = prog_analyzer.analyze_student_progression(all_marks[0], subject_names)
    print(f"   Overall Average: {student_1_analysis['overall_avg']:.2f}")
    print(f"   Booster Subjects: {len(student_1_analysis['booster_subjects'])}")
    for item in student_1_analysis['booster_subjects'][:3]:
        print(f"     - {item['subject']}: {item['avg_marks']:.2f} (+{item['improvement']:.2f})")
    print(f"   At-Risk Subjects: {len(student_1_analysis['at_risk_subjects'])}")
    for item in student_1_analysis['at_risk_subjects']:
        print(f"     - {item['subject']}: {item['avg_marks']:.2f} ({item['status']})")
    
    print("\n" + "="*80)
    print("✨ AFSA Analysis Complete!")
    print("="*80)


if __name__ == "__main__":
    demo_afsa()
