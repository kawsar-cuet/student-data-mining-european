"""
Data Preprocessing Module for Real Educational Dataset
Implements journal methodology for 4,424 students with 35 features
Following the publication-ready methodology from docs/JOURNAL_METHODOLOGY.md
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class RealDataPreprocessor:
    """
    Preprocessing pipeline for real educational dataset
    Implements journal methodology with feature engineering and stratified splits
    """
    
    def __init__(self, data_path='data/educational_data.csv', random_state=42):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the real educational dataset CSV
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.original_data = None
        self.processed_data = None
        self.feature_names = None
        
        print(f"Initializing RealDataPreprocessor for journal methodology")
        print(f"Dataset: {data_path}")
        
    def load_data(self):
        """Load real educational dataset"""
        try:
            self.original_data = pd.read_csv(self.data_path)
            print(f"\n✓ Dataset loaded successfully")
            print(f"  Shape: {self.original_data.shape} (rows × columns)")
            print(f"  Features: {self.original_data.shape[1] - 1} + 1 target")
            
            # Display target distribution
            target_dist = self.original_data['Target'].value_counts()
            print(f"\n  Target Distribution:")
            for status, count in target_dist.items():
                pct = count / len(self.original_data) * 100
                print(f"    • {status}: {count} ({pct:.1f}%)")
            
            return self.original_data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def explore_data(self):
        """Display comprehensive data exploration"""
        if self.original_data is None:
            self.load_data()
        
        print("\n" + "="*90)
        print(" DATA EXPLORATION - REAL EDUCATIONAL DATASET")
        print("="*90)
        
        print(f"\nDataset Dimensions: {self.original_data.shape}")
        print(f"Total Students: {len(self.original_data):,}")
        print(f"Total Features: {len(self.original_data.columns)}")
        
        # Check for missing values
        missing = self.original_data.isnull().sum()
        if missing.sum() > 0:
            print("\n⚠ Missing Values Detected:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values detected (clean dataset)")
        
        # Numerical features summary
        numerical_cols = self.original_data.select_dtypes(include=[np.number]).columns
        print(f"\nNumerical Features: {len(numerical_cols)}")
        
        # Target analysis
        print("\n" + "-"*90)
        print(" TARGET VARIABLE ANALYSIS")
        print("-"*90)
        
        target_counts = self.original_data['Target'].value_counts()
        print("\n3-Class Distribution:")
        for target, count in target_counts.items():
            percentage = (count / len(self.original_data)) * 100
            print(f"  {target:12s}: {count:5d} ({percentage:5.2f}%)")
        
        # Binary dropout analysis
        is_dropout = (self.original_data['Target'] == 'Dropout').sum()
        not_dropout = len(self.original_data) - is_dropout
        print(f"\nBinary Dropout Analysis:")
        print(f"  Not Dropout: {not_dropout:5d} ({not_dropout/len(self.original_data)*100:5.2f}%)")
        print(f"  Dropout    : {is_dropout:5d} ({is_dropout/len(self.original_data)*100:5.2f}%)")
        
        # Key features summary
        print("\n" + "-"*90)
        print(" KEY FEATURES SUMMARY")
        print("-"*90)
        
        key_features = [
            'Age at enrollment',
            'Curricular units 1st sem (grade)',
            'Curricular units 2nd sem (grade)',
            'Curricular units 1st sem (approved)',
            'Curricular units 2nd sem (approved)'
        ]
        
        available_features = [f for f in key_features if f in self.original_data.columns]
        if available_features:
            print(self.original_data[available_features].describe())
        
        return self.original_data
    
    def engineer_features(self, df):
        """
        Create derived features following journal methodology
        
        Feature Engineering Strategy:
        1. Academic Performance Indicators
        2. Engagement Metrics
        3. Socioeconomic Composites
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Dataframe with engineered features
        """
        print("\n" + "="*90)
        print(" FEATURE ENGINEERING (Journal Methodology)")
        print("="*90)
        
        df_eng = df.copy()
        features_created = 0
        
        # === ACADEMIC PERFORMANCE INDICATORS ===
        print("\n📚 Academic Performance Indicators:")
        
        # 1. Total units enrolled
        if 'Curricular units 1st sem (enrolled)' in df.columns and 'Curricular units 2nd sem (enrolled)' in df.columns:
            df_eng['total_units_enrolled'] = (
                df['Curricular units 1st sem (enrolled)'] + 
                df['Curricular units 2nd sem (enrolled)']
            )
            features_created += 1
            print("  ✓ total_units_enrolled")
        
        # 2. Total units approved
        if 'Curricular units 1st sem (approved)' in df.columns and 'Curricular units 2nd sem (approved)' in df.columns:
            df_eng['total_units_approved'] = (
                df['Curricular units 1st sem (approved)'] + 
                df['Curricular units 2nd sem (approved)']
            )
            features_created += 1
            print("  ✓ total_units_approved")
        
        # 3. Success rate (approval rate)
        if 'total_units_approved' in df_eng.columns and 'total_units_enrolled' in df_eng.columns:
            df_eng['success_rate'] = np.where(
                df_eng['total_units_enrolled'] > 0,
                df_eng['total_units_approved'] / df_eng['total_units_enrolled'],
                0
            )
            features_created += 1
            print("  ✓ success_rate")
        
        # 4. Semester consistency (grade variance)
        if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
            df_eng['semester_consistency'] = abs(
                df['Curricular units 1st sem (grade)'] - 
                df['Curricular units 2nd sem (grade)']
            )
            features_created += 1
            print("  ✓ semester_consistency")
        
        # 5. Average grade (both semesters)
        if 'Curricular units 1st sem (grade)' in df.columns and 'Curricular units 2nd sem (grade)' in df.columns:
            df_eng['average_grade'] = (
                df['Curricular units 1st sem (grade)'] + 
                df['Curricular units 2nd sem (grade)']
            ) / 2
            features_created += 1
            print("  ✓ average_grade")
        
        # 6. Academic progression (improvement)
        if 'Curricular units 1st sem (approved)' in df.columns and 'Curricular units 2nd sem (approved)' in df.columns:
            df_eng['academic_progression'] = (
                df['Curricular units 2nd sem (approved)'] - 
                df['Curricular units 1st sem (approved)']
            )
            features_created += 1
            print("  ✓ academic_progression")
        
        # === ENGAGEMENT METRICS ===
        print("\n🎯 Engagement Metrics:")
        
        # 7. Total units without evaluation
        if 'Curricular units 1st sem (without evaluations)' in df.columns and 'Curricular units 2nd sem (without evaluations)' in df.columns:
            df_eng['total_units_no_eval'] = (
                df['Curricular units 1st sem (without evaluations)'] + 
                df['Curricular units 2nd sem (without evaluations)']
            )
            features_created += 1
            print("  ✓ total_units_no_eval")
        
        # 8. Engagement index
        if 'total_units_no_eval' in df_eng.columns and 'total_units_enrolled' in df_eng.columns:
            df_eng['engagement_index'] = np.where(
                df_eng['total_units_enrolled'] > 0,
                1 - (df_eng['total_units_no_eval'] / df_eng['total_units_enrolled']),
                1
            )
            features_created += 1
            print("  ✓ engagement_index")
        
        # 9. Total evaluations
        if 'Curricular units 1st sem (evaluations)' in df.columns and 'Curricular units 2nd sem (evaluations)' in df.columns:
            df_eng['total_evaluations'] = (
                df['Curricular units 1st sem (evaluations)'] + 
                df['Curricular units 2nd sem (evaluations)']
            )
            features_created += 1
            print("  ✓ total_evaluations")
        
        # 10. Evaluation completion rate
        if 'total_evaluations' in df_eng.columns and 'total_units_enrolled' in df_eng.columns:
            df_eng['evaluation_completion_rate'] = np.where(
                df_eng['total_units_enrolled'] > 0,
                df_eng['total_evaluations'] / (df_eng['total_units_enrolled'] * 2),  # Assume 2 evals per unit
                0
            )
            features_created += 1
            print("  ✓ evaluation_completion_rate")
        
        # === SOCIOECONOMIC COMPOSITES ===
        print("\n💰 Socioeconomic Indicators:")
        
        # 11. Parental education level (average)
        if "Mother's qualification" in df.columns and "Father's qualification" in df.columns:
            df_eng['parental_education_level'] = (
                df["Mother's qualification"] + 
                df["Father's qualification"]
            ) / 2
            features_created += 1
            print("  ✓ parental_education_level")
        
        # 12. Financial support indicator
        if 'Scholarship holder' in df.columns and 'Tuition fees up to date' in df.columns:
            df_eng['financial_support'] = (
                df['Scholarship holder'] + 
                df['Tuition fees up to date']
            ) / 2
            features_created += 1
            print("  ✓ financial_support")
        
        print(f"\n✓ Feature engineering complete: {features_created} new features created")
        print(f"  Total features: {len(df_eng.columns)}")
        
        return df_eng
    
    def prepare_data(self, test_size_1=0.15, test_size_2=0.15):
        """
        Complete preprocessing pipeline following journal methodology
        
        Steps:
        1. Load and explore data
        2. Engineer features
        3. Encode target variables (3-class and binary)
        4. Select numerical features
        5. Stratified train-val-test split (70-15-15)
        6. Z-score normalization
        
        Args:
            test_size_1 (float): Test set proportion (default: 0.15)
            test_size_2 (float): Validation set proportion from remaining (default: ~0.15)
        
        Returns:
            tuple: (X_train, X_val, X_test, y_target_train, y_target_val, y_target_test,
                   y_dropout_train, y_dropout_val, y_dropout_test, feature_names)
        """
        print("\n" + "="*90)
        print(" DATA PREPARATION PIPELINE (Journal Methodology)")
        print("="*90)
        
        # Step 1: Load data
        if self.original_data is None:
            self.load_data()
        
        df = self.original_data.copy()
        
        # Step 2: Feature engineering
        df_engineered = self.engineer_features(df)
        
        # Step 3: Encode targets
        print("\n" + "="*90)
        print(" TARGET ENCODING")
        print("="*90)
        
        # Binary dropout target
        y_dropout = (df_engineered['Target'] == 'Dropout').astype(int)
        print(f"\n✓ Binary dropout encoded:")
        print(f"  Not Dropout (0): {(y_dropout == 0).sum()} ({(y_dropout == 0).sum()/len(y_dropout)*100:.1f}%)")
        print(f"  Dropout (1): {(y_dropout == 1).sum()} ({(y_dropout == 1).sum()/len(y_dropout)*100:.1f}%)")
        
        # 3-class target
        y_target = self.label_encoder.fit_transform(df_engineered['Target'])
        print(f"\n✓ 3-class target encoded:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = (y_target == i).sum()
            print(f"  {class_name} ({i}): {count} ({count/len(y_target)*100:.1f}%)")
        
        # Step 4: Select features
        print("\n" + "="*90)
        print(" FEATURE SELECTION")
        print("="*90)
        
        # Remove non-predictive columns
        columns_to_remove = ['Target']
        
        # Select numerical features only for initial model
        numerical_cols = df_engineered.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [col for col in numerical_cols if col not in columns_to_remove]
        
        X = df_engineered[feature_cols].copy()
        
        # Handle any potential NaN values
        if X.isnull().sum().sum() > 0:
            print(f"⚠ Filling {X.isnull().sum().sum()} missing values with 0")
            X = X.fillna(0)
        
        self.feature_names = feature_cols
        print(f"\n✓ Feature selection complete:")
        print(f"  Selected features: {len(feature_cols)}")
        print(f"  Feature types: All numerical")
        
        # Step 5: Stratified split (70-15-15)
        print("\n" + "="*90)
        print(" STRATIFIED TRAIN-VALIDATION-TEST SPLIT")
        print("="*90)
        
        # First split: 85% (train+val) and 15% (test)
        X_temp, X_test, y_target_temp, y_target_test, y_dropout_temp, y_dropout_test = train_test_split(
            X, y_target, y_dropout,
            test_size=test_size_1,
            random_state=self.random_state,
            stratify=y_target
        )
        
        # Second split: split the 85% into 70% (train) and 15% (val)
        val_size_adjusted = test_size_2 / (1 - test_size_1)  # 0.15 / 0.85 ≈ 0.1765
        X_train, X_val, y_target_train, y_target_val, y_dropout_train, y_dropout_val = train_test_split(
            X_temp, y_target_temp, y_dropout_temp,
            test_size=val_size_adjusted,
            random_state=self.random_state,
            stratify=y_target_temp
        )
        
        print(f"\n✓ Stratified split complete:")
        print(f"  Training set:   {len(X_train):5d} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"  Validation set: {len(X_val):5d} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"  Test set:       {len(X_test):5d} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # Verify stratification
        print(f"\n  Target distribution in training set:")
        for i, class_name in enumerate(self.label_encoder.classes_):
            count = (y_target_train == i).sum()
            print(f"    {class_name} ({i}): {count} ({count/len(y_target_train)*100:.1f}%)")
        
        # Step 6: Z-score normalization
        print("\n" + "="*90)
        print(" Z-SCORE NORMALIZATION")
        print("="*90)
        
        # Fit on training data only (prevent data leakage)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"\n✓ Normalization complete:")
        print(f"  Method: Z-score standardization (μ=0, σ=1)")
        print(f"  Fitted on: Training set only")
        print(f"  Applied to: Train, Validation, Test")
        
        # Display sample statistics
        print(f"\n  Sample statistics (after normalization):")
        print(f"    Training set   - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
        print(f"    Validation set - Mean: {X_val_scaled.mean():.6f}, Std: {X_val_scaled.std():.6f}")
        print(f"    Test set       - Mean: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")
        
        print("\n" + "="*90)
        print(" ✓ DATA PREPARATION COMPLETE")
        print("="*90)
        
        return (X_train_scaled, X_val_scaled, X_test_scaled,
                y_target_train, y_target_val, y_target_test,
                y_dropout_train, y_dropout_val, y_dropout_test,
                self.feature_names)
    
    def get_target_labels(self):
        """Get target class labels"""
        if hasattr(self.label_encoder, 'classes_'):
            return self.label_encoder.classes_.tolist()
        return ['Dropout', 'Enrolled', 'Graduate']
    
    def inverse_transform_target(self, y_encoded):
        """Convert encoded targets back to original labels"""
        return self.label_encoder.inverse_transform(y_encoded)
