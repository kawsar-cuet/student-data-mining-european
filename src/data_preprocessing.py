"""
Data Preprocessing Module
Handles data loading, cleaning, feature engineering, and preparation for modeling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for student performance data
    """
    
    def __init__(self, data_path, random_state=42):
        """
        Initialize the preprocessor
        
        Args:
            data_path (str): Path to the CSV dataset
            random_state (int): Random seed for reproducibility
        """
        self.data_path = data_path
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.original_data = None
        self.processed_data = None
        
        print(f"Initializing DataPreprocessor with data from: {data_path}")
        
    def load_data(self):
        """Load data from CSV file"""
        try:
            self.original_data = pd.read_csv(self.data_path)
            print(f"✓ Data loaded successfully: {self.original_data.shape[0]} rows, {self.original_data.shape[1]} columns")
            return self.original_data
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            raise
    
    def explore_data(self):
        """Display basic information about the dataset"""
        if self.original_data is None:
            self.load_data()
        
        print("\n" + "="*80)
        print("DATA EXPLORATION")
        print("="*80)
        
        print("\nDataset Shape:", self.original_data.shape)
        print("\nColumn Names and Types:")
        print(self.original_data.dtypes)
        
        print("\nMissing Values:")
        missing = self.original_data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values detected")
        
        print("\nBasic Statistics:")
        print(self.original_data.describe())
        
        print("\nTarget Variable Distribution:")
        print("\nDropout Status:")
        print(self.original_data['dropout_status'].value_counts())
        print("\nFinal Grades:")
        print(self.original_data['final_grade'].value_counts().sort_index())
        
        return self.original_data.describe()
    
    def clean_data(self, df):
        """
        Clean the dataset
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Cleaned dataframe
        """
        print("\n" + "="*80)
        print("DATA CLEANING")
        print("="*80)
        
        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates()
        print(f"✓ Removed {initial_rows - len(df)} duplicate rows")
        
        # Handle missing values (if any)
        # For this dataset, we'll use simple imputation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Impute numeric columns with median
        if df[numeric_cols].isnull().sum().sum() > 0:
            imputer_num = SimpleImputer(strategy='median')
            df[numeric_cols] = imputer_num.fit_transform(df[numeric_cols])
            print(f"✓ Imputed missing numeric values")
        
        # Impute categorical columns with mode
        if df[categorical_cols].isnull().sum().sum() > 0:
            imputer_cat = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])
            print(f"✓ Imputed missing categorical values")
        
        print(f"✓ Data cleaning complete")
        
        return df
    
    def engineer_features(self, df):
        """
        Create derived features from existing ones
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Dataframe with additional features
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        # Academic consistency
        df['academic_consistency'] = abs(df['cgpa'] - df['previous_semester_cgpa'])
        print("✓ Created 'academic_consistency' feature")
        
        # Workload stress ratio
        df['workload_stress'] = df['study_hours_per_week'] / (df['sleep_hours'] + 0.1)  # avoid division by zero
        print("✓ Created 'workload_stress' feature")
        
        # Resource access score
        df['resource_access'] = (df['internet_access'].map({'Yes': 1, 'No': 0}) + 
                                 df['scholarship'].map({'Yes': 1, 'No': 0})) / 2
        print("✓ Created 'resource_access' feature")
        
        # Academic performance score (composite)
        df['academic_score'] = (
            0.3 * df['cgpa'] + 
            0.2 * (df['midterm_score'] / 100) * 4 +  # normalize to 4.0 scale
            0.2 * (df['quiz_average'] / 100) * 4 +
            0.2 * (df['assignment_submission_rate'] / 100) * 4 +
            0.1 * (df['participation_score'] / 100) * 4
        )
        print("✓ Created 'academic_score' feature")
        
        # Engagement level
        df['engagement_level'] = (
            0.4 * (df['attendance_rate'] / 100) +
            0.3 * (df['library_visits_per_month'] / 20) +  # normalize
            0.3 * (df['extracurricular_activities'] / 5)  # normalize
        )
        print("✓ Created 'engagement_level' feature")
        
        # Risk indicators
        df['high_stress'] = ((df['stress_level'] == 'High') & 
                            (df['sleep_hours'] < 6)).astype(int)
        print("✓ Created 'high_stress' feature")
        
        df['low_motivation'] = (df['motivation_level'] == 'Low').astype(int)
        print("✓ Created 'low_motivation' feature")
        
        print(f"✓ Feature engineering complete: {len(df.columns)} total features")
        
        return df
    
    def encode_features(self, df):
        """
        Encode categorical variables
        
        Args:
            df (DataFrame): Input dataframe
            
        Returns:
            DataFrame: Encoded dataframe
        """
        print("\n" + "="*80)
        print("FEATURE ENCODING")
        print("="*80)
        
        # Binary encoding for Yes/No columns
        binary_cols = ['part_time_job', 'internet_access', 'scholarship', 'health_issues']
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
                print(f"✓ Encoded '{col}' (binary)")
        
        # Label encoding for ordinal features
        ordinal_cols = {
            'parents_education': {'SSC': 0, 'HSC': 1, 'Bachelor': 2, 'Master': 3},
            'stress_level': {'Low': 0, 'Medium': 1, 'High': 2},
            'motivation_level': {'Low': 0, 'Medium': 1, 'High': 2},
            'final_grade': {'D+': 0, 'C': 1, 'C+': 2, 'B-': 3, 'B': 4, 'B+': 5, 'A-': 6, 'A': 7, 'A+': 8}
        }
        
        for col, mapping in ordinal_cols.items():
            if col in df.columns:
                df[col] = df[col].map(mapping)
                print(f"✓ Encoded '{col}' (ordinal)")
        
        # One-hot encoding for nominal features
        nominal_cols = ['gender', 'department']
        df = pd.get_dummies(df, columns=nominal_cols, prefix=nominal_cols, drop_first=True)
        print(f"✓ One-hot encoded nominal features: {nominal_cols}")
        
        # Encode dropout status
        df['dropout_status'] = df['dropout_status'].map({'Yes': 1, 'No': 0})
        print(f"✓ Encoded 'dropout_status' (binary)")
        
        print(f"✓ Encoding complete: {len(df.columns)} features after encoding")
        
        return df
    
    def prepare_data(self, test_size=0.15, val_size=0.15):
        """
        Complete data preparation pipeline
        
        Args:
            test_size (float): Proportion for test set
            val_size (float): Proportion for validation set
            
        Returns:
            tuple: X_train, X_val, X_test, y_train_grade, y_val_grade, y_test_grade,
                   y_train_dropout, y_val_dropout, y_test_dropout, feature_names
        """
        print("\n" + "="*80)
        print("COMPLETE DATA PREPARATION PIPELINE")
        print("="*80)
        
        # Load data
        if self.original_data is None:
            self.load_data()
        
        df = self.original_data.copy()
        
        # Pipeline steps
        df = self.clean_data(df)
        df = self.engineer_features(df)
        
        # Store student info before dropping
        student_info = df[['student_id', 'name']].copy()
        
        # Drop non-feature columns
        drop_cols = ['student_id', 'name']
        df = df.drop(columns=drop_cols)
        
        # Encode features
        df = self.encode_features(df)
        
        # Separate features and targets
        X = df.drop(columns=['final_grade', 'dropout_status'])
        y_grade = df['final_grade']
        y_dropout = df['dropout_status']
        
        feature_names = X.columns.tolist()
        
        # Split data: first into train+val and test
        X_temp, X_test, y_grade_temp, y_grade_test, y_dropout_temp, y_dropout_test = train_test_split(
            X, y_grade, y_dropout, test_size=test_size, random_state=self.random_state, stratify=y_dropout
        )
        
        # Then split train+val into train and val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_grade_train, y_grade_val, y_dropout_train, y_dropout_val = train_test_split(
            X_temp, y_grade_temp, y_dropout_temp, test_size=val_ratio, 
            random_state=self.random_state, stratify=y_dropout_temp
        )
        
        # Normalize features
        X_train = pd.DataFrame(
            self.scaler.fit_transform(X_train), 
            columns=feature_names,
            index=X_train.index
        )
        X_val = pd.DataFrame(
            self.scaler.transform(X_val), 
            columns=feature_names,
            index=X_val.index
        )
        X_test = pd.DataFrame(
            self.scaler.transform(X_test), 
            columns=feature_names,
            index=X_test.index
        )
        
        print("\n" + "="*80)
        print("DATA SPLIT SUMMARY")
        print("="*80)
        print(f"Training set:   {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)")
        print(f"Test set:       {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        print(f"Total features: {len(feature_names)}")
        
        print("\nDropout distribution in training set:")
        print(y_dropout_train.value_counts())
        
        # Store processed data
        self.processed_data = {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_grade_train': y_grade_train, 'y_grade_val': y_grade_val, 'y_grade_test': y_grade_test,
            'y_dropout_train': y_dropout_train, 'y_dropout_val': y_dropout_val, 'y_dropout_test': y_dropout_test,
            'feature_names': feature_names,
            'student_info': student_info
        }
        
        return (X_train, X_val, X_test, 
                y_grade_train, y_grade_val, y_grade_test,
                y_dropout_train, y_dropout_val, y_dropout_test,
                feature_names)
    
    def get_feature_importance(self, importance_values, top_n=15):
        """
        Display feature importance
        
        Args:
            importance_values (array): Feature importance values
            top_n (int): Number of top features to display
        """
        if self.processed_data is None:
            print("Please run prepare_data() first")
            return
        
        feature_names = self.processed_data['feature_names']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importance_values
        }).sort_values('Importance', ascending=False)
        
        print(f"\nTop {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor('data/ulab_students_dataset.csv')
    preprocessor.explore_data()
    
    X_train, X_val, X_test, y_grade_train, y_grade_val, y_grade_test, \
    y_dropout_train, y_dropout_val, y_dropout_test, feature_names = preprocessor.prepare_data()
    
    print("\n✓ Data preprocessing completed successfully!")
