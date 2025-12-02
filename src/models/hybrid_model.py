"""
Hybrid Multi-Task Learning Model
Combines performance and dropout prediction in a single model with shared layers
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import os


class HybridModel:
    """
    Multi-task learning model for both grade prediction and dropout prediction
    Shared feature extraction with two prediction heads
    """
    
    def __init__(self, input_dim, num_grade_classes=9, random_state=42):
        """
        Initialize the hybrid model
        
        Args:
            input_dim (int): Number of input features
            num_grade_classes (int): Number of grade classes
            random_state (int): Random seed
        """
        self.input_dim = input_dim
        self.num_grade_classes = num_grade_classes
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        print(f"Initializing HybridModel: {input_dim} features → " + 
              f"Grade ({num_grade_classes} classes) + Dropout (binary)")
        
    def build_model(self, grade_weight=0.5, dropout_weight=0.5):
        """
        Build the multi-task learning architecture
        
        Args:
            grade_weight: Weight for grade prediction loss
            dropout_weight: Weight for dropout prediction loss
        """
        
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # Shared layers for feature extraction
        shared = layers.Dense(128, activation='relu', name='shared_1')(inputs)
        shared = layers.BatchNormalization(name='shared_bn_1')(shared)
        shared = layers.Dropout(0.3, name='shared_dropout_1')(shared)
        
        shared = layers.Dense(64, activation='relu', name='shared_2')(shared)
        shared = layers.BatchNormalization(name='shared_bn_2')(shared)
        shared = layers.Dropout(0.2, name='shared_dropout_2')(shared)
        
        # Grade prediction head
        grade_branch = layers.Dense(32, activation='relu', name='grade_dense_1')(shared)
        grade_branch = layers.Dropout(0.1, name='grade_dropout')(grade_branch)
        grade_output = layers.Dense(
            self.num_grade_classes, 
            activation='softmax', 
            name='grade_output'
        )(grade_branch)
        
        # Dropout prediction head
        dropout_branch = layers.Dense(16, activation='relu', name='dropout_dense_1')(shared)
        dropout_output = layers.Dense(
            1, 
            activation='sigmoid', 
            name='dropout_output'
        )(dropout_branch)
        
        # Create model with two outputs
        model = models.Model(
            inputs=inputs,
            outputs=[grade_output, dropout_output]
        )
        
        # Compile with weighted losses
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'grade_output': 'categorical_crossentropy',
                'dropout_output': 'binary_crossentropy'
            },
            loss_weights={
                'grade_output': grade_weight,
                'dropout_output': dropout_weight
            },
            metrics={
                'grade_output': ['accuracy'],
                'dropout_output': ['accuracy', keras.metrics.AUC(name='auc')]
            }
        )
        
        self.model = model
        
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE: Hybrid Multi-Task Model")
        print("="*80)
        self.model.summary()
        
        return model
    
    def train(self, X_train, y_grade_train, y_dropout_train, 
              X_val=None, y_grade_val=None, y_dropout_val=None,
              epochs=100, batch_size=16, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_grade_train: Training grade labels
            y_dropout_train: Training dropout labels
            X_val: Validation features
            y_grade_val: Validation grade labels
            y_dropout_val: Validation dropout labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode
            
        Returns:
            History object
        """
        if self.model is None:
            self.build_model()
        
        # Convert grade labels to categorical
        y_grade_train_cat = to_categorical(y_grade_train, num_classes=self.num_grade_classes)
        
        validation_data = None
        if X_val is not None and y_grade_val is not None and y_dropout_val is not None:
            y_grade_val_cat = to_categorical(y_grade_val, num_classes=self.num_grade_classes)
            validation_data = (
                X_val, 
                {'grade_output': y_grade_val_cat, 'dropout_output': y_dropout_val}
            )
        
        # Define callbacks
        callback_list = [
            callbacks.EarlyStopping(
                monitor='val_loss' if validation_data else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss' if validation_data else 'loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        print("\n" + "="*80)
        print("TRAINING HYBRID MULTI-TASK MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("="*80 + "\n")
        
        # Train model
        self.history = self.model.fit(
            X_train,
            {'grade_output': y_grade_train_cat, 'dropout_output': y_dropout_train},
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("\n✓ Training completed!")
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
            
        Returns:
            tuple: (predicted_grades, predicted_dropout)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        grade_probs, dropout_probs = self.model.predict(X, verbose=0)
        
        predicted_grades = np.argmax(grade_probs, axis=1)
        predicted_dropout = (dropout_probs > 0.5).astype(int).flatten()
        
        return predicted_grades, predicted_dropout
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            tuple: (grade_probabilities, dropout_probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        grade_probs, dropout_probs = self.model.predict(X, verbose=0)
        
        return grade_probs, dropout_probs.flatten()
    
    def evaluate(self, X_test, y_grade_test, y_dropout_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_grade_test: Test grade labels
            y_dropout_test: Test dropout labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert grade labels to categorical
        y_grade_test_cat = to_categorical(y_grade_test, num_classes=self.num_grade_classes)
        
        # Evaluate
        results = self.model.evaluate(
            X_test,
            {'grade_output': y_grade_test_cat, 'dropout_output': y_dropout_test},
            verbose=0
        )
        
        # Parse results
        metrics = {
            'total_loss': results[0],
            'grade_loss': results[1],
            'dropout_loss': results[2],
            'grade_accuracy': results[3],
            'dropout_accuracy': results[4],
            'dropout_auc': results[5]
        }
        
        print("\n" + "="*80)
        print("HYBRID MODEL EVALUATION")
        print("="*80)
        print("\nGrade Prediction:")
        print(f"  Loss: {metrics['grade_loss']:.4f}")
        print(f"  Accuracy: {metrics['grade_accuracy']:.4f}")
        print("\nDropout Prediction:")
        print(f"  Loss: {metrics['dropout_loss']:.4f}")
        print(f"  Accuracy: {metrics['dropout_accuracy']:.4f}")
        print(f"  AUC: {metrics['dropout_auc']:.4f}")
        print("\nOverall:")
        print(f"  Total Loss: {metrics['total_loss']:.4f}")
        print("="*80)
        
        return metrics
    
    def save(self, filepath):
        """Save model to file"""
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"✓ Model saved to: {filepath}")
    
    def load(self, filepath):
        """Load model from file"""
        self.model = keras.models.load_model(filepath)
        print(f"✓ Model loaded from: {filepath}")
        return self.model
    
    def get_grade_label(self, class_index):
        """Convert class index to grade label"""
        grade_mapping = {
            0: 'D+', 1: 'C', 2: 'C+', 3: 'B-', 4: 'B', 
            5: 'B+', 6: 'A-', 7: 'A', 8: 'A+'
        }
        return grade_mapping.get(class_index, 'Unknown')
