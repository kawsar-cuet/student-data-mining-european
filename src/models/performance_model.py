"""
Performance Prediction Model
Deep Neural Network for predicting student final grades (multi-class classification)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import os


class PerformanceModel:
    """
    Deep Neural Network for student grade prediction
    Output: Multi-class classification (9 grades: D+ to A+)
    """
    
    def __init__(self, input_dim, num_classes=9, random_state=42):
        """
        Initialize the performance prediction model
        
        Args:
            input_dim (int): Number of input features
            num_classes (int): Number of grade classes (default: 9)
            random_state (int): Random seed
        """
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        print(f"Initializing PerformanceModel: {input_dim} features → {num_classes} classes")
        
    def build_model(self):
        """Build the deep neural network architecture"""
        
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(self.input_dim,)),
            
            # First hidden layer
            layers.Dense(128, activation='relu', name='dense_1'),
            layers.BatchNormalization(name='bn_1'),
            layers.Dropout(0.3, name='dropout_1'),
            
            # Second hidden layer
            layers.Dense(64, activation='relu', name='dense_2'),
            layers.BatchNormalization(name='bn_2'),
            layers.Dropout(0.2, name='dropout_2'),
            
            # Third hidden layer
            layers.Dense(32, activation='relu', name='dense_3'),
            layers.Dropout(0.1, name='dropout_3'),
            
            # Output layer
            layers.Dense(self.num_classes, activation='softmax', name='output')
        ])
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE: Performance Prediction")
        print("="*80)
        self.model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=16, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (integer class labels)
            X_val: Validation features
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            verbose: Verbosity mode
            
        Returns:
            History object
        """
        if self.model is None:
            self.build_model()
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        
        validation_data = None
        if X_val is not None and y_val is not None:
            y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
            validation_data = (X_val, y_val_cat)
        
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
        print("TRAINING PERFORMANCE MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("="*80 + "\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train_cat,
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
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        
        return predicted_classes
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Prediction probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Convert labels to categorical
        y_test_cat = to_categorical(y_test, num_classes=self.num_classes)
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test_cat, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'precision': results[2],
            'recall': results[3]
        }
        
        # Calculate F1-score
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        print("\n" + "="*80)
        print("PERFORMANCE MODEL EVALUATION")
        print("="*80)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
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
