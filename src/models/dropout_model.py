"""
Dropout Prediction Model
Deep Neural Network with Attention for predicting student dropout risk (binary classification)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
import os


class AttentionLayer(layers.Layer):
    """Custom attention layer for feature importance"""
    
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
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
    
    def call(self, x):
        # Attention mechanism
        e = keras.backend.tanh(keras.backend.dot(x, self.W) + self.b)
        a = keras.backend.softmax(e, axis=-1)
        output = x * a
        return output


class DropoutModel:
    """
    Deep Neural Network with Attention for dropout prediction
    Output: Binary classification (dropout vs not dropout)
    """
    
    def __init__(self, input_dim, random_state=42):
        """
        Initialize the dropout prediction model
        
        Args:
            input_dim (int): Number of input features
            random_state (int): Random seed
        """
        self.input_dim = input_dim
        self.random_state = random_state
        self.model = None
        self.history = None
        
        # Set random seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        print(f"Initializing DropoutModel: {input_dim} features → Binary classification")
        
    def build_model(self):
        """Build the deep neural network architecture with attention"""
        
        # Input layer
        inputs = layers.Input(shape=(self.input_dim,), name='input')
        
        # First hidden layer
        x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
        x = layers.BatchNormalization(name='bn_1')(x)
        x = layers.Dropout(0.3, name='dropout_1')(x)
        
        # Attention layer
        x = AttentionLayer(name='attention')(x)
        
        # Second hidden layer
        x = layers.Dense(32, activation='relu', name='dense_2')(x)
        x = layers.Dropout(0.2, name='dropout_2')(x)
        
        # Third hidden layer
        x = layers.Dense(16, activation='relu', name='dense_3')(x)
        
        # Output layer
        outputs = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy',
                    keras.metrics.AUC(name='auc'),
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall')]
        )
        
        self.model = model
        
        print("\n" + "="*80)
        print("MODEL ARCHITECTURE: Dropout Prediction (with Attention)")
        print("="*80)
        self.model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=100, batch_size=16, verbose=1):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels (0 or 1)
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
        
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        
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
        print("TRAINING DROPOUT PREDICTION MODEL")
        print("="*80)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val) if X_val is not None else 0}")
        print(f"Epochs: {epochs}, Batch size: {batch_size}")
        print("="*80 + "\n")
        
        # Train model
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callback_list,
            verbose=verbose
        )
        
        print("\n✓ Training completed!")
        
        return self.history
    
    def predict(self, X, threshold=0.5):
        """
        Make predictions
        
        Args:
            X: Input features
            threshold: Classification threshold (default: 0.5)
            
        Returns:
            Predicted binary labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X, verbose=0)
        predicted_classes = (predictions > threshold).astype(int).flatten()
        
        return predicted_classes
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
            
        Returns:
            Dropout probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X, verbose=0).flatten()
    
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
        
        # Evaluate
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2],
            'precision': results[3],
            'recall': results[4]
        }
        
        # Calculate F1-score
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                 (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0.0
        
        print("\n" + "="*80)
        print("DROPOUT MODEL EVALUATION")
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
        self.model = keras.models.load_model(
            filepath,
            custom_objects={'AttentionLayer': AttentionLayer}
        )
        print(f"✓ Model loaded from: {filepath}")
        return self.model
    
    def get_risk_level(self, probability):
        """Convert dropout probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Medium Risk"
        else:
            return "High Risk"
