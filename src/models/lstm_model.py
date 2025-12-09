#!/usr/bin/env python3
"""
LSTM Model for Bitcoin Price Prediction
Implements a deep learning LSTM model for predicting price movements
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model, Model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l1_l2
    TENSORFLOW_AVAILABLE = True
except ImportError:
    logging.warning("TensorFlow not available. LSTM model will be disabled.")
    TENSORFLOW_AVAILABLE = False

import os
import joblib
from datetime import datetime

# Configure logging
logger = logging.getLogger("lstm_model")

class LSTMModel:
    """LSTM Deep Learning Model for time series prediction"""
    
    def __init__(self, 
                config: Dict[str, Any],
                sequence_length: int = 60,
                n_features: int = 0,
                model_path: Optional[str] = None):
        """
        Initialize the LSTM model
        
        Args:
            config (Dict[str, Any]): Configuration parameters
            sequence_length (int): Length of input sequences
            n_features (int): Number of features in the input data
            model_path (str, optional): Path to load a pre-trained model
        """
        self.config = config
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model_path = model_path
        self.model = None
        self.feature_scaler = None
        self.history = None
        
        # Load model if path provided, otherwise create new model
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif self.n_features > 0:
            self.build_model()
            
        logger.info(f"Initialized LSTM model with {n_features} features and sequence length {sequence_length}")
    
    def build_model(self) -> None:
        """Build the LSTM model architecture"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping model build.")
            return

        try:
            # Reset Keras session
            tf.keras.backend.clear_session()
            
            # Get hyperparameters from config
            lstm_units = self.config.get('lstm_units', [128, 64])
            dropout_rate = self.config.get('dropout_rate', 0.2)
            learning_rate = self.config.get('learning_rate', 0.001)
            l1_reg = self.config.get('l1_reg', 0.0001)
            l2_reg = self.config.get('l2_reg', 0.0001)
            bidirectional = self.config.get('bidirectional', True)
            
            # Define model
            self.model = Sequential()
            
            # First LSTM layer
            if bidirectional:
                self.model.add(Bidirectional(
                    LSTM(lstm_units[0], 
                         return_sequences=(len(lstm_units) > 1),
                         kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)),
                    input_shape=(self.sequence_length, self.n_features)
                ))
            else:
                self.model.add(LSTM(
                    lstm_units[0],
                    return_sequences=(len(lstm_units) > 1),
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg),
                    input_shape=(self.sequence_length, self.n_features)
                ))
            
            self.model.add(Dropout(dropout_rate))
            self.model.add(BatchNormalization())
            
            # Additional LSTM layers
            for i in range(1, len(lstm_units)):
                is_last = i == len(lstm_units) - 1
                if bidirectional:
                    self.model.add(Bidirectional(
                        LSTM(lstm_units[i], 
                             return_sequences=not is_last,
                             kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg))
                    ))
                else:
                    self.model.add(LSTM(
                        lstm_units[i],
                        return_sequences=not is_last,
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)
                    ))
                self.model.add(Dropout(dropout_rate))
                self.model.add(BatchNormalization())
            
            # Output layer - binary classification for price direction
            self.model.add(Dense(1, activation='sigmoid'))
            
            # Compile model
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Summary
            self.model.summary()
            logger.info("LSTM model built successfully")
            
        except Exception as e:
            logger.error(f"Error building LSTM model: {e}")
            raise
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None, 
             batch_size: int = 32, 
             epochs: int = 100,
             save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            batch_size: Batch size for training
            epochs: Number of epochs to train
            save_path: Path to save the trained model
            
        Returns:
            Dict[str, Any]: Training history
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Skipping training.")
                return {}

            if self.model is None:
                if self.n_features == 0:
                    self.n_features = X_train.shape[2]
                self.build_model()
            
            # Set up callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss' if X_val is not None else 'loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss' if X_val is not None else 'loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Add model checkpoint if save_path is provided
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                callbacks.append(ModelCheckpoint(
                    filepath=save_path,
                    monitor='val_loss' if X_val is not None else 'loss',
                    save_best_only=True
                ))
            
            # Train model
            validation_data = (X_val, y_val) if X_val is not None and y_val is not None else None
            
            self.history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=validation_data,
                callbacks=callbacks,
                verbose=1
            ).history
            
            if save_path:
                self.save(save_path)
                
            logger.info(f"Model training completed. Final training accuracy: {self.history['accuracy'][-1]:.4f}")
            if validation_data:
                logger.info(f"Validation accuracy: {self.history['val_accuracy'][-1]:.4f}")
                
            return self.history
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {e}")
            raise
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions with the trained model
        
        Args:
            X: Input features
            
        Returns:
            np.ndarray: Predictions
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Returning zeros.")
                return np.zeros((len(X), 1))

            if self.model is None:
                raise ValueError("Model not initialized. Call build_model() or load() first.")
                
            return self.model.predict(X)
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save(self, path: str) -> None:
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Skipping save.")
                return

            if self.model is None:
                raise ValueError("No model to save.")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            
            # Save metadata
            metadata = {
                'sequence_length': self.sequence_length,
                'n_features': self.n_features,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.joblib"
            joblib.dump(metadata, metadata_path)
            
            logger.info(f"Model saved to {path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load(self, path: str) -> None:
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Skipping load.")
                return

            # Load the model
            self.model = load_model(path)
            
            # Load metadata if available
            metadata_path = f"{os.path.splitext(path)[0]}_metadata.joblib"
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.sequence_length = metadata.get('sequence_length', self.sequence_length)
                self.n_features = metadata.get('n_features', self.n_features)
                self.config = metadata.get('config', self.config)
                
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model on test data
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                logger.warning("TensorFlow not available. Skipping evaluation.")
                return {}

            if self.model is None:
                raise ValueError("Model not initialized. Call build_model() or load() first.")
                
            # Get predictions
            y_pred_prob = self.predict(X_test)
            y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
            
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_prob)
            }
            
            logger.info(f"Model evaluation results: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            raise