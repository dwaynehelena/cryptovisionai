#!/usr/bin/env python3
"""
Transformer Model for Bitcoin Price Prediction
Implements a transformer architecture for predicting price movements
"""

import numpy as np
import logging
from typing import Tuple, Dict, List, Any, Optional
import joblib
from datetime import datetime
import os

try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input, Dense, MultiHeadAttention, LayerNormalization,
        Dropout, GlobalAveragePooling1D, Add
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
    Layer = tf.keras.layers.Layer
except ImportError:
    logging.warning("TensorFlow not available. Transformer model will be disabled.")
    TENSORFLOW_AVAILABLE = False
    Layer = object
    # Define dummy placeholders to prevent NameError in class definitions
    Input = Dense = MultiHeadAttention = LayerNormalization = Dropout = GlobalAveragePooling1D = Add = object
    Model = object
    Adam = object
    EarlyStopping = ModelCheckpoint = ReduceLROnPlateau = object

# Configure logging
logger = logging.getLogger("transformer_model")

class TransformerBlock(Layer):
    """Transformer block with multi-head attention and feed-forward network"""
    
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
        
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        })
        return config

class PositionalEncoding(Layer):
    """Positional encoding layer for transformer models"""
    
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "position": self.position,
            "d_model": self.d_model,
            "pos_encoding": self.pos_encoding,
        })
        return config
        
    def get_angles(self, position, i, d_model):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles
        
    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        
        # Apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # Apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        
        pos_encoding = angle_rads[np.newaxis, ...]
        
        return tf.cast(pos_encoding, dtype=tf.float32)
        
    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

class TransformerModel:
    """
    Transformer-based model for time series prediction
    Using multi-head attention to capture temporal dependencies
    """
    
    def __init__(self, 
                config: Dict[str, Any],
                sequence_length: int = 60,
                n_features: int = 0,
                model_path: Optional[str] = None):
        """
        Initialize the Transformer model
        
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
        self.history = None
        
        # Load model if path provided, otherwise create new model
        if model_path and os.path.exists(model_path):
            self.load(model_path)
        elif self.n_features > 0:
            self.build_model()
            
        logger.info(f"Initialized Transformer model with {n_features} features and sequence length {sequence_length}")
    
    def build_model(self) -> None:
        """Build the Transformer model architecture"""
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available. Skipping model build.")
            return

        try:
            # Reset Keras session
            tf.keras.backend.clear_session()
            
            # Get hyperparameters from config
            embed_dim = self.config.get('embed_dim', 64)
            num_heads = self.config.get('num_heads', 4)
            ff_dim = self.config.get('ff_dim', 128)
            num_transformer_blocks = self.config.get('num_transformer_blocks', 4)
            mlp_units = self.config.get('mlp_units', [128, 64])
            dropout_rate = self.config.get('dropout_rate', 0.2)
            learning_rate = self.config.get('learning_rate', 0.001)
            
            # Define inputs
            inputs = Input(shape=(self.sequence_length, self.n_features))
            
            # Project inputs to embedding dimension
            x = Dense(embed_dim)(inputs)
            
            # Positional encoding
            x = PositionalEncoding(self.sequence_length, embed_dim)(x)
            
            # Transformer blocks
            for _ in range(num_transformer_blocks):
                x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)
            
            # Global pooling to convert sequence to vector
            x = GlobalAveragePooling1D()(x)
            
            # MLP head
            for dim in mlp_units:
                x = Dense(dim, activation="relu")(x)
                x = Dropout(dropout_rate)(x)
                
            # Output layer - binary classification for price direction
            outputs = Dense(1, activation="sigmoid")(x)
            
            # Build and compile model
            self.model = Model(inputs=inputs, outputs=outputs)
            
            optimizer = Adam(learning_rate=learning_rate)
            self.model.compile(
                optimizer=optimizer,
                loss="binary_crossentropy",
                metrics=["accuracy"]
            )
            
            # Summary
            self.model.summary()
            logger.info("Transformer model built successfully")
            
        except Exception as e:
            logger.error(f"Error building Transformer model: {e}")
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
        Train the Transformer model
        
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
            logger.error(f"Error training Transformer model: {e}")
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

            # Register custom objects
            custom_objects = {
                'TransformerBlock': TransformerBlock,
                'PositionalEncoding': PositionalEncoding
            }
            
            # Load the model
            self.model = load_model(path, custom_objects=custom_objects)
            
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