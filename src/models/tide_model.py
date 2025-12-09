import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import logging

logger = logging.getLogger("tide_model")

class TiDEModel:
    def __init__(self, config, sequence_length=60, n_features=14, horizon=1):
        self.config = config
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.horizon = horizon # For now we predict 1 step, but TiDE is good for multi-step
        self.model = None

    def build_model(self):
        """
        Build TiDE (Time-series Dense Encoder) model.
        Reference: https://arxiv.org/abs/2304.08424
        """
        # Hyperparameters
        hidden_dim = self.config.get('hidden_dim', 256)
        decoder_output_dim = self.config.get('decoder_output_dim', 32)
        dropout_rate = self.config.get('dropout_rate', 0.1)
        num_layers = self.config.get('num_layers', 2)
        
        # Inputs
        # Flatten input: (batch, seq_len * n_features)
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        x = layers.Flatten()(inputs)
        
        # Encoder
        # Multiple residual blocks
        for _ in range(num_layers):
            # Residual implementation: x + Dense(Relu(Dense(x)))
            res = layers.Dense(hidden_dim, activation='relu')(x)
            res = layers.Dropout(dropout_rate)(res)
            res = layers.Dense(hidden_dim)(res)
            res = layers.Dropout(dropout_rate)(res)
            
            # Project x to match hidden_dim if needed
            if x.shape[-1] != hidden_dim:
                x_proj = layers.Dense(hidden_dim)(x)
                x = layers.Add()([x_proj, res])
            else:
                x = layers.Add()([x, res])
            
            x = layers.LayerNormalization()(x)
            
        # Decoder
        # Dense decoding to prediction
        x = layers.Dense(decoder_output_dim, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        
        # Final projection to target
        # We are predicting a binary class (Up/Down) for the next step
        # Original TiDE predicts the time series values, but we adapt it for classification here.
        output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=output)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001))
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        logger.info("TiDE model built successfully")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, save_path=None):
        if self.model is None:
            self.build_model()
            
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        if save_path:
            self.save(save_path)
            
        return history.history

    def predict(self, X):
        if self.model is None:
            raise Exception("Model not built or trained")
        return self.model.predict(X)

    def save(self, path):
        if self.model:
            self.model.save(path)
            logger.info(f"Model saved to {path}")

    def load(self, path):
        self.model = models.load_model(path)
        logger.info(f"Model loaded from {path}")
