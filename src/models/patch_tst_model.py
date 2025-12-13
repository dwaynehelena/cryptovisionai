import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import os
import joblib
import logging

logger = logging.getLogger("patch_tst_model")

class PatchTSTModel:
    def __init__(self, config, sequence_length=60, n_features=14):
        self.config = config
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.model = None
        
    def build_model(self):
        """
        Build PatchTST model
        """
        # Hyperparameters
        patch_len = self.config.get('patch_len', 16)
        stride = self.config.get('stride', 8)
        embed_dim = self.config.get('embed_dim', 128)
        num_heads = self.config.get('num_heads', 8)
        ff_dim = self.config.get('ff_dim', 256)
        num_layers = self.config.get('num_layers', 3)
        dropout_rate = self.config.get('dropout_rate', 0.1)
        
        # Calculate number of patches
        num_patches = (self.sequence_length - patch_len) // stride + 1
        
        # Inputs: (batch_size, seq_len, n_features)
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))
        
        # Channel Independence: Treat each feature as a separate time series
        # Reshape to (batch_size, n_features, seq_len, 1) to share weights across features
        x = layers.Permute((2, 1))(inputs) # (batch, n_features, seq_len)
        x = layers.Reshape((self.n_features, self.sequence_length, 1))(x)
        
        # Patching
        # Extract patches: (batch, n_features, num_patches, patch_len)
        # We simulate this by reshaping/slicing or using a custom layer. 
        # For simplicity in Keras, we can use a Conv1D with stride on the last dimension if we treat it right.
        # But standard Conv1D works on the temporal dimension.
        
        # Let's process each feature independently using TimeDistributed or sharing layers
        
        # Shared Encoder
        encoder_input = layers.Input(shape=(self.sequence_length, 1))
        
        # Patch Embedding (Conv1D with kernel=patch_len, stride=stride)
        # Output: (batch, num_patches, embed_dim)
        patches = layers.Conv1D(filters=embed_dim, kernel_size=patch_len, strides=stride, padding='valid')(encoder_input)
        
        # Positional Encoding
        pos_encoding = layers.Embedding(input_dim=num_patches, output_dim=embed_dim)(tf.range(num_patches))
        patches = patches + pos_encoding
        
        # Transformer Encoder
        for _ in range(num_layers):
            # Self Attention
            attention_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(patches, patches)
            attention_output = layers.Dropout(dropout_rate)(attention_output)
            x1 = layers.Add()([patches, attention_output])
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
            
            # Feed Forward
            ffn_output = layers.Dense(ff_dim, activation="gelu")(x1)
            ffn_output = layers.Dense(embed_dim)(ffn_output)
            ffn_output = layers.Dropout(dropout_rate)(ffn_output)
            patches = layers.Add()([x1, ffn_output])
            patches = layers.LayerNormalization(epsilon=1e-6)(patches)
            
        # Flatten patches: (batch, num_patches * embed_dim)
        encoded = layers.Flatten()(patches)
        
        # Head
        output = layers.Dense(1)(encoded) # Predict 1 value per feature (or we can change this)
        
        # Create the shared encoder model
        encoder_model = Model(encoder_input, output)
        
        # Apply shared encoder to each feature
        # inputs: (batch, seq_len, n_features)
        outputs = []
        for i in range(self.n_features):
            # Slice feature i: (batch, seq_len, 1)
            feature_slice = layers.Lambda(lambda x: x[:, :, i:i+1])(inputs)
            out = encoder_model(feature_slice) # (batch, 1)
            outputs.append(out)
            
        # Concatenate feature outputs: (batch, n_features)
        concat_outputs = layers.Concatenate()(outputs)
        
        # Final prediction head (mixing features)
        # We want to predict a single binary target (Up/Down)
        x = layers.Dense(64, activation='relu')(concat_outputs)
        x = layers.Dropout(dropout_rate)(x)
        final_output = layers.Dense(1, activation='sigmoid')(x)
        
        self.model = Model(inputs=inputs, outputs=final_output)
        
        try:
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=self.config.get('learning_rate', 0.001))
        except AttributeError:
             optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.get('learning_rate', 0.001))
        self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        logger.info("PatchTST model built successfully")
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, epochs=50, save_path=None, **kwargs):
        if self.model is None:
            self.build_model()
            
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Filter standard args from kwargs if present to avoid duplication
        fit_kwargs = kwargs.copy()
        if 'verbose' not in fit_kwargs:
            fit_kwargs['verbose'] = 1
            
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val) if X_val is not None else None,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks,
            **fit_kwargs
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
