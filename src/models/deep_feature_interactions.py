#!/usr/bin/env python3
"""
Deep Feature Interactions Module
Captures nonlinear relationships between features and creates enhanced feature representations
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import joblib
import os
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

# Configure logging
logger = logging.getLogger("deep_features")

class FeatureInteractionNetwork(nn.Module):
    """
    Neural network for learning feature interactions
    Uses self-attention and cross-feature layers to capture complex relationships
    """
    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                output_dim: Optional[int] = None, dropout_rate: float = 0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        
        # Self-attention layer
        self.query = nn.Linear(input_dim, 32)
        self.key = nn.Linear(input_dim, 32)
        self.value = nn.Linear(input_dim, 32)
        
        # Feature cross layers
        self.cross_layers = nn.ModuleList()
        self.cross_layers.append(nn.Linear(input_dim, input_dim, bias=False))
        self.cross_layers.append(nn.Linear(input_dim, input_dim, bias=False))
        
        # Deep layers
        self.deep_layers = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.deep_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.deep_layers.append(nn.BatchNorm1d(hidden_dim))
            prev_dim = hidden_dim
            
        # Output layer
        self.output_layer = nn.Linear(prev_dim + 32 + input_dim, self.output_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Original input for residual connection
        original_x = x
        
        # Self-attention mechanism
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Compute attention scores - dot product attention
        attention = F.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(), dim=1)
        attention_out = attention.unsqueeze(1) * v
        
        # Cross network - explicit feature interactions
        cross_x = x
        for layer in self.cross_layers:
            cross = layer(cross_x)
            cross_x = cross_x * cross + cross_x
        
        # Deep network - implicit feature interactions
        deep_x = x
        for i, layer in enumerate(self.deep_layers):
            deep_x = layer(deep_x)
            # Apply batch norm and activation only on dense layers
            if i % 2 == 1:
                deep_x = F.relu(deep_x)
                deep_x = self.dropout(deep_x)
        
        # Combine all feature representations
        combined = torch.cat([deep_x, attention_out, cross_x], dim=1)
        output = self.output_layer(combined)
        
        # Residual connection to preserve original features
        if self.output_dim == self.input_dim:
            output = output + original_x
            
        return output

class DeepFeatureInteractions:
    """
    Main class for learning and applying deep feature interactions
    """
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize deep feature interactions model
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.batch_size = self.config.get('batch_size', 64)
        self.epochs = self.config.get('epochs', 50)
        self.learning_rate = self.config.get('learning_rate', 0.001)
        self.early_stopping = self.config.get('early_stopping', True)
        self.patience = self.config.get('patience', 10)
        self.hidden_dims = self.config.get('hidden_dims', [64, 32])
        self.dropout_rate = self.config.get('dropout_rate', 0.2)
        
        # Logger
        self.logger = logging.getLogger('deep_features')
        self.logger.info(f"DeepFeatureInteractions initialized, using device: {self.device}")
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None, 
            validation_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None,
            feature_names: Optional[List[str]] = None) -> None:
        """
        Train the feature interaction network
        
        Args:
            X: Input features
            y: Optional target variable for supervised tuning
            validation_data: Optional validation data for early stopping
            feature_names: Optional list of feature names
        """
        from sklearn.preprocessing import StandardScaler
        
        # Store feature names
        self.feature_names = feature_names or X.columns.tolist()
        
        # Normalize inputs
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X_scaled)
        
        if y is not None:
            y_tensor = torch.FloatTensor(y.values.reshape(-1, 1))
            dataset = TensorDataset(X_tensor, y_tensor)
            supervised = True
        else:
            dataset = TensorDataset(X_tensor, X_tensor)  # Autoencoder style
            supervised = False
            
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Create validation dataloader if provided
        val_dataloader = None
        if validation_data is not None:
            X_val, y_val = validation_data
            X_val_scaled = self.scaler.transform(X_val)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            
            if supervised:
                y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1))
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            else:
                val_dataset = TensorDataset(X_val_tensor, X_val_tensor)
                
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        input_dim = X.shape[1]
        output_dim = 1 if supervised else input_dim
        
        self.model = FeatureInteractionNetwork(
            input_dim=input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout_rate=self.dropout_rate
        ).to(self.device)
        
        # Optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Loss function
        if supervised:
            criterion = nn.MSELoss()
        else:
            criterion = nn.MSELoss()  # Could also use other reconstruction losses
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate loss
                if not supervised:
                    loss = criterion(outputs, inputs)  # Reconstruct original features
                else:
                    loss = criterion(outputs, targets)
                    
                # Backward pass & optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
            train_loss /= len(dataloader)
            
            # Validation phase
            val_loss = 0
            if val_dataloader:
                self.model.eval()
                with torch.no_grad():
                    for inputs, targets in val_dataloader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)
                        
                        outputs = self.model(inputs)
                        
                        if not supervised:
                            loss = criterion(outputs, inputs)
                        else:
                            loss = criterion(outputs, targets)
                            
                        val_loss += loss.item()
                        
                val_loss /= len(val_dataloader)
                
                # Early stopping logic
                if self.early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        self.best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch+1}")
                        # Restore best model
                        self.model.load_state_dict(self.best_model_state)
                        break
            
            # Log progress
            log_msg = f"Epoch {epoch+1}/{self.epochs}, Train loss: {train_loss:.6f}"
            if val_dataloader:
                log_msg += f", Val loss: {val_loss:.6f}"
                
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                self.logger.info(log_msg)
            
        self.logger.info("Training complete")
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate enhanced features using the trained model
        
        Args:
            X: Input features
            
        Returns:
            Enhanced feature representation
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        # Set model to eval mode
        self.model.eval()
        
        # Generate enhanced features
        with torch.no_grad():
            enhanced_features = self.model(X_tensor).cpu().numpy()
            
        return enhanced_features
    
    def analyze_feature_interactions(self, X: pd.DataFrame, 
                                    top_k: int = 10) -> pd.DataFrame:
        """
        Analyze which features interact most strongly
        
        Args:
            X: Input features
            top_k: Number of top interactions to return
            
        Returns:
            DataFrame with most important feature interactions
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Get weights from cross layers
        cross_weights = []
        for layer in self.model.cross_layers:
            cross_weights.append(layer.weight.data.cpu().numpy())
            
        # Combine weights from all cross layers
        combined_weights = np.abs(np.mean([w for w in cross_weights], axis=0))
        
        # Create feature interaction matrix
        n_features = combined_weights.shape[0]
        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(n_features)]
        
        interactions = []
        for i in range(n_features):
            for j in range(i+1, n_features):
                # Calculate interaction strength
                interaction_strength = combined_weights[i, j] + combined_weights[j, i]
                interactions.append({
                    'feature_1': feature_names[i],
                    'feature_2': feature_names[j],
                    'strength': interaction_strength
                })
                
        # Convert to DataFrame and sort
        interactions_df = pd.DataFrame(interactions)
        interactions_df = interactions_df.sort_values('strength', ascending=False).head(top_k)
        
        return interactions_df
    
    def plot_attention_weights(self, X: pd.DataFrame, 
                              sample_idx: int = 0) -> None:
        """
        Plot attention weights for a sample
        
        Args:
            X: Input features
            sample_idx: Index of sample to plot attention for
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
            
        # Scale inputs
        X_scaled = self.scaler.transform(X)
        X_sample = torch.FloatTensor(X_scaled[sample_idx:sample_idx+1]).to(self.device)
        
        # Get attention weights
        self.model.eval()
        with torch.no_grad():
            q = self.model.query(X_sample)
            k = self.model.key(X_sample)
            attention = F.softmax(torch.bmm(q.unsqueeze(1), k.unsqueeze(2)).squeeze(), dim=1)
            
        # Plot attention heatmap
        plt.figure(figsize=(10, 8))
        feature_names = self.feature_names if self.feature_names else [f"Feature_{i}" for i in range(X.shape[1])]
        
        attention_matrix = attention.cpu().numpy().reshape(-1, 1)
        df = pd.DataFrame(attention_matrix, index=feature_names, columns=['Attention'])
        df = df.sort_values('Attention', ascending=False)
        
        sns.barplot(x='Attention', y=df.index, data=df)
        plt.title('Feature Attention Weights')
        plt.tight_layout()
        
        # Save figure
        save_dir = os.path.join(os.getcwd(), 'models', 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f'feature_attention_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(save_dir, filename))
        self.logger.info(f"Feature attention plot saved to {filename}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the feature interaction model
        
        Args:
            filepath: Path to save model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
            
        # Save torch model state dict
        torch.save(self.model.state_dict(), f"{filepath}_state.pth")
        
        # Save other components
        model_data = {
            'config': self.config,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'input_dim': self.model.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.model.output_dim,
            'dropout_rate': self.dropout_rate
        }
        
        joblib.dump(model_data, f"{filepath}_data.joblib")
        self.logger.info(f"Feature interaction model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a saved feature interaction model
        
        Args:
            filepath: Path to load model from
        """
        # Load model metadata
        model_data = joblib.load(f"{filepath}_data.joblib")
        
        # Extract components
        self.config = model_data.get('config', self.config)
        self.scaler = model_data.get('scaler')
        self.feature_names = model_data.get('feature_names')
        input_dim = model_data.get('input_dim')
        hidden_dims = model_data.get('hidden_dims', self.hidden_dims)
        output_dim = model_data.get('output_dim')
        dropout_rate = model_data.get('dropout_rate', self.dropout_rate)
        
        # Recreate model
        self.model = FeatureInteractionNetwork(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        # Load state dict
        self.model.load_state_dict(torch.load(f"{filepath}_state.pth", map_location=self.device))
        self.model.eval()
        
        self.logger.info(f"Feature interaction model loaded from {filepath}")

if __name__ == "__main__":
    # Simple test
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Generate correlated features
    X = np.random.randn(n_samples, n_features)
    X[:, 1] = X[:, 0] * 0.5 + np.random.randn(n_samples) * 0.1  # Correlated
    X[:, 3] = X[:, 2] * -0.7 + np.random.randn(n_samples) * 0.2  # Negatively correlated
    X[:, 5] = X[:, 4] * X[:, 6] + np.random.randn(n_samples) * 0.1  # Interaction
    
    # Create nonlinear target
    y = 0.3 * X[:, 0] + 0.5 * X[:, 1] + 0.7 * X[:, 2] * X[:, 3] + np.sin(X[:, 4]) + np.random.randn(n_samples) * 0.1
    
    # Convert to pandas
    feature_names = [f'feature_{i}' for i in range(n_features)]
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y, name='target')
    
    # Create and train feature interaction model
    model = DeepFeatureInteractions(config={
        'batch_size': 64,
        'epochs': 30,
        'hidden_dims': [64, 32],
        'learning_rate': 0.001
    })
    
    # Split into train and validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_df, y_series, test_size=0.2, random_state=42)
    
    # Train supervised
    model.fit(X_train, y_train, validation_data=(X_val, y_val), feature_names=feature_names)
    
    # Transform data
    X_enhanced = model.transform(X_df)
    print(f"Enhanced features shape: {X_enhanced.shape}")
    
    # Analyze interactions
    interactions = model.analyze_feature_interactions(X_df, top_k=5)
    print("Top feature interactions:")
    print(interactions)
    
    # Visualize attention
    model.plot_attention_weights(X_df)