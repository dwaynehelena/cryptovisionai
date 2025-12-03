#!/usr/bin/env python3
"""
Regime-Specific Ensemble Model Implementation
Utilizes enhanced market regime detection to train and deploy regime-specific models
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Union, Optional, Tuple
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, precision_score, recall_score, f1_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime
import joblib
import os
import logging
from .enhanced_regime import EnhancedRegimeDetector

# Configure logging
logger = logging.getLogger("regime_ensemble")

class RegimeSpecificEnsembleModel:
    """
    Ensemble model that trains separate models for different market regimes
    and uses the appropriate one for prediction based on detected regime
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize regime-specific ensemble model
        
        Args:
            config: Configuration parameters for the model
        """
        self.config = config or {}
        
        # Regime detection
        self.regime_detector = EnhancedRegimeDetector(
            config=self.config.get('regime_detector_config', {})
        )
        self.n_regimes = self.config.get('n_regimes', 4)
        self.regime_detector.n_regimes = self.n_regimes
        
        # Models for each regime
        self.regime_models = {}
        self.fallback_model = None  # Used when no regime-specific model is available
        
        # Model configuration
        self.model_types = self.config.get('model_types', ['xgboost', 'lightgbm', 'random_forest'])
        self.target_type = self.config.get('target_type', 'regression')  # 'regression' or 'classification'
        self.use_voting = self.config.get('use_voting', True)  # Use voting for ensemble
        
        # Performance metrics for each regime
        self.performance_metrics = {}
        self.feature_importance = {}
        
        # Thresholds
        self.min_samples_per_regime = self.config.get('min_samples_per_regime', 50)
        
        # Metadata
        self.metadata = {
            'training_date': None,
            'n_samples': 0,
            'regime_distribution': {},
            'model_info': {}
        }
        
        self.logger = logging.getLogger('regime_ensemble')
    
    def _get_base_model(self, model_type: str):
        """Get a base model of specified type"""
        if self.target_type == 'regression':
            if model_type == 'xgboost':
                return xgb.XGBRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == 'lightgbm':
                return lgb.LGBMRegressor(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == 'random_forest':
                return RandomForestRegressor(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
        else:  # classification
            if model_type == 'xgboost':
                return xgb.XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == 'lightgbm':
                return lgb.LGBMClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            elif model_type == 'random_forest':
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=5,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    def _create_ensemble(self, X_train, y_train):
        """Create an ensemble of models"""
        models = {}
        
        for model_type in self.model_types:
            models[model_type] = self._get_base_model(model_type)
            models[model_type].fit(X_train, y_train)
            
        return models
    
    def _create_regime_model(self, regime_id: int, X_train, y_train):
        """Create a model for a specific regime"""
        self.logger.info(f"Training models for regime {regime_id}")
        
        regime_model = {
            'models': {},
            'voting_weights': None,
            'training_size': len(y_train),
            'feature_names': None
        }
        
        # Train individual models
        for model_type in self.model_types:
            try:
                model = self._get_base_model(model_type)
                model.fit(X_train, y_train)
                regime_model['models'][model_type] = model
                
                # Get feature importance if available
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[f'regime_{regime_id}_{model_type}'] = model.feature_importances_
            except Exception as e:
                self.logger.error(f"Error training {model_type} for regime {regime_id}: {e}")
        
        # Determine voting weights based on validation performance if enough data
        if len(y_train) >= 100:
            X_train_sub, X_val, y_train_sub, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            weights = {}
            for model_type, model in regime_model['models'].items():
                y_pred = model.predict(X_val)
                
                if self.target_type == 'regression':
                    # Lower error means better performance (invert for weights)
                    error = mean_squared_error(y_val, y_pred, squared=False)  # RMSE
                    weights[model_type] = 1.0 / (error + 1e-10)
                else:
                    # Higher accuracy means better performance
                    accuracy = accuracy_score(y_val, y_pred)
                    weights[model_type] = accuracy
            
            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v/total_weight for k, v in weights.items()}
            
            regime_model['voting_weights'] = weights
        else:
            # Equal weights if not enough data
            weights = {model_type: 1.0/len(self.model_types) for model_type in self.model_types}
            regime_model['voting_weights'] = weights
        
        return regime_model
    
    def preprocess_data(self, X, y=None):
        """
        Preprocess data by extracting prices for regime detection
        
        Args:
            X: Features dataframe/array
            y: Optional target values
            
        Returns:
            X with additional regime features, y
        """
        # Extract price column for regime detection
        # Assumes price column exists, or first column is price
        if isinstance(X, pd.DataFrame) and 'price' in X.columns:
            prices = X['price'].values
        elif isinstance(X, pd.DataFrame) and 'close' in X.columns:
            prices = X['close'].values
        else:
            # Try to use the first column as price
            if isinstance(X, pd.DataFrame):
                prices = X.iloc[:, 0].values
            else:
                prices = X[:, 0]
            
            self.logger.warning("Price column not found, using first column for regime detection")
        
        # Get volumes if available
        volumes = None
        if isinstance(X, pd.DataFrame) and 'volume' in X.columns:
            volumes = X['volume'].values
        
        # Detect regimes
        regimes, regime_confidence = self.regime_detector.predict_regime(prices, volumes)
        
        # Generate regime features
        regime_features = self.regime_detector.generate_regime_features(prices, volumes)
        
        # Convert X to DataFrame if it's not already
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        
        # Add regime features to X
        # First create descriptive column names
        n_regime_features = regime_features.shape[1]
        regime_col_names = []
        
        # One-hot regime columns
        for i in range(self.n_regimes):
            regime_col_names.append(f'regime_{i}_active')
            
        # Additional regime features
        current_idx = self.n_regimes
        
        # Add probability column if present (depends on generate_regime_features params)
        if n_regime_features > self.n_regimes + 3:  # If we have more than one-hot + 3 base features
            regime_col_names.append('regime_probability')
            current_idx += 1
            
        # Add standard regime features
        regime_col_names.extend([
            'days_in_regime',
            'recent_regime_switches',
            'transition_probability',
            'periods_to_change'
        ])
        
        # Ensure we have the right number of columns
        regime_col_names = regime_col_names[:n_regime_features]
        
        # Handle different array lengths - regime features might be shorter than original X
        offset = len(X) - len(regime_features)
        
        if offset > 0:
            # Pre-fill with zeros
            padding = np.zeros((offset, regime_features.shape[1]))
            regime_features = np.vstack([padding, regime_features])
            
            self.logger.info(f"Added {offset} rows of padding for regime features")
            
        # Add regime features to X
        for i, col_name in enumerate(regime_col_names):
            X[col_name] = regime_features[:, i]
        
        # Add current regime as a single column
        if offset > 0:
            regimes_padded = np.hstack([np.zeros(offset), regimes])
        else:
            regimes_padded = regimes
            
        X['current_regime'] = regimes_padded.astype(int)
        
        return X, y
    
    def fit(self, X, y):
        """
        Fit regime-specific ensemble models
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Self
        """
        self.logger.info("Fitting regime-specific ensemble model")
        
        # First, detect regimes
        X, y = self.preprocess_data(X, y)
        
        # Record feature names
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        # Get the regime assignments
        regimes = X['current_regime'].values
        
        # Record metadata
        self.metadata['training_date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.metadata['n_samples'] = len(y)
        
        # Count samples in each regime
        unique_regimes, regime_counts = np.unique(regimes, return_counts=True)
        self.metadata['regime_distribution'] = {int(r): int(c) for r, c in zip(unique_regimes, regime_counts)}
        
        self.logger.info(f"Regime distribution: {self.metadata['regime_distribution']}")
        
        # Train regime-specific models
        for regime_id in unique_regimes:
            regime_id = int(regime_id)
            
            # Skip if regime_id is 0 and it was due to padding
            if regime_id == 0 and not np.any(X.iloc[:, -self.n_regimes-1:-1].sum(axis=1) > 0):
                self.logger.warning(f"Skipping regime {regime_id} as it appears to be padding")
                continue
                
            # Filter data for this regime
            mask = regimes == regime_id
            X_regime = X[mask]
            y_regime = y[mask]
            
            # Check if we have enough samples
            if len(y_regime) < self.min_samples_per_regime:
                self.logger.warning(f"Not enough samples for regime {regime_id}: {len(y_regime)} < {self.min_samples_per_regime}")
                continue
            
            # Train models for this regime
            self.regime_models[regime_id] = self._create_regime_model(regime_id, X_regime, y_regime)
            
            # Record model info in metadata
            self.metadata['model_info'][f'regime_{regime_id}'] = {
                'n_samples': len(y_regime),
                'models': list(self.regime_models[regime_id]['models'].keys()),
                'weights': self.regime_models[regime_id]['voting_weights']
            }
        
        # Train a fallback model using all data
        self.logger.info("Training fallback model on all data")
        self.fallback_model = self._create_regime_model(-1, X, y)  # -1 indicates fallback model
        self.metadata['model_info']['fallback'] = {
            'n_samples': len(y),
            'models': list(self.fallback_model['models'].keys()),
            'weights': self.fallback_model['voting_weights']
        }
        
        return self
    
    def predict(self, X):
        """
        Make predictions using regime-specific models
        
        Args:
            X: Features
            
        Returns:
            Predictions
        """
        # First, detect regimes and add regime features
        X, _ = self.preprocess_data(X)
        
        # Get the regime assignments
        regimes = X['current_regime'].values
        
        # Make predictions
        predictions = np.zeros(len(X))
        
        for i, regime_id in enumerate(regimes):
            regime_id = int(regime_id)
            
            # Choose appropriate model
            if regime_id in self.regime_models:
                regime_model = self.regime_models[regime_id]
            else:
                self.logger.warning(f"No model for regime {regime_id}, using fallback model")
                regime_model = self.fallback_model
            
            # Make prediction for this sample using voting ensemble
            x_i = X.iloc[[i]] if isinstance(X, pd.DataFrame) else X[[i]]
            
            # Ensemble prediction with voting
            if self.use_voting:
                # Weighted average of model predictions
                pred_sum = 0
                weight_sum = 0
                
                for model_type, weight in regime_model['voting_weights'].items():
                    if model_type in regime_model['models']:
                        model_pred = regime_model['models'][model_type].predict(x_i)[0]
                        pred_sum += model_pred * weight
                        weight_sum += weight
                
                if weight_sum > 0:
                    predictions[i] = pred_sum / weight_sum
            else:
                # Just use the first available model
                model_type = list(regime_model['models'].keys())[0]
                predictions[i] = regime_model['models'][model_type].predict(x_i)[0]
        
        return predictions
    
    def evaluate(self, X, y):
        """
        Evaluate model performance overall and by regime
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary of performance metrics
        """
        self.logger.info("Evaluating regime-specific ensemble model")
        
        # Detect regimes
        X, y = self.preprocess_data(X, y)
        
        # Get predictions
        y_pred = self.predict(X)
        
        # Overall performance
        metrics = {}
        
        if self.target_type == 'regression':
            metrics['overall'] = {
                'rmse': mean_squared_error(y, y_pred, squared=False),
                'mae': mean_absolute_error(y, y_pred)
            }
        else:
            metrics['overall'] = {
                'accuracy': accuracy_score(y, y_pred),
                'precision': precision_score(y, y_pred, average='weighted'),
                'recall': recall_score(y, y_pred, average='weighted'),
                'f1': f1_score(y, y_pred, average='weighted')
            }
        
        # Performance by regime
        regimes = X['current_regime'].values
        unique_regimes = np.unique(regimes)
        
        for regime_id in unique_regimes:
            regime_id = int(regime_id)
            mask = regimes == regime_id
            
            if np.sum(mask) < 5:  # Need at least 5 samples
                continue
                
            y_regime = y[mask]
            y_pred_regime = y_pred[mask]
            
            if self.target_type == 'regression':
                metrics[f'regime_{regime_id}'] = {
                    'rmse': mean_squared_error(y_regime, y_pred_regime, squared=False),
                    'mae': mean_absolute_error(y_regime, y_pred_regime),
                    'n_samples': int(np.sum(mask))
                }
            else:
                metrics[f'regime_{regime_id}'] = {
                    'accuracy': accuracy_score(y_regime, y_pred_regime),
                    'precision': precision_score(y_regime, y_pred_regime, average='weighted'),
                    'recall': recall_score(y_regime, y_pred_regime, average='weighted'),
                    'f1': f1_score(y_regime, y_pred_regime, average='weighted'),
                    'n_samples': int(np.sum(mask))
                }
        
        self.performance_metrics = metrics
        return metrics
    
    def save_model(self, directory):
        """
        Save the regime-specific ensemble model
        
        Args:
            directory: Directory to save the model
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        # Save metadata
        metadata_path = os.path.join(directory, 'regime_ensemble_metadata.joblib')
        joblib.dump({
            'metadata': self.metadata,
            'performance_metrics': self.performance_metrics,
            'feature_importance': self.feature_importance,
            'feature_names': getattr(self, 'feature_names', None),
            'config': self.config
        }, metadata_path)
        
        # Save regime models
        for regime_id, regime_model in self.regime_models.items():
            regime_dir = os.path.join(directory, f'regime_{regime_id}')
            if not os.path.exists(regime_dir):
                os.makedirs(regime_dir)
                
            for model_type, model in regime_model['models'].items():
                model_path = os.path.join(regime_dir, f'{model_type}_model.joblib')
                joblib.dump(model, model_path)
        
        # Save fallback model
        fallback_dir = os.path.join(directory, 'fallback')
        if not os.path.exists(fallback_dir):
            os.makedirs(fallback_dir)
            
        for model_type, model in self.fallback_model['models'].items():
            model_path = os.path.join(fallback_dir, f'{model_type}_model.joblib')
            joblib.dump(model, model_path)
        
        # Save regime detector
        detector_path = os.path.join(directory, 'regime_detector.joblib')
        self.regime_detector.save_model(detector_path)
        
        # Save weights
        weights_data = {
            'fallback': self.fallback_model['voting_weights']
        }
        for regime_id, regime_model in self.regime_models.items():
            weights_data[f'regime_{regime_id}'] = regime_model['voting_weights']
            
        weights_path = os.path.join(directory, 'voting_weights.joblib')
        joblib.dump(weights_data, weights_path)
        
        self.logger.info(f"Regime-specific ensemble model saved to {directory}")
    
    def load_model(self, directory):
        """
        Load the regime-specific ensemble model
        
        Args:
            directory: Directory to load the model from
        """
        # Load metadata
        metadata_path = os.path.join(directory, 'regime_ensemble_metadata.joblib')
        metadata_dict = joblib.load(metadata_path)
        
        self.metadata = metadata_dict.get('metadata', {})
        self.performance_metrics = metadata_dict.get('performance_metrics', {})
        self.feature_importance = metadata_dict.get('feature_importance', {})
        self.feature_names = metadata_dict.get('feature_names')
        self.config = metadata_dict.get('config', self.config)
        
        # Update configuration
        if 'n_regimes' in self.config:
            self.n_regimes = self.config['n_regimes']
        
        # Load regime detector
        detector_path = os.path.join(directory, 'regime_detector.joblib')
        self.regime_detector.load_model(detector_path)
        
        # Load weights
        weights_path = os.path.join(directory, 'voting_weights.joblib')
        weights_data = joblib.load(weights_path)
        
        # Load regime models
        self.regime_models = {}
        for regime_key, regime_info in self.metadata['model_info'].items():
            if regime_key == 'fallback':
                continue
                
            regime_id = int(regime_key.split('_')[1])
            regime_dir = os.path.join(directory, f'regime_{regime_id}')
            
            regime_model = {
                'models': {},
                'voting_weights': weights_data.get(f'regime_{regime_id}', {}),
                'training_size': regime_info['n_samples'],
                'feature_names': self.feature_names
            }
            
            for model_type in regime_info['models']:
                model_path = os.path.join(regime_dir, f'{model_type}_model.joblib')
                if os.path.exists(model_path):
                    regime_model['models'][model_type] = joblib.load(model_path)
                    
            self.regime_models[regime_id] = regime_model
        
        # Load fallback model
        fallback_dir = os.path.join(directory, 'fallback')
        fallback_info = self.metadata['model_info'].get('fallback', {})
        
        self.fallback_model = {
            'models': {},
            'voting_weights': weights_data.get('fallback', {}),
            'training_size': fallback_info.get('n_samples', 0),
            'feature_names': self.feature_names
        }
        
        for model_type in fallback_info.get('models', []):
            model_path = os.path.join(fallback_dir, f'{model_type}_model.joblib')
            if os.path.exists(model_path):
                self.fallback_model['models'][model_type] = joblib.load(model_path)
        
        self.logger.info(f"Regime-specific ensemble model loaded from {directory}")
    
    def plot_regime_performance(self, figsize=(12, 8)):
        """
        Plot performance metrics by regime
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.performance_metrics:
            self.logger.warning("No performance metrics available. Run evaluate() first.")
            return None
        
        # Extract metrics
        regimes = []
        rmse_values = []
        mae_values = []
        n_samples = []
        
        for key, metrics in self.performance_metrics.items():
            if key == 'overall':
                continue
                
            regimes.append(key)
            
            if self.target_type == 'regression':
                rmse_values.append(metrics['rmse'])
                mae_values.append(metrics['mae'])
            else:
                # For classification, use accuracy and f1
                rmse_values.append(metrics['accuracy'])
                mae_values.append(metrics['f1'])
                
            n_samples.append(metrics['n_samples'])
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Plot RMSE or Accuracy
        x_pos = np.arange(len(regimes))
        
        if self.target_type == 'regression':
            bars = ax1.bar(x_pos, rmse_values)
            ax1.set_title('RMSE by Regime')
            ax1.set_ylabel('RMSE')
        else:
            bars = ax1.bar(x_pos, rmse_values)
            ax1.set_title('Accuracy by Regime')
            ax1.set_ylabel('Accuracy')
            
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(regimes)
        
        # Add regime labels if available
        if hasattr(self.regime_detector, 'regime_labels') and self.regime_detector.regime_labels:
            labels = []
            for regime in regimes:
                regime_id = int(regime.split('_')[1])
                label = self.regime_detector.regime_labels.get(f'ensemble_{regime_id}', regime)
                labels.append(label)
            ax1.set_xticklabels(labels, rotation=45, ha='right')
        
        # Plot MAE or F1
        if self.target_type == 'regression':
            bars = ax2.bar(x_pos, mae_values)
            ax2.set_title('MAE by Regime')
            ax2.set_ylabel('MAE')
        else:
            bars = ax2.bar(x_pos, mae_values)
            ax2.set_title('F1 Score by Regime')
            ax2.set_ylabel('F1 Score')
            
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(regimes)
        
        # Add regime labels if available
        if hasattr(self.regime_detector, 'regime_labels') and self.regime_detector.regime_labels:
            ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # Add overall performance line
        overall_metrics = self.performance_metrics.get('overall', {})
        if self.target_type == 'regression':
            overall_rmse = overall_metrics.get('rmse', 0)
            overall_mae = overall_metrics.get('mae', 0)
            ax1.axhline(y=overall_rmse, color='r', linestyle='-', label=f'Overall: {overall_rmse:.4f}')
            ax2.axhline(y=overall_mae, color='r', linestyle='-', label=f'Overall: {overall_mae:.4f}')
        else:
            overall_acc = overall_metrics.get('accuracy', 0)
            overall_f1 = overall_metrics.get('f1', 0)
            ax1.axhline(y=overall_acc, color='r', linestyle='-', label=f'Overall: {overall_acc:.4f}')
            ax2.axhline(y=overall_f1, color='r', linestyle='-', label=f'Overall: {overall_f1:.4f}')
        
        ax1.legend()
        ax2.legend()
        
        plt.tight_layout()
        
        # Save figure
        save_dir = os.path.join(os.getcwd(), 'models', 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f'regime_performance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(save_dir, filename))
        self.logger.info(f"Performance plot saved to {filename}")
        
        return fig

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate synthetic data with different regimes
    np.random.seed(42)
    
    # First regime - positive relationship
    X1 = np.random.normal(0, 1, (300, 5))
    y1 = 0.5 * X1[:, 0] + 0.3 * X1[:, 1] + 0.1 * np.random.normal(0, 1, 300)
    
    # Second regime - negative relationship
    X2 = np.random.normal(0, 1, (200, 5))
    y2 = -0.7 * X2[:, 0] - 0.2 * X2[:, 1] + 0.1 * np.random.normal(0, 1, 200)
    
    # Third regime - no relationship, just noise
    X3 = np.random.normal(0, 1, (150, 5))
    y3 = 0.05 * np.random.normal(0, 1, 150)
    
    # Combine data
    X = np.vstack([X1, X2, X3])
    y = np.hstack([y1, y2, y3])
    
    # Add 'price' column for regime detection
    price = np.cumsum(0.01 * np.random.normal(0, 1, len(X)) + 0.002)
    price = 100 * np.exp(price)  # Make it look like a price series
    
    X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    X_df['price'] = price
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, shuffle=False)
    
    # Create and train regime-specific ensemble
    model = RegimeSpecificEnsembleModel(config={'n_regimes': 3})
    model.fit(X_train, y_train)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test)
    print("Performance metrics:")
    for regime, regime_metrics in metrics.items():
        print(f"{regime}: {regime_metrics}")
    
    # Plot performance
    model.plot_regime_performance()
    
    # Test saving and loading
    model.save_model(os.path.join(os.getcwd(), 'models', 'regime_ensemble'))
    
    new_model = RegimeSpecificEnsembleModel()
    new_model.load_model(os.path.join(os.getcwd(), 'models', 'regime_ensemble'))
    
    # Evaluate loaded model
    new_metrics = new_model.evaluate(X_test, y_test)
    print("\nLoaded model performance:")
    for regime, regime_metrics in new_metrics.items():
        print(f"{regime}: {regime_metrics}")