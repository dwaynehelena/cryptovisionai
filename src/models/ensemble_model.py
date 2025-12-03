#!/usr/bin/env python3
"""
Ensemble Model for Bitcoin Price Prediction
Combines multiple models for improved prediction accuracy
"""

import numpy as np
import pandas as pd
import os
import logging
import joblib
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import json
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb
# Use our compatibility wrapper instead of direct import
from ..utils.lightgbm_compat import LGBMClassifier, LGBMRegressor, LIGHTGBM_AVAILABLE, lgb
import itertools
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTETomek, SMOTEENN
import random
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter

# Import custom models
from .lstm_model import LSTMModel
from .transformer_model import TransformerModel
from .ensemble_optimizer import EnsembleOptimizer

# Configure logging
logger = logging.getLogger("ensemble_model")

# New Feature Interaction Generator Class
class FeatureInteractionGenerator:
    """
    Identifies and generates important feature interactions to improve model performance.
    Uses mutual information, correlation analysis and tree-based methods to discover interactions.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize feature interaction generator
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        self.max_interactions = self.config.get('max_interactions', 20)
        self.interaction_threshold = self.config.get('interaction_threshold', 0.01)
        self.logger = logging.getLogger('feature_interaction')
        
    def generate_polynomial_features(self, X: np.ndarray, feature_names: List[str] = None,
                                  degree: int = 2, top_k: int = 10) -> Tuple[np.ndarray, List[str]]:
        """
        Generate polynomial features for important feature interactions
        
        Args:
            X: Input feature matrix
            feature_names: Names of features (for interpretability)
            degree: Maximum polynomial degree
            top_k: Maximum number of interactions per degree
            
        Returns:
            Tuple of augmented feature matrix and new feature names
        """
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.feature_selection import mutual_info_regression
        
        self.logger.info(f"Generating polynomial features with degree={degree}, top_k={top_k}")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        # Original features and names
        X_augmented = X.copy()
        augmented_names = feature_names.copy()
        
        # Calculate feature importance using mutual information
        try:
            # Generate synthetic target for mutual info calculation
            # This works better than random target for finding good interactions
            synthetic_y = np.mean(X, axis=1) + np.std(X, axis=1)
            mi_scores = mutual_info_regression(X, synthetic_y, random_state=self.random_state)
            
            # Sort features by importance
            important_indices = np.argsort(-mi_scores)
            
            # Use top 50% features for interactions to avoid combinatorial explosion
            num_to_use = max(3, X.shape[1] // 2)
            selected_indices = important_indices[:num_to_use]
            
            # For pairs, use only the selected important features
            if degree >= 2:
                pairs = list(itertools.combinations(selected_indices, 2))
                # Sort pairs by combined importance
                pairs_scores = [(i, j, mi_scores[i] + mi_scores[j]) for i, j in pairs]
                pairs_scores.sort(key=lambda x: x[2], reverse=True)
                
                # Take top_k pairs
                top_pairs = pairs_scores[:top_k]
                
                # Generate pair interaction features
                for i, j, _ in top_pairs:
                    interaction = X[:, i] * X[:, j]
                    X_augmented = np.column_stack((X_augmented, interaction))
                    augmented_names.append(f"{feature_names[i]}*{feature_names[j]}")
                    
                    # Also add division if no zeros
                    if not np.any(X[:, j] == 0):
                        ratio = X[:, i] / (X[:, j] + 1e-10)
                        # Remove extreme outliers
                        ratio = np.clip(ratio, -1e5, 1e5)
                        X_augmented = np.column_stack((X_augmented, ratio))
                        augmented_names.append(f"{feature_names[i]}/{feature_names[j]}")
            
            # For triplets, be even more selective
            if degree >= 3:
                # Take top features only
                triplet_indices = selected_indices[:min(5, len(selected_indices))]
                triplets = list(itertools.combinations(triplet_indices, 3))
                
                # Limit number of triplets
                max_triplets = min(top_k // 2, len(triplets))
                selected_triplets = triplets[:max_triplets]
                
                # Generate triplet interaction features
                for i, j, k in selected_triplets:
                    interaction = X[:, i] * X[:, j] * X[:, k]
                    X_augmented = np.column_stack((X_augmented, interaction))
                    augmented_names.append(f"{feature_names[i]}*{feature_names[j]}*{feature_names[k]}")
            
            # Add squares for important features
            for idx in selected_indices[:min(5, len(selected_indices))]:
                squared = X[:, idx] ** 2
                X_augmented = np.column_stack((X_augmented, squared))
                augmented_names.append(f"{feature_names[idx]}^2")
                
            self.logger.info(f"Generated {X_augmented.shape[1] - X.shape[1]} interaction features")
            return X_augmented, augmented_names
            
        except Exception as e:
            self.logger.error(f"Error generating polynomial features: {e}")
            return X, feature_names
            
    def detect_tree_based_interactions(self, X: np.ndarray, y: np.ndarray, 
                                     feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Detect feature interactions using tree-based models
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Tuple of augmented feature matrix and new feature names
        """
        self.logger.info("Detecting tree-based feature interactions")
        
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Train a tree-based model
            rf_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state
            )
            rf_model.fit(X, y)
            
            # Get feature importances
            importances = rf_model.feature_importances_
            
            # Select top features
            top_indices = np.argsort(-importances)[:10]
            
            # Generate interactions between top features
            X_augmented = X.copy()
            augmented_names = feature_names.copy()
            
            # Create interactions
            for i, j in itertools.combinations(top_indices, 2):
                # Multiply features
                interaction = X[:, i] * X[:, j]
                X_augmented = np.column_stack((X_augmented, interaction))
                augmented_names.append(f"{feature_names[i]}*{feature_names[j]}")
                
                # Also try difference for time series
                diff = X[:, i] - X[:, j]
                X_augmented = np.column_stack((X_augmented, diff))
                augmented_names.append(f"{feature_names[i]}-{feature_names[j]}")
            
            self.logger.info(f"Generated {X_augmented.shape[1] - X.shape[1]} tree-based interaction features")
            return X_augmented, augmented_names
            
        except Exception as e:
            self.logger.error(f"Error detecting tree-based interactions: {e}")
            return X, feature_names
            
    def detect_correlation_based_interactions(self, X: np.ndarray, y: np.ndarray, 
                                           feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create features based on highly correlated feature groups
        
        Args:
            X: Feature matrix
            y: Target variable
            feature_names: Names of features
            
        Returns:
            Tuple of augmented feature matrix and new feature names
        """
        self.logger.info("Detecting correlation-based feature interactions")
        
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
            # Calculate correlation matrix
            corr_matrix = np.corrcoef(X.T)
            np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
            
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(corr_matrix.shape[0]):
                for j in range(i+1, corr_matrix.shape[0]):
                    corr = abs(corr_matrix[i, j])
                    if corr > 0.7:  # Highly correlated
                        high_corr_pairs.append((i, j, corr))
            
            # Sort by correlation strength
            high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
            
            # Take top pairs
            X_augmented = X.copy()
            augmented_names = feature_names.copy()
            
            for i, j, corr in high_corr_pairs[:self.max_interactions]:
                # For correlated features, create difference and ratio
                diff = X[:, i] - X[:, j]
                X_augmented = np.column_stack((X_augmented, diff))
                augmented_names.append(f"{feature_names[i]}-{feature_names[j]}")
                
                # Add ratio if no zeros
                if not np.any(X[:, j] == 0):
                    ratio = X[:, i] / (X[:, j] + 1e-10)
                    ratio = np.clip(ratio, -1e5, 1e5)  # Remove extremes
                    X_augmented = np.column_stack((X_augmented, ratio))
                    augmented_names.append(f"{feature_names[i]}/{feature_names[j]}")
                
                # For highly positively correlated features, also average them
                if corr_matrix[i, j] > 0.8:
                    avg = (X[:, i] + X[:, j]) / 2
                    X_augmented = np.column_stack((X_augmented, avg))
                    augmented_names.append(f"avg({feature_names[i]},{feature_names[j]})")
                    
            self.logger.info(f"Generated {X_augmented.shape[1] - X.shape[1]} correlation-based interaction features")
            return X_augmented, augmented_names
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation-based interactions: {e}")
            return X, feature_names
    
    def find_temporal_interactions(self, X: np.ndarray, time_periods: int = 5,
                                feature_names: List[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Find interactions between features across time periods
        Useful for time series data where columns represent different time periods
        
        Args:
            X: Feature matrix
            time_periods: Number of time periods in the data
            feature_names: Names of features
            
        Returns:
            Tuple of augmented feature matrix and new feature names
        """
        self.logger.info(f"Finding temporal interactions across {time_periods} periods")
        
        try:
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
            if X.shape[1] % time_periods != 0:
                self.logger.warning(f"Feature count {X.shape[1]} not divisible by time periods {time_periods}")
                return X, feature_names
                
            # Determine features per period
            features_per_period = X.shape[1] // time_periods
            
            # Reshape data to (samples, periods, features)
            X_reshaped = X.reshape(X.shape[0], time_periods, features_per_period)
            
            # Calculate temporal differences (t - t-1)
            X_diff = np.zeros((X.shape[0], time_periods-1, features_per_period))
            for t in range(1, time_periods):
                X_diff[:, t-1] = X_reshaped[:, t] - X_reshaped[:, t-1]
                
            # Calculate momentum (normalized differences)
            X_momentum = np.zeros((X.shape[0], time_periods-1, features_per_period))
            for t in range(1, time_periods):
                # Avoid division by zero
                denominator = np.abs(X_reshaped[:, t-1]) + 1e-10
                X_momentum[:, t-1] = (X_reshaped[:, t] - X_reshaped[:, t-1]) / denominator
                
            # Reshape back to 2D
            X_diff_flat = X_diff.reshape(X.shape[0], -1)
            X_momentum_flat = X_momentum.reshape(X.shape[0], -1)
            
            # Create new feature names
            diff_names = []
            for t in range(1, time_periods):
                for f in range(features_per_period):
                    orig_idx = f + (t-1)*features_per_period
                    if orig_idx < len(feature_names):
                        base_name = feature_names[orig_idx]
                        diff_names.append(f"diff_{base_name}_t{t}")
            
            momentum_names = []
            for t in range(1, time_periods):
                for f in range(features_per_period):
                    orig_idx = f + (t-1)*features_per_period
                    if orig_idx < len(feature_names):
                        base_name = feature_names[orig_idx]
                        momentum_names.append(f"momentum_{base_name}_t{t}")
            
            # Combine original and new features
            X_augmented = np.hstack([X, X_diff_flat, X_momentum_flat])
            augmented_names = feature_names + diff_names + momentum_names
            
            self.logger.info(f"Generated {X_augmented.shape[1] - X.shape[1]} temporal interaction features")
            return X_augmented, augmented_names
            
        except Exception as e:
            self.logger.error(f"Error finding temporal interactions: {e}")
            return X, feature_names
    
    def generate_all_interaction_features(self, X: np.ndarray, y: np.ndarray = None,
                                       feature_names: List[str] = None,
                                       time_series: bool = True,
                                       time_periods: int = 5) -> Tuple[np.ndarray, List[str]]:
        """
        Generate all types of interaction features
        
        Args:
            X: Feature matrix
            y: Target variable (optional)
            feature_names: Names of features
            time_series: Whether data is time series
            time_periods: Number of time periods if time series
            
        Returns:
            Tuple of augmented feature matrix and new feature names
        """
        self.logger.info("Generating all types of interaction features")
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            
        X_augmented = X.copy()
        augmented_names = feature_names.copy()
        
        # Apply polynomial interactions first
        X_augmented, augmented_names = self.generate_polynomial_features(
            X_augmented, augmented_names, degree=2, top_k=min(10, self.max_interactions)
        )
        
        # Apply tree-based interactions if we have labels
        if y is not None:
            X_augmented, augmented_names = self.detect_tree_based_interactions(
                X_augmented, y, augmented_names
            )
            
        # Apply correlation-based interactions
        X_augmented, augmented_names = self.detect_correlation_based_interactions(
            X_augmented, y, augmented_names
        )
        
        # Apply temporal interactions if time series
        if time_series and X.shape[1] % time_periods == 0:
            X_augmented, augmented_names = self.find_temporal_interactions(
                X_augmented, time_periods, augmented_names
            )
            
        # Feature selection if too many were created
        if X_augmented.shape[1] > 3 * X.shape[1] and y is not None:
            self.logger.info(f"Applying feature selection to reduce from {X_augmented.shape[1]} features")
            try:
                from sklearn.feature_selection import SelectFromModel
                selector = SelectFromModel(
                    RandomForestClassifier(n_estimators=50, random_state=self.random_state),
                    threshold="median"
                )
                selector.fit(X_augmented, y)
                
                # Get selected feature indices
                selected_indices = selector.get_support(indices=True)
                
                # Keep only selected features
                X_augmented = X_augmented[:, selected_indices]
                augmented_names = [augmented_names[i] for i in selected_indices]
                
                self.logger.info(f"Reduced to {X_augmented.shape[1]} features after selection")
            except Exception as e:
                self.logger.error(f"Feature selection failed: {e}")
        
        return X_augmented, augmented_names

# Data augmentation utilities
class DataAugmentation:
    """
    Data augmentation techniques for cryptocurrency market data
    Provides methods to generate synthetic data points and handle imbalanced classes
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data augmentation module
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.random_state = self.config.get('random_state', 42)
        
    def apply_smote(self, X: np.ndarray, y: np.ndarray, method: str = 'regular') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE or variant to balance classes
        
        Args:
            X: Input features
            y: Target labels
            method: SMOTE variant to use ('regular', 'borderline', 'adasyn', 'kmeans', 'tomek', 'enn')
            
        Returns:
            Tuple of resampled features and targets
        """
        smote_ratio = self.config.get('smote_ratio', 'auto')
        k_neighbors = self.config.get('k_neighbors', 5)
        
        logger.info(f"Applying {method} SMOTE with ratio {smote_ratio}")
        
        try:
            if method == 'regular':
                sampler = SMOTE(sampling_strategy=smote_ratio, random_state=self.random_state, k_neighbors=k_neighbors)
            elif method == 'borderline':
                sampler = BorderlineSMOTE(sampling_strategy=smote_ratio, random_state=self.random_state, k_neighbors=k_neighbors)
            elif method == 'adasyn':
                sampler = ADASYN(sampling_strategy=smote_ratio, random_state=self.random_state, n_neighbors=k_neighbors)
            elif method == 'kmeans':
                sampler = KMeansSMOTE(sampling_strategy=smote_ratio, random_state=self.random_state, k_neighbors=k_neighbors)
            elif method == 'tomek':
                sampler = SMOTETomek(sampling_strategy=smote_ratio, random_state=self.random_state, smote=SMOTE(k_neighbors=k_neighbors))
            elif method == 'enn':
                sampler = SMOTEENN(sampling_strategy=smote_ratio, random_state=self.random_state, smote=SMOTE(k_neighbors=k_neighbors))
            else:
                logger.warning(f"Unknown SMOTE method: {method}, using regular SMOTE")
                sampler = SMOTE(sampling_strategy=smote_ratio, random_state=self.random_state, k_neighbors=k_neighbors)
                
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Log the class distribution change
            before_counts = np.unique(y, return_counts=True)[1]
            after_counts = np.unique(y_resampled, return_counts=True)[1]
            logger.info(f"Before {method} SMOTE: {before_counts}, after: {after_counts}")
            
            return X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Error applying {method} SMOTE: {e}")
            logger.warning("Returning original data without resampling")
            return X, y
    
    def generate_synthetic_market_conditions(self, X: np.ndarray, y: np.ndarray, 
                                           condition_type: str = 'volatile', 
                                           n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic data for specific market conditions
        
        Args:
            X: Original feature data
            y: Original target data
            condition_type: Type of market condition to generate ('volatile', 'trending_up', 'trending_down', 'sideways')
            n_samples: Number of synthetic samples to generate
            
        Returns:
            Tuple of synthetic features and targets
        """
        logger.info(f"Generating {n_samples} synthetic samples for {condition_type} market condition")
        
        # Find samples that match the condition type
        condition_indices = self._find_condition_indices(X, y, condition_type)
        
        if len(condition_indices) < 5:
            logger.warning(f"Not enough samples for {condition_type} condition, using random samples")
            condition_indices = np.random.choice(len(X), size=min(len(X), 20), replace=False)
        
        # Subset data to condition samples
        X_condition = X[condition_indices]
        y_condition = y[condition_indices]
        
        # Generate synthetic samples based on these samples
        try:
            if len(X_condition) < 2:
                # Not enough samples for SMOTE, use noise-based augmentation
                X_synthetic, y_synthetic = self._noise_based_augmentation(X_condition, y_condition, n_samples)
            else:
                # Use SMOTE to generate similar samples
                oversampler = SMOTE(sampling_strategy='auto', k_neighbors=min(len(X_condition)-1, 5), random_state=self.random_state)
                
                # If we need more samples than SMOTE would naturally generate, repeat the process
                X_aug, y_aug = X_condition, y_condition
                X_synthetic, y_synthetic = np.empty((0, X.shape[1])), np.empty(0)
                
                while len(X_synthetic) < n_samples:
                    # Apply SMOTE to generate new samples
                    X_temp, y_temp = oversampler.fit_resample(X_aug, y_aug)
                    
                    # Extract only the synthetic samples
                    mask = np.ones(len(X_temp), dtype=bool)
                    mask[:len(X_aug)] = False
                    
                    X_new = X_temp[mask]
                    y_new = y_temp[mask]
                    
                    # Add some noise for diversity
                    X_new = self._add_noise(X_new, noise_level=0.05)
                    
                    # Add to synthetic dataset
                    X_synthetic = np.vstack((X_synthetic, X_new))
                    y_synthetic = np.append(y_synthetic, y_new)
                    
                    # Update X_aug with some new synthetic samples to continue generation
                    X_aug = np.vstack((X_condition, X_new[:10]))
                    y_aug = np.append(y_condition, y_new[:10])
                
                # Trim to the requested number of samples
                X_synthetic = X_synthetic[:n_samples]
                y_synthetic = y_synthetic[:n_samples]
                
            logger.info(f"Generated {len(X_synthetic)} synthetic {condition_type} samples")
            return X_synthetic, y_synthetic
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return np.empty((0, X.shape[1])), np.empty(0)
    
    def _find_condition_indices(self, X: np.ndarray, y: np.ndarray, condition_type: str) -> np.ndarray:
        """
        Find indices corresponding to specific market conditions
        
        Args:
            X: Feature data
            y: Target data
            condition_type: Type of condition to find
            
        Returns:
            Array of indices matching the condition
        """
        # Try to identify feature column types based on feature names in config
        volatility_cols = []
        trend_cols = []
        
        # Determine feature columns by types
        feature_map = self.config.get('feature_map', {})
        for i in range(X.shape[1]):
            if self._check_feature_type(i, 'volatility'):
                volatility_cols.append(i)
            if self._check_feature_type(i, 'trend'):
                trend_cols.append(i)
                
        if not volatility_cols and not trend_cols:
            # Fallback if no feature map: use reasonable heuristics
            # Assume last columns might be volatility indicators
            volatility_cols = list(range(max(0, X.shape[1] - 5), X.shape[1]))
            # Assume early columns might be trend indicators
            trend_cols = list(range(min(5, X.shape[1])))
        
        # Find indices based on condition type
        if condition_type == 'volatile' and volatility_cols:
            # Find samples with high volatility
            volatility_features = X[:, volatility_cols]
            mean_volatility = np.mean(volatility_features, axis=1)
            threshold = np.percentile(mean_volatility, 75)
            return np.where(mean_volatility > threshold)[0]
            
        elif condition_type == 'trending_up' and trend_cols:
            # Find samples with strong upward trend
            trend_features = X[:, trend_cols]
            mean_trend = np.mean(trend_features, axis=1)
            threshold = np.percentile(mean_trend, 75)
            return np.where(mean_trend > threshold)[0]
            
        elif condition_type == 'trending_down' and trend_cols:
            # Find samples with strong downward trend
            trend_features = X[:, trend_cols]
            mean_trend = np.mean(trend_features, axis=1)
            threshold = np.percentile(mean_trend, 25)
            return np.where(mean_trend < threshold)[0]
            
        elif condition_type == 'sideways':
            # Find samples with low volatility
            if volatility_cols:
                volatility_features = X[:, volatility_cols]
                mean_volatility = np.mean(volatility_features, axis=1)
                threshold = np.percentile(mean_volatility, 25)
                return np.where(mean_volatility < threshold)[0]
            
        # Fallback to random selection
        return np.random.choice(len(X), size=min(len(X), int(len(X)*0.1)), replace=False)
    
    def _check_feature_type(self, feature_idx: int, feature_type: str) -> bool:
        """Check if a feature corresponds to a specific type based on config"""
        feature_map = self.config.get('feature_map', {})
        return feature_idx in feature_map.get(feature_type, [])
    
    def _noise_based_augmentation(self, X_orig: np.ndarray, y_orig: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic samples by adding noise to original data
        
        Args:
            X_orig: Original feature data
            y_orig: Original target data
            n_samples: Number of samples to generate
            
        Returns:
            Tuple of synthetic features and targets
        """
        # Repeat original samples to get enough samples
        indices = np.random.choice(len(X_orig), size=n_samples, replace=True)
        X_synthetic = X_orig[indices].copy()
        y_synthetic = y_orig[indices].copy()
        
        # Add random noise to each feature
        X_synthetic = self._add_noise(X_synthetic, noise_level=0.1)
        
        return X_synthetic, y_synthetic
    
    def _add_noise(self, X: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add random noise to features
        
        Args:
            X: Feature data
            noise_level: Scale of noise to add
            
        Returns:
            Features with added noise
        """
        # Calculate standard deviation for each feature
        std = np.std(X, axis=0)
        
        # Generate noise with appropriate scale
        noise = np.random.normal(0, noise_level, X.shape)
        noise = noise * std
        
        # Add noise to features
        X_noisy = X + noise
        
        return X_noisy
    
    def temporal_data_segmentation(self, X: np.ndarray, y: np.ndarray, timestamps: np.ndarray, 
                                  n_segments: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Segment data based on time periods to capture different market conditions
        
        Args:
            X: Feature data
            y: Target data
            timestamps: Array of timestamps for each sample
            n_segments: Number of segments to create
            
        Returns:
            List of (X, y) tuples for each segment
        """
        if len(timestamps) != len(X):
            logger.error("Timestamps array length does not match data length")
            return [(X, y)]
        
        # Sort data by timestamp
        sorted_indices = np.argsort(timestamps)
        X_sorted = X[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # Create equal-sized segments
        segment_size = len(X) // n_segments
        segments = []
        
        for i in range(n_segments):
            start_idx = i * segment_size
            # For the last segment, include any remaining samples
            end_idx = (i + 1) * segment_size if i < n_segments - 1 else len(X)
            
            X_segment = X_sorted[start_idx:end_idx]
            y_segment = y_sorted[start_idx:end_idx]
            
            segments.append((X_segment, y_segment))
            
        logger.info(f"Created {len(segments)} temporal segments from data")
        return segments
    
    def add_market_regime_features(self, X: np.ndarray, market_regimes: np.ndarray) -> np.ndarray:
        """
        Add market regime indicators as features
        
        Args:
            X: Feature data
            market_regimes: Array of market regime labels
            
        Returns:
            Enhanced feature array with market regime indicators
        """
        # One-hot encode market regimes
        unique_regimes = np.unique(market_regimes)
        regime_features = np.zeros((len(X), len(unique_regimes)))
        
        for i, regime in enumerate(unique_regimes):
            regime_features[:, i] = (market_regimes == regime).astype(float)
        
        # Concatenate with original features
        X_enhanced = np.hstack((X, regime_features))
        
        logger.info(f"Added {len(unique_regimes)} market regime features")
        return X_enhanced
    
    def detect_market_regimes(self, price_data: np.ndarray, returns: np.ndarray, 
                            volatility: np.ndarray, window_size: int = 20, 
                            n_regimes: int = 3) -> np.ndarray:
        """
        Detect market regimes using unsupervised clustering
        
        Args:
            price_data: Array of price data
            returns: Array of returns
            volatility: Array of volatility measures
            window_size: Size of rolling window
            n_regimes: Number of market regimes to detect
            
        Returns:
            Array of market regime labels
        """
        if len(price_data) < window_size:
            logger.warning("Price data too short for regime detection")
            return np.zeros(len(price_data))
        
        # Create features for clustering
        features = np.zeros((len(price_data) - window_size + 1, 3))
        
        for i in range(len(features)):
            # Volatility (std of returns)
            features[i, 0] = np.std(returns[i:i+window_size])
            # Momentum (mean of returns)
            features[i, 1] = np.mean(returns[i:i+window_size])
            # Price trend
            features[i, 2] = price_data[i+window_size-1] / price_data[i] - 1
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_regimes, random_state=self.random_state)
        regimes = kmeans.fit_predict(features_scaled)
        
        # Expand regime labels to match original data length
        full_regimes = np.zeros(len(price_data), dtype=int)
        full_regimes[:window_size-1] = regimes[0]  # Fill beginning
        full_regimes[window_size-1:] = regimes
        
        logger.info(f"Detected {n_regimes} market regimes")
        return full_regimes
        
    def generate_adversarial_examples(self, X: np.ndarray, y: np.ndarray, model,
                                    epsilon: float = 0.05, n_samples: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate adversarial examples that preserve the input pattern but change the model prediction
        
        Args:
            X: Original feature data
            y: Original target data
            model: Model object with predict_proba method
            epsilon: Maximum perturbation size
            n_samples: Number of adversarial samples to generate
            
        Returns:
            Tuple of adversarial features and targets
        """
        logger.info(f"Generating {n_samples} adversarial examples with epsilon={epsilon}")
        
        if not hasattr(model, 'predict_proba'):
            logger.error("Model must have predict_proba method for adversarial generation")
            return np.empty((0, X.shape[1])), np.empty(0)
            
        # Sample original data points to modify
        if len(X) <= n_samples:
            indices = np.arange(len(X))
        else:
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            
        X_base = X[indices].copy()
        y_base = y[indices].copy()
        
        # Get initial predictions
        y_prob = model.predict_proba(X_base)
        
        # Initialize adversarial examples
        X_adv = X_base.copy()
        
        # Calculate feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            # Use a uniform distribution if importance is not available
            importance = np.ones(X.shape[1])
            
        # Normalize importance
        importance = importance / np.sum(importance)
        
        # Multiply by sign to increase or decrease depending on target
        sign = np.ones(len(X_base))
        sign[y_prob[:, 1] >= 0.5] = -1  # Flip high confidence predictions
        
        # Apply targeted perturbation
        for i in range(len(X_base)):
            # Calculate perturbation magnitude inversely proportional to feature importance
            perturbation = sign[i] * epsilon * np.random.uniform(0.5, 1.0, X.shape[1]) * importance
            X_adv[i] += perturbation * np.std(X, axis=0)
        
        logger.info(f"Generated {len(X_adv)} adversarial examples")
        return X_adv, y_base
        
    def time_series_perturbation(self, X: np.ndarray, y: np.ndarray, 
                               sequence_length: int = 60, 
                               n_samples: int = 100,
                               perturbation_type: str = 'mixed') -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply time series specific perturbations that preserve patterns while introducing realistic variations
        
        Args:
            X: Original feature data (structured as sequences)
            y: Original target data
            sequence_length: Number of time steps per sample (for reshaping)
            n_samples: Number of perturbed samples to generate
            perturbation_type: Type of perturbation ('mixed', 'trend', 'noise', 'temporal')
            
        Returns:
            Tuple of perturbed features and targets
        """
        logger.info(f"Generating {n_samples} time series perturbations of type '{perturbation_type}'")
        
        # Sample original data points to modify
        if len(X) <= n_samples:
            indices = np.random.choice(len(X), size=n_samples, replace=True)
        else:
            indices = np.random.choice(len(X), size=n_samples, replace=False)
            
        X_base = X[indices].copy()
        y_base = y[indices].copy()
        
        # Initialize perturbed data
        X_perturbed = X_base.copy()
        
        # Reshape if X is flat (samples, features) to (samples, sequence_length, features_per_step)
        original_shape = X_base.shape
        is_flat = len(original_shape) == 2
        
        if is_flat:
            features_per_step = X_base.shape[1] // sequence_length
            if features_per_step * sequence_length != X_base.shape[1]:
                logger.warning("Feature dimensions not divisible by sequence length. Using flat perturbation.")
            else:
                X_base = X_base.reshape(X_base.shape[0], sequence_length, features_per_step)
                X_perturbed = X_perturbed.reshape(X_perturbed.shape[0], sequence_length, features_per_step)
        
        # Apply perturbation based on type
        if perturbation_type == 'trend' or perturbation_type == 'mixed':
            # Add small trend changes
            if not is_flat or features_per_step * sequence_length == original_shape[1]:
                for i in range(len(X_base)):
                    for feat in range(X_base.shape[-1]):
                        # Apply small trend modification
                        trend = np.linspace(-0.02, 0.02, sequence_length) * np.random.uniform(-1, 1)
                        X_perturbed[i, :, feat] += trend * np.std(X_base[i, :, feat])
        
        if perturbation_type == 'noise' or perturbation_type == 'mixed':
            # Add colored noise
            noise_level = 0.02
            if not is_flat or features_per_step * sequence_length == original_shape[1]:
                for i in range(len(X_base)):
                    for feat in range(X_base.shape[-1]):
                        # Generate pink noise (1/f noise)
                        noise = np.random.normal(0, noise_level, sequence_length)
                        # Smooth the noise to make it more realistic
                        noise = savgol_filter(noise, min(11, sequence_length - sequence_length % 2 + 1), 3)
                        X_perturbed[i, :, feat] += noise * np.std(X_base[i, :, feat])
        
        if perturbation_type == 'temporal' or perturbation_type == 'mixed':
            # Apply small time warping
            if not is_flat or features_per_step * sequence_length == original_shape[1]:
                for i in range(len(X_base)):
                    for feat in range(X_base.shape[-1]):
                        # Temporal warping: slightly stretch or compress segments randomly
                        orig_seq = X_base[i, :, feat]
                        # Create random indices for warping (keep endpoints fixed)
                        indices = np.sort(np.concatenate([
                            [0],
                            np.random.uniform(0, sequence_length, size=5),
                            [sequence_length-1]
                        ]))
                        indices = np.clip(indices, 0, sequence_length-1)
                        # Target indices with slight random shifts
                        targets = np.sort(np.concatenate([
                            [0], 
                            indices[1:-1] + np.random.uniform(-2, 2, size=5),
                            [sequence_length-1]
                        ]))
                        targets = np.clip(targets, 0, sequence_length-1)
                        # Interpolate
                        warped = np.interp(
                            np.arange(sequence_length),
                            targets,
                            orig_seq[indices.astype(int)]
                        )
                        X_perturbed[i, :, feat] = warped
        
        # Reshape back to original format if needed
        if is_flat:
            X_perturbed = X_perturbed.reshape(original_shape)
        
        logger.info(f"Generated {len(X_perturbed)} perturbed time series samples")
        return X_perturbed, y_base
        
    def weighted_augmentation(self, X: np.ndarray, y: np.ndarray, 
                            methods: List[str], 
                            weights: List[float] = None,
                            n_samples_total: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply multiple augmentation methods with specified weights
        
        Args:
            X: Original feature data
            y: Original target data
            methods: List of augmentation methods to use
            weights: List of weights for each method (normalized internally)
            n_samples_total: Total number of samples to generate
            
        Returns:
            Tuple of augmented features and targets
        """
        if weights is None:
            weights = [1.0] * len(methods)
            
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate number of samples for each method
        n_samples = [int(w * n_samples_total) for w in weights]
        # Ensure we have exactly n_samples_total
        n_samples[-1] = n_samples_total - sum(n_samples[:-1])
        
        logger.info(f"Applying weighted augmentation with methods: {methods}, weights: {weights}")
        
        X_augmented = np.empty((0, X.shape[1]))
        y_augmented = np.empty(0)
        
        # Generate samples for each method
        for method, samples in zip(methods, n_samples):
            if samples <= 0:
                continue
                
            X_new, y_new = np.empty((0, X.shape[1])), np.empty(0)
            
            if method == 'smote':
                X_new, y_new = self.apply_smote(X, y)
            elif method == 'borderline_smote':
                X_new, y_new = self.apply_smote(X, y, method='borderline')
            elif method == 'adasyn':
                X_new, y_new = self.apply_smote(X, y, method='adasyn')
            elif method == 'volatile':
                X_new, y_new = self.generate_synthetic_market_conditions(X, y, condition_type='volatile', n_samples=samples)
            elif method == 'trending_up':
                X_new, y_new = self.generate_synthetic_market_conditions(X, y, condition_type='trending_up', n_samples=samples)
            elif method == 'trending_down':
                X_new, y_new = self.generate_synthetic_market_conditions(X, y, condition_type='trending_down', n_samples=samples)
            elif method == 'sideways':
                X_new, y_new = self.generate_synthetic_market_conditions(X, y, condition_type='sideways', n_samples=samples)
            elif method == 'noise':
                X_new, y_new = self._noise_based_augmentation(X, y, n_samples=samples)
            elif method.startswith('time_series_'):
                perturbation_type = method.split('_')[-1]
                X_new, y_new = self.time_series_perturbation(X, y, perturbation_type=perturbation_type, n_samples=samples)
            else:
                logger.warning(f"Unknown augmentation method: {method}")
                continue
                
            # Keep only the requested number of samples
            if len(X_new) > samples:
                indices = np.random.choice(len(X_new), size=samples, replace=False)
                X_new, y_new = X_new[indices], y_new[indices]
            
            # Add to augmented dataset
            X_augmented = np.vstack((X_augmented, X_new))
            y_augmented = np.append(y_augmented, y_new)
            
        logger.info(f"Generated {len(X_augmented)} samples with weighted augmentation")
        return X_augmented, y_augmented
        
    def generate_backtesting_scenarios(self, X: np.ndarray, y: np.ndarray,
                                     price_data: np.ndarray,
                                     scenario_types: List[str] = ['crash', 'boom', 'volatile', 'sideways'],
                                     n_scenarios: int = 4) -> List[Tuple[np.ndarray, np.ndarray, str]]:
        """
        Generate realistic backtesting scenarios by selecting and modifying historical data
        
        Args:
            X: Feature data
            y: Target data
            price_data: Historical price data corresponding to X and y
            scenario_types: Types of scenarios to generate
            n_scenarios: Number of scenarios per type
            
        Returns:
            List of (X_scenario, y_scenario, scenario_name) tuples
        """
        logger.info(f"Generating {n_scenarios} backtesting scenarios of types: {scenario_types}")
        
        if len(X) != len(price_data):
            logger.error("Price data length does not match feature data length")
            return []
            
        scenarios = []
        window_size = 60  # Default window size for scenarios
        
        for scenario_type in scenario_types:
            for i in range(n_scenarios):
                scenario_name = f"{scenario_type}_scenario_{i+1}"
                
                if scenario_type == 'crash':
                    # Find periods of significant price drops
                    returns = np.diff(price_data) / price_data[:-1]
                    rolling_return = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
                    crash_indices = np.argsort(rolling_return)[:10]  # Top 10 worst periods
                    
                    if len(crash_indices) == 0:
                        logger.warning(f"No crash periods found for {scenario_name}")
                        continue
                        
                    # Select a random crash period
                    start_idx = np.random.choice(crash_indices)
                    end_idx = min(start_idx + window_size*2, len(X))
                    
                    X_scenario = X[start_idx:end_idx].copy()
                    y_scenario = y[start_idx:end_idx].copy()
                    
                    # Further amplify the crash characteristics
                    if hasattr(self, '_find_condition_indices'):
                        # Find trend columns and amplify downward movement
                        trend_cols = []
                        for j in range(X.shape[1]):
                            if self._check_feature_type(j, 'trend'):
                                trend_cols.append(j)
                                
                        if trend_cols:
                            for col in trend_cols:
                                # Amplify negative trends
                                mask = X_scenario[:, col] < 0
                                X_scenario[mask, col] *= 1.2  # 20% worse
                
                elif scenario_type == 'boom':
                    # Find periods of significant price increases
                    returns = np.diff(price_data) / price_data[:-1]
                    rolling_return = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
                    boom_indices = np.argsort(rolling_return)[-10:]  # Top 10 best periods
                    
                    if len(boom_indices) == 0:
                        logger.warning(f"No boom periods found for {scenario_name}")
                        continue
                        
                    # Select a random boom period
                    start_idx = np.random.choice(boom_indices)
                    end_idx = min(start_idx + window_size*2, len(X))
                    
                    X_scenario = X[start_idx:end_idx].copy()
                    y_scenario = y[start_idx:end_idx].copy()
                    
                    # Further amplify the boom characteristics
                    if hasattr(self, '_find_condition_indices'):
                        # Find trend columns and amplify upward movement
                        trend_cols = []
                        for j in range(X.shape[1]):
                            if self._check_feature_type(j, 'trend'):
                                trend_cols.append(j)
                                
                        if trend_cols:
                            for col in trend_cols:
                                # Amplify positive trends
                                mask = X_scenario[:, col] > 0
                                X_scenario[mask, col] *= 1.2  # 20% better
                
                elif scenario_type == 'volatile':
                    # Generate volatile market conditions
                    X_scenario, y_scenario = self.generate_synthetic_market_conditions(
                        X, y, condition_type='volatile', n_samples=window_size*2
                    )
                    
                elif scenario_type == 'sideways':
                    # Generate sideways market conditions
                    X_scenario, y_scenario = self.generate_synthetic_market_conditions(
                        X, y, condition_type='sideways', n_samples=window_size*2
                    )
                
                else:
                    logger.warning(f"Unknown scenario type: {scenario_type}")
                    continue
                
                scenarios.append((X_scenario, y_scenario, scenario_name))
                logger.info(f"Generated {scenario_name} with {len(X_scenario)} samples")
        
        return scenarios

# Dynamic weight optimizer class continues below...
class DynamicWeightOptimizer:
    """
    Dynamic weight optimizer for ensemble models
    Adjusts model weights based on recent performance
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the dynamic weight optimizer
        
        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.weight_history = []
        self.performance_history = []
        self.learning_rate = self.config.get('learning_rate', 0.05)
        self.window_size = self.config.get('window_size', 10)
        self.min_weight = self.config.get('min_weight', 0.05)
        self.max_weight = self.config.get('max_weight', 0.8)
        
        logger.info(f"Initialized DynamicWeightOptimizer with learning_rate={self.learning_rate}")

class EnsembleModel:
    """
    Ensemble model for cryptocurrency price prediction
    Combines multiple models to improve prediction accuracy
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize ensemble model
        
        Args:
            config: Configuration parameters for the ensemble
        """
        self.config = config or {}
        self.ensemble_type = self.config.get('ensemble_type', 'voting')
        self.meta_learner_type = self.config.get('meta_learner_type', 'lightgbm')
        
        # Initialize model containers
        self.models = {}
        self.preprocessing = {}
        self.ensemble = None
        self.meta_learner = None
        self.feature_importances = None
        
        # Initialize feature names - default to empty list if not provided during training
        self.feature_names = []
        
        # Configure logger
        self.logger = logging.getLogger("ensemble_model")
        
    def train_base_models(self, data_dict: Dict[str, Any], model_configs: Dict[str, Dict], 
                        save_dir: str = None) -> Dict[str, Any]:
        """
        Train all base models in the ensemble
        
        Args:
            data_dict: Dictionary containing training data
            model_configs: Dictionary with model configurations
            save_dir: Directory to save trained models
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("Training base models for ensemble")
        
        # Extract data
        X_train = data_dict.get('X_train')
        y_train = data_dict.get('y_train')
        X_val = data_dict.get('X_val')
        y_val = data_dict.get('y_val')
        
        if X_train is None or y_train is None:
            self.logger.error("Training data not provided")
            return {'error': 'Training data not provided'}
        
        # Dictionary to store results
        results = {}
        
        # Containers for voting ensemble
        estimators = []
        
        # Train Random Forest if specified
        if 'random_forest' in model_configs:
            self.logger.info("Training Random Forest model")
            rf_config = model_configs['random_forest']
            
            rf_model = RandomForestClassifier(
                n_estimators=rf_config.get('n_estimators', 100),
                max_depth=rf_config.get('max_depth', None),
                min_samples_split=rf_config.get('min_samples_split', 2),
                min_samples_leaf=rf_config.get('min_samples_leaf', 1),
                bootstrap=rf_config.get('bootstrap', True),
                random_state=42
            )
            
            rf_model.fit(X_train, y_train)
            
            # Store model
            self.models['random_forest'] = rf_model
            
            # Add to voting ensemble
            weight = rf_config.get('weight', 1.0)
            estimators.append(('random_forest', rf_model, weight))
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = rf_model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                self.logger.info(f"Random Forest validation accuracy: {accuracy:.4f}")
                results['random_forest'] = {'accuracy': accuracy}
            
            # Save model if directory is provided
            if save_dir:
                model_path = os.path.join(save_dir, 'random_forest_model.joblib')
                joblib.dump(rf_model, model_path)
                self.logger.info(f"Random Forest model saved to {model_path}")
        
        # Train XGBoost if specified
        if 'xgboost' in model_configs:
            self.logger.info("Training XGBoost model")
            xgb_config = model_configs['xgboost']
            
            xgb_model = xgb.XGBClassifier(
                n_estimators=xgb_config.get('n_estimators', 100),
                max_depth=xgb_config.get('max_depth', 6),
                learning_rate=xgb_config.get('learning_rate', 0.3),
                subsample=xgb_config.get('subsample', 0.8),
                colsample_bytree=xgb_config.get('colsample_bytree', 0.8),
                gamma=xgb_config.get('gamma', 0),
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            xgb_model.fit(X_train, y_train)
            
            # Store model
            self.models['xgboost'] = xgb_model
            
            # Add to voting ensemble
            weight = xgb_config.get('weight', 1.0)
            estimators.append(('xgboost', xgb_model, weight))
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = xgb_model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                self.logger.info(f"XGBoost validation accuracy: {accuracy:.4f}")
                results['xgboost'] = {'accuracy': accuracy}
            
            # Save model if directory is provided
            if save_dir:
                model_path = os.path.join(save_dir, 'xgboost_model.joblib')
                joblib.dump(xgb_model, model_path)
                self.logger.info(f"XGBoost model saved to {model_path}")
        
        # Train LightGBM if specified
        if 'lightgbm' in model_configs:
            self.logger.info("Training LightGBM model")
            lgb_config = model_configs['lightgbm']
            
            lgb_model = LGBMClassifier(
                n_estimators=lgb_config.get('n_estimators', 100),
                max_depth=lgb_config.get('max_depth', -1),
                learning_rate=lgb_config.get('learning_rate', 0.3),
                subsample=lgb_config.get('subsample', 0.8),
                colsample_bytree=lgb_config.get('colsample_bytree', 0.8),
                reg_alpha=lgb_config.get('reg_alpha', 0.0),
                reg_lambda=lgb_config.get('reg_lambda', 0.0),
                random_state=42
            )
            
            lgb_model.fit(X_train, y_train)
            
            # Store model
            self.models['lightgbm'] = lgb_model
            
            # Add to voting ensemble
            weight = lgb_config.get('weight', 1.0)
            estimators.append(('lightgbm', lgb_model, weight))
            
            # Evaluate on validation set if provided
            if X_val is not None and y_val is not None:
                y_pred = lgb_model.predict(X_val)
                accuracy = accuracy_score(y_val, y_pred)
                self.logger.info(f"LightGBM validation accuracy: {accuracy:.4f}")
                results['lightgbm'] = {'accuracy': accuracy}
            
            # Save model if directory is provided
            if save_dir:
                model_path = os.path.join(save_dir, 'lightgbm_model.joblib')
                joblib.dump(lgb_model, model_path)
                self.logger.info(f"LightGBM model saved to {model_path}")
        
        # Create and train ensemble
        if len(estimators) > 1:
            self.logger.info(f"Creating {self.ensemble_type} ensemble with {len(estimators)} base models")
            
            if self.ensemble_type == 'voting':
                # Create voting classifier
                self.ensemble = VotingClassifier(
                    estimators=[(name, clf) for name, clf, _ in estimators],
                    voting='soft',
                    weights=[weight for _, _, weight in estimators]
                )
                
                # Fit ensemble
                self.ensemble.fit(X_train, y_train)
                
                # Evaluate ensemble
                if X_val is not None and y_val is not None:
                    y_pred = self.ensemble.predict(X_val)
                    ensemble_accuracy = accuracy_score(y_val, y_pred)
                    self.logger.info(f"Ensemble validation accuracy: {ensemble_accuracy:.4f}")
                    results['ensemble'] = {'accuracy': ensemble_accuracy}
                
                # Save ensemble if directory is provided
                if save_dir:
                    ensemble_path = os.path.join(save_dir, 'voting_classifier.joblib')
                    joblib.dump(self.ensemble, ensemble_path)
                    self.logger.info(f"Voting ensemble saved to {ensemble_path}")
                    
            elif self.ensemble_type == 'stacking':
                # Stacking implementation can be added here
                self.logger.info("Stacking ensemble is not implemented yet")
                
        # Extract feature importances if available
        self._extract_feature_importances()
        
        # Save metadata if directory is provided
        if save_dir:
            metadata = {
                'ensemble_type': self.ensemble_type,
                'models': list(self.models.keys()),
                'trained_at': datetime.now().isoformat(),
                'performance': results
            }
            
            metadata_path = os.path.join(save_dir, 'ensemble_metadata.joblib')
            joblib.dump(metadata, metadata_path)
            self.logger.info(f"Ensemble metadata saved to {metadata_path}")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the ensemble

        Args:
            X: Input features as a NumPy array

        Returns:
            Array of predicted probabilities for positive class
        """
        if not isinstance(X, pd.DataFrame):
            # Convert to DataFrame and ensure feature names match those used during training
            X = pd.DataFrame(X, columns=self.feature_names)

        if self.ensemble is None:
            self.logger.error("Ensemble model is not initialized")
            return np.array([])

        try:
            # Get probability predictions using ensemble
            y_prob = self.ensemble.predict_proba(X)
            return y_prob[:, 1]  # Return probabilities for the positive class
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return np.array([])
    
    def predict_binary(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the ensemble
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of binary predictions (0 or 1)
        """
        if self.ensemble is None:
            self.logger.error("Ensemble model has not been trained")
            return np.zeros(len(X))
        
        try:
            return self.ensemble.predict(X)
        except Exception as e:
            self.logger.error(f"Error during binary prediction: {e}")
            return np.zeros(len(X))
    
    def evaluate(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the ensemble model
        
        Args:
            X: Feature matrix
            y_true: Ground truth labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.ensemble is None:
            self.logger.error("Ensemble model has not been trained")
            return {'error': 'Model not trained'}
        
        try:
            # Get predictions
            y_pred = self.predict_binary(X)
            y_proba = self.predict(X)
            
            # Calculate metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            # Calculate AUC only if we have probabilities and both classes are present
            if len(np.unique(y_true)) > 1 and hasattr(self.ensemble, 'predict_proba'):
                auc = roc_auc_score(y_true, y_proba)
            else:
                auc = np.nan
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc
            }
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            return {'error': str(e)}
    
    def evaluate_individual_models(self, X: np.ndarray, y_true: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Evaluate each model in the ensemble individually
        
        Args:
            X: Feature matrix
            y_true: Ground truth labels
            
        Returns:
            Dictionary with evaluation metrics for each model
        """
        results = {}
        
        for name, model in self.models.items():
            try:
                # Get binary predictions
                y_pred = model.predict(X)
                
                # Calculate metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Calculate AUC if predict_proba is available
                if hasattr(model, 'predict_proba') and len(np.unique(y_true)) > 1:
                    y_proba = model.predict_proba(X)[:, 1]
                    auc = roc_auc_score(y_true, y_proba)
                else:
                    auc = np.nan
                
                results[name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc': auc
                }
            except Exception as e:
                self.logger.error(f"Error evaluating {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def save(self, save_dir: str) -> bool:
        """
        Save the ensemble model and its components
        
        Args:
            save_dir: Directory to save models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save individual models
            for name, model in self.models.items():
                model_path = os.path.join(save_dir, f'{name}_model.joblib')
                joblib.dump(model, model_path)
            
            # Save ensemble if available
            if self.ensemble is not None:
                ensemble_name = self.ensemble_type + '_classifier.joblib'
                ensemble_path = os.path.join(save_dir, ensemble_name)
                joblib.dump(self.ensemble, ensemble_path)
            
            # Save metadata
            metadata = {
                'ensemble_type': self.ensemble_type,
                'models': list(self.models.keys()),
                'saved_at': datetime.now().isoformat(),
                'feature_importances': self.feature_importances
            }
            
            metadata_path = os.path.join(save_dir, 'ensemble_metadata.joblib')
            joblib.dump(metadata, metadata_path)
            
            self.logger.info(f"Ensemble model saved to {save_dir}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving ensemble model: {e}")
            return False
    
    def load(self, load_dir: str) -> bool:
        """
        Load ensemble model from directory
        
        Args:
            load_dir: Directory containing saved models
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if directory exists
            if not os.path.exists(load_dir):
                self.logger.error(f"Model directory does not exist: {load_dir}")
                return False
                
            # Load metadata
            metadata_path = os.path.join(load_dir, 'ensemble_metadata.joblib')
            if os.path.exists(metadata_path):
                metadata = joblib.load(metadata_path)
                self.ensemble_type = metadata.get('ensemble_type', 'voting')
                
                # Load individual models
                model_names = metadata.get('models', [])
                for name in model_names:
                    model_path = os.path.join(load_dir, f'{name}_model.joblib')
                    if os.path.exists(model_path):
                        self.models[name] = joblib.load(model_path)
                        self.logger.info(f"Loaded base model: {name}")
                    else:
                        self.logger.warning(f"Model file not found: {model_path}")
                
                # Load ensemble
                ensemble_name = self.ensemble_type + '_classifier.joblib'
                ensemble_path = os.path.join(load_dir, ensemble_name)
                if os.path.exists(ensemble_path):
                    self.ensemble = joblib.load(ensemble_path)
                    # Verify the ensemble model is properly loaded
                    if not hasattr(self.ensemble, 'estimators_'):
                        self.logger.warning("Loaded ensemble doesn't have estimators_ attribute. Attempting to reconstruct...")
                        # Try to reconstruct from individual models
                        if len(self.models) > 1:
                            from sklearn.ensemble import VotingClassifier
                            estimators = [(name, model) for name, model in self.models.items()]
                            self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
                            self.logger.info(f"Reconstructed ensemble with {len(estimators)} models")
                    else:
                        self.logger.info(f"Loaded ensemble with {len(self.ensemble.estimators_)} models")
                else:
                    self.logger.warning(f"Ensemble file not found: {ensemble_path}")
                    # Try to reconstruct from individual models
                    if len(self.models) > 1:
                        from sklearn.ensemble import VotingClassifier
                        estimators = [(name, model) for name, model in self.models.items()]
                        self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
                        self.logger.info(f"Reconstructed ensemble with {len(estimators)} models")
                
                # Load feature importances
                self.feature_importances = metadata.get('feature_importances')
                
                # Load or infer feature names (needed for prediction)
                if 'feature_names' in metadata:
                    self.feature_names = metadata.get('feature_names')
                    self.logger.info(f"Loaded {len(self.feature_names)} feature names from metadata")
                else:
                    # Try to extract feature names from a model with feature_names_ attribute
                    for _, model in self.models.items():
                        if hasattr(model, 'feature_names_'):
                            self.feature_names = model.feature_names_
                            self.logger.info(f"Extracted {len(self.feature_names)} feature names from model")
                            break
                    
                    # If no feature names found, create generic ones based on the first model's input dimension
                    if not self.feature_names:
                        for name, model in self.models.items():
                            if hasattr(model, 'n_features_in_'):
                                n_features = model.n_features_in_
                                self.feature_names = [f'feature_{i}' for i in range(n_features)]
                                self.logger.info(f"Created {len(self.feature_names)} generic feature names")
                                break
                
                # Final verification
                if self.ensemble is None:
                    self.logger.error("Failed to load or reconstruct ensemble model")
                    return False
                    
                self.logger.info(f"Ensemble model loaded from {load_dir}")
                return True
            else:
                self.logger.error(f"Metadata file not found in {load_dir}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error loading ensemble model: {e}")
            return False
    
    def _extract_feature_importances(self):
        """Extract feature importances from base models"""
        feature_importances = {}
        
        # Extract importances from models that support it
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                feature_importances[name] = model.feature_importances_
        
        self.feature_importances = feature_importances
    
    def plot_feature_importance(self, feature_names=None, top_n=20, save_path=None):
        """
        Plot feature importances
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to display
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure with feature importance plots
        """
        if not self.feature_importances:
            self.logger.error("No feature importances available")
            return None
        
        n_models = len(self.feature_importances)
        if n_models == 0:
            return None
        
        # Create figure
        fig, axes = plt.subplots(n_models, 1, figsize=(10, 5*n_models), squeeze=False)
        
        for i, (name, importances) in enumerate(self.feature_importances.items()):
            ax = axes[i, 0]
            
            # Sort importances
            if feature_names is None:
                feature_names = [f"Feature {j}" for j in range(len(importances))]
            
            # Sort and get top_n features
            indices = np.argsort(importances)[::-1][:top_n]
            top_importances = importances[indices]
            top_names = [feature_names[j] for j in indices]
            
            # Plot
            ax.barh(range(len(top_importances)), top_importances, align='center')
            ax.set_yticks(range(len(top_importances)))
            ax.set_yticklabels(top_names)
            ax.set_title(f"{name} Feature Importance")
            ax.set_xlabel('Importance')
        
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path)
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig

class RegimeSpecificEnsemble:
    """Train separate models for each market regime to improve prediction accuracy"""
    
    def __init__(self, base_models=None, model_configs=None, n_regimes=4, regime_feature='market_regime_cluster'):
        """
        Initialize the regime-specific ensemble
        
        Args:
            base_models: Dictionary of base models to use (optional)
            model_configs: Configurations for models
            n_regimes: Number of regimes to model separately
            regime_feature: Feature name that identifies the market regime
        """
        self.base_models = base_models or {}
        self.model_configs = model_configs or {}
        self.n_regimes = n_regimes
        self.regime_feature = regime_feature
        self.regime_models = {}
        self.scaler = None
        self.feature_selectors = {}
        self.logger = logging.getLogger('RegimeSpecificEnsemble')
        
    def train(self, X, y, regimes, feature_names=None):
        """
        Train specialized models for each regime
        
        Args:
            X: Features for training
            y: Target labels
            regimes: Array of regime labels
            feature_names: List of feature names for interpretability
        """
        self.logger.info("Training regime-specific ensemble models")
        unique_regimes = np.unique(regimes)
        self.feature_names = feature_names
        
        # Ensure we have regimes
        if len(unique_regimes) <= 1:
            self.logger.warning(f"Only one regime detected: {unique_regimes[0]}. Falling back to standard ensemble.")
            # Create a single ensemble model for all data
            ensemble = EnsembleModel(self.model_configs)
            ensemble.train(X, y)
            self.regime_models[-1] = ensemble  # Use -1 as key for the general model
            return self
            
        # Train a specialized model for each regime
        for regime in unique_regimes:
            mask = regimes == regime
            
            # Skip if too few samples
            if np.sum(mask) < 100:
                self.logger.warning(f"Insufficient data for regime {regime} (only {np.sum(mask)} samples), skipping")
                continue
                
            self.logger.info(f"Training model for regime {regime} with {np.sum(mask)} samples")
            
            # Extract data for this regime
            X_regime = X[mask]
            y_regime = y[mask]
            
            # Customize hyperparameters for this regime
            regime_configs = self._customize_configs_for_regime(regime)
            
            # Create and train ensemble for this regime
            ensemble = EnsembleModel(regime_configs)
            ensemble.train(X_regime, y_regime, feature_names=feature_names)
            
            # Store the model
            self.regime_models[regime] = ensemble
            
        # Train a fallback model on all data for regimes without enough data
        if len(self.regime_models) < len(unique_regimes):
            self.logger.info("Training fallback model for unseen regimes")
            ensemble = EnsembleModel(self.model_configs)
            ensemble.train(X, y, feature_names=feature_names)
            self.regime_models[-1] = ensemble  # Use -1 as key for the general model
            
        return self
    
    def _customize_configs_for_regime(self, regime):
        """
        Adjust hyperparameters based on regime characteristics
        
        Args:
            regime: The regime identifier
            
        Returns:
            Dictionary of configurations customized for the regime
        """
        regime_configs = self.model_configs.copy() if self.model_configs else {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 42
            },
            'xgboost': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'lightgbm': {
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            }
        }
        
        # High volatility regime adjustments
        if regime in [2, 3]:  # Assuming regimes 2,3 are high volatility
            self.logger.info(f"Customizing for high volatility regime {regime}")
            if 'random_forest' in regime_configs:
                regime_configs['random_forest']['max_depth'] = min(regime_configs['random_forest'].get('max_depth', 10), 6)
                regime_configs['random_forest']['min_samples_split'] = 8
                regime_configs['random_forest']['n_estimators'] = 150
                
            if 'xgboost' in regime_configs:
                regime_configs['xgboost']['max_depth'] = min(regime_configs['xgboost'].get('max_depth', 6), 4)
                regime_configs['xgboost']['subsample'] = 0.7
                regime_configs['xgboost']['learning_rate'] = 0.05
                
            if 'lightgbm' in regime_configs:
                regime_configs['lightgbm']['max_depth'] = min(regime_configs['lightgbm'].get('max_depth', 6), 4)
                regime_configs['lightgbm']['subsample'] = 0.7
                regime_configs['lightgbm']['learning_rate'] = 0.05
                regime_configs['lightgbm']['reg_lambda'] = 3.0
                
        # Trending regime adjustments
        elif regime in [0, 1]:  # Assuming regimes 0,1 are trending
            self.logger.info(f"Customizing for trending regime {regime}")
            if 'random_forest' in regime_configs:
                regime_configs['random_forest']['n_estimators'] = 120
                
            if 'xgboost' in regime_configs:
                regime_configs['xgboost']['learning_rate'] = 0.08
                regime_configs['xgboost']['gamma'] = 0.1
                
            if 'lightgbm' in regime_configs:
                regime_configs['lightgbm']['num_leaves'] = 31
                regime_configs['lightgbm']['learning_rate'] = 0.08
                
        return regime_configs
    
    def predict(self, X, current_regime=None):
        """
        Make predictions using the appropriate regime model
        
        Args:
            X: Features for prediction
            current_regime: Current market regime if known
            
        Returns:
            Array of predictions
        """
        if current_regime is not None and current_regime in self.regime_models:
            # Use the regime-specific model if available
            return self.regime_models[current_regime].predict(X)
        else:
            # If regime is unknown or no specific model exists
            if -1 in self.regime_models:
                # Use the general model if available
                return self.regime_models[-1].predict(X)
            else:
                # Ensemble predictions from all models
                predictions = []
                weights = []
                
                for regime, model in self.regime_models.items():
                    pred = model.predict(X)
                    predictions.append(pred)
                    weights.append(1.0)  # Equal weight by default
                
                # Weight by model performance if metrics are available
                if hasattr(self, 'model_metrics') and self.model_metrics:
                    weights = [self.model_metrics.get(regime, {}).get('f1_score', 1.0) 
                              for regime in self.regime_models.keys()]
                
                # Normalize weights
                weights = np.array(weights) / sum(weights)
                
                # Weighted average of predictions
                return np.average(np.array(predictions), axis=0, weights=weights)
    
    def predict_proba(self, X, current_regime=None):
        """
        Make probability predictions using the appropriate regime model
        
        Args:
            X: Features for prediction
            current_regime: Current market regime if known
            
        Returns:
            Array of probability predictions
        """
        if current_regime is not None and current_regime in self.regime_models:
            # Use the regime-specific model if available
            return self.regime_models[current_regime].predict_proba(X)
        else:
            # Same logic as predict() but for probabilities
            if -1 in self.regime_models:
                return self.regime_models[-1].predict_proba(X)
            else:
                proba_predictions = []
                weights = []
                
                for regime, model in self.regime_models.items():
                    proba = model.predict_proba(X)
                    proba_predictions.append(proba)
                    weights.append(1.0)
                
                # Weight by model performance if metrics are available
                if hasattr(self, 'model_metrics') and self.model_metrics:
                    weights = [self.model_metrics.get(regime, {}).get('f1_score', 1.0) 
                              for regime in self.regime_models.keys()]
                
                # Normalize weights
                weights = np.array(weights) / sum(weights)
                
                # Weighted average of probability predictions
                return np.average(np.array(proba_predictions), axis=0, weights=weights)
    
    def evaluate(self, X, y, regimes):
        """
        Evaluate model performance across all regimes
        
        Args:
            X: Features
            y: Target labels
            regimes: Regime labels
            
        Returns:
            Dictionary of metrics by regime
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        unique_regimes = np.unique(regimes)
        metrics = {}
        
        # Overall performance
        y_pred = self.predict(X)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        overall_metrics = {
            'accuracy': accuracy_score(y, y_pred_binary),
            'precision': precision_score(y, y_pred_binary),
            'recall': recall_score(y, y_pred_binary),
            'f1_score': f1_score(y, y_pred_binary),
            'roc_auc': roc_auc_score(y, y_pred)
        }
        metrics['overall'] = overall_metrics
        
        # Per-regime performance
        for regime in unique_regimes:
            mask = regimes == regime
            if np.sum(mask) > 20:  # Need enough samples for meaningful evaluation
                X_regime = X[mask]
                y_regime = y[mask]
                
                # Get predictions for this regime
                y_pred_regime = self.predict(X_regime, current_regime=regime)
                y_pred_regime_binary = (y_pred_regime > 0.5).astype(int)
                
                regime_metrics = {
                    'accuracy': accuracy_score(y_regime, y_pred_regime_binary),
                    'precision': precision_score(y_regime, y_pred_regime_binary),
                    'recall': recall_score(y_regime, y_pred_regime_binary),
                    'f1_score': f1_score(y_regime, y_pred_regime_binary),
                    'roc_auc': roc_auc_score(y_regime, y_pred_regime)
                }
                metrics[regime] = regime_metrics
        
        self.model_metrics = metrics
        return metrics
    
    def save(self, path):
        """Save the ensemble to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Serialize the model
        joblib.dump(self, path)
        self.logger.info(f"RegimeSpecificEnsemble saved to {path}")
        
    @classmethod
    def load(cls, path):
        """Load the ensemble from disk"""
        model = joblib.load(path)
        return model