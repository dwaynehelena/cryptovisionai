#!/usr/bin/env python3
"""
Sentiment Feature Weight Optimizer - Optimizes the weights of individual sentiment features
based on their historical predictive power for cryptocurrency price movements.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentWeightOptimizer:
    """
    Optimizes weights for sentiment features based on their predictive power
    for future price movements using machine learning techniques.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the sentiment feature weight optimizer
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', 30)
        self.forecast_horizon = self.config.get('forecast_horizon', 24)  # hours
        self.alpha = self.config.get('regularization_alpha', 0.1)
        self.optimization_interval = self.config.get('optimization_interval', 24)  # hours
        self.last_optimization = None
        self.feature_importances = {}
        self.model = None
        self.scaler = StandardScaler()
        
    def optimize_weights(self, 
                         sentiment_features: pd.DataFrame, 
                         price_data: pd.DataFrame,
                         force_update: bool = False) -> pd.DataFrame:
        """
        Optimize weights for different sentiment features based on their ability
        to predict future price movements
        
        Args:
            sentiment_features (pd.DataFrame): DataFrame containing sentiment-based features
            price_data (pd.DataFrame): OHLCV price data
            force_update (bool): Force update even if optimization interval hasn't passed
            
        Returns:
            pd.DataFrame: DataFrame with optimized weighted features
        """
        current_time = datetime.now()
        
        # Check if we need to update weights
        if (not force_update and self.last_optimization and 
            (current_time - self.last_optimization) < timedelta(hours=self.optimization_interval)):
            logger.info("Using cached sentiment feature weights")
            
            if self.model is not None and len(self.feature_importances) > 0:
                # Apply existing model
                return self._apply_optimized_weights(sentiment_features)
            
        logger.info("Optimizing sentiment feature weights")
        
        # Calculate target variable: future returns
        if 'close' not in price_data.columns:
            logger.error("Price data must contain 'close' column")
            return sentiment_features
            
        try:
            # Create target variable - future price change
            target = price_data['close'].pct_change(self.forecast_horizon).shift(-self.forecast_horizon)
            
            # Align data
            common_index = sentiment_features.index.intersection(target.index)
            if len(common_index) < max(30, self.forecast_horizon * 2):
                logger.warning("Not enough overlapping data points for optimization")
                return sentiment_features
                
            X = sentiment_features.loc[common_index].copy()
            y = target.loc[common_index].copy()
            
            # Drop rows with NaN
            valid_mask = ~(X.isna().any(axis=1) | y.isna())
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 30:
                logger.warning("Not enough valid data points after removing NaNs")
                return sentiment_features
                
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                index=X.index, 
                columns=X.columns
            )
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)
            best_model = None
            best_score = float('-inf')
            
            # Try different regularization strengths
            for alpha in [0.001, 0.01, 0.1, 1.0]:
                # Try both Ridge and Lasso
                for model_class in [Ridge, Lasso]:
                    model = model_class(alpha=alpha, random_state=42)
                    
                    # Cross-validation
                    scores = []
                    for train_idx, test_idx in tscv.split(X_scaled):
                        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # Fit
                        model.fit(X_train, y_train)
                        
                        # Score - use directional accuracy instead of RMSE
                        y_pred = model.predict(X_test)
                        directional_accuracy = np.mean((y_test > 0) == (y_pred > 0))
                        scores.append(directional_accuracy)
                    
                    avg_score = np.mean(scores)
                    
                    if avg_score > best_score:
                        best_score = avg_score
                        best_model = model_class(alpha=alpha, random_state=42)
            
            # Train final model on full dataset
            if best_model is not None:
                best_model.fit(X_scaled, y)
                self.model = best_model
                
                # Extract feature importances
                self.feature_importances = {
                    feature: abs(coef)
                    for feature, coef in zip(X.columns, best_model.coef_)
                }
                
                # Normalize to sum to 1.0
                total_importance = sum(self.feature_importances.values())
                if total_importance > 0:
                    self.feature_importances = {
                        k: v / total_importance 
                        for k, v in self.feature_importances.items()
                    }
                    
                logger.info(f"Top 5 important sentiment features: {sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]}")
                
                # Update timestamp
                self.last_optimization = current_time
                
                # Apply optimized weights
                return self._apply_optimized_weights(sentiment_features)
            else:
                logger.warning("Failed to find a good model for sentiment feature optimization")
                return sentiment_features
                
        except Exception as e:
            logger.error(f"Error optimizing sentiment feature weights: {str(e)}")
            logger.exception("Optimization error details")
            return sentiment_features
    
    def _apply_optimized_weights(self, sentiment_features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply optimized weights to the sentiment features
        
        Args:
            sentiment_features (pd.DataFrame): Original sentiment features
            
        Returns:
            pd.DataFrame: DataFrame with weighted features
        """
        if not self.feature_importances or len(self.feature_importances) == 0:
            return sentiment_features
            
        result = pd.DataFrame(index=sentiment_features.index)
        
        try:
            # Apply weights to each feature
            for feature, weight in self.feature_importances.items():
                if feature in sentiment_features.columns:
                    result[f'weighted_{feature}'] = sentiment_features[feature] * weight
                    
            # Create combined weighted sentiment score
            features_to_combine = [f for f in sentiment_features.columns 
                                  if f in self.feature_importances]
            
            if features_to_combine:
                # Calculate weighted sum
                result['combined_weighted_sentiment'] = sum(
                    sentiment_features[feature] * self.feature_importances.get(feature, 0)
                    for feature in features_to_combine
                )
                
                # Add smoothed version
                result['combined_weighted_sentiment_smooth'] = result['combined_weighted_sentiment'].ewm(span=12).mean()
                
                # Add momentum of weighted sentiment
                for period in [6, 12, 24]:
                    result[f'weighted_sentiment_momentum_{period}h'] = result['combined_weighted_sentiment'].diff(period)
                    
        except Exception as e:
            logger.error(f"Error applying optimized weights: {str(e)}")
            
        return result
    
    def get_feature_importances(self) -> Dict[str, float]:
        """
        Get the current feature importance weights
        
        Returns:
            Dict[str, float]: Dictionary mapping feature names to their importance weights
        """
        return self.feature_importances
        
    def detect_regime_change(self, 
                            sentiment_features: pd.DataFrame, 
                            price_data: pd.DataFrame, 
                            window: int = 72) -> bool:
        """
        Detect if there's been a significant change in the relationship between
        sentiment and price movement (regime change)
        
        Args:
            sentiment_features (pd.DataFrame): Sentiment features
            price_data (pd.DataFrame): Price data
            window (int): Window size for analysis
            
        Returns:
            bool: True if regime change detected, False otherwise
        """
        if len(sentiment_features) < window * 2 or 'close' not in price_data.columns:
            return False
            
        try:
            # Get most important feature
            if not self.feature_importances:
                most_imp_feature = sentiment_features.columns[0]  # default to first
            else:
                most_imp_feature = max(self.feature_importances.items(), key=lambda x: x[1])[0]
                
            if most_imp_feature not in sentiment_features.columns:
                most_imp_feature = sentiment_features.columns[0]
                
            # Calculate rolling correlation
            sentiment_series = sentiment_features[most_imp_feature]
            price_returns = price_data['close'].pct_change()
            
            # Align
            common_index = sentiment_series.index.intersection(price_returns.index)
            sentiment_aligned = sentiment_series.loc[common_index]
            returns_aligned = price_returns.loc[common_index]
            
            # Calculate rolling correlations
            rolling_corr = sentiment_aligned.rolling(window=window).corr(returns_aligned)
            
            # Detect significant changes in correlation
            if len(rolling_corr) >= window * 2:
                prev_period = rolling_corr.iloc[-window*2:-window].mean()
                current_period = rolling_corr.iloc[-window:].mean()
                
                # Check if correlation changed by more than 0.3 or sign flipped
                if abs(prev_period - current_period) > 0.3 or prev_period * current_period < 0:
                    logger.info(f"Sentiment regime change detected: correlation changed from {prev_period:.3f} to {current_period:.3f}")
                    return True
                    
        except Exception as e:
            logger.warning(f"Error in regime change detection: {str(e)}")
            
        return False