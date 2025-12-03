#!/usr/bin/env python3
"""
Sentiment Weight Optimizer - Dynamically adjusts weights for different sentiment sources
based on their historical correlation with price movements
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class SentimentWeightOptimizer:
    """
    Dynamically optimize weights for different sentiment sources based on their
    historical predictive power for price movements.
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize the sentiment weight optimizer
        
        Args:
            config (Dict): Configuration dictionary
        """
        self.config = config or {}
        self.lookback_days = self.config.get('lookback_days', 30)
        self.prediction_horizon = self.config.get('prediction_horizon', 24)  # hours
        self.source_weights = {
            'news': 0.33,
            'twitter': 0.33,
            'reddit': 0.34
        }
        self.last_optimization = None
        self.optimization_interval = self.config.get('optimization_interval', 7)  # days
        
    def optimize_weights(self, 
                        sentiment_data: Dict[str, pd.DataFrame], 
                        price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Optimize weights for different sentiment sources based on their correlation
        with future price movements
        
        Args:
            sentiment_data (Dict[str, pd.DataFrame]): Dictionary of sentiment DataFrames by source
            price_data (pd.DataFrame): Price data with OHLCV columns
            
        Returns:
            Dict[str, float]: Optimized weights for each source
        """
        # Check if we need to update weights
        current_time = datetime.now()
        if (self.last_optimization and 
            (current_time - self.last_optimization).days < self.optimization_interval):
            logger.info("Using cached sentiment weights")
            return self.source_weights
            
        logger.info("Optimizing sentiment source weights")
        
        try:
            # Calculate future returns
            future_returns = price_data['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
            
            # Calculate correlations between each source and future returns
            correlations = {}
            prediction_errors = {}
            
            for source, df in sentiment_data.items():
                if source not in correlations:
                    correlations[source] = 0.0
                    prediction_errors[source] = float('inf')
                
                # Skip if DataFrame is empty
                if df.empty:
                    continue
                
                # Align data
                common_index = df.index.intersection(price_data.index)
                if len(common_index) < 10:  # Need enough data points
                    continue
                    
                aligned_sentiment = df.loc[common_index]
                aligned_price = price_data.loc[common_index]
                aligned_future_returns = future_returns.loc[common_index]
                
                # Drop NaNs
                valid_data = pd.concat([
                    aligned_sentiment['sentiment_score'], 
                    aligned_future_returns
                ], axis=1).dropna()
                
                if len(valid_data) < 10:  # Need enough data points
                    continue
                
                # Calculate correlation
                corr = valid_data['sentiment_score'].corr(valid_data[valid_data.columns[1]])
                correlations[source] = abs(corr)  # Use absolute correlation
                
                # Calculate prediction error using linear regression
                try:
                    X = valid_data[['sentiment_score']].values
                    y = valid_data[valid_data.columns[1]].values
                    
                    model = LinearRegression()
                    model.fit(X, y)
                    predictions = model.predict(X)
                    
                    mse = mean_squared_error(y, predictions)
                    prediction_errors[source] = mse
                except Exception as e:
                    logger.warning(f"Error calculating prediction error for {source}: {e}")
            
            # Calculate weights based on correlations and prediction errors
            total_corr = sum(correlations.values())
            total_inv_error = sum(1/err if err > 0 else 0 for err in prediction_errors.values())
            
            # Combined metric (50% correlation, 50% inverse error)
            weights = {}
            for source in sentiment_data.keys():
                if total_corr > 0 and total_inv_error > 0:
                    corr_weight = correlations.get(source, 0) / total_corr if total_corr > 0 else 0
                    error_weight = (1/prediction_errors.get(source, float('inf'))) / total_inv_error if total_inv_error > 0 else 0
                    weights[source] = 0.5 * corr_weight + 0.5 * error_weight
                else:
                    weights[source] = 1.0 / len(sentiment_data)
            
            # Normalize weights to sum to 1.0
            total_weight = sum(weights.values())
            if total_weight > 0:
                self.source_weights = {k: v/total_weight for k, v in weights.items()}
            
            # Update last optimization time
            self.last_optimization = current_time
            
            logger.info(f"Optimized sentiment weights: {self.source_weights}")
            return self.source_weights
            
        except Exception as e:
            logger.error(f"Error optimizing sentiment weights: {e}")
            return self.source_weights
            
    def detect_sentiment_trend_change(self, 
                                     sentiment_data: pd.DataFrame, 
                                     window: int = 24) -> Tuple[bool, float]:
        """
        Detect significant changes in sentiment trend
        
        Args:
            sentiment_data (pd.DataFrame): Sentiment data
            window (int): Window size for trend analysis
            
        Returns:
            Tuple[bool, float]: (is_trend_change, trend_strength)
        """
        if sentiment_data.empty or len(sentiment_data) < window * 2:
            return False, 0.0
            
        # Calculate the trend slope using rolling regression
        try:
            x = np.array(range(window))
            
            # Initialize arrays
            slopes = np.zeros(len(sentiment_data) - window + 1)
            
            # Calculate rolling slopes
            for i in range(len(sentiment_data) - window + 1):
                y = sentiment_data['sentiment_score'].iloc[i:i+window].values
                slope, _ = np.polyfit(x, y, 1)
                slopes[i] = slope
            
            # Check for trend change (sign change in slope)
            if len(slopes) >= 2:
                previous_slope = slopes[-2]
                current_slope = slopes[-1]
                
                # Trend change if sign changes
                trend_change = (previous_slope * current_slope) < 0
                
                # Calculate trend strength
                trend_strength = abs(current_slope)
                
                return trend_change, trend_strength
            
        except Exception as e:
            logger.warning(f"Error detecting sentiment trend change: {e}")
            
        return False, 0.0
        
    def calculate_sentiment_divergence(self, 
                                      sentiment_data: pd.DataFrame, 
                                      price_data: pd.DataFrame,
                                      window: int = 24) -> pd.DataFrame:
        """
        Calculate divergence between sentiment and price movements
        
        Args:
            sentiment_data (pd.DataFrame): Sentiment data
            price_data (pd.DataFrame): Price data
            window (int): Window for divergence calculation
            
        Returns:
            pd.DataFrame: DataFrame with divergence indicators
        """
        result = pd.DataFrame(index=sentiment_data.index)
        
        try:
            # Calculate normalized price and sentiment
            price_changes = price_data['close'].pct_change()
            sentiment_changes = sentiment_data['sentiment_score'].diff()
            
            # Calculate the 5-period z-score for each
            price_zscore = (price_changes.rolling(window=window).mean() - 
                           price_changes.rolling(window=window*3).mean()) / price_changes.rolling(window=window*3).std()
            
            sentiment_zscore = (sentiment_changes.rolling(window=window).mean() - 
                               sentiment_changes.rolling(window=window*3).mean()) / sentiment_changes.rolling(window=window*3).std()
            
            # Calculate divergence (difference between z-scores)
            result['sentiment_price_divergence'] = sentiment_zscore - price_zscore
            
            # Calculate divergence strength (absolute value)
            result['divergence_strength'] = result['sentiment_price_divergence'].abs()
            
            # Calculate divergence direction (positive = sentiment leading price)
            result['divergence_direction'] = np.sign(result['sentiment_price_divergence'])
            
        except Exception as e:
            logger.warning(f"Error calculating sentiment divergence: {e}")
            
        return result