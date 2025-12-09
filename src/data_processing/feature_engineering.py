#!/usr/bin/env python3
"""
Feature Engineering Module - Extracts and transforms features from raw cryptocurrency data
for use in predictive models
"""

import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import logging

class CryptoFeatureEngineer:
    """
    Enhanced feature engineering for cryptocurrency prediction
    Includes technical indicators, on-chain metrics, sentiment analysis,
    and cross-asset correlation features
    """
    
    def __init__(self, feature_config=None):
        """
        Initialize the feature engineer
        
        Args:
            feature_config: Dictionary of feature configurations
        """
        self.feature_config = feature_config or {
            'technical_indicators': True,
            'volatility_features': True,
            'volume_features': True,
            'pattern_features': True,
            'temporal_features': True,
            'price_derivatives': True,
            'liquidity_features': False,  # Requires additional data
            'sentiment_features': False,  # Requires additional data
            'on_chain_metrics': False     # Requires additional data
        }
        self.scaler = StandardScaler()
        self.logger = logging.getLogger('CryptoFeatureEngineer')
        
    def transform(self, df, fit=False):
        """
        Transform data with advanced features
        
        Args:
            df: DataFrame with OHLCV data
            fit: Whether to fit the scaler
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Generating advanced features")
        
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Technical indicators
        if self.feature_config.get('technical_indicators', True):
            result = self._add_technical_indicators(result)
        
        # Volatility features
        if self.feature_config.get('volatility_features', True):
            result = self._add_volatility_features(result)
        
        # Volume features
        if self.feature_config.get('volume_features', True):
            result = self._add_volume_features(result)
            
        # Chart pattern features
        if self.feature_config.get('pattern_features', True):
            result = self._add_pattern_features(result)
            
        # Temporal features
        if self.feature_config.get('temporal_features', True):
            result = self._add_temporal_features(result)
            
        # Price derivative features
        if self.feature_config.get('price_derivatives', True):
            result = self._add_price_derivatives(result)
            
        # On-chain metrics if available
        if self.feature_config.get('on_chain_metrics', False) and 'active_addresses' in df.columns:
            result = self._add_on_chain_features(result)
            
        # Sentiment features if available
        if self.feature_config.get('sentiment_features', False) and 'sentiment_score' in df.columns:
            result = self._add_sentiment_features(result)
            
        # Liquidity features if available
        if self.feature_config.get('liquidity_features', False) and 'bid_ask_spread' in df.columns:
            result = self._add_liquidity_features(result)
            
        # Drop NaN values
        result = result.replace([np.inf, -np.inf], np.nan)
        result = result.dropna()
        
        # Select only numeric columns
        numeric_cols = result.select_dtypes(include=[np.number]).columns.tolist()
        result = result[numeric_cols]
        
        self.feature_names = numeric_cols
        return result
        
    def _add_technical_indicators(self, df):
        """Add technical analysis indicators"""
        try:
            # Moving averages
            for window in [3, 5, 7, 14, 21, 50, 100, 200]:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window).mean()
                
            # SMA ratios (trend strength)
            df['sma_ratio_short'] = df['sma_7'] / df['sma_21']
            df['sma_ratio_long'] = df['sma_21'] / df['sma_100']
            
            # MACD
            df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(
                df['close'], fastperiod=12, slowperiod=26, signalperiod=9
            )
            
            # RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            df['rsi_short'] = talib.RSI(df['close'], timeperiod=5)
            
            # Stochastic Oscillator
            df['slowk'], df['slowd'] = talib.STOCH(
                df['high'], df['low'], df['close'], 
                fastk_period=14, slowk_period=3, slowk_matype=0, 
                slowd_period=3, slowd_matype=0
            )
            
            # Bollinger Bands
            df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(
                df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
            )
            df['bbands_width'] = (df['bbands_upper'] - df['bbands_lower']) / df['bbands_middle']
            df['bbands_position'] = (df['close'] - df['bbands_lower']) / (df['bbands_upper'] - df['bbands_lower'])
            
            # Commodity Channel Index
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Average Directional Movement Index
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Parabolic SAR
            df['sar'] = talib.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            df['sar_ratio'] = df['close'] / df['sar']
            
            # ATR - Average True Range
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # OBV - On Balance Volume
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['obv_change'] = df['obv'].pct_change()
            
            # Rate of Change
            df['roc'] = talib.ROC(df['close'], timeperiod=10)
            
            # Williams %R
            df['willr'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
        
        return df
        
    def _add_volatility_features(self, df):
        """Add volatility-based features"""
        try:
            # Calculate returns
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            # Historical volatility
            for window in [3, 5, 10, 20, 30, 200]:
                df[f'volatility_{window}d'] = df['log_returns'].rolling(window=window).std()
            
            # Volatility ratio (short-term vs long-term)
            df['volatility_ratio'] = df['volatility_5d'] / df['volatility_30d']
            
            # Normalized true range
            df['true_range'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['normalized_range'] = df['true_range'] / df['close']
            
            # Garman-Klass volatility
            df['gk_vol'] = np.sqrt(
                0.5 * np.log(df['high'] / df['low'])**2 - 
                (2 * np.log(2) - 1) * np.log(df['close'] / df['open'])**2
            )
            
            # Parkinson volatility estimator
            df['parkinson_vol'] = np.sqrt(
                (1.0 / (4.0 * np.log(2.0))) * 
                np.log(df['high'] / df['low'])**2
            )
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility features: {e}")
            
        return df
        
    def _add_volume_features(self, df):
        """Add volume-based features"""
        try:
            # Volume moving averages
            for window in [5, 10, 20]:
                df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
            
            # Relative volume
            df['relative_volume'] = df['volume'] / df['volume_sma_20']
            
            # Price-volume relationship
            df['price_volume_rank'] = df['close'] * df['volume']
            df['price_volume_rank_zscore'] = zscore(df['price_volume_rank'], nan_policy='omit')
            
            # Volume momentum
            df['volume_change'] = df['volume'].pct_change()
            df['volume_momentum'] = df['volume_change'].rolling(window=5).mean()
            
            # Chaikin Money Flow
            money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
            money_flow_volume = money_flow_multiplier * df['volume']
            df['cmf'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()
            
            # Ease of movement
            distance_moved = (df['high'] + df['low']) / 2 - (df['high'].shift(1) + df['low'].shift(1)) / 2
            box_ratio = (df['volume'] / 1000000) / (df['high'] - df['low'])
            df['eom'] = distance_moved / box_ratio
            df['eom_ma'] = df['eom'].rolling(window=14).mean()
            
        except Exception as e:
            self.logger.error(f"Error calculating volume features: {e}")
            
        return df
        
    def _add_pattern_features(self, df):
        """Add chart pattern recognition features"""
        try:
            # Create a dictionary to hold all the pattern features
            pattern_features = {}
            
            # Bullish patterns
            pattern_features['pattern_hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_inverted_hammer'] = talib.CDLINVERTEDHAMMER(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_engulfing_bullish'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_three_white_soldiers'] = talib.CDL3WHITESOLDIERS(df['open'], df['high'], df['low'], df['close'])
            
            # Bearish patterns
            pattern_features['pattern_hanging_man'] = talib.CDLHANGINGMAN(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_engulfing_bearish'] = -talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_three_black_crows'] = talib.CDL3BLACKCROWS(df['open'], df['high'], df['low'], df['close'])
            
            # Continuation patterns
            pattern_features['pattern_doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            pattern_features['pattern_harami'] = talib.CDLHARAMI(df['open'], df['high'], df['low'], df['close'])
            
            # Add all pattern features at once
            patterns_df = pd.DataFrame(pattern_features, index=df.index)
            df = pd.concat([df, patterns_df], axis=1)
            
            # Consolidate bullish/bearish signals
            bullish_patterns = ['pattern_hammer', 'pattern_inverted_hammer', 'pattern_engulfing_bullish', 
                           'pattern_morning_star', 'pattern_three_white_soldiers']
            bearish_patterns = ['pattern_hanging_man', 'pattern_shooting_star', 'pattern_engulfing_bearish',
                           'pattern_evening_star', 'pattern_three_black_crows']
            
            # Create consolidated features
            df['bullish_pattern_count'] = df[bullish_patterns].clip(lower=0).sum(axis=1)
            df['bearish_pattern_count'] = df[bearish_patterns].clip(lower=0).sum(axis=1)
            df['pattern_strength'] = df['bullish_pattern_count'] - df['bearish_pattern_count']
            
        except Exception as e:
            self.logger.error(f"Error calculating pattern features: {e}")
            
        return df
        
    def _add_temporal_features(self, df):
        """Add time-based features"""
        try:
            # Ensure we have a proper datetime index
            if df.index.dtype != 'datetime64[ns]' and 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.set_index('timestamp')
            
            # Create a dictionary for all temporal features
            temporal_features = {}
            
            # Time features
            temporal_features['hour'] = df.index.hour
            temporal_features['day'] = df.index.day
            temporal_features['day_of_week'] = df.index.dayofweek
            temporal_features['month'] = df.index.month
            temporal_features['quarter'] = df.index.quarter
            temporal_features['year'] = df.index.year
            
            # Cyclic encoding of time features
            temporal_features['hour_sin'] = np.sin(temporal_features['hour'] * (2 * np.pi / 24))
            temporal_features['hour_cos'] = np.cos(temporal_features['hour'] * (2 * np.pi / 24))
            temporal_features['day_sin'] = np.sin((temporal_features['day'] - 1) * (2 * np.pi / 31))
            temporal_features['day_cos'] = np.cos((temporal_features['day'] - 1) * (2 * np.pi / 31))
            temporal_features['month_sin'] = np.sin((temporal_features['month'] - 1) * (2 * np.pi / 12))
            temporal_features['month_cos'] = np.cos((temporal_features['month'] - 1) * (2 * np.pi / 12))
            temporal_features['day_of_week_sin'] = np.sin(temporal_features['day_of_week'] * (2 * np.pi / 7))
            temporal_features['day_of_week_cos'] = np.cos(temporal_features['day_of_week'] * (2 * np.pi / 7))
            
            # Weekend indicator
            temporal_features['is_weekend'] = [1 if x >= 5 else 0 for x in temporal_features['day_of_week']]
            
            # Add all temporal features at once
            temporal_df = pd.DataFrame(temporal_features, index=df.index)
            df = pd.concat([df, temporal_df], axis=1)
            
        except Exception as e:
            self.logger.error(f"Error calculating temporal features: {e}")
            
        return df
        
    def _add_price_derivatives(self, df):
        """Add price derivative features"""
        try:
            # Create a dictionary for all price derivative features
            derivatives = {}
            
            # Price momentum
            for window in [1, 3, 5, 10, 21, 50, 200]:
                derivatives[f'momentum_{window}d'] = df['close'] / df['close'].shift(window) - 1
            
            # Price acceleration (calculate after adding momentum features to the dictionary)
            for window in [3, 5, 10]:
                momentum_key = f'momentum_{window}d'
                derivatives[f'acceleration_{window}d'] = derivatives[momentum_key].diff()
                
            # Rate of change in different timeframes
            for window in [1, 3, 5, 10, 21]:
                derivatives[f'roc_{window}d'] = (df['close'] - df['close'].shift(window)) / df['close'].shift(window) * 100
                
            # Returns over different periods
            for window in [1, 3, 5, 10, 21]:
                derivatives[f'return_{window}d'] = df['close'].pct_change(window)
                
            # Log returns over different periods (better for statistical properties)
            for window in [1, 3, 5, 10, 21]:
                price_ratio = df['close'] / df['close'].shift(window)
                derivatives[f'log_return_{window}d'] = np.log(price_ratio.replace([np.inf, -np.inf, 0], np.nan))
            
            # Add all price derivative features at once    
            derivatives_df = pd.DataFrame(derivatives, index=df.index)
            df = pd.concat([df, derivatives_df], axis=1)
                
        except Exception as e:
            self.logger.error(f"Error calculating price derivative features: {e}")
            
        return df
        
    def _add_on_chain_features(self, df):
        """Add on-chain metrics features if available"""
        try:
            if all(col in df.columns for col in ['active_addresses', 'transaction_count', 'average_transaction_value']):
                # On-chain activity features
                df['active_addresses_change'] = df['active_addresses'].pct_change()
                df['transaction_count_change'] = df['transaction_count'].pct_change()
                df['transaction_value_change'] = df['average_transaction_value'].pct_change()
                
                # Activity ratios
                df['txn_per_address'] = df['transaction_count'] / df['active_addresses']
                df['value_per_txn'] = df['average_transaction_value'] / df['transaction_count']
                
                # Rolling metrics
                for window in [7, 14, 30]:
                    df[f'active_addresses_sma_{window}d'] = df['active_addresses'].rolling(window).mean()
                    df[f'active_addresses_ratio_{window}d'] = df['active_addresses'] / df[f'active_addresses_sma_{window}d']
                    
                    df[f'transaction_count_sma_{window}d'] = df['transaction_count'].rolling(window).mean()
                    df[f'transaction_count_ratio_{window}d'] = df['transaction_count'] / df[f'transaction_count_sma_{window}d']
                
        except Exception as e:
            self.logger.error(f"Error calculating on-chain features: {e}")
            
        return df
        
    def _add_sentiment_features(self, df):
        """Add sentiment-based features if available"""
        try:
            if 'sentiment_score' in df.columns:
                # Simple sentiment features
                df['sentiment_change'] = df['sentiment_score'].pct_change()
                
                # Rolling sentiment metrics
                for window in [3, 7, 14]:
                    df[f'sentiment_sma_{window}d'] = df['sentiment_score'].rolling(window).mean()
                    df[f'sentiment_std_{window}d'] = df['sentiment_score'].rolling(window).std()
                    
                # Sentiment momentum
                df['sentiment_momentum'] = df['sentiment_score'] - df['sentiment_sma_7d']
                
                # Sentiment divergence with price
                price_momentum = df['close'].pct_change(7)
                sentiment_momentum = df['sentiment_score'].pct_change(7)
                df['price_sentiment_divergence'] = price_momentum - sentiment_momentum
                
        except Exception as e:
            self.logger.error(f"Error calculating sentiment features: {e}")
            
        return df
        
    def _add_liquidity_features(self, df):
        """Add liquidity-based features if available"""
        try:
            if all(col in df.columns for col in ['bid_ask_spread', 'order_book_depth']):
                # Simple liquidity features
                df['relative_spread'] = df['bid_ask_spread'] / df['close']
                
                # Rolling liquidity metrics
                for window in [3, 7, 14]:
                    df[f'spread_sma_{window}d'] = df['bid_ask_spread'].rolling(window).mean()
                    df[f'depth_sma_{window}d'] = df['order_book_depth'].rolling(window).mean()
                    
                # Liquidity change metrics
                df['spread_change'] = df['bid_ask_spread'].pct_change()
                df['depth_change'] = df['order_book_depth'].pct_change()
                
                # Liquidity ratio to volatility
                for window in [7, 14]:
                    liquidity_ratio = df[f'spread_sma_{window}d'] / df[f'volatility_{window}d']
                    df[f'liquidity_volatility_ratio_{window}d'] = liquidity_ratio
                
        except Exception as e:
            self.logger.error(f"Error calculating liquidity features: {e}")
            
        return df
        
    def extract_features(self, df, feature_set="standard", include_target=False):
        """
        Extract features from raw OHLCV data
        
        Args:
            df: DataFrame with OHLCV data
            feature_set: Type of feature set to use ('standard', 'advanced', 'minimal')
            include_target: Whether to include target variable in the output
                
        Returns:
            DataFrame with extracted features
        """
        self.logger.info(f"Extracting features with feature set: {feature_set}")
        
        try:
            if feature_set == "minimal":
                # Use only price and volume-based features
                self.feature_config = {
                    'technical_indicators': True,
                    'volatility_features': True,
                    'volume_features': True,
                    'pattern_features': False,
                    'temporal_features': False,
                    'price_derivatives': True,
                    'liquidity_features': False,
                    'sentiment_features': False,
                    'on_chain_metrics': False
                }
            elif feature_set == "advanced":
                # Use all available features
                self.feature_config = {
                    'technical_indicators': True,
                    'volatility_features': True,
                    'volume_features': True,
                    'pattern_features': True,
                    'temporal_features': True,
                    'price_derivatives': True,
                    'liquidity_features': True,
                    'sentiment_features': True,
                    'on_chain_metrics': True
                }
            else:  # standard
                # Use default feature set
                self.feature_config = {
                    'technical_indicators': True,
                    'volatility_features': True,
                    'volume_features': True,
                    'pattern_features': True,
                    'temporal_features': True,
                    'price_derivatives': True,
                    'liquidity_features': False,
                    'sentiment_features': False,
                    'on_chain_metrics': False
                }
                
            # Transform the data with the selected feature set
            result = self.transform(df)
            
            # Create target variable if requested
            if include_target and len(df) > 1:
                # Default target: next period's return
                result['target'] = df['close'].shift(-1) / df['close'] - 1
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error extracting features: {e}")
            raise
        
    def save_scaler(self, path):
        """Save the feature scaler to disk"""
        import joblib
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.scaler, path)
        self.logger.info(f"Feature scaler saved to {path}")
        
    @classmethod
    def load_scaler(cls, path):
        """Load the feature scaler from disk"""
        import joblib
        return joblib.load(path)

# Add an alias for the class to maintain backward compatibility
FeatureEngineering = CryptoFeatureEngineer