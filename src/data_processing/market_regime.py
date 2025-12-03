import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import logging
from scipy.stats import zscore
from sklearn.decomposition import PCA
import joblib
import os

class MarketRegimeDetector:
    """
    Detects different market regimes using unsupervised learning techniques
    Market regimes include: trending bullish, trending bearish, high volatility, range-bound
    """
    
    def __init__(self, n_regimes=4, lookback_window=20):
        """
        Initialize the market regime detector
        
        Args:
            n_regimes: Number of market regimes to identify
            lookback_window: Window for feature calculation
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=n_regimes, random_state=42)
        self.pca = PCA(n_components=2)
        self.logger = logging.getLogger('MarketRegimeDetector')
        
    def extract_regime_features(self, prices):
        """
        Extract features that help identify market regimes
        
        Args:
            prices: DataFrame with 'close', 'high', 'low', 'volume' columns
            
        Returns:
            DataFrame with regime features
        """
        self.logger.info("Extracting market regime features")
        df = prices.copy()
        
        # Price based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Volatility features
        df['volatility'] = df['log_returns'].rolling(window=self.lookback_window).std()
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        df['rolling_range'] = df['high_low_range'].rolling(window=self.lookback_window).mean()
        
        # Trend features
        df['sma_short'] = df['close'].rolling(window=5).mean()
        df['sma_medium'] = df['close'].rolling(window=20).mean()
        df['sma_long'] = df['close'].rolling(window=50).mean()
        df['trend_strength'] = (df['sma_short'] - df['sma_long']) / df['sma_long']
        df['trend_consistency'] = df['log_returns'].rolling(window=self.lookback_window).apply(
            lambda x: np.sum(np.sign(x)) / len(x), raw=True)
            
        # Volume features
        df['volume_change'] = df['volume'].pct_change()
        df['volume_sma'] = df['volume'].rolling(window=self.lookback_window).mean()
        df['relative_volume'] = df['volume'] / df['volume_sma']
        
        # Momentum features
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_slope'] = df['rsi'].diff(5)
        
        # Mean reversion potential
        df['zscore_price'] = zscore(df['close'].rolling(window=self.lookback_window).mean(), nan_policy='omit')
        
        # Drop NaN values
        regime_features = df.dropna().copy()
        
        return regime_features
    
    def _calculate_rsi(self, prices, window=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def fit(self, prices):
        """
        Fit the regime detection model
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            Self
        """
        # Extract features
        regime_features = self.extract_regime_features(prices)
        
        # Select features for clustering
        features_for_clustering = [
            'volatility', 'rolling_range', 'trend_strength', 
            'trend_consistency', 'relative_volume', 'rsi'
        ]
        
        # Ensure all features exist
        existing_features = [f for f in features_for_clustering if f in regime_features.columns]
        if len(existing_features) < 3:
            self.logger.warning(f"Not enough regime features available. Using default features.")
            # Use whatever numerical features we have
            existing_features = regime_features.select_dtypes(include=[np.number]).columns.tolist()
            
        X = regime_features[existing_features].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensionality for better clustering
        X_pca = self.pca.fit_transform(X_scaled)
        
        # Fit KMeans
        self.kmeans.fit(X_scaled)
        
        # Store feature names for prediction
        self.feature_names = existing_features
        
        self.logger.info(f"Fitted market regime detector with features: {existing_features}")
        return self
    
    def predict(self, prices):
        """
        Predict the market regime for new data
        
        Args:
            prices: DataFrame with price data
            
        Returns:
            Array with regime labels
        """
        # Extract features
        regime_features = self.extract_regime_features(prices)
        
        # Ensure we have the necessary features
        missing_features = [f for f in self.feature_names if f not in regime_features.columns]
        if missing_features:
            self.logger.warning(f"Missing features for regime prediction: {missing_features}")
            return np.zeros(len(regime_features))
        
        # Select and scale the features
        X = regime_features[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        
        # Predict regimes
        regimes = self.kmeans.predict(X_scaled)
        
        return regimes
    
    def label_regimes(self, prices, regimes):
        """
        Add interpretable labels to the numeric regimes
        
        Args:
            prices: DataFrame with price data
            regimes: Array with regime labels from predict()
            
        Returns:
            DataFrame with regime labels and descriptions
        """
        # Combine prices and regimes
        df = prices.copy()
        df['regime'] = regimes
        
        # Calculate metrics for each regime
        regime_stats = {}
        for regime in range(self.n_regimes):
            mask = df['regime'] == regime
            if not any(mask):
                continue
                
            regime_data = df[mask]
            returns = regime_data['close'].pct_change().dropna()
            
            regime_stats[regime] = {
                'mean_return': returns.mean(),
                'volatility': returns.std(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0,
                'skew': returns.skew(),
                'count': len(regime_data)
            }
        
        # Assign labels based on statistics
        labels = {}
        for regime, stats in regime_stats.items():
            if stats['mean_return'] > 0.001:  # Strong positive returns
                if stats['volatility'] > 0.02:  # High volatility
                    labels[regime] = 'High Volatility Bullish'
                else:
                    labels[regime] = 'Trending Bullish'
            elif stats['mean_return'] < -0.001:  # Strong negative returns
                if stats['volatility'] > 0.02:  # High volatility
                    labels[regime] = 'High Volatility Bearish'
                else:
                    labels[regime] = 'Trending Bearish'
            else:  # Sideways
                if stats['volatility'] > 0.015:
                    labels[regime] = 'High Volatility Sideways'
                else:
                    labels[regime] = 'Range-Bound'
        
        # Add descriptive labels
        df['regime_label'] = df['regime'].map(labels)
        
        self.regime_labels = labels
        return df
    
    def save(self, path):
        """Save the market regime detector to disk"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Serialize the model
        joblib.dump(self, path)
        self.logger.info(f"MarketRegimeDetector saved to {path}")
        
    @classmethod
    def load(cls, path):
        """Load the market regime detector from disk"""
        model = joblib.load(path)
        return model