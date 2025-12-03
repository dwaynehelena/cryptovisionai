#!/usr/bin/env python3
"""
Market Regime Detection Module
Identifies different market states (trending, ranging, volatile) for adaptive strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from scipy import stats
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import logging
from datetime import datetime
import joblib
import os
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger("market_regime")

class MarketRegimeDetector:
    """
    Detects market regimes using multiple techniques:
    1. Hidden Markov Models
    2. Volatility clustering
    3. Trend strength indicators
    4. Spectral analysis
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize market regime detector
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.n_regimes = self.config.get('n_regimes', 3)
        self.lookback_window = self.config.get('lookback_window', 20)
        self.random_state = self.config.get('random_state', 42)
        self.hmm_model = None
        self.kmeans_model = None
        self.regime_features = {}
        self.scaler = StandardScaler()
        self.logger = logging.getLogger('market_regime')
        
    def _extract_market_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Extract features for regime detection from price series
        
        Args:
            prices: Array of price values
            
        Returns:
            Feature matrix for regime detection
        """
        if len(prices) < self.lookback_window + 5:
            raise ValueError(f"Price series too short, need at least {self.lookback_window + 5} points")
        
        # Calculate returns
        returns = np.diff(prices) / prices[:-1]
        
        # Extract features
        feature_list = []
        
        # Process each window
        for i in range(len(returns) - self.lookback_window + 1):
            window = returns[i:i+self.lookback_window]
            
            # Volatility features
            volatility = np.std(window)
            recent_volatility = np.std(window[-5:])
            volatility_change = recent_volatility / (np.std(window[:-5]) + 1e-10) - 1
            
            # Trend features
            price_window = prices[i:i+self.lookback_window+1]
            trend, _, _, _, _ = stats.linregress(np.arange(len(price_window)), price_window)
            trend_strength = abs(trend) / (volatility + 1e-10)
            
            # Momentum
            momentum = (price_window[-1] / price_window[0]) - 1
            
            # Statistical features
            skew = stats.skew(window)
            kurtosis = stats.kurtosis(window)
            
            # Drawdown
            max_drawdown = (np.min(price_window) / np.max(price_window) - 1)
            
            # Autocorrelation (mean reversion vs momentum)
            acf_1 = np.corrcoef(window[:-1], window[1:])[0, 1]
            
            # Spectral features - detect cyclicality
            from scipy import signal
            try:
                f, Pxx = signal.periodogram(window)
                peak_freq = f[np.argmax(Pxx)]
                spectral_entropy = -np.sum(Pxx * np.log2(Pxx + 1e-10)) / np.log2(len(Pxx))
            except:
                peak_freq = 0
                spectral_entropy = 0
            
            # Combine features
            features = [
                volatility, 
                volatility_change,
                trend_strength,
                momentum,
                skew,
                kurtosis,
                max_drawdown,
                acf_1,
                peak_freq,
                spectral_entropy
            ]
            
            feature_list.append(features)
            
        return np.array(feature_list)
        
    def fit_hmm(self, prices: np.ndarray) -> hmm.GaussianHMM:
        """
        Fit Hidden Markov Model for regime detection
        
        Args:
            prices: Array of price values
            
        Returns:
            Fitted HMM model
        """
        self.logger.info(f"Fitting HMM with {self.n_regimes} regimes")
        
        features = self._extract_market_features(prices)
        scaled_features = self.scaler.fit_transform(features)
        
        # Reduce dimensionality for better HMM fitting
        if scaled_features.shape[1] > 3:
            pca = PCA(n_components=min(3, scaled_features.shape[1]))
            scaled_features = pca.fit_transform(scaled_features)
            
        # Try different covariance types
        best_score = -np.inf
        best_model = None
        
        for cov_type in ['spherical', 'diag', 'full']:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=cov_type,
                    random_state=self.random_state,
                    n_iter=100
                )
                model.fit(scaled_features)
                score = model.score(scaled_features)
                
                if score > best_score:
                    best_score = score
                    best_model = model
            except Exception as e:
                self.logger.warning(f"HMM fitting failed for {cov_type}: {e}")
        
        if best_model is None:
            raise ValueError("HMM model fitting failed for all covariance types")
            
        self.hmm_model = best_model
        return best_model
        
    def predict_regime_hmm(self, prices: np.ndarray) -> np.ndarray:
        """
        Predict regimes using HMM model
        
        Args:
            prices: Array of price values
            
        Returns:
            Array of regime labels
        """
        if self.hmm_model is None:
            self.fit_hmm(prices)
            
        features = self._extract_market_features(prices)
        scaled_features = self.scaler.transform(features)
        
        # Apply PCA if it was used during training
        if scaled_features.shape[1] > self.hmm_model.n_features:
            pca = PCA(n_components=self.hmm_model.n_features)
            scaled_features = pca.fit_transform(scaled_features)
        elif scaled_features.shape[1] < self.hmm_model.n_features:
            # Pad with zeros if needed
            pad_width = self.hmm_model.n_features - scaled_features.shape[1]
            scaled_features = np.pad(scaled_features, ((0, 0), (0, pad_width)))
        
        # Get regime probabilities and most likely regime
        regime_probs = self.hmm_model.predict_proba(scaled_features)
        regimes = self.hmm_model.predict(scaled_features)
        
        return regimes, regime_probs
        
    def fit_kmeans(self, prices: np.ndarray) -> KMeans:
        """
        Fit KMeans for regime detection
        
        Args:
            prices: Array of price values
            
        Returns:
            Fitted KMeans model
        """
        self.logger.info(f"Fitting KMeans with {self.n_regimes} regimes")
        
        features = self._extract_market_features(prices)
        scaled_features = self.scaler.fit_transform(features)
        
        self.kmeans_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10
        )
        self.kmeans_model.fit(scaled_features)
        
        # Analyze clusters to understand regimes
        self.analyze_clusters(scaled_features, self.kmeans_model.labels_)
        
        return self.kmeans_model
    
    def analyze_clusters(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Analyze regime clusters to assign meaningful labels
        
        Args:
            features: Feature matrix
            labels: Cluster labels
        """
        # Map back to original feature space
        original_features = self.scaler.inverse_transform(features)
        
        # For each cluster, compute mean of features
        for i in range(self.n_regimes):
            cluster_features = original_features[labels == i]
            
            if len(cluster_features) > 0:
                mean_features = np.mean(cluster_features, axis=0)
                
                # Store feature means for later interpretation
                self.regime_features[i] = mean_features
                
                # Simple logic to label regimes
                volatility = mean_features[0]  # Assuming first feature is volatility
                trend = mean_features[2]  # Assuming this is trend strength
                
                if volatility > np.median(original_features[:, 0]) * 1.5:
                    regime_type = "Volatile"
                elif abs(trend) > np.median(np.abs(original_features[:, 2])) * 1.2:
                    if trend > 0:
                        regime_type = "Bullish Trend"
                    else:
                        regime_type = "Bearish Trend"
                else:
                    regime_type = "Range Bound"
                
                self.logger.info(f"Cluster {i} interpreted as: {regime_type}")
                
                # Associate descriptive label with cluster
                self.regime_features[f"label_{i}"] = regime_type
    
    def predict_regime_kmeans(self, prices: np.ndarray) -> np.ndarray:
        """
        Predict regimes using KMeans model
        
        Args:
            prices: Array of price values
            
        Returns:
            Array of regime labels
        """
        if self.kmeans_model is None:
            self.fit_kmeans(prices)
            
        features = self._extract_market_features(prices)
        scaled_features = self.scaler.transform(features)
        
        regimes = self.kmeans_model.predict(scaled_features)
        return regimes
    
    def detect_regime_changes(self, regimes: np.ndarray, threshold: int = 3) -> List[int]:
        """
        Detect points where market regime changes
        
        Args:
            regimes: Array of regime labels
            threshold: Number of consecutive same regimes to confirm change
            
        Returns:
            List of indices where regime changes
        """
        change_points = []
        
        for i in range(threshold, len(regimes)):
            # Check if we've had a consistent different regime for 'threshold' periods
            if all(regimes[i-j] != regimes[i-threshold-1] for j in range(threshold)):
                change_points.append(i-threshold)
                
        return change_points
    
    def plot_regimes(self, prices: np.ndarray, regimes: np.ndarray, 
                   change_points: List[int] = None, title: str = 'Market Regimes'):
        """
        Plot price series with market regimes highlighted
        
        Args:
            prices: Array of price values
            regimes: Array of regime labels
            change_points: Indices of regime change points
            title: Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot price series
        plt.plot(prices, color='black', alpha=0.6)
        
        # Color regions by regime
        offset = len(prices) - len(regimes)
        
        # Create a colormap for regimes
        colors = ['lightblue', 'lightgreen', 'salmon', 'orchid', 'orange', 'cyan']
        
        for i in range(len(regimes)):
            plt.axvspan(i+offset, i+offset+1, alpha=0.3, color=colors[regimes[i] % len(colors)])
        
        # Mark regime changes
        if change_points:
            for cp in change_points:
                plt.axvline(x=cp+offset, color='red', linestyle='--', alpha=0.7)
        
        # Add labels and descriptions if available
        if self.regime_features:
            legend_elements = []
            from matplotlib.patches import Patch
            
            for i in range(self.n_regimes):
                if f"label_{i}" in self.regime_features:
                    label = self.regime_features[f"label_{i}"]
                    legend_elements.append(Patch(facecolor=colors[i % len(colors)], 
                                                alpha=0.3, label=label))
            
            plt.legend(handles=legend_elements)
        
        plt.title(title)
        plt.ylabel('Price')
        plt.xlabel('Time')
        plt.tight_layout()
        
        # Save figure
        save_dir = os.path.join(os.getcwd(), 'models', 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f'market_regimes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(save_dir, filename))
        self.logger.info(f"Regime plot saved to {filename}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save regime detection models
        
        Args:
            filepath: Path to save models
        """
        model_data = {
            'hmm_model': self.hmm_model,
            'kmeans_model': self.kmeans_model,
            'scaler': self.scaler,
            'regime_features': self.regime_features,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Market regime model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load regime detection models
        
        Args:
            filepath: Path to load models from
        """
        model_data = joblib.load(filepath)
        
        self.hmm_model = model_data.get('hmm_model')
        self.kmeans_model = model_data.get('kmeans_model')
        self.scaler = model_data.get('scaler')
        self.regime_features = model_data.get('regime_features', {})
        self.config = model_data.get('config', self.config)
        
        self.logger.info(f"Market regime model loaded from {filepath}")
    
    def generate_regime_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Generate regime-based features for model input
        
        Args:
            prices: Array of price values
            
        Returns:
            Array of regime features
        """
        if self.kmeans_model is None:
            self.fit_kmeans(prices)
            
        # Get regime predictions
        regimes = self.predict_regime_kmeans(prices)
        
        # For the most recent data points, create features
        regime_features = []
        offset = len(prices) - len(regimes)
        
        for i in range(len(regimes)):
            current_regime = regimes[i]
            
            # One-hot encode the current regime
            one_hot = np.zeros(self.n_regimes)
            one_hot[current_regime] = 1
            
            # Count days in current regime
            days_in_regime = 1
            j = i - 1
            while j >= 0 and regimes[j] == current_regime:
                days_in_regime += 1
                j -= 1
                
            # Detect if close to regime change (recent regime switches)
            regime_switches = 0
            for j in range(max(0, i-5), i):
                if j > 0 and regimes[j] != regimes[j-1]:
                    regime_switches += 1
            
            # Create feature vector
            features = np.concatenate([
                one_hot,
                [days_in_regime / 20.0],  # Normalize
                [regime_switches / 3.0]   # Normalize
            ])
            
            regime_features.append(features)
            
        return np.array(regime_features)

if __name__ == "__main__":
    # Simple test
    np.random.seed(42)
    
    # Generate synthetic price series with different regimes
    n_points = 200
    
    # First regime - trending up with low volatility
    prices1 = np.cumsum(np.random.normal(0.001, 0.01, n_points)) + 100
    
    # Second regime - high volatility
    prices2 = np.cumsum(np.random.normal(0, 0.03, n_points)) + prices1[-1]
    
    # Third regime - trending down
    prices3 = np.cumsum(np.random.normal(-0.001, 0.01, n_points)) + prices2[-1]
    
    # Fourth regime - range bound
    noise = np.random.normal(0, 0.005, n_points)
    cycle = 0.02 * np.sin(np.linspace(0, 6*np.pi, n_points))
    prices4 = prices3[-1] + noise + cycle
    
    # Combine all
    prices = np.concatenate([prices1, prices2, prices3, prices4])
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the regime detector
    detector = MarketRegimeDetector(config={'n_regimes': 4})
    
    # Fit and predict
    detector.fit_kmeans(prices)
    regimes = detector.predict_regime_kmeans(prices)
    
    # Plot results
    change_points = detector.detect_regime_changes(regimes)
    detector.plot_regimes(prices, regimes, change_points, 'Synthetic Price Series with Regimes')