#!/usr/bin/env python3
"""
Enhanced Market Regime Detection
Implements advanced techniques for identifying market regimes
with improved accuracy and stability
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.pipeline import Pipeline
import logging
from datetime import datetime
import joblib
import os
from hmmlearn import hmm
from scipy import stats, signal
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logger = logging.getLogger("enhanced_regime")

class EnhancedRegimeDetector:
    """
    Advanced market regime detection using multiple techniques:
    1. Enhanced Hidden Markov Models with optimized parameters
    2. Multi-resolution volatility analysis
    3. Trend persistence metrics
    4. Wavelets for cycle detection
    5. Dynamic Time Warping for pattern matching
    6. Ensemble approach combining multiple detection methods
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize enhanced regime detector
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.n_regimes = self.config.get('n_regimes', 4)  # Default to 4 regimes
        self.lookback_window = self.config.get('lookback_window', 60)  # Longer window
        self.short_window = self.config.get('short_window', 20)
        self.random_state = self.config.get('random_state', 42)
        self.use_ensemble = self.config.get('use_ensemble', True)
        
        # Models
        self.hmm_model = None
        self.kmeans_model = None
        self.spectral_model = None
        self.ensemble_weights = self.config.get('ensemble_weights', [0.5, 0.3, 0.2])  # HMM, KMeans, Spectral
        
        # Data transformation
        self.scaler = RobustScaler()
        self.pca = None
        self.regime_features = {}
        self.regime_labels = {}
        self.feature_importance = None
        
        # Advanced detection
        self.use_wavelets = self.config.get('use_wavelets', True)
        self.use_dtw = self.config.get('use_dtw', False)  # Dynamic Time Warping is computationally expensive
        
        self.logger = logging.getLogger('enhanced_regime')
        
    def _extract_advanced_features(self, prices: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
        """
        Extract enhanced feature set for regime detection
        
        Args:
            prices: Array of price values
            volumes: Optional array of volume values
            
        Returns:
            Feature matrix for regime detection
        """
        if len(prices) < self.lookback_window + 15:
            raise ValueError(f"Price series too short, need at least {self.lookback_window + 15} points")
        
        # Calculate returns at different timeframes
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.log(prices[1:]) - np.log(prices[:-1])
        
        # Extract features
        feature_list = []
        
        # Process each window
        for i in range(len(returns) - self.lookback_window + 1):
            window = returns[i:i+self.lookback_window]
            log_window = log_returns[i:i+self.lookback_window]
            price_window = prices[i:i+self.lookback_window+1]
            
            # Multi-timeframe volatility features
            volatility_long = np.std(window)
            volatility_medium = np.std(window[-self.short_window:])
            volatility_short = np.std(window[-10:])
            volatility_ratio_s_m = volatility_short / (volatility_medium + 1e-10)
            volatility_ratio_m_l = volatility_medium / (volatility_long + 1e-10)
            
            # GARCH-like volatility persistence
            abs_returns = np.abs(window)
            vol_persistence = np.corrcoef(abs_returns[:-1], abs_returns[1:])[0, 1]
            
            # Trend features with multiple timeframes
            trend_long, _, r_value_long, _, _ = stats.linregress(np.arange(len(price_window)), price_window)
            trend_medium, _, r_value_medium, _, _ = stats.linregress(np.arange(self.short_window+1), price_window[-self.short_window-1:])
            trend_short, _, r_value_short, _, _ = stats.linregress(np.arange(11), price_window[-11:])
            
            # R-squared as trend strength measures
            trend_strength_long = r_value_long**2
            trend_strength_medium = r_value_medium**2
            trend_strength_short = r_value_short**2
            
            # Trend persistence - does the trend tend to continue?
            trend_persistence = np.corrcoef(window[:-1], window[1:])[0, 1]
            
            # Momentum at different timeframes
            momentum_long = (price_window[-1] / price_window[0]) - 1
            momentum_medium = (price_window[-1] / price_window[-self.short_window-1]) - 1
            momentum_short = (price_window[-1] / price_window[-11]) - 1
            
            # Higher-order statistical features
            skew = stats.skew(window)
            kurtosis = stats.kurtosis(window)
            
            # Drawdown metrics
            rolling_max = np.maximum.accumulate(price_window)
            drawdowns = price_window / rolling_max - 1
            max_drawdown = np.min(drawdowns)
            avg_drawdown = np.mean(drawdowns)
            drawdown_duration = np.sum(drawdowns < -0.01)  # Duration of significant drawdowns
            
            # Autocorrelation at different lags for mean reversion vs trend following
            acf_1 = np.corrcoef(window[:-1], window[1:])[0, 1] if len(window) > 1 else 0
            acf_2 = np.corrcoef(window[:-2], window[2:])[0, 1] if len(window) > 2 else 0
            acf_5 = np.corrcoef(window[:-5], window[5:])[0, 1] if len(window) > 5 else 0
            
            # Advanced spectral features - detect cyclicality
            try:
                # Compute periodogram
                f, Pxx = signal.periodogram(window, detrend='linear')
                peak_freq = f[np.argmax(Pxx[1:]) + 1]  # Skip zero frequency
                spectral_entropy = -np.sum(Pxx * np.log2(Pxx + 1e-10)) / np.log2(len(Pxx))
                
                # Power in different frequency bands
                low_freq_power = np.sum(Pxx[1:len(Pxx)//10]) / (np.sum(Pxx) + 1e-10)
                mid_freq_power = np.sum(Pxx[len(Pxx)//10:len(Pxx)//3]) / (np.sum(Pxx) + 1e-10)
                high_freq_power = np.sum(Pxx[len(Pxx)//3:]) / (np.sum(Pxx) + 1e-10)
            except Exception as e:
                self.logger.warning(f"Error in spectral analysis: {e}")
                peak_freq = 0
                spectral_entropy = 0
                low_freq_power = 0
                mid_freq_power = 0
                high_freq_power = 0
            
            # Volume-based features (if available)
            vol_features = []
            if volumes is not None and i < len(volumes) - self.lookback_window + 1:
                vol_window = volumes[i:i+self.lookback_window]
                vol_mean = np.mean(vol_window)
                vol_std = np.std(vol_window)
                recent_vol_ratio = np.mean(vol_window[-5:]) / (vol_mean + 1e-10)
                vol_trend, _, vol_r_value, _, _ = stats.linregress(np.arange(len(vol_window)), vol_window)
                vol_price_corr = np.corrcoef(window, vol_window)[0, 1]
                vol_features = [vol_mean, vol_std, recent_vol_ratio, vol_trend, vol_price_corr]
            
            # Wavelet features if enabled
            wavelet_features = []
            if self.use_wavelets:
                try:
                    import pywt
                    # Perform wavelet decomposition
                    coeffs = pywt.wavedec(window, 'db4', level=3)
                    wavelet_energy = [np.sum(np.square(c)) for c in coeffs]
                    total_energy = sum(wavelet_energy) + 1e-10
                    wavelet_energy_ratio = [e/total_energy for e in wavelet_energy]
                    wavelet_features = wavelet_energy_ratio
                except Exception as e:
                    self.logger.warning(f"Error in wavelet analysis: {e}")
                    wavelet_features = [0, 0, 0, 0]
            
            # Combine all features
            features = [
                # Volatility features
                volatility_long, 
                volatility_medium,
                volatility_short,
                volatility_ratio_s_m,
                volatility_ratio_m_l,
                vol_persistence,
                
                # Trend features
                trend_long,
                trend_medium,
                trend_short, 
                trend_strength_long,
                trend_strength_medium,
                trend_strength_short,
                trend_persistence,
                
                # Momentum
                momentum_long,
                momentum_medium,
                momentum_short,
                
                # Statistical
                skew,
                kurtosis,
                
                # Drawdown
                max_drawdown,
                avg_drawdown,
                drawdown_duration,
                
                # Autocorrelation
                acf_1,
                acf_2,
                acf_5,
                
                # Spectral
                peak_freq,
                spectral_entropy,
                low_freq_power,
                mid_freq_power,
                high_freq_power
            ]
            
            # Add optional features
            if vol_features:
                features.extend(vol_features)
            if wavelet_features:
                features.extend(wavelet_features)
            
            feature_list.append(features)
            
        return np.array(feature_list)
    
    def _determine_feature_importance(self, features: np.ndarray):
        """Calculate feature importance for interpretability"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            # Use a forest to determine feature importance
            if features.shape[0] > 10:  # Need enough samples
                # Create synthetic target (cluster quality measure)
                pca_result = PCA(n_components=1).fit_transform(features).ravel()
                
                rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=self.random_state)
                rf.fit(features, pca_result)
                
                self.feature_importance = rf.feature_importances_
        except Exception as e:
            self.logger.warning(f"Could not determine feature importance: {e}")
            self.feature_importance = None
        
    def fit_hmm(self, prices: np.ndarray, volumes: np.ndarray = None) -> hmm.GaussianHMM:
        """
        Fit Enhanced Hidden Markov Model for regime detection
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Fitted HMM model
        """
        self.logger.info(f"Fitting Enhanced HMM with {self.n_regimes} regimes")
        
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.fit_transform(features)
        
        # Apply dimensionality reduction for better HMM fitting
        n_components = min(5, scaled_features.shape[1])
        self.pca = PCA(n_components=n_components)
        reduced_features = self.pca.fit_transform(scaled_features)
        
        # Determine feature importance
        self._determine_feature_importance(scaled_features)
        
        # Try different HMM configurations
        best_score = -np.inf
        best_model = None
        
        # Test different covariance types
        for cov_type in ['spherical', 'diag', 'full']:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=cov_type,
                    random_state=self.random_state,
                    n_iter=200
                )
                model.fit(reduced_features)
                score = model.score(reduced_features)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    self.logger.info(f"New best HMM model: {cov_type} with score {score:.2f}")
            except Exception as e:
                self.logger.warning(f"HMM fitting failed for {cov_type}: {e}")
        
        if best_model is None:
            raise ValueError("HMM model fitting failed for all covariance types")
            
        self.hmm_model = best_model
        
        # Analyze the regimes
        hmm_regimes = best_model.predict(reduced_features)
        self._analyze_hmm_regimes(features, hmm_regimes)
        
        return best_model
    
    def _analyze_hmm_regimes(self, features: np.ndarray, regimes: np.ndarray):
        """Analyze HMM regimes to assign meaningful labels"""
        regime_data = {}
        
        # For each regime, compute statistics on original features
        for i in range(self.n_regimes):
            regime_mask = regimes == i
            if np.sum(regime_mask) == 0:
                continue
                
            regime_features = features[regime_mask]
            mean_features = np.mean(regime_features, axis=0)
            std_features = np.std(regime_features, axis=0)
            
            regime_data[i] = {
                'mean': mean_features,
                'std': std_features,
                'count': np.sum(regime_mask),
                'frequency': np.sum(regime_mask) / len(regimes)
            }
            
        # Interpret regimes based on key metrics
        # Using indices based on the feature extraction order
        volatility_idx = 0  # Long-term volatility
        trend_idx = 7       # Long-term trend
        momentum_idx = 13   # Long-term momentum
        
        # Assign interpretable labels to regimes
        for i, data in regime_data.items():
            volatility = data['mean'][volatility_idx]
            trend = data['mean'][trend_idx]
            momentum = data['mean'][momentum_idx]
            
            # Set threshold relative to all regimes
            vol_values = [r['mean'][volatility_idx] for r in regime_data.values()]
            trend_values = [abs(r['mean'][trend_idx]) for r in regime_data.values()]
            
            if volatility > np.median(vol_values) * 1.3:
                if trend > 0:
                    label = "Volatile Bullish"
                else:
                    label = "Volatile Bearish"
            elif abs(trend) > np.median(trend_values) * 1.2:
                if trend > 0:
                    label = "Strong Bullish Trend"
                else:
                    label = "Strong Bearish Trend"
            elif abs(trend) > np.median(trend_values) * 0.8:
                if trend > 0:
                    label = "Moderate Bullish Trend"
                else:
                    label = "Moderate Bearish Trend"
            else:
                if volatility < np.median(vol_values) * 0.7:
                    label = "Low Volatility Range"
                else:
                    label = "Neutral/Range Bound"
            
            # Store label
            self.regime_labels[f"hmm_{i}"] = label
            self.logger.info(f"HMM Regime {i} interpreted as: {label}")
    
    def predict_regime_hmm(self, prices: np.ndarray, volumes: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict regimes using HMM model with probabilities
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Tuple of (regime labels, regime probabilities)
        """
        if self.hmm_model is None:
            self.fit_hmm(prices, volumes)
            
        # Extract features and transform
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.transform(features)
        reduced_features = self.pca.transform(scaled_features)
        
        # Get predictions and probabilities
        regimes = self.hmm_model.predict(reduced_features)
        probabilities = self.hmm_model.predict_proba(reduced_features)
        
        return regimes, probabilities
    
    def fit_kmeans(self, prices: np.ndarray, volumes: np.ndarray = None) -> KMeans:
        """
        Fit KMeans clustering for regime detection
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Fitted KMeans model
        """
        self.logger.info(f"Fitting KMeans with {self.n_regimes} regimes")
        
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.fit_transform(features)
        
        # Try multiple initializations for better clustering
        self.kmeans_model = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=20,  # More initializations for better results
            max_iter=500
        )
        
        self.kmeans_model.fit(scaled_features)
        
        # Analyze the clusters
        self._analyze_kmeans_regimes(features, self.kmeans_model.labels_)
        
        return self.kmeans_model
    
    def _analyze_kmeans_regimes(self, features: np.ndarray, labels: np.ndarray):
        """Analyze KMeans regimes to assign meaningful labels"""
        # Similar to HMM analysis
        regime_data = {}
        
        # For each regime, compute statistics
        for i in range(self.n_regimes):
            regime_mask = labels == i
            if np.sum(regime_mask) == 0:
                continue
                
            regime_features = features[regime_mask]
            mean_features = np.mean(regime_features, axis=0)
            std_features = np.std(regime_features, axis=0)
            
            regime_data[i] = {
                'mean': mean_features,
                'std': std_features,
                'count': np.sum(regime_mask),
                'frequency': np.sum(regime_mask) / len(labels)
            }
            
        # Interpret regimes - same indices as in HMM analysis
        volatility_idx = 0  # Long-term volatility
        trend_idx = 7       # Long-term trend
        momentum_idx = 13   # Long-term momentum
        
        # Assign interpretable labels to regimes (similar to HMM)
        for i, data in regime_data.items():
            volatility = data['mean'][volatility_idx]
            trend = data['mean'][trend_idx]
            momentum = data['mean'][momentum_idx]
            
            # Set threshold relative to all regimes
            vol_values = [r['mean'][volatility_idx] for r in regime_data.values()]
            trend_values = [abs(r['mean'][trend_idx]) for r in regime_data.values()]
            
            # Similar logic to HMM but with slightly different thresholds
            if volatility > np.median(vol_values) * 1.3:
                if trend > 0:
                    label = "Volatile Bullish"
                else:
                    label = "Volatile Bearish"
            elif abs(trend) > np.median(trend_values) * 1.2:
                if trend > 0:
                    label = "Strong Bullish Trend"
                else:
                    label = "Strong Bearish Trend"
            elif abs(trend) > np.median(trend_values) * 0.8:
                if trend > 0:
                    label = "Moderate Bullish Trend"
                else:
                    label = "Moderate Bearish Trend"
            else:
                if volatility < np.median(vol_values) * 0.7:
                    label = "Low Volatility Range"
                else:
                    label = "Neutral/Range Bound"
            
            # Store label
            self.regime_labels[f"kmeans_{i}"] = label
            self.logger.info(f"KMeans Regime {i} interpreted as: {label}")
    
    def predict_regime_kmeans(self, prices: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
        """
        Predict regimes using KMeans model
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Array of regime labels
        """
        if self.kmeans_model is None:
            self.fit_kmeans(prices, volumes)
            
        # Extract features and transform
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.transform(features)
        
        # Get predictions
        return self.kmeans_model.predict(scaled_features)
    
    def fit_spectral_clustering(self, prices: np.ndarray, volumes: np.ndarray = None) -> SpectralClustering:
        """
        Fit Spectral Clustering for regime detection (captures non-linear structures)
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Fitted Spectral Clustering model
        """
        self.logger.info(f"Fitting Spectral Clustering with {self.n_regimes} regimes")
        
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.fit_transform(features)
        
        # Spectral clustering for non-linear regime boundaries
        self.spectral_model = SpectralClustering(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            affinity='nearest_neighbors',
            n_neighbors=min(30, max(5, len(scaled_features) // 10))
        )
        
        labels = self.spectral_model.fit_predict(scaled_features)
        
        # Store feature vectors for prediction
        self.spectral_features = scaled_features
        self.spectral_labels = labels
        
        # Analyze the clusters
        self._analyze_spectral_regimes(features, labels)
        
        return self.spectral_model
    
    def _analyze_spectral_regimes(self, features: np.ndarray, labels: np.ndarray):
        """Analyze Spectral Clustering regimes"""
        # Similar to previous analyses
        regime_data = {}
        
        for i in range(self.n_regimes):
            regime_mask = labels == i
            if np.sum(regime_mask) == 0:
                continue
                
            regime_features = features[regime_mask]
            mean_features = np.mean(regime_features, axis=0)
            
            regime_data[i] = {
                'mean': mean_features,
                'count': np.sum(regime_mask),
                'frequency': np.sum(regime_mask) / len(labels)
            }
            
        # Use same indices as before
        volatility_idx = 0
        trend_idx = 7
        
        # Assign labels
        for i, data in regime_data.items():
            volatility = data['mean'][volatility_idx]
            trend = data['mean'][trend_idx]
            
            # Similar logic to previous methods
            vol_values = [r['mean'][volatility_idx] for r in regime_data.values()]
            trend_values = [abs(r['mean'][trend_idx]) for r in regime_data.values()]
            
            if volatility > np.median(vol_values) * 1.3:
                if trend > 0:
                    label = "Volatile Bullish"
                else:
                    label = "Volatile Bearish"
            elif abs(trend) > np.median(trend_values) * 1.2:
                if trend > 0:
                    label = "Strong Bullish Trend"
                else:
                    label = "Strong Bearish Trend"
            else:
                if volatility < np.median(vol_values) * 0.7:
                    label = "Low Volatility Range"
                else:
                    label = "Neutral/Range Bound"
            
            # Store label
            self.regime_labels[f"spectral_{i}"] = label
            self.logger.info(f"Spectral Regime {i} interpreted as: {label}")
    
    def predict_regime_spectral(self, prices: np.ndarray, volumes: np.ndarray = None) -> np.ndarray:
        """
        Predict regimes using Spectral Clustering
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Array of regime labels
        """
        if self.spectral_model is None:
            self.fit_spectral_clustering(prices, volumes)
            
        # For prediction with spectral clustering, we use nearest neighbors
        # to assign new points to clusters since spectral clustering itself
        # doesn't have a predict method
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.transform(features)
        
        # Use kNN to assign to clusters
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.spectral_features, self.spectral_labels)
        
        return knn.predict(scaled_features)
    
    def fit_ensemble(self, prices: np.ndarray, volumes: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Fit all regime detection models and create an ensemble
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Tuple of (ensemble regime predictions, model dictionary)
        """
        self.logger.info("Fitting regime detection ensemble")
        
        # Fit individual models
        features = self._extract_advanced_features(prices, volumes)
        scaled_features = self.scaler.fit_transform(features)
        
        # First, run all individual models
        hmm_model = self.fit_hmm(prices, volumes)
        hmm_regimes, _ = self.predict_regime_hmm(prices, volumes)
        
        kmeans_model = self.fit_kmeans(prices, volumes)
        kmeans_regimes = self.predict_regime_kmeans(prices, volumes)
        
        if self.ensemble_weights[2] > 0:
            spectral_model = self.fit_spectral_clustering(prices, volumes)
            spectral_regimes = self.predict_regime_spectral(prices, volumes)
        else:
            spectral_regimes = None
        
        # Now create the ensemble prediction
        ensemble_regimes = self._combine_regime_predictions(
            hmm_regimes, kmeans_regimes, spectral_regimes
        )
        
        # Analyze the ensemble regimes
        self._analyze_ensemble_regimes(features, ensemble_regimes)
        
        return ensemble_regimes
    
    def _combine_regime_predictions(self, hmm_regimes, kmeans_regimes, spectral_regimes=None):
        """
        Combine predictions from multiple models into ensemble prediction
        
        Args:
            hmm_regimes: Regime predictions from HMM
            kmeans_regimes: Regime predictions from KMeans
            spectral_regimes: Regime predictions from Spectral Clustering
            
        Returns:
            Array of ensemble regime labels
        """
        # Normalize the weights
        weights = np.array(self.ensemble_weights)
        if spectral_regimes is None:
            weights = weights[:2]
            weights = weights / np.sum(weights)
        else:
            weights = weights / np.sum(weights)
        
        # Align regime labels across methods (they might have different numbering)
        regimes_list = [hmm_regimes, kmeans_regimes]
        if spectral_regimes is not None:
            regimes_list.append(spectral_regimes)
        
        # Use Hungarian algorithm to find optimal matching between regimes
        from scipy.optimize import linear_sum_assignment
        aligned_regimes = []
        
        # Use the first method (HMM) as reference
        reference = hmm_regimes
        aligned_regimes.append(reference)
        
        # Align others to reference
        for other_regimes in regimes_list[1:]:
            # Build confusion matrix
            matrix = np.zeros((self.n_regimes, self.n_regimes))
            for i in range(len(reference)):
                if i < len(other_regimes):
                    matrix[reference[i], other_regimes[i]] += 1
            
            # Find optimal assignment
            row_ind, col_ind = linear_sum_assignment(-matrix)
            
            # Create mapping
            regime_map = {old: new for old, new in zip(range(self.n_regimes), col_ind)}
            
            # Map regimes
            aligned = np.array([regime_map.get(r, r) for r in other_regimes])
            aligned_regimes.append(aligned)
        
        # Weighted voting for final regime
        ensemble_votes = np.zeros((len(reference), self.n_regimes))
        
        for i, (regimes, w) in enumerate(zip(aligned_regimes, weights)):
            # One-hot encode and weight
            for j in range(len(regimes)):
                if j < len(ensemble_votes):
                    ensemble_votes[j, regimes[j]] += w
        
        # Take regime with highest weight for each time point
        ensemble_regimes = np.argmax(ensemble_votes, axis=1)
        
        return ensemble_regimes
    
    def _analyze_ensemble_regimes(self, features: np.ndarray, regimes: np.ndarray):
        """Analyze the ensemble regimes"""
        # Similar to previous analyses
        regime_data = {}
        
        for i in range(self.n_regimes):
            regime_mask = regimes == i
            if np.sum(regime_mask) == 0:
                continue
                
            regime_features = features[regime_mask]
            mean_features = np.mean(regime_features, axis=0)
            
            regime_data[i] = {
                'mean': mean_features,
                'count': np.sum(regime_mask),
                'frequency': np.sum(regime_mask) / len(regimes)
            }
            
        # Assign labels using same logic as before
        volatility_idx = 0
        trend_idx = 7
        
        for i, data in regime_data.items():
            volatility = data['mean'][volatility_idx]
            trend = data['mean'][trend_idx]
            
            vol_values = [r['mean'][volatility_idx] for r in regime_data.values()]
            trend_values = [abs(r['mean'][trend_idx]) for r in regime_data.values()]
            
            if volatility > np.median(vol_values) * 1.3:
                if trend > 0:
                    label = "Volatile Bullish"
                else:
                    label = "Volatile Bearish"
            elif abs(trend) > np.median(trend_values) * 1.2:
                if trend > 0:
                    label = "Strong Bullish Trend"
                else:
                    label = "Strong Bearish Trend"
            elif abs(trend) > np.median(trend_values) * 0.8:
                if trend > 0:
                    label = "Moderate Bullish Trend"
                else:
                    label = "Moderate Bearish Trend"
            else:
                if volatility < np.median(vol_values) * 0.7:
                    label = "Low Volatility Range"
                else:
                    label = "Neutral/Range Bound"
            
            # Store label
            self.regime_labels[f"ensemble_{i}"] = label
            self.logger.info(f"Ensemble Regime {i} interpreted as: {label}")
    
    def predict_regime(self, prices: np.ndarray, volumes: np.ndarray = None) -> Tuple[np.ndarray, Dict]:
        """
        Predict regimes using all available models
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            
        Returns:
            Tuple of (regime predictions, confidence scores)
        """
        # Check if models are fitted
        if self.hmm_model is None:
            self.fit_ensemble(prices, volumes)
            
        # Get predictions from all models
        hmm_regimes, hmm_probs = self.predict_regime_hmm(prices, volumes)
        kmeans_regimes = self.predict_regime_kmeans(prices, volumes)
        
        if self.use_ensemble and self.ensemble_weights[2] > 0:
            spectral_regimes = self.predict_regime_spectral(prices, volumes)
        else:
            spectral_regimes = None
        
        # Combine predictions if using ensemble
        if self.use_ensemble:
            ensemble_regimes = self._combine_regime_predictions(
                hmm_regimes, kmeans_regimes, spectral_regimes
            )
            
            # Calculate confidence scores
            confidence = {}
            
            # HMM provides direct probabilities
            for i in range(len(hmm_regimes)):
                if i < len(hmm_probs):
                    regime = ensemble_regimes[i]
                    confidence[i] = {
                        'regime': regime,
                        'probability': float(np.max(hmm_probs[i])),
                        'label': self.regime_labels.get(f"ensemble_{regime}", f"Regime {regime}")
                    }
            
            return ensemble_regimes, confidence
        else:
            # If not using ensemble, return HMM regimes and probabilities
            confidence = {}
            for i in range(len(hmm_regimes)):
                if i < len(hmm_probs):
                    regime = hmm_regimes[i]
                    confidence[i] = {
                        'regime': regime,
                        'probability': float(np.max(hmm_probs[i])),
                        'label': self.regime_labels.get(f"hmm_{regime}", f"Regime {regime}")
                    }
            
            return hmm_regimes, confidence
    
    def detect_regime_transitions(self, regimes: np.ndarray, threshold: int = 3) -> List[int]:
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
            if all(regimes[i-j] == regimes[i] for j in range(threshold)) and regimes[i] != regimes[i-threshold-1]:
                change_points.append(i-threshold)
                
        return change_points
    
    def predict_next_regime_transition(self, prices: np.ndarray, regimes: np.ndarray, 
                                     lookback: int = 60) -> Tuple[float, int, str]:
        """
        Predict likelihood of regime transition in the near future
        
        Args:
            prices: Price series
            regimes: Current regime assignments
            lookback: Amount of history to consider
            
        Returns:
            Tuple of (transition_probability, expected_periods, likely_next_regime)
        """
        if len(regimes) < lookback:
            return 0.0, -1, "unknown"
            
        # Extract recent history
        recent_regimes = regimes[-lookback:]
        current_regime = regimes[-1]
        
        # Calculate statistics of regime durations
        durations = {}
        transitions = {}
        
        current_regime_idx = 0
        current_duration = 1
        
        for i in range(1, len(regimes)):
            if regimes[i] == regimes[i-1]:
                current_duration += 1
            else:
                # Record duration
                regime = regimes[i-1]
                if regime not in durations:
                    durations[regime] = []
                durations[regime].append(current_duration)
                
                # Record transition
                if regime not in transitions:
                    transitions[regime] = {}
                    
                next_regime = regimes[i]
                if next_regime not in transitions[regime]:
                    transitions[regime][next_regime] = 0
                    
                transitions[regime][next_regime] += 1
                
                # Reset duration counter
                current_duration = 1
        
        # Add the final regime's duration
        if regimes[-1] not in durations:
            durations[regimes[-1]] = []
        durations[regimes[-1]].append(current_duration)
        
        # Calculate average duration for the current regime
        if current_regime in durations and len(durations[current_regime]) > 0:
            avg_duration = np.mean(durations[current_regime])
            current_duration = 1
            
            # Count backward to find current duration
            for i in range(len(regimes)-2, -1, -1):
                if regimes[i] == current_regime:
                    current_duration += 1
                else:
                    break
            
            # Calculate transition probability
            transition_probability = current_duration / avg_duration
            transition_probability = min(0.95, transition_probability)  # Cap at 95%
            
            # Calculate remaining periods
            remaining_periods = max(1, int(avg_duration - current_duration))
            
            # Determine likely next regime
            likely_next_regime = "unknown"
            if current_regime in transitions and transitions[current_regime]:
                likely_next_regime = max(transitions[current_regime].items(), key=lambda x: x[1])[0]
                
            return transition_probability, remaining_periods, str(likely_next_regime)
        else:
            return 0.1, -1, "unknown"  # Default values if no history
    
    def plot_regimes(self, prices: np.ndarray, regimes: np.ndarray, 
                   volumes: np.ndarray = None,
                   change_points: List[int] = None, 
                   title: str = 'Enhanced Market Regimes'):
        """
        Create advanced visualization of market regimes
        
        Args:
            prices: Array of price values
            regimes: Array of regime labels
            volumes: Optional volume data
            change_points: Indices of regime change points
            title: Plot title
        """
        plt.figure(figsize=(16, 10))
        
        # Create a 3-row subplot
        gs = plt.GridSpec(3, 1, height_ratios=[3, 1, 1])
        
        # First subplot - prices and regimes
        ax1 = plt.subplot(gs[0])
        
        # Plot price series
        ax1.plot(prices, color='black', alpha=0.7, linewidth=1.5)
        
        # Color regions by regime
        offset = len(prices) - len(regimes)
        
        # Create a colormap for regimes
        colors = ['lightblue', 'lightgreen', 'salmon', 'orchid', 'orange', 'cyan',
                 'lightgray', 'yellow']
        
        for i in range(len(regimes)):
            ax1.axvspan(i+offset, i+offset+1, alpha=0.3, color=colors[regimes[i] % len(colors)])
        
        # Mark regime changes
        if change_points:
            for cp in change_points:
                plt.axvline(x=cp+offset, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # Add labels for regimes
        legend_elements = []
        from matplotlib.patches import Patch
        
        for i in range(self.n_regimes):
            key = f"ensemble_{i}" if self.use_ensemble else f"hmm_{i}"
            if key in self.regime_labels:
                label = self.regime_labels[key]
                legend_elements.append(Patch(facecolor=colors[i % len(colors)], 
                                            alpha=0.3, label=label))
        
        ax1.legend(handles=legend_elements, loc='upper left')
        ax1.set_title(title)
        ax1.set_ylabel('Price')
        ax1.set_xticklabels([])  # Hide x-axis labels for top plot
        
        # Second subplot - volatility
        ax2 = plt.subplot(gs[1], sharex=ax1)
        
        # Calculate volatility (20-day rolling)
        window = min(20, len(prices)//10)
        returns = np.diff(prices) / prices[:-1]
        volatility = np.zeros(len(returns))
        
        for i in range(window, len(returns)):
            volatility[i] = np.std(returns[i-window:i]) * np.sqrt(252)  # Annualized
        
        # Plot volatility
        ax2.plot(np.arange(len(volatility)) + 1, volatility, color='darkblue', alpha=0.7)
        ax2.set_ylabel('Volatility')
        ax2.set_xticklabels([])  # Hide x-axis labels for middle plot
        
        # Third subplot - volume if available, otherwise momentum
        ax3 = plt.subplot(gs[2], sharex=ax1)
        
        if volumes is not None and len(volumes) >= len(prices):
            # Plot volume
            ax3.bar(np.arange(len(volumes[:len(prices)])), volumes[:len(prices)], color='gray', alpha=0.5)
            ax3.set_ylabel('Volume')
        else:
            # Calculate momentum (price / 20-day MA)
            momentum = np.zeros(len(prices))
            for i in range(window, len(prices)):
                momentum[i] = prices[i] / np.mean(prices[i-window:i]) - 1
            
            ax3.plot(momentum, color='darkgreen', alpha=0.7)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.2)
            ax3.set_ylabel('Momentum')
        
        ax3.set_xlabel('Time')
        
        plt.tight_layout()
        
        # Save figure
        save_dir = os.path.join(os.getcwd(), 'models', 'plots')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        filename = f'enhanced_regimes_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        plt.savefig(os.path.join(save_dir, filename))
        self.logger.info(f"Enhanced regime plot saved to {filename}")
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save enhanced regime detection models
        
        Args:
            filepath: Path to save models
        """
        model_data = {
            'hmm_model': self.hmm_model,
            'kmeans_model': self.kmeans_model,
            'spectral_features': getattr(self, 'spectral_features', None),
            'spectral_labels': getattr(self, 'spectral_labels', None),
            'scaler': self.scaler,
            'pca': self.pca,
            'regime_labels': self.regime_labels,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_data, filepath)
        self.logger.info(f"Enhanced regime model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load enhanced regime detection models
        
        Args:
            filepath: Path to load models from
        """
        model_data = joblib.load(filepath)
        
        self.hmm_model = model_data.get('hmm_model')
        self.kmeans_model = model_data.get('kmeans_model')
        self.spectral_features = model_data.get('spectral_features')
        self.spectral_labels = model_data.get('spectral_labels')
        self.scaler = model_data.get('scaler')
        self.pca = model_data.get('pca')
        self.regime_labels = model_data.get('regime_labels', {})
        self.feature_importance = model_data.get('feature_importance')
        self.config = model_data.get('config', self.config)
        
        self.logger.info(f"Enhanced regime model loaded from {filepath}")
    
    def generate_regime_features(self, prices: np.ndarray, 
                              volumes: np.ndarray = None,
                              include_probabilities: bool = True) -> np.ndarray:
        """
        Generate regime-based features for model input
        
        Args:
            prices: Array of price values
            volumes: Optional volume data
            include_probabilities: Whether to include regime probabilities as features
            
        Returns:
            Array of regime features
        """
        # Get regime predictions
        regimes, confidence = self.predict_regime(prices, volumes)
        
        # For the most recent data points, create features
        regime_features = []
        offset = len(prices) - len(regimes)
        
        for i in range(len(regimes)):
            current_regime = regimes[i]
            
            # One-hot encode the current regime
            one_hot = np.zeros(self.n_regimes)
            one_hot[current_regime] = 1
            
            # Get regime probability if available
            prob_features = []
            if include_probabilities and i in confidence:
                prob_features = [confidence[i]['probability']]
            
            # Count days in current regime
            days_in_regime = 1
            j = i - 1
            while j >= 0 and regimes[j] == current_regime:
                days_in_regime += 1
                j -= 1
            
            # Normalize days in regime
            days_feature = min(days_in_regime / 30.0, 1.0)  # Cap at 1 month
                
            # Detect if close to regime change (recent regime switches)
            regime_switches = 0
            for j in range(max(0, i-5), i):
                if j > 0 and regimes[j] != regimes[j-1]:
                    regime_switches += 1
            
            # Predict likelihood of regime transition
            transition_prob, periods_to_change, _ = self.predict_next_regime_transition(
                prices[:offset+i+1], regimes[:i+1]
            )
            
            # Normalize periods to change
            if periods_to_change > 0:
                periods_feature = min(periods_to_change / 20.0, 1.0)  # Cap at 20 days
            else:
                periods_feature = 0.0
            
            # Create feature vector
            features = np.concatenate([
                one_hot,                           # Current regime (one-hot)
                prob_features,                     # Regime probability
                [days_feature],                    # Days in current regime (normalized)
                [regime_switches / 3.0],           # Recent regime switches (normalized)
                [transition_prob],                 # Probability of transition
                [periods_feature]                  # Periods until likely change (normalized)
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
    
    # Test the enhanced regime detector
    detector = EnhancedRegimeDetector(config={'n_regimes': 4})
    
    # Fit and predict
    regimes = detector.fit_ensemble(prices)
    
    # Plot results
    change_points = detector.detect_regime_transitions(regimes)
    detector.plot_regimes(prices, regimes, change_points=change_points, title='Enhanced Regime Detection')