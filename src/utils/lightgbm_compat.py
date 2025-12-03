#!/usr/bin/env python3
"""
LightGBM Compatibility Module

This module provides fallback mechanisms when LightGBM cannot be installed
on systems like macOS (especially with Apple Silicon).
"""

import logging
import sys
import os

logger = logging.getLogger(__name__)

# Try to import LightGBM
LIGHTGBM_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
    logger.info(f"Using LightGBM version {lgb.__version__}")
except ImportError:
    logger.warning("LightGBM not available. Using scikit-learn fallback models instead.")
    lgb = None

# If LightGBM is not available, provide fallback implementations
if not LIGHTGBM_AVAILABLE:
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    
    class LGBMClassifier:
        """Fallback implementation of LGBMClassifier using scikit-learn's GradientBoostingClassifier"""
        
        def __init__(self, **kwargs):
            logger.warning("Using scikit-learn GradientBoostingClassifier as fallback for LGBMClassifier")
            self.model = GradientBoostingClassifier(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3)
            )
        
        def fit(self, X, y, **kwargs):
            return self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
        
        def predict_proba(self, X):
            return self.model.predict_proba(X)
        
        def get_params(self, deep=True):
            return self.model.get_params(deep)
        
        def set_params(self, **params):
            return self.model.set_params(**params)

    class LGBMRegressor:
        """Fallback implementation of LGBMRegressor using scikit-learn's GradientBoostingRegressor"""
        
        def __init__(self, **kwargs):
            logger.warning("Using scikit-learn GradientBoostingRegressor as fallback for LGBMRegressor")
            self.model = GradientBoostingRegressor(
                n_estimators=kwargs.get('n_estimators', 100),
                learning_rate=kwargs.get('learning_rate', 0.1),
                max_depth=kwargs.get('max_depth', 3)
            )
        
        def fit(self, X, y, **kwargs):
            return self.model.fit(X, y)
        
        def predict(self, X):
            return self.model.predict(X)
        
        def get_params(self, deep=True):
            return self.model.get_params(deep)
        
        def set_params(self, **params):
            return self.model.set_params(**params)

else:
    # If LightGBM is available, just expose the real classes
    LGBMClassifier = lgb.LGBMClassifier
    LGBMRegressor = lgb.LGBMRegressor

# Expose functions and classes to make imports cleaner
__all__ = ['LGBMClassifier', 'LGBMRegressor', 'LIGHTGBM_AVAILABLE', 'lgb']