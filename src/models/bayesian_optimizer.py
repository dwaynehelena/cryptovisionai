#!/usr/bin/env python3
"""
Bayesian Optimizer for hyperparameter tuning
Uses Bayesian optimization to find optimal hyperparameters for machine learning models
"""

import numpy as np
import pandas as pd
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
import matplotlib.pyplot as plt

# Import scikit-optimize for Bayesian Optimization
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# Configure logging
logger = logging.getLogger("bayesian_optimizer")


class BayesianOptimizer:
    """
    Bayesian Optimizer for hyperparameter tuning
    Uses Gaussian Process to model the objective function and find optimal hyperparameters
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the Bayesian Optimizer
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.n_calls = self.config.get('n_calls', 30)  # Number of optimization iterations
        self.n_initial_points = self.config.get('n_initial_points', 10)  # Initial random points
        self.random_state = self.config.get('random_state', 42)
        self.verbose = self.config.get('verbose', True)
        self.save_dir = self.config.get('save_dir', 'models/ensemble/')
        self.optimization_history = []
        self.best_params = None
        self.best_score = None
        
        logger.info(f"Initialized BayesianOptimizer with {self.n_calls} iterations and {self.n_initial_points} initial points")
    
    def optimize_lightgbm(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters for LightGBM model
        
        Args:
            X: Training features
            y: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Optimized hyperparameters
        """
        import lightgbm as lgb
        from sklearn.model_selection import cross_val_score
        
        logger.info("Starting Bayesian optimization for LightGBM")
        
        # Define the search space for LightGBM hyperparameters
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(3, 15, name='max_depth'),
            Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Integer(1, 100, name='min_child_samples')
        ]
        
        # Define the objective function to minimize (negative cross-validation score)
        @use_named_args(space)
        def objective(**params):
            # Create LightGBM model with current parameters
            model = lgb.LGBMClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                min_child_samples=params['min_child_samples'],
                random_state=self.random_state
            )
            
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                mean_score = scores.mean()
                
                # We want to maximize the score, so return negative
                result = -mean_score
                
                if self.verbose:
                    logger.info(f"Parameters: {params}, Score: {mean_score:.4f}")
                
                # Store parameters and score in history
                self.optimization_history.append({
                    'params': params,
                    'score': mean_score,
                    'timestamp': datetime.now().isoformat()
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimization iteration: {e}")
                # Return a large value for failed runs
                return 999.0
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Extract best parameters
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4],
            'min_child_samples': result.x[5]
        }
        best_score = -result.fun
        
        # Save best parameters and score
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        # Save optimization results
        self._save_optimization_results()
        
        return best_params
    
    def optimize_xgboost(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters for XGBoost model
        
        Args:
            X: Training features
            y: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Optimized hyperparameters
        """
        import xgboost as xgb
        from sklearn.model_selection import cross_val_score
        
        logger.info("Starting Bayesian optimization for XGBoost")
        
        # Define the search space for XGBoost hyperparameters
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(3, 15, name='max_depth'),
            Real(0.01, 0.3, prior='log-uniform', name='learning_rate'),
            Real(0.5, 1.0, name='subsample'),
            Real(0.5, 1.0, name='colsample_bytree'),
            Real(0.01, 10.0, prior='log-uniform', name='gamma'),
            Real(0.01, 10.0, prior='log-uniform', name='min_child_weight')
        ]
        
        # Define the objective function to minimize (negative cross-validation score)
        @use_named_args(space)
        def objective(**params):
            # Create XGBoost model with current parameters
            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                gamma=params['gamma'],
                min_child_weight=params['min_child_weight'],
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                mean_score = scores.mean()
                
                # We want to maximize the score, so return negative
                result = -mean_score
                
                if self.verbose:
                    logger.info(f"Parameters: {params}, Score: {mean_score:.4f}")
                
                # Store parameters and score in history
                self.optimization_history.append({
                    'params': params,
                    'score': mean_score,
                    'timestamp': datetime.now().isoformat()
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimization iteration: {e}")
                # Return a large value for failed runs
                return 999.0
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Extract best parameters
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'learning_rate': result.x[2],
            'subsample': result.x[3],
            'colsample_bytree': result.x[4],
            'gamma': result.x[5],
            'min_child_weight': result.x[6]
        }
        best_score = -result.fun
        
        # Save best parameters and score
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        # Save optimization results
        self._save_optimization_results()
        
        return best_params
    
    def optimize_rf(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """
        Optimize hyperparameters for Random Forest model
        
        Args:
            X: Training features
            y: Training target
            cv: Number of cross-validation folds
            
        Returns:
            Dict[str, Any]: Optimized hyperparameters
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_score
        
        logger.info("Starting Bayesian optimization for Random Forest")
        
        # Define the search space for Random Forest hyperparameters
        space = [
            Integer(50, 300, name='n_estimators'),
            Integer(3, 15, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf'),
            Categorical(['sqrt', 'log2', None], name='max_features')
        ]
        
        # Define the objective function to minimize (negative cross-validation score)
        @use_named_args(space)
        def objective(**params):
            # Create Random Forest model with current parameters
            model = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                min_samples_split=params['min_samples_split'],
                min_samples_leaf=params['min_samples_leaf'],
                max_features=params['max_features'],
                random_state=self.random_state,
                n_jobs=-1
            )
            
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)
                mean_score = scores.mean()
                
                # We want to maximize the score, so return negative
                result = -mean_score
                
                if self.verbose:
                    logger.info(f"Parameters: {params}, Score: {mean_score:.4f}")
                
                # Store parameters and score in history
                self.optimization_history.append({
                    'params': params,
                    'score': mean_score,
                    'timestamp': datetime.now().isoformat()
                })
                
                return result
                
            except Exception as e:
                logger.error(f"Error in optimization iteration: {e}")
                # Return a large value for failed runs
                return 999.0
        
        # Run Bayesian optimization
        result = gp_minimize(
            objective,
            space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
            verbose=self.verbose
        )
        
        # Extract best parameters
        best_params = {
            'n_estimators': result.x[0],
            'max_depth': result.x[1],
            'min_samples_split': result.x[2],
            'min_samples_leaf': result.x[3],
            'max_features': result.x[4]
        }
        best_score = -result.fun
        
        # Save best parameters and score
        self.best_params = best_params
        self.best_score = best_score
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best score: {best_score:.4f}")
        
        # Save optimization results
        self._save_optimization_results()
        
        return best_params
    
    def plot_optimization_results(self, save_path: Optional[str] = None) -> None:
        """
        Plot the optimization results
        
        Args:
            save_path: Path to save the plot
        """
        if not hasattr(self, 'result') or self.result is None:
            logger.warning("No optimization results available to plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Plot convergence
        ax = plt.subplot(1, 1, 1)
        plot_convergence(self.result, ax=ax)
        
        plt.title('Optimization Convergence')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            logger.info(f"Optimization plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _save_optimization_results(self) -> None:
        """
        Save optimization results to a JSON file
        """
        os.makedirs(self.save_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.save_dir, f"optimization_results_{timestamp}.json")
        
        results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'history': self.optimization_history,
            'config': {
                'n_calls': self.n_calls,
                'n_initial_points': self.n_initial_points,
                'random_state': self.random_state,
            }
        }
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {results_path}")