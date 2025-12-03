#!/usr/bin/env python3
"""
Ensemble model optimizer module that finds optimal weights
for ensemble models using various optimization techniques
"""
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Union, Tuple, Any, Callable, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
import os
from datetime import datetime
import joblib
from scipy.optimize import minimize, differential_evolution, dual_annealing

# Configure logging
logger = logging.getLogger("ensemble_optimizer")

class EnsembleOptimizer:
    """
    Class for optimizing ensemble model weights
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the ensemble optimizer
        
        Args:
            config: Configuration dictionary with optimizer parameters
        """
        self.config = config or {}
        self.best_weights = None
        self.best_score = -np.inf
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y, y_pred: precision_score(y, y_pred, zero_division=0),
            'recall': lambda y, y_pred: recall_score(y, y_pred, zero_division=0),
            'f1': lambda y, y_pred: f1_score(y, y_pred, zero_division=0),
            'roc_auc': roc_auc_score
        }
        
        # Dynamic optimization parameters
        self.model_performance_history = {}  # Tracks historical performance of models
        self.weight_history = {}             # Tracks history of weight adjustments
        self.dynamic_window_size = self.config.get('dynamic_window_size', 30)  # Window size for dynamic optimization
        self.weight_momentum = self.config.get('weight_momentum', 0.8)  # Momentum factor for smooth weight transitions
        self.performance_decay = self.config.get('performance_decay', 0.95)  # Decay factor for older performance
        
    def optimize_weights(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                      method: str = 'grid_search', metric: str = 'f1',
                      n_trials: int = 100, save_path: str = None) -> Dict[str, Any]:
        """
        Optimize ensemble weights using various methods
        
        Args:
            models_proba: List of model probability predictions (each with shape n_samples)
            y_true: True labels
            method: Optimization method ('grid_search', 'scipy', 'evolutionary', 'annealing')
            metric: Metric to optimize
            n_trials: Number of trials for grid search
            save_path: Path to save optimization results
            
        Returns:
            Dictionary with optimization results
        """
        # Check inputs
        if not models_proba:
            raise ValueError("No models provided for optimization")
        
        n_models = len(models_proba)
        logger.info(f"Optimizing weights for {n_models} models using {method} method")
        
        # Convert all predictions to numpy arrays for consistency
        models_proba = [np.array(proba).flatten() for proba in models_proba]
        y_true = np.array(y_true).flatten()
        
        # Select optimization method
        if method == 'grid_search':
            weights, score = self._grid_search_weights(models_proba, y_true, metric, n_trials)
        elif method == 'scipy':
            weights, score = self._scipy_optimize_weights(models_proba, y_true, metric)
        elif method == 'evolutionary':
            weights, score = self._evolutionary_optimize_weights(models_proba, y_true, metric)
        elif method == 'annealing':
            weights, score = self._annealing_optimize_weights(models_proba, y_true, metric)
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Store best weights and score
        self.best_weights = weights
        self.best_score = score
        
        # Create result dictionary
        result = {
            'weights': weights,
            'score': score,
            'metric': metric,
            'method': method,
            'n_models': n_models
        }
        
        # Save results if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            joblib.dump(result, save_path)
            logger.info(f"Saved optimization results to {save_path}")
        
        logger.info(f"Optimal weights: {weights}")
        logger.info(f"Best {metric} score: {score:.4f}")
        
        return result
    
    def optimize_weights_dynamic(self, models_proba_dict: Dict[str, np.ndarray], y_true: np.ndarray,
                             timestamp: datetime = None, metric: str = 'f1',
                             save_history: bool = True) -> Dict[str, Any]:
        """
        Dynamically optimize ensemble weights based on recent performance
        
        Args:
            models_proba_dict: Dictionary of model name to probability predictions
            y_true: True labels
            timestamp: Timestamp for this optimization run (defaults to now)
            metric: Metric to optimize
            save_history: Whether to save history of performance and weights
            
        Returns:
            Dictionary with optimization results including dynamic weights
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Convert models_proba_dict to list format for internal use
        model_names = list(models_proba_dict.keys())
        models_proba = [models_proba_dict[name] for name in model_names]
        
        # Calculate current performance for each model
        current_performance = {}
        for i, name in enumerate(model_names):
            y_pred = (models_proba[i] >= 0.5).astype(int)
            score = self._calculate_metric(y_true, y_pred, models_proba[i], metric)
            current_performance[name] = score
            
        # Update performance history
        for name, score in current_performance.items():
            if name not in self.model_performance_history:
                self.model_performance_history[name] = []
            self.model_performance_history[name].append((timestamp, score))
            
        # Prune history to keep only recent window
        for name in self.model_performance_history:
            if len(self.model_performance_history[name]) > self.dynamic_window_size:
                self.model_performance_history[name] = self.model_performance_history[name][-self.dynamic_window_size:]
        
        # Calculate dynamic weights based on recent performance
        dynamic_weights = self._calculate_dynamic_weights(model_names, metric)
        
        # Apply smoothing using previous weights and momentum if available
        if model_names[0] in self.weight_history and len(self.weight_history[model_names[0]]) > 0:
            for i, name in enumerate(model_names):
                prev_weight = self.weight_history[name][-1][1]
                dynamic_weights[i] = self.weight_momentum * prev_weight + (1 - self.weight_momentum) * dynamic_weights[i]
                
        # Normalize weights to sum to 1
        dynamic_weights = dynamic_weights / np.sum(dynamic_weights)
        
        # Update weight history
        for i, name in enumerate(model_names):
            if name not in self.weight_history:
                self.weight_history[name] = []
            self.weight_history[name].append((timestamp, dynamic_weights[i]))
            
        # Calculate weighted ensemble predictions with new weights
        ensemble_proba = self._weighted_average(models_proba, dynamic_weights)
        y_pred = (ensemble_proba >= 0.5).astype(int)
        dynamic_score = self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
        
        # Create result dictionary with weights mapped to model names
        weights_dict = {name: dynamic_weights[i] for i, name in enumerate(model_names)}
        
        result = {
            'dynamic_weights': weights_dict,
            'weights_array': dynamic_weights, 
            'model_names': model_names,
            'score': dynamic_score,
            'metric': metric,
            'timestamp': timestamp,
            'model_performance': current_performance
        }
        
        logger.info(f"Dynamic weight optimization: {metric} = {dynamic_score:.4f}")
        logger.info(f"Dynamic weights: {weights_dict}")
        
        return result
        
    def _calculate_dynamic_weights(self, model_names: List[str], metric: str) -> np.ndarray:
        """
        Calculate dynamic weights based on recent performance history
        
        Args:
            model_names: List of model names
            metric: Metric being optimized
            
        Returns:
            Array of dynamically optimized weights
        """
        n_models = len(model_names)
        dynamic_weights = np.ones(n_models)
        
        # If no history, use equal weights
        if not self.model_performance_history or not all(name in self.model_performance_history for name in model_names):
            logger.info("Insufficient history for dynamic weights, using equal weights")
            return dynamic_weights / n_models
            
        # Calculate weighted average of recent performance for each model
        for i, name in enumerate(model_names):
            history = self.model_performance_history[name]
            if not history:
                dynamic_weights[i] = 1.0
                continue
                
            # Apply exponential decay to emphasize recent performance
            # Newer entries have higher weight in the average
            scores = [score for _, score in history]
            timestamps = [(ts - history[0][0]).total_seconds() for ts, _ in history]
            max_time = max(timestamps) if timestamps else 1
            
            if max_time > 0:
                # Normalize timestamps to [0,1] range and apply decay
                time_weights = np.array([self.performance_decay ** ((max_time - t) / max_time) for t in timestamps])
                time_weights = time_weights / np.sum(time_weights)  # Normalize weights
                
                # Calculate weighted performance
                weighted_performance = np.sum(np.array(scores) * time_weights)
                dynamic_weights[i] = max(0.001, weighted_performance)  # Ensure positive weight
            else:
                dynamic_weights[i] = scores[-1]  # Use latest score
        
        # Normalize weights to sum to 1
        return dynamic_weights / np.sum(dynamic_weights)
        
    def get_model_performance_trend(self, model_name: str, as_dataframe: bool = False) -> Union[List[Tuple[datetime, float]], pd.DataFrame]:
        """
        Get historical performance trend for a specific model
        
        Args:
            model_name: Name of the model
            as_dataframe: Whether to return as pandas DataFrame
            
        Returns:
            List of (timestamp, score) tuples or DataFrame with performance history
        """
        if model_name not in self.model_performance_history:
            logger.warning(f"No performance history for model {model_name}")
            return [] if not as_dataframe else pd.DataFrame(columns=['timestamp', 'score'])
            
        history = self.model_performance_history[model_name]
        
        if as_dataframe:
            return pd.DataFrame(history, columns=['timestamp', 'score'])
        return history
        
    def get_weight_trend(self, model_name: str, as_dataframe: bool = False) -> Union[List[Tuple[datetime, float]], pd.DataFrame]:
        """
        Get historical weight trend for a specific model
        
        Args:
            model_name: Name of the model
            as_dataframe: Whether to return as pandas DataFrame
            
        Returns:
            List of (timestamp, weight) tuples or DataFrame with weight history
        """
        if model_name not in self.weight_history:
            logger.warning(f"No weight history for model {model_name}")
            return [] if not as_dataframe else pd.DataFrame(columns=['timestamp', 'weight'])
            
        history = self.weight_history[model_name]
        
        if as_dataframe:
            return pd.DataFrame(history, columns=['timestamp', 'weight'])
        return history
        
    def plot_performance_trends(self, save_path: Optional[str] = None):
        """
        Plot performance trends for all models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.model_performance_history:
            logger.warning("No performance history to plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        for name, history in self.model_performance_history.items():
            timestamps = [ts for ts, _ in history]
            scores = [score for _, score in history]
            plt.plot(timestamps, scores, marker='o', label=name)
            
        plt.title('Model Performance Trends')
        plt.xlabel('Time')
        plt.ylabel('Performance Score')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Performance trends plot saved to {save_path}")
        else:
            plt.show()
            
    def plot_weight_trends(self, save_path: Optional[str] = None):
        """
        Plot weight trends for all models
        
        Args:
            save_path: Path to save the plot
        """
        if not self.weight_history:
            logger.warning("No weight history to plot")
            return
            
        plt.figure(figsize=(12, 6))
        
        for name, history in self.weight_history.items():
            timestamps = [ts for ts, _ in history]
            weights = [weight for _, weight in history]
            plt.plot(timestamps, weights, marker='o', label=name)
            
        plt.title('Model Weight Trends')
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            logger.info(f"Weight trends plot saved to {save_path}")
        else:
            plt.show()

    def save_optimization_history(self, save_path: str):
        """
        Save the optimization history to a file
        
        Args:
            save_path: Path to save the history
        """
        history = {
            'model_performance_history': self.model_performance_history,
            'weight_history': self.weight_history
        }
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(history, save_path)
        logger.info(f"Optimization history saved to {save_path}")
        
    def load_optimization_history(self, load_path: str):
        """
        Load optimization history from a file
        
        Args:
            load_path: Path to load the history from
        """
        if not os.path.exists(load_path):
            logger.warning(f"History file {load_path} not found")
            return
            
        history = joblib.load(load_path)
        self.model_performance_history = history.get('model_performance_history', {})
        self.weight_history = history.get('weight_history', {})
        logger.info(f"Loaded optimization history from {load_path}")
    
    def _grid_search_weights(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                          metric: str = 'f1', n_trials: int = 100) -> Tuple[np.ndarray, float]:
        """
        Find optimal weights using random grid search
        
        Args:
            models_proba: List of model probability predictions
            y_true: True labels
            metric: Metric to optimize
            n_trials: Number of random trials
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        n_models = len(models_proba)
        best_weights = np.ones(n_models) / n_models  # Default: equal weights
        best_score = -np.inf
        
        # Special case for single model
        if n_models == 1:
            y_pred = (models_proba[0] >= 0.5).astype(int)
            score = self._calculate_metric(y_true, y_pred, models_proba[0], metric)
            return np.array([1.0]), score
        
        # Try equal weights first (baseline)
        equal_weights = np.ones(n_models) / n_models
        ensemble_proba = self._weighted_average(models_proba, equal_weights)
        y_pred = (ensemble_proba >= 0.5).astype(int)
        equal_score = self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
        
        best_weights = equal_weights
        best_score = equal_score
        
        logger.info(f"Baseline (equal weights): {metric} = {equal_score:.4f}")
        
        # Random search for better weights
        for i in range(n_trials):
            # Generate random weights that sum to 1
            weights = np.random.dirichlet(np.ones(n_models))
            
            # Calculate weighted ensemble predictions
            ensemble_proba = self._weighted_average(models_proba, weights)
            y_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate score
            score = self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
            
            # Update best if improved
            if score > best_score:
                best_score = score
                best_weights = weights
                logger.debug(f"New best weights found: {best_weights}, {metric} = {best_score:.4f}")
        
        logger.info(f"Grid search completed: best {metric} = {best_score:.4f}")
        return best_weights, best_score
    
    def _scipy_optimize_weights(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                             metric: str = 'f1') -> Tuple[np.ndarray, float]:
        """
        Find optimal weights using scipy's optimize.minimize
        
        Args:
            models_proba: List of model probability predictions
            y_true: True labels
            metric: Metric to optimize
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        n_models = len(models_proba)
        
        # Special case for single model
        if n_models == 1:
            y_pred = (models_proba[0] >= 0.5).astype(int)
            score = self._calculate_metric(y_true, y_pred, models_proba[0], metric)
            return np.array([1.0]), score
        
        # Define objective function to minimize (negative to maximize original metric)
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted ensemble predictions
            ensemble_proba = self._weighted_average(models_proba, weights)
            y_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate negative score
            score = -self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
            return score
        
        # Initial guess: equal weights
        x0 = np.ones(n_models) / n_models
        
        # Constraints: weights sum to 1 (handled in objective by normalization)
        
        # Bounds: all weights between 0 and 1
        bounds = [(0.001, 1.0)] * n_models  # Small positive lower bound to avoid degenerate solutions
        
        # Run optimization
        result = minimize(
            objective,
            x0, 
            method='L-BFGS-B',
            bounds=bounds
        )
        
        # Normalize final weights
        weights = result.x / np.sum(result.x)
        score = -result.fun  # Convert back to positive score
        
        logger.info(f"Scipy optimization completed: best {metric} = {score:.4f}")
        return weights, score
    
    def _evolutionary_optimize_weights(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                                    metric: str = 'f1') -> Tuple[np.ndarray, float]:
        """
        Find optimal weights using differential evolution
        
        Args:
            models_proba: List of model probability predictions
            y_true: True labels
            metric: Metric to optimize
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        n_models = len(models_proba)
        
        # Special case for single model
        if n_models == 1:
            y_pred = (models_proba[0] >= 0.5).astype(int)
            score = self._calculate_metric(y_true, y_pred, models_proba[0], metric)
            return np.array([1.0]), score
        
        # Define objective function to minimize (negative to maximize original metric)
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted ensemble predictions
            ensemble_proba = self._weighted_average(models_proba, weights)
            y_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate negative score
            score = -self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
            return score
        
        # Bounds: all weights between 0 and 1
        bounds = [(0.001, 1.0)] * n_models
        
        # Run differential evolution
        result = differential_evolution(
            objective,
            bounds,
            strategy='best1bin',
            maxiter=50,
            popsize=15,
            mutation=(0.5, 1.0),
            recombination=0.7
        )
        
        # Normalize final weights
        weights = result.x / np.sum(result.x)
        score = -result.fun  # Convert back to positive score
        
        logger.info(f"Evolutionary optimization completed: best {metric} = {score:.4f}")
        return weights, score
    
    def _annealing_optimize_weights(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                                 metric: str = 'f1') -> Tuple[np.ndarray, float]:
        """
        Find optimal weights using simulated annealing
        
        Args:
            models_proba: List of model probability predictions
            y_true: True labels
            metric: Metric to optimize
            
        Returns:
            Tuple of (best_weights, best_score)
        """
        n_models = len(models_proba)
        
        # Special case for single model
        if n_models == 1:
            y_pred = (models_proba[0] >= 0.5).astype(int)
            score = self._calculate_metric(y_true, y_pred, models_proba[0], metric)
            return np.array([1.0]), score
        
        # Define objective function to minimize (negative to maximize original metric)
        def objective(weights):
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate weighted ensemble predictions
            ensemble_proba = self._weighted_average(models_proba, weights)
            y_pred = (ensemble_proba >= 0.5).astype(int)
            
            # Calculate negative score
            score = -self._calculate_metric(y_true, y_pred, ensemble_proba, metric)
            return score
        
        # Bounds: all weights between 0 and 1
        bounds = [(0.001, 1.0)] * n_models
        
        # Run dual annealing
        result = dual_annealing(
            objective,
            bounds,
            maxiter=100
        )
        
        # Normalize final weights
        weights = result.x / np.sum(result.x)
        score = -result.fun  # Convert back to positive score
        
        logger.info(f"Annealing optimization completed: best {metric} = {score:.4f}")
        return weights, score
    
    def _weighted_average(self, models_proba: List[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        Calculate weighted average of model predictions
        
        Args:
            models_proba: List of model probability predictions
            weights: Array of weights for each model
            
        Returns:
            Array of weighted average predictions
        """
        # Ensure weights are normalized
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate weighted sum
        weighted_sum = np.zeros_like(models_proba[0])
        for i, proba in enumerate(models_proba):
            weighted_sum += weights[i] * proba
            
        return weighted_sum
    
    def _calculate_metric(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        y_pred_proba: np.ndarray, metric: str) -> float:
        """
        Calculate specified metric
        
        Args:
            y_true: True labels
            y_pred: Predicted binary labels
            y_pred_proba: Predicted probabilities
            metric: Metric name
            
        Returns:
            Metric value
        """
        try:
            if metric == 'roc_auc':
                return roc_auc_score(y_true, y_pred_proba)
            elif metric in self.metrics:
                return self.metrics[metric](y_true, y_pred)
            else:
                logger.warning(f"Unknown metric: {metric}, using accuracy")
                return accuracy_score(y_true, y_pred)
        except Exception as e:
            logger.warning(f"Error calculating {metric}: {e}, returning -inf")
            return -np.inf
            
    def apply_ensemble_weights(self, models_proba: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
        """
        Apply ensemble weights to model predictions
        
        Args:
            models_proba: List of model probability predictions
            weights: Array of weights for each model (uses best_weights if None)
            
        Returns:
            Array of weighted ensemble predictions
        """
        weights = weights if weights is not None else self.best_weights
        
        if weights is None:
            # If no weights specified or optimized, use equal weights
            weights = np.ones(len(models_proba)) / len(models_proba)
            
        return self._weighted_average(models_proba, weights)
    
    def evaluate_ensemble(self, models_proba: List[np.ndarray], y_true: np.ndarray,
                       weights: np.ndarray = None) -> Dict[str, float]:
        """
        Evaluate ensemble model with specified or best weights
        
        Args:
            models_proba: List of model probability predictions
            y_true: True labels
            weights: Array of weights for each model (uses best_weights if None)
            
        Returns:
            Dictionary with evaluation metrics
        """
        weights = weights if weights is not None else self.best_weights
        
        # Use equal weights if no optimized weights available
        if weights is None:
            weights = np.ones(len(models_proba)) / len(models_proba)
            
        # Calculate weighted ensemble predictions
        ensemble_proba = self._weighted_average(models_proba, weights)
        y_pred = (ensemble_proba >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        for metric_name, metric_func in self.metrics.items():
            try:
                if metric_name == 'roc_auc':
                    metrics[metric_name] = metric_func(y_true, ensemble_proba)
                else:
                    metrics[metric_name] = metric_func(y_true, y_pred)
            except Exception as e:
                logger.warning(f"Error calculating {metric_name}: {e}")
                metrics[metric_name] = float('nan')
                
        return metrics