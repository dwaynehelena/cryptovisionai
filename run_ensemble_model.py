#!/usr/bin/env python3
"""
Script to run the Ensemble Model with enhanced accuracy
"""

import os
import sys
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# Ensure the src directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", "ensemble_model_training.log")),
        logging.StreamHandler()
    ]
)

# Import the ensemble model and other required modules
from src.models.ensemble_model import EnsembleModel, DataAugmentation
from src.data_processing.binance_connector import BinanceConnector
from src.data_processing.feature_engineering import FeatureEngineering

class WalkForwardValidator:
    """Class for performing walk-forward validation on time series data"""
    
    def __init__(self, n_splits=5, test_size=0.2):
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y=None):
        """Generate indices for walk-forward validation
        
        Args:
            X: Features DataFrame with time series index
            y: Target variable
            
        Yields:
            train_indices, test_indices for each fold
        """
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        indices = np.arange(n_samples)
        
        # Generate splits that maintain temporal ordering
        for i in range(self.n_splits):
            # Calculate the end of each window, shifting forward each time
            end_point = n_samples - (self.n_splits - i - 1) * test_size
            start_point = max(0, end_point - test_size)
            
            # Get train and test indices for this split
            test_indices = indices[start_point:end_point]
            train_indices = indices[:start_point]
            
            yield train_indices, test_indices
    
    def get_performance(self, model, X, y, metric_func):
        """Evaluate model performance across all folds
        
        Args:
            model: Model with fit and predict methods
            X: Features
            y: Target
            metric_func: Function to evaluate performance
            
        Returns:
            Average performance across folds and individual fold performances
        """
        performances = []
        
        for train_idx, test_idx in self.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Clone model to avoid reusing the same instance
            model = clone(model)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            performance = metric_func(y_test, y_pred)
            performances.append(performance)
            
        return np.mean(performances), performances

def run_ensemble_model(symbols=None, feature_set="advanced", n_features=136, epochs=150, use_augmentation=True):
    """
    Run the ensemble model with enhanced accuracy
    
    Args:
        symbols: List of symbols to train on (default: ["BTCUSDT", "ETHUSDT", "BNBUSDT"])
        feature_set: Feature set to use (default: "advanced")
        n_features: Number of top features to select (default: 136)
        epochs: Number of epochs for deep learning models (default: 150)
        use_augmentation: Whether to use advanced data augmentation (default: True)
    """
    print(f"Running enhanced ensemble model training...")
    
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]
    
    # Initialize components
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    binance_connector = BinanceConnector(use_testnet=True, config_path=config_path)
    feature_engineering = FeatureEngineering()
    
    # Initialize data collection
    all_features = pd.DataFrame()
    all_targets = pd.Series()
    symbol_data = {}  # Store data by symbol for sentiment analysis later
    
    # Collect data for all symbols with longer history for better training
    for symbol in symbols:
        print(f"Fetching historical data for {symbol}...")
        
        # Get more historical data (increase limit for larger training dataset)
        interval = "1h"
        limit = 3000  # Increased from 2000 to 3000 for more training data
        
        df = binance_connector.get_historical_klines(symbol, interval, limit)
        if df is None or df.empty:
            print(f"Could not get data for {symbol}, skipping.")
            continue

        # Store raw data for sentiment analysis
        symbol_data[symbol] = df.copy()
        
        # Generate features with the specified feature set
        print(f"Generating features for {symbol} using {feature_set} feature set...")
        features_df = feature_engineering.extract_features(df, feature_set=feature_set)
        
        if features_df is None or features_df.empty:
            print(f"Failed to generate features for {symbol}, skipping.")
            continue
        
        # Drop NaN values
        features_df = features_df.dropna()
        
        # Add symbol as a one-hot feature
        features_df[f'symbol_{symbol}'] = 1
        
        # Extract features and target
        drop_cols = []
        if 'target_direction' in features_df.columns:
            drop_cols.append('target_direction')
            y = features_df['target_direction']
        else:
            # Create a default target if not available
            print(f"Warning: 'target_direction' not found in features, creating synthetic target...")
            # Create a binary target based on price movements (close vs previous close)
            # Select only the close prices that correspond to our features_df timestamps
            matching_df = df[df.index.isin(features_df.index)]
            
            if len(matching_df) != len(features_df):
                print(f"Warning: Length mismatch. Features: {len(features_df)}, Original data: {len(matching_df)}")
                # Ensure we're only using data points that exist in both dataframes
                common_index = features_df.index.intersection(matching_df.index)
                features_df = features_df.loc[common_index]
                matching_df = matching_df.loc[common_index]
                
            close_prices = matching_df['close'].values
            price_changes = np.diff(close_prices)
            # Pad with a zero at the beginning to maintain length
            price_changes = np.insert(price_changes, 0, 0)
            y = pd.Series((price_changes > 0).astype(int), index=features_df.index)
            features_df['target_direction'] = y
            drop_cols.append('target_direction')
        
        if 'target_return' in features_df.columns:
            drop_cols.append('target_return')
        
        # Now drop only columns that exist
        X = features_df.drop(drop_cols, axis=1) if drop_cols else features_df
        
        # Store in our collection
        if all_features.empty:
            all_features = X
            all_targets = y
        else:
            # Ensure columns match before concatenating
            common_cols = set(all_features.columns).intersection(set(X.columns))
            all_features = pd.concat([all_features[list(common_cols)], X[list(common_cols)]])
            all_targets = pd.concat([all_targets, y])
    
    if all_features.empty:
        print("No valid data collected. Exiting.")
        return
    
    print(f"Total dataset size: {len(all_features)} samples with {all_features.shape[1]} features")
    
    # Feature selection to improve model performance
    print("Performing feature selection...")
    selector = SelectKBest(f_classif, k=min(n_features, all_features.shape[1]))
    X_selected = selector.fit_transform(all_features, all_targets)
    
    # Get the selected feature names for analysis
    selected_indices = selector.get_support(indices=True)
    selected_features = all_features.columns[selected_indices].tolist()
    print(f"Selected top {len(selected_features)} features")
    
    # Create a feature importance plot
    plt.figure(figsize=(12, 6))
    feature_scores = selector.scores_[selected_indices]
    plt.bar(range(len(selected_features)), feature_scores)
    plt.xticks(range(len(selected_features)), [f[:15] for f in selected_features], rotation=90)
    plt.title('Feature Importance Scores')
    plt.tight_layout()
    
    # Save the feature importance plot
    os.makedirs("models/plots", exist_ok=True)
    plt.savefig("models/plots/feature_importance.png")
    print(f"Feature importance plot saved to models/plots/feature_importance.png")
    
    # Scale the features for better model performance
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    # Implementation of Walk-Forward Validation
    print("Performing walk-forward cross validation...")
    
    # Use WalkForwardValidator instead of TimeSeriesSplit for more robust validation
    n_folds = 5
    validator = WalkForwardValidator(n_splits=n_folds, test_size=0.15)
    
    # Store results for each fold
    fold_metrics = []
    best_metrics = {}
    
    # Initialize data augmentation if enabled
    augmentation_config = {
        'random_state': 42,
        'feature_map': {
            'volatility': [],  # Will be populated based on feature names
            'trend': [],       # Will be populated based on feature names
        }
    }
    
    # Identify volatility and trend features for better augmentation
    for i, feature in enumerate(selected_features):
        if any(term in feature.lower() for term in ['vol', 'std', 'rsi', 'atr']):
            augmentation_config['feature_map']['volatility'].append(i)
        if any(term in feature.lower() for term in ['ma', 'ema', 'trend', 'momentum', 'macd']):
            augmentation_config['feature_map']['trend'].append(i)
    
    data_augmenter = DataAugmentation(augmentation_config) if use_augmentation else None
    
    # Run training/validation across folds
    for fold, (train_indices, test_indices) in enumerate(validator.split(X_scaled)):
        print(f"\n=== Fold {fold+1}/{n_folds} ===")
        
        # Split data for this fold
        X_train = X_scaled[train_indices]
        y_train = all_targets.iloc[train_indices].values
        X_test = X_scaled[test_indices]
        y_test = all_targets.iloc[test_indices].values
        
        # Further split train into train and validation
        val_size = int(0.2 * len(X_train))
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_train = X_train[:-val_size]
        y_train = y_train[:-val_size]
        
        print(f"Fold {fold+1} - Training set: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")
        
        # Apply data augmentation if enabled
        if use_augmentation and data_augmenter:
            print("Applying data augmentation techniques...")
            
            # Getting raw price data for market regime detection (using first symbol as proxy)
            # This is simplified - in production you would want to use the actual price data for each sample
            if symbol_data and len(symbol_data) > 0:
                first_symbol = list(symbol_data.keys())[0]
                price_data = symbol_data[first_symbol]['close'].values[-len(X_train):]
                # Extend if needed
                if len(price_data) < len(X_train):
                    price_data = np.pad(price_data, (len(X_train) - len(price_data), 0), 'edge')
                
                # Calculate returns for regime detection
                returns = np.diff(price_data) / price_data[:-1]
                returns = np.insert(returns, 0, 0)  # Add leading zero for consistent length
                
                # Calculate volatility using rolling window
                window = 20
                volatility = np.zeros_like(price_data)
                for i in range(window, len(price_data)):
                    volatility[i] = np.std(returns[i-window:i])
            else:
                # Fallback if no price data
                price_data = np.linspace(1000, 2000, len(X_train))
                returns = np.random.normal(0, 0.02, len(X_train))
                volatility = np.abs(returns)
            
            # Generate synthetic samples for different market conditions
            print("Generating synthetic market conditions...")
            X_volatile, y_volatile = data_augmenter.generate_synthetic_market_conditions(
                X_train, y_train, condition_type='volatile', n_samples=int(len(X_train)*0.2)
            )
            
            X_trend_up, y_trend_up = data_augmenter.generate_synthetic_market_conditions(
                X_train, y_train, condition_type='trending_up', n_samples=int(len(X_train)*0.2)
            )
            
            X_trend_down, y_trend_down = data_augmenter.generate_synthetic_market_conditions(
                X_train, y_train, condition_type='trending_down', n_samples=int(len(X_train)*0.2)
            )
            
            # Apply time series perturbation for more realistic variations
            print("Applying time series perturbations...")
            X_perturbed, y_perturbed = data_augmenter.time_series_perturbation(
                X_train, y_train, perturbation_type='mixed', n_samples=int(len(X_train)*0.2)
            )
            
            # Combine original and augmented data
            X_train_aug = np.vstack([X_train, X_volatile, X_trend_up, X_trend_down, X_perturbed])
            y_train_aug = np.concatenate([y_train, y_volatile, y_trend_up, y_trend_down, y_perturbed])
            
            print(f"Original training samples: {len(X_train)}")
            print(f"Augmented training samples: {len(X_train_aug)}")
            
            # Use augmented data for training
            X_train = X_train_aug
            y_train = y_train_aug
        
        # Enhanced model configuration with better hyperparameters
        model_config = {
            "ensemble_type": "voting",  # options: "voting" or "stacking"
            "meta_learner_type": "lightgbm"  # for stacking ensemble
        }
        
        # Initialize the model
        model = EnsembleModel(model_config)
        
        # Define which models to train with optimized hyperparameters
        model_configs = {
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'bootstrap': True,
                'weight': 1.0
            },
            'xgboost': {
                'n_estimators': 150,
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0.1,
                'weight': 1.2
            },
            'lightgbm': {
                'n_estimators': 150,
                'max_depth': 8,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'weight': 1.5
            }
        }
        
        # Prepare data dictionary for training
        data_dict = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'sequence_length': 60
        }
        
        # Create temporary folder for this fold's models
        fold_dir = f"models/ensemble_fold_{fold+1}"
        os.makedirs(fold_dir, exist_ok=True)
        
        # Train the models for this fold
        print(f"Training fold {fold+1} models...")
        model.train_base_models(data_dict, model_configs, fold_dir)
        
        # Evaluate on test data for this fold
        metrics = model.evaluate(X_test, y_test)
        fold_metrics.append(metrics)
        
        print(f"Fold {fold+1} metrics:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Save best model based on accuracy
        if fold == 0 or metrics['accuracy'] > best_metrics.get('accuracy', 0):
            best_metrics = metrics
            best_fold = fold + 1
            best_models = {
                'model': model,
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'model_configs': model_configs,
                'selector': selector,
                'scaler': scaler
            }
    
    # Calculate average performance across folds
    avg_metrics = {}
    for metric in fold_metrics[0].keys():
        avg_metrics[metric] = np.mean([fold[metric] for fold in fold_metrics])
    
    print("\nAverage metrics across all folds:")
    for metric_name, value in avg_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    print(f"\nBest model was from fold {best_fold}")
    
    # Use the best model and retrain on all data
    print("\nRetraining best model on all data...")
    
    # Enhanced model configuration with better hyperparameters
    model_config = {
        "ensemble_type": "voting",  # options: "voting" or "stacking"
        "meta_learner_type": "lightgbm"  # for stacking ensemble
    }
    
    # Initialize the final model
    final_model = EnsembleModel(model_config)
    
    # Prepare full data for final training (still keeping validation)
    val_size = int(0.1 * len(X_scaled))
    X_train_full = X_scaled[:-val_size]
    y_train_full = all_targets.iloc[:-val_size].values
    X_val_full = X_scaled[-val_size:]
    y_val_full = all_targets.iloc[-val_size:].values
    
    # Apply data augmentation to the full training set if enabled
    if use_augmentation and data_augmenter:
        print("\nApplying data augmentation to final training set...")
        
        # For final training, use a combination of augmentation techniques with weights
        print("Applying weighted augmentation...")
        X_aug_full, y_aug_full = data_augmenter.weighted_augmentation(
            X_train_full, y_train_full,
            methods=['volatile', 'trending_up', 'trending_down', 'time_series_mixed', 'smote'],
            weights=[0.2, 0.2, 0.2, 0.3, 0.1],
            n_samples_total=int(len(X_train_full) * 0.5)  # Add 50% more samples
        )
        
        # Generate some adversarial examples using a simple model for robustness
        if hasattr(best_models.get('model', {}), 'models') and 'random_forest' in best_models['model'].models:
            print("Generating adversarial examples...")
            initial_model = best_models['model'].models['random_forest']
            X_adv, y_adv = data_augmenter.generate_adversarial_examples(
                X_train_full, y_train_full, initial_model, epsilon=0.05, 
                n_samples=int(len(X_train_full) * 0.1)
            )
            
            # Combine original and all augmented data for final training
            X_train_full = np.vstack([X_train_full, X_aug_full, X_adv])
            y_train_full = np.concatenate([y_train_full, y_aug_full, y_adv])
        else:
            # Just use the weighted augmentation if no model available
            X_train_full = np.vstack([X_train_full, X_aug_full])
            y_train_full = np.concatenate([y_train_full, y_aug_full])
        
        print(f"Final augmented training set size: {len(X_train_full)} samples")
    
    print(f"Final training set: {X_train_full.shape}, Validation set: {X_val_full.shape}")
    
    # Prepare data dictionary for final training
    data_dict_full = {
        'X_train': X_train_full,
        'y_train': y_train_full,
        'X_val': X_val_full,
        'y_val': y_val_full,
        'sequence_length': 60
    }
    
    # Train the models
    save_dir = "models/ensemble"  # This is where the dashboard looks for models by default
    os.makedirs(save_dir, exist_ok=True)
    print("Training final models with optimized hyperparameters...")
    results = final_model.train_base_models(data_dict_full, best_models['model_configs'], save_dir)
    print("Base models trained successfully")
    
    # Evaluate on validation data
    print("\nEvaluating final ensemble model...")
    metrics = final_model.evaluate(X_val_full, y_val_full)
    print("Ensemble model evaluation:")
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.4f}")
    
    # Evaluate individual models
    print("\nEvaluating individual models...")
    model_metrics = final_model.evaluate_individual_models(X_val_full, y_val_full)
    print("\nIndividual model performance:")
    for model_name, metrics in model_metrics.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"  {metric_name}: {value:.4f}")
    
    # Save the trained ensemble model
    print("\nSaving enhanced ensemble model...")
    final_model.save(save_dir)
    
    # Save the feature selector and scaler for preprocessing future data
    joblib_dir = os.path.join(save_dir, "preprocessing")
    os.makedirs(joblib_dir, exist_ok=True)
    
    import joblib
    joblib.dump(selector, os.path.join(joblib_dir, "feature_selector.joblib"))
    joblib.dump(scaler, os.path.join(joblib_dir, "scaler.joblib"))
    joblib.dump(selected_features, os.path.join(joblib_dir, "selected_features.joblib"))
    
    print(f"Ensemble model and preprocessing tools saved to {save_dir}")
    
    # Make predictions on test data
    print("\nTesting model on recent data...")
    predictions = final_model.predict(X_val_full[:10])
    binary_predictions = final_model.predict_binary(X_val_full[:10])
    actual = y_val_full[:10]
    
    print("\nPrediction results on test data:")
    print(f"{'Pred Prob':<10}{'Prediction':<12}{'Actual':<10}{'Correct':<10}")
    correct_count = 0
    for i, (prob, pred, act) in enumerate(zip(predictions, binary_predictions, actual)):
        is_correct = pred == act
        if is_correct:
            correct_count += 1
        print(f"{prob:.4f}{'     UP' if pred == 1 else '   DOWN'}{' '*5}{'UP' if act == 1 else 'DOWN'}{' '*5}{'✓' if is_correct else '✗'}")
    
    accuracy = correct_count / len(actual) * 100
    print(f"\nSample accuracy: {accuracy:.2f}%")
    
    # Record the training results for comparison
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    training_summary = {
        'timestamp': timestamp,
        'symbols': symbols,
        'feature_set': feature_set,
        'n_selected_features': len(selected_features),
        'ensemble_metrics': metrics,
        'individual_metrics': model_metrics,
        'fold_metrics': [dict(fold) for fold in fold_metrics],
        'avg_fold_metrics': avg_metrics,
    }
    
    # Save the training summary
    import json
    with open(os.path.join(save_dir, f"training_summary_{timestamp}.json"), "w") as f:
        json.dump({k: str(v) if isinstance(v, np.ndarray) else v for k, v in training_summary.items()}, f, indent=4)
    
    print("\nEnhanced ensemble model training completed successfully")
    
    return final_model, metrics

if __name__ == "__main__":
    # Make sure required directories exist
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/ensemble", exist_ok=True)
    
    # For sklearn's clone function 
    from sklearn.base import clone
    
    # Run the model with optimized parameters
    run_ensemble_model(
        symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOTUSDT"],  # Added more cryptocurrencies
        feature_set="advanced",
        n_features=136,  # Updated to 136 features instead of 60
        epochs=150
    )