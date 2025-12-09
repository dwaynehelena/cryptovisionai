import os
print("DEBUG: Script started", flush=True)
import sys
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import optuna
import joblib
import gc
from typing import Tuple, Any, List
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.binance_connector import BinanceConnector
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.patch_tst_model import PatchTSTModel
from src.models.tide_model import TiDEModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/automl_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("automl_training")

def load_data(symbol: str = "BTCUSDT", limit: int = 50000) -> pd.DataFrame:
    """Load and preprocess data"""
    config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
    connector = BinanceConnector(use_testnet=False, config_path=config_path)
    
    # Fetch data (5m interval for "Next 5 Minutes" prediction)
    # Reducing limit slightly to 250k initial check to be safe with RAM, then we can scale up
    interval = "5m"
    start_date = "1 Jan, 2023"
    start_ts = int(datetime.strptime(start_date, "%d %b, %Y").timestamp() * 1000)
    
    logger.info(f"Fetching 5m data for {symbol} from {start_date}...")
    df = connector.get_historical_klines(
        symbol, 
        interval, 
        limit=limit, 
        start_time=datetime.fromtimestamp(start_ts/1000)
    )
    
    # Memory Optimization: Downcast floats immediately
    float_cols = df.select_dtypes(include=['float64']).columns
    df[float_cols] = df[float_cols].astype('float32')
    
    return df

def prepare_data(df: pd.DataFrame, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    """Feature engineering and data preparation"""
    fe = FeatureEngineering()
    
    # Generate features
    # Note: feature_engineering.py might default to float64, so we cast again
    logger.info("Generating features...")
    df_features = fe.extract_features(df, feature_set="advanced")
    
    # Force float32
    df_features = df_features.astype('float32')
    
    # Create target (next candle direction for trend)
    # 1 if close[t+1] > close[t], else 0
    close_prices = df['close'].values
    next_close = pd.Series(close_prices).shift(-1).values
    price_changes = next_close - close_prices
    y = (price_changes > 0).astype(int)
    
    # Drop last row (NaN target)
    df_features = df_features.iloc[:-1]
    y = y[:-1]
    
    # Scale features
    logger.info("Scaling features...")
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_features).astype('float32')
    
    # Garbage collection
    del df, df_features, close_prices, next_close, price_changes
    gc.collect()
    
    # Create sequences
    logger.info("Creating sequences...")
    X_seq, y_seq = [], []
    for i in range(len(X_scaled) - sequence_length):
        X_seq.append(X_scaled[i:i+sequence_length])
        y_seq.append(y[i+sequence_length])
        
    X_seq = np.array(X_seq, dtype='float32')
    y_seq = np.array(y_seq, dtype='int32') # Use int32 for targets
    
    # Split train/val
    train_size = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    return X_train, y_train, X_val, y_val, scaler

def objective(trial):
    """Optuna objective function"""
    
    # 1. Suggest Model Type
    model_type = trial.suggest_categorical("model_type", ["PatchTST", "TiDE"])
    
    # Common Hyperparams
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256]) # Increased batch size for speed
    
    # Global variables for data (passed implicitly or we should wrap in a class)
    # For now relying on global scope as per original script design
    
    config = {
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "dropout_rate": trial.suggest_float("dropout_rate", 0.1, 0.4)
    }
    
    if model_type == "PatchTST":
        config.update({
            "patch_len": trial.suggest_categorical("patch_len", [8, 16]),
            "stride": trial.suggest_categorical("stride", [8]), 
            "embed_dim": trial.suggest_int("embed_dim", 32, 64, step=32), # Reduced for memory
            "num_heads": trial.suggest_categorical("num_heads", [4]),
            "ff_dim": trial.suggest_int("ff_dim", 64, 128, step=64),
            "num_layers": trial.suggest_int("num_layers", 1, 2)
        })
        model = PatchTSTModel(config, sequence_length=60, n_features=X_train.shape[2])
        
    else: # TiDE
        config.update({
            "hidden_dim": trial.suggest_int("hidden_dim", 128, 512, step=128),
            "decoder_output_dim": trial.suggest_int("decoder_output_dim", 16, 64, step=16),
            "num_layers": trial.suggest_int("num_layers", 1, 3)
        })
        model = TiDEModel(config, sequence_length=60, n_features=X_train.shape[2])
        
    # 2. Train Model
    try:
        # Check for NaN and fix if necessary (float32 conversion can sometimes cause issues)
        if np.isnan(X_train).any():
            logger.warning("NaN found in training data. Filling with 0.")
            X_train = np.nan_to_num(X_train)
            
        history = model.train(
            X_train, y_train,
            X_val=X_val, y_val=y_val,
            batch_size=batch_size,
            epochs=10, # Keep it short for AutoML
            save_path=None
        )
        
        # 3. Return Metric
        if 'val_accuracy' in history and len(history['val_accuracy']) > 0:
            return history['val_accuracy'][-1]
        else:
            return 0.0
        
    except Exception as e:
        logger.error(f"Trial failed: {e}")
        return 0.0
    finally:
        # Clear memory after trial
        tf.keras.backend.clear_session()
        gc.collect()

if __name__ == "__main__":
    # List of symbols to train
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
    
    for symbol in symbols:
        logger.info(f"Starting AutoML for {symbol}...")
        
        try:
            # Load Data
            df = load_data(symbol, limit=100000) # Reduced limit to 100k for safety initially
            global X_train, y_train, X_val, y_val
            X_train, y_train, X_val, y_val, scaler = prepare_data(df)
            
            logger.info(f"Data prepared: Train shape: {X_train.shape}, Val shape: {X_val.shape}")
            
            # Run Optimization
            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=10) # 10 trials per symbol
            
            # Save Best Model Info
            logger.info(f"Best trial for {symbol}:")
            logger.info(f"  Value: {study.best_trial.value}")
            logger.info(f"  Params: {study.best_trial.params}")
            
            # Save scaler
            joblib.dump(scaler, f"models/automl/{symbol}_scaler.pkl")
            
            # Save optimization results
            df_results = study.trials_dataframe()
            df_results.to_csv(f"models/automl/{symbol}_optimization_results.csv")
            
            # Clean up global validation data to free memory for next symbol
            del X_train, y_train, X_val, y_val, scaler, df
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error processing {symbol}: {e}")
            continue
