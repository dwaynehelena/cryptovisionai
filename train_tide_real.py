
import os
import time
import numpy as np
import pandas as pd
import yaml
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from src.data_processing.binance_connector import BinanceConnector
from src.models.tide_model import TiDEModel

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("train_tide")

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def prepare_data(df, sequence_length, target_horizon=1):
    """
    Prepare data for TiDE model training (sequence to single step classification)
    Target: 1 if price moves UP, 0 if DOWN
    """
    # Drop NaN values created by technical indicators (e.g., rolling windows)
    df = df.dropna()
    
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'ignore', 'close_time']]
    data = df[feature_cols].values
    
    # Check for infinity
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    X, y = [], []
    
    # Calculate returns for target
    # Calculate returns for target
    # If close price at t+horizon > close price at t, then 1, else 0
    close_prices = df['close'].values
    
    for i in range(len(data) - sequence_length - target_horizon):
        X.append(data_scaled[i : i + sequence_length])
        
        current_close = close_prices[i + sequence_length - 1]
        future_close = close_prices[i + sequence_length + target_horizon - 1]
        
        # Classification target: 1 (Up) vs 0 (Down)
        target = 1 if future_close > current_close else 0
        y.append(target)
        
    return np.array(X), np.array(y), scaler

def train_tide():
    logger.info("Starting TiDE model training (Massive 3-Year Run)...")
    
    config = load_config()
    
    # Initialize Binance Connector - Force Mainnet (Public Data)
    connector = BinanceConnector(
        use_testnet=False,
        api_key=None, 
        api_secret=None
    )
    
    # Calculate start time for 3 years ago
    import datetime
    start_dt = datetime.datetime.now() - datetime.timedelta(days=1095)
    
    symbol = "BTCUSDT"
    interval = "5m"
    limit = 500000 # Fetch up to 500k candles (~3 years of 5m data is ~315k)
    
    logger.info(f"Fetching klines for {symbol} {interval} starting from {start_dt}...")
    df = connector.get_historical_klines(symbol, interval, limit=limit, start_time=start_dt)
    
    if df is None:
        logger.error("Failed to fetch data.")
        return
    
    logger.info(f"Fetched {len(df)} rows.")
    
    # Prepare data
    # Override sequence length
    config['model']['model_config']['models']['tide']['sequence_length'] = 120
    seq_len = 120
    target_horizon = 1 # Predict 1 step ahead (5m)
    
    n_features = len([col for col in df.columns if col not in ['timestamp', 'ignore', 'close_time']])
    
    logger.info(f"Preparing data with sequence_length={seq_len}, n_features={n_features}, horizon={target_horizon}")
    
    X, y, scaler = prepare_data(df, seq_len)
    
    # Split train/test
    split = int(len(X) * 0.9) # 90/10 split for more training data
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    logger.info(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Testing shapes: X={X_test.shape}, y={y_test.shape}")
    
    tide_config = config['model']['model_config']['models']['tide']
    tide_config['learning_rate'] = 0.0005
    tide_config['dropout_rate'] = 0.1
    tide_config['hidden_dim'] = 256
    tide_config['num_layers'] = 3
    
    # Update n_features in config matching actual data
    model = TiDEModel(tide_config, sequence_length=seq_len, n_features=n_features)
    model.build_model()
    
    logger.info("Training...")
    history = model.train(
        X_train, y_train, 
        X_val=X_test, y_val=y_test, 
        epochs=50, 
        batch_size=1024, # Large batch size for speed
        save_path="models/tide_standalone.h5"
    )
    
    # Evaluation
    logger.info("Evaluating...")
    predictions = model.predict(X_test)
    pred_classes = (predictions > 0.5).astype(int).flatten()
    
    accuracy = np.mean(pred_classes == y_test)
    logger.info(f"Final Accuracy: {accuracy * 100:.2f}%")
    
    if accuracy < 0.6:
        logger.warning("Accuracy is below 60%. You might need to tune hyperparameters.")
    else:
        logger.info("Success! Accuracy is above 60%.")

if __name__ == "__main__":
    train_tide()
