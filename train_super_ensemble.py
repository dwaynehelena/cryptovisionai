
import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime

# Add project root
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.hybrid_ensemble import HybridEnsemble
from src.data_processing.binance_connector import BinanceConnector
from src.data_processing.feature_engineering import FeatureEngineering
from sklearn.preprocessing import StandardScaler
from numpy.lib.stride_tricks import sliding_window_view

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("train_super")

def prepare_data_sequence(symbol):
    connector = BinanceConnector(use_testnet=False)
    # Get plenty of data for "Maximum Accuracy"
    df = connector.get_historical_klines(symbol, "5m", limit=30000, start_time=datetime.now() - pd.Timedelta(days=100))
    if df is None: return None
    
    fe = FeatureEngineering()
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    df_features = fe.extract_features(df, feature_set="advanced")
    
    y = df_features['target'].values
    X = df_features.drop(columns=['target']).values.astype('float32')
    y = y.astype('int32')
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    seq_len = 60
    X_windows = sliding_window_view(X_scaled, window_shape=seq_len, axis=0)
    X_seq = X_windows.transpose(0, 2, 1) # (N, Seq, Feat)
    y_seq = y[seq_len-1:]
    
    # Align lengths
    min_len = min(len(X_seq), len(y_seq))
    X_seq = X_seq[:min_len]
    y_seq = y_seq[:min_len]
    
    split = int(len(X_seq) * 0.85)
    return {
        'dl': {
            'X_train': X_seq[:split], 'y_train': y_seq[:split],
            'X_val': X_seq[split:], 'y_val': y_seq[split:]
        }
    }

def main():
    symbol = "BTCUSDT"
    logger.info(f"Preparing Super Ensemble for {symbol}...")
    
    data = prepare_data_sequence(symbol)
    if not data: 
        logger.error("No data")
        return
        
    ensemble = HybridEnsemble()
    
    logger.info("Training Individual Components (TiDE, PatchTST, Mamba)...")
    ensemble.train_component_models(data)
    
    logger.info("Optimizing Mixture...")
    acc = ensemble.optimize_weights(data['dl']['X_val'], data['dl']['y_val'])
    
    logger.info(f"FINAL ENSEMBLE ACCURACY: {acc:.4f}")
    
    # Save
    ensemble.save("models/super_ensemble_BTC")

if __name__ == "__main__":
    main()
