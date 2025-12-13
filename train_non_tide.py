
import os
import sys
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
import gc
import joblib
from datetime import datetime
from typing import Dict, Any, Tuple

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.binance_connector import BinanceConnector
from src.data_processing.feature_engineering import FeatureEngineering
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel
from src.models.patch_tst_model import PatchTSTModel
from src.models.ensemble_model import EnsembleModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("train_non_tide")

def load_data(symbol: str = "BTCUSDT", limit: int = 30000) -> pd.DataFrame:
    """Load data"""
    connector = BinanceConnector(use_testnet=False)
    interval = "5m"
    start_dt = datetime.now() - pd.Timedelta(days=90) # Last 90 days for speed
    
    logger.info(f"Fetching data for {symbol}...")
    df = connector.get_historical_klines(symbol, interval, limit=limit, start_time=start_dt)
    return df

def prepare_data(df: pd.DataFrame, sequence_length: int = 60) ->  Dict[str, Any]:
    """Prepare data for both DL (sequences) and Ensemble (flat)"""
    fe = FeatureEngineering()
    
    # Target: Next Close > Current Close
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    
    # Extract features (handles dropping NaNs)
    df_features = fe.extract_features(df, feature_set="advanced")
    
    # Separate
    y = df_features['target'].values
    X_df = df_features.drop(columns=['target'])
    X = X_df.values.astype('float32')
    y = y.astype('int32')
    
    # Scale
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # --- Sequence Data (for DL) ---
    from numpy.lib.stride_tricks import sliding_window_view
    
    # Create windows (Samples, SeqLen, Features)
    # sliding_window_view returns (N, WinSize, Features) if axis=0? No.
    # shape (N, Features) -> window(axis=0) -> (N-Win+1, Features, WinSize)
    X_windows = sliding_window_view(X_scaled, window_shape=sequence_length, axis=0)
    # Transpose to (Samples, WinSize, Features)
    X_seq = X_windows.transpose(0, 2, 1)
    
    # Align y for sequences (y at end of window)
    # If window is [0..59], it predicts y[59]?
    # In df, X[t] predicts y[t] (which is close[t+1] > close[t])
    # So if we use X[t-59]...X[t] to predict, we are predicting trend at t.
    # So y should take the index corresponding to the last element of the window.
    # sliding_window_view[0] is rows 0..59. Last row is 59.
    # So y aligned is y[59...]
    y_seq = y[sequence_length-1:]
    
    # Truncate X_seq to match y_seq if needed (usually sliding_window_view reduces size correctly)
    # len(X_seq) = N - seq_len + 1
    # len(y_seq) = N - seq_len + 1
    
    # --- Flat Data (for Ensemble) ---
    # Ensemble usually predicts row t using row t features.
    # But here we probably want to be consistent with time-series testing.
    # We will use the same split.
    # But EnsembleModel takes 'X_train' and 'y_train'.
    # We can just use the scaled data, but we must ensure no lookahead bias if using just row t.
    # extract_features usually adds lagging features, so row t has history.
    
    # Split Train/Val (80/20)
    split_idx = int(len(X_seq) * 0.8)
    
    DL_data = {
        'X_train': X_seq[:split_idx],
        'y_train': y_seq[:split_idx],
        'X_val': X_seq[split_idx:],
        'y_val': y_seq[split_idx:],
        'scaler': scaler
    }
    
    # For Ensemble, align with the sequence data indices
    # The sequences start at index 0 (which is real index sequence_length-1 in X)
    # So we used X[sequence_length-1:] for the "current time" of the prediction?
    # Actually, Ensemble just needs X and y aligned.
    X_flat_aligned = X_scaled[sequence_length-1:]
    y_flat_aligned = y[sequence_length-1:]
    
    Ensemble_data = {
        'X_train': X_flat_aligned[:split_idx],
        'y_train': y_flat_aligned[:split_idx],
        'X_val': X_flat_aligned[split_idx:],
        'y_val': y_flat_aligned[split_idx:]
    }
    
    return {'dl': DL_data, 'ensemble': Ensemble_data}

def train_all():
    symbols = [
        "BTCUSDT", "ETHUSDT" #, "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "LUNAUSDT", "SUIUSDT", "ZECUSDT"
        # Limiting to 2 main symbols for demonstration speed as requested "see accuracy", 
        # but user asked "retrain all of them". I will do main ones first or all if fast.
        # Let's do BTC and ETH first to ensure it works, then maybe loop more if successful.
        # Actually user said "retrain all of them", implying the list I showed earlier.
        # I'll include all but limit data size/epochs.
    ]
    all_symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT", "DOGEUSDT", "LUNAUSDT", "SUIUSDT", "ZECUSDT"]
    
    results = []

    for symbol in all_symbols:
        logger.info(f"=== Processing {symbol} ===")
        try:
            df = load_data(symbol, limit=20000) # 20k candles ~ 70 days of 5m
            if df is None: continue
            
            data = prepare_data(df)
            dl_data = data['dl']
            ens_data = data['ensemble']
            
            n_features = dl_data['X_train'].shape[2]
            seq_len = 60
            
            symbol_res = {'symbol': symbol}
            
            # 1. LSTM
            logger.info(f"Training LSTM for {symbol}...")
            lstm = LSTMModel({'lstm_units': [64], 'dropout_rate': 0.2, 'epochs': 5}, sequence_length=seq_len, n_features=n_features)
            hist = lstm.train(dl_data['X_train'], dl_data['y_train'], dl_data['X_val'], dl_data['y_val'], epochs=5, batch_size=256)
            if hist:
                symbol_res['LSTM'] = hist['val_accuracy'][-1]
                lstm.save(f"models/lstm_{symbol}.h5")
            
            # 2. Transformer
            logger.info(f"Training Transformer for {symbol}...")
            trans = TransformerModel({'embed_dim': 32, 'num_heads': 2, 'ff_dim': 64, 'num_transformer_blocks': 2}, sequence_length=seq_len, n_features=n_features)
            hist = trans.train(dl_data['X_train'], dl_data['y_train'], dl_data['X_val'], dl_data['y_val'], epochs=5, batch_size=256)
            if hist:
                symbol_res['Transformer'] = hist['val_accuracy'][-1]
                trans.save(f"models/transformer_{symbol}.h5")
                
            # 3. PatchTST
            logger.info(f"Training PatchTST for {symbol}...")
            patch = PatchTSTModel({'patch_len': 16, 'stride': 8, 'embed_dim': 32, 'num_heads': 2, 'num_layers': 2}, sequence_length=seq_len, n_features=n_features)
            try:
                hist = patch.train(dl_data['X_train'], dl_data['y_train'], dl_data['X_val'], dl_data['y_val'], epochs=5, batch_size=256)
                if hist:
                    symbol_res['PatchTST'] = hist['val_accuracy'][-1]
                    # save method might differ for PatchTST, it inherits from BaseModel? Let's assume standard
                    patch.save(f"models/patchtst_{symbol}.h5")
            except Exception as e:
                logger.error(f"PatchTST failed: {e}")
                symbol_res['PatchTST'] = 0.0

            # 4. Ensemble
            logger.info(f"Training Ensemble (RF, XGB) for {symbol}...")
            ensemble = EnsembleModel()
            # We use smaller estimators for speed
            ens_config = {
                'random_forest': {'n_estimators': 50, 'max_depth': 10},
                'xgboost': {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1},
                # 'lightgbm': {'n_estimators': 50} 
            }
            ens_results = ensemble.train_base_models(ens_data, ens_config, save_dir=f"models/ensemble_{symbol}")
            
            if 'random_forest' in ens_results:
                symbol_res['RandomForest'] = ens_results['random_forest']['accuracy']
            if 'xgboost' in ens_results:
                symbol_res['XGBoost'] = ens_results['xgboost']['accuracy']
                
            results.append(symbol_res)
            
            # Cleanup
            tf.keras.backend.clear_session()
            gc.collect()
            
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")
            import traceback
            traceback.print_exc()
            
    # Print Summary
    print("\n" + "="*60)
    print(f"{'Symbol':<10} {'LSTM':<10} {'Transf':<10} {'PatchTST':<10} {'RanForest':<10} {'XGBoost':<10}")
    print("-" * 60)
    for res in results:
        print(f"{res['symbol']:<10} "
              f"{res.get('LSTM', 0):.4f}     "
              f"{res.get('Transformer', 0):.4f}     "
              f"{res.get('PatchTST', 0):.4f}     "
              f"{res.get('RandomForest', 0):.4f}     "
              f"{res.get('XGBoost', 0):.4f}")
    print("="*60 + "\n")

if __name__ == "__main__":
    train_all()
