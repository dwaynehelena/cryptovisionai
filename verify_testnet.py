
import sys
import os
import logging
import time
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# Setup path
sys.path.append(os.getcwd())

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_trading")

from src.trading.trading_system import TradingSystem
import src.trading.trading_system 

# Mock Model to force signals
class MockModel:
    def __init__(self):
        self.sequence_length = 60
        
    def predict(self, X):
        # Force a strong BUY signal
        # Output: [p_hold, p_buy, p_sell]
        return np.array([[0.05, 0.90, 0.05]])

def verify_on_testnet():
    logger.info("--- Starting Testnet Verification ---")
    
    # Load Real Config
    import yaml
    with open("config/config.yaml", 'r') as f:
        real_config = yaml.safe_load(f)
        
    # Override for Testnet Safety
    config = real_config.copy()
    config['use_testnet'] = True
    config['api_key'] = real_config['api']['binance']['testnet']['api_key']
    config['api_secret'] = real_config['api']['binance']['testnet']['api_secret']
    config['initial_capital'] = 1000.0
    config['risk_management'] = {
        'max_position_size': 10.0, # Small size
        'risk_per_trade': 1.0,
        'stop_loss_percent': 2.0
    }
    
    # Initialize System
    logger.info("Initializing Trading System...")
    system = TradingSystem(config)
    
    # Inject Mock Model because real training is still running
    logger.info("Injecting Mock Model (Forced BUY signal)...")
    system.model = MockModel()
    
    # Check Connectivity
    status = system.check_readiness()
    logger.info(f"System Status: {status}")
    
    if not status['components']['binance_api']['status'] == 'operational':
        logger.error("Binance Testnet API not reachable. Check network/keys.")
        return

    # Create Dummy Data for BTCUSDT to generate features
    # We need to fetch real data so indicators work, but we'll force the prediction
    symbol = "BTCUSDT"
    logger.info(f"Fetching fetching real market data for {symbol} to generate valid indicators...")
    df = system._get_historical_data(symbol)
    
    if df is None:
        logger.error("Failed to fetch data.")
        return
        
    logger.info(f"Generating features...")
    features = system._generate_features(df)
    
    if features is None:
        logger.error("Feature generation failed.")
        return

    # FORCE PREDICTION
    logger.info("Generating Prediction from Mock Model...")
    signal = system._get_prediction(features)
    logger.info(f"Generated Signal: {signal}")
    
    if signal['signal'] != 'buy':
        logger.error("Mock model failed to produce BUY signal.")
        return
        
    # MANUALLY TRIGGER SIGNAL PROCESSING
    logger.info("Executing Trade Logic...")
    
    # We want to see if it creates an order. 
    # We can spy on binance_connector.create_order OR let it run real.
    # User asked to "prove on testnet", so let's run REAL.
    
    system._process_signal(symbol, df, signal)
    
    # Check Portfolio
    time.sleep(2) # Wait for execution
    
    open_positions = system.portfolio.get_all_positions()['open_positions']
    logger.info(f"Open Positions: {len(open_positions)}")
    
    if len(open_positions) > 0:
        logger.info("SUCCESS: Position Opened on Testnet!")
        logger.info(open_positions[0])
        
        # Cleanup: Close the position
        logger.info("Closing position to cleanup...")
        pos_id = open_positions[0]['position_id']
        # Fetch current price
        ticker = system.binance_connector.get_ticker(symbol)
        price = float(ticker['price'])
        # Close
        system.binance_connector.create_order(symbol, "SELL", "MARKET", open_positions[0]['amount'])
        system.portfolio.close_position(pos_id, price)
        logger.info("Position Closed.")
    else:
        logger.error("FAILED: No position opened. Check logs for Risk Manager rejection or API errors.")

if __name__ == "__main__":
    verify_on_testnet()
