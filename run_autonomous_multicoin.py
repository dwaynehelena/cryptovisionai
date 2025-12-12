#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
import requests
import pandas as pd
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autonomous_multicoin.log')
    ]
)
logger = logging.getLogger("AUTO_BOT_MULTI")

# Constants
PYTHON_EXEC = "./cryptovision_py39_env/bin/python"
TRAIN_SCRIPT = "train_advanced.py"
MODELS_DIR = "models"
TOP_N = 10

def get_top_volume_coins(n=10):
    """
    Fetch top N crypto pairs by 24h Quote Volume (Most Popular), avoiding stablecoins.
    """
    logger.info("🔍 Scanning for Top Volume Coins (24h)...")
    
    # Stablecoins to exclude (Base asset)
    STABLECOINS = {'USDC', 'FDUSD', 'TUSD', 'DAI', 'USDE', 'USDP', 'BUSD', 'EUR', 'AEUR', 'WBTC'} 
    # Blacklist known problematic symbols (e.g. delisted/renamed but reporting volume)
    BLACKLIST = {'ATUSDT'} 
    
    try:
        url = "https://api.binance.com/api/v3/ticker/24hr"
        response = requests.get(url)
        data = response.json()
        
        candidates = []
        for p in data:
            symbol = p['symbol']
            if not symbol.endswith('USDT'):
                continue
            
            if symbol in BLACKLIST:
                continue
                
            base_asset = symbol[:-4]
            if base_asset in STABLECOINS:
                continue
            
            # Exclude leveraged tokens
            if any(x in symbol for x in ['UP', 'DOWN', 'BULL', 'BEAR']):
                continue
                
            vol = float(p['quoteVolume'])
            candidates.append({
                'symbol': symbol,
                'volume': vol,
                'lastPrice': float(p['lastPrice'])
            })
        
        # Sort by Volume (Desc)
        candidates.sort(key=lambda x: x['volume'], reverse=True)
        
        top_n = candidates[:n]
        logger.info(f"🏆 Top {n} Most Popular (Volume):")
        for i, p in enumerate(top_n):
            vol_m = p['volume'] / 1_000_000
            logger.info(f"   {i+1}. {p['symbol']}: ${vol_m:.2f}M Vol")
            
        return [p['symbol'] for p in top_n]
        
    except Exception as e:
        logger.error(f"Error scanning market: {e}")
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"] # Fallback to blue chips

def ensure_models_trained(symbols, config):
    """
    Ensure a Triple Barrier TiDE model exists for each symbol.
    """
    logger.info("🧠 Verifying Models...")
    
    for symbol in symbols:
        model_file = f"tide_winning_{symbol}.h5"
        model_path = os.path.join(MODELS_DIR, model_file)
        
        if os.path.exists(model_path):
            logger.info(f"✅ Model exists for {symbol}")
        else:
            logger.info(f"⏳ Training model for {symbol}...")
            # Check config for feature set
            feature_set = config['model'].get('feature_set', 'standard')
            
            try:
                subprocess.run(
                    [PYTHON_EXEC, TRAIN_SCRIPT, "--symbol", symbol, "--output", model_path, "--feature_set", feature_set],
                    check=True
                )
                logger.info(f"✅ Training completed for {symbol} (Features: {feature_set})")
            except subprocess.CalledProcessError:
                logger.error(f"❌ Training failed for {symbol}")
                # We continue to next symbol
            except KeyboardInterrupt:
                logger.warning("Training interrupted.")
                sys.exit(1)

def start_multicoin_trading_system(symbols):
    logger.info("📈 Initializing Multi-Coin Trading System...")
    logger.info(f"   Assets: {symbols}")
    
    try:
        sys.path.append(os.getcwd())
        from src.trading.trading_system import TradingSystem
        import yaml
        
        # Load Config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        # Override config mainly for dynamic symbols
        config['trading']['symbols'] = symbols # We will need to support this in TradingSystem
        
        # Initialize
        system = TradingSystem(config)
        
        # INJECT Multi-Model Logic
        # Since TradingSystem natively loads 1 model, we will override its load logic 
        # or it should be updated to handle this config.
        # For this script to work, we assume TradingSystem IS updated.
        
        # Start
        system.start()
        logger.info("✅ Multi-Coin Trading System is RUNNING.")
        
        last_summary_time = 0
        while True:
            time.sleep(10) # Faster heartbeat (10s)
            
            try:
                # Use TradingSystem lock if needed? get_portfolio_summary has internal lock
                summary = system.portfolio.get_portfolio_summary()
                
                pnl = summary['profit_loss']
                initial = config['trading'].get('initial_capital', 10000.0)
                equity = initial + pnl
                roi = (pnl / initial) * 100
                
                # Visualize progress to $10,000 (100 -> 10000 is 9900% ROI)
                target_roi = 9900.0
                progress_pct = (roi / target_roi) * 100
                progress_bar = "#" * int(progress_pct // 5)
                
                logger.info(f"🚀 P/L: ${pnl:+.2f} | Equity: ${equity:.2f} | ROI: {roi:.2f}% | Goal: $10,000")
                
                # Log Open Positions
                open_positions = system.portfolio.positions
                if open_positions:
                    logger.info("   📂 Open Positions:")
                    for pid, pos in open_positions.items():
                        # Calculate PnL (approx if not updated in real-time loop, but update_positions runs in system)
                        # We should trust pos.pnl if it's being updated. 
                        # Note: system.start() runs in background thread? check. 
                        # TradingSystem.start() calls _run_trading_loop in thread.
                        # _run_trading_loop updates positions periodically.
                        logger.info(f"      - [{pos.symbol}] {pos.position_type.upper()} | Amt: {pos.amount:.4f} | Entry: ${pos.entry_price:.2f} | PnL: ${pos.pnl:+.2f} ({((pos.current_price/pos.entry_price - 1) if pos.position_type=='long' else (1 - pos.current_price/pos.entry_price))*100:+.2f}%)")
                else:
                    logger.info("   📂 Open Positions: None")
                # Log Prices & Accuracy
                price_strs = []
                accuracies = getattr(system, 'model_accuracies', {})
                for s in system.active_symbols:
                    ticker = system.binance_connector.get_ticker(s)
                    if 'price' in ticker:
                         acc_str = ""
                         if s in accuracies:
                             acc_str = f" [Acc: {accuracies[s]:.2f}]"
                         price_strs.append(f"{s}: ${float(ticker['price']):.4f}{acc_str}")
                
                if price_strs:
                    logger.info("💰 Prices: " + " | ".join(price_strs))

                if equity >= 10000.0:
                     logger.info("🎉🎉🎉 MOONSHOT! $10,000 TARGET ACHIEVED! 🎉🎉🎉")
                     # We don't stop, we keep earning!
            except Exception as e:
                logger.error(f"Heartbeat Error: {e}")
                # Don't crash main loop
                time.sleep(5)
                continue
                
    except Exception as e:
        logger.error(f"❌ System Crashed: {e}")
        import traceback
        traceback.print_exc()

def main():
    logger.info("==========================================================")
    logger.info("   🤖 CRYPTOVISION MULTI-COIN AUTONOMOUS BOT 🤖   ")
    logger.info("==========================================================")
    
    # Load Config (needed for ensure_models_trained)
    import yaml
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # 1. Get Top 10
    symbols = get_top_volume_coins(TOP_N)
    
    # 2. Ensure Models are Trained
    ensure_models_trained(symbols, config)
    
    # 3. Start Trading
    start_multicoin_trading_system(symbols)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped.")
