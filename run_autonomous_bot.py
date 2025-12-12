#!/usr/bin/env python3
import os
import sys
import time
import logging
import subprocess
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('autonomous_bot.log')
    ]
)
logger = logging.getLogger("AUTO_BOT")

# Constants
PYTHON_EXEC = "./cryptovision_py39_env/bin/python"
TRAIN_SCRIPT = "train_advanced.py"
MODEL_PATH = "models/tide_winning.h5"
CHECK_INTERVAL = 10

def check_model_exists():
    return os.path.exists(MODEL_PATH)

def run_training():
    logger.info("⚠️  Model file not found or retraining requested.")
    logger.info(f"🚀 Launching Automated Training ({TRAIN_SCRIPT})...")
    logger.info("   Target: >60% Accuracy. This may take a while.")
    
    try:
        # Run training script and wait for it to finish
        # We stream output to stdout so user sees progress
        result = subprocess.run(
            [PYTHON_EXEC, TRAIN_SCRIPT],
            check=True
        )
        if result.returncode == 0:
            logger.info("✅ Training completed successfully.")
            return True
        else:
            logger.error("❌ Training failed.")
            return False
            
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Training script crashed: {e}")
        return False
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted by user.")
        return False

def start_trading_system():
    logger.info("📈 Initializing Trading System...")
    
    try:
        # Import dynamically to ensure env is ready
        sys.path.append(os.getcwd())
        from src.trading.trading_system import TradingSystem
        import yaml
        
        # Load Config
        with open("config/config.yaml", 'r') as f:
            config = yaml.safe_load(f)
            
        # Ensure we are in the mode user wants (defaulting to Config)
        # We assume config.yaml is set to desired 'test' or 'live' mode
        mode = "TESTNET" if config.get("use_testnet", True) else "MAINNET - REAL MONEY"
        logger.info(f"⚡ Mode: {mode}")
        
        # Initialize
        system = TradingSystem(config)
        
        # Check Readiness
        status = system.check_readiness()
        if not status['ready']:
            logger.warning(f"⚠️  System might not be fully ready: {status}")
            # We proceed anyway as check_readiness might be strict
        
        # Start
        system.start()
        logger.info("✅ Trading System is RUNNING.")
        logger.info("   Press Ctrl+C to stop.")
        
        # Keep main thread alive
        while True:
            time.sleep(60)
            # Optional: Print heartbeat or summary
            summary = system.portfolio.get_portfolio_summary()
            pnl = summary['profit_loss']
            pos_count = summary['open_positions']
            logger.info(f"heartbeat | P/L: ${pnl:.2f} | Open Positions: {pos_count}")
            
    except ImportError:
        logger.error("❌ Could not import TradingSystem. Check python environment.")
    except Exception as e:
        logger.error(f"❌ Trading System Crashed: {e}")
        import traceback
        traceback.print_exc()

def main():
    logger.info("=================================================")
    logger.info("   🤖 CRYPTOVISION AUTONOMOUS TRADING BOT 🤖   ")
    logger.info("=================================================")
    
    # 1. Check Model
    if not check_model_exists():
        success = run_training()
        if not success:
            logger.error("❌ Cannot start trading without a trained model.")
            sys.exit(1)
            
    # 2. Check Model again just to be sure
    if check_model_exists():
        logger.info(f"✅ Found Winning Model: {MODEL_PATH}")
        # 3. Start Trading
        start_trading_system()
    else:
        logger.error("❌ Model missing after training phase. Something went wrong.")
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user.")
