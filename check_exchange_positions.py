import sys
import os
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SYNC_CHECK")

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing.binance_connector import BinanceConnector
from src.utils.alerts import AlertManager

def check_positions():
    logger.info("🔍 Connecting to Binance Testnet...")
    
    # Initialize connector (it handles config loading internally usually, 
    # but we might need to manually ensuring config is picked up if it relies on init args)
    # Looking at run_autonomous_multicoin.py, it loads config first.
    # We'll rely on BinanceConnector's internal config loading if it exists, or pass it.
    
    # Actually BinanceConnector takes config_path.
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "config.yaml")
    
    try:
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Pass both use_testnet and config_path explicitly
        logger.info(f"Using config: {config_path}")
        connector = BinanceConnector(use_testnet=True, config_path=config_path)
        
        if not connector.client:
            logger.error("❌ Failed to initialize Client. Check API keys in config.")
            return

        logger.info("✅ Connected.")
        
        # Get Account Info
        account_info = connector.get_account_info()
        if not account_info or 'balances' not in account_info:
            logger.error("❌ Failed to fetch account info")
            return

        # Filter for non-zero balances (Positions)
        # Note: In Spot (Testnet), "Positions" are just Balances of the asset (e.g. BTC, ETH).
        # Short positions in Spot are tricky - usually done via Margin or Futures.
        # But this bot seems to trade Spot (BUY/SELL).
        # Wait, the bot logs said "Opened SHORT position". 
        # If it's Spot, you can't short unless it's Selling existing holdings.
        # If the bot assumes "Short" = Sell and hold USDT, then the "Position" is just USDT.
        # Let's check what the bot actually DOES for "Short".
        # TradingSystem._open_position line 1335: "order_side = 'BUY' if ... 'long' else 'SELL'".
        # If it Sells to Open, it implies we successfully sold an asset we didn't have? No, Spot rejects that.
        # Unless it's Futures? config says "use_testnet: true". BinanceConnector uses "spot" endpoints normally.
        # Let's see what "Assets" we actually hold.
        
        balances = account_info['balances']
        active_assets = []
        for b in balances:
            free = float(b['free'])
            locked = float(b['locked'])
            if free > 0 or locked > 0:
                active_assets.append({
                    "asset": b['asset'],
                    "free": free,
                    "locked": locked,
                    "total": free + locked
                })
        
        logger.info(f"💰 Wallet Balances (Exchange):")
        for asset in active_assets:
             logger.info(f"   - {asset['asset']}: {asset['total']:.8f}")

        # Check for Local Portfolio
        local_portfolio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "portfolio.json")
        if os.path.exists(local_portfolio_path):
             import json
             with open(local_portfolio_path, 'r') as f:
                 local_data = json.load(f)
             logger.info(f"📂 Local Portfolio: {len(local_data.get('positions', {}))} positions")
             for pid, pos in local_data.get('positions', {}).items():
                 logger.info(f"   - [{pos['symbol']}] {pos['position_type']} (Amt: {pos['amount']})")
        else:
             logger.info("📂 Local Portfolio: Empty / No File")

    except Exception as e:
        logger.error(f"Error during check: {e}")

if __name__ == "__main__":
    check_positions()
