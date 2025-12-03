#!/usr/bin/env python3
"""
Main Application - Connects all components of the Bitcoin price prediction and trading system
"""

import os
import logging
import argparse
import yaml
import threading
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Import custom modules
from .data_processing.binance_connector import BinanceConnector
from .data_processing.feature_engineering import FeatureEngineering
from .models.ensemble_model import EnsembleModel
from .trading.trading_system import TradingSystem
from .visualization.dashboard import Dashboard

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join("logs", f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("app")

class CryptoVisionAI:
    """
    Main application class that connects all components of the system
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the application
        
        Args:
            config_path (str): Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.binance_connector = None
        self.feature_engineering = None
        self.model = None
        self.trading_system = None
        self.dashboard = None
        
        # Setup directories
        self._setup_directories()
        
        logger.info("CryptoVisionAI initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Dict[str, Any]: Configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            logger.info("Using default configuration")
            return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """
        Create default configuration
        
        Returns:
            Dict[str, Any]: Default configuration dictionary
        """
        config = {
            "general": {
                "mode": "test",  # 'test', 'paper', 'live'
                "debug": True
            },
            "binance": {
                "use_testnet": True,
                "api_key": "",
                "api_secret": ""
            },
            "trading": {
                "base_asset": "BTC",
                "quote_asset": "USDT",
                "initial_capital": 10000.0,
                "update_interval_seconds": 60,
                "signal_interval_minutes": 60,
                "test_mode": True
            },
            "risk_management": {
                "max_portfolio_risk": 5.0,
                "max_position_size": 20.0,
                "max_open_positions": 5,
                "stop_loss_percent": 3.0,
                "take_profit_percent": 6.0,
                "max_drawdown_limit": 20.0,
                "risk_per_trade": 1.0,
                "kelly_fraction": 0.5
            },
            "model": {
                "model_path": "models/ensemble",
                "feature_set": "advanced",
                "model_config": {
                    "ensemble_type": "voting"
                }
            },
            "dashboard": {
                "refresh_interval": 30,
                "port": 8050,
                "host": "0.0.0.0"
            }
        }
        return config
    
    def _setup_directories(self) -> None:
        """Create necessary directories if they don't exist"""
        directories = ["data", "logs", "models"]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def initialize_components(self) -> None:
        """Initialize all system components"""
        try:
            # Initialize Binance connector
            binance_config = self.config.get("binance", {})
            self.binance_connector = BinanceConnector(
                use_testnet=binance_config.get("use_testnet", True),
                config_path=self.config_path if hasattr(self, 'config_path') else "config/config.yaml"
            )
            logger.info("Binance connector initialized")
            
            # Initialize feature engineering
            self.feature_engineering = FeatureEngineering()
            logger.info("Feature engineering initialized")
            
            # Initialize model
            model_config = self.config.get("model", {}).get("model_config", {})
            self.model = EnsembleModel(model_config)
            
            # Load model if path exists
            model_path = self.config.get("model", {}).get("model_path", "models/ensemble")
            if os.path.exists(model_path):
                self.model.load(model_path)
                logger.info(f"Model loaded from {model_path}")
            else:
                logger.warning(f"Model path {model_path} doesn't exist. Running with untrained model.")
            
            # Initialize trading system
            trading_config = {
                **self.config.get("trading", {}),
                "risk_management": self.config.get("risk_management", {}),
                "model_config": model_config,
                "model_path": model_path
            }
            self.trading_system = TradingSystem(trading_config)
            logger.info("Trading system initialized")
            
            # Initialize dashboard
            dashboard_config = self.config.get("dashboard", {})
            self.dashboard = Dashboard(
                trading_system=self.trading_system,
                refresh_interval=dashboard_config.get("refresh_interval", 30)
            )
            logger.info("Dashboard initialized")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def start(self) -> None:
        """Start the application"""
        try:
            logger.info("Starting CryptoVisionAI")
            
            # Initialize components if not already done
            if self.trading_system is None:
                self.initialize_components()
            
            # Start trading system in a separate thread
            trading_thread = threading.Thread(target=self._start_trading)
            trading_thread.daemon = True
            trading_thread.start()
            logger.info("Trading thread started")
            
            # Start dashboard
            dashboard_config = self.config.get("dashboard", {})
            self.dashboard.start(
                debug=self.config.get("general", {}).get("debug", False),
                host=dashboard_config.get("host", "0.0.0.0"),
                port=dashboard_config.get("port", 8050)
            )
            
        except Exception as e:
            logger.error(f"Error starting application: {e}")
            self.stop()
            raise
    
    def _start_trading(self) -> None:
        """Start the trading system"""
        try:
            # Only start automatically if in paper or live mode
            mode = self.config.get("general", {}).get("mode", "test")
            if mode in ["paper", "live"]:
                self.trading_system.start()
                logger.info(f"Trading system started in {mode} mode")
        except Exception as e:
            logger.error(f"Error starting trading system: {e}")
    
    def stop(self) -> None:
        """Stop the application"""
        try:
            logger.info("Stopping CryptoVisionAI")
            
            # Stop trading system
            if self.trading_system:
                self.trading_system.stop()
                logger.info("Trading system stopped")
            
            # Stop dashboard
            if self.dashboard:
                self.dashboard.stop()
                logger.info("Dashboard stopped")
                
        except Exception as e:
            logger.error(f"Error stopping application: {e}")
            raise


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CryptoVisionAI - Bitcoin price prediction and trading system")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["test", "paper", "live"], help="Trading mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Create the application
        app = CryptoVisionAI(args.config)
        
        # Override settings from command line if provided
        if args.mode:
            app.config["general"]["mode"] = args.mode
        if args.debug:
            app.config["general"]["debug"] = True
        
        # Initialize and start the application
        app.initialize_components()
        app.start()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
    except Exception as e:
        logger.exception(f"Unhandled exception: {e}")