#!/usr/bin/env python3
"""
Startup Script for CryptoVisionAI Dashboard

This script initializes and starts the CryptoVisionAI trading system and dashboard.
It handles configuration loading, logging setup, and provides command line options
for customizing the startup process.
"""

import os
import sys
import argparse
import logging
import yaml
from datetime import datetime
import importlib.util
import importlib

# Ensure the src directory is in the path for imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Set up logging before imports to catch any early errors
log_dir = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("startup")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CryptoVisionAI Dashboard Startup")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["test", "paper", "live"], default="test", help="Trading mode")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", type=int, help="Dashboard port")
    parser.add_argument("--host", type=str, help="Dashboard host")
    parser.add_argument("--symbol", type=str, default="BTCUSDT", help="Trading symbol")
    parser.add_argument("--no-trading", action="store_true", help="Start dashboard without trading system")
    return parser.parse_args()

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error loading configuration from {config_path}: {e}")
        logger.info("Using default configuration")
        return None

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = [
        # Package name and import name (if different)
        ("dash", "dash"),
        ("plotly", "plotly"),
        ("pandas", "pandas"),
        ("numpy", "numpy"),
        ("dash-bootstrap-components", "dash_bootstrap_components"),
        ("python-binance", "binance"),
        ("pyyaml", "yaml"),
        ("scikit-learn", "sklearn"),
        ("tensorflow", "tensorflow"),
        ("xgboost", "xgboost")
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            # Try two methods of checking if a package is installed
            try:
                # Method 1: Try to import the module
                importlib.import_module(import_name)
            except ImportError:
                # Method 2: Check if the module spec exists
                if importlib.util.find_spec(import_name) is None:
                    missing_packages.append(package_name)
        except Exception:
            missing_packages.append(package_name)
    
    if missing_packages:
        logger.warning(f"Missing required packages: {', '.join(missing_packages)}")
        logger.warning("Please install missing packages: pip install -r requirements.txt")
        return False
    
    return True

def create_asset_directories():
    """Create asset directories for the dashboard"""
    assets_dir = os.path.join(os.path.dirname(__file__), "src", "visualization", "assets")
    os.makedirs(assets_dir, exist_ok=True)
    
    # Create or check for logo file
    logo_path = os.path.join(assets_dir, "logo.png")
    if not os.path.exists(logo_path):
        logger.warning(f"Logo file not found at {logo_path}")
        logger.info("Dashboard will run without a logo")

def main():
    """Main function to start the dashboard"""
    # Parse arguments
    args = parse_arguments()
    
    # Check dependencies
    if not check_dependencies():
        logger.warning("Missing dependencies may cause issues")
    
    # Create necessary directories
    create_asset_directories()
    
    # Load configuration
    config = load_config(args.config)
    if config is None:
        logger.error("Failed to load configuration. Exiting.")
        sys.exit(1)
        
    # Override config with command line arguments
    if args.mode:
        config["general"]["mode"] = args.mode
    if args.debug:
        config["general"]["debug"] = args.debug
    if args.port:
        config["dashboard"]["port"] = args.port
    if args.host:
        config["dashboard"]["host"] = args.host
    
    logger.info(f"Starting CryptoVisionAI in {config['general']['mode']} mode")
    
    try:
        # Start the application - import here to avoid circular imports
        from src.app import CryptoVisionAI
        
        # Create application instance
        app = CryptoVisionAI(args.config)
        
        # Initialize components
        app.initialize_components()
        
        # Start application
        app.start()
        
    except KeyboardInterrupt:
        logger.info("Application interrupted by user. Shutting down...")
    except Exception as e:
        logger.exception(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()