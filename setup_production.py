#!/usr/bin/env python3
"""
Simplified Production Setup Script for CryptoVisionAI

This script helps update the production configuration with secure credentials
and validates critical settings.
"""

import os
import sys
import yaml
import secrets
import getpass
import hashlib
import datetime
from pathlib import Path

# Configuration file paths
CONFIG_PATH = "config/production_config.yaml"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "="*80)
    print(f"{title.center(80)}")
    print("="*80 + "\n")

def print_success(message):
    """Print a success message"""
    print(f"✓ {message}")

def print_warning(message):
    """Print a warning message"""
    print(f"⚠️  {message}")

def print_error(message):
    """Print an error message"""
    print(f"✗ {message}")

def load_config(config_path):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        print_error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def save_config(config, config_path):
    """Save configuration to YAML file"""
    try:
        # Create backup
        if os.path.exists(config_path):
            backup_path = f"{config_path}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.rename(config_path, backup_path)
            print_success(f"Created backup at {backup_path}")
            
        # Save new config
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        print_success(f"Configuration saved to {config_path}")
    except Exception as e:
        print_error(f"Failed to save config: {e}")
        sys.exit(1)

def update_api_keys(config):
    """Update API keys in configuration"""
    print("\n--- API Key Configuration ---")
    print("Please enter your Binance production API credentials:")
    
    api_key = input("API Key [leave blank to skip]: ")
    if api_key:
        api_secret = getpass.getpass("API Secret: ")
        
        # Update in both locations
        config["api"]["binance"]["mainnet"]["api_key"] = api_key
        config["api"]["binance"]["mainnet"]["api_secret"] = api_secret
        config["binance"]["api_key"] = api_key
        config["binance"]["api_secret"] = api_secret
        
        print_success("API keys updated")
    else:
        print_warning("API keys not updated - you'll need to edit the config file manually")

def update_email_config(config):
    """Update email notification settings"""
    print("\n--- Email Alert Configuration ---")
    enable = input("Enable email alerts? (y/n) [n]: ").lower() == 'y'
    
    config["alerts"]["email"]["enabled"] = enable
    if enable:
        config["alerts"]["email"]["smtp_server"] = input("SMTP Server [smtp.gmail.com]: ") or "smtp.gmail.com"
        config["alerts"]["email"]["smtp_port"] = int(input("SMTP Port [587]: ") or "587")
        config["alerts"]["email"]["smtp_user"] = input("SMTP Username (email address): ")
        config["alerts"]["email"]["smtp_password"] = getpass.getpass("SMTP Password/App Password: ")
        config["alerts"]["email"]["sender"] = input("Sender Email [same as username]: ") or config["alerts"]["email"]["smtp_user"]
        
        # Recipients
        recipients = []
        print("Enter email recipients (one per line, leave blank to finish):")
        while True:
            recipient = input("> ")
            if not recipient:
                break
            recipients.append(recipient)
        
        if recipients:
            config["alerts"]["email"]["recipients"] = recipients
        
        print_success("Email alert configuration updated")
    else:
        print_warning("Email alerts disabled")

def update_auth_config(config):
    """Update authentication settings"""
    print("\n--- Authentication Configuration ---")
    # Always enable authentication for production
    config["auth"]["enabled"] = True
    
    # Generate secure JWT secret
    jwt_secret = secrets.token_hex(32)  # 64 character hex string
    config["auth"]["jwt_secret"] = jwt_secret
    print_success("Generated secure JWT secret")
    
    # Set admin password
    print("Enter password for 'admin' user (leave blank to skip):")
    password = getpass.getpass("> ")
    if password:
        # Simple password hashing for demonstration
        # In production, you should use a proper hashing library like bcrypt
        hash_value = hashlib.sha256(password.encode()).hexdigest()
        config["auth"]["users"]["admin"]["password_hash"] = hash_value
        print_success("Admin password updated")
    else:
        print_warning("Admin password not updated")

def update_risk_settings(config):
    """Update risk management settings"""
    print("\n--- Risk Management Settings ---")
    print("Current settings (recommended values in brackets):")
    print(f"- max_portfolio_risk: {config['risk_management']['max_portfolio_risk']}% (2-5%)")
    print(f"- max_position_size: {config['risk_management']['max_position_size']}% (3-5%)")
    print(f"- max_open_positions: {config['risk_management']['max_open_positions']} (3-5)")
    print(f"- stop_loss_percent: {config['risk_management']['stop_loss_percent']}% (3-5%)")
    
    if input("Do you want to adjust these settings? (y/n) [n]: ").lower() == 'y':
        try:
            config["risk_management"]["max_portfolio_risk"] = float(input(f"max_portfolio_risk [{config['risk_management']['max_portfolio_risk']}%]: ") or config["risk_management"]["max_portfolio_risk"])
            config["risk_management"]["max_position_size"] = float(input(f"max_position_size [{config['risk_management']['max_position_size']}%]: ") or config["risk_management"]["max_position_size"])
            config["risk_management"]["max_open_positions"] = int(input(f"max_open_positions [{config['risk_management']['max_open_positions']}]: ") or config["risk_management"]["max_open_positions"])
            config["risk_management"]["stop_loss_percent"] = float(input(f"stop_loss_percent [{config['risk_management']['stop_loss_percent']}%]: ") or config["risk_management"]["stop_loss_percent"])
            print_success("Risk management settings updated")
        except ValueError:
            print_error("Invalid input. Using default values.")

def update_trading_mode(config):
    """Configure trading mode settings"""
    print("\n--- Trading Mode Configuration ---")
    print("CRITICAL: These settings determine whether real trades will be executed")
    
    # Start with safe settings
    config["binance"]["use_testnet"] = True
    config["trading"]["test_mode"] = True
    
    real_trading = input("Enable REAL trading with REAL funds? (yes/no) [no]: ").lower() == 'yes'
    
    if real_trading:
        print_warning("⚠️  WARNING: You are enabling REAL trading with REAL funds!")
        confirm = input("Type 'I UNDERSTAND THE RISKS' to confirm: ")
        
        if confirm == "I UNDERSTAND THE RISKS":
            config["binance"]["use_testnet"] = False
            config["trading"]["test_mode"] = False
            print_warning("REAL trading has been ENABLED!")
        else:
            print_warning("Confirmation failed. Keeping safe settings (testnet/test_mode)")
    else:
        print_success("Using safe settings: testnet=True, test_mode=True (no real trades)")

def main():
    """Main function"""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    print_header("CryptoVisionAI Production Setup")
    print("This script will help you configure essential settings for production deployment.")
    print("Note: This is a simplified version of the full production transition process.")
    
    if not os.path.exists(CONFIG_PATH):
        print_error(f"Configuration file not found: {CONFIG_PATH}")
        sys.exit(1)
        
    # Load existing configuration
    config = load_config(CONFIG_PATH)
    
    # Update configurations
    update_trading_mode(config)
    update_api_keys(config)
    update_email_config(config)
    update_auth_config(config)
    update_risk_settings(config)
    
    # Save updated configuration
    save_config(config, CONFIG_PATH)
    
    # Print next steps
    print("\n--- Next Steps ---")
    print("1. Review the production checklist: docs/production_checklist.md")
    print("2. Test your configuration in test mode first before enabling real trading")
    print("3. Start the system using: ./start_dashboard.sh --config config/production_config.yaml")
    print("\nRemember to monitor the system closely when first moving to production!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        sys.exit(0)