#!/usr/bin/env python3
"""
Production Transition Script for CryptoVisionAI

This script guides you through the process of transitioning from test/development
to production mode, performing safety checks and configuration validation.
"""

import os
import sys
import yaml
import shutil
import getpass
import logging
import hashlib
import datetime
import subprocess
from typing import Dict, Any, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/production_transition_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    ]
)
logger = logging.getLogger("production_transition")

# Define colors for terminal output
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

def print_header(title: str) -> None:
    """Print a formatted header"""
    width = 80
    print("\n" + "="*width)
    print(f"{Colors.BOLD}{Colors.BLUE}{title.center(width)}{Colors.END}")
    print("="*width + "\n")

def print_success(message: str) -> None:
    """Print a success message"""
    print(f"{Colors.GREEN}✓ {message}{Colors.END}")

def print_warning(message: str) -> None:
    """Print a warning message"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_error(message: str) -> None:
    """Print an error message"""
    print(f"{Colors.RED}✗ {message}{Colors.END}")

def print_step(step: str, index: int = None) -> None:
    """Print a step message"""
    if index:
        print(f"{Colors.BOLD}Step {index}: {step}{Colors.END}")
    else:
        print(f"{Colors.BOLD}{step}{Colors.END}")

def confirm(message: str) -> bool:
    """Ask for user confirmation"""
    response = input(f"{Colors.YELLOW}{message} (y/n): {Colors.END}").lower().strip()
    return response in ('y', 'yes')

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)

def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    try:
        with open(config_path, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)
        print_success(f"Configuration saved to {config_path}")
    except Exception as e:
        print_error(f"Failed to save config to {config_path}: {e}")
        logger.error(f"Failed to save config to {config_path}: {e}")

def check_api_keys(config: Dict[str, Any]) -> bool:
    """Check if API keys are configured properly"""
    placeholder_keys = [
        "YOUR_PRODUCTION_API_KEY",
        "YOUR_PRODUCTION_API_SECRET",
        "YOUR_API_KEY",
        "YOUR_API_SECRET",
        "YOUR_MAINNET_API_KEY",
        "YOUR_MAINNET_API_SECRET"
    ]
    
    # Check API keys in binance section
    binance_api_key = config.get("binance", {}).get("api_key", "")
    binance_api_secret = config.get("binance", {}).get("api_secret", "")
    
    # Check API keys in api.binance.mainnet section
    mainnet_api_key = config.get("api", {}).get("binance", {}).get("mainnet", {}).get("api_key", "")
    mainnet_api_secret = config.get("api", {}).get("binance", {}).get("mainnet", {}).get("api_secret", "")
    
    if any(key in [binance_api_key, binance_api_secret, mainnet_api_key, mainnet_api_secret] for key in placeholder_keys):
        print_error("API keys contain placeholder values")
        return False
        
    if not binance_api_key or not binance_api_secret or not mainnet_api_key or not mainnet_api_secret:
        print_error("API keys are empty")
        return False
        
    if binance_api_key != mainnet_api_key:
        print_warning("API key mismatch between 'binance' and 'api.binance.mainnet' sections")
        
    print_success("API keys appear to be properly configured")
    return True

def check_alert_configuration(config: Dict[str, Any]) -> bool:
    """Check if alerts are properly configured"""
    alerts_config = config.get("alerts", {})
    
    if not alerts_config:
        print_error("Alerts configuration is missing")
        return False
        
    alerts_enabled = alerts_config.get("alerts_enabled", False)
    if not alerts_enabled:
        print_warning("Alerts are not enabled")
        return False
        
    test_mode = alerts_config.get("test_mode", True)
    if test_mode:
        print_warning("Alerts are in test mode - they won't be sent to external services")
        
    email_enabled = alerts_config.get("email", {}).get("enabled", False)
    if not email_enabled:
        print_warning("Email alerts are not enabled")
    else:
        email_config = alerts_config.get("email", {})
        if "YOUR_" in email_config.get("smtp_password", ""):
            print_error("Email SMTP password contains placeholder value")
            return False
        if not email_config.get("recipients", []):
            print_error("No email recipients configured")
            return False
            
    sms_enabled = alerts_config.get("sms", {}).get("enabled", False)
    if not sms_enabled:
        print_warning("SMS alerts are not enabled - recommended for critical alerts")
        
    print_success("Alert configuration appears to be valid")
    return True

def check_risk_parameters(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Check risk management parameters for production safety"""
    risk_config = config.get("risk_management", {})
    warnings = []
    
    # Check parameters against safe thresholds
    if risk_config.get("max_position_size", 20.0) > 5.0:
        warnings.append(f"max_position_size is high: {risk_config.get('max_position_size')}% (recommended ≤5%)")
        
    if risk_config.get("max_open_positions", 10) > 5:
        warnings.append(f"max_open_positions is high: {risk_config.get('max_open_positions')} (recommended ≤5)")
        
    if risk_config.get("stop_loss_percent", 5.0) > 5.0:
        warnings.append(f"stop_loss_percent is high: {risk_config.get('stop_loss_percent')}% (recommended ≤5%)")
        
    if risk_config.get("risk_per_trade", 1.0) > 1.0:
        warnings.append(f"risk_per_trade is high: {risk_config.get('risk_per_trade')}% (recommended ≤1%)")
        
    if risk_config.get("max_drawdown_limit", 20.0) > 15.0:
        warnings.append(f"max_drawdown_limit is high: {risk_config.get('max_drawdown_limit')}% (recommended ≤15%)")
        
    # Return True if no warnings, otherwise False with warnings
    return len(warnings) == 0, warnings

def check_auth_configuration(config: Dict[str, Any]) -> bool:
    """Check if authentication is properly configured"""
    auth_config = config.get("auth", {})
    
    if not auth_config.get("enabled", False):
        print_error("Authentication is not enabled")
        return False
        
    jwt_secret = auth_config.get("jwt_secret", "")
    if not jwt_secret or jwt_secret == "GENERATE_A_STRONG_RANDOM_SECRET_KEY_HERE":
        print_error("JWT secret is not configured")
        return False
        
    users = auth_config.get("users", {})
    if not users:
        print_error("No users configured for authentication")
        return False
        
    has_placeholder_password = False
    for username, user_data in users.items():
        if user_data.get("password_hash", "").startswith("REPLACE_WITH_"):
            print_error(f"User '{username}' has placeholder password hash")
            has_placeholder_password = True
            
    if has_placeholder_password:
        return False
        
    print_success("Authentication configuration appears to be valid")
    return True

def generate_production_config() -> Dict[str, Any]:
    """Generate production configuration from template and user input"""
    print_step("Generating production configuration", 1)
    
    # Load template
    template_path = "config/production_config_template.yaml"
    if not os.path.exists(template_path):
        print_error(f"Production template not found at {template_path}")
        sys.exit(1)
        
    config = load_config(template_path)
    
    # Let's set some critical values
    print("\nAPI Key Configuration")
    print("--------------------")
    print("Please enter your Binance production API credentials:")
    
    api_key = getpass.getpass("API Key: ")
    api_secret = getpass.getpass("API Secret: ")
    
    # Update API keys in multiple locations
    config["binance"]["api_key"] = api_key
    config["binance"]["api_secret"] = api_secret
    config["api"]["binance"]["mainnet"]["api_key"] = api_key
    config["api"]["binance"]["mainnet"]["api_secret"] = api_secret
    
    # Email configuration 
    print("\nAlert Email Configuration")
    print("------------------------")
    if confirm("Would you like to configure email alerts?"):
        config["alerts"]["email"]["enabled"] = True
        config["alerts"]["email"]["smtp_server"] = input("SMTP Server: ")
        config["alerts"]["email"]["smtp_port"] = int(input("SMTP Port (usually 587 for TLS): "))
        config["alerts"]["email"]["smtp_user"] = input("SMTP Username: ")
        config["alerts"]["email"]["smtp_password"] = getpass.getpass("SMTP Password: ")
        config["alerts"]["email"]["sender"] = input("Sender Email Address: ")
        
        recipients = []
        print("Enter email recipients (one per line, leave empty to finish):")
        while True:
            recipient = input("> ")
            if not recipient:
                break
            recipients.append(recipient)
        
        if recipients:
            config["alerts"]["email"]["recipients"] = recipients
    else:
        config["alerts"]["email"]["enabled"] = False
    
    # SMS configuration
    print("\nAlert SMS Configuration")
    print("----------------------")
    if confirm("Would you like to configure SMS alerts (requires Twilio account)?"):
        config["alerts"]["sms"]["enabled"] = True
        config["alerts"]["sms"]["account_sid"] = input("Twilio Account SID: ")
        config["alerts"]["sms"]["auth_token"] = getpass.getpass("Twilio Auth Token: ")
        config["alerts"]["sms"]["from_number"] = input("Twilio Phone Number (with +): ")
        
        recipients = []
        print("Enter SMS recipients (one per line, leave empty to finish):")
        while True:
            recipient = input("> ")
            if not recipient:
                break
            recipients.append(recipient)
        
        if recipients:
            config["alerts"]["sms"]["recipients"] = recipients
    else:
        config["alerts"]["sms"]["enabled"] = False
    
    # Risk management
    print("\nRisk Management Configuration")
    print("----------------------------")
    print("Suggested production values are shown in brackets.")
    
    max_position_size = float(input("Maximum position size as percentage of portfolio [3.0]: ") or "3.0")
    max_open_positions = int(input("Maximum number of open positions [3]: ") or "3")
    stop_loss_percent = float(input("Default stop loss percentage [3.0]: ") or "3.0")
    max_drawdown_limit = float(input("Maximum drawdown percentage before trading stops [15.0]: ") or "15.0")
    
    config["risk_management"]["max_position_size"] = max_position_size
    config["risk_management"]["max_open_positions"] = max_open_positions
    config["risk_management"]["stop_loss_percent"] = stop_loss_percent
    config["risk_management"]["max_drawdown_limit"] = max_drawdown_limit
    
    # Authentication
    print("\nAuthentication Configuration")
    print("---------------------------")
    if not config["auth"]["enabled"]:
        print_error("Authentication must be enabled in production")
        config["auth"]["enabled"] = True
        
    # Generate JWT secret
    jwt_secret = hashlib.sha256(os.urandom(32)).hexdigest()
    config["auth"]["jwt_secret"] = jwt_secret
    print_success("Generated secure JWT secret")
    
    # Setup admin user
    admin_password = getpass.getpass("Enter password for 'admin' user: ")
    if admin_password:
        # In a real implementation, you would hash this password properly
        # This is just a placeholder - use proper password hashing in production
        import bcrypt  # You would need to install this
        password_hash = bcrypt.hashpw(admin_password.encode(), bcrypt.gensalt()).decode()
        config["auth"]["users"]["admin"]["password_hash"] = password_hash
        print_success("Admin password set")
    
    return config

def finalize_configuration(config: Dict[str, Any]) -> None:
    """Final checks and save production configuration"""
    print_step("Finalizing configuration", 2)
    
    # Make sure critical production settings are correct
    config["general"]["mode"] = "live"
    config["general"]["debug"] = False
    config["binance"]["use_testnet"] = False
    config["trading"]["test_mode"] = False
    config["alerts"]["test_mode"] = False
    config["dashboard"]["debug"] = False
    
    # Verify settings
    print("\nVerifying critical production settings:")
    print(f"- General mode: {config['general']['mode']}")
    print(f"- Debug mode: {config['general']['debug']}")
    print(f"- Using testnet: {config['binance']['use_testnet']}")
    print(f"- Test trading: {config['trading']['test_mode']}")
    print(f"- Alert test mode: {config['alerts']['test_mode']}")
    
    if not confirm("\nThese settings will execute LIVE trades with REAL money. Are you sure you want to continue?"):
        print_warning("Configuration not saved. Exiting.")
        sys.exit(0)
    
    # Check for critical issues
    api_keys_valid = check_api_keys(config)
    alerts_valid = check_alert_configuration(config)
    risk_valid, risk_warnings = check_risk_parameters(config)
    auth_valid = check_auth_configuration(config)
    
    # Display risk warnings
    if risk_warnings:
        print("\nRisk parameter warnings:")
        for warning in risk_warnings:
            print_warning(warning)
        
    # Check overall validation
    if not api_keys_valid or not auth_valid:
        print_error("\nCritical issues found in configuration. Please fix them before proceeding.")
        if confirm("Would you like to save this configuration anyway for further editing?"):
            output_path = "config/production_config_DRAFT.yaml"
            save_config(config, output_path)
            print_warning(f"Draft configuration saved to {output_path}")
        sys.exit(1)
    
    # Finalize and save
    output_path = "config/production_config.yaml"
    
    # Backup existing file if it exists
    if os.path.exists(output_path):
        backup_path = f"{output_path}.bak.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        shutil.copy2(output_path, backup_path)
        print_success(f"Backup of existing configuration saved to {backup_path}")
    
    # Save new configuration
    save_config(config, output_path)
    print_success(f"Production configuration saved to {output_path}")
    
    # Print next steps
    print("\nNext Steps:")
    print("1. Review the production checklist in docs/production_checklist.md")
    print("2. Test the system with the production configuration in test mode first")
    print("3. When ready, deploy to production using:")
    print(f"   ./start_dashboard.sh --config {output_path}")

def main() -> None:
    """Main function"""
    print_header("CryptoVisionAI Production Transition Script")
    
    print(f"{Colors.RED}{Colors.BOLD}WARNING: This script will help you prepare for production deployment.{Colors.END}")
    print(f"{Colors.RED}{Colors.BOLD}Trading with real funds involves financial risk.{Colors.END}\n")
    
    if not confirm("Do you want to continue with the production setup?"):
        print("Setup canceled.")
        return
    
    # Check for production checklist
    checklist_path = "docs/production_checklist.md"
    if os.path.exists(checklist_path):
        print_success(f"Found production checklist at {checklist_path}")
        if confirm("Would you like to review the production checklist now?"):
            # Try to open the checklist with system default application
            try:
                if sys.platform == "darwin":  # macOS
                    subprocess.call(["open", checklist_path])
                elif sys.platform == "win32":  # Windows
                    os.startfile(checklist_path)
                else:  # Linux
                    subprocess.call(["xdg-open", checklist_path])
            except:
                print_warning(f"Could not open {checklist_path}. Please open it manually.")
    else:
        print_warning(f"Production checklist not found at {checklist_path}")
    
    # Generate production config
    config = generate_production_config()
    
    # Finalize configuration
    finalize_configuration(config)
    
if __name__ == "__main__":
    main()