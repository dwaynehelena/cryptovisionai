#!/usr/bin/env python3
"""
Dependency Manager - Handles checking and installing required dependencies
"""

import subprocess
import sys
import importlib.util
import logging
from typing import Dict, List, Tuple, Optional, Set

# Configure logging
logger = logging.getLogger("dependency_manager")

# Core requirements for the application
CORE_REQUIREMENTS = {
    # Data handling
    "numpy": "1.26.4",
    "pandas": "2.2.0",
    "scipy": "1.12.0",
    
    # Visualization
    "dash": "2.15.0",
    "dash-bootstrap-components": "1.5.0",
    "plotly": "5.18.0",
    "matplotlib": "3.8.2",  # Added for data visualization and analysis
    
    # API connections
    "requests": "2.31.0",
    "python-binance": "1.0.32",
    
    # Machine Learning
    "scikit-learn": "1.4.0",
    "imbalanced-learn": "0.12.0",
    "torch": "2.2.0",
    "tensorflow": "2.15.0",  # Updated to match requirements.txt
    "xgboost": "2.0.3",      # Added xgboost to core requirements
    "lightgbm": "4.3.0",     # Added lightgbm to core requirements
    "ta-lib": "0.6.3",      # Updated to version available in PyPI
    
    # Monitoring
    "psutil": "5.9.8",
    
    # Utilities
    "python-dotenv": "1.0.1",
    "pyyaml": "6.0.1",
}

# Additional requirements for specific modules
MODULE_REQUIREMENTS = {
    "training": {
        "optuna": "3.5.0",
        "lightgbm": "4.3.0",
        "xgboost": "2.0.3"
    },
    "backtesting": {
        "backtrader": "1.9.78.123",
        "pyfolio": "0.9.2"
    },
    "production": {
        "prometheus-client": "0.19.0",
        "fastapi": "0.109.0",
        "uvicorn": "0.27.0"
    }
}

def is_package_installed(package_name: str) -> bool:
    """
    Check if a package is installed
    
    Args:
        package_name: Name of the package to check
        
    Returns:
        bool: True if package is installed, False otherwise
    """
    return importlib.util.find_spec(package_name) is not None

def get_package_version(package_name: str) -> Optional[str]:
    """
    Get the installed version of a package
    
    Args:
        package_name: Name of the package
        
    Returns:
        str: Version string or None if package is not installed
    """
    try:
        module = __import__(package_name)
        return getattr(module, '__version__', None)
    except ImportError:
        return None

def check_dependencies(include_modules: Optional[List[str]] = None) -> Tuple[bool, Dict[str, str]]:
    """
    Check if all required dependencies are installed
    
    Args:
        include_modules: List of additional modules to include in check
        
    Returns:
        Tuple[bool, Dict[str, str]]: (all_installed, {missing_package: required_version})
    """
    all_installed = True
    missing_packages = {}
    
    # Check for environment variables to skip problematic dependencies
    import os
    skip_talib = os.environ.get("SKIP_TALIB", "0") == "1"
    skip_lightgbm = os.environ.get("SKIP_LIGHTGBM", "0") == "1"
    
    # Check core requirements
    for package_name, version in CORE_REQUIREMENTS.items():
        # Skip TA-Lib if environment variable is set
        if package_name == "ta-lib" and skip_talib:
            logger.warning("Skipping TA-Lib check due to SKIP_TALIB environment variable")
            continue
        
        # Skip LightGBM if environment variable is set
        if package_name == "lightgbm" and skip_lightgbm:
            logger.warning("Skipping LightGBM check due to SKIP_LIGHTGBM environment variable")
            continue
            
        if not is_package_installed(package_name):
            all_installed = False
            missing_packages[package_name] = version
    
    # Check additional module requirements
    if include_modules:
        for module_name in include_modules:
            if module_name in MODULE_REQUIREMENTS:
                for package_name, version in MODULE_REQUIREMENTS[module_name].items():
                    if not is_package_installed(package_name):
                        all_installed = False
                        missing_packages[package_name] = version
    
    return all_installed, missing_packages

def install_packages(packages: Dict[str, str], verbose: bool = False) -> Tuple[bool, List[str], List[str]]:
    """
    Install missing packages using pip
    
    Args:
        packages: Dictionary of {package_name: version}
        verbose: Whether to print verbose output
        
    Returns:
        Tuple[bool, List[str], List[str]]: (success, installed_packages, failed_packages)
    """
    installed = []
    failed = []
    
    for package_name, version in packages.items():
        package_spec = f"{package_name}=={version}"
        logger.info(f"Installing {package_spec}")
        
        try:
            output = subprocess.check_output(
                [sys.executable, "-m", "pip", "install", package_spec],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            installed.append(package_name)
            if verbose:
                logger.info(output)
        except subprocess.CalledProcessError as e:
            failed.append(package_name)
            logger.error(f"Failed to install {package_name}: {e.output}")
    
    success = len(failed) == 0
    return success, installed, failed

def ensure_dependencies(include_modules: Optional[List[str]] = None, auto_install: bool = False, verbose: bool = False) -> bool:
    """
    Check dependencies and optionally install missing ones
    
    Args:
        include_modules: List of additional modules to include in check
        auto_install: Whether to automatically install missing packages
        verbose: Whether to print verbose output
        
    Returns:
        bool: True if all dependencies are satisfied after potential installation
    """
    all_installed, missing_packages = check_dependencies(include_modules)
    
    if not all_installed:
        missing_str = ", ".join(missing_packages.keys())
        logger.warning(f"Missing dependencies: {missing_str}")
        
        if auto_install:
            logger.info("Installing missing dependencies...")
            success, installed, failed = install_packages(missing_packages, verbose)
            
            if success:
                logger.info(f"Successfully installed all missing dependencies: {', '.join(installed)}")
                return True
            else:
                if installed:
                    logger.info(f"Successfully installed: {', '.join(installed)}")
                if failed:
                    logger.error(f"Failed to install: {', '.join(failed)}")
                return False
        else:
            logger.info("Use pip to install missing dependencies:")
            for package, version in missing_packages.items():
                logger.info(f"  pip install {package}=={version}")
            return False
    
    return True

def get_all_requirements() -> Dict[str, str]:
    """
    Get all requirements for the project
    
    Returns:
        Dict[str, str]: Dictionary of all package requirements
    """
    all_requirements = {**CORE_REQUIREMENTS}
    
    for module_reqs in MODULE_REQUIREMENTS.values():
        all_requirements.update(module_reqs)
    
    return all_requirements

def generate_requirements_txt(filepath: str = "requirements.txt") -> None:
    """
    Generate requirements.txt file with all dependencies
    
    Args:
        filepath: Path to output file
    """
    all_requirements = get_all_requirements()
    with open(filepath, "w") as f:
        for package, version in sorted(all_requirements.items()):
            f.write(f"{package}=={version}\n")
    
    logger.info(f"Generated {filepath} with {len(all_requirements)} packages")

if __name__ == "__main__":
    # Configure logging for command line usage
    logging.basicConfig(level=logging.INFO)
    
    # If invoked directly, check all dependencies for all modules
    import argparse
    parser = argparse.ArgumentParser(description="Check and install dependencies for CryptoVisionAI")
    parser.add_argument("--install", action="store_true", help="Automatically install missing dependencies")
    parser.add_argument("--verbose", action="store_true", help="Show verbose output")
    parser.add_argument("--modules", nargs="+", help="Specific modules to check dependencies for", 
                        choices=list(MODULE_REQUIREMENTS.keys()) + ["all"])
    parser.add_argument("--generate-requirements", action="store_true", help="Generate requirements.txt file")
    
    args = parser.parse_args()
    
    if args.generate_requirements:
        generate_requirements_txt()
        sys.exit(0)
    
    modules = None
    if args.modules:
        if "all" in args.modules:
            modules = list(MODULE_REQUIREMENTS.keys())
        else:
            modules = args.modules
    
    result = ensure_dependencies(include_modules=modules, auto_install=args.install, verbose=args.verbose)
    sys.exit(0 if result else 1)