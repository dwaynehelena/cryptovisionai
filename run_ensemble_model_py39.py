#!/usr/bin/env python3
"""
Wrapper script to ensure the ensemble model training runs with Python 3.9
This script checks the Python version before executing the main training script
"""

import sys
import os
import platform
import subprocess

def check_python_version():
    """Check if the current Python version is 3.9.x"""
    version_info = sys.version_info
    return version_info.major == 3 and version_info.minor == 9

def main():
    """Main function to check Python version and run the ensemble model"""
    print("CryptoVisionAI Ensemble Model Training")
    print("--------------------------------------")
    print(f"Current Python version: {platform.python_version()}")
    
    if not check_python_version():
        print("\n⚠️ WARNING: This script should be run with Python 3.9.x")
        print("Current training is using incompatible Python version.")
        print("\nOptions to proceed:")
        print("1. Create a new virtual environment with Python 3.9:")
        print("   - Install Python 3.9 if not already installed")
        print("   - Run: python3.9 -m venv cryptovision_py39_env")
        print("   - Activate: source cryptovision_py39_env/bin/activate")
        print("   - Install dependencies: pip install -r requirements.txt")
        print("   - Run this script again: python run_ensemble_model_py39.py")
        print("\n2. Continue anyway with current Python version (not recommended)")
        
        choice = input("\nDo you want to continue anyway? (y/N): ").strip().lower()
        if choice != 'y':
            print("Exiting. Please run with Python 3.9.x for best compatibility.")
            return
        
        print("\n⚠️ Proceeding with incompatible Python version. Some features may not work correctly.")
    
    print("\nStarting ensemble model training...")
    # Import and run the actual model training script
    try:
        import run_ensemble_model
        # Change the parameters if desired
        run_ensemble_model.run_ensemble_model(
            symbols=["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "DOTUSDT"],
            feature_set="advanced",
            n_features=136,
            epochs=150
        )
    except ImportError as e:
        print(f"\n❌ Error importing run_ensemble_model: {e}")
        print("This could be due to missing dependencies. Please ensure all requirements are installed.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error during model training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()