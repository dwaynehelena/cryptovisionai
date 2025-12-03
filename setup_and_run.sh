#!/bin/bash

# CryptoVisionAI Setup & Run Script
# This script creates a virtual environment, installs dependencies, and runs the ensemble model

# Set script to exit immediately if any command fails
set -e

echo "===== CryptoVisionAI Environment Setup & Run ====="
echo "Setting up environment and running ensemble model training..."

# Define environment and log directories
ENV_DIR="venv"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SETUP_LOG="${LOG_DIR}/setup_${TIMESTAMP}.log"

# Create logs directory if it doesn't exist
mkdir -p "$LOG_DIR"

echo "Setting up virtual environment..."

# Check if Python 3 is available
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python 3 is not installed or not in PATH. Please install Python 3.7 or newer."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Found Python version: $PYTHON_VERSION"

# Create virtual environment if it doesn't already exist
if [ ! -d "$ENV_DIR" ]; then
    echo "Creating new virtual environment in $ENV_DIR..."
    $PYTHON_CMD -m venv "$ENV_DIR"
else
    echo "Using existing virtual environment in $ENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source cryptovision_py39_env/bin/activate"

# Check for pip
if ! command -v pip &> /dev/null; then
    echo "Error: pip not found in virtual environment."
    echo "Attempting to install pip..."
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $PYTHON_CMD get-pip.py
    rm get-pip.py
fi

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt 2>&1 | tee -a "$SETUP_LOG"

# Special handling for ta-lib which often has installation issues
if ! pip list | grep -q "ta-lib"; then
    echo "TA-Lib installation may have failed. This is a common issue."
    echo "You may need to install TA-Lib system dependencies first."
    echo "For macOS: brew install ta-lib"
    echo "For Ubuntu: sudo apt-get install build-essential ta-lib"
    echo "Continuing with other packages..."
fi

echo "Dependencies installed. Setting up environment variables..."

# Create .env file if it doesn't exist (with template values)
if [ ! -f ".env" ]; then
    echo "Creating template .env file..."
    cat > .env << EOL
# API Keys (replace with your actual API keys)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here

# Environment Settings
USE_TESTNET=True
DEBUG_MODE=True
LOG_LEVEL=INFO
EOL
    echo "Created .env file template. Please edit with your actual API keys."
fi

# Run the ensemble model training script
echo "Starting ensemble model training..."
$PYTHON_CMD run_ensemble_model.py 2>&1 | tee -a "$SETUP_LOG"

echo "Process complete. Check logs for details."
echo "Log file: $SETUP_LOG"
echo "To activate this environment in the future, run: source cryptovision_py39_env/bin/activate"