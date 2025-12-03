#!/bin/bash
# Unified Environment Script for CryptoVisionAI
# This script creates a single unified environment for all operations

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Set environment name
ENV_NAME="cryptovision_unified_env"
PYTHON_VERSION="3.9"

echo "🔄 Creating unified environment for CryptoVisionAI..."

# Check if the environment already exists
if [ -d "$SCRIPT_DIR/$ENV_NAME" ]; then
    echo "⚠️  Environment $ENV_NAME already exists."
    read -p "Do you want to recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing old environment..."
        rm -rf "$SCRIPT_DIR/$ENV_NAME"
    else
        echo "✅ Using existing environment."
        source "$SCRIPT_DIR/$ENV_NAME/bin/activate"
        echo "Checking dependencies..."
        python -m src.utils.dependency_manager --install
        echo "✅ Environment is ready. Activated: $ENV_NAME"
        exit 0
    fi
fi

echo "🔧 Creating new Python $PYTHON_VERSION virtual environment: $ENV_NAME"
python$PYTHON_VERSION -m venv "$ENV_NAME"

echo "🔌 Activating environment..."
source "$ENV_NAME/bin/activate"

echo "🔄 Upgrading pip..."
pip install --upgrade pip

echo "📦 Installing dependencies..."
# Generate requirements.txt with all necessary packages
python -m src.utils.dependency_manager --generate-requirements
pip install -r requirements.txt

# Create symbolic links to env directories
ln -sf "$ENV_NAME" cryptovision_train_env
ln -sf "$ENV_NAME" cryptovision_py39_env
ln -sf "$ENV_NAME" cryptovision_prod_env

echo "✅ Unified environment setup complete!"
echo "🔍 The environment is now active and can be used for all operations."
echo "📋 To activate this environment in the future, run:"
echo "  source $ENV_NAME/bin/activate"
echo 
echo "🚀 You can now use the following commands to run the application:"
echo "  ./start_dashboard.sh         # Start the dashboard"
echo "  ./run_ensemble_model.py      # Run the ensemble model"
echo "  ./prepare_for_production.py  # Prepare for production deployment"