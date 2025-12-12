#!/bin/bash

# CryptoVisionAI Autonomous Bot Launcher

# Ensure we are in the repo root
if [ ! -f "config/config.yaml" ]; then
    echo "❌ Error: Please run this script from the repository root directory."
    exit 1
fi

echo "🚀 Starting CryptoVisionAI Autonomous Bot (Multi-Coin Top 10)..."
echo "💰 Goal: Moonshot ($100 -> $10,000)"
echo "---------------------------------------------------"

# Check if environment exists
if [ -f "./cryptovision_py39_env/bin/python" ]; then
    PYTHON_EXEC="./cryptovision_py39_env/bin/python"
elif [ -f "./cryptovison_py39_env/bin/python" ]; then
    # Handle potential typo in env name seen earlier
    PYTHON_EXEC="./cryptovison_py39_env/bin/python"
else
    echo "⚠️  Virtual environment not found. Trying system python3..."
    PYTHON_EXEC="python3"
fi

# Run the bot
$PYTHON_EXEC run_autonomous_multicoin.py
