#!/bin/bash
# Improved CryptoVisionAI Dashboard Startup Script

# Set default values
CONFIG_PATH="config/config.yaml"
MODE="test"
PORT=8050
HOST="0.0.0.0"
DEBUG=false
INSTALL_DEPS=false
PY39_ENV="cryptovision_py39_env"
FORCE_REINSTALL=false

# Colors for better output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Display help information
show_help() {
  echo -e "${GREEN}CryptoVisionAI Dashboard Startup${NC}"
  echo ""
  echo "Usage: ./start_dashboard_improved.sh [OPTIONS]"
  echo ""
  echo "Options:"
  echo "  -c, --config PATH     Set path to config file (default: config/config.yaml)"
  echo "  -m, --mode MODE       Set trading mode (test, paper, live) (default: test)"
  echo "  -p, --port PORT       Set dashboard port (default: 8050)"
  echo "  -h, --host HOST       Set dashboard host (default: 0.0.0.0)"
  echo "  -d, --debug           Enable debug mode"
  echo "  -s, --symbol SYMBOL   Set trading symbol (default: BTCUSDT)"
  echo "  -n, --no-trading      Start dashboard without trading system"
  echo "  -i, --install-deps    Install dependencies before starting"
  echo "  -f, --force-reinstall Reinstall all dependencies (useful for resolving issues)"
  echo "  --help                Display this help message"
  echo ""
  exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    -m|--mode)
      MODE="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    -h|--host)
      HOST="$2"
      shift 2
      ;;
    -d|--debug)
      DEBUG=true
      shift
      ;;
    -s|--symbol)
      SYMBOL="$2"
      shift 2
      ;;
    -n|--no-trading)
      NO_TRADING=true
      shift
      ;;
    -i|--install-deps)
      INSTALL_DEPS=true
      shift
      ;;
    -f|--force-reinstall)
      FORCE_REINSTALL=true
      INSTALL_DEPS=true
      shift
      ;;
    --help)
      show_help
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      show_help
      ;;
  esac
done

echo -e "${GREEN}Starting CryptoVisionAI Dashboard...${NC}"
echo "Configuration: $CONFIG_PATH"
echo "Mode: $MODE"
echo "Host: $HOST"
echo "Port: $PORT"

# Check for Python 3.9 environment
if [ -d "$PY39_ENV" ]; then
  echo -e "${GREEN}Found Python 3.9 environment: $PY39_ENV${NC}"
else
  echo -e "${RED}Python 3.9 environment not found at $PY39_ENV${NC}"
  echo -e "${YELLOW}Checking for other Python environments...${NC}"
  
  # Check for alternate environments
  for env in "cryptovision_prod_env" "cryptovision_train_env" "venv" ".venv"; do
    if [ -d "$env" ]; then
      echo -e "${GREEN}Found alternate environment: $env${NC}"
      PY39_ENV="$env"
      break
    fi
  done
  
  if [ "$PY39_ENV" == "cryptovision_py39_env" ]; then
    echo -e "${RED}No Python environment found! Please create one using:${NC}"
    echo -e "python3.9 -m venv cryptovision_py39_env"
    echo -e "Or run the training script first: ./train_with_python39.sh"
    exit 1
  fi
fi

# Activate the Python 3.9 environment
echo -e "${GREEN}Activating Python environment: $PY39_ENV${NC}"
source cryptovision_py39_env/bin/activate"

# Verify Python version in the virtual environment
PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo -e "${GREEN}Using Python $PYTHON_VERSION${NC}"

# Install dependencies if requested
if [ "$INSTALL_DEPS" = true ]; then
  echo -e "${GREEN}Installing/updating dependencies...${NC}"
  pip install --upgrade pip
  
  if [ "$FORCE_REINSTALL" = true ]; then
    pip install --force-reinstall -r requirements.txt
  else
    pip install -r requirements.txt
  fi
fi

# Create assets directory if it doesn't exist
ASSETS_DIR="src/visualization/assets"
mkdir -p "$ASSETS_DIR"

# Check for logo file and create a default one if missing
LOGO_PATH="$ASSETS_DIR/logo.png"
if [ ! -f "$LOGO_PATH" ]; then
  echo -e "${YELLOW}Logo file not found at $LOGO_PATH${NC}"
  echo -e "${GREEN}Creating a simple text-based logo...${NC}"
  
  # Use Python to create a simple text-based logo
  python -c "
import matplotlib.pyplot as plt
import os

plt.figure(figsize=(6, 2))
plt.text(0.5, 0.5, 'CryptoVisionAI', fontsize=28, fontweight='bold', 
         ha='center', va='center', color='#2c3e50')
plt.axis('off')
plt.savefig('$LOGO_PATH', bbox_inches='tight', pad_inches=0.1, dpi=150)
plt.close()
print('Created logo at $LOGO_PATH')
  "
fi

# Build command
CMD="python start_dashboard.py --config $CONFIG_PATH --mode $MODE --host $HOST --port $PORT"

# Add optional arguments
if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug"
  echo -e "${YELLOW}Debug mode: enabled${NC}"
else 
  echo "Debug mode: disabled"
fi

if [ ! -z "$SYMBOL" ]; then
  CMD="$CMD --symbol $SYMBOL"
  echo "Trading symbol: $SYMBOL"
fi

if [ "$NO_TRADING" = true ]; then
  CMD="$CMD --no-trading"
  echo "Trading system: disabled"
else
  echo "Trading system: enabled"
fi

echo ""
echo -e "${GREEN}Executing: $CMD${NC}"
echo ""

# Execute the command
eval $CMD