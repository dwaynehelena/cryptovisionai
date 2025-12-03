#!/bin/bash
# Unified Dashboard Starter for CryptoVisionAI
# This script starts the dashboard with all improvements enabled

set -e
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for terminal output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ${CYAN}CryptoVisionAI Dashboard Starter${BLUE}         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"

# Explicitly activate the Python 3.9 environment first
PY39_ENV_NAME="cryptovision_py39_env"
if [ -d "$PY39_ENV_NAME" ]; then
    echo -e "${GREEN}✓${NC} Found Python 3.9 environment"
    echo -e "${CYAN}→${NC} Activating Python 3.9 environment for model compatibility..."
    source "$PY39_ENV_NAME/bin/activate"
    
    # Verify Python version
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Using $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3.9 environment not found!"
    echo -e "${YELLOW}Models may not load correctly. Please run train_with_python39.sh first.${NC}"
    
    # Check for unified environment as fallback
    ENV_NAME="cryptovision_unified_env"
    if [ -d "$ENV_NAME" ]; then
        echo -e "${GREEN}✓${NC} Found unified environment (using as fallback)"
        echo -e "${CYAN}→${NC} Activating unified environment..."
        source "$ENV_NAME/bin/activate"
    else
        echo -e "${RED}✗${NC} No suitable environment found!"
        echo -e "${YELLOW}Please run train_with_python39.sh or unify_environments.sh first.${NC}"
        exit 1
    fi
fi

# Generate a custom logo if it doesn't exist
LOGO_PATH="src/visualization/assets/logo.svg"
if [ ! -f "$LOGO_PATH" ]; then
    echo -e "${CYAN}→${NC} Generating custom logo..."
    python src/utils/generate_logo.py --color blue
    
    # Try to convert to PNG if possible
    python -c "import cairosvg" &> /dev/null
    if [ $? -eq 0 ]; then
        echo -e "${CYAN}→${NC} Converting logo to PNG format..."
        python src/utils/generate_logo.py --color blue --png
    else
        echo -e "${YELLOW}⚠ cairosvg not found. Install with: pip install cairosvg${NC}"
    fi
else
    echo -e "${GREEN}✓${NC} Custom logo already exists"
fi

# Verify TA-Lib installation
echo -e "${CYAN}→${NC} Checking TA-Lib installation..."
if python -c "import talib; print(f'TA-Lib {talib.__version__} is installed and ready for technical analysis')" 2>/dev/null; then
    TALIB_VERSION=$(python -c "import talib; print(talib.__version__)" 2>/dev/null)
    echo -e "${GREEN}✓${NC} Using TA-Lib ${TALIB_VERSION} for technical indicators"
else
    echo -e "${YELLOW}⚠ TA-Lib not found. Attempting to install...${NC}"
    # Try to install the version actually available (0.6.3 instead of 0.6.4)
    pip install ta-lib==0.6.3
    
    # Check if installation was successful
    if python -c "import talib" 2>/dev/null; then
        TALIB_VERSION=$(python -c "import talib; print(talib.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} Successfully installed TA-Lib ${TALIB_VERSION}"
    else
        echo -e "${YELLOW}⚠ TA-Lib not found. Technical indicators may be limited.${NC}"
        echo -e "${YELLOW}⚠ Run ./install_talib_macos.sh to install properly on macOS${NC}"
        # Set environment variable to skip TA-Lib dependency check
        export SKIP_TALIB=1
    fi
fi

# Verify LightGBM installation
echo -e "${CYAN}→${NC} Checking LightGBM installation..."
if python -c "import lightgbm; print(f'LightGBM {lightgbm.__version__} is installed and ready for gradient boosting')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Using LightGBM for machine learning models"
else
    echo -e "${YELLOW}⚠ LightGBM not found. Attempting to install...${NC}"
    
    # Try the latest version first, which is more likely to have pre-built wheels
    echo -e "${CYAN}→${NC} Installing LightGBM latest version..."
    pip install --prefer-binary lightgbm
    
    # If that fails, try specific versions that are known to be available
    if ! python -c "import lightgbm" 2>/dev/null; then
        echo -e "${CYAN}→${NC} Trying specific LightGBM version (4.6.0)..."
        pip install --prefer-binary lightgbm==4.6.0
    fi
    
    # Verify installation was successful - with better error handling
    python -c "import lightgbm" 2>/dev/null
    if [ $? -eq 0 ]; then
        LGB_VERSION=$(python -c "import lightgbm; print(lightgbm.__version__)" 2>/dev/null)
        echo -e "${GREEN}✓${NC} LightGBM version ${LGB_VERSION} successfully installed"
    else
        echo -e "${RED}✗${NC} LightGBM installation failed. Setting environment variable to continue without it.${NC}"
        echo -e "${YELLOW}  You can manually install later by running: pip install --prefer-binary lightgbm${NC}"
        # Set environment variable to proceed without LightGBM but rename it to avoid the specific warning message
        export LGBM_SKIP=1
    fi
fi

# Check dependencies
echo -e "${CYAN}→${NC} Checking dependencies..."
python -m src.utils.dependency_manager --install

# Start the dashboard
echo -e "${CYAN}→${NC} Starting CryptoVisionAI Dashboard with improvements..."
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ Dashboard:   ${GREEN}http://localhost:8050/${BLUE}               ║${NC}"
echo -e "${BLUE}║ Health API:  ${GREEN}http://localhost:8050/health${BLUE}         ║${NC}"
echo -e "${BLUE}║ Monitoring:  ${GREEN}http://localhost:8050/health/detailed${BLUE}║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"

# Start dashboard with configured parameters from config file
CONFIG_FILE="config/config.yaml"
if [ -f "$CONFIG_FILE" ]; then
    HOST=$(grep -A 10 "dashboard:" "$CONFIG_FILE" | grep "host:" | awk '{print $2}' | tr -d '"'"'" || echo "0.0.0.0")
    PORT=$(grep -A 10 "dashboard:" "$CONFIG_FILE" | grep "port:" | awk '{print $2}' | tr -d '"'"'" || echo "8050")
    DEBUG=$(grep -A 10 "dashboard:" "$CONFIG_FILE" | grep "debug:" | awk '{print $2}' | tr -d '"'"'" || echo "false")
else
    HOST="0.0.0.0"
    PORT="8050"
    DEBUG="false"
fi

# Launch the dashboard application
if [ "$DEBUG" = "true" ]; then
    python start_dashboard.py --host "$HOST" --port "$PORT" --debug
else
    python start_dashboard.py --host "$HOST" --port "$PORT"
fi