#!/bin/bash
# Script to properly install TA-Lib on macOS systems
# For CryptoVisionAI project

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
echo -e "${BLUE}║       ${CYAN}CryptoVisionAI TA-Lib Installer${BLUE}         ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Error: Homebrew is not installed.${NC}"
    echo -e "${YELLOW}Please install Homebrew first:${NC}"
    echo -e "${CYAN}  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"${NC}"
    exit 1
fi

# Check if TA-Lib is installed via Homebrew
echo -e "${CYAN}→${NC} Ensuring TA-Lib C library is installed via Homebrew..."
brew install ta-lib || brew reinstall ta-lib

# Activate the unified environment
ENV_NAME="cryptovision_unified_env"
if [ ! -d "$ENV_NAME" ]; then
    echo -e "${RED}Error: Unified environment not found.${NC}"
    echo -e "${YELLOW}Please run ./unify_environments.sh first${NC}"
    exit 1
else
    echo -e "${CYAN}→${NC} Activating unified environment..."
    source "$ENV_NAME/bin/activate"
fi

# Get TA-Lib paths for better configuration
TALIB_BREW_PREFIX=$(brew --prefix ta-lib)
echo -e "${CYAN}→${NC} TA-Lib Homebrew prefix: ${TALIB_BREW_PREFIX}"

# Download and build from source with proper flags
echo -e "${CYAN}→${NC} Setting up build environment with correct paths..."
export CFLAGS="-I${TALIB_BREW_PREFIX}/include"
export LDFLAGS="-L${TALIB_BREW_PREFIX}/lib"
export TA_INCLUDE_PATH="${TALIB_BREW_PREFIX}/include"
export TA_LIBRARY_PATH="${TALIB_BREW_PREFIX}/lib"
export ARCHFLAGS="-arch $(uname -m)"

# Install wheel first to ensure proper build tools
echo -e "${CYAN}→${NC} Upgrading pip and wheel..."
pip install --upgrade pip wheel setuptools

# Clone TA-Lib from GitHub for source installation
echo -e "${CYAN}→${NC} Downloading TA-Lib Python wrapper source..."
TMP_DIR=$(mktemp -d)
cd $TMP_DIR
git clone https://github.com/mrjbq7/ta-lib.git
cd ta-lib

# Build and install from source
echo -e "${CYAN}→${NC} Building and installing TA-Lib Python wrapper..."
pip install -e .

cd $SCRIPT_DIR

# Verify the installation
echo -e "${CYAN}→${NC} Verifying installation..."
if python -c "import talib; print(f'TA-Lib installed successfully. Version: {talib.__version__}')" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} TA-Lib Python wrapper installed successfully!"
    # Update start_unified_dashboard.sh
    echo -e "${CYAN}→${NC} Updating dashboard script to use TA-Lib..."
    sed -i.bak 's/export SKIP_TALIB=1/# TA-Lib is now properly installed/' start_unified_dashboard.sh
    rm -f start_unified_dashboard.sh.bak
    # Update dependency manager to reflect successful installation
    echo -e "${GREEN}✓${NC} TA-Lib installed successfully!"
else
    echo -e "${RED}Error: TA-Lib Python wrapper installation failed.${NC}"
    echo -e "${YELLOW}Trying alternative installation method...${NC}"
    
    # Alternative method using direct wheel link for Apple Silicon
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${YELLOW}Detected Apple Silicon (arm64). Trying prebuilt wheel...${NC}"
        pip uninstall -y ta-lib || true
        pip install --no-cache-dir --no-binary :all: --force-reinstall ta-lib
    else
        # For Intel Macs
        echo -e "${YELLOW}Detected Intel Mac. Trying with different compile flags...${NC}"
        pip uninstall -y ta-lib || true
        export CFLAGS="-I${TALIB_BREW_PREFIX}/include -march=native"
        export LDFLAGS="-L${TALIB_BREW_PREFIX}/lib"
        pip install --no-cache-dir --no-binary :all: --force-reinstall ta-lib
    fi
    
    # Check if alternative method worked
    if python -c "import talib" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} TA-Lib Python wrapper installed successfully with alternative method!"
        # Update start_unified_dashboard.sh
        sed -i.bak 's/export SKIP_TALIB=1/# TA-Lib is now properly installed/' start_unified_dashboard.sh
        rm -f start_unified_dashboard.sh.bak
    else
        echo -e "${RED}All installation methods failed. Setting up temporary workaround...${NC}"
        echo -e "${YELLOW}The dashboard will work but with limited technical analysis features.${NC}"
        # Keep the SKIP_TALIB flag for now
    fi
fi

echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║ ${GREEN}Installation process completed!${BLUE}                ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo -e "${YELLOW}You can now run ./start_unified_dashboard.sh${NC}"