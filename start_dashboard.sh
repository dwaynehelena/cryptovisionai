#!/bin/bash
# CryptoVisionAI Dashboard Startup Script

# Set default values
CONFIG_PATH="config/config.yaml"
MODE="test"
PORT=8050
HOST="0.0.0.0"
DEBUG=false
INSTALL_DEPS=false
PYTHON_CMD="python3.9"
FORCE_REINSTALL=false

# Display help information
show_help() {
  echo "CryptoVisionAI Dashboard Startup"
  echo ""
  echo "Usage: ./start_dashboard.sh [OPTIONS]"
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
      echo "Unknown option: $1"
      show_help
      ;;
  esac
done

echo "Starting CryptoVisionAI Dashboard..."
echo "Configuration: $CONFIG_PATH"
echo "Mode: $MODE"
echo "Host: $HOST"
echo "Port: $PORT"

# Check if Python 3.9 is installed
if ! command -v $PYTHON_CMD &> /dev/null; then
    # Try to find a Python 3.x if Python 3.9 is not available
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        if [[ "${PYTHON_VERSION:0:1}" == "3" ]]; then
            echo "Python 3.9 not found, but Python $PYTHON_VERSION is available."
            PYTHON_CMD="python3"
        fi
    else
        echo "Error: Python 3.x is not installed or not in PATH"
        echo "Please install Python 3.9 using:"
        echo "  - macOS: brew install python@3.9"
        echo "  - Linux: Use your package manager (apt, yum, etc.)"
        echo "  - Windows: Download from python.org"
        exit 1
    fi
fi

# Verify Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Using Python $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.9" ]]; then
    echo "Warning: Found Python $PYTHON_VERSION instead of Python 3.9"
    echo "This may cause compatibility issues"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Exiting. Please install Python 3.9"
        exit 1
    fi
fi

# Set up virtual environment
setup_venv() {
    if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
        echo "Creating virtual environment with Python $PYTHON_VERSION..."
        $PYTHON_CMD -m venv venv
        if [ $? -ne 0 ]; then
            echo "Failed to create virtual environment. Please install venv package:"
            echo "$PYTHON_CMD -m pip install virtualenv"
            exit 1
        fi
    fi

    # Activate virtual environment
    echo "Activating virtual environment..."
    if [ -d "venv" ]; then
        source cryptovision_py39_env/bin/activate
    else
        source cryptovision_py39_env/bin/activate
    fi

    # Check if activation was successful
    if [ $? -ne 0 ]; then
        echo "Failed to activate virtual environment."
        exit 1
    fi
    
    # Verify Python version in virtual environment
    VENV_PYTHON_VERSION=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    echo "Virtual environment using Python $VENV_PYTHON_VERSION"
}

# Check if a Python package is installed
check_package() {
    python -c "import $1" &> /dev/null
    return $?
}

# Test all required packages
check_dependencies() {
    echo "Checking required dependencies..."
    
    MISSING_DEPS=()
    
    # List of critical packages to check
    PACKAGES=(
        "dash:dash"
        "dash_bootstrap_components:dash-bootstrap-components"
        "binance:python-binance"
        "yaml:pyyaml"
        "sklearn:scikit-learn"
        "talib:ta-lib"
        "pywt:pywavelets"
        "pandas:pandas"
        "numpy:numpy"
        "tensorflow:tensorflow"
        "torch:torch"
        "matplotlib:matplotlib"
        "plotly:plotly"
        "flask:flask"
    )
    
    for pkg in "${PACKAGES[@]}"; do
        IFS=":" read -r import_name package_name <<< "$pkg"
        if ! check_package $import_name; then
            MISSING_DEPS+=("$package_name")
        fi
    done
    
    if [ ${#MISSING_DEPS[@]} -gt 0 ]; then
        echo "Missing required packages: ${MISSING_DEPS[*]}"
        return 1
    else
        echo "All critical dependencies are installed."
        return 0
    fi
}

# Install dependencies
install_dependencies() {
    echo "Installing dependencies..."
    if [ -f "requirements.txt" ]; then
        python -m pip install --upgrade pip
        
        # Force reinstall if requested
        if [ "$FORCE_REINSTALL" = true ]; then
            echo "Force reinstalling all packages..."
            pip install --force-reinstall -r requirements.txt
            if [ $? -ne 0 ]; then
                echo "Failed to reinstall dependencies."
                exit 1
            fi
            return 0
        fi
        
        # Check if ta-lib is installed
        if ! python -c "import talib" &> /dev/null; then
            echo "TA-Lib not found. Installing TA-Lib dependencies..."
            
            # Detect OS and install TA-Lib accordingly
            if [[ "$OSTYPE" == "darwin"* ]]; then
                # macOS
                echo "Installing TA-Lib on macOS using Homebrew..."
                
                # Check if Homebrew is installed
                if ! command -v brew &> /dev/null; then
                    echo "Homebrew not found. Please install Homebrew first:"
                    echo '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
                    echo "Then run this script again."
                    exit 1
                fi
                
                # Install TA-Lib
                brew install ta-lib || echo "TA-Lib may already be installed, trying to continue..."
                
            elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
                # Linux
                echo "Installing TA-Lib on Linux..."
                echo "You may need to enter your password for sudo access."
                
                # Try different package managers
                if command -v apt-get &> /dev/null; then
                    sudo apt-get update
                    sudo apt-get install -y build-essential
                    sudo apt-get install -y ta-lib
                elif command -v yum &> /dev/null; then
                    sudo yum install -y gcc gcc-c++ make
                    sudo yum install -y ta-lib ta-lib-devel
                else
                    echo "Unsupported Linux distribution. Please install TA-Lib manually."
                    echo "See: https://github.com/mrjbq7/ta-lib#dependencies"
                    exit 1
                fi
                
            else
                echo "Unsupported OS. Please install TA-Lib manually:"
                echo "- macOS: brew install ta-lib"
                echo "- Linux: sudo apt-get install ta-lib"
                echo "- Windows: Follow instructions at https://github.com/mrjbq7/ta-lib#windows"
                echo "Then try again."
                exit 1
            fi
            
            # Install Python wrapper
            echo "Installing Python TA-Lib wrapper..."
            pip install TA-Lib
            
            if [ $? -ne 0 ]; then
                echo "Failed to install TA-Lib. Please install manually following the instructions at:"
                echo "https://github.com/mrjbq7/ta-lib#installation"
                exit 1
            fi
        fi
        
        # Check if PyWavelets is installed
        if ! python -c "import pywt" &> /dev/null; then
            echo "PyWavelets not found. Installing..."
            pip install pywavelets
            
            if [ $? -ne 0 ]; then
                echo "Failed to install PyWavelets."
                exit 1
            fi
        fi
        
        # Install other dependencies
        pip install -r requirements.txt
        
        if [ $? -ne 0 ]; then
            echo "Failed to install dependencies."
            exit 1
        fi
        
        # Check for dash-bootstrap-components specifically (common issue)
        if ! python -c "import dash_bootstrap_components" &> /dev/null; then
            echo "Installing dash-bootstrap-components separately..."
            pip install dash-bootstrap-components
        fi
        
        echo "Dependencies installed successfully."
    else
        echo "Warning: requirements.txt not found. Skipping dependency installation."
    fi
}

# Create necessary folders if they don't exist
create_directories() {
    echo "Setting up project directories..."
    mkdir -p data logs models
    
    # Create assets folder for the dashboard
    mkdir -p src/visualization/assets
}

# Setup virtual environment
setup_venv

# Install dependencies if requested or if it's the first run
if [ "$INSTALL_DEPS" = true ] || [ ! -f "venv/installed.flag" ]; then
    install_dependencies
    # Create flag file to indicate dependencies are installed
    touch venv/installed.flag
fi

# Check dependencies
check_dependencies
if [ $? -ne 0 ] && [ "$INSTALL_DEPS" != true ]; then
    echo "Please install missing packages: pip install -r requirements.txt"
    echo "Or run with -i flag to install dependencies: ./start_dashboard.sh -i"
    echo "Missing dependencies may cause issues"
fi

# Ensure necessary directories exist
create_directories

# Check for logo file
LOGO_PATH="src/visualization/assets/logo.png"
if [ ! -f "$LOGO_PATH" ]; then
    echo "Warning: Logo file not found at $(pwd)/$LOGO_PATH"
    echo "Dashboard will run without a logo"
fi

# Build command
CMD="python start_dashboard.py --config $CONFIG_PATH --mode $MODE --host $HOST --port $PORT"

# Add optional arguments
if [ "$DEBUG" = true ]; then
  CMD="$CMD --debug"
  echo "Debug mode: enabled"
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
echo "Executing: $CMD"
echo ""

# Execute the command
eval $CMD