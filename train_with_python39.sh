#!/bin/bash
# Script to set up Python 3.9 environment and run the ensemble model training

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}CryptoVisionAI Model Training Setup${NC}"
echo "========================================"

# Check if Python 3.9 is installed
if command -v python3.9 &> /dev/null; then
    echo -e "${GREEN}✓ Python 3.9 is installed${NC}"
    PYTHON_CMD="python3.9"
elif command -v python3 &> /dev/null && python3 --version | grep -q "Python 3.9"; then
    echo -e "${GREEN}✓ Python 3.9 is installed as python3${NC}"
    PYTHON_CMD="python3"
else
    echo -e "${RED}✗ Python 3.9 is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install Python 3.9 before continuing.${NC}"
    echo -e "On macOS, you can use: brew install python@3.9"
    echo -e "On Ubuntu/Debian: sudo apt install python3.9 python3.9-venv python3.9-dev"
    exit 1
fi

# Environment name
ENV_NAME="cryptovision_py39_env"

# Check if environment already exists
if [ -d "$ENV_NAME" ]; then
    echo -e "${YELLOW}Virtual environment $ENV_NAME already exists.${NC}"
    read -p "Do you want to use the existing environment? (y/N): " USE_EXISTING
    if [[ $USE_EXISTING != "y" && $USE_EXISTING != "Y" ]]; then
        read -p "Do you want to recreate it? This will delete the existing environment. (y/N): " RECREATE
        if [[ $RECREATE == "y" || $RECREATE == "Y" ]]; then
            echo "Removing existing environment..."
            rm -rf "$ENV_NAME"
            echo -e "${GREEN}Creating new virtual environment with Python 3.9...${NC}"
            $PYTHON_CMD -m venv "$ENV_NAME"
        else
            echo -e "${YELLOW}Using existing environment without recreation.${NC}"
        fi
    else
        echo -e "${GREEN}Using existing environment.${NC}"
    fi
else
    echo -e "${GREEN}Creating virtual environment with Python 3.9...${NC}"
    $PYTHON_CMD -m venv "$ENV_NAME"
fi

# Activate the environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source "$ENV_NAME/bin/activate"

# Verify Python version in the virtual environment
VENV_PYTHON_VERSION=$(python --version)
echo -e "${GREEN}Virtual environment Python version: $VENV_PYTHON_VERSION${NC}"

# Install requirements
echo -e "${GREEN}Installing required packages...${NC}"
pip install --upgrade pip
pip install -r requirements.txt

# Optional: Check for TensorFlow compatibility
if pip list | grep -q tensorflow; then
    echo -e "${GREEN}✓ TensorFlow is installed${NC}"
else
    echo -e "${YELLOW}⚠️ Warning: TensorFlow is not installed properly${NC}"
    echo -e "Attempting to install a compatible version..."
    pip install "tensorflow<2.11"
fi

# Optional: Check for other critical packages
for pkg in "pandas" "numpy" "scikit-learn" "xgboost" "lightgbm"; do
    if pip list | grep -q $pkg; then
        echo -e "${GREEN}✓ $pkg is installed${NC}"
    else
        echo -e "${RED}✗ $pkg is missing${NC}"
        echo -e "Attempting to install $pkg..."
        pip install $pkg
    fi
done

# Run the training script
echo -e "\n${GREEN}Starting model training...${NC}"
echo -e "Running ensemble model with 136 features"
echo -e "${YELLOW}This may take a while...${NC}\n"

# Run the training script
python run_ensemble_model.py

# Completion message
echo -e "\n${GREEN}Training process completed!${NC}"
echo -e "If successful, the model has been saved to models/ensemble/"
echo -e "You can now run your dashboard with the newly trained model."