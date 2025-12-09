#!/bin/bash
# Script to clean up Python environments and ensure Python 3.9 is the only one used
# for the CryptoVisionAI project

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

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         ${CYAN}CryptoVisionAI Python Environment Cleanup${BLUE}         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

# Check if Python 3.9 environment exists
PY39_ENV_NAME="cryptovision_py39_env"
if [ ! -d "$PY39_ENV_NAME" ]; then
    echo -e "${RED}✗${NC} Python 3.9 environment not found!"
    echo -e "${YELLOW}Please run train_with_python39.sh first to create the Python 3.9 environment.${NC}"
    exit 1
fi

# Verify Python 3.9 is installed in the environment
echo -e "${CYAN}→${NC} Verifying Python 3.9 environment..."
if source "$PY39_ENV_NAME/bin/activate" && python --version | grep -q "Python 3.9"; then
    PYTHON_VERSION=$(python --version)
    echo -e "${GREEN}✓${NC} Found $PYTHON_VERSION in $PY39_ENV_NAME"
else
    echo -e "${RED}✗${NC} The environment does not contain Python 3.9!"
    echo -e "${YELLOW}Please run train_with_python39.sh to create a proper Python 3.9 environment.${NC}"
    exit 1
fi

# Function to safely remove an environment if it exists
remove_env() {
    local env_name="$1"
    if [ "$env_name" != "$PY39_ENV_NAME" ] && [ -d "$env_name" ]; then
        echo -e "${CYAN}→${NC} Removing environment: $env_name"
        rm -rf "$env_name"
        echo -e "${GREEN}✓${NC} Removed $env_name successfully"
    fi
}

# Remove other virtual environments
echo -e "\n${BLUE}=== Cleaning up other Python environments ===${NC}"

# List of project's virtual environments to check
ENVS_TO_CHECK=(
    "cryptovision_unified_env"
    "cryptovision_train_env"
)

# Preserve production environment if it exists
if [ -d "cryptovision_prod_env" ]; then
    echo -e "${YELLOW}⚠${NC} Production environment detected. Checking its Python version..."
    if source "cryptovision_prod_env/bin/activate" && python --version | grep -q "Python 3.9"; then
        echo -e "${GREEN}✓${NC} Production environment uses Python 3.9 - preserving it"
    else
        ENVS_TO_CHECK+=("cryptovision_prod_env")
    fi
fi

# Remove each environment
for env in "${ENVS_TO_CHECK[@]}"; do
    remove_env "$env"
done

# Update all scripts to use only Python 3.9 environment
echo -e "\n${BLUE}=== Updating scripts to use Python 3.9 ===${NC}"

# Update start_unified_dashboard.sh
echo -e "${CYAN}→${NC} Updating start_unified_dashboard.sh to use Python 3.9 only..."
sed -i.bak 's/ENV_NAME="cryptovision_unified_env"/ENV_NAME="cryptovision_py39_env"/g' start_unified_dashboard.sh
echo -e "${GREEN}✓${NC} Updated start_unified_dashboard.sh"

# Update start_dashboard.sh
if [ -f "start_dashboard.sh" ]; then
    echo -e "${CYAN}→${NC} Updating start_dashboard.sh to use Python 3.9 only..."
    sed -i.bak 's/source .*\/bin\/activate/source cryptovision_py39_env\/bin\/activate/g' start_dashboard.sh
    echo -e "${GREEN}✓${NC} Updated start_dashboard.sh"
fi

# Update start_dashboard_improved.sh
if [ -f "start_dashboard_improved.sh" ]; then
    echo -e "${CYAN}→${NC} Updating start_dashboard_improved.sh to use Python 3.9 only..."
    sed -i.bak 's/source .*\/bin\/activate/source cryptovision_py39_env\/bin\/activate/g' start_dashboard_improved.sh
    echo -e "${GREEN}✓${NC} Updated start_dashboard_improved.sh"
fi

# Update setup_and_run.sh
if [ -f "setup_and_run.sh" ]; then
    echo -e "${CYAN}→${NC} Updating setup_and_run.sh to use Python 3.9 only..."
    sed -i.bak 's/source .*\/bin\/activate/source cryptovision_py39_env\/bin\/activate/g' setup_and_run.sh
    echo -e "${GREEN}✓${NC} Updated setup_and_run.sh"
fi

# Remove backup files
rm -f *.bak

# Create a symbolic link for unified environment if any scripts depend on it
echo -e "${CYAN}→${NC} Creating compatibility symbolic link..."
if [ ! -L "cryptovision_unified_env" ]; then
    ln -s "$PY39_ENV_NAME" "cryptovision_unified_env"
    echo -e "${GREEN}✓${NC} Created compatibility symbolic link: cryptovision_unified_env -> $PY39_ENV_NAME"
fi

echo -e "\n${BLUE}=== Environment Cleanup Summary ===${NC}"
echo -e "${GREEN}✓${NC} Python 3.9 environment ($PY39_ENV_NAME) is set as the default"
echo -e "${GREEN}✓${NC} Removed other Python environments"
echo -e "${GREEN}✓${NC} Updated scripts to use Python 3.9"
echo -e "${GREEN}✓${NC} Created compatibility links for backward compatibility"

echo -e "\n${CYAN}Your CryptoVisionAI project is now configured to use only Python 3.9.${NC}"
echo -e "${YELLOW}Note: This script only affected project-specific Python environments.${NC}"
echo -e "${YELLOW}System-wide Python installations were not modified.${NC}"

# Activate Python 3.9 environment for the current session
source "$PY39_ENV_NAME/bin/activate"
echo -e "\n${GREEN}✓${NC} Activated Python 3.9 environment for current session"
echo -e "${BLUE}Current Python: $(python --version)${NC}"