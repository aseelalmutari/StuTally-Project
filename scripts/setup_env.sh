#!/bin/bash
# scripts/setup_env.sh
# Quick setup script for StuTally environment

set -e  # Exit on error

echo "=========================================="
echo "🚀 StuTally - Environment Setup"
echo "=========================================="
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "Project root: $PROJECT_ROOT"
echo ""

# Check if .env exists
if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠️  .env file already exists${NC}"
    read -p "Do you want to overwrite it? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Keeping existing .env file"
    else
        mv .env .env.backup.$(date +%Y%m%d_%H%M%S)
        echo -e "${GREEN}✅ Backed up existing .env${NC}"
    fi
fi

# Copy .env.example to .env
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        echo -e "${GREEN}✅ Created .env from .env.example${NC}"
    else
        echo -e "${RED}❌ .env.example not found${NC}"
        exit 1
    fi
fi

# Generate secrets
echo ""
echo "🔐 Generating secure secrets..."
SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
JWT_SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_hex(32))')

# Update .env with generated secrets
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
    sed -i '' "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET_KEY/" .env
else
    # Linux
    sed -i "s/SECRET_KEY=.*/SECRET_KEY=$SECRET_KEY/" .env
    sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET_KEY/" .env
fi

echo -e "${GREEN}✅ Secrets generated and added to .env${NC}"

# Create necessary directories
echo ""
echo "📁 Creating directories..."
mkdir -p uploads models data logs static/processed_images

# Create .gitkeep files
touch uploads/.gitkeep
touch models/.gitkeep
touch logs/.gitkeep
touch static/processed_images/.gitkeep

echo -e "${GREEN}✅ Directories created${NC}"

# Check Python version
echo ""
echo "🐍 Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $PYTHON_VERSION"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo ""
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate venv and install requirements
echo ""
echo "📦 Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installed${NC}"

# Create admin user
echo ""
echo "👤 Creating admin user..."
python3 create_user.py

echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review and update .env file with your settings"
echo "2. Add YOLO models to models/ directory"
echo "3. Add Roboflow API key to .env (ROBOFLOW_API_KEY)"
echo "4. Run the app: python app.py"
echo ""
echo "=========================================="

