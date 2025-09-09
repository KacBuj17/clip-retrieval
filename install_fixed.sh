#!/bin/bash

# Create virtual environment
VENV_DIR=".venv"
python3 -m venv "$VENV_DIR"

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Build project and install required Python dependencies
make
pip uninstall -y albumentations
pip install albumentations==1.3.0
pip install "setuptools<80"

# Check if npm is installed, install it if missing.
if ! command -v npm &> /dev/null; then
    echo "npm not found - installing..."
    if command -v apt-get &> /dev/null; then
        apt-get update && \
        apt-get install -y npm && \
        rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
    else
        echo "Unknown package manager – please install Node.js manually."
        exit 1
    fi
else
    echo "npm installed - skipping"
fi

# Clean npm cache
echo "Cleaning npm cache..."
npm cache clean --force || true

# Remove leftover npm logs
rm -rf ~/.npm/_logs


# Install frontend dependencies and build frontend if npm is available
if command -v npm &> /dev/null; then
    echo "npm installed - installing and building frontend"
    cd front
    npm install
    npm run build
    cd ..
else
    echo "npm not found - skipping frontend build"
fi
