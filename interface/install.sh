#!/usr/bin/env bash
#
# QENEX LAB Chat Interface Installation Script
# Version: 3.0-INFINITY
#

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║       QENEX LAB Chat Interface Installation                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️  Please do not run this script as root"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.12+"
    exit 1
fi
echo "✓ Python 3 found: $(python3 --version)"

# Check Bun
if ! command -v bun &> /dev/null; then
    echo "❌ Bun not found. Please install Bun: https://bun.sh"
    exit 1
fi
echo "✓ Bun found: $(bun --version)"

# Install backend dependencies
echo ""
echo "Installing backend dependencies..."
cd /opt/qenex_lab/interface/backend

if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

source venv/bin/activate
pip install -r requirements.txt
deactivate

echo "✓ Backend dependencies installed"

# Install frontend dependencies
echo ""
echo "Installing frontend dependencies..."
cd /opt/qenex_lab/interface/frontend
bun install
echo "✓ Frontend dependencies installed"

# Fix permissions
echo ""
echo "Setting permissions..."
sudo chown -R $USER:$USER /opt/qenex_lab
echo "✓ Permissions set"

# Test installations
echo ""
echo "Testing installations..."

# Test backend
cd /opt/qenex_lab/interface/backend
source venv/bin/activate
python3 -c "import fastapi; print('  FastAPI:', fastapi.__version__)"
python3 -c "import uvicorn; print('  Uvicorn:', uvicorn.__version__)"
deactivate

# Test frontend
cd /opt/qenex_lab/interface/frontend
echo "  Solid.js: $(bun pm ls solid-js | grep solid-js | awk '{print $2}')"
echo "  Vite: $(bun pm ls vite | grep vite | awk '{print $2}')"

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                  Installation Complete!                          ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Launch the chat interface:"
echo "  qlab --chat"
echo ""
echo "Documentation:"
echo "  /opt/qenex_lab/interface/README.md"
echo ""
