#!/bin/bash
# QENEX LAB TUI Run Script

echo "=========================================="
echo "QENEX LAB OMNI-AWARE TUI v1.4.0-INFINITY"
echo "Cyberpunk Terminal Interface"
echo "=========================================="
echo ""

# Check if backend is running
if ! curl -s http://localhost:8765/health > /dev/null 2>&1; then
    echo "⚠️  WARNING: Backend not detected on port 8765"
    echo ""
    echo "Please start the backend first:"
    echo "  cd /opt/qenex_lab/interface/backend"
    echo "  ./start_omni.sh"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "🚀 Launching TUI..."
echo ""

# Run in release mode if built, otherwise dev mode
if [ -f "./target/release/qenex-tui" ]; then
    ./target/release/qenex-tui
else
    cargo run --release
fi
