#!/bin/bash
# Install Rust toolchain for QENEX LAB TUI

echo "=========================================="
echo "QENEX LAB TUI - Rust Installation"
echo "=========================================="
echo ""

# Check if already installed
if command -v cargo &> /dev/null; then
    echo "✓ Rust is already installed!"
    cargo --version
    rustc --version
    echo ""
    echo "Run: ./build.sh to compile the TUI"
    exit 0
fi

echo "📥 Installing Rust toolchain..."
echo ""

# Install Rust using rustup (non-interactive)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# Source cargo environment
source "$HOME/.cargo/env"

echo ""
echo "=========================================="
echo "✅ Rust installed successfully!"
echo ""
cargo --version
rustc --version
echo ""
echo "Next steps:"
echo "  1. Reload shell: source ~/.cargo/env"
echo "  2. Build TUI: ./build.sh"
echo "  3. Run TUI: ./run.sh"
echo "=========================================="
