#!/bin/bash
# QENEX LAB TUI Build Script

set -e

echo "=========================================="
echo "QENEX LAB TUI v1.4.0-INFINITY"
echo "Building high-performance Rust binary..."
echo "=========================================="
echo ""

# Check if Rust is installed
if ! command -v cargo &> /dev/null; then
    echo "❌ Rust/Cargo not found!"
    echo "Install with: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
    exit 1
fi

# Build release binary
echo "🔨 Compiling with maximum optimizations..."
cargo build --release

echo ""
echo "=========================================="
echo "✅ Build complete!"
echo "Binary: ./target/release/qenex-tui"
echo ""
echo "Run with: ./run.sh"
echo "Or: cargo run --release"
echo "=========================================="
