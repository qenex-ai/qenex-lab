#!/bin/bash
# QENEX LAB OMNI_INTEGRATION v1.4.0-INFINITY Startup Script

echo "========================================="
echo "QENEX LAB v1.4.0-INFINITY"
echo "OMNI-AWARE Scientific Intelligence System"
echo "========================================="
echo ""

# Activate virtual environment
source venv/bin/activate

# Start FastAPI server
echo "Starting OMNI-AWARE backend on http://localhost:8765"
echo "Press Ctrl+C to stop"
echo ""

python main.py
