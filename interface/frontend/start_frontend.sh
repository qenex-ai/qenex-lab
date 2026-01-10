#!/bin/bash
# QENEX LAB Frontend Startup Script

cd /opt/qenex_lab/interface/frontend

echo "========================================="
echo "QENEX LAB Frontend"
echo "Starting dev server on port 5173..."
echo "========================================="
echo ""

/usr/bin/bun run dev
