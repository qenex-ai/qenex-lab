"""
QENEX Test Configuration

This conftest.py sets up the Python path so all tests can import
the QENEX packages correctly.
"""
import sys
import os

# Get the workspace root (parent of tests directory)
WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add package directories to path
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex_chem', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex-core', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex-qlang', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex-physics', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex-math', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE_ROOT, 'packages', 'qenex-bio', 'src'))
sys.path.insert(0, WORKSPACE_ROOT)

# Set numpy to raise errors on invalid operations
import numpy as np
# np.seterr(all='raise')  # Can uncomment for strict mode
