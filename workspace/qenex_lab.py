#!/usr/bin/env python3
"""
QENEX LAB - Terminal UI Launcher

Launch the QENEX LAB Terminal User Interface.

Usage:
    python qenex_lab.py
    
    Or with the virtual environment:
    source venv/bin/activate && python qenex_lab.py

Clipboard Support:
    - Ctrl+V / Shift+Insert: Paste from system clipboard
    - Ctrl+C: Copy to system clipboard
    - Ctrl+X: Cut to system clipboard
"""

import sys
import os

# Add package paths
WORKSPACE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WORKSPACE, 'packages', 'qenex-ui', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE, 'packages', 'qenex_chem', 'src'))
sys.path.insert(0, os.path.join(WORKSPACE, 'packages', 'qenex-qlang', 'src'))


def check_dependencies():
    """Check that required dependencies are installed."""
    missing = []
    
    try:
        import textual
    except ImportError:
        missing.append("textual")
    
    try:
        import pyperclip
    except ImportError:
        missing.append("pyperclip")
    
    try:
        import rich
    except ImportError:
        missing.append("rich")
    
    if missing:
        print("Missing dependencies:", ", ".join(missing))
        print("Install with: pip install " + " ".join(missing))
        sys.exit(1)


def main():
    """Launch the QENEX LAB TUI."""
    check_dependencies()
    
    # Import and run the app
    from app import QenexLabApp
    
    app = QenexLabApp()
    app.run()


if __name__ == "__main__":
    main()
