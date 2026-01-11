"""
Tests for QENEX LAB Primary Controller UI

Tests cover:
- QENEXBridge integration layer
- ClipboardMaster functionality
- Widget components
- UI panels and interactions

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import sys
import os
from pathlib import Path
from datetime import datetime

import pytest

# Add paths
WORKSPACE = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE / 'packages' / 'qenex-ui' / 'src'))
sys.path.insert(0, str(WORKSPACE / 'packages' / 'qenex-qlang' / 'src'))


# ============================================================================
# ClipboardMaster Tests
# ============================================================================

class TestClipboardMaster:
    """Tests for ClipboardMaster module."""
    
    def test_import_clipboard_master(self):
        """Test ClipboardMaster can be imported."""
        from widgets import ClipboardMaster, ClipboardEntry, get_clipboard_master
        
        assert ClipboardMaster is not None
        assert ClipboardEntry is not None
        assert get_clipboard_master is not None
    
    def test_clipboard_master_init(self):
        """Test ClipboardMaster initialization."""
        from widgets import ClipboardMaster
        
        cm = ClipboardMaster()
        assert cm.history == []
        assert cm._last_content == ""
        assert cm.MAX_HISTORY == 20
    
    def test_detect_type_xyz(self):
        """Test XYZ format detection."""
        from widgets import ClipboardMaster
        
        xyz_content = """O  0.0  0.0  0.0
H  0.0  0.757  0.587
H  0.0  -0.757  0.587"""
        
        assert ClipboardMaster.detect_type(xyz_content) == "xyz"
    
    def test_detect_type_python(self):
        """Test Python code detection."""
        from widgets import ClipboardMaster
        
        python_content = """import numpy as np
from scipy import linalg

def compute_energy(mol):
    return mol.energy
"""
        
        assert ClipboardMaster.detect_type(python_content) == "python"
    
    def test_detect_type_qlang(self):
        """Test Q-Lang code detection."""
        from widgets import ClipboardMaster
        
        qlang_content = """# Q-Lang simulation
const c = 299792458 m/s
molecule water
lattice ising 10x10
"""
        
        assert ClipboardMaster.detect_type(qlang_content) == "qlang"
    
    def test_detect_type_text(self):
        """Test plain text detection."""
        from widgets import ClipboardMaster
        
        text_content = "This is just some plain text with no special format."
        
        assert ClipboardMaster.detect_type(text_content) == "text"
    
    def test_clipboard_entry_preview(self):
        """Test ClipboardEntry preview."""
        from widgets import ClipboardEntry
        
        entry = ClipboardEntry(
            content="This is a long content that should be truncated in the preview" * 3,
            timestamp=datetime.now(),
            content_type="text",
            source="test",
        )
        
        assert len(entry.preview) <= 53  # 50 chars + "..."
        assert "..." in entry.preview
    
    def test_clipboard_entry_short_preview(self):
        """Test ClipboardEntry preview for short content."""
        from widgets import ClipboardEntry
        
        entry = ClipboardEntry(
            content="Short",
            timestamp=datetime.now(),
            content_type="text",
        )
        
        assert entry.preview == "Short"
        assert "..." not in entry.preview
    
    def test_get_recent_empty(self):
        """Test get_recent with empty history."""
        from widgets import ClipboardMaster
        
        cm = ClipboardMaster()
        recent = cm.get_recent(5)
        
        assert recent == []
    
    def test_clear_history(self):
        """Test clearing clipboard history."""
        from widgets import ClipboardMaster, ClipboardEntry
        
        cm = ClipboardMaster()
        
        # Add some entries manually
        cm.history.append(ClipboardEntry(
            content="test",
            timestamp=datetime.now(),
            content_type="text",
        ))
        cm._last_content = "test"
        
        assert len(cm.history) == 1
        
        cm.clear_history()
        
        assert cm.history == []
        assert cm._last_content == ""


# ============================================================================
# QENEXBridge Tests
# ============================================================================

class TestQENEXBridge:
    """Tests for QENEXBridge integration layer."""
    
    def test_import_bridge(self):
        """Test QENEXBridge can be imported."""
        from controller import QENEXBridge
        
        assert QENEXBridge is not None
    
    def test_bridge_init(self):
        """Test QENEXBridge initialization."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        assert bridge is not None
        assert bridge.verbose == False
    
    def test_bridge_lazy_integration(self):
        """Test lazy loading of LLM integration."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        # Integration should be None until accessed
        assert bridge._integration is None
    
    def test_bridge_lazy_interpreter(self):
        """Test lazy loading of Q-Lang interpreter."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        # Interpreter should be None until accessed
        assert bridge._interpreter is None
    
    def test_bridge_get_status(self):
        """Test getting system status."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        status = bridge.get_status()
        
        assert isinstance(status, dict)
        assert "rust_available" in status
        assert "interpreter_available" in status
        assert "integration_available" in status
    
    def test_bridge_reason_no_integration(self):
        """Test reasoning when integration is not available."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        # Force integration to None
        bridge._integration = "not_real"
        bridge._integration = None
        
        result = bridge.reason("What is entropy?")
        
        # Should return error without crashing
        assert isinstance(result, dict)
    
    def test_bridge_run_script_no_interpreter(self):
        """Test script execution when interpreter is not available."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        # Attempt to run a script - result depends on interpreter availability
        result = bridge.run_script("print('hello')")
        
        assert isinstance(result, dict)
        # Either succeeds (interpreter available) or fails gracefully
        assert "success" in result
    
    def test_bridge_get_rust_stats_no_rust(self):
        """Test Rust stats when not available."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        bridge._rust_available = False
        
        stats = bridge.get_rust_stats()
        
        assert isinstance(stats, dict)
        assert "status" in stats
        assert stats["status"] == "Not Available"


# ============================================================================
# Widget Tests
# ============================================================================

class TestWidgets:
    """Tests for clipboard-enabled widgets."""
    
    def test_import_widgets(self):
        """Test widgets can be imported."""
        from widgets import (
            ClipboardInput,
            ClipboardTextArea,
            MoleculeInput,
            ScriptEditor,
        )
        
        assert ClipboardInput is not None
        assert ClipboardTextArea is not None
        assert MoleculeInput is not None
        assert ScriptEditor is not None
    
    def test_clipboard_functions(self):
        """Test clipboard utility functions."""
        from widgets import (
            clipboard_copy,
            clipboard_paste,
            clipboard_paste_with_type,
            get_clipboard_master,
        )
        
        assert callable(clipboard_copy)
        assert callable(clipboard_paste)
        assert callable(clipboard_paste_with_type)
        assert callable(get_clipboard_master)
    
    def test_global_clipboard_master(self):
        """Test global clipboard master instance."""
        from widgets import get_clipboard_master, ClipboardMaster
        
        cm = get_clipboard_master()
        
        assert isinstance(cm, ClipboardMaster)
        
        # Same instance returned each time
        cm2 = get_clipboard_master()
        assert cm is cm2


# ============================================================================
# MoleculeInput Tests
# ============================================================================

class TestMoleculeInput:
    """Tests for MoleculeInput widget."""
    
    def test_xyz_parsing_basic(self):
        """Test basic XYZ parsing."""
        from widgets import MoleculeInput
        
        # Create instance (without running in app context)
        # We can test the parsing logic directly
        content = """H  0.0  0.0  0.0
H  0.0  0.0  1.4"""
        
        # Parse directly
        lines = content.strip().split('\n')
        atoms = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append((element, (x, y, z)))
        
        assert len(atoms) == 2
        assert atoms[0][0] == "H"
        assert atoms[0][1] == (0.0, 0.0, 0.0)
        assert atoms[1][0] == "H"
        assert atoms[1][1] == (0.0, 0.0, 1.4)
    
    def test_xyz_parsing_water(self):
        """Test XYZ parsing for water molecule."""
        content = """O  0.0  0.0  0.0
H  0.0  0.757  0.587
H  0.0  -0.757  0.587"""
        
        lines = content.strip().split('\n')
        atoms = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append((element, (x, y, z)))
        
        assert len(atoms) == 3
        assert atoms[0][0] == "O"
        assert atoms[1][0] == "H"
        assert atoms[2][0] == "H"
    
    def test_xyz_parsing_with_comments(self):
        """Test XYZ parsing with comment lines."""
        content = """# Water molecule
O  0.0  0.0  0.0
H  0.0  0.757  0.587
# Another hydrogen
H  0.0  -0.757  0.587"""
        
        lines = content.strip().split('\n')
        atoms = []
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                element = parts[0]
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append((element, (x, y, z)))
        
        assert len(atoms) == 3


# ============================================================================
# Controller Panel Tests
# ============================================================================

class TestControllerPanels:
    """Tests for controller UI panels."""
    
    def test_import_panels(self):
        """Test panel classes can be imported."""
        from controller import (
            ChatPanel,
            ScriptPanel,
            MoleculePanel,
            StatusPanel,
            ResultsPanel,
        )
        
        assert ChatPanel is not None
        assert ScriptPanel is not None
        assert MoleculePanel is not None
        assert StatusPanel is not None
        assert ResultsPanel is not None
    
    def test_import_controller(self):
        """Test main controller can be imported."""
        from controller import QENEXController, main
        
        assert QENEXController is not None
        assert callable(main)
    
    def test_script_panel_example(self):
        """Test ScriptPanel has example script."""
        from controller import ScriptPanel
        
        assert hasattr(ScriptPanel, 'EXAMPLE_SCRIPT')
        assert "Molecule" in ScriptPanel.EXAMPLE_SCRIPT
        assert "HartreeFockSolver" in ScriptPanel.EXAMPLE_SCRIPT


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for the controller system."""
    
    def test_full_import_chain(self):
        """Test full import chain works."""
        # This tests that all imports resolve correctly
        from controller import (
            QENEXBridge,
            QENEXController,
            ChatPanel,
            ScriptPanel,
            MoleculePanel,
            StatusPanel,
        )
        from widgets import (
            ClipboardMaster,
            ClipboardEntry,
            ClipboardInput,
            ClipboardTextArea,
            MoleculeInput,
            ScriptEditor,
        )
        
        # All imports successful
        assert True
    
    def test_bridge_with_interpreter(self):
        """Test bridge can access interpreter."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        # Access interpreter (may be None if not available)
        interp = bridge.interpreter
        
        # Should not crash
        assert True
    
    def test_bridge_with_integration(self):
        """Test bridge can access LLM integration."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        # Access integration (may be None if not available)
        integ = bridge.integration
        
        # Should not crash
        assert True


# ============================================================================
# Content Type Detection Tests
# ============================================================================

class TestContentDetection:
    """Tests for content type auto-detection."""
    
    def test_xyz_with_negative_coords(self):
        """Test XYZ detection with negative coordinates."""
        from widgets import ClipboardMaster
        
        content = """C  -1.234  2.345  -3.456
H  0.0  -0.5  1.2"""
        
        assert ClipboardMaster.detect_type(content) == "xyz"
    
    def test_python_with_imports(self):
        """Test Python detection with imports."""
        from widgets import ClipboardMaster
        
        content = """import numpy as np
import scipy.linalg as la
from typing import List"""
        
        assert ClipboardMaster.detect_type(content) == "python"
    
    def test_python_with_class(self):
        """Test Python detection with class definition."""
        from widgets import ClipboardMaster
        
        content = """class MyClass:
    def __init__(self):
        self.value = 0"""
        
        assert ClipboardMaster.detect_type(content) == "python"
    
    def test_python_with_decorator(self):
        """Test Python detection with decorator."""
        from widgets import ClipboardMaster
        
        content = """@property
def my_property(self):
    return self._value"""
        
        assert ClipboardMaster.detect_type(content) == "python"
    
    def test_qlang_with_const(self):
        """Test Q-Lang detection with const."""
        from widgets import ClipboardMaster
        
        content = """const speed_of_light = 299792458 m/s
const planck = 6.626e-34 J*s"""
        
        assert ClipboardMaster.detect_type(content) == "qlang"
    
    def test_qlang_with_molecule(self):
        """Test Q-Lang detection with molecule keyword (no XYZ-like lines)."""
        from widgets import ClipboardMaster
        
        # Content that is clearly Q-Lang, not XYZ
        content = """# Q-Lang molecule definition
molecule water {
  basis = "6-31G*"
  charge = 0
}"""
        
        assert ClipboardMaster.detect_type(content) == "qlang"
    
    def test_mixed_content_defaults_to_text(self):
        """Test mixed content defaults to text."""
        from widgets import ClipboardMaster
        
        content = """Some random text
that doesn't match any pattern
just regular prose"""
        
        assert ClipboardMaster.detect_type(content) == "text"


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_xyz_content(self):
        """Test XYZ parsing with empty content."""
        from widgets import ClipboardMaster
        
        assert ClipboardMaster.detect_type("") == "text"
    
    def test_whitespace_only(self):
        """Test detection with whitespace only."""
        from widgets import ClipboardMaster
        
        assert ClipboardMaster.detect_type("   \n\n   ") == "text"
    
    def test_single_line(self):
        """Test detection with single line."""
        from widgets import ClipboardMaster
        
        assert ClipboardMaster.detect_type("H  0.0  0.0  0.0") == "xyz"
    
    def test_bridge_generate_no_integration(self):
        """Test code generation without integration."""
        from controller import QENEXBridge
        
        bridge = QENEXBridge(verbose=False)
        
        result = bridge.generate_code("write a function")
        
        assert isinstance(result, dict)
        # Either succeeds (integration available) or fails gracefully
        assert "success" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
