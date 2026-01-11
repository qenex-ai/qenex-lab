"""
QENEX LAB Terminal UI - Custom Widgets with Clipboard Master Support

Provides enhanced Input and TextArea widgets that properly handle
system clipboard events (Ctrl+V, Shift+Insert) using pyperclip.

Clipboard Master Features:
- Persistent paste (content doesn't clear on focus loss)
- Paste history with access to recent items
- Auto-detect content type (XYZ, Python, Q-Lang)
- Multi-format paste support

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

from __future__ import annotations

from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import re

from textual.widgets import Input, TextArea
from textual.binding import Binding
from textual.message import Message
from textual import events

# Try to import pyperclip, provide fallback if unavailable
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    pyperclip = None


# ============================================================================
# Clipboard Master Module
# ============================================================================

@dataclass
class ClipboardEntry:
    """A clipboard history entry."""
    content: str
    timestamp: datetime
    content_type: str  # "text", "xyz", "python", "qlang"
    source: str = ""  # optional source identifier
    
    @property
    def preview(self) -> str:
        """Get a short preview of the content."""
        first_line = self.content.split('\n')[0][:50]
        return first_line + "..." if len(self.content) > 50 else first_line


class ClipboardMaster:
    """
    Clipboard Master - Enhanced clipboard management.
    
    Features:
    - Persistent paste history (survives focus changes)
    - Auto-detect content type
    - Access to recent paste items
    - Content type filtering
    """
    
    MAX_HISTORY = 20
    
    def __init__(self):
        self.history: List[ClipboardEntry] = []
        self._last_content: str = ""
    
    def copy(self, text: str, content_type: str = "text", source: str = "") -> bool:
        """
        Copy text to system clipboard and add to history.
        
        Args:
            text: Text to copy
            content_type: Type of content ("text", "xyz", "python", "qlang")
            source: Optional source identifier
        
        Returns:
            True if successful
        """
        if not CLIPBOARD_AVAILABLE:
            return False
        
        try:
            pyperclip.copy(text)
            
            # Add to history
            entry = ClipboardEntry(
                content=text,
                timestamp=datetime.now(),
                content_type=content_type or self.detect_type(text),
                source=source,
            )
            self.history.insert(0, entry)
            
            # Trim history
            if len(self.history) > self.MAX_HISTORY:
                self.history = self.history[:self.MAX_HISTORY]
            
            self._last_content = text
            return True
            
        except Exception:
            return False
    
    def paste(self) -> Tuple[Optional[str], str]:
        """
        Paste text from system clipboard.
        
        Returns:
            Tuple of (content, detected_type) or (None, "")
        """
        if not CLIPBOARD_AVAILABLE:
            return None, ""
        
        try:
            content = pyperclip.paste()
            if content:
                content_type = self.detect_type(content)
                
                # Add to history if new
                if content != self._last_content:
                    entry = ClipboardEntry(
                        content=content,
                        timestamp=datetime.now(),
                        content_type=content_type,
                        source="paste",
                    )
                    self.history.insert(0, entry)
                    
                    if len(self.history) > self.MAX_HISTORY:
                        self.history = self.history[:self.MAX_HISTORY]
                    
                    self._last_content = content
                
                return content, content_type
            
            return None, ""
            
        except Exception:
            return None, ""
    
    def get_recent(self, n: int = 5, content_type: Optional[str] = None) -> List[ClipboardEntry]:
        """
        Get recent clipboard entries.
        
        Args:
            n: Number of entries to return
            content_type: Optional filter by type
        
        Returns:
            List of recent ClipboardEntry objects
        """
        if content_type:
            filtered = [e for e in self.history if e.content_type == content_type]
            return filtered[:n]
        return self.history[:n]
    
    def clear_history(self) -> None:
        """Clear clipboard history."""
        self.history.clear()
        self._last_content = ""
    
    @staticmethod
    def detect_type(content: str) -> str:
        """
        Auto-detect content type.
        
        Returns:
            "xyz", "python", "qlang", or "text"
        """
        lines = content.strip().split('\n')
        
        # Check for XYZ format (element X Y Z)
        xyz_pattern = re.compile(r'^[A-Z][a-z]?\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s+-?\d+\.?\d*\s*$')
        xyz_matches = sum(1 for line in lines if xyz_pattern.match(line.strip()))
        if xyz_matches >= len(lines) * 0.5 and xyz_matches >= 1:
            return "xyz"
        
        # Check for Python code indicators
        python_indicators = [
            r'^import\s+\w+',
            r'^from\s+\w+\s+import',
            r'^def\s+\w+\s*\(',
            r'^class\s+\w+',
            r'^if\s+__name__\s*==',
            r'^\s*@\w+',
            r'print\s*\(',
        ]
        for pattern in python_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return "python"
        
        # Check for Q-Lang indicators
        qlang_indicators = [
            r'^const\s+\w+\s*=',
            r'^molecule\s+\w+',
            r'^lattice\s+\w+',
            r'^proof\s+\w+',
            r'^domain\s+\w+',
            r'#\s*Q-Lang',
        ]
        for pattern in qlang_indicators:
            if re.search(pattern, content, re.MULTILINE):
                return "qlang"
        
        return "text"


# Global clipboard master instance
_clipboard_master = ClipboardMaster()


def get_clipboard_master() -> ClipboardMaster:
    """Get the global clipboard master instance."""
    return _clipboard_master


def clipboard_copy(text: str, content_type: str = "") -> bool:
    """
    Copy text to system clipboard.
    Returns True if successful, False otherwise.
    """
    return _clipboard_master.copy(text, content_type)


def clipboard_paste() -> str | None:
    """
    Paste text from system clipboard.
    Returns the clipboard content or None if unavailable.
    """
    content, _ = _clipboard_master.paste()
    return content


def clipboard_paste_with_type() -> Tuple[Optional[str], str]:
    """
    Paste text from clipboard with auto-detected type.
    Returns (content, type) tuple.
    """
    return _clipboard_master.paste()


class ClipboardInput(Input):
    """
    Enhanced Input widget with system clipboard support.
    
    Handles Ctrl+V and Shift+Insert for pasting from system clipboard
    using pyperclip for cross-platform compatibility.
    """
    
    BINDINGS = [
        Binding("ctrl+v", "paste_clipboard", "Paste", show=False),
        Binding("shift+insert", "paste_clipboard", "Paste", show=False),
        Binding("ctrl+c", "copy_clipboard", "Copy", show=False),
        Binding("ctrl+x", "cut_clipboard", "Cut", show=False),
        Binding("ctrl+a", "select_all", "Select All", show=False),
    ]
    
    def action_paste_clipboard(self) -> None:
        """Paste text from system clipboard at cursor position."""
        clipboard_text = clipboard_paste()
        if clipboard_text:
            # Remove newlines for single-line input
            clipboard_text = clipboard_text.replace('\n', ' ').replace('\r', '')
            self.insert_text_at_cursor(clipboard_text)
    
    def action_copy_clipboard(self) -> None:
        """Copy selected text to system clipboard."""
        # Input widget doesn't have built-in selection, copy entire value
        if self.value:
            clipboard_copy(self.value)
    
    def action_cut_clipboard(self) -> None:
        """Cut all text to system clipboard."""
        if self.value:
            clipboard_copy(self.value)
            self.value = ""
    
    def action_select_all(self) -> None:
        """Select all text (move cursor to end for Input widget)."""
        self.cursor_position = len(self.value)


class ClipboardTextArea(TextArea):
    """
    Enhanced TextArea widget with system clipboard support.
    
    Handles Ctrl+V and Shift+Insert for pasting multi-line text
    from system clipboard. Perfect for pasting molecular geometries
    and Python scripts.
    """
    
    BINDINGS = [
        Binding("ctrl+v", "paste_clipboard", "Paste", show=False),
        Binding("shift+insert", "paste_clipboard", "Paste", show=False),
        Binding("ctrl+c", "copy_clipboard", "Copy", show=False),
        Binding("ctrl+x", "cut_clipboard", "Cut", show=False),
        Binding("ctrl+a", "select_all", "Select All", show=False),
        Binding("ctrl+s", "save_content", "Save", show=False),
    ]
    
    def action_paste_clipboard(self) -> None:
        """Paste text from system clipboard at cursor position."""
        clipboard_text = clipboard_paste()
        if clipboard_text:
            # Normalize line endings
            clipboard_text = clipboard_text.replace('\r\n', '\n').replace('\r', '\n')
            self.insert(clipboard_text)
    
    def action_copy_clipboard(self) -> None:
        """Copy selected text to system clipboard."""
        selected_text = self.selected_text
        if selected_text:
            clipboard_copy(selected_text)
    
    def action_cut_clipboard(self) -> None:
        """Cut selected text to system clipboard."""
        selected_text = self.selected_text
        if selected_text:
            clipboard_copy(selected_text)
            # Use the delete method with selection bounds
            if self.selection:
                self.delete(self.selection.start, self.selection.end)
    
    def action_select_all(self) -> None:
        """Select all text in the text area."""
        self.select_all()
    
    def action_save_content(self) -> None:
        """Trigger save event (handled by parent)."""
        self.post_message(self.Changed(self))


class MoleculeInput(ClipboardTextArea):
    """
    Specialized TextArea for molecular geometry input.
    
    Supports XYZ format and can parse pasted molecular structures.
    """
    
    DEFAULT_CSS = """
    MoleculeInput {
        height: auto;
        min-height: 8;
        max-height: 20;
        border: solid $primary;
    }
    
    MoleculeInput:focus {
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.language = None  # Plain text
    
    def get_xyz_data(self) -> list[tuple[str, tuple[float, float, float]]] | None:
        """
        Parse the content as XYZ format molecular geometry.
        
        Returns list of (element, (x, y, z)) tuples or None if invalid.
        """
        lines = self.text.strip().split('\n')
        atoms = []
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                try:
                    element = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    atoms.append((element, (x, y, z)))
                except (ValueError, IndexError):
                    continue
        
        return atoms if atoms else None


class ScriptEditor(ClipboardTextArea):
    """
    Python script editor with syntax highlighting.
    
    Supports pasting Python scripts and Q-Lang code.
    """
    
    DEFAULT_CSS = """
    ScriptEditor {
        height: 100%;
        border: solid $primary;
    }
    
    ScriptEditor:focus {
        border: solid $accent;
    }
    """
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault('language', 'python')
        kwargs.setdefault('theme', 'monokai')
        super().__init__(*args, **kwargs)
