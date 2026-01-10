"""
QENEX LAB Terminal UI - Custom Widgets with Clipboard Support

Provides enhanced Input and TextArea widgets that properly handle
system clipboard events (Ctrl+V, Shift+Insert) using pyperclip.
"""

from __future__ import annotations

from textual.widgets import Input, TextArea
from textual.binding import Binding
from textual import events

# Try to import pyperclip, provide fallback if unavailable
try:
    import pyperclip
    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False
    pyperclip = None


def clipboard_copy(text: str) -> bool:
    """
    Copy text to system clipboard.
    Returns True if successful, False otherwise.
    """
    if not CLIPBOARD_AVAILABLE:
        return False
    try:
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def clipboard_paste() -> str | None:
    """
    Paste text from system clipboard.
    Returns the clipboard content or None if unavailable.
    """
    if not CLIPBOARD_AVAILABLE:
        return None
    try:
        return pyperclip.paste()
    except Exception:
        return None


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
