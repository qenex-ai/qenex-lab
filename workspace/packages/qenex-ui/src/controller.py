#!/usr/bin/env python3
r"""
QENEX LAB Primary Controller UI
================================
The central control interface for the QENEX Scientific Intelligence Laboratory.

This is the PRIMARY interface that integrates:
- Chat/Reasoning panel linked to `qenex reason` (Scout 17B)
- Script Editor with Run button linked to `qenex run` (Q-Lang)
- Code Generation linked to `qenex generate` (DeepSeek)
- Molecular Visualization using Rust ERI (qenex_accelerate)
- Clipboard Master for persistent paste operations
- System Status and Backend Monitoring

Architecture:
    +-------------------------------------------------------------------------+
    |                    QENEX LAB PRIMARY CONTROLLER                          |
    +-------------------------------------------------------------------------+
    |                                                                          |
    |  +-------------------------------+  +--------------------------------+  |
    |  |     Chat / Reasoning          |  |      Script Editor              |  |
    |  |        (Scout)                |  |    + Run Button (Q-Lang)       |  |
    |  |                               |  |                                 |  |
    |  |  "What is the binding        |  |  ```python                     |  |
    |  |   energy of water?"          |  |  from molecule import Molecule |  |
    |  |                               |  |  ...                           |  |
    |  |  > Scout reasoning...        |  |  ```                            |  |
    |  |                               |  |                                 |  |
    |  |  [Input field]               |  |  [Run] [Generate] [Clear]      |  |
    |  +-------------------------------+  +--------------------------------+  |
    |                                                                          |
    |  +-------------------------------+  +--------------------------------+  |
    |  |     Molecule Viewer           |  |      System Status             |  |
    |  |      (Rust ERI)               |  |                                 |  |
    |  |                               |  |  Scout:    * Connected         |  |
    |  |    O         H               |  |  DeepSeek: * Connected         |  |
    |  |     \       /                 |  |  Rust ERI: * Available         |  |
    |  |      \     /                  |  |  Tokens:   1,234 used          |  |
    |  |       \   /                   |  |                                 |  |
    |  |        \ /                    |  |  Cache Hits: 98.7%             |  |
    |  |         H                     |  |  Threads: 8 pinned             |  |
    |  +-------------------------------+  +--------------------------------+  |
    |                                                                          |
    +-------------------------------------------------------------------------+

100% Local Operation - No External APIs

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

from __future__ import annotations

import sys
import os
import io
import time
import traceback
import asyncio
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

# Add QENEX packages to path
_WORKSPACE = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_WORKSPACE / 'packages' / 'qenex_chem' / 'src'))
sys.path.insert(0, str(_WORKSPACE / 'packages' / 'qenex-qlang' / 'src'))
sys.path.insert(0, str(_WORKSPACE / 'packages' / 'qenex-ui' / 'src'))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer, Grid
from textual.widgets import (
    Header, Footer, Static, Button, Label, Input,
    TabbedContent, TabPane, DataTable, Log,
    LoadingIndicator, ProgressBar, RichLog, Markdown
)
from textual.reactive import reactive
from textual.message import Message
from textual import work
from textual.worker import Worker, get_current_worker

from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.console import Console

# Import clipboard widgets
try:
    from widgets import ClipboardInput, ClipboardTextArea, MoleculeInput, ScriptEditor
except ImportError:
    from .widgets import ClipboardInput, ClipboardTextArea, MoleculeInput, ScriptEditor


# ============================================================================
# QENEX Integration Layer
# ============================================================================

class QENEXBridge:
    """
    Bridge to QENEX CLI functionality.
    
    Provides direct access to:
    - qenex reason (Scout reasoning)
    - qenex run (Q-Lang scripts)
    - qenex generate (DeepSeek code gen)
    - qenex_accelerate (Rust ERI)
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._integration = None
        self._interpreter = None
        self._rust_available = False
        self._init_components()
    
    def _init_components(self):
        """Initialize QENEX components lazily."""
        # Check Rust accelerator
        try:
            import qenex_accelerate as qa
            self._rust_available = True
            if self.verbose:
                stats = qa.scout_report()
                print(f"Rust ERI: {stats.get('num_threads', '?')} threads")
        except ImportError:
            self._rust_available = False
    
    @property
    def integration(self):
        """Lazy load LLM integration."""
        if self._integration is None:
            try:
                from llm_integration import QENEXIntegration, IntegrationConfig
                from llm_backend import BackendType, MockBackend
                
                config = IntegrationConfig(
                    scout_backend=BackendType.MOCK,
                    deepseek_backend=BackendType.MOCK,
                    context_store_path=str(Path.home() / ".qenex" / "contexts"),
                )
                self._integration = QENEXIntegration(config=config, verbose=False)
                
                # Register mock backend
                self._integration.router.register_backend(MockBackend())
                
            except ImportError as e:
                print(f"Warning: LLM integration not available: {e}")
                self._integration = None
        return self._integration
    
    @property
    def interpreter(self):
        """Lazy load Q-Lang interpreter."""
        if self._interpreter is None:
            try:
                from interpreter import QLangInterpreter
                self._interpreter = QLangInterpreter()
            except ImportError as e:
                print(f"Warning: Q-Lang interpreter not available: {e}")
                self._interpreter = None
        return self._interpreter
    
    def reason(self, query: str, mode: str = "REASON") -> Dict[str, Any]:
        """
        Execute Scout reasoning.
        
        Links to: qenex reason "query"
        """
        if self.integration is None:
            return {
                "success": False,
                "output": "LLM integration not available",
                "tokens_generated": 0,
                "backend": "none",
            }
        
        return self.integration.reason(query, mode=mode)
    
    def run_script(self, code: str) -> Dict[str, Any]:
        """
        Execute Q-Lang script.
        
        Links to: qenex run script.ql
        """
        if self.interpreter is None:
            return {
                "success": False,
                "output": "Q-Lang interpreter not available",
                "error": "No interpreter",
            }
        
        output = io.StringIO()
        error_output = io.StringIO()
        
        try:
            with redirect_stdout(output), redirect_stderr(error_output):
                result = self.interpreter.execute(code)
            
            return {
                "success": True,
                "output": output.getvalue(),
                "error": error_output.getvalue(),
                "result": result,
            }
        except Exception as e:
            return {
                "success": False,
                "output": output.getvalue(),
                "error": str(e),
                "traceback": traceback.format_exc(),
            }
    
    def generate_code(self, description: str, language: str = "python") -> Dict[str, Any]:
        """
        Generate code with DeepSeek.
        
        Links to: qenex generate "description"
        """
        if self.integration is None:
            return {
                "success": False,
                "code": "",
                "raw_response": "LLM integration not available",
                "tokens_generated": 0,
            }
        
        return self.integration.generate_code(description, language=language)
    
    def get_rust_stats(self) -> Dict[str, str]:
        """Get Rust ERI statistics."""
        if not self._rust_available:
            return {"status": "Not Available"}
        
        try:
            import qenex_accelerate as qa
            return qa.scout_report()
        except Exception as e:
            return {"status": f"Error: {e}"}
    
    def get_status(self) -> Dict[str, Any]:
        """Get overall system status."""
        status = {
            "rust_available": self._rust_available,
            "interpreter_available": self.interpreter is not None,
            "integration_available": self.integration is not None,
        }
        
        if self.integration:
            int_status = self.integration.get_status()
            status.update({
                "scout_backend": int_status["scout"]["backend"],
                "deepseek_backend": int_status["deepseek"]["backend"],
                "total_tokens": int_status["tokens"]["total_tokens"],
                "available_backends": int_status["available_backends"],
            })
        
        if self._rust_available:
            rust_stats = self.get_rust_stats()
            status.update({
                "rust_threads": rust_stats.get("num_threads", "?"),
                "cache_hit_rate": rust_stats.get("cache_hit_rate", "?"),
            })
        
        return status


# ============================================================================
# UI Components
# ============================================================================

class ChatPanel(Vertical):
    """
    Chat/Reasoning panel linked to Scout.
    
    Features:
    - Input field for queries
    - History of reasoning sessions
    - Mode selection (REASON, SYNTHESIZE, PROVE, etc.)
    """
    
    DEFAULT_CSS = """
    ChatPanel {
        height: 100%;
        border: solid $primary;
        padding: 0 1;
    }
    
    ChatPanel #chat-history {
        height: 1fr;
        border: none;
        background: $surface;
    }
    
    ChatPanel #chat-input-row {
        height: auto;
        padding: 1 0;
    }
    
    ChatPanel #chat-input {
        width: 1fr;
    }
    
    ChatPanel .mode-btn {
        min-width: 10;
    }
    """
    
    class ReasoningRequested(Message):
        """Message sent when user requests reasoning."""
        def __init__(self, query: str, mode: str):
            self.query = query
            self.mode = mode
            super().__init__()
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.current_mode = "REASON"
    
    def compose(self) -> ComposeResult:
        yield Label("Chat / Reasoning (Scout)", classes="section-title")
        yield RichLog(id="chat-history", highlight=True, markup=True)
        
        with Horizontal(id="mode-row"):
            yield Button("Reason", id="mode-reason", classes="mode-btn", variant="primary")
            yield Button("Synth", id="mode-synth", classes="mode-btn")
            yield Button("Prove", id="mode-prove", classes="mode-btn")
            yield Button("Hypo", id="mode-hypo", classes="mode-btn")
        
        with Horizontal(id="chat-input-row"):
            yield ClipboardInput(
                placeholder="Ask Scout a question... (Ctrl+V to paste)",
                id="chat-input"
            )
            yield Button("Send", id="btn-send", variant="success")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        
        if btn == "btn-send":
            self._send_query()
        elif btn == "mode-reason":
            self._set_mode("REASON")
        elif btn == "mode-synth":
            self._set_mode("SYNTHESIZE")
        elif btn == "mode-prove":
            self._set_mode("PROVE")
        elif btn == "mode-hypo":
            self._set_mode("HYPOTHESIZE")
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._send_query()
    
    def _send_query(self) -> None:
        input_widget = self.query_one("#chat-input", ClipboardInput)
        query = input_widget.value.strip()
        
        if query:
            self.post_message(self.ReasoningRequested(query, self.current_mode))
            input_widget.value = ""
    
    def _set_mode(self, mode: str) -> None:
        self.current_mode = mode
        
        # Update button states
        for btn_id, btn_mode in [
            ("mode-reason", "REASON"),
            ("mode-synth", "SYNTHESIZE"),
            ("mode-prove", "PROVE"),
            ("mode-hypo", "HYPOTHESIZE"),
        ]:
            btn = self.query_one(f"#{btn_id}", Button)
            btn.variant = "primary" if btn_mode == mode else "default"
    
    def log(self, message: str) -> None:
        """Log a message to chat history."""
        self.query_one("#chat-history", RichLog).write(message)


class ScriptPanel(Vertical):
    """
    Script Editor panel linked to Q-Lang.
    
    Features:
    - Code editor with syntax highlighting
    - Run button → qenex run
    - Generate button → qenex generate
    - Clipboard support
    """
    
    DEFAULT_CSS = """
    ScriptPanel {
        height: 100%;
        border: solid $success;
        padding: 0 1;
    }
    
    ScriptPanel #script-editor {
        height: 1fr;
    }
    
    ScriptPanel #script-buttons {
        height: auto;
        padding: 1 0;
    }
    """
    
    EXAMPLE_SCRIPT = '''"""
QENEX LAB - Example Script
Paste your script here or use /generate to create code.
"""
from molecule import Molecule
from solver import HartreeFockSolver

# Define H2 molecule
mol = Molecule([
    ('H', (0.0, 0.0, 0.0)),
    ('H', (0.0, 0.0, 1.4))
])

# Compute energy
solver = HartreeFockSolver()
E_elec, E_tot = solver.compute_energy(mol, verbose=True)

print(f"Total Energy: {E_tot:.8f} Ha")
'''
    
    class RunRequested(Message):
        """Message sent when user requests script execution."""
        def __init__(self, code: str):
            self.code = code
            super().__init__()
    
    class GenerateRequested(Message):
        """Message sent when user requests code generation."""
        def __init__(self, description: str):
            self.description = description
            super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Label("Script Editor (Q-Lang / Python)", classes="section-title")
        yield ScriptEditor(self.EXAMPLE_SCRIPT, id="script-editor")
        
        with Horizontal(id="script-buttons"):
            yield Button("Run", id="btn-run-script", variant="success")
            yield Button("Generate", id="btn-generate", variant="primary")
            yield Button("Clear", id="btn-clear-script", variant="error")
            yield Button("Example", id="btn-example")
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn = event.button.id
        
        if btn == "btn-run-script":
            self._run_script()
        elif btn == "btn-generate":
            self._request_generate()
        elif btn == "btn-clear-script":
            self._clear_editor()
        elif btn == "btn-example":
            self._load_example()
    
    def _run_script(self) -> None:
        editor = self.query_one("#script-editor", ScriptEditor)
        code = editor.text.strip()
        if code:
            self.post_message(self.RunRequested(code))
    
    def _request_generate(self) -> None:
        # TODO: Prompt for description
        self.post_message(self.GenerateRequested("Compute the energy of a water molecule"))
    
    def _clear_editor(self) -> None:
        editor = self.query_one("#script-editor", ScriptEditor)
        editor.clear()
    
    def _load_example(self) -> None:
        editor = self.query_one("#script-editor", ScriptEditor)
        editor.clear()
        editor.insert(self.EXAMPLE_SCRIPT)
    
    def set_code(self, code: str) -> None:
        """Set the editor content."""
        editor = self.query_one("#script-editor", ScriptEditor)
        editor.clear()
        editor.insert(code)


class MoleculePanel(Vertical):
    """
    Molecule Viewer panel with Rust ERI integration.
    
    Features:
    - XYZ input with clipboard support
    - ASCII visualization
    - ERI statistics from Rust backend
    """
    
    DEFAULT_CSS = """
    MoleculePanel {
        height: 100%;
        border: solid $warning;
        padding: 0 1;
    }
    
    MoleculePanel #mol-input {
        height: 10;
    }
    
    MoleculePanel #mol-viz {
        height: 1fr;
        background: $surface;
        border: solid $primary;
    }
    
    MoleculePanel #eri-stats {
        height: auto;
        padding: 1;
    }
    """
    
    DEFAULT_MOLECULE = """O  0.0  0.0  0.0
H  0.0  0.757  0.587
H  0.0  -0.757  0.587"""
    
    class MoleculeChanged(Message):
        """Message sent when molecule geometry changes."""
        def __init__(self, atoms: List[Tuple[str, Tuple[float, float, float]]]):
            self.atoms = atoms
            super().__init__()
    
    def compose(self) -> ComposeResult:
        yield Label("Molecule Viewer (Rust ERI)", classes="section-title")
        yield MoleculeInput(self.DEFAULT_MOLECULE, id="mol-input")
        yield Static(id="mol-viz")
        yield Static(id="eri-stats")
    
    def on_mount(self) -> None:
        self._update_visualization()
        self._update_eri_stats()
    
    def _update_visualization(self) -> None:
        """Update ASCII molecule visualization."""
        mol_input = self.query_one("#mol-input", MoleculeInput)
        atoms = mol_input.get_xyz_data()
        
        if not atoms:
            self.query_one("#mol-viz", Static).update("No valid molecule")
            return
        
        # Simple ASCII visualization
        viz_lines = ["Molecule Structure:", ""]
        for i, (element, (x, y, z)) in enumerate(atoms):
            viz_lines.append(f"  {element} ({x:7.3f}, {y:7.3f}, {z:7.3f})")
        
        viz_lines.append("")
        viz_lines.append(f"Atoms: {len(atoms)}")
        
        # Calculate center of mass
        total_mass = 0
        com = [0.0, 0.0, 0.0]
        masses = {'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'S': 32.065}
        
        for element, (x, y, z) in atoms:
            mass = masses.get(element, 12.0)
            total_mass += mass
            com[0] += mass * x
            com[1] += mass * y
            com[2] += mass * z
        
        if total_mass > 0:
            com = [c / total_mass for c in com]
            viz_lines.append(f"Center of Mass: ({com[0]:.3f}, {com[1]:.3f}, {com[2]:.3f})")
        
        self.query_one("#mol-viz", Static).update("\n".join(viz_lines))
    
    def _update_eri_stats(self) -> None:
        """Update ERI statistics from Rust backend."""
        try:
            import qenex_accelerate as qa
            stats = qa.scout_report()
            
            text = Text()
            text.append("Rust ERI Backend\n", style="bold cyan")
            text.append(f"Threads: {stats.get('num_threads', '?')}\n")
            text.append(f"Cache Hit Rate: {stats.get('cache_hit_rate', '?')}\n")
            text.append(f"Primitives: {stats.get('total_primitives', '0')}")
            
            self.query_one("#eri-stats", Static).update(text)
        except ImportError:
            self.query_one("#eri-stats", Static).update("Rust backend not available")


class StatusPanel(Vertical):
    """
    System Status panel.
    
    Shows:
    - Backend connection status
    - Token usage
    - Performance metrics
    """
    
    DEFAULT_CSS = """
    StatusPanel {
        height: 100%;
        border: solid $accent;
        padding: 0 1;
    }
    
    StatusPanel #status-content {
        height: 1fr;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Label("System Status", classes="section-title")
        yield Static(id="status-content")
    
    def update_status(self, status: Dict[str, Any]) -> None:
        """Update the status display."""
        text = Text()
        
        # Backend status
        text.append("Backends\n", style="bold cyan")
        
        scout_ok = status.get("integration_available", False)
        text.append(f"  Scout:    {'[green]OK[/]' if scout_ok else '[red]N/A[/]'}\n", style="")
        
        deepseek_ok = status.get("integration_available", False)
        text.append(f"  DeepSeek: {'[green]OK[/]' if deepseek_ok else '[red]N/A[/]'}\n", style="")
        
        rust_ok = status.get("rust_available", False)
        text.append(f"  Rust ERI: {'[green]OK[/]' if rust_ok else '[red]N/A[/]'}\n", style="")
        
        qlang_ok = status.get("interpreter_available", False)
        text.append(f"  Q-Lang:   {'[green]OK[/]' if qlang_ok else '[red]N/A[/]'}\n", style="")
        
        # Token usage
        text.append("\nToken Usage\n", style="bold cyan")
        text.append(f"  Total: {status.get('total_tokens', 0):,}\n")
        
        # Rust stats
        if rust_ok:
            text.append("\nRust Performance\n", style="bold cyan")
            text.append(f"  Threads: {status.get('rust_threads', '?')}\n")
            text.append(f"  Cache: {status.get('cache_hit_rate', '?')}\n")
        
        # Available backends
        backends = status.get("available_backends", [])
        if backends:
            text.append(f"\nAvailable: {', '.join(backends)}\n", style="dim")
        
        self.query_one("#status-content", Static).update(text)


class ResultsPanel(RichLog):
    """Results panel for computation output."""
    
    DEFAULT_CSS = """
    ResultsPanel {
        height: 100%;
        border: solid $success;
        background: $surface;
    }
    """


# ============================================================================
# Main Application
# ============================================================================

class QENEXController(App):
    """
    QENEX LAB Primary Controller Application.
    
    The central command interface for the scientific laboratory.
    """
    
    TITLE = "QENEX LAB"
    SUB_TITLE = "Primary Controller"
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 3;
        grid-rows: 1fr 1fr 8;
        grid-columns: 1fr 1fr;
    }
    
    #chat-panel {
        column-span: 1;
    }
    
    #script-panel {
        column-span: 1;
    }
    
    #molecule-panel {
        column-span: 1;
    }
    
    #status-panel {
        column-span: 1;
    }
    
    #results-row {
        column-span: 2;
        height: 8;
    }
    
    .section-title {
        text-style: bold;
        color: $text;
        padding: 0 1;
        background: $primary;
        height: 1;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "run_script", "Run Script"),
        Binding("ctrl+enter", "send_query", "Send Query"),
        Binding("ctrl+g", "generate", "Generate"),
        Binding("ctrl+s", "save", "Save"),
        Binding("f1", "help", "Help"),
        Binding("f5", "refresh_status", "Refresh"),
    ]
    
    def __init__(self):
        super().__init__()
        self.bridge = QENEXBridge(verbose=False)
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        yield ChatPanel(id="chat-panel")
        yield ScriptPanel(id="script-panel")
        yield MoleculePanel(id="molecule-panel")
        yield StatusPanel(id="status-panel")
        
        with Horizontal(id="results-row"):
            yield ResultsPanel(id="results", highlight=True, markup=True)
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        self.log_result("[bold green]QENEX LAB[/] Primary Controller initialized")
        self.log_result("100% Local Operation - No External APIs")
        self.log_result("")
        self.log_result("Shortcuts: Ctrl+R=Run, Ctrl+Enter=Send, F1=Help")
        self._refresh_status()
    
    def log_result(self, message: str) -> None:
        """Log to results panel."""
        self.query_one("#results", ResultsPanel).write(message)
    
    def _refresh_status(self) -> None:
        """Refresh system status."""
        status = self.bridge.get_status()
        self.query_one("#status-panel", StatusPanel).update_status(status)
    
    # ========================================================================
    # Event Handlers
    # ========================================================================
    
    def on_chat_panel_reasoning_requested(self, event: ChatPanel.ReasoningRequested) -> None:
        """Handle reasoning request from chat panel."""
        self.run_reasoning(event.query, event.mode)
    
    def on_script_panel_run_requested(self, event: ScriptPanel.RunRequested) -> None:
        """Handle script run request."""
        self.run_script(event.code)
    
    def on_script_panel_generate_requested(self, event: ScriptPanel.GenerateRequested) -> None:
        """Handle code generation request."""
        self.generate_code(event.description)
    
    # ========================================================================
    # Actions
    # ========================================================================
    
    def action_run_script(self) -> None:
        """Run current script."""
        script_panel = self.query_one("#script-panel", ScriptPanel)
        editor = script_panel.query_one("#script-editor", ScriptEditor)
        code = editor.text.strip()
        if code:
            self.run_script(code)
    
    def action_send_query(self) -> None:
        """Send current query."""
        chat_panel = self.query_one("#chat-panel", ChatPanel)
        chat_panel._send_query()
    
    def action_generate(self) -> None:
        """Generate code."""
        self.generate_code("Write a function to compute molecular energy")
    
    def action_refresh_status(self) -> None:
        """Refresh status."""
        self._refresh_status()
        self.log_result("[dim]Status refreshed[/]")
    
    def action_help(self) -> None:
        """Show help."""
        help_text = """
[bold cyan]QENEX LAB Primary Controller - Help[/]

[bold]Keyboard Shortcuts:[/]
  Ctrl+Q          Quit
  Ctrl+R          Run script
  Ctrl+Enter      Send chat query
  Ctrl+G          Generate code
  F1              This help
  F5              Refresh status
  Ctrl+V          Paste from clipboard

[bold]Chat Panel (Scout Reasoning):[/]
  Type a scientific question and press Send or Ctrl+Enter.
  Modes: Reason, Synthesize, Prove, Hypothesize

[bold]Script Panel (Q-Lang / Python):[/]
  Write or paste code, then click Run or press Ctrl+R.
  Use Generate to create code with DeepSeek.

[bold]Molecule Panel:[/]
  Enter XYZ format coordinates for visualization.
  Example:
    O  0.0  0.0  0.0
    H  0.0  0.757  0.587
    H  0.0  -0.757  0.587

[bold]100% Local Operation[/] - All LLM calls use local backends only.
"""
        self.log_result(help_text)
    
    # ========================================================================
    # Workers
    # ========================================================================
    
    @work(thread=True)
    def run_reasoning(self, query: str, mode: str) -> None:
        """Execute Scout reasoning in background."""
        chat = self.query_one("#chat-panel", ChatPanel)
        
        self.call_from_thread(chat.log, f"\n[bold]You ({mode}):[/] {query}")
        self.call_from_thread(chat.log, "[dim]Scout reasoning...[/]")
        
        start = time.time()
        result = self.bridge.reason(query, mode)
        elapsed = time.time() - start
        
        if result["success"]:
            self.call_from_thread(
                chat.log,
                f"\n[bold cyan]Scout:[/]\n{result['output']}"
            )
            self.call_from_thread(
                chat.log,
                f"\n[dim]({elapsed:.1f}s | {result.get('tokens_generated', 0)} tokens | {result.get('backend', 'unknown')})[/]"
            )
        else:
            self.call_from_thread(
                chat.log,
                f"\n[red]Error:[/] {result.get('output', 'Unknown error')}"
            )
        
        self.call_from_thread(self._refresh_status)
    
    @work(thread=True)
    def run_script(self, code: str) -> None:
        """Execute Q-Lang script in background."""
        self.call_from_thread(self.log_result, "\n[bold]Executing script...[/]")
        
        start = time.time()
        result = self.bridge.run_script(code)
        elapsed = time.time() - start
        
        if result["success"]:
            output = result.get("output", "")
            if output:
                for line in output.split('\n'):
                    self.call_from_thread(self.log_result, line)
            
            if result.get("result") is not None:
                self.call_from_thread(
                    self.log_result,
                    f"\n[green]Result:[/] {result['result']}"
                )
            
            self.call_from_thread(
                self.log_result,
                f"\n[green]Script completed in {elapsed:.2f}s[/]"
            )
        else:
            self.call_from_thread(
                self.log_result,
                f"\n[red]Error:[/] {result.get('error', 'Unknown error')}"
            )
            if result.get("traceback"):
                self.call_from_thread(
                    self.log_result,
                    f"[dim]{result['traceback']}[/]"
                )
        
        self.call_from_thread(self._refresh_status)
    
    @work(thread=True)
    def generate_code(self, description: str) -> None:
        """Generate code with DeepSeek in background."""
        self.call_from_thread(
            self.log_result,
            f"\n[bold]Generating code...[/]\nDescription: {description}"
        )
        
        start = time.time()
        result = self.bridge.generate_code(description)
        elapsed = time.time() - start
        
        if result["success"]:
            code = result.get("code", "")
            
            # Update script panel
            script_panel = self.query_one("#script-panel", ScriptPanel)
            self.call_from_thread(script_panel.set_code, code)
            
            self.call_from_thread(
                self.log_result,
                f"\n[green]Code generated in {elapsed:.2f}s ({result.get('tokens_generated', 0)} tokens)[/]"
            )
        else:
            self.call_from_thread(
                self.log_result,
                f"\n[red]Error:[/] {result.get('raw_response', 'Unknown error')}"
            )
        
        self.call_from_thread(self._refresh_status)


# ============================================================================
# Entry Point
# ============================================================================

def main():
    """Launch QENEX LAB Primary Controller."""
    app = QENEXController()
    app.run()


if __name__ == "__main__":
    main()
