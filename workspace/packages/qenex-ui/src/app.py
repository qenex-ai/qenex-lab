"""
QENEX LAB Terminal UI - Main Application

A comprehensive terminal interface for quantum chemistry simulations.
Features:
- Molecular geometry input with clipboard support (Ctrl+V, Shift+Insert)
- Python/Q-Lang script editor
- Real-time SCF computation
- Results visualization
- Scout monitoring dashboard
"""

from __future__ import annotations

import sys
import os
import io
import traceback
from datetime import datetime
from contextlib import redirect_stdout, redirect_stderr

# Add QENEX packages to path
_WORKSPACE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(_WORKSPACE, 'packages', 'qenex_chem', 'src'))
sys.path.insert(0, os.path.join(_WORKSPACE, 'packages', 'qenex-qlang', 'src'))

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import (
    Header, Footer, Static, Button, Label, 
    TabbedContent, TabPane, DataTable, Log,
    LoadingIndicator, ProgressBar, RichLog
)
from textual.reactive import reactive
from textual import work
from textual.worker import Worker, get_current_worker

from rich.syntax import Syntax
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

import pyperclip

# Import custom widgets
from widgets import ClipboardInput, ClipboardTextArea, MoleculeInput, ScriptEditor


# Example molecules for quick testing
EXAMPLE_MOLECULES = {
    "H2": """H  0.0  0.0  0.0
H  0.0  0.0  1.4""",
    "Water": """O  0.0  0.0  0.0
H  0.0  0.757  0.587
H  0.0  -0.757  0.587""",
    "Methane": """C  0.0  0.0  0.0
H  0.629  0.629  0.629
H  -0.629  -0.629  0.629
H  -0.629  0.629  -0.629
H  0.629  -0.629  -0.629""",
    "Ammonia": """N  0.0  0.0  0.0
H  0.0  0.942  0.335
H  0.816  -0.471  0.335
H  -0.816  -0.471  0.335""",
}

EXAMPLE_SCRIPT = '''"""
QENEX LAB - Example Hartree-Fock Calculation
Paste your own Python script here or use this template.
Press Ctrl+V or Shift+Insert to paste from clipboard.
"""
from molecule import Molecule
from solver import HartreeFockSolver

# Define molecular geometry (paste XYZ coordinates)
mol = Molecule([
    ('H', (0.0, 0.0, 0.0)),
    ('H', (0.0, 0.0, 1.4))
])

# Create solver and compute energy
solver = HartreeFockSolver()
E_elec, E_tot = solver.compute_energy(mol, verbose=True)

print(f"\\nResults:")
print(f"  Electronic Energy: {E_elec:.8f} Ha")
print(f"  Total Energy:      {E_tot:.8f} Ha")
'''


class StatusBar(Static):
    """Status bar showing current operation state."""
    
    status = reactive("Ready")
    
    def render(self) -> Text:
        return Text(f" Status: {self.status}", style="bold white on dark_blue")


class ScoutPanel(Static):
    """Panel displaying Scout monitoring statistics."""
    
    DEFAULT_CSS = """
    ScoutPanel {
        height: auto;
        padding: 1;
        background: $surface;
        border: solid $primary;
    }
    """
    
    def compose(self) -> ComposeResult:
        yield Static(id="scout-stats")
    
    def update_stats(self, stats: dict) -> None:
        """Update the displayed statistics."""
        table = Table(title="Scout Monitor", box=None, expand=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in stats.items():
            table.add_row(key.replace('_', ' ').title(), str(value))
        
        self.query_one("#scout-stats", Static).update(table)


class ResultsPanel(RichLog):
    """Panel for displaying computation results."""
    
    DEFAULT_CSS = """
    ResultsPanel {
        height: 100%;
        border: solid $success;
        background: $surface;
    }
    """


class QenexLabApp(App):
    """
    QENEX LAB Terminal UI Application
    
    A comprehensive interface for quantum chemistry simulations with
    full clipboard support for pasting molecular geometries and scripts.
    """
    
    TITLE = "QENEX LAB"
    SUB_TITLE = "Quantum Chemistry Laboratory"
    
    CSS = """
    Screen {
        layout: grid;
        grid-size: 2 2;
        grid-rows: 1fr 1fr;
        grid-columns: 1fr 1fr;
    }
    
    #left-panel {
        row-span: 2;
        height: 100%;
        border: solid $primary;
    }
    
    #molecule-section {
        height: auto;
        max-height: 40%;
        padding: 1;
    }
    
    #script-section {
        height: 1fr;
        padding: 1;
    }
    
    #results-panel {
        height: 100%;
        border: solid $success;
    }
    
    #scout-section {
        height: 100%;
        border: solid $warning;
    }
    
    .section-title {
        text-style: bold;
        color: $text;
        padding: 0 1;
        background: $primary;
    }
    
    .button-row {
        height: auto;
        padding: 1 0;
    }
    
    Button {
        margin: 0 1;
    }
    
    #example-buttons {
        height: auto;
        layout: horizontal;
    }
    
    .example-btn {
        min-width: 10;
    }
    """
    
    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+r", "run_calculation", "Run"),
        Binding("ctrl+e", "run_script", "Execute Script"),
        Binding("ctrl+l", "clear_results", "Clear"),
        Binding("f1", "show_help", "Help"),
        Binding("ctrl+1", "focus_molecule", "Focus Molecule"),
        Binding("ctrl+2", "focus_script", "Focus Script"),
    ]
    
    def __init__(self):
        super().__init__()
        self.solver = None
        self.current_molecule = None
    
    def compose(self) -> ComposeResult:
        yield Header()
        
        with Vertical(id="left-panel"):
            # Molecule Input Section
            with Vertical(id="molecule-section"):
                yield Label("Molecular Geometry (XYZ Format)", classes="section-title")
                yield Label("Paste with Ctrl+V or Shift+Insert", classes="hint")
                yield MoleculeInput(
                    EXAMPLE_MOLECULES["H2"],
                    id="molecule-input"
                )
                with Horizontal(id="example-buttons", classes="button-row"):
                    yield Button("H2", id="btn-h2", classes="example-btn", variant="default")
                    yield Button("H2O", id="btn-water", classes="example-btn", variant="default")
                    yield Button("CH4", id="btn-methane", classes="example-btn", variant="default")
                    yield Button("NH3", id="btn-ammonia", classes="example-btn", variant="default")
                with Horizontal(classes="button-row"):
                    yield Button("Run HF", id="btn-run", variant="success")
                    yield Button("Optimize", id="btn-optimize", variant="primary")
                    yield Button("Clear", id="btn-clear", variant="error")
            
            # Script Editor Section
            with Vertical(id="script-section"):
                yield Label("Python Script Editor", classes="section-title")
                yield ScriptEditor(
                    EXAMPLE_SCRIPT,
                    id="script-editor"
                )
                with Horizontal(classes="button-row"):
                    yield Button("Execute", id="btn-execute", variant="success")
                    yield Button("Load Example", id="btn-load-example", variant="default")
        
        # Results Panel
        with Vertical(id="results-panel"):
            yield Label("Computation Results", classes="section-title")
            yield ResultsPanel(id="results-log", highlight=True, markup=True)
        
        # Scout Monitoring Section
        with Vertical(id="scout-section"):
            yield Label("Scout Monitor", classes="section-title")
            yield ScoutPanel(id="scout-panel")
            yield Static(id="system-info")
        
        yield Footer()
    
    def on_mount(self) -> None:
        """Initialize the application on mount."""
        self.log_message("[bold green]QENEX LAB[/] initialized")
        self.log_message("Paste molecular geometries with [bold]Ctrl+V[/] or [bold]Shift+Insert[/]")
        self.log_message("")
        self.update_scout_stats()
        self.update_system_info()
    
    def log_message(self, message: str) -> None:
        """Log a message to the results panel."""
        log = self.query_one("#results-log", ResultsPanel)
        log.write(message)
    
    def update_scout_stats(self) -> None:
        """Update Scout monitoring statistics."""
        try:
            import qenex_accelerate as qa
            stats = qa.scout_report()
            self.query_one("#scout-panel", ScoutPanel).update_stats(stats)
        except ImportError:
            self.query_one("#scout-panel", ScoutPanel).update_stats({
                "status": "Rust backend not available"
            })
    
    def update_system_info(self) -> None:
        """Update system information display."""
        info_text = Text()
        info_text.append("\nSystem Information\n", style="bold cyan")
        info_text.append(f"Python: {sys.version.split()[0]}\n", style="dim")
        
        try:
            import numpy as np
            info_text.append(f"NumPy: {np.__version__}\n", style="dim")
        except:
            pass
        
        try:
            import qenex_accelerate as qa
            info_text.append("Rust Backend: Available\n", style="green")
            info_text.append(f"Threads: {qa.scout_get_num_threads()}\n", style="dim")
        except:
            info_text.append("Rust Backend: Not Available\n", style="red")
        
        self.query_one("#system-info", Static).update(info_text)
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button press events."""
        button_id = event.button.id
        
        if button_id == "btn-h2":
            self.set_molecule("H2")
        elif button_id == "btn-water":
            self.set_molecule("Water")
        elif button_id == "btn-methane":
            self.set_molecule("Methane")
        elif button_id == "btn-ammonia":
            self.set_molecule("Ammonia")
        elif button_id == "btn-run":
            self.run_calculation()
        elif button_id == "btn-optimize":
            self.run_optimization()
        elif button_id == "btn-clear":
            self.clear_molecule_input()
        elif button_id == "btn-execute":
            self.run_script()
        elif button_id == "btn-load-example":
            self.load_example_script()
    
    def set_molecule(self, name: str) -> None:
        """Set the molecule input to a predefined example."""
        if name in EXAMPLE_MOLECULES:
            mol_input = self.query_one("#molecule-input", MoleculeInput)
            mol_input.clear()
            mol_input.insert(EXAMPLE_MOLECULES[name])
            self.log_message(f"Loaded [bold]{name}[/] molecule")
    
    def clear_molecule_input(self) -> None:
        """Clear the molecule input field."""
        mol_input = self.query_one("#molecule-input", MoleculeInput)
        mol_input.clear()
        self.log_message("Cleared molecule input")
    
    def load_example_script(self) -> None:
        """Load the example script into the editor."""
        editor = self.query_one("#script-editor", ScriptEditor)
        editor.clear()
        editor.insert(EXAMPLE_SCRIPT)
        self.log_message("Loaded example script")
    
    def action_clear_results(self) -> None:
        """Clear the results log."""
        log = self.query_one("#results-log", ResultsPanel)
        log.clear()
        self.log_message("[dim]Results cleared[/]")
    
    def action_focus_molecule(self) -> None:
        """Focus the molecule input."""
        self.query_one("#molecule-input").focus()
    
    def action_focus_script(self) -> None:
        """Focus the script editor."""
        self.query_one("#script-editor").focus()
    
    def action_run_calculation(self) -> None:
        """Run the HF calculation (keyboard shortcut)."""
        self.run_calculation()
    
    def action_run_script(self) -> None:
        """Execute the script (keyboard shortcut)."""
        self.run_script()
    
    def action_show_help(self) -> None:
        """Show help information."""
        help_text = """
[bold cyan]QENEX LAB Help[/]

[bold]Keyboard Shortcuts:[/]
  Ctrl+V / Shift+Insert  Paste from clipboard
  Ctrl+C                 Copy to clipboard
  Ctrl+R                 Run HF calculation
  Ctrl+E                 Execute script
  Ctrl+L                 Clear results
  Ctrl+1                 Focus molecule input
  Ctrl+2                 Focus script editor
  Ctrl+Q                 Quit

[bold]Molecule Input:[/]
  Enter XYZ format coordinates:
  Element  X  Y  Z
  
  Example:
  H  0.0  0.0  0.0
  H  0.0  0.0  1.4

[bold]Script Editor:[/]
  Write Python code using QENEX modules.
  Available imports:
  - molecule.Molecule
  - solver.HartreeFockSolver
  - optimizer.GeometryOptimizer
"""
        self.log_message(help_text)
    
    @work(thread=True)
    def run_calculation(self) -> None:
        """Run Hartree-Fock calculation on the input molecule."""
        worker = get_current_worker()
        
        try:
            # Get molecule data
            mol_input = self.query_one("#molecule-input", MoleculeInput)
            atoms = mol_input.get_xyz_data()
            
            if not atoms:
                self.call_from_thread(
                    self.log_message,
                    "[red]Error:[/] Invalid or empty molecular geometry"
                )
                return
            
            self.call_from_thread(
                self.log_message,
                f"\n[bold]Starting Hartree-Fock calculation...[/]"
            )
            self.call_from_thread(
                self.log_message,
                f"Atoms: {len(atoms)}"
            )
            
            # Import and create molecule
            from molecule import Molecule
            from solver import HartreeFockSolver
            
            mol = Molecule(atoms)
            solver = HartreeFockSolver()
            
            # Capture output
            output = io.StringIO()
            with redirect_stdout(output):
                E_elec, E_tot = solver.compute_energy(mol, verbose=True)
            
            # Log captured output
            captured = output.getvalue()
            if captured:
                for line in captured.split('\n'):
                    if line.strip():
                        self.call_from_thread(self.log_message, f"[dim]{line}[/]")
            
            # Log results
            self.call_from_thread(
                self.log_message,
                f"\n[bold green]Calculation Complete![/]"
            )
            self.call_from_thread(
                self.log_message,
                f"[cyan]Electronic Energy:[/] {E_elec:.10f} Ha"
            )
            self.call_from_thread(
                self.log_message,
                f"[cyan]Total Energy:[/]      {E_tot:.10f} Ha"
            )
            
            # Update Scout stats
            self.call_from_thread(self.update_scout_stats)
            
        except Exception as e:
            self.call_from_thread(
                self.log_message,
                f"[red]Error:[/] {str(e)}"
            )
            self.call_from_thread(
                self.log_message,
                f"[dim]{traceback.format_exc()}[/]"
            )
    
    @work(thread=True)
    def run_optimization(self) -> None:
        """Run geometry optimization on the input molecule."""
        worker = get_current_worker()
        
        try:
            mol_input = self.query_one("#molecule-input", MoleculeInput)
            atoms = mol_input.get_xyz_data()
            
            if not atoms:
                self.call_from_thread(
                    self.log_message,
                    "[red]Error:[/] Invalid or empty molecular geometry"
                )
                return
            
            self.call_from_thread(
                self.log_message,
                f"\n[bold]Starting Geometry Optimization...[/]"
            )
            
            from molecule import Molecule
            from solver import HartreeFockSolver
            from optimizer import GeometryOptimizer
            
            mol = Molecule(atoms)
            solver = HartreeFockSolver()
            opt = GeometryOptimizer(solver)
            
            # Capture output
            output = io.StringIO()
            with redirect_stdout(output):
                opt_mol, history = opt.optimize(mol, max_steps=10)
            
            captured = output.getvalue()
            if captured:
                for line in captured.split('\n'):
                    if line.strip():
                        self.call_from_thread(self.log_message, f"[dim]{line}[/]")
            
            self.call_from_thread(
                self.log_message,
                f"\n[bold green]Optimization Complete![/]"
            )
            self.call_from_thread(
                self.log_message,
                f"Steps: {len(history)}"
            )
            self.call_from_thread(
                self.log_message,
                f"Final Energy: {history[-1]:.10f} Ha"
            )
            
            # Update molecule input with optimized geometry
            new_xyz = "\n".join([
                f"{el}  {pos[0]:.6f}  {pos[1]:.6f}  {pos[2]:.6f}"
                for el, pos in opt_mol.atoms
            ])
            
            def update_input():
                mol_input = self.query_one("#molecule-input", MoleculeInput)
                mol_input.clear()
                mol_input.insert(new_xyz)
            
            self.call_from_thread(update_input)
            self.call_from_thread(self.update_scout_stats)
            
        except Exception as e:
            self.call_from_thread(
                self.log_message,
                f"[red]Error:[/] {str(e)}"
            )
    
    @work(thread=True)
    def run_script(self) -> None:
        """Execute the Python script in the editor."""
        worker = get_current_worker()
        
        try:
            editor = self.query_one("#script-editor", ScriptEditor)
            script = editor.text
            
            if not script.strip():
                self.call_from_thread(
                    self.log_message,
                    "[red]Error:[/] Empty script"
                )
                return
            
            self.call_from_thread(
                self.log_message,
                f"\n[bold]Executing script...[/]"
            )
            
            # Create execution namespace
            namespace = {
                '__name__': '__main__',
                '__file__': '<script>',
            }
            
            # Capture output
            output = io.StringIO()
            error_output = io.StringIO()
            
            with redirect_stdout(output), redirect_stderr(error_output):
                exec(script, namespace)
            
            # Log captured output
            stdout_text = output.getvalue()
            stderr_text = error_output.getvalue()
            
            if stdout_text:
                for line in stdout_text.split('\n'):
                    self.call_from_thread(self.log_message, line)
            
            if stderr_text:
                for line in stderr_text.split('\n'):
                    self.call_from_thread(self.log_message, f"[yellow]{line}[/]")
            
            self.call_from_thread(
                self.log_message,
                f"\n[bold green]Script execution complete![/]"
            )
            self.call_from_thread(self.update_scout_stats)
            
        except Exception as e:
            self.call_from_thread(
                self.log_message,
                f"[red]Script Error:[/] {str(e)}"
            )
            self.call_from_thread(
                self.log_message,
                f"[dim]{traceback.format_exc()}[/]"
            )


def main():
    """Entry point for QENEX LAB TUI."""
    app = QenexLabApp()
    app.run()


if __name__ == "__main__":
    main()
