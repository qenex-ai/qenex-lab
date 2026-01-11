#!/usr/bin/env python3
"""
QENEX - Unified Scientific Intelligence Laboratory CLI
=======================================================
The main entry point for the QENEX Lab ecosystem.

100% Local Operation - No External API Dependencies

Usage:
    qenex                           # Start interactive REPL
    qenex reason "query"            # Scientific reasoning with Scout
    qenex generate "description"    # Code generation with DeepSeek
    qenex experiment start <name>   # Start a new experiment
    qenex dashboard                 # Launch web dashboard
    qenex run <script.ql>           # Run Q-Lang script
    qenex status                    # Show system status
    qenex help                      # Show help

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                        QENEX CLI                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │                                                                   │
    │   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐            │
    │   │ reason  │  │generate │  │  expt   │  │  run    │            │
    │   │ (Scout) │  │(DeepSeek│  │(Orchestr│  │ (Q-Lang)│            │
    │   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘            │
    │        │            │            │            │                   │
    │        └────────────┴────────────┴────────────┘                   │
    │                          │                                        │
    │                   ┌──────▼──────┐                                │
    │                   │ Integration │                                │
    │                   │    Layer    │                                │
    │                   └──────┬──────┘                                │
    │                          │                                        │
    │              ┌───────────┼───────────┐                           │
    │              ▼           ▼           ▼                           │
    │          Ollama     llama.cpp      vLLM                          │
    │          (local)     (local)      (local)                        │
    └──────────────────────────────────────────────────────────────────┘

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import argparse
import time
import json
import readline
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add package paths
SCRIPT_DIR = Path(__file__).parent
PACKAGES_DIR = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PACKAGES_DIR / "qenex_chem" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-bio" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-physics" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-math" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-astro" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-neuro" / "src"))
sys.path.insert(0, str(PACKAGES_DIR / "qenex-climate" / "src"))

# Version
__version__ = "0.5.0"

# ANSI colors for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def color(text: str, c: str) -> str:
    """Apply color to text if terminal supports it."""
    if sys.stdout.isatty():
        return f"{c}{text}{Colors.ENDC}"
    return text


# ==============================================================================
# Component Imports (Lazy Loading)
# ==============================================================================

_integration = None
_interpreter = None
_orchestrator = None
_dashboard = None
_research = None
_collaboration = None


def get_integration():
    """Lazy load LLM integration."""
    global _integration
    if _integration is None:
        try:
            from llm_integration import QENEXIntegration, IntegrationConfig
            from llm_backend import BackendType
            
            config = IntegrationConfig(
                scout_backend=BackendType.MOCK,  # Default to mock for safety
                deepseek_backend=BackendType.MOCK,
                context_store_path=str(Path.home() / ".qenex" / "contexts"),
            )
            _integration = QENEXIntegration(config=config, verbose=False)
            
            # Register mock backend by default
            from llm_backend import MockBackend
            _integration.router.register_backend(MockBackend())
            
        except ImportError as e:
            print(color(f"Warning: LLM integration not available: {e}", Colors.YELLOW))
            _integration = None
    return _integration


def get_interpreter():
    """Lazy load Q-Lang interpreter."""
    global _interpreter
    if _interpreter is None:
        try:
            from interpreter import QLangInterpreter
            _interpreter = QLangInterpreter()
        except ImportError as e:
            print(color(f"Warning: Q-Lang interpreter not available: {e}", Colors.YELLOW))
            _interpreter = None
    return _interpreter


def get_orchestrator():
    """Lazy load experiment orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        try:
            from orchestrator import ExperimentOrchestrator
            _orchestrator = ExperimentOrchestrator(verbose=False)
        except ImportError as e:
            print(color(f"Warning: Orchestrator not available: {e}", Colors.YELLOW))
            _orchestrator = None
    return _orchestrator


def get_research():
    """Lazy load research engine."""
    global _research
    if _research is None:
        try:
            from research import ResearchEngine
            _research = ResearchEngine(verbose=False)
        except ImportError as e:
            print(color(f"Warning: Research engine not available: {e}", Colors.YELLOW))
            _research = None
    return _research


# ==============================================================================
# Command Handlers
# ==============================================================================

def cmd_reason(args):
    """Execute scientific reasoning with Scout."""
    integration = get_integration()
    if not integration:
        print(color("Error: LLM integration not available", Colors.RED))
        return 1
    
    query = " ".join(args.query)
    mode = args.mode.upper() if args.mode else "REASON"
    
    print(color(f"\n🧠 Scout Reasoning [{mode}]", Colors.CYAN))
    print(color("=" * 60, Colors.DIM))
    print(color(f"Query: {query}", Colors.DIM))
    print(color("=" * 60, Colors.DIM))
    
    start = time.time()
    result = integration.reason(query, mode=mode)
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"\n{result['output']}")
        print(color(f"\n─── {elapsed:.2f}s | {result['tokens_generated']} tokens | {result['backend']} ───", Colors.DIM))
    else:
        print(color(f"Error: {result.get('output', 'Unknown error')}", Colors.RED))
        return 1
    
    return 0


def cmd_generate(args):
    """Generate code with DeepSeek."""
    integration = get_integration()
    if not integration:
        print(color("Error: LLM integration not available", Colors.RED))
        return 1
    
    description = " ".join(args.description)
    language = args.language or "python"
    mode = args.mode.upper() if args.mode else "GENERATE"
    
    print(color(f"\n💻 DeepSeek Code Generation [{language}]", Colors.CYAN))
    print(color("=" * 60, Colors.DIM))
    print(color(f"Task: {description}", Colors.DIM))
    print(color("=" * 60, Colors.DIM))
    
    start = time.time()
    result = integration.generate_code(description, language=language, mode=mode)
    elapsed = time.time() - start
    
    if result["success"]:
        print(f"\n```{language}")
        print(result['code'])
        print("```")
        print(color(f"\n─── {elapsed:.2f}s | {result['tokens_generated']} tokens | {result['backend']} ───", Colors.DIM))
        
        # Optionally save to file
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result['code'])
            print(color(f"✓ Saved to {args.output}", Colors.GREEN))
    else:
        print(color(f"Error: {result.get('raw_response', 'Unknown error')}", Colors.RED))
        return 1
    
    return 0


def cmd_experiment(args):
    """Manage experiments."""
    orchestrator = get_orchestrator()
    if not orchestrator:
        print(color("Error: Orchestrator not available", Colors.RED))
        return 1
    
    subcmd = args.subcmd
    
    if subcmd == "start":
        name = args.name or f"experiment_{int(time.time())}"
        print(color(f"\n🔬 Starting experiment: {name}", Colors.CYAN))
        orchestrator.create_experiment(name, description=args.description or "")
        print(color(f"✓ Experiment '{name}' created", Colors.GREEN))
        
    elif subcmd == "list":
        print(color("\n📋 Experiments", Colors.CYAN))
        experiments = orchestrator.list_experiments()
        if experiments:
            for exp in experiments:
                status_color = Colors.GREEN if exp.get("status") == "completed" else Colors.YELLOW
                print(f"  • {exp['name']} [{color(exp.get('status', 'unknown'), status_color)}]")
        else:
            print(color("  No experiments found", Colors.DIM))
            
    elif subcmd == "status":
        if args.name:
            exp = orchestrator.get_experiment(args.name)
            if exp:
                print(color(f"\n🔬 Experiment: {exp['name']}", Colors.CYAN))
                print(f"  Status: {exp.get('status', 'unknown')}")
                print(f"  Created: {exp.get('created', 'unknown')}")
                print(f"  Steps: {len(exp.get('steps', []))}")
            else:
                print(color(f"Experiment '{args.name}' not found", Colors.RED))
        else:
            # Show current experiment
            current = orchestrator.current_experiment
            if current:
                print(color(f"\n🔬 Current: {current['name']}", Colors.CYAN))
            else:
                print(color("No active experiment", Colors.DIM))
    
    elif subcmd == "run":
        if not args.name:
            print(color("Error: Experiment name required", Colors.RED))
            return 1
        print(color(f"\n▶️  Running experiment: {args.name}", Colors.CYAN))
        result = orchestrator.run_experiment(args.name)
        if result.get("success"):
            print(color("✓ Experiment completed successfully", Colors.GREEN))
        else:
            print(color(f"✗ Experiment failed: {result.get('error', 'unknown')}", Colors.RED))
            
    else:
        print(color(f"Unknown subcommand: {subcmd}", Colors.RED))
        return 1
    
    return 0


def cmd_run(args):
    """Run a Q-Lang script."""
    interpreter = get_interpreter()
    if not interpreter:
        print(color("Error: Q-Lang interpreter not available", Colors.RED))
        return 1
    
    script_path = Path(args.script)
    if not script_path.exists():
        print(color(f"Error: Script not found: {script_path}", Colors.RED))
        return 1
    
    print(color(f"\n📜 Running: {script_path.name}", Colors.CYAN))
    print(color("=" * 60, Colors.DIM))
    
    try:
        with open(script_path) as f:
            code = f.read()
        
        start = time.time()
        result = interpreter.execute(code)
        elapsed = time.time() - start
        
        if result is not None:
            print(f"\nResult: {result}")
        
        print(color(f"\n─── Completed in {elapsed:.3f}s ───", Colors.DIM))
        return 0
        
    except Exception as e:
        print(color(f"Error: {e}", Colors.RED))
        return 1


def cmd_dashboard(args):
    """Launch the web dashboard."""
    try:
        from dashboard import create_app
        
        host = args.host or "127.0.0.1"
        port = args.port or 8050
        
        print(color(f"\n🌐 Starting QENEX Dashboard", Colors.CYAN))
        print(color(f"   URL: http://{host}:{port}", Colors.GREEN))
        print(color("   Press Ctrl+C to stop\n", Colors.DIM))
        
        app = create_app()
        app.run_server(host=host, port=port, debug=args.debug)
        
    except ImportError as e:
        print(color(f"Error: Dashboard not available: {e}", Colors.RED))
        print(color("Install with: pip install dash plotly", Colors.DIM))
        return 1
    except KeyboardInterrupt:
        print(color("\n\nDashboard stopped", Colors.DIM))
        return 0


def cmd_status(args):
    """Show system status."""
    print(color("\n" + "=" * 60, Colors.CYAN))
    print(color("           QENEX Scientific Laboratory", Colors.BOLD))
    print(color(f"                   v{__version__}", Colors.DIM))
    print(color("=" * 60, Colors.CYAN))
    
    # Check components
    components = [
        ("Q-Lang Interpreter", get_interpreter),
        ("LLM Integration", get_integration),
        ("Experiment Orchestrator", get_orchestrator),
        ("Research Engine", get_research),
    ]
    
    print(color("\n📦 Components:", Colors.BOLD))
    for name, loader in components:
        try:
            obj = loader()
            if obj:
                print(f"  {color('✓', Colors.GREEN)} {name}")
            else:
                print(f"  {color('○', Colors.YELLOW)} {name} (not loaded)")
        except Exception as e:
            print(f"  {color('✗', Colors.RED)} {name} ({e})")
    
    # Check LLM backends
    integration = get_integration()
    if integration:
        print(color("\n🔌 LLM Backends:", Colors.BOLD))
        for backend in integration.router.get_available_backends():
            available = backend.is_available()
            status = color("●", Colors.GREEN) if available else color("○", Colors.RED)
            print(f"  {status} {backend.name} ({backend.backend_type.value})")
        
        print(color("\n📊 Token Usage:", Colors.BOLD))
        tokens = integration.total_tokens.to_dict()
        print(f"  Prompt: {tokens['prompt_tokens']:,}")
        print(f"  Completion: {tokens['completion_tokens']:,}")
        print(f"  Total: {tokens['total_tokens']:,}")
        print(f"  Operations: {tokens['operations']}")
    
    # Check scientific packages
    print(color("\n🔬 Scientific Packages:", Colors.BOLD))
    packages = [
        ("qenex_chem", "Quantum Chemistry"),
        ("qenex-bio", "Biological Systems"),
        ("qenex-physics", "Physics Simulation"),
        ("qenex-math", "Formal Mathematics"),
        ("qenex-astro", "Astrophysics"),
        ("qenex-neuro", "Neuroscience"),
        ("qenex-climate", "Climate Science"),
    ]
    
    for pkg_name, description in packages:
        pkg_path = PACKAGES_DIR / pkg_name.replace("-", "_") / "src"
        alt_path = PACKAGES_DIR / pkg_name / "src"
        exists = pkg_path.exists() or alt_path.exists()
        status = color("✓", Colors.GREEN) if exists else color("○", Colors.DIM)
        print(f"  {status} {description}")
    
    print(color("\n" + "=" * 60 + "\n", Colors.DIM))
    return 0


def cmd_config(args):
    """Configure QENEX settings."""
    config_path = Path.home() / ".qenex" / "config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load existing config
    config = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    
    if args.show:
        print(color("\n⚙️  QENEX Configuration", Colors.CYAN))
        print(color(f"   Path: {config_path}", Colors.DIM))
        print()
        if config:
            for key, value in config.items():
                print(f"  {key}: {value}")
        else:
            print(color("  (no configuration set)", Colors.DIM))
    
    elif args.set:
        key, value = args.set.split("=", 1)
        config[key] = value
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(color(f"✓ Set {key} = {value}", Colors.GREEN))
    
    elif args.backend:
        config["default_backend"] = args.backend
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(color(f"✓ Default backend set to: {args.backend}", Colors.GREEN))
    
    else:
        print(color("Use --show, --set key=value, or --backend <name>", Colors.DIM))
    
    return 0


# ==============================================================================
# Interactive REPL
# ==============================================================================

class QENEXRepl:
    """Interactive QENEX REPL."""
    
    BANNER = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   ██████╗ ███████╗███╗   ██╗███████╗██╗  ██╗                ║
    ║  ██╔═══██╗██╔════╝████╗  ██║██╔════╝╚██╗██╔╝                ║
    ║  ██║   ██║█████╗  ██╔██╗ ██║█████╗   ╚███╔╝                 ║
    ║  ██║▄▄ ██║██╔══╝  ██║╚██╗██║██╔══╝   ██╔██╗                 ║
    ║  ╚██████╔╝███████╗██║ ╚████║███████╗██╔╝ ██╗                ║
    ║   ╚══▀▀═╝ ╚══════╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝                ║
    ║                                                              ║
    ║         Scientific Intelligence Laboratory v{version}          ║
    ║              100% Local • No External APIs                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    
    Commands:
      /reason <query>       Scientific reasoning with Scout
      /generate <task>      Code generation with DeepSeek
      /run <script.ql>      Run Q-Lang script
      /experiment <cmd>     Manage experiments
      /status               System status
      /help                 Show help
      /quit                 Exit REPL
    
    Or type Q-Lang code directly.
    """
    
    def __init__(self):
        self.running = True
        self.interpreter = None
        self.integration = None
        self.history_file = Path.home() / ".qenex" / "history"
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Setup readline
        try:
            readline.read_history_file(self.history_file)
        except FileNotFoundError:
            pass
        readline.set_history_length(1000)
    
    def start(self):
        """Start the REPL."""
        print(self.BANNER.format(version=__version__))
        
        # Lazy load components
        print(color("Loading components...", Colors.DIM))
        self.interpreter = get_interpreter()
        self.integration = get_integration()
        print(color("Ready!\n", Colors.GREEN))
        
        while self.running:
            try:
                line = input(color("qenex> ", Colors.CYAN))
                if not line.strip():
                    continue
                
                self.process_line(line.strip())
                
            except KeyboardInterrupt:
                print("\n" + color("(Use /quit to exit)", Colors.DIM))
            except EOFError:
                self.running = False
                print()
        
        # Save history
        readline.write_history_file(self.history_file)
        print(color("Goodbye!", Colors.DIM))
    
    def process_line(self, line: str):
        """Process a REPL input line."""
        if line.startswith("/"):
            self.process_command(line)
        else:
            self.process_qlang(line)
    
    def process_command(self, line: str):
        """Process a slash command."""
        parts = line[1:].split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""
        
        if cmd in ("quit", "exit", "q"):
            self.running = False
            
        elif cmd == "help":
            self.show_help()
            
        elif cmd == "status":
            cmd_status(argparse.Namespace())
            
        elif cmd == "reason":
            if not args:
                print(color("Usage: /reason <query>", Colors.YELLOW))
                return
            if self.integration:
                result = self.integration.reason(args)
                if result["success"]:
                    print(f"\n{result['output']}\n")
                else:
                    print(color(f"Error: {result.get('output')}", Colors.RED))
            else:
                print(color("LLM integration not available", Colors.RED))
                
        elif cmd == "generate":
            if not args:
                print(color("Usage: /generate <description>", Colors.YELLOW))
                return
            if self.integration:
                result = self.integration.generate_code(args)
                if result["success"]:
                    print(f"\n```python\n{result['code']}\n```\n")
                else:
                    print(color(f"Error: {result.get('raw_response')}", Colors.RED))
            else:
                print(color("LLM integration not available", Colors.RED))
                
        elif cmd == "run":
            if not args:
                print(color("Usage: /run <script.ql>", Colors.YELLOW))
                return
            ns = argparse.Namespace(script=args)
            cmd_run(ns)
            
        elif cmd == "experiment":
            exp_parts = args.split(maxsplit=1)
            subcmd = exp_parts[0] if exp_parts else "status"
            name = exp_parts[1] if len(exp_parts) > 1 else None
            ns = argparse.Namespace(subcmd=subcmd, name=name, description=None)
            cmd_experiment(ns)
            
        elif cmd == "clear":
            os.system('clear' if os.name == 'posix' else 'cls')
            
        else:
            print(color(f"Unknown command: /{cmd}", Colors.RED))
            print(color("Type /help for available commands", Colors.DIM))
    
    def process_qlang(self, line: str):
        """Process Q-Lang code."""
        if not self.interpreter:
            print(color("Q-Lang interpreter not available", Colors.RED))
            return
        
        try:
            result = self.interpreter.execute(line)
            if result is not None:
                print(f"→ {result}")
        except Exception as e:
            print(color(f"Error: {e}", Colors.RED))
    
    def show_help(self):
        """Show REPL help."""
        print(color("""
╭──────────────────────────────────────────────────────────────╮
│                      QENEX REPL Help                         │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  COMMANDS (prefix with /)                                    │
│  ─────────────────────────                                   │
│  /reason <query>      Deep scientific reasoning              │
│  /generate <task>     Generate code                          │
│  /run <file.ql>       Execute Q-Lang script                  │
│  /experiment <cmd>    Experiment management                  │
│  /status              Show system status                     │
│  /clear               Clear screen                           │
│  /help                This help                              │
│  /quit                Exit REPL                              │
│                                                              │
│  Q-LANG (type directly)                                      │
│  ──────────────────────                                      │
│  c = 299792458 m/s           # Physical constants            │
│  E = m * c^2                 # Formulas                      │
│  molecule H2O                # Chemistry                     │
│  lattice ising 10x10         # Physics                       │
│  proof start "theorem"       # Mathematics                   │
│                                                              │
│  REASONING MODES                                             │
│  ───────────────                                             │
│  /reason --mode=synthesize   # Cross-reference sources       │
│  /reason --mode=prove        # Construct formal proofs       │
│  /reason --mode=hypothesize  # Generate hypotheses           │
│                                                              │
╰──────────────────────────────────────────────────────────────╯
        """, Colors.DIM))


# ==============================================================================
# Main Entry Point
# ==============================================================================

def create_parser():
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="qenex",
        description="QENEX Scientific Intelligence Laboratory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  qenex                                    Start interactive REPL
  qenex reason "What is dark matter?"      Scientific reasoning
  qenex generate "FFT implementation"      Code generation
  qenex experiment start my_exp            Start experiment
  qenex run simulation.ql                  Run Q-Lang script
  qenex status                             Show system status
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"QENEX v{__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # reason command
    reason_parser = subparsers.add_parser("reason", help="Scientific reasoning with Scout")
    reason_parser.add_argument("query", nargs="+", help="The reasoning query")
    reason_parser.add_argument("--mode", "-m", choices=["reason", "synthesize", "prove", "hypothesize", "verify", "analyze"],
                               help="Reasoning mode")
    
    # generate command
    gen_parser = subparsers.add_parser("generate", help="Code generation with DeepSeek")
    gen_parser.add_argument("description", nargs="+", help="Description of code to generate")
    gen_parser.add_argument("--language", "-l", default="python", help="Target language")
    gen_parser.add_argument("--mode", "-m", choices=["generate", "optimize", "test", "document", "explain"],
                            help="Generation mode")
    gen_parser.add_argument("--output", "-o", help="Output file")
    
    # experiment command
    exp_parser = subparsers.add_parser("experiment", help="Manage experiments")
    exp_parser.add_argument("subcmd", choices=["start", "list", "status", "run", "stop"],
                            help="Experiment subcommand")
    exp_parser.add_argument("name", nargs="?", help="Experiment name")
    exp_parser.add_argument("--description", "-d", help="Experiment description")
    
    # run command
    run_parser = subparsers.add_parser("run", help="Run Q-Lang script")
    run_parser.add_argument("script", help="Q-Lang script file")
    
    # dashboard command
    dash_parser = subparsers.add_parser("dashboard", help="Launch web dashboard")
    dash_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    dash_parser.add_argument("--port", "-p", type=int, default=8050, help="Port number")
    dash_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # status command
    subparsers.add_parser("status", help="Show system status")
    
    # config command
    config_parser = subparsers.add_parser("config", help="Configure settings")
    config_parser.add_argument("--show", action="store_true", help="Show configuration")
    config_parser.add_argument("--set", help="Set key=value")
    config_parser.add_argument("--backend", help="Set default LLM backend")
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Dispatch to command handlers
    if args.command is None:
        # No command - start REPL
        repl = QENEXRepl()
        repl.start()
        return 0
    
    handlers = {
        "reason": cmd_reason,
        "generate": cmd_generate,
        "experiment": cmd_experiment,
        "run": cmd_run,
        "dashboard": cmd_dashboard,
        "status": cmd_status,
        "config": cmd_config,
    }
    
    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
