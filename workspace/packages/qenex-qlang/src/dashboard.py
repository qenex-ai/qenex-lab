"""
QENEX Lab Web Dashboard
=======================

Interactive web interface for the QENEX Scientific Laboratory.

Features:
- Q-Lang REPL with syntax highlighting
- Research paper search and analysis
- Scout 10M scientific reasoning
- DeepSeek code generation
- Experiment orchestrator pipelines
- Visualization of results

Usage:
    python dashboard.py          # Start on default port 7860
    python dashboard.py --port 8080 --share  # Share publicly

Requirements:
    pip install gradio>=4.0.0 plotly pandas

Author: QENEX Sovereign Agent
Version: 1.0.0
"""

import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
from datetime import datetime

# Add package path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try to import gradio
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    gr = None

# Import QENEX modules
try:
    from interpreter import QLangInterpreter
    INTERPRETER_AVAILABLE = True
except ImportError:
    INTERPRETER_AVAILABLE = False
    QLangInterpreter = None

try:
    from research import ResearchEngine, ResearchMode
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False
    ResearchEngine = None

try:
    from scout_10m import ScoutReasoner, ScoutMode
    SCOUT_AVAILABLE = True
except ImportError:
    SCOUT_AVAILABLE = False
    ScoutReasoner = None

try:
    from deepseek import DeepSeekEngine, TargetLanguage
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    DeepSeekEngine = None

try:
    from orchestrator import (
        ExperimentOrchestrator,
        Pipeline,
        create_research_pipeline,
        create_implementation_pipeline,
        create_proof_pipeline,
    )
    ORCHESTRATOR_AVAILABLE = True
except ImportError:
    ORCHESTRATOR_AVAILABLE = False
    ExperimentOrchestrator = None

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# ============================================================================
# Global State
# ============================================================================

class QenexDashboardState:
    """Global state for the dashboard."""
    
    def __init__(self):
        self.interpreter: Optional[QLangInterpreter] = None
        self.research: Optional[ResearchEngine] = None
        self.scout: Optional[ScoutReasoner] = None
        self.deepseek: Optional[DeepSeekEngine] = None
        self.orchestrator: Optional[ExperimentOrchestrator] = None
        self.history: List[Dict[str, Any]] = []
        self.papers: Dict[str, Any] = {}
        
    def init_interpreter(self):
        """Initialize Q-Lang interpreter."""
        if self.interpreter is None and INTERPRETER_AVAILABLE:
            self.interpreter = QLangInterpreter()
        return self.interpreter
    
    def init_research(self):
        """Initialize research engine."""
        if self.research is None and RESEARCH_AVAILABLE:
            self.research = ResearchEngine(verbose=False)
        return self.research
    
    def init_scout(self):
        """Initialize Scout reasoner."""
        if self.scout is None and SCOUT_AVAILABLE:
            self.scout = ScoutReasoner(verbose=False)
        return self.scout
    
    def init_deepseek(self):
        """Initialize DeepSeek engine."""
        if self.deepseek is None and DEEPSEEK_AVAILABLE:
            self.deepseek = DeepSeekEngine(verbose=False)
        return self.deepseek
    
    def init_orchestrator(self):
        """Initialize orchestrator."""
        if self.orchestrator is None and ORCHESTRATOR_AVAILABLE:
            self.orchestrator = ExperimentOrchestrator(verbose=False)
        return self.orchestrator


# Global state instance
_state = QenexDashboardState()


# ============================================================================
# Q-Lang REPL Functions
# ============================================================================

def execute_qlang(code: str, history: str) -> Tuple[str, str]:
    """Execute Q-Lang code and return output."""
    interpreter = _state.init_interpreter()
    if interpreter is None:
        return "Error: Q-Lang interpreter not available", history
    
    # Capture output
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            interpreter.execute(code)
        
        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()
        
        if errors:
            output += f"\n[Errors]\n{errors}"
        
        if not output.strip():
            output = "[OK] Executed successfully"
            
    except Exception as e:
        output = f"[Error] {type(e).__name__}: {str(e)}"
    
    # Update history
    timestamp = datetime.now().strftime("%H:%M:%S")
    new_entry = f"[{timestamp}] >>> {code}\n{output}\n"
    updated_history = history + new_entry if history else new_entry
    
    return output, updated_history


def get_qlang_context() -> str:
    """Get current Q-Lang context variables."""
    interpreter = _state.init_interpreter()
    if interpreter is None:
        return "Interpreter not available"
    
    ctx = interpreter.context
    lines = ["Current Variables:"]
    for key, value in sorted(ctx.items()):
        if not key.startswith('_'):
            val_str = str(value)[:100]
            lines.append(f"  {key}: {val_str}")
    
    return "\n".join(lines)


# ============================================================================
# Research Functions
# ============================================================================

def search_papers(query: str, max_results: int = 5) -> Tuple[str, str]:
    """Search for papers on arXiv."""
    research = _state.init_research()
    if research is None:
        return "Error: Research engine not available", ""
    
    try:
        result = research.search(query, max_results=max_results)
        
        if not result.search_results:
            return f"No papers found for: {query}", ""
        
        # Format results
        lines = [f"Found {len(result.search_results)} papers for '{query}':\n"]
        
        for i, paper in enumerate(result.search_results, 1):
            lines.append(f"{i}. **{paper.title}**")
            lines.append(f"   Authors: {', '.join(paper.authors[:3])}")
            lines.append(f"   ID: {paper.paper_id}")
            lines.append(f"   Abstract: {paper.abstract[:200]}...")
            lines.append("")
        
        # Store for later use
        for paper in result.search_results:
            _state.papers[paper.paper_id] = paper
        
        return "\n".join(lines), result.summary
        
    except Exception as e:
        return f"Error: {str(e)}", ""


def fetch_paper(identifier: str) -> str:
    """Fetch a specific paper."""
    research = _state.init_research()
    if research is None:
        return "Error: Research engine not available"
    
    try:
        result = research.fetch(identifier)
        
        if result.papers:
            paper = result.papers[0]
            _state.papers[paper.paper_id] = paper
            
            return f"""
**{paper.title}**

Authors: {', '.join(paper.authors)}
Date: {paper.date}
ArXiv ID: {paper.arxiv_id or 'N/A'}

Abstract:
{paper.abstract}

Sections: {len(paper.sections)}
Full Text Length: {len(paper.full_text)} characters
"""
        else:
            return f"Paper not found: {identifier}"
            
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Scout Functions  
# ============================================================================

def scout_reason(prompt: str, mode: str = "reason") -> str:
    """Use Scout for scientific reasoning."""
    scout = _state.init_scout()
    if scout is None:
        return "Error: Scout engine not available"
    
    try:
        mode_enum = ScoutMode[mode.upper()] if hasattr(ScoutMode, mode.upper()) else ScoutMode.REASON
        result = scout.reason(prompt, mode=mode_enum)
        
        return f"""
**Scout Response ({mode.upper()})**

{result.output}

---
Tokens: {result.tokens_used} | Time: {result.elapsed_ms:.1f}ms
"""
    except Exception as e:
        return f"Error: {str(e)}"


def load_paper_to_scout(paper_id: str) -> str:
    """Load a paper into Scout's context."""
    scout = _state.init_scout()
    if scout is None:
        return "Error: Scout engine not available"
    
    paper = _state.papers.get(paper_id)
    if paper is None:
        return f"Paper not found: {paper_id}. Search or fetch it first."
    
    try:
        if hasattr(paper, 'full_text') and paper.full_text:
            scout.context.add_chunk(paper.full_text, paper_id, "research_paper")
            return f"Loaded paper {paper_id} into Scout context ({len(paper.full_text)} chars)"
        else:
            return f"Paper {paper_id} has no full text available"
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# DeepSeek Functions
# ============================================================================

def generate_code(prompt: str, language: str = "python") -> str:
    """Generate code using DeepSeek."""
    deepseek = _state.init_deepseek()
    if deepseek is None:
        return "Error: DeepSeek engine not available"
    
    try:
        result = deepseek.generate(prompt, language=language)
        
        return f"""
**Generated Code ({language})**

```{language}
{result.code}
```

---
Tokens: {result.tokens_used} | Time: {result.elapsed_ms:.1f}ms
"""
    except Exception as e:
        return f"Error: {str(e)}"


def list_templates() -> str:
    """List available code templates."""
    deepseek = _state.init_deepseek()
    if deepseek is None:
        return "Error: DeepSeek engine not available"
    
    try:
        templates = deepseek.list_templates()
        lines = ["Available Templates:\n"]
        for name, desc in templates.items():
            lines.append(f"  **{name}**: {desc}")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {str(e)}"


# ============================================================================
# Orchestrator Functions
# ============================================================================

def run_research_pipeline(topic: str, max_papers: int = 5) -> str:
    """Run an automated research pipeline."""
    orchestrator = _state.init_orchestrator()
    if orchestrator is None:
        return "Error: Orchestrator not available"
    
    try:
        pipeline = create_research_pipeline(topic, max_papers=max_papers)
        run = orchestrator.run(pipeline)
        
        lines = [f"**Research Pipeline: {topic}**\n"]
        lines.append(f"Status: {run.status.upper()}")
        lines.append(f"Steps: {len(run.results)}")
        lines.append("")
        
        for name, result in run.results.items():
            status_icon = "✓" if result.status.value == "completed" else "✗"
            lines.append(f"{status_icon} {name}: {result.status.value} ({result.duration_ms:.1f}ms)")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error: {str(e)}"


def run_implementation_pipeline(description: str, language: str = "python") -> str:
    """Run an implementation pipeline."""
    orchestrator = _state.init_orchestrator()
    if orchestrator is None:
        return "Error: Orchestrator not available"
    
    try:
        pipeline = create_implementation_pipeline(description, language=language)
        run = orchestrator.run(pipeline)
        
        lines = [f"**Implementation Pipeline**\n"]
        lines.append(f"Description: {description[:50]}...")
        lines.append(f"Language: {language}")
        lines.append(f"Status: {run.status.upper()}")
        lines.append("")
        
        for name, result in run.results.items():
            status_icon = "✓" if result.status.value == "completed" else "✗"
            lines.append(f"{status_icon} {name}: {result.status.value}")
        
        # Include generated code if available
        if "generate" in run.context:
            gen_result = run.context["generate"]
            if hasattr(gen_result, "code"):
                lines.append(f"\n**Generated Code:**\n```{language}\n{gen_result.code}\n```")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error: {str(e)}"


def get_orchestrator_status() -> str:
    """Get orchestrator status."""
    orchestrator = _state.init_orchestrator()
    if orchestrator is None:
        return "Orchestrator not available"
    
    runs = orchestrator.get_runs()
    
    lines = ["**Orchestrator Status**\n"]
    lines.append(f"Cache: {'enabled' if orchestrator.cache_enabled else 'disabled'}")
    lines.append(f"Total Runs: {len(runs)}")
    lines.append("")
    
    if runs:
        lines.append("Recent Runs:")
        for run in runs[-5:]:
            lines.append(f"  - {run.pipeline.name}: {run.status}")
    
    return "\n".join(lines)


# ============================================================================
# Dashboard UI
# ============================================================================

def get_system_status() -> str:
    """Get overall system status."""
    status_lines = ["**QENEX Lab System Status**\n"]
    
    components = [
        ("Gradio", GRADIO_AVAILABLE),
        ("Q-Lang Interpreter", INTERPRETER_AVAILABLE),
        ("Research Engine", RESEARCH_AVAILABLE),
        ("Scout 10M", SCOUT_AVAILABLE),
        ("DeepSeek", DEEPSEEK_AVAILABLE),
        ("Orchestrator", ORCHESTRATOR_AVAILABLE),
        ("Plotly", PLOTLY_AVAILABLE),
        ("Pandas", PANDAS_AVAILABLE),
    ]
    
    for name, available in components:
        icon = "✓" if available else "✗"
        status_lines.append(f"{icon} {name}")
    
    return "\n".join(status_lines)


def create_dashboard():
    """Create the Gradio dashboard."""
    if not GRADIO_AVAILABLE:
        print("Error: Gradio not installed. Run: pip install gradio>=4.0.0")
        return None
    
    # Custom CSS
    custom_css = """
    .qenex-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .qenex-title {
        color: #00d4ff;
        font-size: 2em;
        font-weight: bold;
    }
    .output-box {
        font-family: 'Fira Code', 'Monaco', monospace;
        font-size: 14px;
    }
    """
    
    with gr.Blocks(
        title="QENEX Scientific Laboratory",
        css=custom_css,
        theme=gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="blue",
        )
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🧬 QENEX Scientific Laboratory
        
        **The Git of the Universe** - Unified Scientific Discovery Engine
        
        Orchestrate research across Physics, Chemistry, Biology, Mathematics, and more.
        """)
        
        with gr.Tabs():
            
            # ===== Q-Lang REPL Tab =====
            with gr.Tab("⚡ Q-Lang REPL"):
                gr.Markdown("Execute Q-Lang commands interactively.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        code_input = gr.Code(
                            label="Q-Lang Code",
                            language="python",
                            lines=10,
                            value="# Enter Q-Lang commands\nprint(\"Hello QENEX!\")\n\n# Try: scout status\n# Try: research search \"quantum computing\"\n# Try: polyglot status"
                        )
                        
                        with gr.Row():
                            run_btn = gr.Button("▶ Execute", variant="primary")
                            clear_btn = gr.Button("🗑 Clear")
                    
                    with gr.Column(scale=1):
                        context_display = gr.Textbox(
                            label="Context Variables",
                            lines=10,
                            interactive=False
                        )
                
                output_display = gr.Textbox(
                    label="Output",
                    lines=8,
                    interactive=False
                )
                
                history_display = gr.Textbox(
                    label="Session History",
                    lines=10,
                    interactive=False
                )
                
                run_btn.click(
                    execute_qlang,
                    inputs=[code_input, history_display],
                    outputs=[output_display, history_display]
                ).then(
                    get_qlang_context,
                    outputs=[context_display]
                )
                
                clear_btn.click(
                    lambda: ("", "", ""),
                    outputs=[output_display, history_display, context_display]
                )
            
            # ===== Research Tab =====
            with gr.Tab("📚 Research"):
                gr.Markdown("Search and analyze scientific papers from arXiv.")
                
                with gr.Row():
                    with gr.Column():
                        search_query = gr.Textbox(
                            label="Search Query",
                            placeholder="e.g., quantum error correction, protein folding",
                            lines=1
                        )
                        max_results = gr.Slider(
                            minimum=1, maximum=20, value=5, step=1,
                            label="Max Results"
                        )
                        search_btn = gr.Button("🔍 Search", variant="primary")
                    
                    with gr.Column():
                        paper_id_input = gr.Textbox(
                            label="Paper ID",
                            placeholder="e.g., arxiv:2301.12345",
                            lines=1
                        )
                        fetch_btn = gr.Button("📄 Fetch Paper")
                
                search_results = gr.Markdown(label="Search Results")
                search_summary = gr.Textbox(label="Summary", lines=2)
                
                paper_details = gr.Markdown(label="Paper Details")
                
                search_btn.click(
                    search_papers,
                    inputs=[search_query, max_results],
                    outputs=[search_results, search_summary]
                )
                
                fetch_btn.click(
                    fetch_paper,
                    inputs=[paper_id_input],
                    outputs=[paper_details]
                )
            
            # ===== Scout Tab =====
            with gr.Tab("🧠 Scout 10M"):
                gr.Markdown("Scientific reasoning with 10M token context window.")
                
                with gr.Row():
                    with gr.Column(scale=2):
                        scout_prompt = gr.Textbox(
                            label="Reasoning Prompt",
                            placeholder="Enter your scientific question...",
                            lines=5
                        )
                        
                        scout_mode = gr.Dropdown(
                            choices=["reason", "hypothesize", "prove", "synthesize", "critique"],
                            value="reason",
                            label="Reasoning Mode"
                        )
                        
                        scout_btn = gr.Button("🧠 Reason", variant="primary")
                    
                    with gr.Column(scale=1):
                        load_paper_id = gr.Textbox(
                            label="Load Paper to Context",
                            placeholder="Paper ID from search"
                        )
                        load_paper_btn = gr.Button("📥 Load Paper")
                        load_status = gr.Textbox(label="Load Status", lines=2)
                
                scout_output = gr.Markdown(label="Scout Response")
                
                scout_btn.click(
                    scout_reason,
                    inputs=[scout_prompt, scout_mode],
                    outputs=[scout_output]
                )
                
                load_paper_btn.click(
                    load_paper_to_scout,
                    inputs=[load_paper_id],
                    outputs=[load_status]
                )
            
            # ===== DeepSeek Tab =====
            with gr.Tab("💻 DeepSeek"):
                gr.Markdown("Scientific code generation with DeepSeek-Coder.")
                
                with gr.Row():
                    with gr.Column():
                        gen_prompt = gr.Textbox(
                            label="Code Description",
                            placeholder="Describe what you want to implement...",
                            lines=5
                        )
                        
                        gen_language = gr.Dropdown(
                            choices=["python", "julia", "rust", "cpp", "lean", "latex"],
                            value="python",
                            label="Target Language"
                        )
                        
                        gen_btn = gr.Button("⚡ Generate", variant="primary")
                    
                    with gr.Column():
                        templates_btn = gr.Button("📋 List Templates")
                        templates_output = gr.Markdown()
                
                code_output = gr.Markdown(label="Generated Code")
                
                gen_btn.click(
                    generate_code,
                    inputs=[gen_prompt, gen_language],
                    outputs=[code_output]
                )
                
                templates_btn.click(
                    list_templates,
                    outputs=[templates_output]
                )
            
            # ===== Orchestrator Tab =====
            with gr.Tab("🔄 Orchestrator"):
                gr.Markdown("Automated multi-step scientific workflows.")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Research Pipeline")
                        research_topic = gr.Textbox(
                            label="Research Topic",
                            placeholder="e.g., quantum computing optimization"
                        )
                        research_papers = gr.Slider(1, 10, 5, step=1, label="Papers to Analyze")
                        research_pipeline_btn = gr.Button("🔬 Run Research Pipeline", variant="primary")
                    
                    with gr.Column():
                        gr.Markdown("### Implementation Pipeline")
                        impl_desc = gr.Textbox(
                            label="Implementation Description",
                            placeholder="e.g., Binary search tree with balancing"
                        )
                        impl_lang = gr.Dropdown(
                            choices=["python", "julia", "rust"],
                            value="python",
                            label="Language"
                        )
                        impl_pipeline_btn = gr.Button("🛠 Run Implementation Pipeline", variant="primary")
                
                pipeline_output = gr.Markdown(label="Pipeline Results")
                
                with gr.Row():
                    status_btn = gr.Button("📊 Status")
                    status_output = gr.Markdown()
                
                research_pipeline_btn.click(
                    run_research_pipeline,
                    inputs=[research_topic, research_papers],
                    outputs=[pipeline_output]
                )
                
                impl_pipeline_btn.click(
                    run_implementation_pipeline,
                    inputs=[impl_desc, impl_lang],
                    outputs=[pipeline_output]
                )
                
                status_btn.click(
                    get_orchestrator_status,
                    outputs=[status_output]
                )
            
            # ===== System Tab =====
            with gr.Tab("⚙️ System"):
                gr.Markdown("System status and configuration.")
                
                status_display = gr.Markdown(value=get_system_status())
                refresh_btn = gr.Button("🔄 Refresh Status")
                
                refresh_btn.click(
                    get_system_status,
                    outputs=[status_display]
                )
                
                gr.Markdown("""
                ### Q-Lang Commands Reference
                
                ```
                # Research
                research search "topic"
                research fetch arxiv:ID
                
                # Scout Reasoning
                scout status
                scout reason "question"
                scout hypothesize "topic"
                
                # DeepSeek Code Gen
                deepseek generate "description"
                deepseek templates
                
                # Orchestrator
                orchestrator status
                orchestrator research "topic"
                orchestrator implement "description"
                
                # Polyglot Compute
                polyglot matmul A B result
                polyglot eigen M
                
                # Julia Bridge
                julia benchmark 1000
                julia matmul A B result
                ```
                """)
        
        # Footer
        gr.Markdown("""
        ---
        **QENEX Lab** | Built with the Trinity Pipeline: Scout 17B + DeepSeek + Scout CLI
        """)
    
    return demo


def launch_dashboard(port: int = 7860, share: bool = False):
    """Launch the dashboard."""
    demo = create_dashboard()
    if demo is None:
        return
    
    print(f"""
╔═══════════════════════════════════════════════════════════════════╗
║                   QENEX Scientific Laboratory                       ║
║                        Web Dashboard                                 ║
╠═══════════════════════════════════════════════════════════════════╣
║  Local:  http://localhost:{port}                                    ║
║  Share:  {'Enabled (public URL will be shown below)' if share else 'Disabled'}             ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    demo.launch(
        server_port=port,
        share=share,
        show_error=True,
    )


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QENEX Lab Web Dashboard")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on")
    parser.add_argument("--share", action="store_true", help="Create public URL")
    parser.add_argument("--status", action="store_true", help="Show system status and exit")
    
    args = parser.parse_args()
    
    if args.status:
        print(get_system_status())
    else:
        if not GRADIO_AVAILABLE:
            print("Error: Gradio not installed.")
            print("Install with: pip install gradio>=4.0.0")
            sys.exit(1)
        
        launch_dashboard(port=args.port, share=args.share)
