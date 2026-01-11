"""
QENEX LLM Integration Layer (100% Local)
=========================================
Connects Scout 17B and DeepSeek to LOCAL LLM backends only.

This module bridges the gap between:
- Scout 10M (scientific reasoning) and actual LLM inference
- DeepSeek (code generation) and actual LLM inference
- Context persistence for long-running sessions
- Experiment state management

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    QENEX LLM Integration                          │
    │                    (100% Local Operation)                         │
    ├──────────────────────────────────────────────────────────────────┤
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐           │
    │  │   Scout      │  │   DeepSeek   │  │  Experiment  │           │
    │  │  Reasoner    │  │   Engine     │  │    State     │           │
    │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘           │
    │         │                 │                 │                    │
    │         └─────────────────┼─────────────────┘                    │
    │                           │                                      │
    │                    ┌──────▼──────┐                               │
    │                    │  LLM Router │                               │
    │                    └──────┬──────┘                               │
    │                           │                                      │
    │              ┌────────────┼────────────┐                         │
    │              ▼            ▼            ▼                         │
    │          Ollama      llama.cpp       vLLM                        │
    │         (local)       (local)       (local)                      │
    └──────────────────────────────────────────────────────────────────┘

Supported LOCAL Backends:
- Ollama: HTTP REST API to local Ollama server (localhost:11434)
- llama.cpp: Direct subprocess or local server mode
- vLLM: Local high-throughput serving with PagedAttention
- Mock: Testing without real LLM

NO EXTERNAL APIs - All operations run 100% locally for:
- Data sovereignty
- Offline operation
- Reproducibility
- Cost control

Features:
- Seamless integration with Scout for reasoning tasks
- Seamless integration with DeepSeek for code generation
- Automatic model selection based on task type
- Context persistence across sessions
- Experiment state checkpointing
- Token usage tracking and optimization

Q-Lang Commands:
    integrate status              # Show integration status
    integrate configure           # Configure LLM settings
    integrate scout <backend>     # Set Scout backend
    integrate deepseek <backend>  # Set DeepSeek backend
    integrate benchmark           # Run performance benchmark

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import time
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# Import QENEX components
try:
    from .llm_backend import (
        LLMRouter, LLMBackend, BackendType,
        GenerationConfig, GenerationResult,
        OllamaBackend, LlamaCppBackend, VLLMBackend,
        OpenAICompatBackend, MockBackend,
        create_default_router,
    )
    from .context_store import ContextStore, Scout10MContext, ContextChunk
    from .scout_10m import ScoutReasoner, ScoutMode, ScoutResult
    from .deepseek import DeepSeekEngine, DeepSeekMode, CodeGenerationResult, TargetLanguage
except ImportError:
    from llm_backend import (
        LLMRouter, LLMBackend, BackendType,
        GenerationConfig, GenerationResult,
        OllamaBackend, LlamaCppBackend, VLLMBackend,
        OpenAICompatBackend, MockBackend,
        create_default_router,
    )
    from context_store import ContextStore, ContextChunk
    try:
        from scout_10m import ScoutReasoner, ScoutMode, ScoutResult, Scout10MContext
    except ImportError:
        ScoutReasoner = None
        ScoutMode = None
        ScoutResult = None
        Scout10MContext = None
    try:
        from deepseek import DeepSeekEngine, DeepSeekMode, CodeGenerationResult, TargetLanguage
    except ImportError:
        DeepSeekEngine = None
        DeepSeekMode = None
        CodeGenerationResult = None
        TargetLanguage = None


class TaskType(Enum):
    """Types of tasks for model routing."""
    REASONING = "reasoning"          # Deep scientific reasoning
    CODE_GENERATION = "code"         # Code generation
    ANALYSIS = "analysis"            # Data/text analysis
    PROOF = "proof"                  # Mathematical proofs
    HYPOTHESIS = "hypothesis"        # Hypothesis generation
    DOCUMENTATION = "documentation"  # Doc generation
    TRANSLATION = "translation"      # Code translation
    GENERAL = "general"              # General chat/completion


@dataclass
class IntegrationConfig:
    """Configuration for LLM integration."""
    # Scout settings
    scout_backend: BackendType = BackendType.OLLAMA
    scout_model: str = "llama2:70b"  # Reasoning model
    scout_temperature: float = 0.3   # Lower for reasoning
    scout_max_tokens: int = 4096
    
    # DeepSeek settings
    deepseek_backend: BackendType = BackendType.OLLAMA
    deepseek_model: str = "deepseek-coder:33b"  # Code model
    deepseek_temperature: float = 0.2  # Lower for code
    deepseek_max_tokens: int = 8192
    
    # General settings
    enable_context_persistence: bool = True
    context_store_path: str = ""
    auto_checkpoint: bool = True
    checkpoint_interval: int = 10  # Checkpoint every N operations
    
    # Token tracking
    track_tokens: bool = True
    token_budget: Optional[int] = None  # Max tokens per session


@dataclass
class TokenUsage:
    """Track token usage across operations."""
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    operations: int = 0
    
    def add(self, prompt: int, completion: int):
        """Add token usage."""
        self.total_prompt_tokens += prompt
        self.total_completion_tokens += completion
        self.total_tokens += prompt + completion
        self.operations += 1
    
    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "operations": self.operations,
        }


class IntegratedScoutReasoner:
    """
    Scout Reasoner with real LLM backend integration.
    
    Wraps the Scout 10M context engine with actual LLM calls
    instead of simulated responses.
    """
    
    # System prompt for scientific reasoning
    SYSTEM_PROMPT = """You are Scout, a specialized scientific reasoning engine.

Your capabilities:
1. DEEP REASONING: Multi-step chain-of-thought for complex problems
2. SYNTHESIS: Cross-reference and integrate information from multiple sources
3. PROOF CONSTRUCTION: Build formal mathematical and logical proofs
4. HYPOTHESIS GENERATION: Propose novel scientific hypotheses
5. VERIFICATION: Validate claims against loaded context and known physics

Scientific domains you specialize in:
- Physics (quantum mechanics, relativity, thermodynamics, astrophysics)
- Chemistry (quantum chemistry, molecular dynamics, materials science)
- Biology (genomics, proteomics, systems biology, evolution)
- Mathematics (proofs, analysis, algebra, topology)
- Computer Science (algorithms, complexity, formal verification)

When reasoning:
- Always show your work step-by-step
- Cite sources from the loaded context when available
- Quantify uncertainty and confidence levels
- Flag potential errors or inconsistencies
- Connect findings across different domains

Format your responses with clear structure using markdown headers."""

    def __init__(
        self,
        router: LLMRouter,
        config: IntegrationConfig,
        context: Optional[Any] = None,
        verbose: bool = True,
    ):
        self.router = router
        self.config = config
        self.verbose = verbose
        self.token_usage = TokenUsage()
        
        # Initialize context
        if context is None and Scout10MContext is not None:
            self.context = Scout10MContext(verbose=False)
        else:
            self.context = context
        
        # Mode-specific prompts
        self.mode_prompts = {
            "REASON": "Please provide deep, step-by-step reasoning to address this query. Use chain-of-thought to break down the problem.",
            "SYNTHESIZE": "Please synthesize information from all available context to address this query. Cross-reference findings across different sources.",
            "PROVE": "Please construct a formal proof for this statement. Use rigorous mathematical notation and state all assumptions.",
            "HYPOTHESIZE": "Please generate novel scientific hypotheses related to this domain. For each, state the hypothesis, reasoning, testability, and plausibility.",
            "VERIFY": "Please verify this claim against known physical laws, mathematical consistency, and dimensional analysis.",
            "ANALYZE": "Please provide a comprehensive analysis including overview, strengths, weaknesses, and recommendations.",
        }
        
        if verbose:
            print("🧠 Integrated Scout Reasoner initialized")
    
    def _build_prompt(self, query: str, mode: str, context_str: str = "") -> str:
        """Build the full prompt for reasoning."""
        mode_instruction = self.mode_prompts.get(mode, self.mode_prompts["REASON"])
        
        prompt_parts = [self.SYSTEM_PROMPT]
        
        if context_str:
            prompt_parts.append(f"\n\n## LOADED CONTEXT\n\n{context_str}")
        
        prompt_parts.append(f"\n\n## TASK: {mode}\n\n{mode_instruction}")
        prompt_parts.append(f"\n\n## QUERY\n\n{query}")
        prompt_parts.append("\n\n## RESPONSE\n")
        
        return "\n".join(prompt_parts)
    
    def reason(
        self,
        query: str,
        mode: str = "REASON",
        use_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute reasoning with real LLM backend.
        
        Args:
            query: The reasoning query
            mode: Reasoning mode (REASON, SYNTHESIZE, PROVE, etc.)
            use_context: Whether to include loaded context
        
        Returns:
            Dictionary with reasoning result
        """
        start = time.time()
        
        # Build context string if available
        context_str = ""
        if use_context and self.context is not None:
            if hasattr(self.context, 'build_context_string'):
                context_str = self.context.build_context_string(max_tokens=50000)
            elif hasattr(self.context, 'chunks'):
                # Build from chunks directly
                context_parts = []
                for chunk in list(self.context.chunks.values())[:20]:
                    context_parts.append(f"### {chunk.source}\n{chunk.content[:2000]}")
                context_str = "\n\n".join(context_parts)
        
        # Build prompt
        prompt = self._build_prompt(query, mode, context_str)
        
        # Configure generation
        gen_config = GenerationConfig(
            max_tokens=self.config.scout_max_tokens,
            temperature=self.config.scout_temperature,
            top_p=0.95,
        )
        
        # Select backend and model
        if self.config.scout_backend in self.router.backends:
            self.router.select_backend(self.config.scout_backend)
        self.router.set_model(self.config.scout_model)
        
        # Generate
        result = self.router.generate(prompt, config=gen_config)
        
        elapsed = (time.time() - start) * 1000
        
        # Track tokens
        if self.config.track_tokens:
            self.token_usage.add(result.tokens_prompt, result.tokens_generated)
        
        # Build response
        response = {
            "success": result.finish_reason != "error",
            "mode": mode,
            "output": result.text,
            "query": query,
            "tokens_prompt": result.tokens_prompt,
            "tokens_generated": result.tokens_generated,
            "elapsed_ms": elapsed,
            "backend": result.backend,
            "model": result.model,
            "context_chunks": len(self.context.chunks) if self.context and hasattr(self.context, 'chunks') else 0,
        }
        
        if self.verbose:
            print(f"✅ Scout reasoning complete ({elapsed:.0f}ms, {result.tokens_generated} tokens)")
        
        return response


class IntegratedDeepSeekEngine:
    """
    DeepSeek Engine with real LLM backend integration.
    
    Wraps the DeepSeek code generation with actual LLM calls.
    """
    
    # System prompt for code generation
    SYSTEM_PROMPT = """You are DeepSeek-Coder, a specialized AI for scientific code generation.

Your capabilities:
1. Generate high-quality scientific computing code
2. Implement mathematical algorithms precisely
3. Write well-documented, type-hinted code
4. Follow best practices for numerical computing
5. Generate comprehensive test suites

Languages you excel at:
- Python (NumPy, SciPy, SymPy, JAX)
- Julia (high-performance numerics)
- Rust (systems programming, FFI)
- Q-Lang (scientific DSL)

Code style requirements:
- Clear variable names reflecting domain concepts
- Comprehensive docstrings (Google style for Python)
- Type hints where applicable
- Error handling for edge cases
- Numerical stability considerations

Always output code in markdown code blocks with language annotation."""

    # Mode-specific instructions
    MODE_PROMPTS = {
        "GENERATE": "Generate code that implements the following functionality:",
        "OPTIMIZE": "Optimize the following code for performance while maintaining correctness:",
        "TEST": "Generate comprehensive unit tests for the following code:",
        "DOCUMENT": "Generate comprehensive documentation for the following code:",
        "TRANSLATE": "Translate the following code to {target_language}:",
        "EXPLAIN": "Explain in detail what the following code does:",
        "DEBUG": "Identify and fix bugs in the following code:",
        "COMPLETE": "Complete the following code:",
    }

    def __init__(
        self,
        router: LLMRouter,
        config: IntegrationConfig,
        verbose: bool = True,
    ):
        self.router = router
        self.config = config
        self.verbose = verbose
        self.token_usage = TokenUsage()
        self.generation_history: List[Dict[str, Any]] = []
        
        if verbose:
            print("💻 Integrated DeepSeek Engine initialized")
    
    def _build_prompt(
        self,
        description: str,
        mode: str,
        language: str,
        context: str = "",
    ) -> str:
        """Build the full prompt for code generation."""
        mode_instruction = self.MODE_PROMPTS.get(mode, self.MODE_PROMPTS["GENERATE"])
        
        # Handle translation target
        if "{target_language}" in mode_instruction:
            mode_instruction = mode_instruction.format(target_language=language)
        
        prompt_parts = [
            self.SYSTEM_PROMPT,
            f"\n\n## TARGET LANGUAGE: {language}",
            f"\n\n## TASK: {mode}",
            f"\n{mode_instruction}",
        ]
        
        if context:
            prompt_parts.append(f"\n\n## EXISTING CODE/CONTEXT\n```\n{context}\n```")
        
        prompt_parts.append(f"\n\n## DESCRIPTION\n{description}")
        prompt_parts.append("\n\n## GENERATED CODE\n")
        
        return "\n".join(prompt_parts)
    
    def generate(
        self,
        description: str,
        language: str = "python",
        mode: str = "GENERATE",
        context: str = "",
    ) -> Dict[str, Any]:
        """
        Generate code with real LLM backend.
        
        Args:
            description: What the code should do
            language: Target programming language
            mode: Generation mode
            context: Optional existing code context
        
        Returns:
            Dictionary with generated code
        """
        start = time.time()
        
        # Build prompt
        prompt = self._build_prompt(description, mode, language, context)
        
        # Configure generation
        gen_config = GenerationConfig(
            max_tokens=self.config.deepseek_max_tokens,
            temperature=self.config.deepseek_temperature,
            top_p=0.95,
            stop=["```\n\n", "## "],  # Stop after code block
        )
        
        # Select backend and model
        if self.config.deepseek_backend in self.router.backends:
            self.router.select_backend(self.config.deepseek_backend)
        self.router.set_model(self.config.deepseek_model)
        
        # Generate
        result = self.router.generate(prompt, config=gen_config)
        
        elapsed = (time.time() - start) * 1000
        
        # Extract code from response
        code = self._extract_code(result.text, language)
        
        # Track tokens
        if self.config.track_tokens:
            self.token_usage.add(result.tokens_prompt, result.tokens_generated)
        
        # Build response
        response = {
            "success": result.finish_reason != "error",
            "mode": mode,
            "language": language,
            "code": code,
            "raw_response": result.text,
            "description": description,
            "tokens_prompt": result.tokens_prompt,
            "tokens_generated": result.tokens_generated,
            "elapsed_ms": elapsed,
            "backend": result.backend,
            "model": result.model,
        }
        
        self.generation_history.append(response)
        
        if self.verbose:
            print(f"✅ DeepSeek generation complete ({elapsed:.0f}ms, {len(code)} chars)")
        
        return response
    
    def _extract_code(self, text: str, language: str) -> str:
        """Extract code from LLM response."""
        import re
        
        # Try to find code block with language annotation
        pattern = rf"```{language}\n(.*?)```"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        # Try generic code block
        pattern = r"```\n?(.*?)```"
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Return full text if no code block found
        return text.strip()


class ExperimentStateManager:
    """
    Manages experiment state with persistence.
    
    Features:
    - Auto-checkpoint after N operations
    - Save/restore experiment context
    - Track experiment history
    - Store intermediate results
    """
    
    def __init__(
        self,
        store: ContextStore,
        experiment_name: str = "default",
        auto_checkpoint: bool = True,
        checkpoint_interval: int = 10,
    ):
        self.store = store
        self.experiment_name = experiment_name
        self.auto_checkpoint = auto_checkpoint
        self.checkpoint_interval = checkpoint_interval
        
        self.state: Dict[str, Any] = {
            "name": experiment_name,
            "created": time.time(),
            "modified": time.time(),
            "operations": [],
            "results": {},
            "metadata": {},
        }
        
        self.operation_count = 0
        self.checkpoint_count = 0
    
    def record_operation(
        self,
        operation_type: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
    ) -> None:
        """Record an operation in experiment history."""
        self.state["operations"].append({
            "type": operation_type,
            "timestamp": time.time(),
            "input": input_data,
            "output_summary": self._summarize(output_data),
        })
        
        self.state["modified"] = time.time()
        self.operation_count += 1
        
        # Auto-checkpoint
        if self.auto_checkpoint and self.operation_count % self.checkpoint_interval == 0:
            self.checkpoint()
    
    def _summarize(self, data: Dict[str, Any], max_len: int = 500) -> Dict[str, Any]:
        """Summarize output data for storage."""
        summary = {}
        for key, value in data.items():
            if isinstance(value, str) and len(value) > max_len:
                summary[key] = value[:max_len] + "..."
            elif isinstance(value, (int, float, bool)):
                summary[key] = value
            elif isinstance(value, dict):
                summary[key] = self._summarize(value, max_len // 2)
            elif isinstance(value, list):
                summary[key] = f"[list of {len(value)} items]"
            else:
                summary[key] = str(type(value).__name__)
        return summary
    
    def store_result(self, key: str, value: Any) -> None:
        """Store a named result."""
        self.state["results"][key] = {
            "value": value,
            "timestamp": time.time(),
        }
    
    def get_result(self, key: str) -> Optional[Any]:
        """Get a stored result."""
        if key in self.state["results"]:
            return self.state["results"][key]["value"]
        return None
    
    def checkpoint(self, name: Optional[str] = None) -> Path:
        """Save current state to disk."""
        checkpoint_name = name or f"{self.experiment_name}_checkpoint_{self.checkpoint_count}"
        
        # Create mock context for storage
        context = type('MockContext', (), {
            'chunks': {},
            'total_tokens': 0,
            'conversation_history': [],
        })()
        
        # Add state as a chunk
        state_chunk = ContextChunk(
            id=f"state_{self.checkpoint_count}",
            content=json.dumps(self.state, indent=2, default=str),
            token_count=len(json.dumps(self.state)) // 4,
            source=f"experiment:{self.experiment_name}",
            chunk_type="state",
            metadata={"checkpoint": self.checkpoint_count},
        )
        context.chunks[state_chunk.id] = state_chunk
        context.total_tokens = state_chunk.token_count
        
        path = self.store.save(
            context,
            name=checkpoint_name,
            description=f"Experiment checkpoint {self.checkpoint_count}",
            tags=["experiment", "checkpoint", self.experiment_name],
        )
        
        self.checkpoint_count += 1
        return path
    
    def restore(self, path: str) -> bool:
        """Restore state from checkpoint."""
        try:
            context = self.store.load(path)
            
            # Find state chunk
            for chunk in context.chunks.values():
                if chunk.chunk_type == "state":
                    self.state = json.loads(chunk.content)
                    return True
            
            return False
        except Exception as e:
            print(f"Failed to restore: {e}")
            return False


class QENEXIntegration:
    """
    Main integration class for QENEX LLM system.
    
    Provides unified access to:
    - Scout reasoning with real LLM
    - DeepSeek code generation with real LLM
    - Context persistence
    - Experiment state management
    """
    
    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        verbose: bool = True,
    ):
        self.config = config or IntegrationConfig()
        self.verbose = verbose
        
        # Initialize router
        self.router = create_default_router(verbose=False)
        
        # Initialize context store
        store_path = self.config.context_store_path or str(Path.home() / ".qenex" / "contexts")
        self.context_store = ContextStore(base_dir=store_path, verbose=False)
        
        # Initialize Scout context
        if Scout10MContext is not None:
            self.scout_context = Scout10MContext(verbose=False)
        else:
            self.scout_context = None
        
        # Initialize integrated components
        self.scout = IntegratedScoutReasoner(
            self.router, self.config, self.scout_context, verbose=False
        )
        self.deepseek = IntegratedDeepSeekEngine(
            self.router, self.config, verbose=False
        )
        
        # Initialize experiment manager
        self.experiment = ExperimentStateManager(
            self.context_store,
            auto_checkpoint=self.config.auto_checkpoint,
            checkpoint_interval=self.config.checkpoint_interval,
        )
        
        # Token tracking
        self.total_tokens = TokenUsage()
        
        if verbose:
            self._print_status()
    
    def _print_status(self):
        """Print integration status."""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           QENEX LLM Integration                              ║
    ║           Scout + DeepSeek + Persistence                     ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        print(f"    🧠 Scout Backend: {self.config.scout_backend.value}")
        print(f"    💻 DeepSeek Backend: {self.config.deepseek_backend.value}")
        print(f"    📁 Context Store: {self.context_store.base_dir}")
        print(f"    ✅ Available Backends: {', '.join(b.name for b in self.router.get_available_backends())}")
        print()
    
    def reason(self, query: str, mode: str = "REASON") -> Dict[str, Any]:
        """Execute Scout reasoning."""
        result = self.scout.reason(query, mode)
        
        # Record in experiment
        self.experiment.record_operation(
            f"scout_{mode.lower()}",
            {"query": query, "mode": mode},
            result,
        )
        
        # Aggregate tokens
        self.total_tokens.add(result["tokens_prompt"], result["tokens_generated"])
        
        return result
    
    def generate_code(
        self,
        description: str,
        language: str = "python",
        mode: str = "GENERATE",
    ) -> Dict[str, Any]:
        """Execute DeepSeek code generation."""
        result = self.deepseek.generate(description, language, mode)
        
        # Record in experiment
        self.experiment.record_operation(
            f"deepseek_{mode.lower()}",
            {"description": description, "language": language},
            result,
        )
        
        # Aggregate tokens
        self.total_tokens.add(result["tokens_prompt"], result["tokens_generated"])
        
        return result
    
    def load_context(self, path: str) -> None:
        """Load context for Scout reasoning."""
        if self.scout_context is not None:
            self.context_store.load(path, into_context=self.scout_context, merge=True)
    
    def save_context(self, name: str) -> Path:
        """Save current Scout context."""
        if self.scout_context is not None:
            return self.context_store.save(self.scout_context, name=name)
        return Path()
    
    def checkpoint(self) -> Path:
        """Create experiment checkpoint."""
        return self.experiment.checkpoint()
    
    def get_status(self) -> Dict[str, Any]:
        """Get integration status."""
        return {
            "scout": {
                "backend": self.config.scout_backend.value,
                "model": self.config.scout_model,
                "context_chunks": len(self.scout_context.chunks) if self.scout_context else 0,
            },
            "deepseek": {
                "backend": self.config.deepseek_backend.value,
                "model": self.config.deepseek_model,
                "generations": len(self.deepseek.generation_history),
            },
            "tokens": self.total_tokens.to_dict(),
            "experiment": {
                "name": self.experiment.experiment_name,
                "operations": len(self.experiment.state["operations"]),
                "checkpoints": self.experiment.checkpoint_count,
            },
            "available_backends": [b.name for b in self.router.get_available_backends()],
        }


def handle_integrate_command(integration: QENEXIntegration, line: str, context: dict) -> None:
    """
    Handle integration commands from Q-Lang interpreter.
    
    Commands:
        integrate status              # Show integration status
        integrate configure key=value # Configure settings
        integrate scout <backend>     # Set Scout backend
        integrate deepseek <backend>  # Set DeepSeek backend
        integrate checkpoint          # Create checkpoint
        integrate benchmark           # Run benchmark
        integrate help                # Show help
    """
    parts = line.split(maxsplit=2)
    
    if len(parts) < 2:
        print("❌ Usage: integrate <command> [args...]")
        return
    
    cmd = parts[1].lower()
    
    try:
        if cmd == "status":
            status = integration.get_status()
            print("\n🔗 QENEX Integration Status")
            print("=" * 50)
            print(f"\n  Scout:")
            print(f"    Backend: {status['scout']['backend']}")
            print(f"    Model: {status['scout']['model']}")
            print(f"    Context: {status['scout']['context_chunks']} chunks")
            print(f"\n  DeepSeek:")
            print(f"    Backend: {status['deepseek']['backend']}")
            print(f"    Model: {status['deepseek']['model']}")
            print(f"    Generations: {status['deepseek']['generations']}")
            print(f"\n  Tokens:")
            for k, v in status['tokens'].items():
                print(f"    {k}: {v:,}")
            print(f"\n  Experiment:")
            print(f"    Name: {status['experiment']['name']}")
            print(f"    Operations: {status['experiment']['operations']}")
            print(f"    Checkpoints: {status['experiment']['checkpoints']}")
            print(f"\n  Available Backends: {', '.join(status['available_backends'])}")
            print()
        
        elif cmd == "scout":
            if len(parts) < 3:
                print(f"Current Scout backend: {integration.config.scout_backend.value}")
                return
            
            backend_name = parts[2].lower()
            backend_map = {
                'ollama': BackendType.OLLAMA,
                'llamacpp': BackendType.LLAMACPP,
                'vllm': BackendType.VLLM,
                'mock': BackendType.MOCK,
            }
            
            if backend_name in backend_map:
                integration.config.scout_backend = backend_map[backend_name]
                print(f"✅ Scout backend set to: {backend_name}")
            else:
                print(f"❌ Unknown backend: {backend_name}")
        
        elif cmd == "deepseek":
            if len(parts) < 3:
                print(f"Current DeepSeek backend: {integration.config.deepseek_backend.value}")
                return
            
            backend_name = parts[2].lower()
            backend_map = {
                'ollama': BackendType.OLLAMA,
                'llamacpp': BackendType.LLAMACPP,
                'vllm': BackendType.VLLM,
                'mock': BackendType.MOCK,
            }
            
            if backend_name in backend_map:
                integration.config.deepseek_backend = backend_map[backend_name]
                print(f"✅ DeepSeek backend set to: {backend_name}")
            else:
                print(f"❌ Unknown backend: {backend_name}")
        
        elif cmd == "configure":
            if len(parts) < 3:
                print("Current configuration:")
                print(f"  scout_model: {integration.config.scout_model}")
                print(f"  scout_temperature: {integration.config.scout_temperature}")
                print(f"  deepseek_model: {integration.config.deepseek_model}")
                print(f"  deepseek_temperature: {integration.config.deepseek_temperature}")
                return
            
            for kv in parts[2].split():
                if '=' in kv:
                    key, value = kv.split('=', 1)
                    if hasattr(integration.config, key):
                        old_value = getattr(integration.config, key)
                        if isinstance(old_value, float):
                            setattr(integration.config, key, float(value))
                        elif isinstance(old_value, int):
                            setattr(integration.config, key, int(value))
                        elif isinstance(old_value, bool):
                            setattr(integration.config, key, value.lower() == 'true')
                        else:
                            setattr(integration.config, key, value)
                        print(f"✅ Set {key} = {value}")
        
        elif cmd == "checkpoint":
            path = integration.checkpoint()
            print(f"✅ Checkpoint saved: {path}")
        
        elif cmd == "benchmark":
            print("\n🏃 Running LLM Benchmark...")
            print("=" * 50)
            
            # Test each available backend
            for backend in integration.router.get_available_backends():
                print(f"\n  Testing {backend.name}...")
                
                integration.router.select_backend(backend.backend_type)
                
                start = time.time()
                result = integration.router.generate(
                    "Write a simple Python function to calculate factorial.",
                    config=GenerationConfig(max_tokens=256),
                )
                elapsed = (time.time() - start) * 1000
                
                print(f"    ⏱️  Latency: {elapsed:.0f}ms")
                print(f"    📊 Tokens: {result.tokens_generated}")
                if result.tokens_generated > 0:
                    print(f"    🚀 Speed: {result.tokens_generated / (elapsed/1000):.1f} tokens/sec")
            
            print()
        
        elif cmd == "help":
            print("""
📖 Integration Commands
========================

  integrate status              Show integration status
  integrate scout <backend>     Set Scout backend (ollama/llamacpp/vllm/mock)
  integrate deepseek <backend>  Set DeepSeek backend
  integrate configure [k=v...]  Configure settings
  integrate checkpoint          Save experiment checkpoint
  integrate benchmark           Run performance benchmark

Settings:
  scout_model, scout_temperature, scout_max_tokens
  deepseek_model, deepseek_temperature, deepseek_max_tokens
            """)
        
        else:
            print(f"❌ Unknown command: {cmd}")
    
    except Exception as e:
        print(f"❌ Integration error: {e}")


# Availability flag
HAS_INTEGRATION = True


# Demo
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX LLM Integration Demo                             ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create integration
    integration = QENEXIntegration(verbose=True)
    
    # Show status
    status = integration.get_status()
    print(f"Available backends: {status['available_backends']}")
    
    # Test Scout reasoning
    print("\n" + "=" * 60)
    print("Testing Scout Reasoning...")
    print("=" * 60)
    
    result = integration.reason(
        "Derive the relationship between entropy and information",
        mode="REASON"
    )
    print(f"\nResult preview: {result['output'][:500]}...")
    
    # Test DeepSeek code generation
    print("\n" + "=" * 60)
    print("Testing DeepSeek Code Generation...")
    print("=" * 60)
    
    result = integration.generate_code(
        "Calculate the eigenvalues of a symmetric matrix",
        language="python"
    )
    print(f"\nGenerated code:\n{result['code'][:500]}...")
    
    # Create checkpoint
    print("\n" + "=" * 60)
    print("Creating checkpoint...")
    print("=" * 60)
    
    path = integration.checkpoint()
    print(f"Checkpoint saved: {path}")
    
    print("\n✅ Demo complete!")
