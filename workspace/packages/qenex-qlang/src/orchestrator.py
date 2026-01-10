"""
QENEX Experiment Orchestrator
=============================

Automated multi-step scientific workflow engine that orchestrates:
- Research: Literature search and paper analysis
- Scout 10M: Scientific reasoning with 10M token context
- DeepSeek: Code generation and optimization
- Polyglot: Multi-language computation dispatch
- Validation: Result verification and reproducibility

The orchestrator enables:
1. Reproducible experiment pipelines
2. Automated hypothesis-to-code workflows
3. Multi-paper synthesis and analysis
4. Checkpoint/resume for long-running experiments
5. Parallel execution of independent steps

Usage:
    from orchestrator import ExperimentOrchestrator, Pipeline, Step
    
    # Define a pipeline
    pipeline = Pipeline("quantum_research")
    pipeline.add_step(Step("search", "research", {"query": "quantum error correction"}))
    pipeline.add_step(Step("analyze", "scout", {"mode": "reason", "prompt": "Key findings?"}))
    pipeline.add_step(Step("implement", "deepseek", {"prompt": "Implement stabilizer codes"}))
    
    # Execute
    orchestrator = ExperimentOrchestrator()
    results = orchestrator.run(pipeline)

Author: QENEX Sovereign Agent
Version: 1.0.0
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable, Union
from enum import Enum
from datetime import datetime
import json
import hashlib
import uuid
import time
import os
import sys
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Robust imports
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import QENEX modules
try:
    from research import ResearchEngine, Paper, ResearchResult
    RESEARCH_AVAILABLE = True
except ImportError:
    RESEARCH_AVAILABLE = False
    ResearchEngine = None

try:
    from scout_10m import ScoutReasoner, ScoutMode, ReasoningResult
    SCOUT_AVAILABLE = True
except ImportError:
    SCOUT_AVAILABLE = False
    ScoutReasoner = None

try:
    from deepseek import DeepSeekEngine, GenerationResult
    DEEPSEEK_AVAILABLE = True
except ImportError:
    DEEPSEEK_AVAILABLE = False
    DeepSeekEngine = None

try:
    from polyglot import PolyglotDispatcher, Backend
    POLYGLOT_AVAILABLE = True
except ImportError:
    POLYGLOT_AVAILABLE = False
    PolyglotDispatcher = None


# ============================================================================
# Mock Classes for Graceful Degradation
# ============================================================================

class MockReasoningResult:
    """Mock reasoning result when Scout is not available."""
    def __init__(self, prompt: str):
        self.output = f"[Mock Scout Response to: {prompt[:100]}...]"
        self.mode = "mock"
        self.tokens_used = 0
        self.elapsed_ms = 0.0
        self.success = True


class MockScoutReasoner:
    """Mock Scout reasoner for graceful degradation."""
    
    class MockContext:
        def add_chunk(self, text, source, type_): pass
        def clear(self): pass
    
    def __init__(self):
        self.context = self.MockContext()
    
    def reason(self, prompt: str, mode=None) -> MockReasoningResult:
        return MockReasoningResult(prompt)


class MockGenerationResult:
    """Mock generation result when DeepSeek is not available."""
    def __init__(self, prompt: str, language: str = "python"):
        self.code = f"# Mock code for: {prompt[:50]}...\npass"
        self.language = language
        self.tokens_used = 0
        self.elapsed_ms = 0.0
        self.success = True


class MockDeepSeekEngine:
    """Mock DeepSeek engine for graceful degradation."""
    
    def generate(self, prompt: str, language: str = "python") -> MockGenerationResult:
        return MockGenerationResult(prompt, language)
    
    def optimize(self, code: str) -> MockGenerationResult:
        return MockGenerationResult(f"optimize: {code[:50]}")
    
    def generate_tests(self, code: str) -> MockGenerationResult:
        return MockGenerationResult(f"test: {code[:50]}")


class MockPolyglotResult:
    """Mock polyglot result."""
    def __init__(self, operation: str):
        self.result = None
        self.backend = "mock"
        self.elapsed_ms = 0.0
        self.operation = operation


class MockPolyglotDispatcher:
    """Mock Polyglot dispatcher for graceful degradation."""
    
    def matmul(self, a, b):
        import numpy as np
        result = MockPolyglotResult("matmul")
        result.result = np.dot(a, b)
        return result
    
    def eigensolve(self, m):
        import numpy as np
        result = MockPolyglotResult("eigen")
        result.result = np.linalg.eig(m)
        return result
    
    def solve(self, a, b):
        import numpy as np
        result = MockPolyglotResult("solve")
        result.result = np.linalg.solve(a, b)
        return result
    
    def fft(self, data):
        import numpy as np
        result = MockPolyglotResult("fft")
        result.result = np.fft.fft(data)
        return result


# ============================================================================
# Core Enums and Types
# ============================================================================

class StepType(Enum):
    """Types of pipeline steps."""
    RESEARCH = "research"       # Literature search and analysis
    SCOUT = "scout"             # Scientific reasoning
    DEEPSEEK = "deepseek"       # Code generation
    POLYGLOT = "polyglot"       # Computation dispatch
    VALIDATE = "validate"       # Result validation
    CUSTOM = "custom"           # User-defined function
    PARALLEL = "parallel"       # Parallel sub-steps
    CONDITIONAL = "conditional" # Conditional branching
    LOOP = "loop"               # Iterative execution
    CHECKPOINT = "checkpoint"   # Save state


class StepStatus(Enum):
    """Status of a pipeline step."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    CACHED = "cached"


@dataclass
class StepResult:
    """Result of executing a pipeline step."""
    step_id: str
    step_name: str
    status: StepStatus
    output: Any = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_ms: float = 0.0
    cached: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "step_id": self.step_id,
            "step_name": self.step_name,
            "status": self.status.value,
            "output": str(self.output)[:1000] if self.output else None,
            "error": self.error,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "cached": self.cached,
            "metadata": self.metadata
        }


@dataclass
class Step:
    """A single step in a pipeline."""
    name: str
    step_type: Union[StepType, str]
    params: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[str] = None  # Python expression for conditional execution
    retry_count: int = 0
    timeout_seconds: float = 300.0
    cache_key: Optional[str] = None  # For caching results
    
    def __post_init__(self):
        if isinstance(self.step_type, str):
            self.step_type = StepType(self.step_type)
        self.id = f"{self.name}_{uuid.uuid4().hex[:8]}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.step_type.value,
            "params": self.params,
            "depends_on": self.depends_on,
            "condition": self.condition,
            "retry_count": self.retry_count,
            "timeout_seconds": self.timeout_seconds
        }


@dataclass 
class Pipeline:
    """A scientific experiment pipeline."""
    name: str
    description: str = ""
    steps: List[Step] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        self.id = f"pipeline_{uuid.uuid4().hex[:12]}"
        self.created_at = datetime.now()
    
    def add_step(self, step: Step) -> "Pipeline":
        """Add a step to the pipeline. Returns self for chaining."""
        self.steps.append(step)
        return self
    
    def add_research_step(self, name: str, query: str, max_results: int = 5) -> "Pipeline":
        """Add a research step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.RESEARCH,
            params={"action": "search", "query": query, "max_results": max_results}
        ))
    
    def add_scout_step(self, name: str, prompt: str, mode: str = "reason", 
                       depends_on: List[str] = None) -> "Pipeline":
        """Add a Scout reasoning step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.SCOUT,
            params={"mode": mode, "prompt": prompt},
            depends_on=depends_on or []
        ))
    
    def add_deepseek_step(self, name: str, prompt: str, language: str = "python",
                          depends_on: List[str] = None) -> "Pipeline":
        """Add a DeepSeek code generation step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.DEEPSEEK,
            params={"action": "generate", "prompt": prompt, "language": language},
            depends_on=depends_on or []
        ))
    
    def add_compute_step(self, name: str, operation: str, 
                         depends_on: List[str] = None, **kwargs) -> "Pipeline":
        """Add a Polyglot computation step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.POLYGLOT,
            params={"operation": operation, **kwargs},
            depends_on=depends_on or []
        ))
    
    def add_validation_step(self, name: str, checks: List[str],
                            depends_on: List[str] = None) -> "Pipeline":
        """Add a validation step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.VALIDATE,
            params={"checks": checks},
            depends_on=depends_on or []
        ))
    
    def add_custom_step(self, name: str, function: Callable,
                        depends_on: List[str] = None, **kwargs) -> "Pipeline":
        """Add a custom function step."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.CUSTOM,
            params={"function": function, **kwargs},
            depends_on=depends_on or []
        ))
    
    def add_parallel_steps(self, name: str, steps: List[Step]) -> "Pipeline":
        """Add parallel execution of multiple steps."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.PARALLEL,
            params={"sub_steps": steps}
        ))
    
    def add_checkpoint(self, name: str, depends_on: List[str] = None) -> "Pipeline":
        """Add a checkpoint for resumption."""
        return self.add_step(Step(
            name=name,
            step_type=StepType.CHECKPOINT,
            depends_on=depends_on or []
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize pipeline to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "steps": [s.to_dict() for s in self.steps],
            "variables": self.variables,
            "metadata": self.metadata
        }
    
    def to_json(self) -> str:
        """Serialize pipeline to JSON."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Pipeline":
        """Deserialize pipeline from dictionary."""
        pipeline = cls(
            name=data["name"],
            description=data.get("description", ""),
            variables=data.get("variables", {}),
            metadata=data.get("metadata", {})
        )
        pipeline.id = data.get("id", pipeline.id)
        for step_data in data.get("steps", []):
            step = Step(
                name=step_data["name"],
                step_type=step_data["type"],
                params=step_data.get("params", {}),
                depends_on=step_data.get("depends_on", []),
                condition=step_data.get("condition"),
                retry_count=step_data.get("retry_count", 0),
                timeout_seconds=step_data.get("timeout_seconds", 300.0)
            )
            step.id = step_data.get("id", step.id)
            pipeline.steps.append(step)
        return pipeline
    
    @classmethod
    def from_json(cls, json_str: str) -> "Pipeline":
        """Deserialize pipeline from JSON."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class ExperimentRun:
    """A single run of a pipeline."""
    pipeline: Pipeline
    run_id: str = field(default_factory=lambda: f"run_{uuid.uuid4().hex[:12]}")
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"
    results: Dict[str, StepResult] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)  # Shared context between steps
    checkpoints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize run to dictionary."""
        return {
            "run_id": self.run_id,
            "pipeline_id": self.pipeline.id,
            "pipeline_name": self.pipeline.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "status": self.status,
            "results": {k: v.to_dict() for k, v in self.results.items()},
            "context_keys": list(self.context.keys())
        }


class ExperimentOrchestrator:
    """
    The main orchestrator for scientific experiments.
    
    Coordinates execution of pipelines across all QENEX subsystems.
    """
    
    def __init__(self, 
                 workspace_dir: str = None,
                 cache_enabled: bool = True,
                 max_parallel: int = 4,
                 verbose: bool = True):
        """
        Initialize the orchestrator.
        
        Args:
            workspace_dir: Directory for storing experiment data
            cache_enabled: Whether to cache step results
            max_parallel: Maximum parallel step execution
            verbose: Whether to print progress
        """
        self.workspace_dir = Path(workspace_dir or "/opt/qenex_lab/workspace/experiments")
        self.cache_enabled = cache_enabled
        self.max_parallel = max_parallel
        self.verbose = verbose
        
        # Initialize engines lazily
        self._research: Optional[ResearchEngine] = None
        self._scout: Optional[ScoutReasoner] = None
        self._deepseek: Optional[DeepSeekEngine] = None
        self._polyglot: Optional[PolyglotDispatcher] = None
        
        # Cache storage
        self._cache: Dict[str, Any] = {}
        
        # Run history
        self._runs: List[ExperimentRun] = []
        
        # Custom step handlers
        self._custom_handlers: Dict[str, Callable] = {}
        
        # Ensure workspace exists
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        if self.verbose:
            print(f"[Orchestrator] Initialized with workspace: {self.workspace_dir}")
            self._print_capabilities()
    
    def _print_capabilities(self):
        """Print available capabilities."""
        caps = []
        if RESEARCH_AVAILABLE:
            caps.append("Research")
        if SCOUT_AVAILABLE:
            caps.append("Scout-10M")
        if DEEPSEEK_AVAILABLE:
            caps.append("DeepSeek")
        if POLYGLOT_AVAILABLE:
            caps.append("Polyglot")
        print(f"[Orchestrator] Available engines: {', '.join(caps)}")
    
    @property
    def research(self) -> "ResearchEngine":
        """Lazy-load research engine."""
        if self._research is None:
            if not RESEARCH_AVAILABLE:
                raise RuntimeError("Research engine not available")
            self._research = ResearchEngine(verbose=self.verbose)
        return self._research
    
    @property
    def scout(self) -> "ScoutReasoner":
        """Lazy-load Scout reasoner."""
        if self._scout is None:
            if SCOUT_AVAILABLE:
                self._scout = ScoutReasoner(verbose=self.verbose)
            else:
                # Create a mock scout for graceful degradation
                self._scout = MockScoutReasoner()
        return self._scout
    
    @property
    def deepseek(self) -> "DeepSeekEngine":
        """Lazy-load DeepSeek engine."""
        if self._deepseek is None:
            if DEEPSEEK_AVAILABLE:
                self._deepseek = DeepSeekEngine(verbose=self.verbose)
            else:
                # Create a mock deepseek for graceful degradation
                self._deepseek = MockDeepSeekEngine()
        return self._deepseek
    
    @property
    def polyglot(self) -> "PolyglotDispatcher":
        """Lazy-load Polyglot dispatcher."""
        if self._polyglot is None:
            if POLYGLOT_AVAILABLE:
                self._polyglot = PolyglotDispatcher(verbose=self.verbose)
            else:
                # Create a mock polyglot for graceful degradation
                self._polyglot = MockPolyglotDispatcher()
        return self._polyglot
    
    def register_handler(self, name: str, handler: Callable):
        """Register a custom step handler."""
        self._custom_handlers[name] = handler
        if self.verbose:
            print(f"[Orchestrator] Registered custom handler: {name}")
    
    def _compute_cache_key(self, step: Step, context: Dict[str, Any]) -> str:
        """Compute cache key for a step."""
        # Filter out non-serializable params (like functions)
        serializable_params = {}
        for k, v in step.params.items():
            if callable(v):
                serializable_params[k] = f"<function:{id(v)}>"
            elif isinstance(v, (str, int, float, bool, list, dict, type(None))):
                serializable_params[k] = v
            else:
                serializable_params[k] = str(v)[:100]
        
        key_data = {
            "type": step.step_type.value,
            "params": serializable_params,
            "context_refs": {k: str(v)[:100] for k, v in context.items() 
                           if k in step.depends_on}
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _check_cache(self, cache_key: str) -> Optional[Any]:
        """Check if result is cached."""
        if not self.cache_enabled:
            return None
        return self._cache.get(cache_key)
    
    def _store_cache(self, cache_key: str, result: Any):
        """Store result in cache."""
        if self.cache_enabled:
            self._cache[cache_key] = result
    
    def _execute_research_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a research step."""
        action = step.params.get("action", "search")
        
        if action == "search":
            query = step.params.get("query", "")
            # Interpolate context variables
            for key, value in context.items():
                query = query.replace(f"${{{key}}}", str(value))
            max_results = step.params.get("max_results", 5)
            return self.research.search(query, max_results=max_results)
        
        elif action == "fetch":
            identifier = step.params.get("identifier", "")
            return self.research.fetch(identifier)
        
        elif action == "analyze":
            paper_id = step.params.get("paper_id")
            if paper_id and paper_id in context:
                paper = context[paper_id]
                return self.research.analyze(paper)
            return None
        
        elif action == "review":
            topic = step.params.get("topic", "")
            max_papers = step.params.get("max_papers", 5)
            return self.research.literature_review(topic, max_papers=max_papers)
        
        else:
            raise ValueError(f"Unknown research action: {action}")
    
    def _execute_scout_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a Scout reasoning step."""
        mode_str = step.params.get("mode", "reason")
        mode = ScoutMode[mode_str.upper()] if hasattr(ScoutMode, mode_str.upper()) else ScoutMode.REASON
        
        prompt = step.params.get("prompt", "")
        
        # Interpolate context variables
        for key, value in context.items():
            if isinstance(value, str):
                prompt = prompt.replace(f"${{{key}}}", value)
            elif hasattr(value, "__str__"):
                prompt = prompt.replace(f"${{{key}}}", str(value)[:500])
        
        # Load context from previous steps if specified
        context_sources = step.params.get("context_sources", [])
        for source in context_sources:
            if source in context:
                data = context[source]
                if hasattr(data, "full_text"):
                    self.scout.context.add_chunk(data.full_text, source, "research")
                elif isinstance(data, str):
                    self.scout.context.add_chunk(data, source, "text")
        
        return self.scout.reason(prompt, mode=mode)
    
    def _execute_deepseek_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a DeepSeek code generation step."""
        action = step.params.get("action", "generate")
        
        if action == "generate":
            prompt = step.params.get("prompt", "")
            language = step.params.get("language", "python")
            
            # Interpolate context
            for key, value in context.items():
                if isinstance(value, str):
                    prompt = prompt.replace(f"${{{key}}}", value)
            
            return self.deepseek.generate(prompt, language=language)
        
        elif action == "optimize":
            code_ref = step.params.get("code_ref", "")
            if code_ref in context:
                code = context[code_ref]
                if hasattr(code, "code"):
                    code = code.code
                return self.deepseek.optimize(code)
            return None
        
        elif action == "test":
            code_ref = step.params.get("code_ref", "")
            if code_ref in context:
                code = context[code_ref]
                if hasattr(code, "code"):
                    code = code.code
                return self.deepseek.generate_tests(code)
            return None
        
        else:
            raise ValueError(f"Unknown deepseek action: {action}")
    
    def _execute_polyglot_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a Polyglot computation step."""
        import numpy as np
        
        operation = step.params.get("operation", "")
        
        if operation == "matmul":
            a_ref = step.params.get("a", "")
            b_ref = step.params.get("b", "")
            a = context.get(a_ref, np.eye(3))
            b = context.get(b_ref, np.eye(3))
            return self.polyglot.matmul(a, b)
        
        elif operation == "eigen":
            m_ref = step.params.get("matrix", "")
            m = context.get(m_ref, np.eye(3))
            return self.polyglot.eigensolve(m)
        
        elif operation == "solve":
            a_ref = step.params.get("a", "")
            b_ref = step.params.get("b", "")
            a = context.get(a_ref, np.eye(3))
            b = context.get(b_ref, np.ones(3))
            return self.polyglot.solve(a, b)
        
        elif operation == "fft":
            data_ref = step.params.get("data", "")
            data = context.get(data_ref, np.zeros(64))
            return self.polyglot.fft(data)
        
        else:
            raise ValueError(f"Unknown polyglot operation: {operation}")
    
    def _execute_validate_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a validation step."""
        checks = step.params.get("checks", [])
        results = {}
        
        # Safe builtins for validation expressions
        safe_builtins = {
            "len": len, "abs": abs, "min": min, "max": max,
            "sum": sum, "all": all, "any": any, "round": round,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "set": set, "tuple": tuple,
            "True": True, "False": False, "None": None,
            "isinstance": isinstance, "type": type,
        }
        
        # Merge context with safe builtins
        eval_globals = {"__builtins__": safe_builtins}
        eval_locals = dict(context)
        
        for check in checks:
            try:
                # Evaluate check expression with context
                result = eval(check, eval_globals, eval_locals)
                results[check] = {"passed": bool(result), "value": result}
            except Exception as e:
                results[check] = {"passed": False, "error": str(e)}
        
        all_passed = all(r.get("passed", False) for r in results.values())
        return {"passed": all_passed, "checks": results}
    
    def _execute_custom_step(self, step: Step, context: Dict[str, Any]) -> Any:
        """Execute a custom function step."""
        func = step.params.get("function")
        if callable(func):
            return func(context, **{k: v for k, v in step.params.items() if k != "function"})
        
        # Check registered handlers
        handler_name = step.params.get("handler")
        if handler_name and handler_name in self._custom_handlers:
            return self._custom_handlers[handler_name](context, **step.params)
        
        raise ValueError("No callable function or registered handler for custom step")
    
    def _execute_parallel_step(self, step: Step, context: Dict[str, Any], 
                                run: ExperimentRun) -> Dict[str, Any]:
        """Execute parallel sub-steps."""
        sub_steps = step.params.get("sub_steps", [])
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_parallel) as executor:
            futures = {}
            for sub_step in sub_steps:
                future = executor.submit(self._execute_step, sub_step, context, run)
                futures[future] = sub_step.name
            
            for future in as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    results[name] = result
                except Exception as e:
                    results[name] = {"error": str(e)}
        
        return results
    
    def _execute_step(self, step: Step, context: Dict[str, Any], 
                      run: ExperimentRun) -> StepResult:
        """Execute a single pipeline step."""
        start_time = datetime.now()
        
        if self.verbose:
            print(f"[Step] Executing: {step.name} ({step.step_type.value})")
        
        # Check condition
        if step.condition:
            try:
                if not eval(step.condition, {"__builtins__": {}}, context):
                    return StepResult(
                        step_id=step.id,
                        step_name=step.name,
                        status=StepStatus.SKIPPED,
                        start_time=start_time,
                        end_time=datetime.now()
                    )
            except Exception as e:
                return StepResult(
                    step_id=step.id,
                    step_name=step.name,
                    status=StepStatus.FAILED,
                    error=f"Condition evaluation failed: {e}",
                    start_time=start_time,
                    end_time=datetime.now()
                )
        
        # Check cache
        cache_key = self._compute_cache_key(step, context)
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            end_time = datetime.now()
            return StepResult(
                step_id=step.id,
                step_name=step.name,
                status=StepStatus.CACHED,
                output=cached_result,
                start_time=start_time,
                end_time=end_time,
                duration_ms=(end_time - start_time).total_seconds() * 1000,
                cached=True
            )
        
        # Execute based on type
        output = None
        error = None
        status = StepStatus.COMPLETED
        
        try:
            if step.step_type == StepType.RESEARCH:
                output = self._execute_research_step(step, context)
            elif step.step_type == StepType.SCOUT:
                output = self._execute_scout_step(step, context)
            elif step.step_type == StepType.DEEPSEEK:
                output = self._execute_deepseek_step(step, context)
            elif step.step_type == StepType.POLYGLOT:
                output = self._execute_polyglot_step(step, context)
            elif step.step_type == StepType.VALIDATE:
                output = self._execute_validate_step(step, context)
            elif step.step_type == StepType.CUSTOM:
                output = self._execute_custom_step(step, context)
            elif step.step_type == StepType.PARALLEL:
                output = self._execute_parallel_step(step, context, run)
            elif step.step_type == StepType.CHECKPOINT:
                output = self._save_checkpoint(step, context, run)
            else:
                raise ValueError(f"Unknown step type: {step.step_type}")
            
            # Store in cache
            self._store_cache(cache_key, output)
            
        except Exception as e:
            status = StepStatus.FAILED
            error = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            if self.verbose:
                print(f"[Step] FAILED: {step.name} - {error}")
        
        end_time = datetime.now()
        duration_ms = (end_time - start_time).total_seconds() * 1000
        
        result = StepResult(
            step_id=step.id,
            step_name=step.name,
            status=status,
            output=output,
            error=error,
            start_time=start_time,
            end_time=end_time,
            duration_ms=duration_ms
        )
        
        if self.verbose:
            status_icon = "OK" if status == StepStatus.COMPLETED else "ERROR"
            print(f"[Step] [{status_icon}] {step.name} ({duration_ms:.1f}ms)")
        
        return result
    
    def _save_checkpoint(self, step: Step, context: Dict[str, Any], 
                         run: ExperimentRun) -> Dict[str, Any]:
        """Save a checkpoint."""
        checkpoint_data = {
            "step_name": step.name,
            "timestamp": datetime.now().isoformat(),
            "context_keys": list(context.keys()),
            "completed_steps": [s for s, r in run.results.items() 
                               if r.status == StepStatus.COMPLETED]
        }
        run.checkpoints[step.name] = checkpoint_data
        
        # Save to disk
        checkpoint_file = self.workspace_dir / f"{run.run_id}_checkpoint_{step.name}.json"
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        return checkpoint_data
    
    def _topological_sort(self, steps: List[Step]) -> List[Step]:
        """Sort steps by dependencies (topological sort)."""
        # Build adjacency list
        graph = {step.name: set(step.depends_on) for step in steps}
        step_map = {step.name: step for step in steps}
        
        # Kahn's algorithm
        in_degree = {name: len(deps) for name, deps in graph.items()}
        queue = [name for name, deg in in_degree.items() if deg == 0]
        sorted_steps = []
        
        while queue:
            name = queue.pop(0)
            if name in step_map:
                sorted_steps.append(step_map[name])
            
            for other_name, deps in graph.items():
                if name in deps:
                    in_degree[other_name] -= 1
                    if in_degree[other_name] == 0:
                        queue.append(other_name)
        
        if len(sorted_steps) != len(steps):
            raise ValueError("Circular dependency detected in pipeline")
        
        return sorted_steps
    
    def run(self, pipeline: Pipeline, 
            resume_from: str = None,
            initial_context: Dict[str, Any] = None) -> ExperimentRun:
        """
        Execute a pipeline.
        
        Args:
            pipeline: The pipeline to execute
            resume_from: Checkpoint name to resume from
            initial_context: Initial context variables
            
        Returns:
            ExperimentRun with all results
        """
        run = ExperimentRun(pipeline=pipeline)
        run.started_at = datetime.now()
        run.status = "running"
        run.context = initial_context or {}
        run.context.update(pipeline.variables)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Orchestrator] Starting pipeline: {pipeline.name}")
            print(f"[Orchestrator] Run ID: {run.run_id}")
            print(f"[Orchestrator] Steps: {len(pipeline.steps)}")
            print(f"{'='*60}\n")
        
        # Sort steps by dependencies
        try:
            sorted_steps = self._topological_sort(pipeline.steps)
        except ValueError as e:
            run.status = "failed"
            run.completed_at = datetime.now()
            print(f"[Orchestrator] ERROR: {e}")
            return run
        
        # Find resume point if specified
        start_index = 0
        if resume_from:
            for i, step in enumerate(sorted_steps):
                if step.name == resume_from:
                    start_index = i
                    break
        
        # Execute steps
        for i, step in enumerate(sorted_steps[start_index:], start=start_index):
            # Check dependencies completed
            deps_satisfied = all(
                dep in run.results and run.results[dep].status == StepStatus.COMPLETED
                for dep in step.depends_on
            )
            
            if not deps_satisfied and step.depends_on:
                result = StepResult(
                    step_id=step.id,
                    step_name=step.name,
                    status=StepStatus.SKIPPED,
                    error="Dependencies not satisfied"
                )
            else:
                result = self._execute_step(step, run.context, run)
            
            run.results[step.name] = result
            
            # Store output in context for downstream steps
            if result.output is not None:
                run.context[step.name] = result.output
            
            # Stop on failure unless retry
            if result.status == StepStatus.FAILED:
                if step.retry_count > 0:
                    for retry in range(step.retry_count):
                        if self.verbose:
                            print(f"[Orchestrator] Retrying {step.name} ({retry+1}/{step.retry_count})")
                        result = self._execute_step(step, run.context, run)
                        run.results[step.name] = result
                        if result.status == StepStatus.COMPLETED:
                            run.context[step.name] = result.output
                            break
                
                if result.status == StepStatus.FAILED:
                    run.status = "failed"
                    break
        
        run.completed_at = datetime.now()
        if run.status == "running":
            run.status = "completed"
        
        # Store run
        self._runs.append(run)
        
        # Save run summary
        self._save_run_summary(run)
        
        if self.verbose:
            self._print_run_summary(run)
        
        return run
    
    def _save_run_summary(self, run: ExperimentRun):
        """Save run summary to disk."""
        summary_file = self.workspace_dir / f"{run.run_id}_summary.json"
        with open(summary_file, "w") as f:
            json.dump(run.to_dict(), f, indent=2)
    
    def _print_run_summary(self, run: ExperimentRun):
        """Print run summary."""
        print(f"\n{'='*60}")
        print(f"[Orchestrator] Pipeline Complete: {run.pipeline.name}")
        print(f"[Orchestrator] Status: {run.status.upper()}")
        if run.started_at and run.completed_at:
            duration = (run.completed_at - run.started_at).total_seconds()
            print(f"[Orchestrator] Duration: {duration:.2f}s")
        
        print(f"\n[Results]")
        for name, result in run.results.items():
            icon = {
                StepStatus.COMPLETED: "OK",
                StepStatus.FAILED: "ERROR",
                StepStatus.SKIPPED: "SKIP",
                StepStatus.CACHED: "CACHED"
            }.get(result.status, "?")
            print(f"  [{icon}] {name}: {result.duration_ms:.1f}ms")
        print(f"{'='*60}\n")
    
    def get_runs(self) -> List[ExperimentRun]:
        """Get all runs."""
        return self._runs
    
    def clear_cache(self):
        """Clear the result cache."""
        self._cache.clear()
        if self.verbose:
            print("[Orchestrator] Cache cleared")


# ============================================================================
# Pre-built Pipeline Templates
# ============================================================================

def create_research_pipeline(topic: str, max_papers: int = 5) -> Pipeline:
    """Create a standard research pipeline."""
    pipeline = Pipeline(
        name=f"research_{topic.replace(' ', '_')[:20]}",
        description=f"Automated research pipeline for: {topic}"
    )
    
    pipeline.add_research_step("search", topic, max_results=max_papers)
    pipeline.add_scout_step(
        "analyze", 
        f"Analyze the key findings from research on: {topic}. "
        f"Context: ${{search}}",
        mode="reason",
        depends_on=["search"]
    )
    pipeline.add_scout_step(
        "hypothesize",
        f"Generate novel research hypotheses based on the analysis. Context: ${{analyze}}",
        mode="hypothesize",
        depends_on=["analyze"]
    )
    pipeline.add_checkpoint("checkpoint_1", depends_on=["hypothesize"])
    
    return pipeline


def create_implementation_pipeline(description: str, language: str = "python") -> Pipeline:
    """Create a code implementation pipeline."""
    pipeline = Pipeline(
        name=f"implement_{language}",
        description=f"Implementation pipeline: {description[:50]}"
    )
    
    pipeline.add_scout_step(
        "design",
        f"Design the architecture for: {description}. "
        f"Specify the key components, data structures, and algorithms.",
        mode="reason"
    )
    pipeline.add_deepseek_step(
        "generate",
        f"Implement based on this design: ${{design}}. Requirements: {description}",
        language=language,
        depends_on=["design"]
    )
    pipeline.add_deepseek_step(
        "test",
        "Generate comprehensive tests for: ${generate}",
        language=language,
        depends_on=["generate"]
    )
    pipeline.add_validation_step(
        "validate",
        ["'def ' in str(generate) or 'function' in str(generate)"],
        depends_on=["generate"]
    )
    
    return pipeline


def create_analysis_pipeline(data_source: str) -> Pipeline:
    """Create a data analysis pipeline."""
    pipeline = Pipeline(
        name="data_analysis",
        description=f"Analysis pipeline for: {data_source}"
    )
    
    pipeline.add_scout_step(
        "understand",
        f"Analyze and describe the structure of: {data_source}",
        mode="reason"
    )
    pipeline.add_deepseek_step(
        "preprocess",
        "Generate data preprocessing code based on: ${understand}",
        language="python",
        depends_on=["understand"]
    )
    pipeline.add_deepseek_step(
        "analyze",
        "Generate statistical analysis code for the preprocessed data",
        language="python",
        depends_on=["preprocess"]
    )
    pipeline.add_deepseek_step(
        "visualize",
        "Generate visualization code for the analysis results",
        language="python",
        depends_on=["analyze"]
    )
    
    return pipeline


def create_proof_pipeline(theorem: str) -> Pipeline:
    """Create a formal proof pipeline."""
    pipeline = Pipeline(
        name="formal_proof",
        description=f"Proof pipeline for: {theorem[:50]}"
    )
    
    pipeline.add_scout_step(
        "decompose",
        f"Decompose this theorem into lemmas: {theorem}",
        mode="reason"
    )
    pipeline.add_scout_step(
        "prove_strategy",
        "Outline a proof strategy based on: ${decompose}",
        mode="prove",
        depends_on=["decompose"]
    )
    pipeline.add_deepseek_step(
        "formalize",
        "Generate Lean 4 proof code for: ${prove_strategy}",
        language="lean",
        depends_on=["prove_strategy"]
    )
    
    return pipeline


# ============================================================================
# Q-Lang Integration
# ============================================================================

def execute_orchestrator_command(interpreter, command: str) -> str:
    """Execute orchestrator commands from Q-Lang."""
    parts = command.strip().split()
    if not parts:
        return "Usage: orchestrator <command> [args]"
    
    cmd = parts[0].lower()
    args = parts[1:]
    
    # Get or create orchestrator
    if not hasattr(interpreter, "_orchestrator"):
        interpreter._orchestrator = ExperimentOrchestrator(verbose=True)
    
    orch = interpreter._orchestrator
    
    if cmd == "status":
        caps = []
        if RESEARCH_AVAILABLE: caps.append("Research")
        if SCOUT_AVAILABLE: caps.append("Scout-10M")
        if DEEPSEEK_AVAILABLE: caps.append("DeepSeek")
        if POLYGLOT_AVAILABLE: caps.append("Polyglot")
        
        return f"""
QENEX Experiment Orchestrator
=============================
Engines: {', '.join(caps)}
Cache: {'enabled' if orch.cache_enabled else 'disabled'}
Runs: {len(orch._runs)}
Workspace: {orch.workspace_dir}
"""
    
    elif cmd == "research" and args:
        topic = " ".join(args)
        pipeline = create_research_pipeline(topic)
        run = orch.run(pipeline)
        return f"Research pipeline completed: {run.status}"
    
    elif cmd == "implement" and args:
        desc = " ".join(args)
        pipeline = create_implementation_pipeline(desc)
        run = orch.run(pipeline)
        return f"Implementation pipeline completed: {run.status}"
    
    elif cmd == "prove" and args:
        theorem = " ".join(args)
        pipeline = create_proof_pipeline(theorem)
        run = orch.run(pipeline)
        return f"Proof pipeline completed: {run.status}"
    
    elif cmd == "runs":
        if not orch._runs:
            return "No runs yet."
        lines = ["Recent Runs:"]
        for run in orch._runs[-5:]:
            lines.append(f"  {run.run_id}: {run.pipeline.name} [{run.status}]")
        return "\n".join(lines)
    
    elif cmd == "clear-cache":
        orch.clear_cache()
        return "Cache cleared."
    
    elif cmd == "help":
        return """
Orchestrator Commands:
  status              - Show orchestrator status
  research <topic>    - Run research pipeline
  implement <desc>    - Run implementation pipeline  
  prove <theorem>     - Run proof pipeline
  runs                - List recent runs
  clear-cache         - Clear result cache
  help                - Show this help
"""
    
    else:
        return f"Unknown command: {cmd}. Use 'orchestrator help' for options."


# ============================================================================
# Main / Demo
# ============================================================================

if __name__ == "__main__":
    print("QENEX Experiment Orchestrator Demo")
    print("=" * 50)
    
    # Create orchestrator
    orch = ExperimentOrchestrator(verbose=True)
    
    # Demo 1: Simple research pipeline
    print("\n[Demo 1] Research Pipeline")
    print("-" * 40)
    
    pipeline = Pipeline(
        name="quantum_computing_research",
        description="Research quantum error correction"
    )
    
    pipeline.add_research_step("search", "quantum error correction", max_results=3)
    pipeline.add_scout_step(
        "analyze",
        "What are the key challenges in quantum error correction?",
        mode="reason",
        depends_on=["search"]
    )
    pipeline.add_checkpoint("checkpoint_1", depends_on=["analyze"])
    
    run = orch.run(pipeline)
    
    print(f"\nPipeline JSON:\n{pipeline.to_json()[:500]}...")
    
    # Demo 2: Custom pipeline with function
    print("\n[Demo 2] Custom Function Pipeline")
    print("-" * 40)
    
    def custom_analysis(context, **kwargs):
        """Custom analysis function."""
        return {
            "context_keys": list(context.keys()),
            "analysis": "Custom analysis complete"
        }
    
    pipeline2 = Pipeline(name="custom_demo")
    pipeline2.add_scout_step("think", "What is 2+2?")
    pipeline2.add_custom_step("custom", custom_analysis, depends_on=["think"])
    
    run2 = orch.run(pipeline2)
    
    # Demo 3: Using template
    print("\n[Demo 3] Template Pipeline")
    print("-" * 40)
    
    impl_pipeline = create_implementation_pipeline(
        "Implement a binary search tree with insert, delete, and search operations",
        language="python"
    )
    
    run3 = orch.run(impl_pipeline)
    
    print("\n[Demo Complete]")
    print(f"Total runs: {len(orch.get_runs())}")
