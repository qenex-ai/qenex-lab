"""
Tests for QENEX Experiment Orchestrator.

Tests cover:
- Pipeline creation and serialization
- Step execution and dependencies
- Caching and checkpoints
- Pre-built templates
- Q-Lang integration
"""

import pytest
import sys
import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "qenex-qlang", "src"))

from orchestrator import (
    ExperimentOrchestrator,
    Pipeline,
    Step,
    StepType,
    StepStatus,
    StepResult,
    ExperimentRun,
    create_research_pipeline,
    create_implementation_pipeline,
    create_analysis_pipeline,
    create_proof_pipeline,
    execute_orchestrator_command,
)


# ============================================================================
# Test Step and StepResult
# ============================================================================

class TestStep:
    """Tests for Step dataclass."""
    
    def test_step_creation(self):
        """Test basic step creation."""
        step = Step(name="test_step", step_type=StepType.SCOUT)
        assert step.name == "test_step"
        assert step.step_type == StepType.SCOUT
        assert step.params == {}
        assert step.depends_on == []
    
    def test_step_with_params(self):
        """Test step with parameters."""
        step = Step(
            name="research",
            step_type=StepType.RESEARCH,
            params={"query": "quantum computing", "max_results": 5}
        )
        assert step.params["query"] == "quantum computing"
        assert step.params["max_results"] == 5
    
    def test_step_with_string_type(self):
        """Test step creation with string type."""
        step = Step(name="scout_step", step_type="scout")
        assert step.step_type == StepType.SCOUT
    
    def test_step_with_dependencies(self):
        """Test step with dependencies."""
        step = Step(
            name="analyze",
            step_type=StepType.SCOUT,
            depends_on=["search", "fetch"]
        )
        assert "search" in step.depends_on
        assert "fetch" in step.depends_on
    
    def test_step_has_unique_id(self):
        """Test that each step has a unique ID."""
        step1 = Step(name="test", step_type=StepType.SCOUT)
        step2 = Step(name="test", step_type=StepType.SCOUT)
        assert step1.id != step2.id
    
    def test_step_to_dict(self):
        """Test step serialization."""
        step = Step(
            name="test",
            step_type=StepType.DEEPSEEK,
            params={"prompt": "Generate code"},
            depends_on=["prev_step"]
        )
        d = step.to_dict()
        assert d["name"] == "test"
        assert d["type"] == "deepseek"
        assert d["params"]["prompt"] == "Generate code"


class TestStepResult:
    """Tests for StepResult dataclass."""
    
    def test_result_creation(self):
        """Test basic result creation."""
        result = StepResult(
            step_id="step_123",
            step_name="test",
            status=StepStatus.COMPLETED,
            output="test output"
        )
        assert result.step_id == "step_123"
        assert result.status == StepStatus.COMPLETED
        assert result.output == "test output"
    
    def test_result_with_timing(self):
        """Test result with timing info."""
        start = datetime.now()
        result = StepResult(
            step_id="step_123",
            step_name="test",
            status=StepStatus.COMPLETED,
            start_time=start,
            end_time=datetime.now(),
            duration_ms=100.5
        )
        assert result.duration_ms == 100.5
        assert result.start_time == start
    
    def test_result_failed(self):
        """Test failed result."""
        result = StepResult(
            step_id="step_123",
            step_name="test",
            status=StepStatus.FAILED,
            error="Something went wrong"
        )
        assert result.status == StepStatus.FAILED
        assert "Something went wrong" in result.error
    
    def test_result_to_dict(self):
        """Test result serialization."""
        result = StepResult(
            step_id="step_123",
            step_name="test",
            status=StepStatus.COMPLETED,
            output={"key": "value"},
            duration_ms=50.0
        )
        d = result.to_dict()
        assert d["step_id"] == "step_123"
        assert d["status"] == "completed"
        assert d["duration_ms"] == 50.0


# ============================================================================
# Test Pipeline
# ============================================================================

class TestPipeline:
    """Tests for Pipeline class."""
    
    def test_pipeline_creation(self):
        """Test basic pipeline creation."""
        pipeline = Pipeline(name="test_pipeline", description="A test pipeline")
        assert pipeline.name == "test_pipeline"
        assert pipeline.description == "A test pipeline"
        assert len(pipeline.steps) == 0
    
    def test_pipeline_add_step(self):
        """Test adding steps to pipeline."""
        pipeline = Pipeline(name="test")
        pipeline.add_step(Step(name="step1", step_type=StepType.SCOUT))
        pipeline.add_step(Step(name="step2", step_type=StepType.DEEPSEEK))
        assert len(pipeline.steps) == 2
    
    def test_pipeline_chaining(self):
        """Test fluent interface for adding steps."""
        pipeline = (Pipeline(name="test")
            .add_step(Step(name="s1", step_type=StepType.SCOUT))
            .add_step(Step(name="s2", step_type=StepType.DEEPSEEK)))
        assert len(pipeline.steps) == 2
    
    def test_pipeline_add_research_step(self):
        """Test adding research step."""
        pipeline = Pipeline(name="test")
        pipeline.add_research_step("search", "quantum computing", max_results=10)
        
        step = pipeline.steps[0]
        assert step.name == "search"
        assert step.step_type == StepType.RESEARCH
        assert step.params["query"] == "quantum computing"
        assert step.params["max_results"] == 10
    
    def test_pipeline_add_scout_step(self):
        """Test adding scout step."""
        pipeline = Pipeline(name="test")
        pipeline.add_scout_step("analyze", "What is the meaning?", mode="reason")
        
        step = pipeline.steps[0]
        assert step.name == "analyze"
        assert step.step_type == StepType.SCOUT
        assert step.params["mode"] == "reason"
    
    def test_pipeline_add_deepseek_step(self):
        """Test adding deepseek step."""
        pipeline = Pipeline(name="test")
        pipeline.add_deepseek_step("generate", "Implement sorting", language="python")
        
        step = pipeline.steps[0]
        assert step.name == "generate"
        assert step.step_type == StepType.DEEPSEEK
        assert step.params["language"] == "python"
    
    def test_pipeline_add_compute_step(self):
        """Test adding polyglot compute step."""
        pipeline = Pipeline(name="test")
        pipeline.add_compute_step("multiply", "matmul", a="A", b="B")
        
        step = pipeline.steps[0]
        assert step.name == "multiply"
        assert step.step_type == StepType.POLYGLOT
        assert step.params["operation"] == "matmul"
    
    def test_pipeline_add_validation_step(self):
        """Test adding validation step."""
        pipeline = Pipeline(name="test")
        pipeline.add_validation_step("check", ["result > 0", "len(data) > 0"])
        
        step = pipeline.steps[0]
        assert step.name == "check"
        assert step.step_type == StepType.VALIDATE
        assert len(step.params["checks"]) == 2
    
    def test_pipeline_add_checkpoint(self):
        """Test adding checkpoint."""
        pipeline = Pipeline(name="test")
        pipeline.add_scout_step("step1", "Do something")
        pipeline.add_checkpoint("cp1", depends_on=["step1"])
        
        step = pipeline.steps[1]
        assert step.name == "cp1"
        assert step.step_type == StepType.CHECKPOINT
    
    def test_pipeline_variables(self):
        """Test pipeline variables."""
        pipeline = Pipeline(
            name="test",
            variables={"topic": "quantum", "max_papers": 10}
        )
        assert pipeline.variables["topic"] == "quantum"
        assert pipeline.variables["max_papers"] == 10
    
    def test_pipeline_to_dict(self):
        """Test pipeline serialization."""
        pipeline = Pipeline(name="test", description="Test pipeline")
        pipeline.add_scout_step("think", "What is 2+2?")
        
        d = pipeline.to_dict()
        assert d["name"] == "test"
        assert d["description"] == "Test pipeline"
        assert len(d["steps"]) == 1
    
    def test_pipeline_to_json(self):
        """Test pipeline JSON serialization."""
        pipeline = Pipeline(name="test")
        pipeline.add_scout_step("think", "Analyze this")
        
        json_str = pipeline.to_json()
        assert '"name": "test"' in json_str
        assert "think" in json_str
    
    def test_pipeline_from_dict(self):
        """Test pipeline deserialization."""
        data = {
            "name": "restored_pipeline",
            "description": "Restored from dict",
            "steps": [
                {"name": "step1", "type": "scout", "params": {"prompt": "Think"}}
            ],
            "variables": {"x": 1}
        }
        
        pipeline = Pipeline.from_dict(data)
        assert pipeline.name == "restored_pipeline"
        assert len(pipeline.steps) == 1
        assert pipeline.variables["x"] == 1
    
    def test_pipeline_from_json(self):
        """Test pipeline JSON deserialization."""
        json_str = '{"name": "json_pipeline", "description": "", "steps": [], "variables": {}}'
        pipeline = Pipeline.from_json(json_str)
        assert pipeline.name == "json_pipeline"
    
    def test_pipeline_roundtrip(self):
        """Test serialization roundtrip."""
        original = Pipeline(name="roundtrip_test", description="Testing roundtrip")
        original.add_scout_step("s1", "First step")
        original.add_deepseek_step("s2", "Second step", depends_on=["s1"])
        original.variables = {"key": "value"}
        
        json_str = original.to_json()
        restored = Pipeline.from_json(json_str)
        
        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.steps) == len(original.steps)
        assert restored.variables == original.variables


# ============================================================================
# Test ExperimentOrchestrator
# ============================================================================

class TestExperimentOrchestrator:
    """Tests for ExperimentOrchestrator."""
    
    def test_orchestrator_creation(self):
        """Test orchestrator creation."""
        orch = ExperimentOrchestrator(verbose=False)
        assert orch.cache_enabled == True
        assert orch.max_parallel == 4
    
    def test_orchestrator_custom_workspace(self, tmp_path):
        """Test orchestrator with custom workspace."""
        orch = ExperimentOrchestrator(
            workspace_dir=str(tmp_path / "experiments"),
            verbose=False
        )
        assert orch.workspace_dir.exists()
    
    def test_register_handler(self):
        """Test registering custom handler."""
        orch = ExperimentOrchestrator(verbose=False)
        
        def my_handler(context, **kwargs):
            return "handled"
        
        orch.register_handler("my_handler", my_handler)
        assert "my_handler" in orch._custom_handlers
    
    def test_cache_operations(self):
        """Test cache storage and retrieval."""
        orch = ExperimentOrchestrator(verbose=False)
        
        orch._store_cache("test_key", {"result": 42})
        cached = orch._check_cache("test_key")
        
        assert cached is not None
        assert cached["result"] == 42
    
    def test_clear_cache(self):
        """Test cache clearing."""
        orch = ExperimentOrchestrator(verbose=False)
        orch._store_cache("key1", "value1")
        orch._store_cache("key2", "value2")
        
        orch.clear_cache()
        
        assert orch._check_cache("key1") is None
        assert orch._check_cache("key2") is None
    
    def test_topological_sort(self):
        """Test dependency sorting."""
        orch = ExperimentOrchestrator(verbose=False)
        
        steps = [
            Step(name="c", step_type=StepType.SCOUT, depends_on=["a", "b"]),
            Step(name="a", step_type=StepType.SCOUT),
            Step(name="b", step_type=StepType.SCOUT, depends_on=["a"]),
        ]
        
        sorted_steps = orch._topological_sort(steps)
        names = [s.name for s in sorted_steps]
        
        assert names.index("a") < names.index("b")
        assert names.index("a") < names.index("c")
        assert names.index("b") < names.index("c")
    
    def test_topological_sort_circular_dependency(self):
        """Test circular dependency detection."""
        orch = ExperimentOrchestrator(verbose=False)
        
        steps = [
            Step(name="a", step_type=StepType.SCOUT, depends_on=["b"]),
            Step(name="b", step_type=StepType.SCOUT, depends_on=["a"]),
        ]
        
        with pytest.raises(ValueError, match="Circular dependency"):
            orch._topological_sort(steps)


class TestPipelineExecution:
    """Tests for pipeline execution."""
    
    def test_run_simple_pipeline(self):
        """Test running a simple pipeline."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="simple_test")
        pipeline.add_scout_step("think", "What is 2+2?")
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert "think" in run.results
        assert run.results["think"].status == StepStatus.COMPLETED
    
    def test_run_pipeline_with_dependencies(self):
        """Test pipeline with dependencies."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="dep_test")
        pipeline.add_scout_step("s1", "First")
        pipeline.add_scout_step("s2", "Second: ${s1}", depends_on=["s1"])
        pipeline.add_scout_step("s3", "Third: ${s2}", depends_on=["s2"])
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert len(run.results) == 3
    
    def test_run_pipeline_with_variables(self):
        """Test pipeline with context variables."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(
            name="var_test",
            variables={"topic": "quantum computing"}
        )
        pipeline.add_scout_step("analyze", "Analyze ${topic}")
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert "topic" in run.context
    
    def test_run_pipeline_with_initial_context(self):
        """Test pipeline with initial context."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="ctx_test")
        pipeline.add_scout_step("think", "Process ${data}")
        
        run = orch.run(pipeline, initial_context={"data": "test data"})
        
        assert "data" in run.context
        assert run.context["data"] == "test data"
    
    def test_run_pipeline_with_checkpoint(self, tmp_path):
        """Test pipeline with checkpoint."""
        orch = ExperimentOrchestrator(
            workspace_dir=str(tmp_path / "experiments"),
            verbose=False
        )
        
        pipeline = Pipeline(name="cp_test")
        pipeline.add_scout_step("s1", "Step 1")
        pipeline.add_checkpoint("cp1", depends_on=["s1"])
        
        run = orch.run(pipeline)
        
        assert "cp1" in run.checkpoints
        # Check checkpoint file was created
        checkpoint_files = list(orch.workspace_dir.glob("*checkpoint*.json"))
        assert len(checkpoint_files) > 0
    
    def test_run_pipeline_with_validation(self):
        """Test pipeline with validation step."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="val_test")
        pipeline.add_validation_step("check", ["1 + 1 == 2", "len('test') == 4"])
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        result = run.results["check"]
        assert result.output["passed"] == True
    
    def test_run_pipeline_validation_failure(self):
        """Test pipeline with failing validation."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="fail_val_test")
        pipeline.add_validation_step("check", ["1 + 1 == 3"])
        
        run = orch.run(pipeline)
        
        result = run.results["check"]
        assert result.output["passed"] == False
    
    def test_run_stores_run_history(self):
        """Test that runs are stored."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline1 = Pipeline(name="test1")
        pipeline1.add_scout_step("s1", "First")
        
        pipeline2 = Pipeline(name="test2")
        pipeline2.add_scout_step("s1", "Second")
        
        orch.run(pipeline1)
        orch.run(pipeline2)
        
        assert len(orch.get_runs()) == 2
    
    def test_run_with_custom_function(self):
        """Test pipeline with custom function step."""
        orch = ExperimentOrchestrator(verbose=False)
        
        def custom_func(context, multiplier=1, **kwargs):
            return {"value": 42 * multiplier}
        
        pipeline = Pipeline(name="custom_test")
        pipeline.add_custom_step("compute", custom_func, multiplier=2)
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert run.results["compute"].output["value"] == 84
    
    def test_run_with_registered_handler(self):
        """Test pipeline with registered handler."""
        orch = ExperimentOrchestrator(verbose=False)
        
        def my_handler(context, **kwargs):
            return "handler result"
        
        orch.register_handler("my_handler", my_handler)
        
        pipeline = Pipeline(name="handler_test")
        pipeline.add_step(Step(
            name="handled",
            step_type=StepType.CUSTOM,
            params={"handler": "my_handler"}
        ))
        
        run = orch.run(pipeline)
        
        assert run.results["handled"].output == "handler result"
    
    def test_conditional_step_executed(self):
        """Test conditional step that executes."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="cond_test", variables={"should_run": True})
        step = Step(
            name="conditional",
            step_type=StepType.SCOUT,
            params={"prompt": "Conditional step"},
            condition="should_run == True"
        )
        pipeline.add_step(step)
        
        run = orch.run(pipeline)
        
        assert run.results["conditional"].status == StepStatus.COMPLETED
    
    def test_conditional_step_skipped(self):
        """Test conditional step that is skipped."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="skip_test", variables={"should_run": False})
        step = Step(
            name="conditional",
            step_type=StepType.SCOUT,
            params={"prompt": "Conditional step"},
            condition="should_run == True"
        )
        pipeline.add_step(step)
        
        run = orch.run(pipeline)
        
        assert run.results["conditional"].status == StepStatus.SKIPPED


class TestExperimentRun:
    """Tests for ExperimentRun."""
    
    def test_run_creation(self):
        """Test run creation."""
        pipeline = Pipeline(name="test")
        run = ExperimentRun(pipeline=pipeline)
        
        assert run.pipeline == pipeline
        assert run.status == "pending"
        assert run.run_id.startswith("run_")
    
    def test_run_to_dict(self):
        """Test run serialization."""
        pipeline = Pipeline(name="test")
        run = ExperimentRun(pipeline=pipeline)
        run.started_at = datetime.now()
        run.status = "completed"
        
        d = run.to_dict()
        
        assert d["pipeline_name"] == "test"
        assert d["status"] == "completed"


# ============================================================================
# Test Pre-built Templates
# ============================================================================

class TestPipelineTemplates:
    """Tests for pre-built pipeline templates."""
    
    def test_create_research_pipeline(self):
        """Test research pipeline template."""
        pipeline = create_research_pipeline("quantum computing", max_papers=3)
        
        assert "research" in pipeline.name or "quantum" in pipeline.name
        assert len(pipeline.steps) >= 3  # search, analyze, hypothesize
        
        # First step should be research
        assert pipeline.steps[0].step_type == StepType.RESEARCH
    
    def test_create_implementation_pipeline(self):
        """Test implementation pipeline template."""
        pipeline = create_implementation_pipeline("Binary search tree", language="python")
        
        assert "implement" in pipeline.name
        assert len(pipeline.steps) >= 3  # design, generate, test
        
        # Should have deepseek steps
        deepseek_steps = [s for s in pipeline.steps if s.step_type == StepType.DEEPSEEK]
        assert len(deepseek_steps) >= 1
    
    def test_create_analysis_pipeline(self):
        """Test analysis pipeline template."""
        pipeline = create_analysis_pipeline("dataset.csv")
        
        assert pipeline.name == "data_analysis"
        assert len(pipeline.steps) >= 3
    
    def test_create_proof_pipeline(self):
        """Test proof pipeline template."""
        pipeline = create_proof_pipeline("Pythagorean theorem")
        
        assert "proof" in pipeline.name
        assert len(pipeline.steps) >= 2
        
        # Should have scout steps
        scout_steps = [s for s in pipeline.steps if s.step_type == StepType.SCOUT]
        assert len(scout_steps) >= 1


# ============================================================================
# Test Q-Lang Integration
# ============================================================================

class TestQLangIntegration:
    """Tests for Q-Lang integration."""
    
    def test_orchestrator_status_command(self):
        """Test status command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "status")
        
        assert "Experiment Orchestrator" in result
        assert "Engines:" in result
    
    def test_orchestrator_help_command(self):
        """Test help command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "help")
        
        assert "Commands:" in result
        assert "research" in result
        assert "implement" in result
    
    def test_orchestrator_runs_command_empty(self):
        """Test runs command with no runs."""
        mock_interpreter = MagicMock()
        # First call creates orchestrator, runs might be there from previous tests
        result = execute_orchestrator_command(mock_interpreter, "runs")
        
        # Should return either "No runs" or "Recent Runs:"
        assert "runs" in result.lower() or "No runs" in result
    
    def test_orchestrator_clear_cache_command(self):
        """Test clear-cache command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "clear-cache")
        
        assert "cleared" in result.lower()
    
    def test_orchestrator_unknown_command(self):
        """Test unknown command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "unknown_cmd")
        
        assert "Unknown command" in result
    
    def test_orchestrator_research_command(self):
        """Test research command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "research quantum error correction")
        
        assert "pipeline" in result.lower()
    
    def test_orchestrator_implement_command(self):
        """Test implement command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "implement binary search")
        
        assert "pipeline" in result.lower()
    
    def test_orchestrator_prove_command(self):
        """Test prove command."""
        mock_interpreter = MagicMock()
        result = execute_orchestrator_command(mock_interpreter, "prove Fermat's theorem")
        
        assert "pipeline" in result.lower()


# ============================================================================
# Test Polyglot Step Execution
# ============================================================================

class TestPolyglotStepExecution:
    """Tests for Polyglot step execution."""
    
    def test_matmul_step(self):
        """Test matrix multiplication step."""
        orch = ExperimentOrchestrator(verbose=False)
        
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        
        pipeline = Pipeline(name="matmul_test", variables={"A": A, "B": B})
        pipeline.add_compute_step("multiply", "matmul", a="A", b="B")
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        result = run.results["multiply"].output
        assert result is not None
    
    def test_eigen_step(self):
        """Test eigenvalue step."""
        orch = ExperimentOrchestrator(verbose=False)
        
        M = np.array([[1, 0], [0, 2]], dtype=float)
        
        pipeline = Pipeline(name="eigen_test", variables={"M": M})
        pipeline.add_compute_step("eigen", "eigen", matrix="M")
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_empty_pipeline(self):
        """Test running empty pipeline."""
        orch = ExperimentOrchestrator(verbose=False)
        pipeline = Pipeline(name="empty")
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert len(run.results) == 0
    
    def test_pipeline_with_all_step_types(self):
        """Test pipeline with all step types."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="all_types")
        pipeline.add_research_step("research", "test query")
        pipeline.add_scout_step("scout", "test prompt")
        pipeline.add_deepseek_step("deepseek", "test code")
        pipeline.add_validation_step("validate", ["True"])
        pipeline.add_checkpoint("checkpoint")
        
        run = orch.run(pipeline)
        
        # All steps should have been attempted
        assert len(run.results) == 5
    
    def test_step_with_retry(self):
        """Test step with retry configuration."""
        step = Step(
            name="retry_step",
            step_type=StepType.SCOUT,
            retry_count=3,
            timeout_seconds=60.0
        )
        
        assert step.retry_count == 3
        assert step.timeout_seconds == 60.0
    
    def test_large_pipeline(self):
        """Test pipeline with many steps."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(name="large")
        for i in range(20):
            deps = [f"s{i-1}"] if i > 0 else []
            pipeline.add_scout_step(f"s{i}", f"Step {i}", depends_on=deps)
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert len(run.results) == 20


# ============================================================================
# Test StepType Enum
# ============================================================================

class TestStepTypeEnum:
    """Tests for StepType enum."""
    
    def test_all_step_types_exist(self):
        """Test all step types are defined."""
        assert StepType.RESEARCH.value == "research"
        assert StepType.SCOUT.value == "scout"
        assert StepType.DEEPSEEK.value == "deepseek"
        assert StepType.POLYGLOT.value == "polyglot"
        assert StepType.VALIDATE.value == "validate"
        assert StepType.CUSTOM.value == "custom"
        assert StepType.PARALLEL.value == "parallel"
        assert StepType.CONDITIONAL.value == "conditional"
        assert StepType.LOOP.value == "loop"
        assert StepType.CHECKPOINT.value == "checkpoint"


class TestStepStatusEnum:
    """Tests for StepStatus enum."""
    
    def test_all_statuses_exist(self):
        """Test all statuses are defined."""
        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"
        assert StepStatus.CACHED.value == "cached"


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_research_workflow(self):
        """Test complete research workflow."""
        orch = ExperimentOrchestrator(verbose=False)
        
        pipeline = Pipeline(
            name="full_research",
            description="Complete research workflow test"
        )
        
        # Add steps
        pipeline.add_research_step("search", "machine learning optimization")
        pipeline.add_scout_step(
            "analyze",
            "Summarize key findings from: ${search}",
            depends_on=["search"]
        )
        pipeline.add_scout_step(
            "hypothesize",
            "Generate hypotheses based on: ${analyze}",
            mode="hypothesize",
            depends_on=["analyze"]
        )
        pipeline.add_deepseek_step(
            "implement",
            "Implement the first hypothesis: ${hypothesize}",
            depends_on=["hypothesize"]
        )
        pipeline.add_validation_step(
            "validate",
            ["'implement' in context"],
            depends_on=["implement"]
        )
        pipeline.add_checkpoint("final_checkpoint", depends_on=["validate"])
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert len(run.results) == 6
        assert "final_checkpoint" in run.checkpoints
    
    def test_context_propagation(self):
        """Test that context properly propagates between steps."""
        orch = ExperimentOrchestrator(verbose=False)
        
        def step1_func(context, **kwargs):
            return {"value": 10}
        
        def step2_func(context, **kwargs):
            step1_result = context.get("step1", {})
            return {"doubled": step1_result.get("value", 0) * 2}
        
        def step3_func(context, **kwargs):
            step2_result = context.get("step2", {})
            return {"final": step2_result.get("doubled", 0) + 5}
        
        pipeline = Pipeline(name="context_test")
        pipeline.add_custom_step("step1", step1_func)
        pipeline.add_custom_step("step2", step2_func, depends_on=["step1"])
        pipeline.add_custom_step("step3", step3_func, depends_on=["step2"])
        
        run = orch.run(pipeline)
        
        assert run.status == "completed"
        assert run.context["step1"]["value"] == 10
        assert run.context["step2"]["doubled"] == 20
        assert run.context["step3"]["final"] == 25


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
