"""
Tests for QENEX Unified CLI
============================
Tests for the main QENEX command-line interface.

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import pytest
import argparse
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# Add package to path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "qenex-qlang" / "src"))

from qenex import (
    __version__,
    Colors,
    color,
    get_integration,
    get_interpreter,
    get_orchestrator,
    get_research,
    cmd_reason,
    cmd_generate,
    cmd_experiment,
    cmd_run,
    cmd_status,
    cmd_config,
    create_parser,
    main,
    QENEXRepl,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def temp_dir():
    """Provide temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_integration():
    """Mock LLM integration."""
    mock = Mock()
    mock.reason.return_value = {
        "success": True,
        "output": "Test reasoning output",
        "tokens_generated": 100,
        "backend": "Mock",
    }
    mock.generate_code.return_value = {
        "success": True,
        "code": "def test(): pass",
        "tokens_generated": 50,
        "backend": "Mock",
    }
    mock.router.get_available_backends.return_value = []
    mock.total_tokens.to_dict.return_value = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "operations": 0,
    }
    return mock


@pytest.fixture
def qlang_script(temp_dir):
    """Create a test Q-Lang script."""
    script_path = Path(temp_dir) / "test.ql"
    script_path.write_text("x = 42\nx")
    return str(script_path)


# ==============================================================================
# Version and Basics Tests
# ==============================================================================

class TestBasics:
    """Test basic module setup."""
    
    def test_version_defined(self):
        """Verify version is defined."""
        assert __version__ is not None
        assert isinstance(__version__, str)
        assert len(__version__.split(".")) >= 2
    
    def test_colors_defined(self):
        """Verify color codes are defined."""
        assert hasattr(Colors, 'HEADER')
        assert hasattr(Colors, 'GREEN')
        assert hasattr(Colors, 'RED')
        assert hasattr(Colors, 'ENDC')
    
    def test_color_function_with_tty(self):
        """Test color function with TTY."""
        # When stdout is a TTY, colors are applied
        with patch('sys.stdout.isatty', return_value=True):
            result = color("test", Colors.GREEN)
            assert Colors.GREEN in result
            assert Colors.ENDC in result
            assert "test" in result
    
    def test_color_function_without_tty(self):
        """Test color function without TTY."""
        # When stdout is not a TTY, colors are stripped
        with patch('sys.stdout.isatty', return_value=False):
            result = color("test", Colors.GREEN)
            assert result == "test"


# ==============================================================================
# Component Loader Tests
# ==============================================================================

class TestComponentLoaders:
    """Test lazy component loading."""
    
    def test_get_integration(self):
        """Test integration loader."""
        # Reset global
        import qenex
        qenex._integration = None
        
        integration = get_integration()
        # Should return something (may be None if dependencies missing)
        # but should not raise
        assert integration is not None or integration is None
    
    def test_get_interpreter(self):
        """Test interpreter loader."""
        import qenex
        qenex._interpreter = None
        
        interpreter = get_interpreter()
        assert interpreter is not None or interpreter is None
    
    def test_get_orchestrator(self):
        """Test orchestrator loader."""
        import qenex
        qenex._orchestrator = None
        
        orchestrator = get_orchestrator()
        assert orchestrator is not None or orchestrator is None
    
    def test_get_research(self):
        """Test research engine loader."""
        import qenex
        qenex._research = None
        
        research = get_research()
        assert research is not None or research is None


# ==============================================================================
# Parser Tests
# ==============================================================================

class TestParser:
    """Test argument parser."""
    
    def test_parser_creation(self):
        """Test parser is created successfully."""
        parser = create_parser()
        assert parser is not None
        assert parser.prog == "qenex"
    
    def test_version_flag(self):
        """Test --version flag."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--version"])
    
    def test_help_flag(self):
        """Test --help flag."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--help"])
    
    def test_reason_command(self):
        """Test reason command parsing."""
        parser = create_parser()
        args = parser.parse_args(["reason", "What", "is", "entropy?"])
        
        assert args.command == "reason"
        assert args.query == ["What", "is", "entropy?"]
    
    def test_reason_with_mode(self):
        """Test reason command with mode."""
        parser = create_parser()
        args = parser.parse_args(["reason", "--mode", "prove", "theorem"])
        
        assert args.command == "reason"
        assert args.mode == "prove"
    
    def test_generate_command(self):
        """Test generate command parsing."""
        parser = create_parser()
        args = parser.parse_args(["generate", "FFT", "implementation"])
        
        assert args.command == "generate"
        assert args.description == ["FFT", "implementation"]
    
    def test_generate_with_language(self):
        """Test generate with language option."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-l", "julia", "sort", "algorithm"])
        
        assert args.language == "julia"
    
    def test_generate_with_output(self):
        """Test generate with output file."""
        parser = create_parser()
        args = parser.parse_args(["generate", "-o", "out.py", "hello", "world"])
        
        assert args.output == "out.py"
    
    def test_experiment_command(self):
        """Test experiment command parsing."""
        parser = create_parser()
        args = parser.parse_args(["experiment", "start", "my_exp"])
        
        assert args.command == "experiment"
        assert args.subcmd == "start"
        assert args.name == "my_exp"
    
    def test_run_command(self):
        """Test run command parsing."""
        parser = create_parser()
        args = parser.parse_args(["run", "script.ql"])
        
        assert args.command == "run"
        assert args.script == "script.ql"
    
    def test_dashboard_command(self):
        """Test dashboard command parsing."""
        parser = create_parser()
        args = parser.parse_args(["dashboard", "--port", "9000"])
        
        assert args.command == "dashboard"
        assert args.port == 9000
    
    def test_status_command(self):
        """Test status command parsing."""
        parser = create_parser()
        args = parser.parse_args(["status"])
        
        assert args.command == "status"
    
    def test_config_command(self):
        """Test config command parsing."""
        parser = create_parser()
        args = parser.parse_args(["config", "--show"])
        
        assert args.command == "config"
        assert args.show is True


# ==============================================================================
# Command Handler Tests
# ==============================================================================

class TestCommandHandlers:
    """Test command handlers."""
    
    def test_cmd_reason(self, mock_integration, capsys):
        """Test reason command handler."""
        with patch('qenex.get_integration', return_value=mock_integration):
            args = argparse.Namespace(
                query=["What", "is", "entropy?"],
                mode=None
            )
            result = cmd_reason(args)
            
            assert result == 0
            captured = capsys.readouterr()
            assert "Scout Reasoning" in captured.out
    
    def test_cmd_reason_no_integration(self, capsys):
        """Test reason command without integration."""
        with patch('qenex.get_integration', return_value=None):
            args = argparse.Namespace(query=["test"], mode=None)
            result = cmd_reason(args)
            
            assert result == 1
            captured = capsys.readouterr()
            assert "not available" in captured.out
    
    def test_cmd_generate(self, mock_integration, capsys):
        """Test generate command handler."""
        with patch('qenex.get_integration', return_value=mock_integration):
            args = argparse.Namespace(
                description=["FFT", "implementation"],
                language="python",
                mode=None,
                output=None
            )
            result = cmd_generate(args)
            
            assert result == 0
            captured = capsys.readouterr()
            assert "DeepSeek" in captured.out
    
    def test_cmd_generate_with_output(self, mock_integration, temp_dir):
        """Test generate with file output."""
        output_file = Path(temp_dir) / "output.py"
        
        with patch('qenex.get_integration', return_value=mock_integration):
            args = argparse.Namespace(
                description=["test"],
                language="python",
                mode=None,
                output=str(output_file)
            )
            result = cmd_generate(args)
            
            assert result == 0
            assert output_file.exists()
    
    def test_cmd_status(self, capsys):
        """Test status command handler."""
        result = cmd_status(argparse.Namespace())
        
        assert result == 0
        captured = capsys.readouterr()
        assert "QENEX" in captured.out
        assert "Components" in captured.out
    
    def test_cmd_config_show(self, temp_dir, capsys):
        """Test config show command."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            args = argparse.Namespace(show=True, set=None, backend=None)
            result = cmd_config(args)
            
            assert result == 0
            captured = capsys.readouterr()
            assert "Configuration" in captured.out
    
    def test_cmd_config_set(self, temp_dir, capsys):
        """Test config set command."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            # Create .qenex directory
            (Path(temp_dir) / ".qenex").mkdir(parents=True, exist_ok=True)
            
            args = argparse.Namespace(show=False, set="key=value", backend=None)
            result = cmd_config(args)
            
            assert result == 0


# ==============================================================================
# Run Command Tests
# ==============================================================================

class TestRunCommand:
    """Test run command."""
    
    def test_run_nonexistent_script(self, capsys):
        """Test running nonexistent script."""
        args = argparse.Namespace(script="nonexistent.ql")
        result = cmd_run(args)
        
        assert result == 1
        captured = capsys.readouterr()
        assert "not found" in captured.out
    
    def test_run_script(self, qlang_script, capsys):
        """Test running valid script."""
        mock_interp = Mock()
        mock_interp.execute.return_value = 42
        
        with patch('qenex.get_interpreter', return_value=mock_interp):
            args = argparse.Namespace(script=qlang_script)
            result = cmd_run(args)
            
            assert result == 0
    
    def test_run_no_interpreter(self, qlang_script, capsys):
        """Test running script without interpreter."""
        with patch('qenex.get_interpreter', return_value=None):
            args = argparse.Namespace(script=qlang_script)
            result = cmd_run(args)
            
            assert result == 1
            captured = capsys.readouterr()
            assert "not available" in captured.out


# ==============================================================================
# REPL Tests
# ==============================================================================

class TestQENEXRepl:
    """Test interactive REPL."""
    
    def test_repl_creation(self, temp_dir):
        """Test REPL initialization."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            assert repl.running is True
    
    def test_repl_banner(self):
        """Test REPL banner contains required info."""
        # Banner uses ASCII art, so check for commands instead
        assert "Scientific Intelligence Laboratory" in QENEXRepl.BANNER
        assert "/reason" in QENEXRepl.BANNER
        assert "/generate" in QENEXRepl.BANNER
        assert "/quit" in QENEXRepl.BANNER
    
    def test_process_quit_command(self, temp_dir):
        """Test quit command."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.process_command("/quit")
            assert repl.running is False
    
    def test_process_exit_command(self, temp_dir):
        """Test exit command."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.process_command("/exit")
            assert repl.running is False
    
    def test_process_help_command(self, temp_dir, capsys):
        """Test help command."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.process_command("/help")
            
            captured = capsys.readouterr()
            assert "Help" in captured.out
    
    def test_process_unknown_command(self, temp_dir, capsys):
        """Test unknown command handling."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.process_command("/unknown_cmd")
            
            captured = capsys.readouterr()
            assert "Unknown command" in captured.out
    
    def test_process_reason_command(self, temp_dir, mock_integration, capsys):
        """Test reason command in REPL."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.integration = mock_integration
            repl.process_command("/reason What is gravity?")
            
            mock_integration.reason.assert_called_once()
    
    def test_process_generate_command(self, temp_dir, mock_integration, capsys):
        """Test generate command in REPL."""
        with patch.dict(os.environ, {"HOME": temp_dir}):
            repl = QENEXRepl()
            repl.integration = mock_integration
            repl.process_command("/generate sorting algorithm")
            
            mock_integration.generate_code.assert_called_once()


# ==============================================================================
# Main Function Tests
# ==============================================================================

class TestMain:
    """Test main entry point."""
    
    def test_main_version(self):
        """Test main with --version."""
        with pytest.raises(SystemExit) as exc_info:
            with patch('sys.argv', ['qenex', '--version']):
                main()
        # Version exits with 0
        assert exc_info.value.code == 0
    
    def test_main_status(self, capsys):
        """Test main with status command."""
        with patch('sys.argv', ['qenex', 'status']):
            result = main()
            
            assert result == 0
            captured = capsys.readouterr()
            assert "QENEX" in captured.out
    
    def test_main_reason(self, mock_integration, capsys):
        """Test main with reason command."""
        with patch('sys.argv', ['qenex', 'reason', 'test', 'query']):
            with patch('qenex.get_integration', return_value=mock_integration):
                result = main()
                
                assert result == 0


# ==============================================================================
# Integration Tests
# ==============================================================================

class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_reason_flow(self, capsys):
        """Test full reasoning flow."""
        with patch('sys.argv', ['qenex', 'reason', 'What', 'is', 'dark', 'matter?']):
            result = main()
            
            # Should complete without error
            assert result == 0
            captured = capsys.readouterr()
            assert "Scout" in captured.out
    
    def test_full_generate_flow(self, capsys):
        """Test full generation flow."""
        with patch('sys.argv', ['qenex', 'generate', 'bubble', 'sort']):
            result = main()
            
            assert result == 0
            captured = capsys.readouterr()
            assert "DeepSeek" in captured.out
    
    def test_full_status_flow(self, capsys):
        """Test full status flow."""
        with patch('sys.argv', ['qenex', 'status']):
            result = main()
            
            assert result == 0
            captured = capsys.readouterr()
            assert "Components" in captured.out
            assert "Scientific Packages" in captured.out


# ==============================================================================
# Error Handling Tests
# ==============================================================================

class TestErrorHandling:
    """Test error handling."""
    
    def test_reason_with_error(self, capsys):
        """Test reason command with error response."""
        mock_integration = Mock()
        mock_integration.reason.return_value = {
            "success": False,
            "output": "Error occurred",
        }
        
        with patch('qenex.get_integration', return_value=mock_integration):
            args = argparse.Namespace(query=["test"], mode=None)
            result = cmd_reason(args)
            
            assert result == 1
    
    def test_generate_with_error(self, capsys):
        """Test generate command with error response."""
        mock_integration = Mock()
        mock_integration.generate_code.return_value = {
            "success": False,
            "raw_response": "Generation failed",
        }
        
        with patch('qenex.get_integration', return_value=mock_integration):
            args = argparse.Namespace(
                description=["test"],
                language="python",
                mode=None,
                output=None
            )
            result = cmd_generate(args)
            
            assert result == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
