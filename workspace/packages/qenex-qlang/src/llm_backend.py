"""
QENEX Local LLM Backend Integration
===================================
Connect Scout 17B and DeepSeek to local LLM backends (Ollama, llama.cpp, vLLM).

Supported Backends:
- Ollama: HTTP REST API for easy model management
- llama.cpp: Direct subprocess or server mode
- vLLM: High-throughput serving with PagedAttention
- OpenAI-compatible: Any API following OpenAI format

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                       LLM Router                                  │
    │            (Automatic backend selection & failover)              │
    ├──────────────────────────────────────────────────────────────────┤
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
    │  │  Ollama  │  │ llama.cpp│  │   vLLM   │  │  OpenAI  │        │
    │  │  Backend │  │  Backend │  │  Backend │  │ Compat.  │        │
    │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘        │
    │       │             │             │             │                │
    │       ▼             ▼             ▼             ▼                │
    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
    │  │HTTP:11434│  │ Subprocess│  │HTTP:8000 │  │ HTTP API │        │
    │  │(default) │  │ or Server│  │(default) │  │          │        │
    │  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
    └──────────────────────────────────────────────────────────────────┘

Q-Lang Commands:
    llm status                    # Show backend status
    llm list                      # List available models
    llm select <backend>          # Select backend (ollama/llamacpp/vllm)
    llm model <name>              # Set model name
    llm generate "prompt"         # Generate text
    llm chat "message"            # Chat completion
    llm stream "prompt"           # Streaming generation
    llm config                    # Show configuration
    llm help                      # Show help

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import json
import time
import asyncio
import subprocess
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Iterator, AsyncIterator, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

# Optional httpx for async HTTP
try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# Optional requests for sync HTTP
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class BackendType(Enum):
    """Supported LLM backend types."""
    OLLAMA = "ollama"
    LLAMACPP = "llamacpp"
    VLLM = "vllm"
    OPENAI_COMPAT = "openai"
    MOCK = "mock"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    stop: List[str] = field(default_factory=list)
    seed: Optional[int] = None
    
    def to_ollama(self) -> Dict[str, Any]:
        """Convert to Ollama API format."""
        opts = {
            "num_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.stop:
            opts["stop"] = self.stop
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts
    
    def to_openai(self) -> Dict[str, Any]:
        """Convert to OpenAI-compatible format."""
        opts = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        if self.stop:
            opts["stop"] = self.stop
        return opts
    
    def to_llamacpp(self) -> Dict[str, Any]:
        """Convert to llama.cpp server format."""
        opts = {
            "n_predict": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.stop:
            opts["stop"] = self.stop
        if self.seed is not None:
            opts["seed"] = self.seed
        return opts


@dataclass
class GenerationResult:
    """Result from LLM generation."""
    text: str
    tokens_generated: int = 0
    tokens_prompt: int = 0
    elapsed_ms: float = 0.0
    model: str = ""
    backend: str = ""
    finish_reason: str = "stop"
    raw_response: Optional[Dict[str, Any]] = None


@dataclass
class ModelInfo:
    """Information about an available model."""
    name: str
    size: str = ""
    family: str = ""
    parameter_count: str = ""
    quantization: str = ""
    modified: str = ""
    digest: str = ""
    
    def __str__(self) -> str:
        parts = [self.name]
        if self.size:
            parts.append(f"({self.size})")
        if self.quantization:
            parts.append(f"[{self.quantization}]")
        return " ".join(parts)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Backend name."""
        pass
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Backend type enum."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available."""
        pass
    
    @abstractmethod
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        pass
    
    @abstractmethod
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text from prompt."""
        pass
    
    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Generate text with streaming."""
        pass
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Chat completion (default: convert to prompt)."""
        # Default implementation: convert messages to prompt
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        prompt = "\n\n".join(prompt_parts)
        
        return self.generate(prompt, model, config)


class OllamaBackend(LLMBackend):
    """
    Ollama backend using HTTP REST API.
    
    Ollama provides easy model management and serving.
    Default endpoint: http://localhost:11434
    
    Features:
    - Model pulling and management
    - Chat and completion APIs
    - Streaming support
    - Model customization
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "llama2",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._client: Optional[Any] = None
    
    @property
    def name(self) -> str:
        return "Ollama"
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA
    
    def _get_client(self):
        """Get HTTP client."""
        if self._client is None:
            if HAS_HTTPX:
                self._client = httpx.Client(timeout=self.timeout)
            elif HAS_REQUESTS:
                self._client = requests.Session()
            else:
                raise ImportError("Neither httpx nor requests installed")
        return self._client
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Any:
        """Make HTTP request."""
        client = self._get_client()
        url = f"{self.base_url}{endpoint}"
        
        if HAS_HTTPX:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
        else:
            resp = client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp.json() if resp.content else {}
    
    def _request_stream(self, method: str, endpoint: str, **kwargs) -> Iterator[Dict]:
        """Make streaming HTTP request."""
        url = f"{self.base_url}{endpoint}"
        
        if HAS_HTTPX:
            with httpx.stream(method, url, timeout=self.timeout, **kwargs) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        yield json.loads(line)
        elif HAS_REQUESTS:
            resp = requests.request(method, url, stream=True, timeout=self.timeout, **kwargs)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield json.loads(line.decode())
        else:
            raise ImportError("Neither httpx nor requests installed")
    
    def is_available(self) -> bool:
        """Check if Ollama server is running."""
        try:
            self._request("GET", "/api/tags")
            return True
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available models in Ollama."""
        try:
            data = self._request("GET", "/api/tags")
            models = []
            
            for m in data.get("models", []):
                name = m.get("name", "")
                details = m.get("details", {})
                
                models.append(ModelInfo(
                    name=name,
                    size=self._format_size(m.get("size", 0)),
                    family=details.get("family", ""),
                    parameter_count=details.get("parameter_size", ""),
                    quantization=details.get("quantization_level", ""),
                    modified=m.get("modified_at", ""),
                    digest=m.get("digest", "")[:12],
                ))
            
            return models
        except Exception:
            return []
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size in bytes to human-readable."""
        if size_bytes == 0:
            return ""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}PB"
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using Ollama."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        start = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": config.to_ollama(),
        }
        
        try:
            data = self._request("POST", "/api/generate", json=payload)
            
            elapsed = (time.time() - start) * 1000
            
            return GenerationResult(
                text=data.get("response", ""),
                tokens_generated=data.get("eval_count", 0),
                tokens_prompt=data.get("prompt_eval_count", 0),
                elapsed_ms=elapsed,
                model=model,
                backend=self.name,
                finish_reason="stop" if data.get("done") else "length",
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                model=model,
                backend=self.name,
                finish_reason="error",
            )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation from Ollama."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": config.to_ollama(),
        }
        
        try:
            for chunk in self._request_stream("POST", "/api/generate", json=payload):
                if "response" in chunk:
                    yield chunk["response"]
        except Exception as e:
            yield f"Error: {e}"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Chat completion using Ollama's native chat API."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        start = time.time()
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": config.to_ollama(),
        }
        
        try:
            data = self._request("POST", "/api/chat", json=payload)
            
            elapsed = (time.time() - start) * 1000
            
            message = data.get("message", {})
            
            return GenerationResult(
                text=message.get("content", ""),
                tokens_generated=data.get("eval_count", 0),
                tokens_prompt=data.get("prompt_eval_count", 0),
                elapsed_ms=elapsed,
                model=model,
                backend=self.name,
                finish_reason="stop" if data.get("done") else "length",
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                model=model,
                backend=self.name,
                finish_reason="error",
            )
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry."""
        try:
            # This is a long-running operation
            payload = {"name": model_name}
            
            for chunk in self._request_stream("POST", "/api/pull", json=payload):
                status = chunk.get("status", "")
                if "error" in status.lower():
                    return False
            
            return True
        except Exception:
            return False


class LlamaCppBackend(LLMBackend):
    """
    llama.cpp backend via subprocess or HTTP server.
    
    Supports both:
    1. Direct subprocess execution of llama-cli
    2. HTTP server mode using llama-server
    
    Features:
    - GGUF model format support
    - Quantization support (Q4, Q8, etc.)
    - CPU and GPU (CUDA/Metal) inference
    - Low memory footprint
    """
    
    def __init__(
        self,
        server_url: Optional[str] = None,
        executable: str = "llama-cli",
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        timeout: float = 120.0,
    ):
        self.server_url = server_url.rstrip("/") if server_url else None
        self.executable = executable
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.timeout = timeout
        self._use_server = server_url is not None
    
    @property
    def name(self) -> str:
        return "llama.cpp"
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.LLAMACPP
    
    def is_available(self) -> bool:
        """Check if llama.cpp is available."""
        if self._use_server:
            try:
                if HAS_HTTPX:
                    resp = httpx.get(f"{self.server_url}/health", timeout=5)
                elif HAS_REQUESTS:
                    resp = requests.get(f"{self.server_url}/health", timeout=5)
                else:
                    return False
                return resp.status_code == 200
            except Exception:
                return False
        else:
            # Check if executable exists
            try:
                result = subprocess.run(
                    [self.executable, "--version"],
                    capture_output=True,
                    timeout=5,
                )
                return result.returncode == 0
            except Exception:
                return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available models (from model directory)."""
        models = []
        
        # Check common locations for GGUF files
        search_paths = [
            Path.home() / ".cache" / "llama.cpp" / "models",
            Path.home() / "models",
            Path("/models"),
            Path("./models"),
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                for gguf_file in search_path.glob("**/*.gguf"):
                    models.append(ModelInfo(
                        name=gguf_file.stem,
                        size=self._format_size(gguf_file.stat().st_size),
                        quantization=self._extract_quant(gguf_file.name),
                    ))
        
        return models
    
    def _format_size(self, size_bytes: int) -> str:
        """Format size."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return f"{size_bytes:.1f}{unit}"
            size_bytes /= 1024
        return f"{size_bytes:.1f}PB"
    
    def _extract_quant(self, filename: str) -> str:
        """Extract quantization from filename."""
        quants = ["Q2_K", "Q3_K", "Q4_K", "Q4_0", "Q4_1", "Q5_K", "Q5_0", "Q5_1", 
                  "Q6_K", "Q8_0", "F16", "F32"]
        filename_upper = filename.upper()
        for q in quants:
            if q in filename_upper:
                return q
        return ""
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using llama.cpp."""
        config = config or GenerationConfig()
        model_path = model or self.model_path
        
        if self._use_server:
            return self._generate_server(prompt, config)
        else:
            return self._generate_subprocess(prompt, model_path, config)
    
    def _generate_server(self, prompt: str, config: GenerationConfig) -> GenerationResult:
        """Generate using llama.cpp server."""
        start = time.time()
        
        payload = {
            "prompt": prompt,
            **config.to_llamacpp(),
        }
        
        try:
            if HAS_HTTPX:
                resp = httpx.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            else:
                raise ImportError("Neither httpx nor requests installed")
            
            elapsed = (time.time() - start) * 1000
            
            return GenerationResult(
                text=data.get("content", ""),
                tokens_generated=data.get("tokens_predicted", 0),
                tokens_prompt=data.get("tokens_evaluated", 0),
                elapsed_ms=elapsed,
                model="llama.cpp",
                backend=self.name,
                finish_reason=data.get("stop_type", "stop"),
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                backend=self.name,
                finish_reason="error",
            )
    
    def _generate_subprocess(
        self,
        prompt: str,
        model_path: Optional[str],
        config: GenerationConfig,
    ) -> GenerationResult:
        """Generate using llama-cli subprocess."""
        if not model_path:
            return GenerationResult(
                text="Error: No model path specified",
                backend=self.name,
                finish_reason="error",
            )
        
        start = time.time()
        
        cmd = [
            self.executable,
            "-m", model_path,
            "-c", str(self.n_ctx),
            "-n", str(config.max_tokens),
            "--temp", str(config.temperature),
            "--top-p", str(config.top_p),
            "--top-k", str(config.top_k),
            "--repeat-penalty", str(config.repeat_penalty),
            "-p", prompt,
        ]
        
        if self.n_gpu_layers > 0:
            cmd.extend(["-ngl", str(self.n_gpu_layers)])
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            
            elapsed = (time.time() - start) * 1000
            
            return GenerationResult(
                text=result.stdout,
                elapsed_ms=elapsed,
                model=Path(model_path).stem if model_path else "unknown",
                backend=self.name,
                finish_reason="stop" if result.returncode == 0 else "error",
            )
        except subprocess.TimeoutExpired:
            return GenerationResult(
                text="Error: Generation timed out",
                backend=self.name,
                finish_reason="timeout",
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                backend=self.name,
                finish_reason="error",
            )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream generation (server mode only)."""
        if not self._use_server:
            yield "Streaming not supported in subprocess mode"
            return
        
        config = config or GenerationConfig()
        
        payload = {
            "prompt": prompt,
            "stream": True,
            **config.to_llamacpp(),
        }
        
        try:
            if HAS_HTTPX:
                with httpx.stream("POST", f"{self.server_url}/completion", 
                                  json=payload, timeout=self.timeout) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            data = json.loads(line[6:])
                            if "content" in data:
                                yield data["content"]
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.server_url}/completion",
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                )
                for line in resp.iter_lines():
                    if line and line.startswith(b"data: "):
                        data = json.loads(line[6:].decode())
                        if "content" in data:
                            yield data["content"]
        except Exception as e:
            yield f"Error: {e}"


class VLLMBackend(LLMBackend):
    """
    vLLM backend for high-throughput inference.
    
    vLLM uses PagedAttention for efficient memory management
    and supports tensor parallelism for multi-GPU inference.
    
    Features:
    - High throughput with continuous batching
    - Efficient memory management with PagedAttention
    - OpenAI-compatible API
    - Streaming support
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str = "EMPTY",
        default_model: str = "default",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "vLLM"
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.VLLM
    
    def _headers(self) -> Dict[str, str]:
        """Get request headers."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def is_available(self) -> bool:
        """Check if vLLM server is running."""
        try:
            if HAS_HTTPX:
                resp = httpx.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=5,
                )
            elif HAS_REQUESTS:
                resp = requests.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=5,
                )
            else:
                return False
            return resp.status_code == 200
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available models in vLLM."""
        try:
            if HAS_HTTPX:
                resp = httpx.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=10,
                )
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.get(
                    f"{self.base_url}/v1/models",
                    headers=self._headers(),
                    timeout=10,
                )
                data = resp.json()
            else:
                return []
            
            models = []
            for m in data.get("data", []):
                models.append(ModelInfo(
                    name=m.get("id", ""),
                    family=m.get("owned_by", ""),
                ))
            
            return models
        except Exception:
            return []
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text using vLLM (via completions API)."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        start = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            **config.to_openai(),
        }
        
        try:
            if HAS_HTTPX:
                resp = httpx.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            else:
                raise ImportError("Neither httpx nor requests installed")
            
            elapsed = (time.time() - start) * 1000
            
            choices = data.get("choices", [])
            text = choices[0].get("text", "") if choices else ""
            finish_reason = choices[0].get("finish_reason", "stop") if choices else "error"
            
            usage = data.get("usage", {})
            
            return GenerationResult(
                text=text,
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
                elapsed_ms=elapsed,
                model=model,
                backend=self.name,
                finish_reason=finish_reason,
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                model=model,
                backend=self.name,
                finish_reason="error",
            )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation from vLLM."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **config.to_openai(),
        }
        
        try:
            if HAS_HTTPX:
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                ) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if choices:
                                yield choices[0].get("text", "")
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                )
                for line in resp.iter_lines():
                    if line:
                        line = line.decode()
                        if line.startswith("data: "):
                            if line == "data: [DONE]":
                                break
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if choices:
                                yield choices[0].get("text", "")
        except Exception as e:
            yield f"Error: {e}"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Chat completion using vLLM's chat API."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        start = time.time()
        
        payload = {
            "model": model,
            "messages": messages,
            **config.to_openai(),
        }
        
        try:
            if HAS_HTTPX:
                resp = httpx.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.base_url}/v1/chat/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            else:
                raise ImportError("Neither httpx nor requests installed")
            
            elapsed = (time.time() - start) * 1000
            
            choices = data.get("choices", [])
            message = choices[0].get("message", {}) if choices else {}
            text = message.get("content", "")
            finish_reason = choices[0].get("finish_reason", "stop") if choices else "error"
            
            usage = data.get("usage", {})
            
            return GenerationResult(
                text=text,
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
                elapsed_ms=elapsed,
                model=model,
                backend=self.name,
                finish_reason=finish_reason,
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                model=model,
                backend=self.name,
                finish_reason="error",
            )


class OpenAICompatBackend(LLMBackend):
    """
    Generic OpenAI-compatible API backend.
    
    Works with any server implementing the OpenAI API format:
    - LocalAI
    - LM Studio
    - text-generation-webui
    - FastChat
    - Any other OpenAI-compatible endpoint
    """
    
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        default_model: str = "default",
        timeout: float = 120.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
    
    @property
    def name(self) -> str:
        return "OpenAI-Compatible"
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OPENAI_COMPAT
    
    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
    
    def is_available(self) -> bool:
        """Check if server is available."""
        try:
            if HAS_HTTPX:
                resp = httpx.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=5)
            elif HAS_REQUESTS:
                resp = requests.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=5)
            else:
                return False
            return resp.status_code in [200, 401, 403]  # Server responds
        except Exception:
            return False
    
    def list_models(self) -> List[ModelInfo]:
        """List available models."""
        try:
            if HAS_HTTPX:
                resp = httpx.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=10)
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.get(f"{self.base_url}/v1/models", headers=self._headers(), timeout=10)
                data = resp.json()
            else:
                return []
            
            return [
                ModelInfo(name=m.get("id", ""), family=m.get("owned_by", ""))
                for m in data.get("data", [])
            ]
        except Exception:
            return []
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate text."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        start = time.time()
        
        payload = {
            "model": model,
            "prompt": prompt,
            **config.to_openai(),
        }
        
        try:
            if HAS_HTTPX:
                resp = httpx.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                data = resp.json()
            else:
                raise ImportError("No HTTP client available")
            
            elapsed = (time.time() - start) * 1000
            
            choices = data.get("choices", [])
            text = choices[0].get("text", "") if choices else ""
            usage = data.get("usage", {})
            
            return GenerationResult(
                text=text,
                tokens_generated=usage.get("completion_tokens", 0),
                tokens_prompt=usage.get("prompt_tokens", 0),
                elapsed_ms=elapsed,
                model=model,
                backend=self.name,
                finish_reason=choices[0].get("finish_reason", "stop") if choices else "error",
                raw_response=data,
            )
        except Exception as e:
            return GenerationResult(
                text=f"Error: {e}",
                model=model,
                backend=self.name,
                finish_reason="error",
            )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation."""
        model = model or self.default_model
        config = config or GenerationConfig()
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            **config.to_openai(),
        }
        
        try:
            if HAS_HTTPX:
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                ) as resp:
                    for line in resp.iter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if choices:
                                yield choices[0].get("text", "")
            elif HAS_REQUESTS:
                resp = requests.post(
                    f"{self.base_url}/v1/completions",
                    headers=self._headers(),
                    json=payload,
                    stream=True,
                    timeout=self.timeout,
                )
                for line in resp.iter_lines():
                    if line:
                        line = line.decode()
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data = json.loads(line[6:])
                            choices = data.get("choices", [])
                            if choices:
                                yield choices[0].get("text", "")
        except Exception as e:
            yield f"Error: {e}"


class MockBackend(LLMBackend):
    """
    Mock backend for testing without a real LLM.
    
    Returns simulated responses for development/testing.
    """
    
    def __init__(self, latency_ms: float = 100.0):
        self.latency_ms = latency_ms
        self._models = [
            ModelInfo(name="mock-7b", size="7B", family="mock"),
            ModelInfo(name="mock-13b", size="13B", family="mock"),
            ModelInfo(name="mock-70b", size="70B", family="mock"),
        ]
    
    @property
    def name(self) -> str:
        return "Mock"
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.MOCK
    
    def is_available(self) -> bool:
        return True
    
    def list_models(self) -> List[ModelInfo]:
        return self._models
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Generate mock response."""
        time.sleep(self.latency_ms / 1000)
        
        # Generate contextual mock response
        if "code" in prompt.lower() or "function" in prompt.lower():
            text = """```python
def example_function():
    \"\"\"Example generated code.\"\"\"
    return "Hello from mock LLM!"
```"""
        elif "?" in prompt:
            text = f"Based on your question about '{prompt[:50]}...', here is my analysis:\n\n1. Point one\n2. Point two\n3. Conclusion"
        else:
            text = f"Mock response to: {prompt[:100]}...\n\nThis is a simulated LLM response for testing purposes."
        
        return GenerationResult(
            text=text,
            tokens_generated=len(text) // 4,
            tokens_prompt=len(prompt) // 4,
            elapsed_ms=self.latency_ms,
            model=model or "mock-7b",
            backend=self.name,
            finish_reason="stop",
        )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream mock response."""
        result = self.generate(prompt, model, config)
        
        # Simulate streaming by yielding word by word
        words = result.text.split()
        for i, word in enumerate(words):
            yield word
            if i < len(words) - 1:
                yield " "
            time.sleep(0.02)  # 20ms per word


class LLMRouter:
    """
    Intelligent router for LLM backends.
    
    Features:
    - Automatic backend selection based on availability
    - Failover support
    - Load balancing (future)
    - Model routing based on task type
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.backends: Dict[BackendType, LLMBackend] = {}
        self.active_backend: Optional[LLMBackend] = None
        self.active_model: Optional[str] = None
        
        # Default configuration
        self.default_config = GenerationConfig()
        
        # Initialize with mock backend
        self.register_backend(MockBackend())
        
        if verbose:
            self._print_status()
    
    def _print_status(self):
        """Print router status."""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           QENEX LLM Router                                   ║
    ║           Intelligent Backend Selection & Failover           ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
    
    def register_backend(self, backend: LLMBackend) -> None:
        """Register a backend."""
        self.backends[backend.backend_type] = backend
        
        # Auto-select first available backend
        if self.active_backend is None and backend.is_available():
            self.active_backend = backend
            if self.verbose:
                print(f"✅ Auto-selected backend: {backend.name}")
    
    def select_backend(self, backend_type: BackendType) -> bool:
        """Select a specific backend."""
        if backend_type not in self.backends:
            if self.verbose:
                print(f"❌ Backend not registered: {backend_type.value}")
            return False
        
        backend = self.backends[backend_type]
        
        if not backend.is_available():
            if self.verbose:
                print(f"❌ Backend not available: {backend.name}")
            return False
        
        self.active_backend = backend
        if self.verbose:
            print(f"✅ Selected backend: {backend.name}")
        return True
    
    def set_model(self, model_name: str) -> None:
        """Set the active model."""
        self.active_model = model_name
        if self.verbose:
            print(f"📦 Model set to: {model_name}")
    
    def get_available_backends(self) -> List[LLMBackend]:
        """Get list of available backends."""
        return [b for b in self.backends.values() if b.is_available()]
    
    def list_all_models(self) -> Dict[str, List[ModelInfo]]:
        """List models from all backends."""
        result = {}
        for backend in self.backends.values():
            if backend.is_available():
                models = backend.list_models()
                if models:
                    result[backend.name] = models
        return result
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
        fallback: bool = True,
    ) -> GenerationResult:
        """
        Generate text using active backend with optional fallback.
        
        Args:
            prompt: Input prompt
            model: Model name (uses active_model if None)
            config: Generation config (uses default if None)
            fallback: Try other backends if active fails
        
        Returns:
            GenerationResult
        """
        model = model or self.active_model
        config = config or self.default_config
        
        # Try active backend first
        if self.active_backend and self.active_backend.is_available():
            result = self.active_backend.generate(prompt, model, config)
            if result.finish_reason != "error":
                return result
        
        # Fallback to other backends
        if fallback:
            for backend in self.backends.values():
                if backend != self.active_backend and backend.is_available():
                    if self.verbose:
                        print(f"⚠️  Falling back to: {backend.name}")
                    result = backend.generate(prompt, model, config)
                    if result.finish_reason != "error":
                        return result
        
        return GenerationResult(
            text="Error: No available backend",
            backend="none",
            finish_reason="error",
        )
    
    def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> Iterator[str]:
        """Stream text generation."""
        model = model or self.active_model
        config = config or self.default_config
        
        if self.active_backend and self.active_backend.is_available():
            yield from self.active_backend.generate_stream(prompt, model, config)
        else:
            yield "Error: No available backend"
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        config: Optional[GenerationConfig] = None,
    ) -> GenerationResult:
        """Chat completion."""
        model = model or self.active_model
        config = config or self.default_config
        
        if self.active_backend and self.active_backend.is_available():
            return self.active_backend.chat(messages, model, config)
        
        return GenerationResult(
            text="Error: No available backend",
            backend="none",
            finish_reason="error",
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get router status."""
        return {
            'active_backend': self.active_backend.name if self.active_backend else None,
            'active_model': self.active_model,
            'registered_backends': [b.name for b in self.backends.values()],
            'available_backends': [b.name for b in self.get_available_backends()],
            'default_config': {
                'max_tokens': self.default_config.max_tokens,
                'temperature': self.default_config.temperature,
                'top_p': self.default_config.top_p,
            },
        }


def handle_llm_command(router: LLMRouter, line: str, context: dict) -> None:
    """
    Handle LLM commands from Q-Lang interpreter.
    
    Commands:
        llm status                    - Show backend status
        llm list                      - List available models
        llm select <backend>          - Select backend
        llm model <name>              - Set model name
        llm generate "prompt"         - Generate text
        llm chat "message"            - Chat completion
        llm stream "prompt"           - Streaming generation
        llm config [key=value...]     - Configure generation params
        llm help                      - Show help
    
    Args:
        router: LLMRouter instance
        line: Command line
        context: Q-Lang context dictionary
    """
    parts = line.split(maxsplit=2)
    
    if len(parts) < 2:
        print("❌ Usage: llm <command> [args...]")
        print("   Commands: status, list, select, model, generate, chat, stream, config, help")
        return
    
    cmd = parts[1].lower()
    
    try:
        if cmd == "status":
            status = router.get_status()
            print("\n🤖 LLM Router Status")
            print("=" * 50)
            print(f"  Active Backend: {status['active_backend'] or 'None'}")
            print(f"  Active Model: {status['active_model'] or 'Default'}")
            print(f"  Registered: {', '.join(status['registered_backends'])}")
            print(f"  Available: {', '.join(status['available_backends'])}")
            print(f"\n  Config:")
            for k, v in status['default_config'].items():
                print(f"    {k}: {v}")
            print()
        
        elif cmd == "list":
            models = router.list_all_models()
            
            if not models:
                print("📦 No models available")
                return
            
            print("\n📦 Available Models")
            print("=" * 60)
            for backend_name, model_list in models.items():
                print(f"\n  [{backend_name}]")
                for model in model_list:
                    print(f"    • {model}")
            print()
        
        elif cmd == "select":
            if len(parts) < 3:
                print("❌ Usage: llm select <backend>")
                print("   Backends: ollama, llamacpp, vllm, openai, mock")
                return
            
            backend_name = parts[2].lower()
            backend_map = {
                'ollama': BackendType.OLLAMA,
                'llamacpp': BackendType.LLAMACPP,
                'llama.cpp': BackendType.LLAMACPP,
                'vllm': BackendType.VLLM,
                'openai': BackendType.OPENAI_COMPAT,
                'mock': BackendType.MOCK,
            }
            
            if backend_name not in backend_map:
                print(f"❌ Unknown backend: {backend_name}")
                return
            
            router.select_backend(backend_map[backend_name])
        
        elif cmd == "model":
            if len(parts) < 3:
                print("❌ Usage: llm model <name>")
                return
            
            router.set_model(parts[2])
        
        elif cmd == "generate":
            if len(parts) < 3:
                print("❌ Usage: llm generate \"<prompt>\"")
                return
            
            prompt = parts[2].strip('"\'')
            
            print("\n🔮 Generating...")
            result = router.generate(prompt)
            
            print(f"\n{result.text}")
            print("-" * 50)
            print(f"⏱️  {result.elapsed_ms:.1f}ms | 🎯 Backend: {result.backend} | 📊 Tokens: {result.tokens_generated}")
            
            context['llm_result'] = result
            context['llm_output'] = result.text
        
        elif cmd == "chat":
            if len(parts) < 3:
                print("❌ Usage: llm chat \"<message>\"")
                return
            
            message = parts[2].strip('"\'')
            
            # Build messages from context history or start fresh
            history = context.get('chat_history', [])
            history.append({"role": "user", "content": message})
            
            print("\n💬 Chatting...")
            result = router.chat(history)
            
            # Add assistant response to history
            history.append({"role": "assistant", "content": result.text})
            context['chat_history'] = history
            
            print(f"\n{result.text}")
            print("-" * 50)
            print(f"⏱️  {result.elapsed_ms:.1f}ms | 🎯 Backend: {result.backend}")
            
            context['llm_result'] = result
        
        elif cmd == "stream":
            if len(parts) < 3:
                print("❌ Usage: llm stream \"<prompt>\"")
                return
            
            prompt = parts[2].strip('"\'')
            
            print("\n🌊 Streaming...")
            print("-" * 50)
            
            full_response = []
            for chunk in router.generate_stream(prompt):
                print(chunk, end="", flush=True)
                full_response.append(chunk)
            
            print("\n" + "-" * 50)
            context['llm_output'] = "".join(full_response)
        
        elif cmd == "config":
            if len(parts) < 3:
                # Show current config
                cfg = router.default_config
                print("\n⚙️  Generation Config")
                print("=" * 40)
                print(f"  max_tokens: {cfg.max_tokens}")
                print(f"  temperature: {cfg.temperature}")
                print(f"  top_p: {cfg.top_p}")
                print(f"  top_k: {cfg.top_k}")
                print(f"  repeat_penalty: {cfg.repeat_penalty}")
                if cfg.stop:
                    print(f"  stop: {cfg.stop}")
                print()
                return
            
            # Parse key=value pairs
            for kv in parts[2].split():
                if '=' in kv:
                    key, value = kv.split('=', 1)
                    key = key.lower()
                    
                    if key == 'max_tokens':
                        router.default_config.max_tokens = int(value)
                    elif key == 'temperature':
                        router.default_config.temperature = float(value)
                    elif key == 'top_p':
                        router.default_config.top_p = float(value)
                    elif key == 'top_k':
                        router.default_config.top_k = int(value)
                    elif key == 'repeat_penalty':
                        router.default_config.repeat_penalty = float(value)
                    
                    print(f"✅ Set {key} = {value}")
        
        elif cmd == "help":
            print("""
📖 LLM Backend Commands
========================

  llm status                  Show backend status and config
  llm list                    List available models from all backends
  llm select <backend>        Select active backend
                              (ollama, llamacpp, vllm, openai, mock)
  llm model <name>            Set model name for generation
  llm generate "prompt"       Generate text from prompt
  llm chat "message"          Chat completion (maintains history)
  llm stream "prompt"         Streaming text generation
  llm config                  Show generation config
  llm config key=value...     Set config (max_tokens, temperature, etc.)

Examples:
  llm select ollama
  llm model llama2:7b
  llm generate "Explain quantum entanglement"
  llm config temperature=0.8 max_tokens=1024
            """)
        
        else:
            print(f"❌ Unknown LLM command: {cmd}")
            print("   Use 'llm help' for available commands")
    
    except Exception as e:
        print(f"❌ LLM Error: {e}")


def create_default_router(verbose: bool = True) -> LLMRouter:
    """
    Create a router with default backends configured.
    
    Registers:
    - Ollama (localhost:11434)
    - llama.cpp server (localhost:8080)
    - vLLM (localhost:8000)
    - Mock (always available)
    """
    router = LLMRouter(verbose=verbose)
    
    # Register backends
    router.register_backend(OllamaBackend())
    router.register_backend(LlamaCppBackend(server_url="http://localhost:8080"))
    router.register_backend(VLLMBackend())
    # Mock is already registered by default
    
    # Try to auto-select best available backend
    for backend_type in [BackendType.OLLAMA, BackendType.VLLM, BackendType.LLAMACPP]:
        if router.select_backend(backend_type):
            break
    
    return router


# For availability checking
HAS_LLM_BACKEND = True


# Demo / Test
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX LLM Backend Demo                                 ║
    ║       Local LLM Integration                                  ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create router
    router = create_default_router(verbose=True)
    
    # Show status
    print("\n" + "=" * 60)
    print("Router Status:")
    print("=" * 60)
    status = router.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")
    
    # List models
    print("\n" + "=" * 60)
    print("Available Models:")
    print("=" * 60)
    models = router.list_all_models()
    for backend, model_list in models.items():
        print(f"\n  [{backend}]")
        for model in model_list:
            print(f"    • {model}")
    
    # Generate with mock
    print("\n" + "=" * 60)
    print("Mock Generation Test:")
    print("=" * 60)
    result = router.generate("Write a Python function to calculate factorial")
    print(f"\n{result.text}")
    print(f"\n⏱️  {result.elapsed_ms:.1f}ms | Backend: {result.backend}")
    
    # Streaming test
    print("\n" + "=" * 60)
    print("Streaming Test:")
    print("=" * 60)
    for chunk in router.generate_stream("Explain quantum computing briefly"):
        print(chunk, end="", flush=True)
    print("\n")
    
    print("✅ Demo complete!")
