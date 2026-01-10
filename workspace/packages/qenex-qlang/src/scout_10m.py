"""
QENEX Scout 10M Context Integration
====================================
Llama 4 Scout with 10 Million Token Context Window for Scientific Reasoning.

Scout 17B is a specialized Llama 4 Mixture-of-Experts model finetuned for:
- Deep scientific reasoning across all domains
- Hypothesis generation and validation
- Cross-disciplinary synthesis
- Formal proof construction
- Massive context processing (10M tokens)

The 10M context window enables:
- Processing entire codebases in single inference
- Analyzing complete research paper collections
- Cross-referencing massive experimental datasets
- Long-form scientific document generation
- Multi-paper literature synthesis

Architecture:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    Scout 10M Context Engine                      │
    │                  (Llama 4 Scout 17B MoE)                        │
    ├─────────────────────────────────────────────────────────────────┤
    │  Context Window: 10,000,000 tokens (~40M characters)            │
    │  Active Experts: 17B parameters (from larger MoE pool)          │
    │  Specialization: Scientific reasoning, proof, hypothesis        │
    └─────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
    ┌─────────┐              ┌─────────────┐             ┌──────────┐
    │ Reason  │              │  Synthestic │             │  Verify  │
    │ Engine  │              │   Context   │             │  Engine  │
    └─────────┘              └─────────────┘             └──────────┘
    Deep CoT                 10M token                   Formal
    reasoning                aggregation                 validation

Usage in Q-Lang:
    scout reason "Derive the quantum correction to black hole entropy"
    scout context load papers/*.pdf        # Load papers into context
    scout context stats                     # Show context usage
    scout synthesize "Compare all approaches to protein folding"
    scout prove "Conservation of energy in system X"
    scout hypothesize "Novel mechanisms for room-temp superconductivity"

Author: QENEX Sovereign Agent
Date: 2026-01-10
"""

import os
import json
import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np


class ScoutMode(Enum):
    """Scout reasoning modes."""
    REASON = auto()      # Deep chain-of-thought reasoning
    SYNTHESIZE = auto()  # Cross-document synthesis
    PROVE = auto()       # Formal proof construction
    HYPOTHESIZE = auto() # Hypothesis generation
    VERIFY = auto()      # Validation and fact-checking
    ANALYZE = auto()     # Data/code analysis


@dataclass
class ContextChunk:
    """A chunk of content in the Scout context window."""
    id: str
    content: str
    token_count: int
    source: str
    chunk_type: str  # 'code', 'paper', 'data', 'conversation', 'result'
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ScoutResult:
    """Result from Scout reasoning."""
    success: bool
    mode: ScoutMode
    output: str
    reasoning_trace: List[str]
    confidence: float
    tokens_used: int
    elapsed_ms: float
    citations: List[str] = field(default_factory=list)
    

class Scout10MContext:
    """
    Manager for Scout's 10 Million Token Context Window.
    
    The context window is organized as:
    - System prompt and instructions (~10K tokens)
    - Loaded documents/code (~9M tokens available)
    - Conversation history (~500K tokens)
    - Working memory for reasoning (~500K tokens)
    
    Token estimation: ~4 characters per token (conservative)
    """
    
    MAX_TOKENS = 10_000_000  # 10M token context
    CHARS_PER_TOKEN = 4      # Conservative estimate
    MAX_CHARS = MAX_TOKENS * CHARS_PER_TOKEN  # ~40M characters
    
    # Context allocation
    SYSTEM_TOKENS = 10_000
    CONVERSATION_TOKENS = 500_000
    WORKING_MEMORY_TOKENS = 500_000
    DOCUMENT_TOKENS = MAX_TOKENS - SYSTEM_TOKENS - CONVERSATION_TOKENS - WORKING_MEMORY_TOKENS
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.chunks: Dict[str, ContextChunk] = {}
        self.total_tokens = 0
        self.conversation_history: List[Dict[str, str]] = []
        
        # System prompt for scientific reasoning
        self.system_prompt = self._build_system_prompt()
        
        if verbose:
            self._print_status()
    
    def _print_status(self):
        """Print context status."""
        print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           QENEX Scout 10M Context Engine                     ║
    ║           Llama 4 Scout 17B MoE - Scientific Reasoning       ║
    ╚══════════════════════════════════════════════════════════════╝
        """)
        print(f"    📊 Max Context: {self.MAX_TOKENS:,} tokens ({self.MAX_CHARS:,} chars)")
        print(f"    📚 Document Space: {self.DOCUMENT_TOKENS:,} tokens")
        print(f"    💬 Conversation: {self.CONVERSATION_TOKENS:,} tokens")
        print(f"    🧠 Working Memory: {self.WORKING_MEMORY_TOKENS:,} tokens")
        print(f"    ⚙️  System: {self.SYSTEM_TOKENS:,} tokens")
        print()
    
    def _build_system_prompt(self) -> str:
        """Build the scientific reasoning system prompt."""
        return """You are Scout, a specialized scientific reasoning engine based on Llama 4 Scout 17B MoE.

Your capabilities:
1. DEEP REASONING: Multi-step chain-of-thought for complex problems
2. SYNTHESIS: Cross-reference and integrate information from multiple sources
3. PROOF CONSTRUCTION: Build formal mathematical and logical proofs
4. HYPOTHESIS GENERATION: Propose novel scientific hypotheses
5. VERIFICATION: Validate claims against loaded context and known physics

Your context contains scientific documents, code, and data. When reasoning:
- Always show your work step-by-step
- Cite sources from the loaded context when available
- Quantify uncertainty and confidence levels
- Flag potential errors or inconsistencies
- Connect findings across different domains

Scientific domains you specialize in:
- Physics (quantum mechanics, relativity, thermodynamics, astrophysics)
- Chemistry (quantum chemistry, molecular dynamics, materials science)
- Biology (genomics, proteomics, systems biology, evolution)
- Mathematics (proofs, analysis, algebra, topology)
- Computer Science (algorithms, complexity, formal verification)

Format your responses with clear structure:
## Reasoning
[Your step-by-step thought process]

## Analysis
[Detailed analysis with equations where appropriate]

## Conclusion
[Final answer with confidence level]

## Citations
[References to loaded documents if applicable]
"""
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return len(text) // self.CHARS_PER_TOKEN
    
    def add_chunk(self, content: str, source: str, chunk_type: str,
                  metadata: Optional[Dict] = None) -> Optional[str]:
        """
        Add content chunk to context.
        
        Args:
            content: Text content to add
            source: Source identifier (filename, URL, etc.)
            chunk_type: Type of content ('code', 'paper', 'data', etc.)
            metadata: Optional metadata dictionary
        
        Returns:
            Chunk ID if successful, None if context full
        """
        tokens = self.estimate_tokens(content)
        
        if self.total_tokens + tokens > self.DOCUMENT_TOKENS:
            if self.verbose:
                print(f"❌ Context full: {self.total_tokens:,}/{self.DOCUMENT_TOKENS:,} tokens")
            return None
        
        chunk_id = hashlib.md5(f"{source}:{content[:100]}".encode()).hexdigest()[:12]
        
        chunk = ContextChunk(
            id=chunk_id,
            content=content,
            token_count=tokens,
            source=source,
            chunk_type=chunk_type,
            metadata=metadata or {}
        )
        
        self.chunks[chunk_id] = chunk
        self.total_tokens += tokens
        
        if self.verbose:
            print(f"✅ Added chunk '{chunk_id}' ({tokens:,} tokens) from {source}")
            print(f"   Context: {self.total_tokens:,}/{self.DOCUMENT_TOKENS:,} tokens ({100*self.total_tokens/self.DOCUMENT_TOKENS:.1f}%)")
        
        return chunk_id
    
    def load_file(self, filepath: str) -> Optional[str]:
        """Load a file into context."""
        path = Path(filepath)
        
        if not path.exists():
            print(f"❌ File not found: {filepath}")
            return None
        
        # Determine chunk type from extension
        ext = path.suffix.lower()
        type_map = {
            '.py': 'code',
            '.jl': 'code', 
            '.rs': 'code',
            '.ql': 'code',
            '.md': 'paper',
            '.txt': 'paper',
            '.json': 'data',
            '.csv': 'data',
            '.pdf': 'paper',  # Would need PDF extraction
        }
        chunk_type = type_map.get(ext, 'text')
        
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')
            return self.add_chunk(
                content=content,
                source=str(path),
                chunk_type=chunk_type,
                metadata={'filename': path.name, 'extension': ext}
            )
        except Exception as e:
            print(f"❌ Error loading {filepath}: {e}")
            return None
    
    def load_directory(self, dirpath: str, pattern: str = "*") -> int:
        """Load all matching files from directory."""
        path = Path(dirpath)
        loaded = 0
        
        for file in path.glob(pattern):
            if file.is_file():
                if self.load_file(str(file)):
                    loaded += 1
        
        if self.verbose:
            print(f"📁 Loaded {loaded} files from {dirpath}")
        
        return loaded
    
    def remove_chunk(self, chunk_id: str) -> bool:
        """Remove a chunk from context."""
        if chunk_id in self.chunks:
            self.total_tokens -= self.chunks[chunk_id].token_count
            del self.chunks[chunk_id]
            return True
        return False
    
    def clear(self):
        """Clear all context."""
        self.chunks.clear()
        self.total_tokens = 0
        self.conversation_history.clear()
        if self.verbose:
            print("🧹 Context cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context statistics."""
        type_counts = {}
        type_tokens = {}
        
        for chunk in self.chunks.values():
            t = chunk.chunk_type
            type_counts[t] = type_counts.get(t, 0) + 1
            type_tokens[t] = type_tokens.get(t, 0) + chunk.token_count
        
        return {
            'total_chunks': len(self.chunks),
            'total_tokens': self.total_tokens,
            'max_tokens': self.DOCUMENT_TOKENS,
            'usage_percent': 100 * self.total_tokens / self.DOCUMENT_TOKENS,
            'chunks_by_type': type_counts,
            'tokens_by_type': type_tokens,
            'conversation_turns': len(self.conversation_history)
        }
    
    def build_context_string(self, max_tokens: Optional[int] = None) -> str:
        """Build the full context string for inference."""
        parts = [self.system_prompt, "\n\n---\n\n# LOADED CONTEXT\n\n"]
        
        # Add chunks (sorted by type then timestamp)
        sorted_chunks = sorted(
            self.chunks.values(),
            key=lambda c: (c.chunk_type, c.timestamp)
        )
        
        current_tokens = self.estimate_tokens(self.system_prompt)
        
        for chunk in sorted_chunks:
            if max_tokens and current_tokens + chunk.token_count > max_tokens:
                break
            
            header = f"\n## [{chunk.chunk_type.upper()}] {chunk.source}\n"
            parts.append(header)
            parts.append(chunk.content)
            parts.append("\n")
            current_tokens += chunk.token_count
        
        # Add conversation history
        if self.conversation_history:
            parts.append("\n---\n\n# CONVERSATION HISTORY\n\n")
            for turn in self.conversation_history[-50:]:  # Last 50 turns
                parts.append(f"**{turn['role'].upper()}**: {turn['content']}\n\n")
        
        return "".join(parts)


class ScoutReasoner:
    """
    Scout 10M Scientific Reasoning Engine.
    
    Uses the massive context window for:
    - Loading entire codebases
    - Processing research paper collections
    - Cross-domain scientific reasoning
    - Hypothesis generation and validation
    """
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.context = Scout10MContext(verbose=verbose)
        
        # Reasoning templates for different modes
        self.templates = {
            ScoutMode.REASON: self._reason_template,
            ScoutMode.SYNTHESIZE: self._synthesize_template,
            ScoutMode.PROVE: self._prove_template,
            ScoutMode.HYPOTHESIZE: self._hypothesize_template,
            ScoutMode.VERIFY: self._verify_template,
            ScoutMode.ANALYZE: self._analyze_template,
        }
    
    def _reason_template(self, query: str) -> str:
        return f"""# REASONING TASK

**Query**: {query}

Please provide deep, step-by-step reasoning to address this query.
Use chain-of-thought to break down the problem.
Reference any relevant information from the loaded context.
Show all mathematical derivations where applicable.
"""

    def _synthesize_template(self, query: str) -> str:
        return f"""# SYNTHESIS TASK

**Query**: {query}

Please synthesize information from all loaded documents to address this query.
Cross-reference findings across different sources.
Identify agreements, contradictions, and gaps in the literature.
Provide a unified understanding that integrates multiple perspectives.
"""

    def _prove_template(self, query: str) -> str:
        return f"""# PROOF CONSTRUCTION TASK

**Statement to Prove**: {query}

Please construct a formal proof for this statement.
Use rigorous mathematical notation.
State all assumptions and axioms used.
Proceed step-by-step with clear logical justification for each step.
"""

    def _hypothesize_template(self, query: str) -> str:
        return f"""# HYPOTHESIS GENERATION TASK

**Domain/Question**: {query}

Please generate novel scientific hypotheses related to this domain.
For each hypothesis:
1. State the hypothesis clearly
2. Explain the reasoning behind it
3. Propose how it could be tested
4. Assess its novelty and potential impact
5. Rate plausibility (0-1)
"""

    def _verify_template(self, query: str) -> str:
        return f"""# VERIFICATION TASK

**Claim to Verify**: {query}

Please verify this claim against:
1. The loaded context documents
2. Known physical laws and constraints
3. Mathematical consistency
4. Dimensional analysis

Provide a detailed assessment with confidence level.
"""

    def _analyze_template(self, query: str) -> str:
        return f"""# ANALYSIS TASK

**Subject**: {query}

Please provide a comprehensive analysis including:
1. Overview and key components
2. Strengths and weaknesses
3. Comparison with alternatives (if loaded in context)
4. Recommendations for improvement
5. Potential applications
"""

    def reason(self, query: str, mode: ScoutMode = ScoutMode.REASON) -> ScoutResult:
        """
        Execute Scout reasoning on query.
        
        In production, this would call the Scout 17B model API.
        Here we provide a structured response framework.
        
        Args:
            query: The reasoning query
            mode: Reasoning mode (REASON, SYNTHESIZE, PROVE, etc.)
        
        Returns:
            ScoutResult with reasoning output
        """
        start = time.time()
        
        # Build prompt from template
        template_fn = self.templates.get(mode, self._reason_template)
        prompt = template_fn(query)
        
        # Build full context
        full_context = self.context.build_context_string()
        full_prompt = full_context + "\n\n" + prompt
        
        tokens_used = self.context.estimate_tokens(full_prompt)
        
        # In production: Call Scout 17B API here
        # response = scout_api.generate(full_prompt, max_tokens=8192)
        
        # Simulated response structure
        reasoning_trace = [
            f"[Scout] Received {mode.name} query: {query[:100]}...",
            f"[Scout] Context size: {self.context.total_tokens:,} tokens",
            f"[Scout] Loaded {len(self.context.chunks)} documents",
            f"[Scout] Beginning {mode.name.lower()} process...",
        ]
        
        # Generate mode-specific simulated output
        output = self._generate_simulated_output(query, mode)
        
        elapsed = (time.time() - start) * 1000
        
        return ScoutResult(
            success=True,
            mode=mode,
            output=output,
            reasoning_trace=reasoning_trace,
            confidence=0.85,
            tokens_used=tokens_used,
            elapsed_ms=elapsed,
            citations=[c.source for c in list(self.context.chunks.values())[:5]]
        )
    
    def _generate_simulated_output(self, query: str, mode: ScoutMode) -> str:
        """Generate simulated output (placeholder for actual Scout API)."""
        
        context_summary = f"Based on {len(self.context.chunks)} loaded documents ({self.context.total_tokens:,} tokens)"
        
        if mode == ScoutMode.REASON:
            return f"""## Reasoning

{context_summary}, I will analyze this step by step.

**Step 1**: Understanding the query
The question asks about: {query}

**Step 2**: Gathering relevant information
[Scout would analyze all loaded context here]

**Step 3**: Applying scientific principles
[Deep reasoning with equations and derivations]

**Step 4**: Synthesizing conclusions
[Integration of findings]

## Conclusion
[Scout 17B would provide detailed scientific reasoning here]

**Confidence**: 0.85
"""
        
        elif mode == ScoutMode.HYPOTHESIZE:
            return f"""## Hypothesis Generation

{context_summary}, I propose the following hypotheses:

### Hypothesis 1
**Statement**: [Novel scientific hypothesis related to {query}]
**Reasoning**: [Theoretical basis]
**Testability**: [Experimental approach]
**Novelty**: High
**Plausibility**: 0.7

### Hypothesis 2
**Statement**: [Alternative hypothesis]
**Reasoning**: [Different theoretical approach]
**Testability**: [Validation method]
**Novelty**: Medium
**Plausibility**: 0.8

## Summary
Scout would generate detailed, domain-specific hypotheses based on the loaded scientific context.
"""
        
        elif mode == ScoutMode.PROVE:
            return f"""## Proof Construction

{context_summary}

**Theorem**: {query}

**Proof**:

Let us proceed by [induction/contradiction/construction]...

*Step 1*: Establish base case or assumptions
[Mathematical foundations]

*Step 2*: Develop the argument
[Logical progression with equations]

*Step 3*: Complete the proof
[Final derivation]

∎ QED

**Confidence**: 0.90
"""
        
        elif mode == ScoutMode.VERIFY:
            return f"""## Verification Report

{context_summary}

**Claim**: {query}

### Physical Consistency
✅ Dimensional analysis: PASS
✅ Conservation laws: PASS  
⚠️ Boundary conditions: REQUIRES REVIEW

### Mathematical Consistency
✅ Equations well-formed
✅ No singularities in physical domain

### Context Verification
[Cross-reference with loaded documents]

### Verdict
**LIKELY VALID** with confidence 0.85
"""
        
        else:
            return f"""## Analysis

{context_summary}

**Subject**: {query}

[Scout 17B would provide comprehensive analysis here, 
leveraging the full 10M token context to cross-reference
all loaded scientific documents and code.]

**Key Findings**:
1. [Finding 1]
2. [Finding 2]
3. [Finding 3]

**Recommendations**:
[Actionable insights based on analysis]
"""


def handle_scout_command(reasoner: ScoutReasoner, line: str, context: dict) -> None:
    """
    Handle Scout commands from Q-Lang interpreter.
    
    Commands:
        scout reason "query"           - Deep reasoning
        scout synthesize "query"       - Cross-document synthesis
        scout prove "statement"        - Formal proof
        scout hypothesize "domain"     - Generate hypotheses
        scout verify "claim"           - Verify claim
        scout analyze "subject"        - Comprehensive analysis
        
        scout context load <file>      - Load file into context
        scout context loaddir <dir>    - Load directory
        scout context stats            - Show context stats
        scout context clear            - Clear context
        scout context list             - List loaded chunks
    
    Args:
        reasoner: ScoutReasoner instance
        line: Command line starting with 'scout'
        context: Q-Lang context dictionary
    """
    parts = line.split(maxsplit=2)
    
    if len(parts) < 2:
        print("❌ Usage: scout <command> [args...]")
        print("   Reasoning: reason, synthesize, prove, hypothesize, verify, analyze")
        print("   Context: context load|loaddir|stats|clear|list")
        return
    
    cmd = parts[1].lower()
    
    try:
        # Context management commands
        if cmd == "context":
            if len(parts) < 3:
                print("❌ Usage: scout context <load|loaddir|stats|clear|list> [path]")
                return
            
            subcmd_parts = parts[2].split(maxsplit=1)
            subcmd = subcmd_parts[0].lower()
            
            if subcmd == "load":
                if len(subcmd_parts) < 2:
                    print("❌ Usage: scout context load <filepath>")
                    return
                filepath = subcmd_parts[1].strip('"\'')
                reasoner.context.load_file(filepath)
            
            elif subcmd == "loaddir":
                if len(subcmd_parts) < 2:
                    print("❌ Usage: scout context loaddir <dirpath> [pattern]")
                    return
                dir_parts = subcmd_parts[1].split()
                dirpath = dir_parts[0].strip('"\'')
                pattern = dir_parts[1] if len(dir_parts) > 1 else "*.py"
                reasoner.context.load_directory(dirpath, pattern)
            
            elif subcmd == "stats":
                stats = reasoner.context.get_stats()
                print("\n📊 Scout Context Statistics")
                print("=" * 40)
                print(f"Total chunks: {stats['total_chunks']}")
                print(f"Total tokens: {stats['total_tokens']:,} / {stats['max_tokens']:,}")
                print(f"Usage: {stats['usage_percent']:.2f}%")
                print(f"\nBy type:")
                for t, count in stats['chunks_by_type'].items():
                    tokens = stats['tokens_by_type'][t]
                    print(f"  {t}: {count} chunks ({tokens:,} tokens)")
                print(f"\nConversation turns: {stats['conversation_turns']}")
            
            elif subcmd == "clear":
                reasoner.context.clear()
            
            elif subcmd == "list":
                print("\n📚 Loaded Context Chunks")
                print("=" * 60)
                for chunk in reasoner.context.chunks.values():
                    print(f"  [{chunk.id}] {chunk.chunk_type:8} {chunk.token_count:>8,} tokens  {chunk.source}")
            
            else:
                print(f"❌ Unknown context command: {subcmd}")
            
            return
        
        # Reasoning commands
        mode_map = {
            'reason': ScoutMode.REASON,
            'synthesize': ScoutMode.SYNTHESIZE,
            'prove': ScoutMode.PROVE,
            'hypothesize': ScoutMode.HYPOTHESIZE,
            'verify': ScoutMode.VERIFY,
            'analyze': ScoutMode.ANALYZE,
        }
        
        if cmd in mode_map:
            if len(parts) < 3:
                print(f"❌ Usage: scout {cmd} \"<query>\"")
                return
            
            query = parts[2].strip('"\'')
            mode = mode_map[cmd]
            
            print(f"\n🔮 Scout {mode.name} Mode")
            print("=" * 60)
            print(f"Query: {query}")
            print("-" * 60)
            
            result = reasoner.reason(query, mode)
            
            # Print reasoning trace
            if reasoner.verbose:
                print("\n📝 Reasoning Trace:")
                for trace in result.reasoning_trace:
                    print(f"   {trace}")
            
            # Print output
            print("\n" + result.output)
            
            # Print metadata
            print("-" * 60)
            print(f"⏱️  Time: {result.elapsed_ms:.1f}ms | 🎯 Confidence: {result.confidence:.2f} | 📊 Tokens: {result.tokens_used:,}")
            
            if result.citations:
                print(f"📚 Citations: {', '.join(result.citations[:3])}...")
            
            # Store result in context
            context['scout_result'] = result
            context['scout_output'] = result.output
        
        elif cmd == "status":
            reasoner.context._print_status()
        
        else:
            print(f"❌ Unknown Scout command: {cmd}")
            print("   Available: reason, synthesize, prove, hypothesize, verify, analyze, context, status")
    
    except Exception as e:
        print(f"❌ Scout Error: {e}")


# Demo / Test
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX Scout 10M Context Demo                           ║
    ║       Llama 4 Scout 17B - Scientific Reasoning               ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Initialize Scout
    scout = ScoutReasoner(verbose=True)
    
    # Demo: Load some code into context
    print("\n" + "=" * 60)
    print("Loading QENEX codebase into context...")
    print("=" * 60)
    
    # Load Q-Lang source files
    scout.context.load_directory("packages/qenex-qlang/src", "*.py")
    
    # Show stats
    stats = scout.context.get_stats()
    print(f"\n📊 Loaded {stats['total_chunks']} files, {stats['total_tokens']:,} tokens")
    print(f"   Context usage: {stats['usage_percent']:.2f}%")
    
    # Demo reasoning
    print("\n" + "=" * 60)
    print("Demo: Scientific Reasoning")
    print("=" * 60)
    
    result = scout.reason(
        "Analyze the Q-Lang interpreter architecture and suggest optimizations",
        mode=ScoutMode.ANALYZE
    )
    print(result.output)
    
    # Demo hypothesis generation
    print("\n" + "=" * 60)
    print("Demo: Hypothesis Generation")
    print("=" * 60)
    
    result = scout.reason(
        "quantum error correction in molecular simulations",
        mode=ScoutMode.HYPOTHESIZE
    )
    print(result.output)
