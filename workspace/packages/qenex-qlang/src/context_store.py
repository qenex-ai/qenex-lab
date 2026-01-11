"""
QENEX Context Persistence Layer
===============================
Save and load Scout 10M context to/from disk for session persistence.

Features:
- Serialize context chunks to JSON/MessagePack
- Compression (gzip/lz4) for large contexts
- Context versioning and metadata
- Index for fast chunk lookup
- Incremental saves (only changed chunks)
- Context merging from multiple files

Architecture:
    ┌──────────────────────────────────────────────────────────────────┐
    │                    Context Store                                  │
    ├──────────────────────────────────────────────────────────────────┤
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
    │  │  Serializer │  │ Compressor  │  │   Index     │              │
    │  │  JSON/MsgPk │  │  gzip/lz4   │  │  Fast Lookup│              │
    │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
    │         └────────────────┼────────────────┘                      │
    │                          ▼                                       │
    │                   ┌─────────────┐                                │
    │                   │  .qctx File │                                │
    │                   │  (context)  │                                │
    │                   └─────────────┘                                │
    └──────────────────────────────────────────────────────────────────┘

File Format (.qctx):
    {
        "version": "1.0",
        "format": "json",  # or "msgpack"
        "compression": "gzip",  # or "lz4", "none"
        "created": "2026-01-11T00:00:00Z",
        "modified": "2026-01-11T00:00:00Z",
        "metadata": {...},
        "index": {
            "chunk_id": {"source": "...", "type": "...", "tokens": 123},
            ...
        },
        "chunks": [...]  # Full chunk data
    }

Q-Lang Commands:
    context save <path>              # Save full context
    context load <path>              # Load context from file
    context export <path> --format   # Export with specific format
    context merge <path>             # Merge context from file
    context list                     # List saved contexts
    context info <path>              # Show context file info
    context diff <path1> <path2>     # Compare two contexts

Author: QENEX Sovereign Agent
Date: 2026-01-11
"""

import os
import sys
import json
import gzip
import time
import shutil
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union, Iterator
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum, auto

# Optional msgpack support
try:
    import msgpack
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False

# Optional lz4 support
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False

# Import Scout types
try:
    from .scout_10m import Scout10MContext, ContextChunk
except ImportError:
    try:
        from scout_10m import Scout10MContext, ContextChunk
    except ImportError:
        # Define minimal types for standalone use
        @dataclass
        class ContextChunk:
            """A chunk of content in the context."""
            id: str
            content: str
            token_count: int
            source: str
            chunk_type: str
            metadata: Dict[str, Any] = field(default_factory=dict)
            timestamp: float = field(default_factory=time.time)
        
        class Scout10MContext:
            """Minimal context stub."""
            def __init__(self):
                self.chunks: Dict[str, ContextChunk] = {}
                self.total_tokens = 0
                self.conversation_history: List[Dict[str, str]] = []


class SerializationFormat(Enum):
    """Supported serialization formats."""
    JSON = "json"
    MSGPACK = "msgpack"


class CompressionType(Enum):
    """Supported compression types."""
    NONE = "none"
    GZIP = "gzip"
    LZ4 = "lz4"


@dataclass
class ContextMetadata:
    """Metadata for a saved context."""
    name: str
    description: str = ""
    version: str = "1.0"
    created: str = ""
    modified: str = ""
    total_chunks: int = 0
    total_tokens: int = 0
    format: str = "json"
    compression: str = "gzip"
    checksum: str = ""
    tags: List[str] = field(default_factory=list)
    custom: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        now = datetime.now(timezone.utc).isoformat()
        if not self.created:
            self.created = now
        if not self.modified:
            self.modified = now


@dataclass
class ChunkIndex:
    """Index entry for fast chunk lookup."""
    chunk_id: str
    source: str
    chunk_type: str
    token_count: int
    offset: int = 0  # Byte offset in file (for streaming)
    size: int = 0    # Byte size in file


@dataclass
class ContextFile:
    """Represents a context file on disk."""
    path: Path
    metadata: ContextMetadata
    index: Dict[str, ChunkIndex]
    chunks: List[ContextChunk] = field(default_factory=list)
    
    def get_chunk_by_source(self, source: str) -> Optional[ContextChunk]:
        """Find chunk by source name."""
        for chunk in self.chunks:
            if chunk.source == source:
                return chunk
        return None
    
    def get_chunks_by_type(self, chunk_type: str) -> List[ContextChunk]:
        """Get all chunks of a given type."""
        return [c for c in self.chunks if c.chunk_type == chunk_type]


class ContextSerializer:
    """Handles serialization/deserialization of context data."""
    
    @staticmethod
    def serialize(data: Any, format: SerializationFormat) -> bytes:
        """Serialize data to bytes."""
        if format == SerializationFormat.JSON:
            return json.dumps(data, indent=2, default=str).encode('utf-8')
        elif format == SerializationFormat.MSGPACK:
            if not HAS_MSGPACK:
                raise ImportError("msgpack not installed. Install with: pip install msgpack")
            return msgpack.packb(data, use_bin_type=True, default=str)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    @staticmethod
    def deserialize(data: bytes, format: SerializationFormat) -> Any:
        """Deserialize bytes to data."""
        if format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif format == SerializationFormat.MSGPACK:
            if not HAS_MSGPACK:
                raise ImportError("msgpack not installed. Install with: pip install msgpack")
            return msgpack.unpackb(data, raw=False)
        else:
            raise ValueError(f"Unknown format: {format}")


class ContextCompressor:
    """Handles compression/decompression of context data."""
    
    @staticmethod
    def compress(data: bytes, compression: CompressionType) -> bytes:
        """Compress data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.compress(data, compresslevel=6)
        elif compression == CompressionType.LZ4:
            if not HAS_LZ4:
                raise ImportError("lz4 not installed. Install with: pip install lz4")
            return lz4.frame.compress(data)
        else:
            raise ValueError(f"Unknown compression: {compression}")
    
    @staticmethod
    def decompress(data: bytes, compression: CompressionType) -> bytes:
        """Decompress data."""
        if compression == CompressionType.NONE:
            return data
        elif compression == CompressionType.GZIP:
            return gzip.decompress(data)
        elif compression == CompressionType.LZ4:
            if not HAS_LZ4:
                raise ImportError("lz4 not installed. Install with: pip install lz4")
            return lz4.frame.decompress(data)
        else:
            raise ValueError(f"Unknown compression: {compression}")


class ContextStore:
    """
    Persistent storage for Scout 10M context.
    
    Manages saving, loading, and merging context data with support
    for multiple formats and compression schemes.
    """
    
    DEFAULT_EXTENSION = ".qctx"
    HEADER_SIZE = 1024  # Reserved header bytes for metadata
    
    def __init__(self, base_dir: Optional[str] = None, verbose: bool = True):
        """
        Initialize context store.
        
        Args:
            base_dir: Base directory for context files (default: ~/.qenex/contexts)
            verbose: Print status messages
        """
        self.verbose = verbose
        
        if base_dir:
            self.base_dir = Path(base_dir)
        else:
            self.base_dir = Path.home() / ".qenex" / "contexts"
        
        # Ensure base directory exists
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache of loaded contexts
        self._cache: Dict[str, ContextFile] = {}
        
        if verbose:
            print(f"📁 Context Store initialized at: {self.base_dir}")
    
    def _compute_checksum(self, data: bytes) -> str:
        """Compute SHA256 checksum of data."""
        return hashlib.sha256(data).hexdigest()[:16]
    
    def _chunk_to_dict(self, chunk: ContextChunk) -> Dict[str, Any]:
        """Convert ContextChunk to dictionary."""
        return {
            'id': chunk.id,
            'content': chunk.content,
            'token_count': chunk.token_count,
            'source': chunk.source,
            'chunk_type': chunk.chunk_type,
            'metadata': chunk.metadata,
            'timestamp': chunk.timestamp,
        }
    
    def _dict_to_chunk(self, data: Dict[str, Any]) -> ContextChunk:
        """Convert dictionary to ContextChunk."""
        return ContextChunk(
            id=data['id'],
            content=data['content'],
            token_count=data['token_count'],
            source=data['source'],
            chunk_type=data['chunk_type'],
            metadata=data.get('metadata', {}),
            timestamp=data.get('timestamp', time.time()),
        )
    
    def save(
        self,
        context: Scout10MContext,
        path: Optional[str] = None,
        name: str = "context",
        description: str = "",
        format: SerializationFormat = SerializationFormat.JSON,
        compression: CompressionType = CompressionType.GZIP,
        tags: Optional[List[str]] = None,
        custom_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        Save context to disk.
        
        Args:
            context: Scout10MContext to save
            path: Output path (default: base_dir/<name>.qctx)
            name: Context name (used if path not specified)
            description: Human-readable description
            format: Serialization format (JSON or MSGPACK)
            compression: Compression type (NONE, GZIP, LZ4)
            tags: Optional tags for organization
            custom_metadata: Custom metadata to include
        
        Returns:
            Path to saved file
        """
        # Determine output path
        if path:
            output_path = Path(path)
            if not output_path.suffix:
                output_path = output_path.with_suffix(self.DEFAULT_EXTENSION)
        else:
            output_path = self.base_dir / f"{name}{self.DEFAULT_EXTENSION}"
        
        # Ensure parent directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build chunk list and index
        chunks_data = []
        index_data = {}
        
        for chunk_id, chunk in context.chunks.items():
            chunk_dict = self._chunk_to_dict(chunk)
            chunks_data.append(chunk_dict)
            index_data[chunk_id] = {
                'source': chunk.source,
                'chunk_type': chunk.chunk_type,
                'token_count': chunk.token_count,
            }
        
        # Build metadata
        metadata = ContextMetadata(
            name=name,
            description=description,
            total_chunks=len(chunks_data),
            total_tokens=context.total_tokens,
            format=format.value,
            compression=compression.value,
            tags=tags or [],
            custom=custom_metadata or {},
        )
        
        # Build complete data structure
        file_data = {
            'version': '1.0',
            'format': format.value,
            'compression': compression.value,
            'metadata': asdict(metadata),
            'index': index_data,
            'conversation_history': context.conversation_history,
            'chunks': chunks_data,
        }
        
        # Serialize
        serialized = ContextSerializer.serialize(file_data, format)
        
        # Compress
        compressed = ContextCompressor.compress(serialized, compression)
        
        # Compute checksum
        metadata.checksum = self._compute_checksum(compressed)
        
        # Update file data with checksum
        file_data['metadata']['checksum'] = metadata.checksum
        serialized = ContextSerializer.serialize(file_data, format)
        compressed = ContextCompressor.compress(serialized, compression)
        
        # Write to disk
        with open(output_path, 'wb') as f:
            f.write(compressed)
        
        if self.verbose:
            size_kb = len(compressed) / 1024
            ratio = len(serialized) / len(compressed) if compressed else 1
            print(f"✅ Context saved: {output_path}")
            print(f"   📊 {metadata.total_chunks} chunks, {metadata.total_tokens:,} tokens")
            print(f"   💾 Size: {size_kb:.1f} KB (compression ratio: {ratio:.1f}x)")
        
        return output_path
    
    def load(
        self,
        path: str,
        into_context: Optional[Scout10MContext] = None,
        merge: bool = False,
    ) -> Scout10MContext:
        """
        Load context from disk.
        
        Args:
            path: Path to context file
            into_context: Existing context to load into (creates new if None)
            merge: If True, merge with existing chunks; if False, replace
        
        Returns:
            Scout10MContext with loaded data
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        
        # Read file
        with open(file_path, 'rb') as f:
            compressed = f.read()
        
        # Detect format from first bytes
        # GZIP starts with 0x1f 0x8b
        # LZ4 starts with 0x04 0x22 0x4d 0x18
        # JSON starts with '{'
        # MessagePack starts with various codes
        
        if compressed[:2] == b'\x1f\x8b':
            compression = CompressionType.GZIP
        elif compressed[:4] == b'\x04\x22\x4d\x18':
            compression = CompressionType.LZ4
        else:
            compression = CompressionType.NONE
        
        # Decompress
        try:
            decompressed = ContextCompressor.decompress(compressed, compression)
        except Exception as e:
            # Try without compression
            decompressed = compressed
            compression = CompressionType.NONE
        
        # Detect serialization format
        try:
            if decompressed[0:1] == b'{':
                format = SerializationFormat.JSON
            else:
                format = SerializationFormat.MSGPACK
        except IndexError:
            raise ValueError("Empty context file")
        
        # Deserialize
        file_data = ContextSerializer.deserialize(decompressed, format)
        
        # Create or use context
        if into_context is None:
            context = Scout10MContext(verbose=False)
        else:
            context = into_context
            if not merge:
                context.chunks.clear()
                context.total_tokens = 0
                context.conversation_history.clear()
        
        # Load chunks
        for chunk_dict in file_data.get('chunks', []):
            chunk = self._dict_to_chunk(chunk_dict)
            
            if merge and chunk.id in context.chunks:
                # Skip existing chunks in merge mode
                continue
            
            context.chunks[chunk.id] = chunk
            context.total_tokens += chunk.token_count
        
        # Load conversation history
        history = file_data.get('conversation_history', [])
        if merge:
            context.conversation_history.extend(history)
        else:
            context.conversation_history = history
        
        if self.verbose:
            metadata = file_data.get('metadata', {})
            print(f"✅ Context loaded: {path}")
            print(f"   📊 {len(context.chunks)} chunks, {context.total_tokens:,} tokens")
            if metadata.get('name'):
                print(f"   📛 Name: {metadata['name']}")
            if metadata.get('description'):
                print(f"   📝 {metadata['description']}")
        
        return context
    
    def get_info(self, path: str) -> Dict[str, Any]:
        """
        Get information about a context file without fully loading it.
        
        Args:
            path: Path to context file
        
        Returns:
            Dictionary with file information
        """
        file_path = Path(path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Context file not found: {path}")
        
        stat = file_path.stat()
        
        # Read and decompress header only
        with open(file_path, 'rb') as f:
            compressed = f.read()
        
        # Detect compression
        if compressed[:2] == b'\x1f\x8b':
            compression = CompressionType.GZIP
        elif compressed[:4] == b'\x04\x22\x4d\x18':
            compression = CompressionType.LZ4
        else:
            compression = CompressionType.NONE
        
        # Decompress
        decompressed = ContextCompressor.decompress(compressed, compression)
        
        # Parse
        if decompressed[0:1] == b'{':
            format = SerializationFormat.JSON
        else:
            format = SerializationFormat.MSGPACK
        
        file_data = ContextSerializer.deserialize(decompressed, format)
        
        metadata = file_data.get('metadata', {})
        index = file_data.get('index', {})
        
        # Compute type distribution
        type_dist = {}
        for entry in index.values():
            t = entry.get('chunk_type', 'unknown')
            type_dist[t] = type_dist.get(t, 0) + 1
        
        return {
            'path': str(file_path),
            'size_bytes': stat.st_size,
            'size_kb': stat.st_size / 1024,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'format': metadata.get('format', format.value),
            'compression': metadata.get('compression', compression.value),
            'version': file_data.get('version', '1.0'),
            'name': metadata.get('name', ''),
            'description': metadata.get('description', ''),
            'total_chunks': metadata.get('total_chunks', len(index)),
            'total_tokens': metadata.get('total_tokens', 0),
            'checksum': metadata.get('checksum', ''),
            'tags': metadata.get('tags', []),
            'created': metadata.get('created', ''),
            'type_distribution': type_dist,
            'conversation_turns': len(file_data.get('conversation_history', [])),
        }
    
    def list_contexts(self, pattern: str = "*.qctx") -> List[Dict[str, Any]]:
        """
        List all context files in base directory.
        
        Args:
            pattern: Glob pattern for files
        
        Returns:
            List of context info dictionaries
        """
        contexts = []
        
        for file_path in self.base_dir.glob(pattern):
            if file_path.is_file():
                try:
                    info = self.get_info(str(file_path))
                    contexts.append(info)
                except Exception as e:
                    contexts.append({
                        'path': str(file_path),
                        'error': str(e),
                    })
        
        # Sort by modification time (newest first)
        contexts.sort(key=lambda x: x.get('modified', ''), reverse=True)
        
        return contexts
    
    def delete(self, path: str) -> bool:
        """
        Delete a context file.
        
        Args:
            path: Path to context file
        
        Returns:
            True if deleted, False if not found
        """
        file_path = Path(path)
        
        if not file_path.exists():
            return False
        
        file_path.unlink()
        
        if self.verbose:
            print(f"🗑️  Deleted: {path}")
        
        return True
    
    def merge(
        self,
        paths: List[str],
        output_path: Optional[str] = None,
        name: str = "merged",
        strategy: str = "latest",
    ) -> Path:
        """
        Merge multiple context files into one.
        
        Args:
            paths: List of context file paths
            output_path: Output path for merged context
            name: Name for merged context
            strategy: Merge strategy for conflicts:
                - 'latest': Keep chunk with latest timestamp
                - 'first': Keep first occurrence
                - 'largest': Keep largest chunk
        
        Returns:
            Path to merged file
        """
        if len(paths) < 2:
            raise ValueError("Need at least 2 files to merge")
        
        merged_context = Scout10MContext(verbose=False)
        chunk_sources: Dict[str, Tuple[ContextChunk, str]] = {}  # id -> (chunk, source_file)
        
        for path in paths:
            context = self.load(path, merge=False)
            
            for chunk_id, chunk in context.chunks.items():
                if chunk_id in chunk_sources:
                    existing_chunk, _ = chunk_sources[chunk_id]
                    
                    # Apply merge strategy
                    if strategy == 'latest':
                        if chunk.timestamp > existing_chunk.timestamp:
                            chunk_sources[chunk_id] = (chunk, path)
                    elif strategy == 'first':
                        pass  # Keep existing
                    elif strategy == 'largest':
                        if chunk.token_count > existing_chunk.token_count:
                            chunk_sources[chunk_id] = (chunk, path)
                else:
                    chunk_sources[chunk_id] = (chunk, path)
            
            # Merge conversation history
            merged_context.conversation_history.extend(context.conversation_history)
        
        # Add all chunks to merged context
        for chunk, _ in chunk_sources.values():
            merged_context.chunks[chunk.id] = chunk
            merged_context.total_tokens += chunk.token_count
        
        # Save merged context
        return self.save(
            merged_context,
            path=output_path,
            name=name,
            description=f"Merged from {len(paths)} contexts",
            tags=['merged'],
        )
    
    def diff(self, path1: str, path2: str) -> Dict[str, Any]:
        """
        Compare two context files.
        
        Args:
            path1: First context file
            path2: Second context file
        
        Returns:
            Dictionary with differences
        """
        ctx1 = self.load(path1, merge=False)
        ctx2 = self.load(path2, merge=False)
        
        ids1 = set(ctx1.chunks.keys())
        ids2 = set(ctx2.chunks.keys())
        
        only_in_1 = ids1 - ids2
        only_in_2 = ids2 - ids1
        in_both = ids1 & ids2
        
        modified = []
        for chunk_id in in_both:
            c1 = ctx1.chunks[chunk_id]
            c2 = ctx2.chunks[chunk_id]
            if c1.content != c2.content or c1.token_count != c2.token_count:
                modified.append(chunk_id)
        
        return {
            'file1': path1,
            'file2': path2,
            'chunks_in_file1': len(ids1),
            'chunks_in_file2': len(ids2),
            'only_in_file1': list(only_in_1),
            'only_in_file2': list(only_in_2),
            'in_both': len(in_both),
            'modified': modified,
            'tokens_in_file1': ctx1.total_tokens,
            'tokens_in_file2': ctx2.total_tokens,
            'token_diff': ctx2.total_tokens - ctx1.total_tokens,
        }
    
    def export_chunks(
        self,
        path: str,
        output_dir: str,
        chunk_type: Optional[str] = None,
    ) -> int:
        """
        Export individual chunks to files.
        
        Args:
            path: Context file path
            output_dir: Output directory for chunks
            chunk_type: Filter by chunk type (optional)
        
        Returns:
            Number of chunks exported
        """
        context = self.load(path, merge=False)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        exported = 0
        
        for chunk in context.chunks.values():
            if chunk_type and chunk.chunk_type != chunk_type:
                continue
            
            # Create filename from source
            safe_name = "".join(c if c.isalnum() or c in '._-' else '_' for c in chunk.source)
            filename = f"{chunk.id}_{safe_name}"[:100]
            
            # Determine extension from chunk type
            ext_map = {'code': '.py', 'paper': '.txt', 'data': '.json'}
            ext = ext_map.get(chunk.chunk_type, '.txt')
            
            chunk_path = output_path / f"{filename}{ext}"
            chunk_path.write_text(chunk.content)
            exported += 1
        
        if self.verbose:
            print(f"📤 Exported {exported} chunks to {output_dir}")
        
        return exported


def handle_context_command(store: ContextStore, context: Scout10MContext, line: str, qlang_context: dict) -> None:
    """
    Handle context persistence commands from Q-Lang interpreter.
    
    Commands:
        context save <path>              - Save full context
        context save <path> --name X     - Save with name
        context load <path>              - Load context from file
        context load <path> --merge      - Merge with existing
        context list                     - List saved contexts
        context info <path>              - Show context file info
        context delete <path>            - Delete context file
        context merge <p1> <p2> [out]    - Merge contexts
        context diff <path1> <path2>     - Compare contexts
        context export <path> <dir>      - Export chunks to files
    
    Args:
        store: ContextStore instance
        context: Current Scout10MContext
        line: Command line
        qlang_context: Q-Lang context dictionary
    """
    parts = line.split()
    
    if len(parts) < 2:
        print("❌ Usage: context <save|load|list|info|delete|merge|diff|export> [args...]")
        return
    
    cmd = parts[1].lower()
    
    try:
        if cmd == "save":
            if len(parts) < 3:
                print("❌ Usage: context save <path> [--name NAME] [--desc DESC]")
                return
            
            path = parts[2]
            name = "context"
            description = ""
            
            # Parse options
            i = 3
            while i < len(parts):
                if parts[i] == "--name" and i + 1 < len(parts):
                    name = parts[i + 1]
                    i += 2
                elif parts[i] == "--desc" and i + 1 < len(parts):
                    description = parts[i + 1]
                    i += 2
                else:
                    i += 1
            
            saved_path = store.save(context, path, name=name, description=description)
            qlang_context['last_saved_context'] = str(saved_path)
        
        elif cmd == "load":
            if len(parts) < 3:
                print("❌ Usage: context load <path> [--merge]")
                return
            
            path = parts[2]
            merge = "--merge" in parts
            
            store.load(path, into_context=context, merge=merge)
            qlang_context['last_loaded_context'] = path
        
        elif cmd == "list":
            contexts = store.list_contexts()
            
            if not contexts:
                print("📂 No saved contexts found")
                return
            
            print("\n📂 Saved Contexts")
            print("=" * 70)
            for ctx in contexts:
                if 'error' in ctx:
                    print(f"  ❌ {ctx['path']}: {ctx['error']}")
                else:
                    name = ctx.get('name', 'unnamed')
                    chunks = ctx.get('total_chunks', 0)
                    tokens = ctx.get('total_tokens', 0)
                    size = ctx.get('size_kb', 0)
                    print(f"  📄 {name:20} | {chunks:4} chunks | {tokens:>10,} tokens | {size:>8.1f} KB")
                    print(f"     {ctx['path']}")
            print()
        
        elif cmd == "info":
            if len(parts) < 3:
                print("❌ Usage: context info <path>")
                return
            
            path = parts[2]
            info = store.get_info(path)
            
            print(f"\n📄 Context File Info: {path}")
            print("=" * 60)
            print(f"  Name: {info.get('name', 'N/A')}")
            print(f"  Description: {info.get('description', 'N/A')}")
            print(f"  Version: {info.get('version', 'N/A')}")
            print(f"  Created: {info.get('created', 'N/A')}")
            print(f"  Modified: {info.get('modified', 'N/A')}")
            print(f"  Size: {info.get('size_kb', 0):.1f} KB")
            print(f"  Format: {info.get('format', 'N/A')}")
            print(f"  Compression: {info.get('compression', 'N/A')}")
            print(f"  Checksum: {info.get('checksum', 'N/A')}")
            print(f"  Chunks: {info.get('total_chunks', 0)}")
            print(f"  Tokens: {info.get('total_tokens', 0):,}")
            print(f"  Conversation turns: {info.get('conversation_turns', 0)}")
            
            if info.get('tags'):
                print(f"  Tags: {', '.join(info['tags'])}")
            
            if info.get('type_distribution'):
                print(f"  Types: {info['type_distribution']}")
            print()
        
        elif cmd == "delete":
            if len(parts) < 3:
                print("❌ Usage: context delete <path>")
                return
            
            path = parts[2]
            if store.delete(path):
                print(f"✅ Deleted: {path}")
            else:
                print(f"❌ File not found: {path}")
        
        elif cmd == "merge":
            if len(parts) < 4:
                print("❌ Usage: context merge <path1> <path2> [output_path]")
                return
            
            path1 = parts[2]
            path2 = parts[3]
            output = parts[4] if len(parts) > 4 else None
            
            merged = store.merge([path1, path2], output_path=output)
            print(f"✅ Merged to: {merged}")
        
        elif cmd == "diff":
            if len(parts) < 4:
                print("❌ Usage: context diff <path1> <path2>")
                return
            
            path1 = parts[2]
            path2 = parts[3]
            
            diff = store.diff(path1, path2)
            
            print(f"\n📊 Context Diff")
            print("=" * 60)
            print(f"  File 1: {diff['file1']}")
            print(f"  File 2: {diff['file2']}")
            print(f"  Chunks in file 1: {diff['chunks_in_file1']}")
            print(f"  Chunks in file 2: {diff['chunks_in_file2']}")
            print(f"  Only in file 1: {len(diff['only_in_file1'])}")
            print(f"  Only in file 2: {len(diff['only_in_file2'])}")
            print(f"  In both: {diff['in_both']}")
            print(f"  Modified: {len(diff['modified'])}")
            print(f"  Token diff: {diff['token_diff']:+,}")
            print()
        
        elif cmd == "export":
            if len(parts) < 4:
                print("❌ Usage: context export <context_path> <output_dir> [--type TYPE]")
                return
            
            ctx_path = parts[2]
            out_dir = parts[3]
            chunk_type = None
            
            if "--type" in parts:
                idx = parts.index("--type")
                if idx + 1 < len(parts):
                    chunk_type = parts[idx + 1]
            
            store.export_chunks(ctx_path, out_dir, chunk_type=chunk_type)
        
        elif cmd == "help":
            print("""
📖 Context Persistence Commands
================================

  context save <path>              Save context to file
    --name NAME                    Set context name
    --desc DESCRIPTION             Set description
    
  context load <path>              Load context from file
    --merge                        Merge with existing (don't replace)
    
  context list                     List all saved contexts
  
  context info <path>              Show context file details
  
  context delete <path>            Delete context file
  
  context merge <p1> <p2> [out]    Merge two contexts
  
  context diff <p1> <p2>           Compare two contexts
  
  context export <path> <dir>      Export chunks to individual files
    --type TYPE                    Filter by chunk type
            """)
        
        else:
            print(f"❌ Unknown context command: {cmd}")
            print("   Available: save, load, list, info, delete, merge, diff, export, help")
    
    except Exception as e:
        print(f"❌ Context error: {e}")


# For availability checking
HAS_CONTEXT_STORE = True


# Demo / Test
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║       QENEX Context Persistence Demo                         ║
    ║       Save/Load Scout 10M Context                            ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create store
    store = ContextStore(base_dir="/tmp/qenex_contexts", verbose=True)
    
    # Create mock context with some data
    context = Scout10MContext(verbose=False)
    
    # Add some test chunks
    context.add_chunk(
        content="def hello_world():\n    print('Hello, World!')\n",
        source="hello.py",
        chunk_type="code",
        metadata={"language": "python"}
    )
    
    context.add_chunk(
        content="# Scientific Paper\n\nThis is a test paper about quantum mechanics.\n\n## Abstract\nWe present results...",
        source="paper.md",
        chunk_type="paper",
        metadata={"title": "Test Paper"}
    )
    
    context.add_chunk(
        content='{"experiment": "test", "results": [1, 2, 3, 4, 5]}',
        source="data.json",
        chunk_type="data",
    )
    
    context.conversation_history = [
        {"role": "user", "content": "What is quantum entanglement?"},
        {"role": "assistant", "content": "Quantum entanglement is a phenomenon..."},
    ]
    
    print(f"\n📊 Mock context created:")
    print(f"   Chunks: {len(context.chunks)}")
    print(f"   Tokens: {context.total_tokens:,}")
    
    # Save context
    print("\n" + "=" * 60)
    print("Saving context...")
    saved_path = store.save(
        context,
        name="demo_context",
        description="Demo context for testing persistence",
        tags=["demo", "test"],
    )
    
    # Get info
    print("\n" + "=" * 60)
    print("Getting file info...")
    info = store.get_info(str(saved_path))
    print(f"   Size: {info['size_kb']:.2f} KB")
    print(f"   Format: {info['format']}")
    print(f"   Compression: {info['compression']}")
    
    # List contexts
    print("\n" + "=" * 60)
    print("Listing contexts...")
    contexts = store.list_contexts()
    for ctx in contexts:
        print(f"   {ctx.get('name', 'unnamed')}: {ctx.get('total_chunks', 0)} chunks")
    
    # Load into new context
    print("\n" + "=" * 60)
    print("Loading into new context...")
    new_context = Scout10MContext(verbose=False)
    store.load(str(saved_path), into_context=new_context)
    print(f"   Loaded chunks: {len(new_context.chunks)}")
    print(f"   Loaded tokens: {new_context.total_tokens:,}")
    print(f"   Conversation turns: {len(new_context.conversation_history)}")
    
    # Verify data integrity
    print("\n" + "=" * 60)
    print("Verifying data integrity...")
    for chunk_id in context.chunks:
        original = context.chunks[chunk_id]
        loaded = new_context.chunks.get(chunk_id)
        if loaded and original.content == loaded.content:
            print(f"   ✅ {chunk_id}: OK")
        else:
            print(f"   ❌ {chunk_id}: MISMATCH")
    
    print("\n✅ Demo complete!")
