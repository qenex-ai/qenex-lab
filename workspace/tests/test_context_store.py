"""
Tests for QENEX Context Persistence Layer.

Tests cover:
- ContextSerializer: JSON/MessagePack serialization
- ContextCompressor: gzip/lz4 compression
- ContextStore: save, load, merge, diff operations
- ContextMetadata: metadata handling
- ChunkIndex: index operations
- Q-Lang integration: command handling
"""

import os
import sys
import json
import gzip
import time
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock

import pytest

# Add source path
sys.path.insert(0, str(Path(__file__).parent.parent / "packages" / "qenex-qlang" / "src"))

from context_store import (
    ContextStore,
    ContextSerializer,
    ContextCompressor,
    ContextMetadata,
    ChunkIndex,
    ContextFile,
    ContextChunk,
    SerializationFormat,
    CompressionType,
    handle_context_command,
    HAS_MSGPACK,
    HAS_LZ4,
    HAS_CONTEXT_STORE,
)

# Try to import Scout10MContext
try:
    from scout_10m import Scout10MContext
except ImportError:
    Scout10MContext = None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    dirpath = tempfile.mkdtemp(prefix="qenex_test_")
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)


@pytest.fixture
def context_store(temp_dir):
    """Create ContextStore instance."""
    return ContextStore(base_dir=temp_dir, verbose=False)


@pytest.fixture
def mock_context():
    """Create mock Scout10MContext with test data."""
    context = Mock()
    context.chunks = {}
    context.total_tokens = 0
    context.conversation_history = []
    
    # Add some test chunks
    chunk1 = ContextChunk(
        id="chunk1",
        content="def test():\n    pass\n",
        token_count=10,
        source="test.py",
        chunk_type="code",
        metadata={"language": "python"},
        timestamp=time.time(),
    )
    chunk2 = ContextChunk(
        id="chunk2",
        content="# Test Document\n\nThis is content.",
        token_count=15,
        source="doc.md",
        chunk_type="paper",
        metadata={},
        timestamp=time.time(),
    )
    
    context.chunks["chunk1"] = chunk1
    context.chunks["chunk2"] = chunk2
    context.total_tokens = 25
    context.conversation_history = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi!"},
    ]
    
    return context


@pytest.fixture
def sample_data():
    """Sample data for serialization tests."""
    return {
        "version": "1.0",
        "metadata": {"name": "test"},
        "chunks": [
            {"id": "1", "content": "hello"},
            {"id": "2", "content": "world"},
        ],
    }


# ============================================================================
# Test ContextSerializer
# ============================================================================

class TestContextSerializer:
    """Tests for ContextSerializer."""
    
    def test_serialize_json(self, sample_data):
        """Test JSON serialization."""
        result = ContextSerializer.serialize(sample_data, SerializationFormat.JSON)
        assert isinstance(result, bytes)
        assert b'"version": "1.0"' in result
    
    def test_deserialize_json(self, sample_data):
        """Test JSON deserialization."""
        serialized = ContextSerializer.serialize(sample_data, SerializationFormat.JSON)
        result = ContextSerializer.deserialize(serialized, SerializationFormat.JSON)
        assert result == sample_data
    
    def test_json_roundtrip(self, sample_data):
        """Test JSON serialization roundtrip."""
        serialized = ContextSerializer.serialize(sample_data, SerializationFormat.JSON)
        result = ContextSerializer.deserialize(serialized, SerializationFormat.JSON)
        assert result["version"] == sample_data["version"]
        assert result["chunks"] == sample_data["chunks"]
    
    @pytest.mark.skipif(not HAS_MSGPACK, reason="msgpack not installed")
    def test_serialize_msgpack(self, sample_data):
        """Test MessagePack serialization."""
        result = ContextSerializer.serialize(sample_data, SerializationFormat.MSGPACK)
        assert isinstance(result, bytes)
        assert len(result) > 0
    
    @pytest.mark.skipif(not HAS_MSGPACK, reason="msgpack not installed")
    def test_deserialize_msgpack(self, sample_data):
        """Test MessagePack deserialization."""
        serialized = ContextSerializer.serialize(sample_data, SerializationFormat.MSGPACK)
        result = ContextSerializer.deserialize(serialized, SerializationFormat.MSGPACK)
        assert result == sample_data
    
    def test_serialize_invalid_format(self, sample_data):
        """Test serialization with invalid format."""
        with pytest.raises(ValueError, match="Unknown format"):
            ContextSerializer.serialize(sample_data, "invalid")
    
    def test_deserialize_invalid_format(self):
        """Test deserialization with invalid format."""
        with pytest.raises(ValueError, match="Unknown format"):
            ContextSerializer.deserialize(b"data", "invalid")


# ============================================================================
# Test ContextCompressor
# ============================================================================

class TestContextCompressor:
    """Tests for ContextCompressor."""
    
    def test_compress_none(self):
        """Test no compression."""
        data = b"test data"
        result = ContextCompressor.compress(data, CompressionType.NONE)
        assert result == data
    
    def test_decompress_none(self):
        """Test no decompression."""
        data = b"test data"
        result = ContextCompressor.decompress(data, CompressionType.NONE)
        assert result == data
    
    def test_compress_gzip(self):
        """Test gzip compression."""
        data = b"test data " * 100
        result = ContextCompressor.compress(data, CompressionType.GZIP)
        assert isinstance(result, bytes)
        assert len(result) < len(data)  # Should be compressed
    
    def test_decompress_gzip(self):
        """Test gzip decompression."""
        data = b"test data " * 100
        compressed = ContextCompressor.compress(data, CompressionType.GZIP)
        result = ContextCompressor.decompress(compressed, CompressionType.GZIP)
        assert result == data
    
    def test_gzip_roundtrip(self):
        """Test gzip compression roundtrip."""
        data = b"Hello World!" * 1000
        compressed = ContextCompressor.compress(data, CompressionType.GZIP)
        decompressed = ContextCompressor.decompress(compressed, CompressionType.GZIP)
        assert decompressed == data
    
    @pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")
    def test_compress_lz4(self):
        """Test lz4 compression."""
        data = b"test data " * 100
        result = ContextCompressor.compress(data, CompressionType.LZ4)
        assert isinstance(result, bytes)
    
    @pytest.mark.skipif(not HAS_LZ4, reason="lz4 not installed")
    def test_decompress_lz4(self):
        """Test lz4 decompression."""
        data = b"test data " * 100
        compressed = ContextCompressor.compress(data, CompressionType.LZ4)
        result = ContextCompressor.decompress(compressed, CompressionType.LZ4)
        assert result == data
    
    def test_compress_invalid_type(self):
        """Test compression with invalid type."""
        with pytest.raises(ValueError, match="Unknown compression"):
            ContextCompressor.compress(b"data", "invalid")


# ============================================================================
# Test ContextMetadata
# ============================================================================

class TestContextMetadata:
    """Tests for ContextMetadata dataclass."""
    
    def test_metadata_creation(self):
        """Test basic metadata creation."""
        meta = ContextMetadata(name="test")
        assert meta.name == "test"
        assert meta.description == ""
        assert meta.version == "1.0"
    
    def test_metadata_with_all_fields(self):
        """Test metadata with all fields."""
        meta = ContextMetadata(
            name="full_test",
            description="A complete test",
            version="2.0",
            total_chunks=10,
            total_tokens=1000,
            format="json",
            compression="gzip",
            tags=["test", "demo"],
            custom={"key": "value"},
        )
        assert meta.name == "full_test"
        assert meta.description == "A complete test"
        assert meta.total_chunks == 10
        assert meta.tags == ["test", "demo"]
    
    def test_metadata_auto_timestamps(self):
        """Test automatic timestamp creation."""
        before = datetime.now(timezone.utc).isoformat()
        meta = ContextMetadata(name="test")
        after = datetime.now(timezone.utc).isoformat()
        
        assert meta.created
        assert meta.modified
        assert meta.created >= before[:19]  # Compare first 19 chars
    
    def test_metadata_default_values(self):
        """Test default values."""
        meta = ContextMetadata(name="test")
        assert meta.tags == []
        assert meta.custom == {}
        assert meta.checksum == ""


# ============================================================================
# Test ChunkIndex
# ============================================================================

class TestChunkIndex:
    """Tests for ChunkIndex dataclass."""
    
    def test_index_creation(self):
        """Test basic index creation."""
        index = ChunkIndex(
            chunk_id="id1",
            source="file.py",
            chunk_type="code",
            token_count=100,
        )
        assert index.chunk_id == "id1"
        assert index.source == "file.py"
        assert index.token_count == 100
    
    def test_index_with_offsets(self):
        """Test index with offset/size."""
        index = ChunkIndex(
            chunk_id="id1",
            source="file.py",
            chunk_type="code",
            token_count=100,
            offset=1024,
            size=512,
        )
        assert index.offset == 1024
        assert index.size == 512


# ============================================================================
# Test ContextFile
# ============================================================================

class TestContextFile:
    """Tests for ContextFile dataclass."""
    
    def test_context_file_creation(self):
        """Test basic file creation."""
        meta = ContextMetadata(name="test")
        ctx_file = ContextFile(
            path=Path("/tmp/test.qctx"),
            metadata=meta,
            index={},
        )
        assert ctx_file.path == Path("/tmp/test.qctx")
        assert ctx_file.metadata.name == "test"
    
    def test_get_chunk_by_source(self):
        """Test finding chunk by source."""
        chunk = ContextChunk(
            id="c1",
            content="test",
            token_count=5,
            source="myfile.py",
            chunk_type="code",
        )
        ctx_file = ContextFile(
            path=Path("/tmp/test.qctx"),
            metadata=ContextMetadata(name="test"),
            index={},
            chunks=[chunk],
        )
        
        result = ctx_file.get_chunk_by_source("myfile.py")
        assert result == chunk
        
        result = ctx_file.get_chunk_by_source("nonexistent.py")
        assert result is None
    
    def test_get_chunks_by_type(self):
        """Test filtering chunks by type."""
        chunks = [
            ContextChunk("c1", "code1", 10, "f1.py", "code"),
            ContextChunk("c2", "doc", 15, "f2.md", "paper"),
            ContextChunk("c3", "code2", 20, "f3.py", "code"),
        ]
        ctx_file = ContextFile(
            path=Path("/tmp/test.qctx"),
            metadata=ContextMetadata(name="test"),
            index={},
            chunks=chunks,
        )
        
        code_chunks = ctx_file.get_chunks_by_type("code")
        assert len(code_chunks) == 2
        
        paper_chunks = ctx_file.get_chunks_by_type("paper")
        assert len(paper_chunks) == 1


# ============================================================================
# Test ContextStore
# ============================================================================

class TestContextStore:
    """Tests for ContextStore main class."""
    
    def test_store_creation(self, temp_dir):
        """Test store initialization."""
        store = ContextStore(base_dir=temp_dir, verbose=False)
        assert store.base_dir == Path(temp_dir)
        assert store.base_dir.exists()
    
    def test_store_default_dir(self):
        """Test store with default directory."""
        store = ContextStore(verbose=False)
        assert store.base_dir == Path.home() / ".qenex" / "contexts"
    
    def test_compute_checksum(self, context_store):
        """Test checksum computation."""
        data = b"test data"
        checksum = context_store._compute_checksum(data)
        assert isinstance(checksum, str)
        assert len(checksum) == 16
    
    def test_chunk_to_dict(self, context_store):
        """Test chunk to dict conversion."""
        chunk = ContextChunk(
            id="c1",
            content="test content",
            token_count=5,
            source="test.py",
            chunk_type="code",
            metadata={"key": "value"},
            timestamp=12345.0,
        )
        result = context_store._chunk_to_dict(chunk)
        
        assert result["id"] == "c1"
        assert result["content"] == "test content"
        assert result["token_count"] == 5
        assert result["source"] == "test.py"
        assert result["metadata"] == {"key": "value"}
    
    def test_dict_to_chunk(self, context_store):
        """Test dict to chunk conversion."""
        data = {
            "id": "c1",
            "content": "test content",
            "token_count": 5,
            "source": "test.py",
            "chunk_type": "code",
            "metadata": {"key": "value"},
            "timestamp": 12345.0,
        }
        result = context_store._dict_to_chunk(data)
        
        assert isinstance(result, ContextChunk)
        assert result.id == "c1"
        assert result.content == "test content"


class TestContextStoreSave:
    """Tests for ContextStore.save()."""
    
    def test_save_basic(self, context_store, mock_context):
        """Test basic save operation."""
        path = context_store.save(mock_context, name="test_save")
        
        assert path.exists()
        assert path.suffix == ".qctx"
    
    def test_save_with_path(self, context_store, mock_context, temp_dir):
        """Test save with explicit path."""
        output_path = Path(temp_dir) / "custom" / "output.qctx"
        path = context_store.save(mock_context, path=str(output_path))
        
        assert path == output_path
        assert path.exists()
    
    def test_save_adds_extension(self, context_store, mock_context, temp_dir):
        """Test that extension is added if missing."""
        output_path = Path(temp_dir) / "no_extension"
        path = context_store.save(mock_context, path=str(output_path))
        
        assert path.suffix == ".qctx"
    
    def test_save_with_description(self, context_store, mock_context):
        """Test save with description."""
        path = context_store.save(
            mock_context,
            name="described",
            description="Test description",
        )
        
        info = context_store.get_info(str(path))
        assert info["description"] == "Test description"
    
    def test_save_with_tags(self, context_store, mock_context):
        """Test save with tags."""
        path = context_store.save(
            mock_context,
            name="tagged",
            tags=["test", "demo"],
        )
        
        info = context_store.get_info(str(path))
        assert info["tags"] == ["test", "demo"]
    
    def test_save_json_format(self, context_store, mock_context):
        """Test save with JSON format."""
        path = context_store.save(
            mock_context,
            name="json_test",
            format=SerializationFormat.JSON,
            compression=CompressionType.NONE,
        )
        
        # Should be readable as JSON (after gzip header check fails)
        with open(path, 'rb') as f:
            content = f.read()
        
        # Should start with JSON brace
        assert content[0:1] == b'{'
    
    def test_save_gzip_compression(self, context_store, mock_context):
        """Test save with gzip compression."""
        path = context_store.save(
            mock_context,
            name="gzip_test",
            compression=CompressionType.GZIP,
        )
        
        with open(path, 'rb') as f:
            header = f.read(2)
        
        # GZIP magic number
        assert header == b'\x1f\x8b'


class TestContextStoreLoad:
    """Tests for ContextStore.load()."""
    
    def test_load_basic(self, context_store, mock_context):
        """Test basic load operation."""
        path = context_store.save(mock_context, name="load_test")
        
        # Create new mock context to load into
        new_context = Mock()
        new_context.chunks = {}
        new_context.total_tokens = 0
        new_context.conversation_history = []
        
        context_store.load(str(path), into_context=new_context)
        
        assert len(new_context.chunks) == 2
        assert new_context.total_tokens == 25
    
    def test_load_creates_context(self, context_store, mock_context):
        """Test load creates new context if none provided."""
        path = context_store.save(mock_context, name="create_test")
        
        # Load without providing context
        result = context_store.load(str(path))
        
        assert hasattr(result, 'chunks')
        assert hasattr(result, 'total_tokens')
    
    def test_load_merge_mode(self, context_store, mock_context):
        """Test load in merge mode."""
        path = context_store.save(mock_context, name="merge_test")
        
        # Create context with existing data
        existing_context = Mock()
        existing_context.chunks = {"existing": ContextChunk("existing", "data", 5, "x.py", "code")}
        existing_context.total_tokens = 5
        existing_context.conversation_history = [{"role": "user", "content": "existing"}]
        
        context_store.load(str(path), into_context=existing_context, merge=True)
        
        # Should have existing + loaded
        assert "existing" in existing_context.chunks
        assert "chunk1" in existing_context.chunks
    
    def test_load_replace_mode(self, context_store, mock_context):
        """Test load in replace mode (default)."""
        path = context_store.save(mock_context, name="replace_test")
        
        # Create context with existing data
        existing_context = Mock()
        existing_context.chunks = {"existing": ContextChunk("existing", "data", 5, "x.py", "code")}
        existing_context.total_tokens = 5
        existing_context.conversation_history = [{"role": "user", "content": "existing"}]
        
        context_store.load(str(path), into_context=existing_context, merge=False)
        
        # Should only have loaded data
        assert "existing" not in existing_context.chunks
        assert "chunk1" in existing_context.chunks
    
    def test_load_file_not_found(self, context_store):
        """Test load with non-existent file."""
        with pytest.raises(FileNotFoundError):
            context_store.load("/nonexistent/path.qctx")
    
    def test_load_preserves_conversation(self, context_store, mock_context):
        """Test that conversation history is loaded."""
        path = context_store.save(mock_context, name="conv_test")
        
        new_context = Mock()
        new_context.chunks = {}
        new_context.total_tokens = 0
        new_context.conversation_history = []
        
        context_store.load(str(path), into_context=new_context)
        
        assert len(new_context.conversation_history) == 2
        assert new_context.conversation_history[0]["role"] == "user"


class TestContextStoreInfo:
    """Tests for ContextStore.get_info()."""
    
    def test_get_info_basic(self, context_store, mock_context):
        """Test basic info retrieval."""
        path = context_store.save(mock_context, name="info_test")
        
        info = context_store.get_info(str(path))
        
        assert info["name"] == "info_test"
        assert info["total_chunks"] == 2
        assert info["total_tokens"] == 25
    
    def test_get_info_type_distribution(self, context_store, mock_context):
        """Test type distribution in info."""
        path = context_store.save(mock_context, name="dist_test")
        
        info = context_store.get_info(str(path))
        
        assert "type_distribution" in info
        assert info["type_distribution"]["code"] == 1
        assert info["type_distribution"]["paper"] == 1
    
    def test_get_info_file_not_found(self, context_store):
        """Test info with non-existent file."""
        with pytest.raises(FileNotFoundError):
            context_store.get_info("/nonexistent/path.qctx")


class TestContextStoreList:
    """Tests for ContextStore.list_contexts()."""
    
    def test_list_empty(self, context_store):
        """Test listing empty directory."""
        contexts = context_store.list_contexts()
        assert contexts == []
    
    def test_list_multiple(self, context_store, mock_context):
        """Test listing multiple contexts."""
        context_store.save(mock_context, name="list_test_1")
        context_store.save(mock_context, name="list_test_2")
        context_store.save(mock_context, name="list_test_3")
        
        contexts = context_store.list_contexts()
        
        assert len(contexts) == 3
        names = [c["name"] for c in contexts]
        assert "list_test_1" in names
        assert "list_test_2" in names
        assert "list_test_3" in names
    
    def test_list_sorted_by_modified(self, context_store, mock_context):
        """Test list is sorted by modification time."""
        context_store.save(mock_context, name="old")
        time.sleep(0.1)  # Ensure different timestamps
        context_store.save(mock_context, name="new")
        
        contexts = context_store.list_contexts()
        
        # Newest first
        assert contexts[0]["name"] == "new"


class TestContextStoreDelete:
    """Tests for ContextStore.delete()."""
    
    def test_delete_existing(self, context_store, mock_context):
        """Test deleting existing file."""
        path = context_store.save(mock_context, name="delete_test")
        
        assert path.exists()
        result = context_store.delete(str(path))
        
        assert result is True
        assert not path.exists()
    
    def test_delete_nonexistent(self, context_store):
        """Test deleting non-existent file."""
        result = context_store.delete("/nonexistent/path.qctx")
        assert result is False


class TestContextStoreMerge:
    """Tests for ContextStore.merge()."""
    
    def test_merge_two_files(self, context_store, temp_dir):
        """Test merging two context files."""
        # Create first context
        ctx1 = Mock()
        ctx1.chunks = {"c1": ContextChunk("c1", "content1", 10, "f1.py", "code")}
        ctx1.total_tokens = 10
        ctx1.conversation_history = []
        path1 = context_store.save(ctx1, name="merge1")
        
        # Create second context
        ctx2 = Mock()
        ctx2.chunks = {"c2": ContextChunk("c2", "content2", 15, "f2.py", "code")}
        ctx2.total_tokens = 15
        ctx2.conversation_history = []
        path2 = context_store.save(ctx2, name="merge2")
        
        # Merge
        merged_path = context_store.merge([str(path1), str(path2)], name="merged")
        
        # Verify merged content
        info = context_store.get_info(str(merged_path))
        assert info["total_chunks"] == 2
    
    def test_merge_requires_two_files(self, context_store, mock_context):
        """Test merge requires at least 2 files."""
        path = context_store.save(mock_context, name="single")
        
        with pytest.raises(ValueError, match="Need at least 2 files"):
            context_store.merge([str(path)])


class TestContextStoreDiff:
    """Tests for ContextStore.diff()."""
    
    def test_diff_identical(self, context_store, mock_context):
        """Test diff of identical files."""
        path1 = context_store.save(mock_context, name="diff1")
        path2 = context_store.save(mock_context, name="diff2")
        
        diff = context_store.diff(str(path1), str(path2))
        
        assert diff["chunks_in_file1"] == 2
        assert diff["chunks_in_file2"] == 2
        assert len(diff["only_in_file1"]) == 0
        assert len(diff["only_in_file2"]) == 0
    
    def test_diff_different(self, context_store, temp_dir):
        """Test diff of different files."""
        # Create first context
        ctx1 = Mock()
        ctx1.chunks = {
            "c1": ContextChunk("c1", "content1", 10, "f1.py", "code"),
            "c2": ContextChunk("c2", "content2", 15, "f2.py", "code"),
        }
        ctx1.total_tokens = 25
        ctx1.conversation_history = []
        path1 = context_store.save(ctx1, name="diff_a")
        
        # Create second context with different chunks
        ctx2 = Mock()
        ctx2.chunks = {
            "c2": ContextChunk("c2", "content2", 15, "f2.py", "code"),
            "c3": ContextChunk("c3", "content3", 20, "f3.py", "code"),
        }
        ctx2.total_tokens = 35
        ctx2.conversation_history = []
        path2 = context_store.save(ctx2, name="diff_b")
        
        diff = context_store.diff(str(path1), str(path2))
        
        assert "c1" in diff["only_in_file1"]
        assert "c3" in diff["only_in_file2"]
        assert diff["in_both"] == 1
        assert diff["token_diff"] == 10


class TestContextStoreExport:
    """Tests for ContextStore.export_chunks()."""
    
    def test_export_all_chunks(self, context_store, mock_context, temp_dir):
        """Test exporting all chunks."""
        path = context_store.save(mock_context, name="export_test")
        
        export_dir = Path(temp_dir) / "exported"
        count = context_store.export_chunks(str(path), str(export_dir))
        
        assert count == 2
        assert export_dir.exists()
        assert len(list(export_dir.iterdir())) == 2
    
    def test_export_filtered_by_type(self, context_store, mock_context, temp_dir):
        """Test exporting filtered by type."""
        path = context_store.save(mock_context, name="filter_test")
        
        export_dir = Path(temp_dir) / "exported_code"
        count = context_store.export_chunks(str(path), str(export_dir), chunk_type="code")
        
        assert count == 1


# ============================================================================
# Test Q-Lang Integration
# ============================================================================

class TestQLangIntegration:
    """Tests for Q-Lang command handling."""
    
    def test_handle_save_command(self, context_store, mock_context, temp_dir, capsys):
        """Test context save command."""
        qlang_context = {}
        save_path = Path(temp_dir) / "cmd_save.qctx"
        
        handle_context_command(
            context_store,
            mock_context,
            f"context save {save_path}",
            qlang_context,
        )
        
        assert save_path.exists()
        assert "last_saved_context" in qlang_context
    
    def test_handle_load_command(self, context_store, mock_context, temp_dir):
        """Test context load command."""
        path = context_store.save(mock_context, name="cmd_load")
        
        new_context = Mock()
        new_context.chunks = {}
        new_context.total_tokens = 0
        new_context.conversation_history = []
        qlang_context = {}
        
        handle_context_command(
            context_store,
            new_context,
            f"context load {path}",
            qlang_context,
        )
        
        assert len(new_context.chunks) == 2
        assert "last_loaded_context" in qlang_context
    
    def test_handle_list_command(self, context_store, mock_context, capsys):
        """Test context list command."""
        context_store.save(mock_context, name="list_cmd_1")
        context_store.save(mock_context, name="list_cmd_2")
        
        handle_context_command(
            context_store,
            mock_context,
            "context list",
            {},
        )
        
        captured = capsys.readouterr()
        assert "list_cmd_1" in captured.out or "Saved Contexts" in captured.out
    
    def test_handle_info_command(self, context_store, mock_context, capsys):
        """Test context info command."""
        path = context_store.save(mock_context, name="info_cmd")
        
        handle_context_command(
            context_store,
            mock_context,
            f"context info {path}",
            {},
        )
        
        captured = capsys.readouterr()
        assert "info_cmd" in captured.out or "Context File Info" in captured.out
    
    def test_handle_delete_command(self, context_store, mock_context, capsys):
        """Test context delete command."""
        path = context_store.save(mock_context, name="delete_cmd")
        assert path.exists()
        
        handle_context_command(
            context_store,
            mock_context,
            f"context delete {path}",
            {},
        )
        
        assert not path.exists()
    
    def test_handle_help_command(self, context_store, mock_context, capsys):
        """Test context help command."""
        handle_context_command(
            context_store,
            mock_context,
            "context help",
            {},
        )
        
        captured = capsys.readouterr()
        assert "save" in captured.out
        assert "load" in captured.out
    
    def test_handle_unknown_command(self, context_store, mock_context, capsys):
        """Test unknown context command."""
        handle_context_command(
            context_store,
            mock_context,
            "context unknown",
            {},
        )
        
        captured = capsys.readouterr()
        assert "Unknown" in captured.out or "❌" in captured.out
    
    def test_handle_missing_args(self, context_store, mock_context, capsys):
        """Test command with missing arguments."""
        handle_context_command(
            context_store,
            mock_context,
            "context save",
            {},
        )
        
        captured = capsys.readouterr()
        assert "Usage" in captured.out


# ============================================================================
# Test Availability Flag
# ============================================================================

class TestAvailability:
    """Tests for module availability flags."""
    
    def test_context_store_flag(self):
        """Test HAS_CONTEXT_STORE flag."""
        assert HAS_CONTEXT_STORE is True
    
    def test_msgpack_flag_type(self):
        """Test HAS_MSGPACK is boolean."""
        assert isinstance(HAS_MSGPACK, bool)
    
    def test_lz4_flag_type(self):
        """Test HAS_LZ4 is boolean."""
        assert isinstance(HAS_LZ4, bool)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_full_save_load_cycle(self, context_store, mock_context):
        """Test complete save/load cycle."""
        # Save
        path = context_store.save(
            mock_context,
            name="integration",
            description="Integration test",
            tags=["test"],
        )
        
        # Get info
        info = context_store.get_info(str(path))
        assert info["name"] == "integration"
        
        # Load into new context
        new_context = Mock()
        new_context.chunks = {}
        new_context.total_tokens = 0
        new_context.conversation_history = []
        
        context_store.load(str(path), into_context=new_context)
        
        # Verify data integrity
        assert len(new_context.chunks) == len(mock_context.chunks)
        for chunk_id, original_chunk in mock_context.chunks.items():
            loaded_chunk = new_context.chunks[chunk_id]
            assert loaded_chunk.content == original_chunk.content
            assert loaded_chunk.source == original_chunk.source
    
    def test_large_context_handling(self, context_store, temp_dir):
        """Test handling of larger contexts."""
        # Create context with many chunks
        ctx = Mock()
        ctx.chunks = {}
        ctx.total_tokens = 0
        ctx.conversation_history = []
        
        for i in range(100):
            chunk = ContextChunk(
                id=f"chunk_{i}",
                content=f"Content for chunk {i}\n" * 100,
                token_count=200,
                source=f"file_{i}.py",
                chunk_type="code" if i % 2 == 0 else "paper",
            )
            ctx.chunks[f"chunk_{i}"] = chunk
            ctx.total_tokens += 200
        
        # Save
        path = context_store.save(ctx, name="large_test")
        
        # Verify compression helps
        info = context_store.get_info(str(path))
        assert info["total_chunks"] == 100
        assert info["size_kb"] < 500  # Should be well compressed
        
        # Load and verify
        new_ctx = Mock()
        new_ctx.chunks = {}
        new_ctx.total_tokens = 0
        new_ctx.conversation_history = []
        
        context_store.load(str(path), into_context=new_ctx)
        assert len(new_ctx.chunks) == 100
    
    def test_unicode_content(self, context_store, temp_dir):
        """Test handling of unicode content."""
        ctx = Mock()
        ctx.chunks = {
            "unicode": ContextChunk(
                id="unicode",
                content="Hello 世界 🌍 Ελληνικά العربية",
                token_count=20,
                source="unicode.txt",
                chunk_type="text",
            )
        }
        ctx.total_tokens = 20
        ctx.conversation_history = [
            {"role": "user", "content": "Привет мир 你好世界"}
        ]
        
        path = context_store.save(ctx, name="unicode_test")
        
        new_ctx = Mock()
        new_ctx.chunks = {}
        new_ctx.total_tokens = 0
        new_ctx.conversation_history = []
        
        context_store.load(str(path), into_context=new_ctx)
        
        assert "世界" in new_ctx.chunks["unicode"].content
        assert "Привет" in new_ctx.conversation_history[0]["content"]
