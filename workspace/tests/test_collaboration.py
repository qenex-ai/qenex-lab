"""
Tests for QENEX Real-Time Collaboration Protocol.

Tests cover:
- Message serialization
- Room state management
- Participant handling
- Lock management
- Event system
- Q-Lang integration
"""

import pytest
import sys
import os
import json
import time
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch

# Add package path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "packages", "qenex-qlang", "src"))

from collaboration import (
    Message,
    MessageType,
    Participant,
    ResourceLock,
    RoomState,
    EventEmitter,
    CollaborationServer,
    CollaborationClient,
    SyncCollaborationClient,
    execute_collab_command,
    WEBSOCKETS_AVAILABLE,
)


# ============================================================================
# Test Message
# ============================================================================

class TestMessage:
    """Tests for Message class."""
    
    def test_message_creation(self):
        """Test basic message creation."""
        msg = Message(type=MessageType.JOIN, payload={"room_id": "test"})
        assert msg.type == MessageType.JOIN
        assert msg.payload["room_id"] == "test"
        assert msg.message_id is not None
    
    def test_message_with_all_fields(self):
        """Test message with all fields."""
        msg = Message(
            type=MessageType.BROADCAST,
            payload={"data": "test"},
            sender_id="client_123",
            room_id="room_456"
        )
        assert msg.sender_id == "client_123"
        assert msg.room_id == "room_456"
    
    def test_message_to_json(self):
        """Test message serialization."""
        msg = Message(
            type=MessageType.UPDATE,
            payload={"path": "a.b", "value": 42}
        )
        json_str = msg.to_json()
        
        assert isinstance(json_str, str)
        data = json.loads(json_str)
        assert data["type"] == "update"
        assert data["payload"]["value"] == 42
    
    def test_message_from_json(self):
        """Test message deserialization."""
        json_str = json.dumps({
            "type": "join",
            "payload": {"room_id": "test_room"},
            "sender_id": "abc",
            "room_id": "test_room",
            "message_id": "msg_123",
            "timestamp": 1234567890
        })
        
        msg = Message.from_json(json_str)
        
        assert msg.type == MessageType.JOIN
        assert msg.payload["room_id"] == "test_room"
        assert msg.sender_id == "abc"
        assert msg.message_id == "msg_123"
    
    def test_message_roundtrip(self):
        """Test serialization roundtrip."""
        original = Message(
            type=MessageType.RESULT,
            payload={"result": [1, 2, 3]},
            sender_id="client_1",
            room_id="room_1"
        )
        
        json_str = original.to_json()
        restored = Message.from_json(json_str)
        
        assert restored.type == original.type
        assert restored.payload == original.payload
        assert restored.sender_id == original.sender_id
        assert restored.room_id == original.room_id


class TestMessageType:
    """Tests for MessageType enum."""
    
    def test_all_message_types_exist(self):
        """Test that all message types are defined."""
        expected_types = [
            "join", "leave", "heartbeat", "ack", "error",
            "sync", "update", "patch",
            "lock", "unlock", "lock_acquired", "lock_denied", "lock_released",
            "broadcast", "direct",
            "experiment_start", "experiment_stop",
            "step_start", "step_complete", "step_failed",
            "result", "hypothesis", "validation", "insight"
        ]
        
        for t in expected_types:
            assert hasattr(MessageType, t.upper())
    
    def test_message_type_values(self):
        """Test message type values."""
        assert MessageType.JOIN.value == "join"
        assert MessageType.BROADCAST.value == "broadcast"
        assert MessageType.LOCK.value == "lock"


# ============================================================================
# Test Participant
# ============================================================================

class TestParticipant:
    """Tests for Participant class."""
    
    def test_participant_creation(self):
        """Test basic participant creation."""
        p = Participant(client_id="client_123", name="Alice")
        assert p.client_id == "client_123"
        assert p.name == "Alice"
        assert p.role == "member"
    
    def test_participant_with_role(self):
        """Test participant with custom role."""
        p = Participant(client_id="owner_1", name="Bob", role="owner")
        assert p.role == "owner"
    
    def test_participant_is_alive(self):
        """Test alive check."""
        p = Participant(client_id="test")
        p.last_heartbeat = time.time()
        
        assert p.is_alive(timeout=30)
        
        # Simulate timeout
        p.last_heartbeat = time.time() - 60
        assert not p.is_alive(timeout=30)
    
    def test_participant_metadata(self):
        """Test participant metadata."""
        p = Participant(
            client_id="test",
            metadata={"role": "researcher", "institution": "MIT"}
        )
        assert p.metadata["institution"] == "MIT"


# ============================================================================
# Test ResourceLock
# ============================================================================

class TestResourceLock:
    """Tests for ResourceLock class."""
    
    def test_lock_creation(self):
        """Test lock creation."""
        lock = ResourceLock(resource_id="data.results", holder_id="client_1")
        assert lock.resource_id == "data.results"
        assert lock.holder_id == "client_1"
        assert lock.expires_at is None
    
    def test_lock_with_expiry(self):
        """Test lock with expiry."""
        lock = ResourceLock(
            resource_id="test",
            holder_id="client_1",
            expires_at=time.time() + 60
        )
        assert not lock.is_expired()
    
    def test_lock_expired(self):
        """Test expired lock."""
        lock = ResourceLock(
            resource_id="test",
            holder_id="client_1",
            expires_at=time.time() - 1
        )
        assert lock.is_expired()
    
    def test_lock_no_expiry_never_expires(self):
        """Test lock without expiry never expires."""
        lock = ResourceLock(resource_id="test", holder_id="client_1")
        assert not lock.is_expired()


# ============================================================================
# Test RoomState
# ============================================================================

class TestRoomState:
    """Tests for RoomState class."""
    
    def test_room_creation(self):
        """Test room creation."""
        room = RoomState(room_id="room_123", name="Test Room")
        assert room.room_id == "room_123"
        assert room.name == "Test Room"
        assert len(room.participants) == 0
        assert room.version == 0
    
    def test_add_participant(self):
        """Test adding participant."""
        room = RoomState(room_id="test")
        p = Participant(client_id="client_1", name="Alice")
        
        room.add_participant(p)
        
        assert "client_1" in room.participants
        assert room.version == 1
    
    def test_remove_participant(self):
        """Test removing participant."""
        room = RoomState(room_id="test")
        p = Participant(client_id="client_1", name="Alice")
        room.add_participant(p)
        
        room.remove_participant("client_1")
        
        assert "client_1" not in room.participants
    
    def test_remove_participant_releases_locks(self):
        """Test that removing participant releases their locks."""
        room = RoomState(room_id="test")
        p = Participant(client_id="client_1")
        room.add_participant(p)
        room.acquire_lock("resource_1", "client_1")
        
        assert "resource_1" in room.locks
        
        room.remove_participant("client_1")
        
        assert "resource_1" not in room.locks
    
    def test_acquire_lock(self):
        """Test acquiring a lock."""
        room = RoomState(room_id="test")
        
        result = room.acquire_lock("data.results", "client_1", ttl=60)
        
        assert result is True
        assert "data.results" in room.locks
        assert room.locks["data.results"].holder_id == "client_1"
    
    def test_acquire_lock_already_held(self):
        """Test acquiring already held lock."""
        room = RoomState(room_id="test")
        room.acquire_lock("resource", "client_1")
        
        result = room.acquire_lock("resource", "client_2")
        
        assert result is False
    
    def test_acquire_lock_same_holder(self):
        """Test extending own lock."""
        room = RoomState(room_id="test")
        room.acquire_lock("resource", "client_1", ttl=60)
        
        result = room.acquire_lock("resource", "client_1", ttl=120)
        
        assert result is True  # Should extend
    
    def test_release_lock(self):
        """Test releasing a lock."""
        room = RoomState(room_id="test")
        room.acquire_lock("resource", "client_1")
        
        result = room.release_lock("resource", "client_1")
        
        assert result is True
        assert "resource" not in room.locks
    
    def test_release_lock_wrong_holder(self):
        """Test releasing lock by non-holder."""
        room = RoomState(room_id="test")
        room.acquire_lock("resource", "client_1")
        
        result = room.release_lock("resource", "client_2")
        
        assert result is False
        assert "resource" in room.locks
    
    def test_update_state(self):
        """Test state update."""
        room = RoomState(room_id="test")
        
        room.update_state("results.accuracy", 0.95, "client_1")
        
        assert room.state["results"]["accuracy"] == 0.95
        assert len(room.history) == 1
    
    def test_update_state_nested(self):
        """Test nested state update."""
        room = RoomState(room_id="test")
        
        room.update_state("a.b.c.d", "value", "client_1")
        
        assert room.state["a"]["b"]["c"]["d"] == "value"
    
    def test_to_dict(self):
        """Test room serialization."""
        room = RoomState(room_id="test", name="Test Room")
        room.add_participant(Participant(client_id="c1", name="Alice"))
        room.update_state("key", "value", "c1")
        
        d = room.to_dict()
        
        assert d["room_id"] == "test"
        assert d["name"] == "Test Room"
        assert "c1" in d["participants"]
        assert d["state"]["key"] == "value"


# ============================================================================
# Test EventEmitter
# ============================================================================

class TestEventEmitter:
    """Tests for EventEmitter class."""
    
    def test_emit_event(self):
        """Test emitting events."""
        emitter = EventEmitter()
        results = []
        
        emitter.on("test", lambda x: results.append(x))
        emitter.emit("test", 42)
        
        assert results == [42]
    
    def test_multiple_handlers(self):
        """Test multiple handlers for same event."""
        emitter = EventEmitter()
        results = []
        
        emitter.on("test", lambda: results.append(1))
        emitter.on("test", lambda: results.append(2))
        emitter.emit("test")
        
        assert results == [1, 2]
    
    def test_remove_handler(self):
        """Test removing handler."""
        emitter = EventEmitter()
        results = []
        
        handler = lambda: results.append(1)
        emitter.on("test", handler)
        emitter.off("test", handler)
        emitter.emit("test")
        
        assert results == []
    
    def test_handler_error_doesnt_stop_others(self):
        """Test that handler error doesn't stop other handlers."""
        emitter = EventEmitter()
        results = []
        
        def bad_handler():
            raise ValueError("oops")
        
        emitter.on("test", bad_handler)
        emitter.on("test", lambda: results.append(1))
        emitter.emit("test")
        
        assert results == [1]
    
    def test_emit_async(self):
        """Test async event emission."""
        emitter = EventEmitter()
        results = []
        
        async def async_handler(x):
            results.append(x)
        
        emitter.on("test", async_handler)
        
        # Run async emit in event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(emitter.emit_async("test", 42))
        finally:
            loop.close()
        
        assert results == [42]


# ============================================================================
# Test CollaborationServer (Unit Tests)
# ============================================================================

class TestCollaborationServerUnit:
    """Unit tests for CollaborationServer."""
    
    def test_server_creation(self):
        """Test server creation."""
        server = CollaborationServer(host="127.0.0.1", port=9999)
        assert server.host == "127.0.0.1"
        assert server.port == 9999
        assert len(server.rooms) == 0
    
    def test_server_room_management(self):
        """Test room management."""
        server = CollaborationServer()
        
        # Create room
        server.rooms["room_1"] = RoomState(room_id="room_1", name="Test")
        
        assert "room_1" in server.rooms
        assert server.rooms["room_1"].name == "Test"


# ============================================================================
# Test CollaborationClient (Unit Tests)
# ============================================================================

class TestCollaborationClientUnit:
    """Unit tests for CollaborationClient."""
    
    def test_client_creation(self):
        """Test client creation."""
        client = CollaborationClient(server_url="ws://test:8765", name="TestClient")
        assert client.server_url == "ws://test:8765"
        assert client.name == "TestClient"
        assert client.current_room is None
    
    def test_client_default_name(self):
        """Test default client name."""
        client = CollaborationClient()
        assert client.name.startswith("Client_")
    
    def test_register_message_handler(self):
        """Test registering message handler."""
        client = CollaborationClient()
        
        handler = MagicMock()
        client.on_message(MessageType.BROADCAST, handler)
        
        assert handler in client._message_handlers[MessageType.BROADCAST]
    
    def test_update_room_state(self):
        """Test local room state update."""
        client = CollaborationClient()
        
        state_dict = {
            "room_id": "room_1",
            "name": "Test Room",
            "state": {"key": "value"},
            "version": 5
        }
        
        client._update_room_state(state_dict)
        
        assert client.room_state is not None
        assert client.room_state.room_id == "room_1"
        assert client.room_state.version == 5


# ============================================================================
# Test SyncCollaborationClient
# ============================================================================

class TestSyncCollaborationClient:
    """Tests for SyncCollaborationClient."""
    
    def test_sync_client_creation(self):
        """Test sync client creation."""
        client = SyncCollaborationClient(
            server_url="ws://test:8765",
            name="SyncTest"
        )
        assert client._client.name == "SyncTest"
    
    def test_sync_client_properties(self):
        """Test sync client properties."""
        client = SyncCollaborationClient()
        
        # Before connect
        assert client.client_id is None
        assert client.current_room is None


# ============================================================================
# Test Q-Lang Integration
# ============================================================================

class TestQLangIntegration:
    """Tests for Q-Lang integration."""
    
    def test_collab_status_not_connected(self):
        """Test status when not connected."""
        mock_interpreter = MagicMock()
        mock_interpreter._collab_client = None
        
        result = execute_collab_command(mock_interpreter, "status")
        
        assert "Not connected" in result
    
    def test_collab_help(self):
        """Test help command."""
        mock_interpreter = MagicMock()
        
        result = execute_collab_command(mock_interpreter, "help")
        
        assert "Commands" in result
        assert "connect" in result
        assert "join" in result
        assert "broadcast" in result
    
    def test_collab_unknown_command(self):
        """Test unknown command."""
        mock_interpreter = MagicMock()
        
        result = execute_collab_command(mock_interpreter, "invalid_cmd")
        
        assert "Unknown command" in result
    
    def test_collab_disconnect_not_connected(self):
        """Test disconnect when not connected."""
        mock_interpreter = MagicMock()
        mock_interpreter._collab_client = None
        
        result = execute_collab_command(mock_interpreter, "disconnect")
        
        assert "Disconnected" in result
    
    def test_collab_join_not_connected(self):
        """Test join when not connected."""
        mock_interpreter = MagicMock()
        mock_interpreter._collab_client = None
        
        result = execute_collab_command(mock_interpreter, "join test_room")
        
        assert "Not connected" in result
    
    def test_collab_broadcast_not_connected(self):
        """Test broadcast when not connected."""
        mock_interpreter = MagicMock()
        mock_interpreter._collab_client = None
        
        result = execute_collab_command(mock_interpreter, "broadcast hello")
        
        assert "Not connected" in result


# ============================================================================
# Integration Tests (with mocked WebSocket)
# ============================================================================

class TestIntegration:
    """Integration tests."""
    
    def test_full_room_workflow(self):
        """Test complete room workflow."""
        # Create room
        room = RoomState(room_id="exp_123", name="Quantum Experiment")
        
        # Add participants
        alice = Participant(client_id="alice", name="Alice", role="owner")
        bob = Participant(client_id="bob", name="Bob")
        
        room.add_participant(alice)
        room.add_participant(bob)
        
        assert len(room.participants) == 2
        
        # Alice acquires lock
        room.acquire_lock("results", "alice")
        
        # Alice updates state
        room.update_state("results.energy", -75.5, "alice")
        room.update_state("results.converged", True, "alice")
        
        # Alice releases lock
        room.release_lock("results", "alice")
        
        # Bob can now update
        room.update_state("results.verified", True, "bob")
        
        # Verify state
        assert room.state["results"]["energy"] == -75.5
        assert room.state["results"]["converged"] == True
        assert room.state["results"]["verified"] == True
        assert len(room.history) == 3
    
    def test_message_protocol(self):
        """Test message protocol flow."""
        # Simulate join message
        join_msg = Message(
            type=MessageType.JOIN,
            payload={"room_id": "test", "name": "Alice"}
        )
        
        json_str = join_msg.to_json()
        restored = Message.from_json(json_str)
        
        assert restored.type == MessageType.JOIN
        assert restored.payload["name"] == "Alice"
        
        # Simulate update message
        update_msg = Message(
            type=MessageType.UPDATE,
            payload={"path": "results.accuracy", "value": 0.95},
            room_id="test"
        )
        
        json_str = update_msg.to_json()
        restored = Message.from_json(json_str)
        
        assert restored.type == MessageType.UPDATE
        assert restored.payload["path"] == "results.accuracy"
    
    def test_concurrent_lock_requests(self):
        """Test handling concurrent lock requests."""
        room = RoomState(room_id="test")
        
        # First request wins
        assert room.acquire_lock("resource", "client_1") is True
        assert room.acquire_lock("resource", "client_2") is False
        assert room.acquire_lock("resource", "client_3") is False
        
        # Only original holder can release
        assert room.release_lock("resource", "client_2") is False
        assert room.release_lock("resource", "client_1") is True
        
        # Now others can acquire
        assert room.acquire_lock("resource", "client_2") is True


# ============================================================================
# Test Availability
# ============================================================================

class TestAvailability:
    """Test module availability."""
    
    def test_websockets_flag(self):
        """Test websockets availability flag."""
        assert isinstance(WEBSOCKETS_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
