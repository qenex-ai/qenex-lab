"""
QENEX Real-Time Collaboration Protocol
======================================

Distributed experiment coordination for multi-agent scientific discovery.

Features:
- WebSocket-based real-time communication
- Shared experiment state synchronization
- Event-driven architecture for experiment updates
- Room-based collaboration (multiple experiments)
- Conflict resolution for concurrent modifications
- Heartbeat/presence tracking
- Message queuing for offline clients

Protocol Messages:
- JOIN: Join an experiment room
- LEAVE: Leave an experiment room
- SYNC: Full state synchronization
- UPDATE: Incremental state update
- LOCK: Acquire resource lock
- UNLOCK: Release resource lock
- BROADCAST: Send to all room members
- HEARTBEAT: Keep-alive signal

Usage:
    # Server
    server = CollaborationServer(port=8765)
    await server.start()
    
    # Client
    client = CollaborationClient("ws://localhost:8765")
    await client.connect()
    await client.join_room("experiment_123")
    await client.broadcast({"type": "result", "data": {...}})

Author: QENEX Sovereign Agent
Version: 1.0.0
"""

import asyncio
import json
import uuid
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional, Callable, Set
from enum import Enum
from datetime import datetime
from collections import defaultdict
import threading
import queue

# Try to import websockets
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    from websockets.client import WebSocketClientProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    WebSocketServerProtocol = Any
    WebSocketClientProtocol = Any


# ============================================================================
# Protocol Messages
# ============================================================================

class MessageType(Enum):
    """Types of protocol messages."""
    # Connection
    JOIN = "join"
    LEAVE = "leave"
    HEARTBEAT = "heartbeat"
    ACK = "ack"
    ERROR = "error"
    
    # State
    SYNC = "sync"
    UPDATE = "update"
    PATCH = "patch"
    
    # Coordination
    LOCK = "lock"
    UNLOCK = "unlock"
    LOCK_ACQUIRED = "lock_acquired"
    LOCK_DENIED = "lock_denied"
    LOCK_RELEASED = "lock_released"
    
    # Communication
    BROADCAST = "broadcast"
    DIRECT = "direct"
    
    # Experiment
    EXPERIMENT_START = "experiment_start"
    EXPERIMENT_STOP = "experiment_stop"
    STEP_START = "step_start"
    STEP_COMPLETE = "step_complete"
    STEP_FAILED = "step_failed"
    RESULT = "result"
    
    # Discovery
    HYPOTHESIS = "hypothesis"
    VALIDATION = "validation"
    INSIGHT = "insight"


@dataclass
class Message:
    """A protocol message."""
    type: MessageType
    payload: Dict[str, Any] = field(default_factory=dict)
    sender_id: str = ""
    room_id: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    timestamp: float = field(default_factory=time.time)
    
    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps({
            "type": self.type.value,
            "payload": self.payload,
            "sender_id": self.sender_id,
            "room_id": self.room_id,
            "message_id": self.message_id,
            "timestamp": self.timestamp
        })
    
    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Deserialize from JSON."""
        data = json.loads(json_str)
        return cls(
            type=MessageType(data["type"]),
            payload=data.get("payload", {}),
            sender_id=data.get("sender_id", ""),
            room_id=data.get("room_id", ""),
            message_id=data.get("message_id", uuid.uuid4().hex[:12]),
            timestamp=data.get("timestamp", time.time())
        )


# ============================================================================
# State Management
# ============================================================================

@dataclass
class Participant:
    """A participant in a collaboration room."""
    client_id: str
    name: str = "Anonymous"
    role: str = "member"  # owner, admin, member, observer
    joined_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_alive(self, timeout: float = 30.0) -> bool:
        """Check if participant is still connected."""
        return (time.time() - self.last_heartbeat) < timeout


@dataclass
class ResourceLock:
    """A lock on a shared resource."""
    resource_id: str
    holder_id: str
    acquired_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if lock has expired."""
        if self.expires_at is None:
            return False
        return time.time() > self.expires_at


@dataclass
class RoomState:
    """State of a collaboration room."""
    room_id: str
    name: str = ""
    created_at: float = field(default_factory=time.time)
    participants: Dict[str, Participant] = field(default_factory=dict)
    locks: Dict[str, ResourceLock] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    version: int = 0
    
    def add_participant(self, participant: Participant):
        """Add a participant to the room."""
        self.participants[participant.client_id] = participant
        self.version += 1
    
    def remove_participant(self, client_id: str):
        """Remove a participant from the room."""
        if client_id in self.participants:
            del self.participants[client_id]
            # Release any locks held by this participant
            locks_to_release = [
                rid for rid, lock in self.locks.items()
                if lock.holder_id == client_id
            ]
            for rid in locks_to_release:
                del self.locks[rid]
            self.version += 1
    
    def acquire_lock(self, resource_id: str, client_id: str, 
                     ttl: float = 60.0) -> bool:
        """Try to acquire a lock on a resource."""
        if resource_id in self.locks:
            lock = self.locks[resource_id]
            if lock.holder_id == client_id:
                # Extend existing lock
                lock.expires_at = time.time() + ttl
                return True
            if lock.is_expired():
                # Take over expired lock
                pass
            else:
                return False
        
        self.locks[resource_id] = ResourceLock(
            resource_id=resource_id,
            holder_id=client_id,
            expires_at=time.time() + ttl
        )
        self.version += 1
        return True
    
    def release_lock(self, resource_id: str, client_id: str) -> bool:
        """Release a lock on a resource."""
        if resource_id not in self.locks:
            return False
        if self.locks[resource_id].holder_id != client_id:
            return False
        del self.locks[resource_id]
        self.version += 1
        return True
    
    def update_state(self, path: str, value: Any, client_id: str):
        """Update state at a specific path."""
        # Simple dot-notation path update
        parts = path.split(".")
        current = self.state
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
        
        # Record in history
        self.history.append({
            "timestamp": time.time(),
            "client_id": client_id,
            "path": path,
            "value": str(value)[:100]
        })
        if len(self.history) > 1000:
            self.history = self.history[-500:]
        
        self.version += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "room_id": self.room_id,
            "name": self.name,
            "created_at": self.created_at,
            "participants": {
                k: asdict(v) for k, v in self.participants.items()
            },
            "locks": {
                k: asdict(v) for k, v in self.locks.items()
            },
            "state": self.state,
            "version": self.version
        }


# ============================================================================
# Event System
# ============================================================================

class EventEmitter:
    """Simple event emitter for pub/sub."""
    
    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = defaultdict(list)
    
    def on(self, event: str, handler: Callable):
        """Register an event handler."""
        self._handlers[event].append(handler)
    
    def off(self, event: str, handler: Callable):
        """Unregister an event handler."""
        if handler in self._handlers[event]:
            self._handlers[event].remove(handler)
    
    def emit(self, event: str, *args, **kwargs):
        """Emit an event to all handlers."""
        for handler in self._handlers[event]:
            try:
                handler(*args, **kwargs)
            except Exception as e:
                print(f"[EventEmitter] Handler error: {e}")
    
    async def emit_async(self, event: str, *args, **kwargs):
        """Emit an event asynchronously."""
        for handler in self._handlers[event]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(*args, **kwargs)
                else:
                    handler(*args, **kwargs)
            except Exception as e:
                print(f"[EventEmitter] Async handler error: {e}")


# ============================================================================
# Collaboration Server
# ============================================================================

class CollaborationServer:
    """
    WebSocket server for real-time collaboration.
    
    Manages rooms, participants, and message routing.
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        self.host = host
        self.port = port
        self.rooms: Dict[str, RoomState] = {}
        self.clients: Dict[str, WebSocketServerProtocol] = {}
        self.client_rooms: Dict[str, Set[str]] = defaultdict(set)
        self.events = EventEmitter()
        self._running = False
        self._server = None
    
    async def start(self):
        """Start the collaboration server."""
        if not WEBSOCKETS_AVAILABLE:
            print("[Server] WebSockets not available. Install: pip install websockets")
            return
        
        self._running = True
        self._server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port
        )
        print(f"[Server] Collaboration server started on ws://{self.host}:{self.port}")
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_loop())
        
        await self._server.wait_closed()
    
    async def stop(self):
        """Stop the server."""
        self._running = False
        if self._server:
            self._server.close()
            await self._server.wait_closed()
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a client connection."""
        client_id = uuid.uuid4().hex[:12]
        self.clients[client_id] = websocket
        print(f"[Server] Client connected: {client_id}")
        
        try:
            async for raw_message in websocket:
                try:
                    message = Message.from_json(raw_message)
                    message.sender_id = client_id
                    await self._handle_message(client_id, message)
                except json.JSONDecodeError:
                    await self._send_error(client_id, "Invalid JSON")
                except Exception as e:
                    await self._send_error(client_id, str(e))
        finally:
            # Clean up on disconnect
            await self._handle_disconnect(client_id)
    
    async def _handle_message(self, client_id: str, message: Message):
        """Handle an incoming message."""
        handlers = {
            MessageType.JOIN: self._handle_join,
            MessageType.LEAVE: self._handle_leave,
            MessageType.HEARTBEAT: self._handle_heartbeat,
            MessageType.UPDATE: self._handle_update,
            MessageType.LOCK: self._handle_lock,
            MessageType.UNLOCK: self._handle_unlock,
            MessageType.BROADCAST: self._handle_broadcast,
            MessageType.DIRECT: self._handle_direct,
            MessageType.SYNC: self._handle_sync,
        }
        
        handler = handlers.get(message.type)
        if handler:
            await handler(client_id, message)
        else:
            # Default: broadcast to room
            if message.room_id:
                await self._broadcast_to_room(message.room_id, message, exclude={client_id})
    
    async def _handle_join(self, client_id: str, message: Message):
        """Handle room join request."""
        room_id = message.payload.get("room_id", message.room_id)
        name = message.payload.get("name", f"User_{client_id[:6]}")
        
        # Create room if it doesn't exist
        if room_id not in self.rooms:
            self.rooms[room_id] = RoomState(
                room_id=room_id,
                name=message.payload.get("room_name", f"Room {room_id}")
            )
        
        room = self.rooms[room_id]
        
        # Add participant
        participant = Participant(
            client_id=client_id,
            name=name,
            role="owner" if len(room.participants) == 0 else "member",
            metadata=message.payload.get("metadata", {})
        )
        room.add_participant(participant)
        self.client_rooms[client_id].add(room_id)
        
        # Send ack with room state
        await self._send(client_id, Message(
            type=MessageType.ACK,
            payload={
                "action": "join",
                "room_id": room_id,
                "client_id": client_id,
                "state": room.to_dict()
            },
            room_id=room_id
        ))
        
        # Notify others
        await self._broadcast_to_room(room_id, Message(
            type=MessageType.JOIN,
            payload={"participant": asdict(participant)},
            sender_id=client_id,
            room_id=room_id
        ), exclude={client_id})
        
        await self.events.emit_async("join", client_id, room_id, participant)
    
    async def _handle_leave(self, client_id: str, message: Message):
        """Handle room leave request."""
        room_id = message.payload.get("room_id", message.room_id)
        
        if room_id in self.rooms:
            self.rooms[room_id].remove_participant(client_id)
            self.client_rooms[client_id].discard(room_id)
            
            # Notify others
            await self._broadcast_to_room(room_id, Message(
                type=MessageType.LEAVE,
                payload={"client_id": client_id},
                sender_id=client_id,
                room_id=room_id
            ))
            
            # Remove empty room
            if not self.rooms[room_id].participants:
                del self.rooms[room_id]
        
        await self._send(client_id, Message(
            type=MessageType.ACK,
            payload={"action": "leave", "room_id": room_id}
        ))
    
    async def _handle_heartbeat(self, client_id: str, message: Message):
        """Handle heartbeat."""
        for room_id in self.client_rooms[client_id]:
            if room_id in self.rooms:
                room = self.rooms[room_id]
                if client_id in room.participants:
                    room.participants[client_id].last_heartbeat = time.time()
        
        await self._send(client_id, Message(
            type=MessageType.ACK,
            payload={"action": "heartbeat", "server_time": time.time()}
        ))
    
    async def _handle_update(self, client_id: str, message: Message):
        """Handle state update."""
        room_id = message.room_id
        if room_id not in self.rooms:
            await self._send_error(client_id, f"Room not found: {room_id}")
            return
        
        room = self.rooms[room_id]
        path = message.payload.get("path", "")
        value = message.payload.get("value")
        
        # Check for lock
        if path in room.locks:
            lock = room.locks[path]
            if lock.holder_id != client_id and not lock.is_expired():
                await self._send(client_id, Message(
                    type=MessageType.ERROR,
                    payload={"error": f"Resource locked by {lock.holder_id}"}
                ))
                return
        
        room.update_state(path, value, client_id)
        
        # Broadcast update
        await self._broadcast_to_room(room_id, Message(
            type=MessageType.UPDATE,
            payload={"path": path, "value": value, "version": room.version},
            sender_id=client_id,
            room_id=room_id
        ))
    
    async def _handle_lock(self, client_id: str, message: Message):
        """Handle lock request."""
        room_id = message.room_id
        if room_id not in self.rooms:
            await self._send_error(client_id, f"Room not found: {room_id}")
            return
        
        room = self.rooms[room_id]
        resource_id = message.payload.get("resource_id", "")
        ttl = message.payload.get("ttl", 60.0)
        
        if room.acquire_lock(resource_id, client_id, ttl):
            await self._send(client_id, Message(
                type=MessageType.LOCK_ACQUIRED,
                payload={"resource_id": resource_id, "ttl": ttl},
                room_id=room_id
            ))
            await self._broadcast_to_room(room_id, Message(
                type=MessageType.LOCK,
                payload={"resource_id": resource_id, "holder_id": client_id},
                sender_id=client_id,
                room_id=room_id
            ), exclude={client_id})
        else:
            holder = room.locks.get(resource_id)
            await self._send(client_id, Message(
                type=MessageType.LOCK_DENIED,
                payload={
                    "resource_id": resource_id,
                    "holder_id": holder.holder_id if holder else None
                },
                room_id=room_id
            ))
    
    async def _handle_unlock(self, client_id: str, message: Message):
        """Handle unlock request."""
        room_id = message.room_id
        if room_id not in self.rooms:
            return
        
        room = self.rooms[room_id]
        resource_id = message.payload.get("resource_id", "")
        
        if room.release_lock(resource_id, client_id):
            await self._broadcast_to_room(room_id, Message(
                type=MessageType.LOCK_RELEASED,
                payload={"resource_id": resource_id},
                sender_id=client_id,
                room_id=room_id
            ))
    
    async def _handle_broadcast(self, client_id: str, message: Message):
        """Handle broadcast to room."""
        room_id = message.room_id
        if room_id:
            await self._broadcast_to_room(room_id, message, exclude={client_id})
    
    async def _handle_direct(self, client_id: str, message: Message):
        """Handle direct message."""
        target_id = message.payload.get("target_id", "")
        if target_id in self.clients:
            await self._send(target_id, message)
    
    async def _handle_sync(self, client_id: str, message: Message):
        """Handle sync request."""
        room_id = message.room_id
        if room_id in self.rooms:
            await self._send(client_id, Message(
                type=MessageType.SYNC,
                payload={"state": self.rooms[room_id].to_dict()},
                room_id=room_id
            ))
    
    async def _handle_disconnect(self, client_id: str):
        """Handle client disconnect."""
        print(f"[Server] Client disconnected: {client_id}")
        
        # Leave all rooms
        for room_id in list(self.client_rooms[client_id]):
            if room_id in self.rooms:
                self.rooms[room_id].remove_participant(client_id)
                await self._broadcast_to_room(room_id, Message(
                    type=MessageType.LEAVE,
                    payload={"client_id": client_id, "reason": "disconnect"},
                    room_id=room_id
                ))
                # Remove empty room
                if not self.rooms[room_id].participants:
                    del self.rooms[room_id]
        
        del self.client_rooms[client_id]
        del self.clients[client_id]
        
        await self.events.emit_async("disconnect", client_id)
    
    async def _send(self, client_id: str, message: Message):
        """Send message to a client."""
        if client_id in self.clients:
            try:
                await self.clients[client_id].send(message.to_json())
            except Exception as e:
                print(f"[Server] Send error to {client_id}: {e}")
    
    async def _send_error(self, client_id: str, error: str):
        """Send error to a client."""
        await self._send(client_id, Message(
            type=MessageType.ERROR,
            payload={"error": error}
        ))
    
    async def _broadcast_to_room(self, room_id: str, message: Message, 
                                  exclude: Set[str] = None):
        """Broadcast message to all room participants."""
        exclude = exclude or set()
        if room_id not in self.rooms:
            return
        
        for client_id in self.rooms[room_id].participants:
            if client_id not in exclude:
                await self._send(client_id, message)
    
    async def _cleanup_loop(self):
        """Periodic cleanup of stale connections and locks."""
        while self._running:
            await asyncio.sleep(30)
            
            # Clean up stale participants
            for room_id, room in list(self.rooms.items()):
                stale = [
                    cid for cid, p in room.participants.items()
                    if not p.is_alive(timeout=60)
                ]
                for client_id in stale:
                    room.remove_participant(client_id)
                    await self._broadcast_to_room(room_id, Message(
                        type=MessageType.LEAVE,
                        payload={"client_id": client_id, "reason": "timeout"},
                        room_id=room_id
                    ))
                
                # Clean expired locks
                expired = [
                    rid for rid, lock in room.locks.items()
                    if lock.is_expired()
                ]
                for resource_id in expired:
                    del room.locks[resource_id]
                    await self._broadcast_to_room(room_id, Message(
                        type=MessageType.LOCK_RELEASED,
                        payload={"resource_id": resource_id, "reason": "expired"},
                        room_id=room_id
                    ))


# ============================================================================
# Collaboration Client
# ============================================================================

class CollaborationClient:
    """
    WebSocket client for real-time collaboration.
    
    Connects to a CollaborationServer and participates in rooms.
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765", name: str = None):
        self.server_url = server_url
        self.name = name or f"Client_{uuid.uuid4().hex[:6]}"
        self.client_id: Optional[str] = None
        self.current_room: Optional[str] = None
        self.room_state: Optional[RoomState] = None
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._connected = False
        self._message_handlers: Dict[MessageType, List[Callable]] = defaultdict(list)
        self.events = EventEmitter()
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._receive_task: Optional[asyncio.Task] = None
    
    def on_message(self, msg_type: MessageType, handler: Callable):
        """Register a message handler."""
        self._message_handlers[msg_type].append(handler)
    
    async def connect(self) -> bool:
        """Connect to the server."""
        if not WEBSOCKETS_AVAILABLE:
            print("[Client] WebSockets not available. Install: pip install websockets")
            return False
        
        try:
            self._websocket = await websockets.connect(self.server_url)
            self._connected = True
            
            # Start receive loop
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            print(f"[Client] Connected to {self.server_url}")
            return True
            
        except Exception as e:
            print(f"[Client] Connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the server."""
        self._connected = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        if self._receive_task:
            self._receive_task.cancel()
        
        if self._websocket:
            await self._websocket.close()
            self._websocket = None
        
        print("[Client] Disconnected")
    
    async def join_room(self, room_id: str, room_name: str = None,
                        metadata: Dict[str, Any] = None) -> bool:
        """Join a collaboration room."""
        if not self._connected:
            return False
        
        await self._send(Message(
            type=MessageType.JOIN,
            payload={
                "room_id": room_id,
                "room_name": room_name or f"Room {room_id}",
                "name": self.name,
                "metadata": metadata or {}
            },
            room_id=room_id
        ))
        
        self.current_room = room_id
        return True
    
    async def leave_room(self, room_id: str = None):
        """Leave a collaboration room."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        await self._send(Message(
            type=MessageType.LEAVE,
            payload={"room_id": room_id},
            room_id=room_id
        ))
        
        if room_id == self.current_room:
            self.current_room = None
            self.room_state = None
    
    async def update(self, path: str, value: Any, room_id: str = None):
        """Update state at a path."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        await self._send(Message(
            type=MessageType.UPDATE,
            payload={"path": path, "value": value},
            room_id=room_id
        ))
    
    async def acquire_lock(self, resource_id: str, ttl: float = 60.0,
                           room_id: str = None) -> bool:
        """Try to acquire a lock."""
        room_id = room_id or self.current_room
        if not room_id:
            return False
        
        await self._send(Message(
            type=MessageType.LOCK,
            payload={"resource_id": resource_id, "ttl": ttl},
            room_id=room_id
        ))
        return True  # Actual result comes via message
    
    async def release_lock(self, resource_id: str, room_id: str = None):
        """Release a lock."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        await self._send(Message(
            type=MessageType.UNLOCK,
            payload={"resource_id": resource_id},
            room_id=room_id
        ))
    
    async def broadcast(self, payload: Dict[str, Any], room_id: str = None):
        """Broadcast a message to all room participants."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        await self._send(Message(
            type=MessageType.BROADCAST,
            payload=payload,
            room_id=room_id
        ))
    
    async def send_direct(self, target_id: str, payload: Dict[str, Any]):
        """Send a direct message to another client."""
        await self._send(Message(
            type=MessageType.DIRECT,
            payload={"target_id": target_id, **payload}
        ))
    
    async def sync(self, room_id: str = None):
        """Request full state sync."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        await self._send(Message(
            type=MessageType.SYNC,
            room_id=room_id
        ))
    
    async def send_experiment_update(self, event_type: str, data: Dict[str, Any],
                                      room_id: str = None):
        """Send an experiment-related update."""
        room_id = room_id or self.current_room
        if not room_id:
            return
        
        msg_type = {
            "start": MessageType.EXPERIMENT_START,
            "stop": MessageType.EXPERIMENT_STOP,
            "step_start": MessageType.STEP_START,
            "step_complete": MessageType.STEP_COMPLETE,
            "step_failed": MessageType.STEP_FAILED,
            "result": MessageType.RESULT,
            "hypothesis": MessageType.HYPOTHESIS,
            "validation": MessageType.VALIDATION,
            "insight": MessageType.INSIGHT,
        }.get(event_type, MessageType.BROADCAST)
        
        await self._send(Message(
            type=msg_type,
            payload=data,
            room_id=room_id
        ))
    
    async def _send(self, message: Message):
        """Send a message to the server."""
        if self._websocket and self._connected:
            try:
                await self._websocket.send(message.to_json())
            except Exception as e:
                print(f"[Client] Send error: {e}")
    
    async def _receive_loop(self):
        """Receive and process messages."""
        try:
            async for raw_message in self._websocket:
                try:
                    message = Message.from_json(raw_message)
                    await self._handle_message(message)
                except json.JSONDecodeError:
                    print("[Client] Invalid message received")
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[Client] Receive error: {e}")
            self._connected = False
    
    async def _handle_message(self, message: Message):
        """Handle an incoming message."""
        # Update client_id from ACK
        if message.type == MessageType.ACK:
            if "client_id" in message.payload:
                self.client_id = message.payload["client_id"]
            if "state" in message.payload:
                self._update_room_state(message.payload["state"])
        
        # Update room state on sync
        if message.type == MessageType.SYNC:
            if "state" in message.payload:
                self._update_room_state(message.payload["state"])
        
        # Call registered handlers
        for handler in self._message_handlers[message.type]:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                print(f"[Client] Handler error: {e}")
        
        # Emit event
        await self.events.emit_async(message.type.value, message)
    
    def _update_room_state(self, state_dict: Dict[str, Any]):
        """Update local room state from dict."""
        if not state_dict:
            return
        
        self.room_state = RoomState(
            room_id=state_dict.get("room_id", ""),
            name=state_dict.get("name", ""),
            created_at=state_dict.get("created_at", time.time()),
            state=state_dict.get("state", {}),
            version=state_dict.get("version", 0)
        )
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeats."""
        try:
            while self._connected:
                await asyncio.sleep(15)
                if self._connected:
                    await self._send(Message(type=MessageType.HEARTBEAT))
        except asyncio.CancelledError:
            pass


# ============================================================================
# Synchronous Wrapper
# ============================================================================

class SyncCollaborationClient:
    """
    Synchronous wrapper for CollaborationClient.
    
    Runs the async client in a background thread.
    """
    
    def __init__(self, server_url: str = "ws://localhost:8765", name: str = None):
        self._client = CollaborationClient(server_url, name)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._message_queue: queue.Queue = queue.Queue()
    
    def connect(self) -> bool:
        """Connect to server (blocking)."""
        self._loop = asyncio.new_event_loop()
        
        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()
        
        self._thread = threading.Thread(target=run_loop, daemon=True)
        self._thread.start()
        
        future = asyncio.run_coroutine_threadsafe(
            self._client.connect(),
            self._loop
        )
        return future.result(timeout=10)
    
    def disconnect(self):
        """Disconnect from server."""
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._client.disconnect(),
                self._loop
            ).result(timeout=5)
            self._loop.call_soon_threadsafe(self._loop.stop)
    
    def join_room(self, room_id: str, **kwargs) -> bool:
        """Join a room."""
        future = asyncio.run_coroutine_threadsafe(
            self._client.join_room(room_id, **kwargs),
            self._loop
        )
        return future.result(timeout=5)
    
    def leave_room(self, room_id: str = None):
        """Leave a room."""
        asyncio.run_coroutine_threadsafe(
            self._client.leave_room(room_id),
            self._loop
        ).result(timeout=5)
    
    def update(self, path: str, value: Any, room_id: str = None):
        """Update state."""
        asyncio.run_coroutine_threadsafe(
            self._client.update(path, value, room_id),
            self._loop
        )
    
    def broadcast(self, payload: Dict[str, Any], room_id: str = None):
        """Broadcast message."""
        asyncio.run_coroutine_threadsafe(
            self._client.broadcast(payload, room_id),
            self._loop
        )
    
    @property
    def client_id(self) -> Optional[str]:
        return self._client.client_id
    
    @property
    def current_room(self) -> Optional[str]:
        return self._client.current_room


# ============================================================================
# Q-Lang Integration
# ============================================================================

def execute_collab_command(interpreter, command: str) -> str:
    """Execute collaboration commands from Q-Lang."""
    parts = command.strip().split()
    if not parts:
        return "Usage: collab <command> [args]"
    
    cmd = parts[0].lower()
    args = parts[1:]
    
    # Get or create client
    if not hasattr(interpreter, "_collab_client"):
        interpreter._collab_client = None
    
    client = interpreter._collab_client
    
    if cmd == "status":
        if client and client._connected:
            return f"""
Collaboration Status
====================
Connected: Yes
Server: {client.server_url}
Client ID: {client.client_id}
Name: {client.name}
Current Room: {client.current_room}
"""
        else:
            return "Not connected. Use 'collab connect <url>' to connect."
    
    elif cmd == "connect":
        url = args[0] if args else "ws://localhost:8765"
        name = args[1] if len(args) > 1 else None
        
        # Use sync client for Q-Lang
        interpreter._collab_client = SyncCollaborationClient(url, name)
        if interpreter._collab_client.connect():
            return f"Connected to {url}"
        else:
            return f"Failed to connect to {url}"
    
    elif cmd == "disconnect":
        if client:
            client.disconnect()
            interpreter._collab_client = None
        return "Disconnected"
    
    elif cmd == "join" and args:
        if not client:
            return "Not connected"
        room_id = args[0]
        client.join_room(room_id)
        return f"Joined room: {room_id}"
    
    elif cmd == "leave":
        if not client:
            return "Not connected"
        room_id = args[0] if args else None
        client.leave_room(room_id)
        return "Left room"
    
    elif cmd == "broadcast" and args:
        if not client:
            return "Not connected"
        message = " ".join(args)
        client.broadcast({"message": message})
        return f"Broadcast: {message}"
    
    elif cmd == "update" and len(args) >= 2:
        if not client:
            return "Not connected"
        path = args[0]
        value = " ".join(args[1:])
        client.update(path, value)
        return f"Updated {path} = {value}"
    
    elif cmd == "help":
        return """
Collaboration Commands
======================
collab status                  - Show connection status
collab connect <url> [name]    - Connect to server
collab disconnect              - Disconnect from server
collab join <room_id>          - Join a collaboration room
collab leave [room_id]         - Leave current/specified room
collab broadcast <message>     - Broadcast to room
collab update <path> <value>   - Update shared state
collab help                    - Show this help
"""
    
    else:
        return f"Unknown command: {cmd}. Use 'collab help' for options."


# ============================================================================
# Demo / Main
# ============================================================================

async def demo_server():
    """Run demo server."""
    server = CollaborationServer(port=8765)
    
    server.events.on("join", lambda cid, rid, p: 
        print(f"[Event] {p.name} joined {rid}"))
    server.events.on("disconnect", lambda cid: 
        print(f"[Event] Client {cid} disconnected"))
    
    await server.start()


async def demo_client(name: str, room: str):
    """Run demo client."""
    client = CollaborationClient(name=name)
    
    def on_broadcast(msg):
        print(f"[{name}] Received: {msg.payload}")
    
    client.on_message(MessageType.BROADCAST, on_broadcast)
    
    if await client.connect():
        await client.join_room(room)
        
        for i in range(3):
            await asyncio.sleep(2)
            await client.broadcast({"from": name, "count": i})
        
        await asyncio.sleep(5)
        await client.disconnect()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "server":
            print("Starting collaboration server...")
            asyncio.run(demo_server())
        elif sys.argv[1] == "client":
            name = sys.argv[2] if len(sys.argv) > 2 else "TestClient"
            room = sys.argv[3] if len(sys.argv) > 3 else "test_room"
            asyncio.run(demo_client(name, room))
    else:
        print("""
QENEX Collaboration Protocol Demo
=================================

Usage:
    python collaboration.py server              # Start server
    python collaboration.py client [name] [room]  # Start client

Or import and use programmatically:
    from collaboration import CollaborationServer, CollaborationClient
""")
