#!/usr/bin/env python3
"""
QENEX LAB Chat Backend
FastAPI server with WebSocket and SSE support for Trinity Pipeline
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import asyncio
import json
from trinity_router import TrinityRouter
from expert_system import ExpertSystem
from pdf_generator import generate_qenex_paper
from semantic_cache import SemanticCache
from context_bridge import GlobalContextBridge
from multi_expert_search import MultiExpertSearch
from reasoning_engine import ReasoningEngine
from tool_system_simple import SimpleToolSystem


app = FastAPI(
    title="QENEX LAB Chat API",
    version="3.0-INFINITY",
    description="Advanced Scientific Computing Laboratory - Chat Interface"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Trinity components
trinity_router = TrinityRouter()
expert_system = ExpertSystem()

# Initialize Semantic Cache (with embedding model)
print("[Backend] Initializing semantic cache...")
cache = SemanticCache()
print("[Backend] ✓ Semantic cache ready")

# Initialize OMNI_INTEGRATION components
print("[Backend] ========================================")
print("[Backend] INITIALIZING OMNI_INTEGRATION v1.4.0-INFINITY")
print("[Backend] ========================================")

print("[Backend] [1/3] Initializing Global Context Bridge...")
context_bridge = GlobalContextBridge()
print("[Backend] ✓ Global Context Bridge ready")

print("[Backend] [2/3] Initializing Multi-Expert Search...")
multi_expert = MultiExpertSearch()
print("[Backend] ✓ Multi-Expert Search ready")

print("[Backend] [3/3] Initializing Reasoning Engine...")
reasoning_engine = ReasoningEngine(context_bridge, multi_expert, trinity_router)
print("[Backend] ✓ Reasoning Engine ready")

print("[Backend] [4/4] Initializing Simple Tool System...")
tool_system = SimpleToolSystem(multi_expert)
print("[Backend] ✓ Simple Tool System ready (DeepSeek → Scout 17B)")

print("[Backend] ========================================")
print("[Backend] OMNI_INTEGRATION INITIALIZATION COMPLETE")
print("[Backend] System is now OMNI-AWARE")
print("[Backend] ========================================")


class ChatMessage(BaseModel):
    content: str
    enable_validation: bool = False  # Optional Scout CLI validation (slower but more rigorous)


class PublishRequest(BaseModel):
    topic: str


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "operational",
        "version": "1.4.0-INFINITY (OMNI-AWARE)",
        "lab": "QENEX LAB",
        "trinity_pipeline": "active",
        "experts": "18/18",
        "cache": cache.stats(),
        "omni_integration": {
            "context_bridge": context_bridge.get_stats(),
            "multi_expert": multi_expert.get_stats(),
            "reasoning_engine": "active"
        }
    }


@app.get("/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return cache.stats()


@app.post("/cache/clear")
async def clear_cache():
    """Clear all caches"""
    cache.clear()
    return {"status": "cleared"}


@app.post("/chat/message")
async def chat_message(message: ChatMessage):
    """OMNI-AWARE Streaming chat endpoint with Trinity Blueprint and semantic caching"""
    print(f"[Chat API] Received message: {message.content[:50]}...")
    print(f"[Chat API] Validation enabled: {message.enable_validation}")

    # Check cache FIRST (cache key includes validation flag)
    cache_key = f"{message.content}:validation={message.enable_validation}"
    cached_response = cache.get(cache_key)

    if cached_response:
        print(f"[Chat API] 🚀 Cache hit! Returning cached OMNI-AWARE response")
        async def cached_event_generator():
            # Send context event (from cache)
            if "context" in cached_response:
                yield {
                    "event": "context",
                    "data": json.dumps(cached_response["context"])
                }

            # Send model event
            yield {
                "event": "model",
                "data": json.dumps({
                    "model": cached_response["model"],
                    "cached": True,
                    "omni_aware": True
                })
            }

            # Send message
            yield {
                "event": "message",
                "data": json.dumps({"content": cached_response["response"]})
            }

            # Send completion
            yield {
                "event": "done",
                "data": json.dumps({
                    "status": "complete",
                    "cached": True,
                    "omni_aware": True
                })
            }
        return EventSourceResponse(cached_event_generator())

    # Cache miss - continue with OMNI-AWARE inference
    async def event_generator():
        full_response = ""
        context_metadata = None

        try:
            # Stream response with Trinity Blueprint context
            async for event in reasoning_engine.stream_response_with_context(
                query=message.content,
                enable_validation=message.enable_validation
            ):
                # Capture context metadata for caching
                if event['event'] == 'context':
                    context_metadata = json.loads(event['data'])

                # Accumulate response content
                if event['event'] == 'message':
                    message_data = json.loads(event['data'])
                    full_response += message_data.get('content', '')

                # Yield event to client
                yield event

            # Cache the complete OMNI-AWARE response
            if full_response:
                # Determine model from context (or default)
                model = "unknown"
                if context_metadata:
                    # Extract model from earlier events (would need to track this)
                    model = "deepseek"  # Default for now

                cache.set(cache_key, {
                    "model": model,
                    "response": full_response,
                    "context": context_metadata,
                    "validation_enabled": message.enable_validation
                }, model)

                print(f"[Chat API] ✓ Cached OMNI-AWARE response")

        except Exception as e:
            print(f"[Chat API] Error: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(event_generator())


@app.post("/chat/simple")
async def chat_simple(message: ChatMessage):
    """
    Simple tool-calling architecture: User → DeepSeek → Scout 17B (when needed)

    DeepSeek acts as main conversational AI and calls Scout 17B as a specialized tool
    for physics/theory questions. User always gets an answer.
    """
    print(f"[Chat API Simple] Received: {message.content[:50]}...")

    async def tool_event_generator():
        try:
            async for event in tool_system.stream_response(message.content):
                event_type = event.get("type")

                if event_type == "content":
                    # DeepSeek's response content
                    yield {
                        "event": "message",
                        "data": json.dumps({"content": event["content"]})
                    }

                elif event_type == "tool_call":
                    # DeepSeek is calling a tool
                    yield {
                        "event": "tool_call",
                        "data": json.dumps({
                            "tool": event.get("tool", "unknown"),
                            "status": event.get("status", "calling")
                        })
                    }

                elif event_type == "tool_result":
                    # Tool execution result
                    yield {
                        "event": "tool_result",
                        "data": json.dumps({
                            "tool": event.get("tool", "unknown"),
                            "result": event.get("result", "")
                        })
                    }

            # Completion
            yield {
                "event": "done",
                "data": json.dumps({"status": "complete"})
            }

        except Exception as e:
            print(f"[Chat API Simple] Error: {e}")
            import traceback
            traceback.print_exc()
            yield {
                "event": "error",
                "data": json.dumps({"error": str(e)})
            }

    return EventSourceResponse(tool_event_generator())


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time expert status updates"""
    await websocket.accept()
    print("[WebSocket] Client connected")

    try:
        # Subscribe to expert status changes
        async for status in expert_system.subscribe():
            await websocket.send_json({
                "type": "expert_status",
                "experts": status
            })
    except WebSocketDisconnect:
        print("[WebSocket] Client disconnected")
    except Exception as e:
        print(f"[WebSocket] Error: {e}")


@app.post("/publish")
async def publish_paper(request: PublishRequest):
    """Generate PDF via /publish command"""
    print(f"[Publish API] Generating paper: {request.topic}")
    pdf_path = await generate_qenex_paper(request.topic)
    return {"pdf": pdf_path, "status": "generated"}


if __name__ == "__main__":
    import uvicorn
    print("Starting QENEX LAB Chat Backend...")
    uvicorn.run(app, host="0.0.0.0", port=8765)
