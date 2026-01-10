#!/usr/bin/env python3
"""
QENEX LAB Simple Tool System - HTTP API Version
User → DeepSeek API → (optionally) Scout API

Uses HTTP calls to local services:
- DeepSeek: http://localhost:8080
- Scout: http://localhost:8085
"""

import json
import asyncio
import httpx
from typing import AsyncIterator, Dict


class SimpleToolSystem:
    """DeepSeek + Scout via HTTP APIs (hugepages-backed services)"""

    DEEPSEEK_URL = "http://localhost:8080"
    SCOUT_URL = "http://localhost:8085"

    def __init__(self, multi_expert):
        self.multi_expert = multi_expert
        self.scout_available = False
        self.deepseek_available = False

        # Check service availability
        print("[Simple Tool System] Checking local API services...")
        asyncio.get_event_loop().run_until_complete(self._check_services())

    async def _check_services(self):
        """Check which services are available"""
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check DeepSeek
            try:
                resp = await client.get(f"{self.DEEPSEEK_URL}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("model_loaded"):
                        self.deepseek_available = True
                        print(f"[Simple Tool System] ✓ DeepSeek API ready at {self.DEEPSEEK_URL}")
            except Exception as e:
                print(f"[Simple Tool System] ⚠ DeepSeek API not available: {e}")

            # Check Scout
            try:
                resp = await client.get(f"{self.SCOUT_URL}/health")
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("model_loaded"):
                        self.scout_available = True
                        print(f"[Simple Tool System] ✓ Scout API ready at {self.SCOUT_URL}")
            except Exception as e:
                print(f"[Simple Tool System] ⚠ Scout API not available: {e}")

        if not self.deepseek_available and not self.scout_available:
            print("[Simple Tool System] ⚠ No LLM services available!")

    async def call_scout(self, physics_query: str) -> str:
        """Call Scout Master Brain API for physics reasoning"""
        print(f"[Simple Tool] Calling Scout API: '{physics_query[:50]}...'")

        if not self.scout_available:
            return "Scout Master Brain not available"

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(
                    f"{self.SCOUT_URL}/v1/reason",
                    json={
                        "query": physics_query,
                        "max_tokens": 2048,
                        "temperature": 0.3
                    }
                )
                if resp.status_code == 200:
                    data = resp.json()
                    reasoning = data.get("reasoning", "")
                    confidence = data.get("confidence", 0)
                    experts = data.get("experts_consulted", [])

                    response = f"""Scout Master Brain Analysis:

{reasoning}

---
Confidence: {confidence:.2f}
Experts Consulted: {', '.join(experts)}"""
                    print(f"[Simple Tool] Scout response: {len(response)} chars")
                    return response
                else:
                    return f"Scout API error: {resp.status_code}"
        except Exception as e:
            print(f"[Simple Tool] Scout API error: {e}")
            return f"Scout API error: {str(e)}"

    async def call_deepseek(self, query: str, system_prompt: str = None, stream: bool = True) -> AsyncIterator[str]:
        """Call DeepSeek API for general queries"""
        if not self.deepseek_available:
            yield "DeepSeek API not available"
            return

        if system_prompt is None:
            system_prompt = """You are DeepSeek Coder in QENEX LAB, backed by Scout CLI (18-expert physics system).

Your capabilities:
- Programming & Software Development (your specialty)
- Physics & Science (via Scout CLI - you automatically consult it for complex physics)
- Mathematics, Algorithms, Data Structures
- General problem-solving and explanations

Be helpful, confident, and thorough. You have expert systems backing you!"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                if stream:
                    # Streaming response
                    async with client.stream(
                        "POST",
                        f"{self.DEEPSEEK_URL}/v1/chat/completions",
                        json={
                            "model": "deepseek-coder-6.7b",
                            "messages": messages,
                            "stream": True,
                            "max_tokens": 2048,
                            "temperature": 0.7
                        }
                    ) as resp:
                        async for line in resp.aiter_lines():
                            if line.startswith("data: "):
                                data_str = line[6:].strip()
                                if data_str and data_str != "[DONE]":
                                    try:
                                        chunk = json.loads(data_str)
                                        if "choices" in chunk:
                                            delta = chunk["choices"][0].get("delta", {})
                                            content = delta.get("content", "")
                                            if content:
                                                yield content
                                    except json.JSONDecodeError:
                                        pass
                else:
                    # Non-streaming response
                    resp = await client.post(
                        f"{self.DEEPSEEK_URL}/v1/chat/completions",
                        json={
                            "model": "deepseek-coder-6.7b",
                            "messages": messages,
                            "stream": False,
                            "max_tokens": 2048,
                            "temperature": 0.7
                        }
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        yield content
                    else:
                        yield f"DeepSeek API error: {resp.status_code}"
        except Exception as e:
            print(f"[Simple Tool] DeepSeek API error: {e}")
            yield f"DeepSeek API error: {str(e)}"

    async def stream_response(self, user_query: str) -> AsyncIterator[Dict]:
        """
        Stream response from DeepSeek API

        If DeepSeek determines it needs physics help, consult Scout first.
        """
        print(f"[Simple Tool] Processing: '{user_query[:60]}...'")

        # Check if this is a complex question that needs expert help
        needs_expert = any(keyword in user_query.lower() for keyword in [
            'quantum', 'physics', 'relativity', 'thermodynamics', 'entropy',
            'lagrangian', 'entanglement', 'superconductor', 'gravitational',
            'schwarzschild', 'einstein', 'derive', 'prove', 'metric', 'field equations',
            'explain', 'theory', 'principle', 'law', 'equation'
        ])

        # AUTOMATIC SCOUT CONSULTATION: If it's a complex question, call Scout FIRST
        if needs_expert and self.scout_available:
            print(f"[Simple Tool] Detected complex question - consulting Scout API first")

            # Notify user
            yield {
                "type": "tool_call",
                "tool": "scout",
                "status": "calling"
            }

            # Call Scout API
            scout_response = await self.call_scout(user_query)

            yield {
                "type": "tool_result",
                "tool": "scout",
                "result": f"Scout consulted ({len(scout_response)} chars)"
            }

            # Now have DeepSeek explain the concept using the Scout analysis
            print(f"[Simple Tool] DeepSeek synthesizing Scout's answer...")

            system_prompt = """You are a helpful AI assistant in QENEX LAB with broad knowledge.

CRITICAL INSTRUCTION:
- User asked a question
- Scout Master Brain (17B MoE physics system) has provided expert analysis
- You MUST synthesize this into a clear, educational answer
- Use examples, analogies, and clear explanations
- Be confident and thorough

NEVER say "I can't" or "beyond my expertise" - Scout has your back!"""

            message = f"""User asked: {user_query}

Scout Master Brain Analysis:
{scout_response}

Please synthesize this into a clear, thorough explanation with examples."""

            async for token in self.call_deepseek(message, system_prompt):
                yield {
                    "type": "content",
                    "content": token
                }

        else:
            # Simple question or Scout not available - let DeepSeek answer directly
            print(f"[Simple Tool] DeepSeek answering directly via API...")

            try:
                async for token in self.call_deepseek(user_query):
                    yield {
                        "type": "content",
                        "content": token
                    }
            except Exception as e:
                print(f"[Simple Tool] Error: {e}")
                import traceback
                traceback.print_exc()
                yield {
                    "type": "error",
                    "error": str(e)
                }
