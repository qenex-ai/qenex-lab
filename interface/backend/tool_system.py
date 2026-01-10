#!/usr/bin/env python3
"""
QENEX LAB Tool System
DeepSeek uses Scout 17B as a specialized physics/theory consultant
"""

import httpx
import json
from typing import AsyncIterator, Dict, Any


class ToolSystem:
    """Manages tool calls from DeepSeek to Scout 17B"""

    OLLAMA_URL = "http://localhost:11434/v1"

    # Tool definitions for DeepSeek
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "consult_scout_physics",
                "description": "Consult Scout 17B (Llama-based physics expert) for complex physics, quantum mechanics, relativity, thermodynamics, or theoretical questions. Use this when the query requires deep physics knowledge or validation against fundamental laws.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "physics_query": {
                            "type": "string",
                            "description": "The physics or theoretical question to ask Scout 17B"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional context about what you've already told the user"
                        }
                    },
                    "required": ["physics_query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_discoveries",
                "description": "Search QENEX LAB's indexed knowledge base of 82 research documents, discoveries, and papers. Use this to find relevant historical research or documented findings.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "What to search for in the knowledge base"
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of documents to retrieve (default: 3)",
                            "default": 3
                        }
                    },
                    "required": ["search_query"]
                }
            }
        }
    ]

    def __init__(self, context_bridge, multi_expert):
        self.context_bridge = context_bridge
        self.multi_expert = multi_expert
        print("[Tool System] Initialized with Scout 17B and Discovery Search tools")

    async def consult_scout_physics(self, physics_query: str, context: str = "") -> str:
        """Call Scout 17B for physics expertise"""
        print(f"[Tool System] 🔬 DeepSeek calling Scout 17B: '{physics_query[:60]}...'")

        # Build prompt for Scout
        scout_prompt = f"""You are Scout 17B, a physics and theoretical expert with deep knowledge of quantum mechanics, relativity, thermodynamics, and the Unified Lagrangian.

Physics Query: {physics_query}
"""
        if context:
            scout_prompt += f"\nContext from conversation: {context}\n"

        scout_prompt += "\nProvide a rigorous, physics-grounded response. Cite fundamental principles and conservation laws."

        # Call Scout 17B
        full_response = ""
        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.OLLAMA_URL}/chat/completions",
                json={
                    "model": "scout-17b:local",
                    "messages": [{"role": "user", "content": scout_prompt}],
                    "stream": True,
                    "temperature": 0.3  # Lower temperature for physics accuracy
                },
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data and data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        full_response += content
                            except json.JSONDecodeError:
                                pass

        print(f"[Tool System] ✓ Scout 17B response: {len(full_response)} chars")
        return full_response

    async def search_discoveries(self, search_query: str, top_k: int = 3) -> str:
        """Search indexed knowledge base"""
        print(f"[Tool System] 📚 DeepSeek searching discoveries: '{search_query[:60]}...'")

        # Use context bridge
        context = await self.context_bridge.gather_context(search_query, top_k=top_k)

        # Format results
        results = []
        for doc in context['discovery_files']:
            results.append(f"**{doc['name']}** (relevance: {doc['relevance_score']:.3f})")
            # Get content preview
            content_preview = doc.get('content_preview', doc.get('content', ''))[:500]
            results.append(f"Content: {content_preview}\n")

        discoveries_text = "\n".join(results)
        print(f"[Tool System] ✓ Found {len(context['discovery_files'])} relevant documents")
        return f"Found {len(context['discovery_files'])} relevant documents:\n\n{discoveries_text}"

    async def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """Execute a tool call from DeepSeek"""
        if tool_name == "consult_scout_physics":
            return await self.consult_scout_physics(
                physics_query=arguments.get("physics_query", ""),
                context=arguments.get("context", "")
            )
        elif tool_name == "search_discoveries":
            return await self.search_discoveries(
                search_query=arguments.get("search_query", ""),
                top_k=arguments.get("top_k", 3)
            )
        else:
            return f"Error: Unknown tool '{tool_name}'"

    async def stream_with_tools(self, user_query: str) -> AsyncIterator[Dict]:
        """
        Stream response from DeepSeek with tool-calling capability

        DeepSeek can call Scout 17B or search discoveries when needed
        """
        print(f"[Tool System] Processing query with tools: '{user_query[:60]}...'")
        print(f"[Tool System] OLLAMA_URL: {self.OLLAMA_URL}")

        messages = [
            {
                "role": "system",
                "content": """You are DeepSeek Coder, integrated with QENEX LAB's scientific intelligence system.

You have access to specialized tools:
1. **consult_scout_physics**: Call Scout 17B (Llama-based physics expert) for complex physics, quantum mechanics, relativity, or theoretical questions
2. **search_discoveries**: Search 82 indexed research documents and discoveries

When users ask about:
- Physics, quantum mechanics, relativity, thermodynamics → USE consult_scout_physics
- Research history, documented findings, papers → USE search_discoveries
- Simple coding, math, general questions → Answer directly

Be conversational and helpful. When you use tools, explain what you're doing."""
            },
            {
                "role": "user",
                "content": user_query
            }
        ]

        # Initial call to DeepSeek
        tool_calls_made = []
        max_iterations = 3  # Prevent infinite loops

        for iteration in range(max_iterations):
            print(f"[Tool System] Iteration {iteration + 1}: Calling DeepSeek...")

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.OLLAMA_URL}/chat/completions",
                    json={
                        "model": "deepseek-coder-tools:local",
                        "messages": messages,
                        "tools": self.TOOLS,
                        "stream": True,
                        "temperature": 0.7
                    },
                ) as response:
                    current_tool_call = None
                    assistant_message = {"role": "assistant", "content": ""}

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if data and data != "[DONE]":
                                try:
                                    chunk = json.loads(data)
                                    if "choices" in chunk and len(chunk["choices"]) > 0:
                                        choice = chunk["choices"][0]
                                        delta = choice.get("delta", {})

                                        # Handle content (text response)
                                        content = delta.get("content")
                                        if content:
                                            assistant_message["content"] += content
                                            yield {
                                                "type": "content",
                                                "content": content
                                            }

                                        # Handle tool calls
                                        tool_calls = delta.get("tool_calls")
                                        if tool_calls:
                                            for tool_call in tool_calls:
                                                if tool_call.get("function"):
                                                    current_tool_call = tool_call
                                                    print(f"[Tool System] 🔧 DeepSeek requesting tool: {tool_call['function']['name']}")

                                        # Check if response is complete
                                        finish_reason = choice.get("finish_reason")
                                        if finish_reason == "tool_calls" and current_tool_call:
                                            # DeepSeek wants to call a tool
                                            tool_name = current_tool_call["function"]["name"]
                                            tool_args = json.loads(current_tool_call["function"]["arguments"])

                                            yield {
                                                "type": "tool_call",
                                                "tool": tool_name,
                                                "arguments": tool_args
                                            }

                                            # Execute tool
                                            tool_result = await self.execute_tool(tool_name, tool_args)

                                            yield {
                                                "type": "tool_result",
                                                "tool": tool_name,
                                                "result": tool_result[:500] + "..." if len(tool_result) > 500 else tool_result
                                            }

                                            # Add tool call and result to messages
                                            messages.append({
                                                "role": "assistant",
                                                "tool_calls": [current_tool_call]
                                            })
                                            messages.append({
                                                "role": "tool",
                                                "tool_call_id": current_tool_call.get("id", "tool_0"),
                                                "name": tool_name,
                                                "content": tool_result
                                            })

                                            tool_calls_made.append(tool_name)
                                            break  # Exit stream, continue loop

                                        elif finish_reason in ["stop", None]:
                                            # Normal completion, add to messages and exit
                                            if assistant_message["content"]:
                                                messages.append(assistant_message)
                                            return

                                except json.JSONDecodeError:
                                    pass

            # If we made a tool call, continue loop for DeepSeek to process result
            if not tool_calls_made or len(tool_calls_made) >= max_iterations:
                break

        print(f"[Tool System] ✓ Completed with {len(tool_calls_made)} tool calls")
