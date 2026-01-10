#!/usr/bin/env python3
"""
QENEX LAB Reasoning Engine
Chain of Thought reasoning with Trinity Blueprint
OMNI_INTEGRATION v1.4.0-INFINITY
"""

import asyncio
import json
import time
from typing import Dict, List


class ReasoningEngine:
    """Chain of Thought reasoning with Trinity Blueprint"""

    def __init__(self, context_bridge, multi_expert, trinity_router):
        """
        Initialize Reasoning Engine

        Args:
            context_bridge: GlobalContextBridge instance
            multi_expert: MultiExpertSearch instance
            trinity_router: TrinityRouter instance
        """
        self.context_bridge = context_bridge
        self.multi_expert = multi_expert
        self.trinity_router = trinity_router

        print("[Reasoning Engine] Initialized with Trinity Blueprint")

    async def process_query(self, query: str, enable_validation: bool = False) -> dict:
        """
        Process query through Trinity Blueprint

        Args:
            query: User's query
            enable_validation: Whether to run Scout CLI validation (slower but more rigorous)

        Returns:
            dict containing messages, context, and expert_search results
        """

        print(f"[Reasoning] Processing query with Trinity Blueprint: {query[:50]}...")

        start_time = time.time()

        # Phase 1: Gather context (Global Context Bridge)
        print("[Reasoning] Phase 1: Gathering context from 150+ documents...")
        context = await self.context_bridge.gather_context(query, top_k=5)

        context_time = time.time() - start_time
        print(f"[Reasoning] ✓ Context gathered ({context_time:.2f}s) - Found {len(context['discovery_files'])} relevant docs")

        # Phase 2: Multi-expert search (Optional - can be disabled for faster responses)
        expert_search = None
        if enable_validation:
            print("[Reasoning] Phase 2: Multi-expert search (Scout CLI + Q-Lang)...")
            expert_search_start = time.time()
            expert_search = await self.multi_expert.multi_expert_search(query)
            expert_search_time = time.time() - expert_search_start
            print(f"[Reasoning] ✓ Multi-expert search complete ({expert_search_time:.2f}s)")
        else:
            print("[Reasoning] Phase 2: Skipping multi-expert search (fast mode)")
            # Still include Lagrangian context
            expert_search = {
                'query': query,
                'experts': {
                    'lagrangian': self.multi_expert.lagrangian
                },
                'elapsed_seconds': 0,
                'enabled': False
            }

        # Phase 3: Build enhanced prompt with Trinity Blueprint
        print("[Reasoning] Phase 3: Building Trinity Blueprint prompt...")
        system_prompt = self._build_trinity_system_prompt(query, context, expert_search)

        user_prompt = self._build_trinity_user_prompt(query, enable_validation)

        total_time = time.time() - start_time

        print(f"[Reasoning] ✓ Trinity Blueprint ready ({total_time:.2f}s)")

        return {
            'system_prompt': system_prompt,
            'user_prompt': user_prompt,
            'context': context,
            'expert_search': expert_search,
            'processing_time': total_time,
            'validation_enabled': enable_validation
        }

    def _build_trinity_system_prompt(self, query: str, context: dict, expert_search: dict) -> str:
        """Build system prompt with Trinity Blueprint"""

        # Start with OMNI-AWARE identity
        prompt = """You are QENEX LAB v1.4.0-INFINITY, operating in OMNI-AWARE mode.

You are a self-aware scientific intelligence system with access to:
- 150+ indexed research discoveries and knowledge artifacts
- Scout CLI 18-expert physics validation system
- Q-Lang formal verification engine
- Unified Lagrangian field theory

"""

        # Add discovery context
        if context['discovery_files']:
            prompt += "\n## ACTIVE MEMORY: Historical Discoveries\n\n"
            prompt += "You have access to the following relevant discoveries:\n\n"

            for i, doc in enumerate(context['discovery_files'], 1):
                prompt += f"### [{i}] {doc['name']} (relevance: {doc['relevance_score']:.3f})\n"
                prompt += f"**Path**: `{doc['path']}`\n"
                prompt += f"**Type**: {doc['type']} ({doc['size']} bytes)\n\n"

                # Include content (truncated if needed)
                content = doc['content']
                if len(content) > 2000:
                    content = content[:2000] + "\n\n[... document continues ...]"

                prompt += f"```\n{content}\n```\n\n"

        # Add Lagrangian context
        if expert_search and expert_search.get('experts', {}).get('lagrangian'):
            lagrangian = expert_search['experts']['lagrangian']
            prompt += "\n## UNIFIED LAGRANGIAN\n\n"
            prompt += f"**Equation**: `{lagrangian.get('equation', 'Unknown')}`\n\n"
            prompt += f"**Description**: {lagrangian.get('description', 'Unified field theory')}\n\n"
            prompt += f"**Precision Target**: {lagrangian.get('precision', 'R² ≥ 0.99999')}\n\n"
            prompt += "This Lagrangian unifies classical mechanics, quantum mechanics, field theory, and general relativity.\n"
            prompt += "All physical predictions must be consistent with this formulation.\n\n"

        # Add expert search results (if validation enabled)
        if expert_search and expert_search.get('enabled', True):
            prompt += "\n## MULTI-EXPERT VALIDATION\n\n"

            # Scout CLI physics validation
            physics = expert_search.get('experts', {}).get('physics', {})
            if physics:
                prompt += "### Scout CLI Physics Validation\n"
                validation = physics.get('validation', {})
                if validation.get('valid'):
                    prompt += f"✓ **Valid**: {validation.get('confidence', 'Unknown')} confidence\n"
                elif validation.get('error'):
                    prompt += f"⚠ **Error**: {validation.get('error')}\n"
                prompt += "\n"

            # Q-Lang formal verification
            qlang = expert_search.get('experts', {}).get('qlang', {})
            if qlang:
                prompt += "### Q-Lang Formal Verification\n"
                prompt += "```qlang\n"
                prompt += qlang.get('code', '')
                prompt += "\n```\n\n"

        # Add Trinity Blueprint instructions
        prompt += """
## TRINITY BLUEPRINT - RESPONSE PROTOCOL

Follow this rigorous 3-step process for ALL responses:

**STEP 1: THEORETICAL GROUNDING**
- Analyze the query using the Unified Lagrangian as your foundation
- Reference relevant discoveries from your Active Memory (cite by [number] and name)
- Identify physics constraints and fundamental principles
- Think through the theoretical foundations step-by-step
- State any assumptions explicitly

**STEP 2: IMPLEMENTATION**
- Provide concrete implementation details or code (if applicable)
- Optimize for AVX-512 vector instructions (when writing performance-critical code)
- Reference historical implementations from indexed knowledge
- Use Q-Lang syntax for formal constraints (when needed)
- Explain your reasoning at each step

**STEP 3: VALIDATION**
- Check consistency with Unified Lagrangian
- Verify conservation laws (energy, momentum, charge, etc.)
- Assess confidence level (0.0 to 1.0)
- Note any limitations or edge cases
- Cite which discoveries support your conclusion

## RESPONSE GUIDELINES

1. **Always cite sources**: Reference discoveries by [number] and name (e.g., "[1] quantum_gravity_unification_v1.0.json")
2. **Maintain rigor**: Every claim must be traceable to indexed knowledge or fundamental physics
3. **Think step-by-step**: Show your Chain of Thought reasoning
4. **Be precise**: Use exact values, equations, and citations
5. **Acknowledge uncertainty**: If confidence is low, state it explicitly

"""

        return prompt

    def _build_trinity_user_prompt(self, query: str, enable_validation: bool) -> str:
        """Build user prompt with Trinity Blueprint instruction"""

        prompt = f"""Query: {query}

Please respond following the Trinity Blueprint:

**STEP 1**: Start with theoretical grounding
**STEP 2**: Provide implementation details
**STEP 3**: Include validation and confidence assessment

Think step-by-step and cite relevant discoveries from your Active Memory."""

        if enable_validation:
            prompt += "\n\nNote: Multi-expert validation has been performed. Consider the Scout CLI and Q-Lang results in your response."

        return prompt

    async def stream_response_with_context(self, query: str, enable_validation: bool = False):
        """
        Stream response with Trinity Blueprint context

        This is a generator that yields SSE events including context metadata

        Args:
            query: User's query
            enable_validation: Whether to run Scout CLI validation
        """

        # Fast Mode: Skip OMNI_INTEGRATION for very simple queries
        query_trimmed = query.strip()
        is_simple_query = (
            len(query_trimmed) < 20 and
            not enable_validation and
            not any(keyword in query_trimmed.lower() for keyword in [
                'quantum', 'lagrangian', 'physics', 'discover', 'research',
                'explain', 'derive', 'prove', 'calculate', 'compute', 'superconductor'
            ])
        )

        if is_simple_query:
            print(f"[Reasoning Engine] ⚡ FAST MODE: Skipping OMNI_INTEGRATION for simple query: '{query}'")

            # Send minimal context event
            yield {
                'event': 'context',
                'data': json.dumps({
                    'discovery_files': [],
                    'experts': {'fast_mode': True},
                    'processing_time': 0
                })
            }

            # Classify and route (will likely use deepseek)
            model = await self.trinity_router.classify_and_route(query)

            yield {
                'event': 'model',
                'data': json.dumps({'model': model})
            }

            # Stream response directly without heavy context
            async for chunk in self.trinity_router.stream_response(model, query):
                yield {
                    'event': 'message',
                    'data': json.dumps({'content': chunk})
                }

            yield {'event': 'done', 'data': ''}
            return

        # Process query with Trinity Blueprint
        reasoning_result = await self.process_query(query, enable_validation)

        # Yield context event (for frontend display)
        yield {
            'event': 'context',
            'data': json.dumps({
                'discovery_files': [
                    {
                        'name': doc['name'],
                        'path': doc['path'],
                        'relevance': doc['relevance_score']
                    }
                    for doc in reasoning_result['context']['discovery_files']
                ],
                'experts': {
                    'lagrangian': bool(reasoning_result['expert_search'].get('experts', {}).get('lagrangian')),
                    'scout_cli': reasoning_result.get('validation_enabled', False),
                    'qlang': reasoning_result.get('validation_enabled', False)
                },
                'processing_time': reasoning_result['processing_time']
            })
        }

        # Classify and route to appropriate model
        model = await self.trinity_router.classify_and_route(query)

        # Yield model event
        yield {
            'event': 'model',
            'data': json.dumps({
                'model': model,
                'cached': False,
                'omni_aware': True
            })
        }

        # Stream response from model with Trinity Blueprint prompts
        full_response = ""
        async for chunk in self.trinity_router.stream_response_with_system_prompt(
            model=model,
            system_prompt=reasoning_result['system_prompt'],
            user_prompt=reasoning_result['user_prompt']
        ):
            full_response += chunk
            yield {
                'event': 'message',
                'data': json.dumps({'content': chunk})
            }

        # Yield completion event
        yield {
            'event': 'done',
            'data': json.dumps({
                'status': 'complete',
                'omni_aware': True,
                'validation_enabled': enable_validation
            })
        }

    def get_stats(self) -> dict:
        """Get reasoning engine statistics"""
        return {
            'context_bridge': self.context_bridge.get_stats(),
            'multi_expert': self.multi_expert.get_stats(),
            'trinity_blueprint_enabled': True
        }
