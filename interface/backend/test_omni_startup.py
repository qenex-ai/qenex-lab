#!/usr/bin/env python3
"""Test OMNI_INTEGRATION startup and initialization"""

import sys
print("[Test] Testing OMNI_INTEGRATION v1.4.0-INFINITY startup...")

# Test imports
print("[Test] Testing imports...")
from context_bridge import GlobalContextBridge
from multi_expert_search import MultiExpertSearch
from reasoning_engine import ReasoningEngine
from trinity_router import TrinityRouter
print("[Test] ✓ All modules imported successfully")

# Initialize components
print("\n[Test] ========================================")
print("[Test] INITIALIZING OMNI_INTEGRATION v1.4.0-INFINITY")
print("[Test] ========================================\n")

print("[Test] [1/4] Initializing Trinity Router...")
trinity_router = TrinityRouter()
print("[Test] ✓ Trinity Router ready")

print("\n[Test] [2/4] Initializing Global Context Bridge...")
context_bridge = GlobalContextBridge()
print("[Test] ✓ Global Context Bridge ready")
print(f"[Test] Stats: {context_bridge.get_stats()}")

print("\n[Test] [3/4] Initializing Multi-Expert Search...")
multi_expert = MultiExpertSearch()
print("[Test] ✓ Multi-Expert Search ready")
print(f"[Test] Stats: {multi_expert.get_stats()}")

print("\n[Test] [4/4] Initializing Reasoning Engine...")
reasoning_engine = ReasoningEngine(context_bridge, multi_expert, trinity_router)
print("[Test] ✓ Reasoning Engine ready")

print("\n[Test] ========================================")
print("[Test] OMNI_INTEGRATION INITIALIZATION COMPLETE")
print("[Test] System is now OMNI-AWARE")
print("[Test] ========================================")

# Test context gathering
print("\n[Test] Testing context gathering for sample query...")
import asyncio

async def test_context():
    query = "What is quantum gravity unification?"
    context = await context_bridge.gather_context(query, top_k=3)
    print(f"[Test] ✓ Found {len(context['discovery_files'])} relevant documents")
    for i, doc in enumerate(context['discovery_files'][:3], 1):
        print(f"[Test]   {i}. {doc['name']} (relevance: {doc['relevance_score']:.3f})")

asyncio.run(test_context())

print("\n[Test] ✅ ALL TESTS PASSED - OMNI_INTEGRATION IS OPERATIONAL")
