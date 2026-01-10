#!/usr/bin/env python3
"""
QENEX LAB Expert System
18-expert validation system with real-time status tracking
OPTIMIZED: Async validation with process pool
"""

import asyncio
from typing import Dict, AsyncIterator
from concurrent.futures import ProcessPoolExecutor
import subprocess
import json


EXPERTS = [
    "Physics", "Math", "Quantum", "Relativity", "Cosmology", "Thermo",
    "E&M", "Nuclear", "Particle", "Astro", "Materials", "Compute",
    "Info", "Stats", "Algebra", "Geometry", "Topology", "Analysis"
]


def _run_scout_validation(expression: str) -> dict:
    """Worker function for process pool (must be picklable)"""
    try:
        result = subprocess.run(
            ["/opt/qenex/scout-cli/target/release/scout", "validate", expression],
            capture_output=True,
            text=True,
            timeout=5  # Reduced from 30s
        )
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        return {"confidence": 0.0, "valid": False, "error": "Timeout"}
    except FileNotFoundError:
        return {"confidence": 0.0, "valid": False, "error": "Scout CLI not found"}
    except Exception as e:
        return {"confidence": 0.0, "valid": False, "error": str(e)}


class ExpertSystem:
    def __init__(self):
        self.status: Dict[str, str] = {expert: "idle" for expert in EXPERTS}
        self._subscribers = []

        # Process pool with 4 workers for async validation
        self.pool = ProcessPoolExecutor(max_workers=4)
        print(f"[Expert System] Initialized with {len(EXPERTS)} experts + 4 Scout CLI workers")

    async def validate(self, expression: str) -> dict:
        """DEPRECATED: Use validate_async() instead"""
        return await self.validate_async(expression)

    async def validate_async(self, expression: str) -> dict:
        """Async validation using process pool (non-blocking)"""
        print(f"[Expert System] Starting async validation")

        # Mark experts as thinking
        for expert in EXPERTS:
            self.status[expert] = "thinking"
        await self._broadcast_status()

        # Run validation in process pool (non-blocking)
        loop = asyncio.get_event_loop()
        try:
            validation = await loop.run_in_executor(
                self.pool,
                _run_scout_validation,
                expression
            )
            print(f"[Expert System] Validation complete: {validation}")
        except Exception as e:
            print(f"[Expert System] Validation error: {e}")
            validation = {"confidence": 0.0, "valid": False, "error": str(e)}

        # Update status
        status_result = "validated" if validation.get("valid", False) else "error"
        for expert in EXPERTS:
            self.status[expert] = status_result
        await self._broadcast_status()

        # Reset to idle after 2 seconds
        await asyncio.sleep(2)
        for expert in EXPERTS:
            self.status[expert] = "idle"
        await self._broadcast_status()

        return validation

    def shutdown(self):
        """Cleanup process pool"""
        self.pool.shutdown(wait=True)
        print("[Expert System] Process pool shut down")

    async def subscribe(self) -> AsyncIterator[Dict[str, str]]:
        """Subscribe to expert status updates"""
        queue: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(queue)
        print(f"[Expert System] New subscriber (total: {len(self._subscribers)})")

        try:
            while True:
                status = await queue.get()
                yield status
        finally:
            self._subscribers.remove(queue)
            print(f"[Expert System] Subscriber removed (remaining: {len(self._subscribers)})")

    async def _broadcast_status(self):
        """Broadcast current status to all subscribers"""
        for queue in self._subscribers:
            try:
                await queue.put(self.status.copy())
            except Exception as e:
                print(f"[Expert System] Error broadcasting: {e}")
