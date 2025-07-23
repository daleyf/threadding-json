"""
common.py

Shared data models and lightweight helpers for distributor & analyzer.
"""
from __future__ import annotations
import asyncio
from typing import Dict, Any, List

from pydantic import BaseModel


# ---------- Pydantic models ---------- #
class LogMessage(BaseModel):
    ts: str               # ISO-8601 timestamp
    level: str            # INFO / WARN / ERROR …
    category: str         # filename / tag
    payload: Dict[str, Any]


class Packet(BaseModel):
    agent_id: str
    messages: List[LogMessage]


# ---------- Tiny async utilities ---------- #
class InMemQueue:
    """Async FIFO wrapper."""
    def __init__(self) -> None:
        self._q: asyncio.Queue[Packet] = asyncio.Queue(maxsize=10_000)

    async def put(self, item: Packet) -> None:
        await self._q.put(item)

    async def get(self) -> Packet:
        return await self._q.get()


class RoundRobinRouter:
    """Lock-free async round-robin router."""
    def __init__(self, targets: List[str]) -> None:
        self._targets = targets
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str:
        async with self._lock:
            tgt = self._targets[self._idx]
            self._idx = (self._idx + 1) % len(self._targets)
            return tgt
