"""Token-bucket rate limiter, one bucket per API key.

Each key gets `rpm` tokens per minute, refilled continuously, with bursts up to `rpm`.
In-memory, so it is per-instance; see README roadmap for a Redis-backed version.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Bucket:
    capacity: float
    tokens: float
    refill_per_sec: float
    updated: float


class RateLimiter:
    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._buckets: dict[str, Bucket] = {}

    def check(self, key: str, rpm: int) -> tuple[bool, float]:
        """Consume one token. Returns (allowed, retry_after_seconds)."""
        now = self._clock()
        b = self._buckets.get(key)
        if b is None or b.capacity != rpm:
            b = self._buckets[key] = Bucket(rpm, rpm, rpm / 60.0, now)
        b.tokens = min(b.capacity, b.tokens + (now - b.updated) * b.refill_per_sec)
        b.updated = now
        if b.tokens >= 1:
            b.tokens -= 1
            return True, 0.0
        return False, (1 - b.tokens) / b.refill_per_sec

    def remaining(self, key: str) -> int:
        b = self._buckets.get(key)
        return int(b.tokens) if b else -1
