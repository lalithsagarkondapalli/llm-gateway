"""Per-model circuit breaker.

CLOSED    -> requests flow normally; consecutive failures are counted.
OPEN      -> after `failure_threshold` consecutive failures; requests skip this model.
HALF_OPEN -> after `cooldown_seconds`; one trial request is let through.
             success -> CLOSED, failure -> OPEN again.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from enum import StrEnum


class State(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 15.0,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._state = State.CLOSED
        self._failures = 0
        self._opened_at = 0.0
        self._trial_in_flight = False

    @property
    def state(self) -> State:
        if self._state is State.OPEN and self._clock() - self._opened_at >= self.cooldown_seconds:
            self._state = State.HALF_OPEN
            self._trial_in_flight = False
        return self._state

    def allow_request(self) -> bool:
        state = self.state
        if state is State.CLOSED:
            return True
        if state is State.HALF_OPEN and not self._trial_in_flight:
            self._trial_in_flight = True
            return True
        return False

    def record_success(self) -> None:
        self._state = State.CLOSED
        self._failures = 0
        self._trial_in_flight = False

    def record_failure(self) -> None:
        self._failures += 1
        if self._state is State.HALF_OPEN or self._failures >= self.failure_threshold:
            self._state = State.OPEN
            self._opened_at = self._clock()
            self._trial_in_flight = False
