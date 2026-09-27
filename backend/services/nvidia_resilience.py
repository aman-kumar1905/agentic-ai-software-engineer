"""
Process-local NVIDIA capacity resilience.

NVIDIA's hosted endpoint is a shared worker pool that can return
"503 ResourceExhausted: Worker local total request limit reached" under
contention that has nothing to do with this application's own request rate
(confirmed via direct raw-HTTP investigation, 2026-09). This module bounds
how hard a single process leans on that endpoint during an outage: a
concurrency guard caps simultaneous in-flight NVIDIA requests, and a circuit
breaker fails fast during a sustained outage instead of continuing to retry
into it.

All state here is deliberately process-local, in-memory, module-level - it
must never be persisted into LangGraph checkpoints or shared across
processes/replicas. A distributed version of this would need a shared store
(e.g. Redis); that is out of scope here by design (single-process deployment
today), not an oversight.
"""

import random
import threading
import time

from backend.core.config import (
    NVIDIA_MAX_CONCURRENT_REQUESTS,
    NVIDIA_CIRCUIT_FAILURE_THRESHOLD,
    NVIDIA_CIRCUIT_COOLDOWN_SECONDS,
)

nvidia_semaphore = threading.Semaphore(NVIDIA_MAX_CONCURRENT_REQUESTS)

# Module-level alias (not a direct `time.sleep` reference at call sites) so
# tests can monkeypatch backend.services.nvidia_resilience.sleep in
# isolation, without touching the global `time` module and risking
# unrelated tests elsewhere that rely on real elapsed time.
sleep = time.sleep


def compute_backoff_delay(attempt: int) -> float:
    """
    Bounded exponential backoff with jitter: attempt 0 -> ~2s, 1 -> ~4s,
    2 -> ~8s, each plus up to 25% jitter (never negative, never unbounded).
    """
    base = 2.0 * (2**attempt)
    return base + random.uniform(0, base * 0.25)


class NvidiaCircuitBreaker:
    """
    Simple process-local circuit breaker over NVIDIA transient-capacity
    failures. CLOSED -> OPEN after `failure_threshold` consecutive
    failures; OPEN fails fast until `cooldown_seconds` elapses, then admits
    exactly one HALF_OPEN probe request. A successful call from any state
    resets to CLOSED; a failed probe re-opens (restarting the cooldown).
    """

    def __init__(self, failure_threshold: int, cooldown_seconds: float):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._consecutive_failures = 0
        self._opened_at = None
        self._probe_in_flight = False

    def _state_locked(self) -> str:
        if self._opened_at is None:
            return "CLOSED"
        if time.time() - self._opened_at >= self.cooldown_seconds:
            return "HALF_OPEN"
        return "OPEN"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state_locked()

    def allow_request(self) -> bool:
        """
        Returns True if a request may proceed now. While HALF_OPEN, admits
        exactly one probe at a time - concurrent callers during the same
        window are rejected until that probe resolves (success or failure).
        """
        with self._lock:
            state = self._state_locked()
            if state == "CLOSED":
                return True
            if state == "OPEN":
                return False
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    def record_success(self) -> bool:
        """Resets failure state. Returns True if this represents a recovery
        (the circuit had failures or was open) worth logging."""
        with self._lock:
            was_unhealthy = self._consecutive_failures > 0 or self._opened_at is not None
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False
            return was_unhealthy

    def record_failure(self) -> bool:
        """Records a failure. Returns True if this failure just opened (or
        re-opened, via a failed recovery probe) the circuit."""
        with self._lock:
            was_probe = self._probe_in_flight
            self._probe_in_flight = False
            self._consecutive_failures += 1
            just_opened = was_probe or self._consecutive_failures >= self.failure_threshold
            if just_opened:
                self._opened_at = time.time()
            return just_opened

    def reset(self) -> None:
        """Test-only: restores a fresh CLOSED state with no failure history."""
        with self._lock:
            self._consecutive_failures = 0
            self._opened_at = None
            self._probe_in_flight = False


nvidia_circuit = NvidiaCircuitBreaker(
    failure_threshold=NVIDIA_CIRCUIT_FAILURE_THRESHOLD,
    cooldown_seconds=NVIDIA_CIRCUIT_COOLDOWN_SECONDS,
)
