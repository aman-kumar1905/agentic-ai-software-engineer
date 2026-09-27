"""
Tests for backend/services/nvidia_resilience.py and its integration into
backend/observability/telemetry.py: bounded exponential-backoff retry for
NVIDIA transient (503/ResourceExhausted) capacity failures, a process-local
concurrency guard, and a circuit breaker that fails fast during a sustained
outage instead of continuing to retry into it.

All tests here are fully mocked/deterministic (no real network calls) - the
real, live NVIDIA + Groq smoke tests are run manually as one-off scripts,
matching this codebase's existing convention for provider investigations
(see the scratchpad scripts referenced in git history), not committed here.
"""
import threading
import time

import httpx
import pytest

from backend.core.config import NVIDIA_MAX_RETRIES, NVIDIA_MAX_CONCURRENT_REQUESTS
from backend.observability.telemetry import invoke_structured, _invoke_nvidia_with_resilience
from backend.services.errors import LLMTransientError, LLMTimeoutError
from backend.services.nvidia_resilience import NvidiaCircuitBreaker, compute_backoff_delay
from backend.schemas.routing import RoutingDecision, TaskType


class FakeFlakyNvidiaLLM:
    """NVIDIA-shaped fake that raises a transient 503 for the first
    `fail_times` invocations, then returns `result` on every call after."""

    def __init__(self, fail_times, result=None, provider="nvidia", model_name="test-model"):
        self._provider = provider
        self.provider = provider
        self.model = model_name
        self.model_name = model_name
        self.fail_times = fail_times
        self.result = result
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        self.invocations += 1
        if self.invocations <= self.fail_times:
            raise httpx.NetworkError("503 Service Unavailable")
        return self.result


# --- Backoff schedule --------------------------------------------------------

def test_compute_backoff_delay_schedule():
    """attempt 0/1/2 -> approximately 2s/4s/8s, each with up to 25% jitter,
    never negative, never unbounded."""
    for attempt, base in enumerate((2.0, 4.0, 8.0)):
        for _ in range(20):
            delay = compute_backoff_delay(attempt)
            assert base <= delay <= base * 1.25


# --- Bounded retry on transient failures -------------------------------------

def test_nvidia_transient_failure_retries_then_succeeds():
    """A 503 that clears within the bounded retry budget must succeed
    without ever propagating an error to the caller."""
    llm = FakeFlakyNvidiaLLM(
        fail_times=2,
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Recovered after retry",
        ),
    )

    result = _invoke_nvidia_with_resilience(llm, RoutingDecision, "Prompt")

    assert result.task_type == TaskType.BUG_FIX
    assert llm.invocations == 3  # 2 failures + 1 success, all within budget


def test_nvidia_repeated_failure_retry_bounded_no_infinite_loop():
    """A 503 that never clears must stop at exactly NVIDIA_MAX_RETRIES
    retries (NVIDIA_MAX_RETRIES + 1 total attempts) - never loop forever."""
    llm = FakeFlakyNvidiaLLM(fail_times=10_000)

    with pytest.raises(LLMTransientError):
        _invoke_nvidia_with_resilience(llm, RoutingDecision, "Prompt")

    assert llm.invocations == NVIDIA_MAX_RETRIES + 1


def test_nvidia_timeout_is_not_retried_by_capacity_backoff():
    """A timeout is a distinct classification from a transient 503 - the
    NVIDIA-specific capacity backoff must not retry it (existing timeout
    handling/telemetry is unaffected and untouched by this feature)."""
    llm = FakeFlakyNvidiaLLM(fail_times=10_000)
    llm.invoke = lambda prompt: (_ for _ in ()).throw(TimeoutError("Request timed out after 30.0 seconds"))

    with pytest.raises(LLMTimeoutError):
        _invoke_nvidia_with_resilience(llm, RoutingDecision, "Prompt")


def test_nvidia_success_does_not_invoke_fallback(monkeypatch):
    """A normal NVIDIA success must never call get_llm() for a fallback
    provider at all."""
    llm = FakeFlakyNvidiaLLM(
        fail_times=0,
        result=RoutingDecision(
            task_type=TaskType.GENERAL,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="ok",
        ),
    )

    def unexpected_get_llm(provider=None, **kw):
        raise AssertionError(f"get_llm() must not be called on NVIDIA success (provider={provider!r})")

    monkeypatch.setattr("backend.services.llm.get_llm", unexpected_get_llm)

    result = invoke_structured(llm, RoutingDecision, "Prompt")
    assert result.task_type == TaskType.GENERAL
    assert llm.invocations == 1


# --- Circuit breaker ----------------------------------------------------------

def test_circuit_breaker_opens_after_threshold_and_fails_fast():
    cb = NvidiaCircuitBreaker(failure_threshold=2, cooldown_seconds=10.0)

    assert cb.state == "CLOSED"
    assert cb.allow_request() is True

    assert cb.record_failure() is False  # 1st failure: below threshold
    assert cb.state == "CLOSED"

    assert cb.record_failure() is True  # 2nd failure: threshold hit, opens
    assert cb.state == "OPEN"
    assert cb.allow_request() is False  # fails fast while open


def test_circuit_breaker_half_open_probe_then_recovers():
    cb = NvidiaCircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

    cb.record_failure()
    assert cb.state == "OPEN"
    assert cb.allow_request() is False

    time.sleep(0.06)
    assert cb.state == "HALF_OPEN"
    assert cb.allow_request() is True  # admits exactly one probe
    assert cb.allow_request() is False  # concurrent caller rejected mid-probe

    assert cb.record_success() is True  # recovery - was unhealthy
    assert cb.state == "CLOSED"
    assert cb.allow_request() is True


def test_circuit_breaker_failed_probe_reopens_and_restarts_cooldown():
    cb = NvidiaCircuitBreaker(failure_threshold=1, cooldown_seconds=0.05)

    cb.record_failure()
    time.sleep(0.06)
    assert cb.allow_request() is True  # probe admitted

    assert cb.record_failure() is True  # probe itself failed -> re-opens
    assert cb.state == "OPEN"
    assert cb.allow_request() is False


def test_circuit_breaker_open_causes_fail_fast_without_calling_provider(monkeypatch):
    """When the circuit is open, _invoke_nvidia_with_resilience must raise
    immediately without ever calling the underlying provider client."""
    from backend.services import nvidia_resilience

    forced_open = NvidiaCircuitBreaker(failure_threshold=1, cooldown_seconds=999.0)
    forced_open.record_failure()
    monkeypatch.setattr(nvidia_resilience, "nvidia_circuit", forced_open)

    llm = FakeFlakyNvidiaLLM(fail_times=0, result="should never be reached")

    with pytest.raises(LLMTransientError, match="circuit breaker open"):
        _invoke_nvidia_with_resilience(llm, RoutingDecision, "Prompt")

    assert llm.invocations == 0


# --- Concurrency guard --------------------------------------------------------

def test_nvidia_concurrency_guard_bounds_simultaneous_requests():
    """No more than NVIDIA_MAX_CONCURRENT_REQUESTS calls may be executing
    the underlying provider invocation at the same time."""
    active = 0
    peak = 0
    lock = threading.Lock()

    class ConcurrencyProbeLLM:
        _provider = "nvidia"
        provider = "nvidia"
        model = "test-model"
        model_name = "test-model"

        def with_structured_output(self, schema, include_raw=False):
            return self

        def invoke(self, prompt):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return RoutingDecision(
                task_type=TaskType.GENERAL,
                requires_planning=False,
                requires_knowledge=False,
                reasoning="ok",
            )

    threads = [
        threading.Thread(target=_invoke_nvidia_with_resilience, args=(ConcurrencyProbeLLM(), RoutingDecision, "Prompt"))
        for _ in range(NVIDIA_MAX_CONCURRENT_REQUESTS + 3)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert peak <= NVIDIA_MAX_CONCURRENT_REQUESTS
