"""
LLM token usage and cost telemetry.

Wraps structured-output LLM calls to capture prompt/completion token counts
and an approximate USD cost, without changing the calling agent functions'
signatures or return types (existing callers and test mocks are unaffected).
Usage from calls made inside a `collect_usage()` block is accumulated and
handed back to the caller to merge into the graph state's `metrics` dict.
"""

import contextvars
from contextlib import contextmanager
from typing import Any, Dict, Optional

ZERO_USAGE: Dict[str, Any] = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "estimated_cost_usd": 0.0,
    # False until a call actually reports usage. Some providers/paths (e.g.
    # ChatNVIDIA's with_structured_output, which doesn't support
    # include_raw) never expose token counts at all - that's "unmeasured",
    # not "zero cost", and callers should render it differently.
    "tracked": False,
}

# Approximate USD cost per 1K tokens, keyed by a substring match against the
# model identifier (case-insensitive). First match wins; extend as needed.
_PRICING_PER_1K = {
    "claude": {"prompt": 0.003, "completion": 0.015},
    "sonnet": {"prompt": 0.003, "completion": 0.015},
    "gemini": {"prompt": 0.000075, "completion": 0.0003},
    "gpt-4o": {"prompt": 0.0025, "completion": 0.01},
    "llama": {"prompt": 0.0002, "completion": 0.0002},
}
_FALLBACK_PRICING = {"prompt": 0.0005, "completion": 0.0015}

_current_usage: "contextvars.ContextVar[Optional[Dict[str, Any]]]" = contextvars.ContextVar(
    "_current_usage", default=None
)


def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Approximates USD cost for a call from a small per-model pricing table."""
    model_key = (model or "").lower()
    rates = _FALLBACK_PRICING
    for key, value in _PRICING_PER_1K.items():
        if key in model_key:
            rates = value
            break
    return (prompt_tokens / 1000) * rates["prompt"] + (completion_tokens / 1000) * rates["completion"]


def extract_usage(raw_message: Any, model: str) -> Dict[str, Any]:
    """
    Safely extracts token usage from a LangChain AIMessage: tries the
    standard `usage_metadata` field first, then the provider-specific
    `response_metadata['token_usage']` shape. Never raises; defaults to
    zeroed counts if the provider didn't report usage.
    """
    prompt_tokens = 0
    completion_tokens = 0
    tracked = False
    try:
        usage = getattr(raw_message, "usage_metadata", None)
        if usage:
            prompt_tokens = usage.get("input_tokens", 0) or 0
            completion_tokens = usage.get("output_tokens", 0) or 0
            tracked = True
        else:
            token_usage = (getattr(raw_message, "response_metadata", None) or {}).get(
                "token_usage", {}
            )
            if token_usage:
                prompt_tokens = token_usage.get("prompt_tokens", 0) or 0
                completion_tokens = token_usage.get("completion_tokens", 0) or 0
                tracked = True
    except Exception:
        prompt_tokens = 0
        completion_tokens = 0
        tracked = False

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "estimated_cost_usd": estimate_cost_usd(model, prompt_tokens, completion_tokens),
        "tracked": tracked,
    }


def merge_usage(a: Optional[Dict[str, Any]], b: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Adds two usage dicts together, returning a new dict. Missing keys default to 0.
    `tracked` is True if either side actually measured real usage."""
    a = a or ZERO_USAGE
    b = b or ZERO_USAGE
    return {
        "prompt_tokens": a.get("prompt_tokens", 0) + b.get("prompt_tokens", 0),
        "completion_tokens": a.get("completion_tokens", 0) + b.get("completion_tokens", 0),
        "total_tokens": a.get("total_tokens", 0) + b.get("total_tokens", 0),
        "estimated_cost_usd": a.get("estimated_cost_usd", 0.0) + b.get("estimated_cost_usd", 0.0),
        "tracked": bool(a.get("tracked", False) or b.get("tracked", False)),
    }


_current_run_context: "contextvars.ContextVar[Optional[Dict[str, str]]]" = contextvars.ContextVar(
    "_current_run_context", default=None
)


@contextmanager
def run_context(run_id: str, organization_id: str = "default-org"):
    """Context manager setting active run_id and organization_id for LLM and resilience telemetry."""
    token = _current_run_context.set({"run_id": run_id, "organization_id": organization_id})
    try:
        yield
    finally:
        _current_run_context.reset(token)


@contextmanager
def collect_usage():
    """
    Context manager that accumulates token usage for every LLM call made
    through `invoke_structured` inside its block. Yields the running dict.
    """
    token = _current_usage.set(dict(ZERO_USAGE))
    try:
        yield _current_usage.get()
    finally:
        _current_usage.reset(token)


def _infer_provider(llm: Any) -> str:
    """Infers provider identifier from client attributes or class name."""
    if hasattr(llm, "_provider") and getattr(llm, "_provider"):
        return str(getattr(llm, "_provider")).lower()
    if hasattr(llm, "provider") and getattr(llm, "provider"):
        return str(getattr(llm, "provider")).lower()
    cls_name = type(llm).__name__.lower()
    if "nvidia" in cls_name:
        return "nvidia"
    if "google" in cls_name or "gemini" in cls_name:
        return "gemini"
    if "openai" in cls_name:
        return "openai"
    if "groq" in cls_name:
        return "groq"
    from backend.core.config import LLM_PROVIDER
    return (LLM_PROVIDER or "nvidia").lower()


def _invoke_nvidia_single_format_structured(llm: Any, schema: Any, prompt: str) -> Any:
    """
    Single-round-trip structured-output attempt for NVIDIA-hosted models.

    Binds the direct, NIM-native `guided_json` top-level parameter - the
    same field ChatNVIDIA.with_structured_output()'s own "direct format"
    uses internally (see the installed langchain-nvidia-ai-endpoints
    source: `nvext_param = {"guided_json": json_schema}` passed via
    `super().bind(**nvext_param, ...)`) - via the same public `.bind()`
    mechanism Runnable already exposes, then parses the JSON from the
    response using this module's own extraction/validation. Returns None
    (never raises) on an empty or unparseable completion, so the caller's
    existing empty-completion / malformed-JSON handling in
    `_invoke_single_provider` applies exactly as it does for any other
    provider, without duplicating that classification logic here.

    NOT the OpenAI-compatible `response_format={"type": "json_schema", ...}`
    parameter this used before 2026-09-24: six bounded, single-attempt raw
    HTTPS requests directly against NVIDIA's API (bypassing this SDK
    entirely) proved `response_format` requests intermittently receive
    "503 ResourceExhausted: Worker local total request limit reached
    (16/16)" - reproduced with a minimal 1-field schema and the real
    RoutingDecision schema alike, while a same-schema request using direct
    `guided_json` succeeded - confirming NVIDIA routes `response_format`
    requests through a small, separately-pooled, contended set of workers
    on the hosted endpoint, unrelated to schema size/content or anything
    this application controls.

    Deliberately NOT `nvext={"guided_json": ...}` (the SDK's other
    supported format): the same investigation proved that for this model,
    nvext.guided_json returns HTTP 200 while silently ignoring the schema
    entirely (a plain-text completion, not JSON) - worse than a
    classifiable failure, since it would look like success and then fail
    Pydantic validation non-deterministically depending on what the model
    happened to say.
    """
    import re

    guided_schema = schema.model_json_schema()
    # Only the network call itself is left free to raise: a timeout, 5xx, or
    # rate limit here is a real failure the caller needs to see immediately
    # (see the caller's comment on why it isn't swallowed). A response that
    # arrives but is empty or fails schema validation is this function's own
    # "didn't work" outcome, not the caller's problem to classify - it's
    # reported back as None, same as any other unusable completion.
    raw = llm.bind(guided_json=guided_schema).invoke(prompt)
    if not raw.content or not raw.content.strip():
        return None
    match = re.search(r"\{.*\}", raw.content, re.DOTALL)
    if not match:
        return None
    try:
        return schema.model_validate_json(match.group(0))
    except Exception:
        return None


def _invoke_single_provider(llm: Any, schema: Any, prompt: str) -> Any:
    """
    Performs the structured output invocation against a single provider instance.
    Records token usage into the active collect_usage() block if present.
    """
    import time
    import re
    from backend.services.errors import (
        classify_llm_exception,
        LLMAuthenticationError,
        LLMInvalidRequestError,
        LLMMalformedResponseError,
        LLMRateLimitError,
        LLMTimeoutError,
    )

    model_name = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
    schema_name = getattr(schema, "__name__", str(schema))
    # ChatNVIDIA.with_structured_output(), for hosted endpoints, always
    # tries up to 3 request formats in sequence (OpenAI response_format,
    # then guided_json direct, then guided_json via nvext) whenever a
    # format's parser can't build the schema object - see the installed
    # langchain-nvidia-ai-endpoints==1.4.3 source, whose own docstring
    # calls formats 2-3 "mostly defensive" since format 1 is "expected to
    # succeed" for hosted models. For a NVIDIA-hosted model that returns an
    # empty completion (the model itself produced nothing, not a
    # format-compatibility problem), all 3 formats fail identically, so the
    # caller pays for 3 full round-trips - confirmed on real hardware (with
    # the then-current openai/gpt-oss-20b) to take up to ~130s - before
    # this function's own raw-invoke fallback further below even runs.
    # Binding the request directly via `_invoke_nvidia_single_format_structured`
    # (using direct guided_json - see that function's own docstring for why
    # response_format and nvext.guided_json are both deliberately avoided)
    # gets the identical "expected to succeed" attempt in a single call, so
    # an empty completion is discovered - and classified as
    # LLMMalformedResponseError by the unchanged logic below - without the
    # other two redundant attempts. Generalized to every NVIDIA-hosted
    # model (2026-09-24, alongside the openai/gpt-oss-20b -> nemotron
    # migration): the underlying with_structured_output() 3-format-cascade
    # behavior this works around is a property of ChatNVIDIA's hosted-model
    # path in general, not specific to any one model family - gating it to
    # a "gpt-oss" substring match would have silently stopped applying this
    # optimization the moment the configured NVIDIA model changed. Every
    # other provider keeps the original with_structured_output() path.
    is_nvidia_hosted_model = _infer_provider(llm) == "nvidia"
    print(f"[LLM] Requesting structured output for {schema_name} from {model_name}...", flush=True)
    t0 = time.time()

    for attempt in range(5):
        try:
            result = None
            try:
                structured = llm.with_structured_output(schema, include_raw=True)
                result = structured.invoke(prompt)
            except (NotImplementedError, Exception) as inner_exc:
                if "guided_json" in str(inner_exc) or "[400]" in str(inner_exc):
                    pass
                elif isinstance(inner_exc, NotImplementedError):
                    if is_nvidia_hosted_model:
                        # Deliberately NOT wrapped in a swallowing
                        # try/except like the branch below: a genuine
                        # network failure (timeout, 5xx, rate limit) here
                        # must propagate immediately so it gets classified
                        # and can trigger the safe provider fallback below,
                        # rather than being silently discarded and paying
                        # for a second, likely-equally-slow direct-invoke
                        # attempt. Only an empty/malformed *response*
                        # (returned as None, not raised) falls through to
                        # that direct-invoke fallback - the same graceful
                        # degradation every other provider already gets.
                        res = _invoke_nvidia_single_format_structured(llm, schema, prompt)
                        if res is not None:
                            print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                            return res
                    else:
                        try:
                            structured = llm.with_structured_output(schema)
                            res = structured.invoke(prompt)
                            if res is not None:
                                print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                                return res
                        except Exception:
                            pass
                else:
                    raise inner_exc

            if isinstance(result, dict):
                if result.get("parsing_error") is not None:
                    raise result["parsing_error"]
                if result.get("parsed") is not None:
                    raw = result.get("raw")
                    if raw is not None:
                        model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
                        collector = _current_usage.get()
                        if collector is not None:
                            call_usage = extract_usage(raw, model)
                            collector.update(merge_usage(collector, call_usage))
                    print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                    return result["parsed"]
            elif result is not None:
                print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                return result

            # Direct fallback if with_structured_output produced None or failed
            raw = llm.invoke(prompt)
            model = getattr(llm, "model", None) or getattr(llm, "model_name", "unknown")
            collector = _current_usage.get()
            if collector is not None:
                call_usage = extract_usage(raw, model)
                collector.update(merge_usage(collector, call_usage))

            # A genuinely empty completion body is never meaningfully
            # parseable - detected here, before any parsing is attempted,
            # rather than let it fall through the regex match (which
            # simply won't match) and the "convert to JSON" re-prompt
            # below into model_validate_json("") at the bottom, which
            # raises a Pydantic ValidationError that would otherwise be a
            # misleading way to discover the same fact. Raising the
            # correctly-classified, retryable error immediately also
            # skips a second, near-certain-to-fail LLM call (the re-prompt
            # asking the model to "convert" nothing into JSON).
            if not raw.content or not raw.content.strip():
                raise LLMMalformedResponseError(
                    f"LLM returned an empty completion for structured output ({schema_name})",
                    provider=_infer_provider(llm),
                )

            match = re.search(r"\{.*\}", raw.content, re.DOTALL)
            if match and hasattr(schema, "model_validate_json"):
                try:
                    print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
                    return schema.model_validate_json(match.group(0))
                except Exception:
                    pass

            # Re-prompt model to extract strict JSON if output was free-form markdown
            schema_doc = schema.model_json_schema() if hasattr(schema, "model_json_schema") else str(schema)
            conv_prompt = (
                f"Convert this text into a single valid JSON object matching this schema:\n{schema_doc}\n\n"
                f"Input text:\n{raw.content}\n\nReturn ONLY the JSON object, no markdown, no explanation."
            )
            conv_res = llm.invoke(conv_prompt)
            conv_match = re.search(r"\{.*\}", conv_res.content, re.DOTALL)
            print(f"[LLM] Response received in {time.time()-t0:.2f}s", flush=True)
            if conv_match and hasattr(schema, "model_validate_json"):
                return schema.model_validate_json(conv_match.group(0))
            if hasattr(schema, "model_validate_json"):
                return schema.model_validate_json(conv_res.content)
            return conv_res.content

        except Exception as e:
            # Fast fail on non-retriable exceptions, and on malformed
            # responses (also fast-failed here, not retried internally,
            # because their text is model-controlled and could otherwise
            # coincidentally match the "429"/"guided_json"/"[400]" retry
            # checks below) - malformed responses still propagate up to
            # invoke_structured's own bounded primary -> fallback provider
            # switch, which is where their retry actually belongs.
            classified = classify_llm_exception(e)
            if isinstance(
                classified,
                (
                    LLMAuthenticationError,
                    LLMInvalidRequestError,
                    LLMTimeoutError,
                    LLMMalformedResponseError,
                    # A rate-limit/quota error is fast-failed immediately,
                    # never retried within this same provider call: a 429
                    # backed by a per-second rate limit won't clear in the
                    # ~1-3s this loop could wait, and a 429 backed by a
                    # daily/monthly quota (the real production case -
                    # generativelanguage.googleapis.com's free-tier
                    # GenerateRequestsPerDayPerProjectPerModel quota) won't
                    # clear for hours - retrying either just spends bounded
                    # request budget for no chance of success. The caller
                    # (invoke_structured) is where retry-shaped recovery
                    # belongs for a rate limit: switching to a DIFFERENT
                    # credentialed provider, not re-asking the same one.
                    LLMRateLimitError,
                ),
            ):
                raise classified

            if ("guided_json" in str(e) or "[400]" in str(e)) and attempt < 2:
                try:
                    raw = llm.invoke(prompt)
                    match = re.search(r"\{.*\}", raw.content, re.DOTALL)
                    if match and hasattr(schema, "model_validate_json"):
                        return schema.model_validate_json(match.group(0))
                except Exception:
                    pass
            raise classified


def _invoke_nvidia_with_resilience(llm: Any, schema: Any, prompt: str) -> Any:
    """
    Wraps _invoke_single_provider with NVIDIA-specific capacity resilience: a
    process-local concurrency guard bounding simultaneous in-flight NVIDIA
    requests, a bounded exponential-backoff retry for transient
    (503/ResourceExhausted) failures, and a circuit breaker that fails fast
    during a sustained NVIDIA capacity outage instead of continuing to retry
    into it (see backend/services/nvidia_resilience.py).

    Deliberately scoped to `type(classified) is LLMTransientError` (the
    generic 5xx/"service unavailable" case), not `isinstance` - a subclass
    like LLMMalformedResponseError must keep propagating on the first
    failure exactly as before, unretried here, since a malformed response is
    a response-quality problem this retry (built for capacity contention)
    isn't meant to address. Every other classification (timeout, rate
    limit, auth, invalid request) is untouched and propagates immediately,
    so invoke_structured's existing primary -> fallback switch is unaffected
    for those cases.
    """
    from backend.services.errors import classify_llm_exception, LLMTransientError
    from backend.services.nvidia_resilience import (
        nvidia_semaphore,
        nvidia_circuit,
        compute_backoff_delay,
        sleep as _nvidia_sleep,
    )
    from backend.core.config import NVIDIA_MAX_RETRIES

    state = nvidia_circuit.state
    if not nvidia_circuit.allow_request():
        print("[LLM] NVIDIA circuit breaker OPEN - failing fast without calling provider", flush=True)
        raise LLMTransientError(
            "NVIDIA circuit breaker open - provider capacity outage, failing fast",
            provider="nvidia",
        )
    if state == "HALF_OPEN":
        print("[LLM] NVIDIA circuit breaker HALF-OPEN - allowing recovery probe request", flush=True)

    with nvidia_semaphore:
        for attempt in range(NVIDIA_MAX_RETRIES + 1):
            try:
                result = _invoke_single_provider(llm, schema, prompt)
                if nvidia_circuit.record_success():
                    print("[LLM] NVIDIA circuit breaker CLOSED - provider healthy again", flush=True)
                return result
            except Exception as e:
                classified = classify_llm_exception(e, provider="nvidia")
                if type(classified) is not LLMTransientError:
                    raise classified
                if nvidia_circuit.record_failure():
                    print(
                        f"[LLM] NVIDIA circuit breaker OPENED after repeated capacity failures - "
                        f"cooling down {nvidia_circuit.cooldown_seconds:.0f}s",
                        flush=True,
                    )
                if attempt < NVIDIA_MAX_RETRIES:
                    delay = compute_backoff_delay(attempt)
                    print(
                        f"[LLM] NVIDIA transient capacity failure (503/ResourceExhausted), "
                        f"retry {attempt + 1}/{NVIDIA_MAX_RETRIES} in {delay:.2f}s...",
                        flush=True,
                    )
                    _nvidia_sleep(delay)
                    continue
                print(
                    f"[LLM] NVIDIA retries exhausted after {NVIDIA_MAX_RETRIES + 1} attempts.",
                    flush=True,
                )
                raise classified


def _dispatch_invoke(llm: Any, schema: Any, prompt: str) -> Any:
    """Routes NVIDIA calls through the capacity-resilience wrapper; every
    other provider is unaffected and calls _invoke_single_provider directly."""
    if _infer_provider(llm) == "nvidia":
        return _invoke_nvidia_with_resilience(llm, schema, prompt)
    return _invoke_single_provider(llm, schema, prompt)


def invoke_structured(
    llm: Any,
    schema: Any,
    prompt: str,
    run_id: Optional[str] = None,
    organization_id: Optional[str] = None,
) -> Any:
    """
    Invokes `llm` for structured `schema` output with bounded request timeout,
    token telemetry recording, structured error classification, and bounded safe provider fallback.

    Maximum provider attempts: 3 (Primary -> Fallback -> Structured Error), with
    one bounded exception: if the Fallback provider itself fails specifically
    with a rate-limit/quota error AND a third, distinct, credentialed provider
    is configured, exactly one further attempt is made against that provider
    before failing closed - a rate-limited provider is known-exhausted, so
    re-raising immediately would give up on a real, already-available
    alternative. No provider is ever attempted more than once, and no
    provider is ever selected without a valid, present API key.
    Fallback is ONLY allowed for explicitly classified transient failures and timeouts.
    Authentication errors, invalid requests, and policy/schema failures NEVER trigger fallback.
    """
    import time
    from backend.services.errors import (
        classify_llm_exception,
        is_fallback_eligible,
        LLMTimeoutError,
        LLMRateLimitError,
        LLMMalformedResponseError,
    )
    from backend.services.llm import get_llm, get_fallback_provider
    from backend.observability.collector import telemetry_collector
    from backend.schemas.telemetry import TelemetryEventType

    primary_provider = _infer_provider(llm)
    t0 = time.time()

    # Attempt 1: Primary Provider
    try:
        return _dispatch_invoke(llm, schema, prompt)
    except Exception as primary_exc:
        elapsed_ms = (time.time() - t0) * 1000.0
        classified_primary = classify_llm_exception(primary_exc, provider=primary_provider)
        classified_primary.elapsed_ms = elapsed_ms

        # Check fallback eligibility (ONLY transient / timeout / rate limit)
        if not is_fallback_eligible(classified_primary):
            print(
                f"[LLM] Primary provider '{primary_provider}' failed with non-transient error: "
                f"{type(classified_primary).__name__}. Failing closed immediately without fallback.",
                flush=True,
            )
            raise classified_primary

        fallback_provider = get_fallback_provider(primary=primary_provider)
        if not fallback_provider or fallback_provider == primary_provider:
            print(
                f"[LLM] Primary provider '{primary_provider}' failed with {type(classified_primary).__name__}, "
                f"but no alternative fallback provider is configured. Failing closed.",
                flush=True,
            )
            raise classified_primary

        # Resolve tenant/run context for sanitized resilience telemetry
        run_ctx = _current_run_context.get() or {}
        effective_run_id = run_id or run_ctx.get("run_id", "adhoc-run")
        effective_org_id = organization_id or run_ctx.get("organization_id", "default-org")

        if isinstance(classified_primary, LLMTimeoutError):
            failure_cat = "LLM_TIMEOUT"
        elif isinstance(classified_primary, LLMRateLimitError):
            failure_cat = "LLM_RATE_LIMIT"
        elif isinstance(classified_primary, LLMMalformedResponseError):
            failure_cat = "LLM_MALFORMED_RESPONSE"
        else:
            failure_cat = "LLM_TRANSIENT_FAILURE"

        # Emit PROVIDER_FALLBACK resilience event (Attempt 1 -> Attempt 2)
        telemetry_collector.on_provider_fallback(
            run_id=effective_run_id,
            organization_id=effective_org_id,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            failure_category=failure_cat,
            failure_type=type(classified_primary).__name__,
            attempt_number=1,
            elapsed_ms=elapsed_ms,
            model=getattr(llm, "model", None) or getattr(llm, "model_name", None),
        )

        print(
            f"[LLM] SAFE PROVIDER FALLBACK: Primary provider '{primary_provider}' failed with "
            f"{type(classified_primary).__name__} after {elapsed_ms:.1f}ms. "
            f"Switching to fallback provider '{fallback_provider}' (attempt 2 of 2)...",
            flush=True,
        )

        # Attempt 2: Fallback Provider
        try:
            fallback_llm = get_llm(provider=fallback_provider)
        except Exception as init_exc:
            print(
                f"[LLM] Failed to initialize fallback provider '{fallback_provider}': {init_exc}. "
                f"Total provider attempts exhausted.",
                flush=True,
            )
            # Distinct, durable telemetry signal for "a fallback provider was
            # selected but never actually ran" - reuses the existing
            # PROVIDER_FALLBACK event via record_event() (no new event type
            # or collector method needed) with an explicit outcome field, so
            # this is distinguishable from a fallback attempt that ran and
            # also failed. Deliberately records only pre-known-safe fields -
            # never str(init_exc) or anything else derived from the raw
            # exception, which could in principle echo credential-bearing
            # configuration (e.g. a client SDK's own error text).
            telemetry_collector.record_event(
                run_id=effective_run_id,
                organization_id=effective_org_id,
                event_type=TelemetryEventType.PROVIDER_FALLBACK,
                duration_ms=elapsed_ms,
                metadata={
                    "run_id": effective_run_id,
                    "organization_id": effective_org_id,
                    "primary_provider": primary_provider,
                    "fallback_provider": fallback_provider,
                    "failure_category": failure_cat,
                    "outcome": "fallback_init_failed",
                    "attempt_number": 1,
                },
            )
            # Exception chaining: init_exc is preserved as __cause__ instead
            # of being silently discarded. What's raised is unchanged
            # (still classified_primary) - fail-closed behavior is preserved.
            raise classified_primary from init_exc

        t1 = time.time()
        try:
            result = _dispatch_invoke(fallback_llm, schema, prompt)
            print(
                f"[LLM] Fallback provider '{fallback_provider}' succeeded in {time.time()-t1:.2f}s.",
                flush=True,
            )
            return result
        except Exception as fallback_exc:
            fb_elapsed_ms = (time.time() - t1) * 1000.0
            classified_fallback = classify_llm_exception(fallback_exc, provider=fallback_provider)
            classified_fallback.elapsed_ms = fb_elapsed_ms
            print(
                f"[LLM] Fallback provider '{fallback_provider}' also failed with "
                f"{type(classified_fallback).__name__}.",
                flush=True,
            )

            # Bounded escape hatch, rate-limit only: a rate-limited fallback
            # is known-exhausted for this invocation (retrying IT would just
            # repeat the same 429/quota error - see the fast-fail in
            # _invoke_single_provider), but a completely different,
            # untouched, credentialed provider might still work right now.
            # Tried at most once - if it also fails, its own classified
            # error is raised below with no further fallback attempted.
            if isinstance(classified_fallback, LLMRateLimitError):
                second_fallback_provider = get_fallback_provider(
                    primary=primary_provider, exclude=[fallback_provider]
                )
                if second_fallback_provider:
                    telemetry_collector.on_provider_fallback(
                        run_id=effective_run_id,
                        organization_id=effective_org_id,
                        primary_provider=fallback_provider,
                        fallback_provider=second_fallback_provider,
                        failure_category="LLM_RATE_LIMIT",
                        failure_type=type(classified_fallback).__name__,
                        attempt_number=2,
                        elapsed_ms=fb_elapsed_ms,
                        model=getattr(fallback_llm, "model", None) or getattr(fallback_llm, "model_name", None),
                    )
                    print(
                        f"[LLM] Fallback provider '{fallback_provider}' was rate-limited. "
                        f"Trying second fallback provider '{second_fallback_provider}' "
                        f"(final attempt, 3 of 3)...",
                        flush=True,
                    )
                    try:
                        second_fallback_llm = get_llm(provider=second_fallback_provider)
                        t2 = time.time()
                        result = _dispatch_invoke(second_fallback_llm, schema, prompt)
                        print(
                            f"[LLM] Second fallback provider '{second_fallback_provider}' "
                            f"succeeded in {time.time()-t2:.2f}s.",
                            flush=True,
                        )
                        return result
                    except Exception as second_fallback_exc:
                        classified_second_fallback = classify_llm_exception(
                            second_fallback_exc, provider=second_fallback_provider
                        )
                        print(
                            f"[LLM] Second fallback provider '{second_fallback_provider}' also failed "
                            f"with {type(classified_second_fallback).__name__}. "
                            f"Maximum provider attempts (3) reached. Raising final structured error.",
                            flush=True,
                        )
                        raise classified_second_fallback from fallback_exc

            print(
                "[LLM] Maximum provider attempts reached. Raising final structured error.",
                flush=True,
            )
            raise classified_fallback

