import os
from typing import Iterable, Optional

from backend.core.config import (
    LLM_PROVIDER,
    LLM_MODEL_NAME,
    LLM_REQUEST_TIMEOUT_SECONDS,
    NVIDIA_API_KEY,
    GOOGLE_API_KEY,
    OPENAI_API_KEY,
    GROQ_API_KEY,
    validate_config,
)

# Fallback model per provider, used only when LLM_MODEL_NAME isn't set.
# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning replaces openai/gpt-oss-20b
# (2026-09-24): a read-only investigation with raw minimal HTTPS requests
# against the exact same NVIDIA endpoint and credentials confirmed
# openai/gpt-oss-20b itself was stalled (95+s hangs), while other NVIDIA
# models on the identical endpoint/credentials responded in ~1-7s - not a
# prompt-length, reasoning_effort, retry-loop, or multi-request application
# bug. If this model goes end-of-life too, set LLM_MODEL_NAME in .env
# rather than editing this file.
# gemini-3.6-flash replaces gemini-2.0-flash (run_120607d608c7, 2026-09-19):
# Google decommissioned gemini-2.0-flash (404 NOT_FOUND, "no longer
# available"), which broke the NVIDIA->Gemini fallback path itself - the
# fallback-model-scoping fix (PR #21) was working correctly, Gemini's own
# default was just stale. Confirmed via the account's live ListModels API
# (generateContent supported) before pinning.
_DEFAULT_MODELS = {
    "nvidia": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "gemini": "gemini-3.6-flash",
    "openai": "gpt-4o",
    # openai/gpt-oss-20b, hosted on Groq (not NVIDIA - same open-weight model,
    # different infrastructure). Verified live (2026-09-27) against the
    # installed langchain-groq==1.1.3: with_structured_output(RoutingDecision,
    # include_raw=True) returns a correctly parsed result in <1s. The larger
    # openai/gpt-oss-120b was tried first and rejected - it failed the exact
    # same live request with "Tool choice is required, but model did not call
    # a tool" (400), producing free-form JSON prose instead of a tool call.
    "groq": "openai/gpt-oss-20b",
}


def get_llm(provider: Optional[str] = None, timeout: Optional[float] = None):
    """
    Initializes and returns the configured LLM client with bounded timeout.
    Supports 'nvidia', 'gemini', and 'openai'.
    """
    eff_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    default_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", str(LLM_REQUEST_TIMEOUT_SECONDS)))
    eff_timeout = float(timeout) if timeout is not None else default_timeout
    validate_config(eff_provider)

    # LLM_MODEL_NAME is a single, provider-agnostic override with no
    # provider of its own attached to it - it exists to pin the
    # CONFIGURED PRIMARY provider to a specific model (e.g. NVIDIA to a
    # specific hosted model after an upstream deprecation or stall), not
    # to name a model for whichever provider happens to be requested.
    # Applying it unconditionally meant a safe provider fallback (e.g.
    # NVIDIA -> Gemini, backend/observability/telemetry.py's explicit
    # get_llm(provider=fallback_provider) call) reused the PRIMARY
    # provider's model name against the FALLBACK provider's API, which has
    # no such model - confirmed in production as a Gemini 404 for a
    # NVIDIA-only model name. Every other call site in this codebase calls
    # get_llm() with no explicit provider at all, so eff_provider is always
    # the configured primary there already - this guard changes nothing
    # for them.
    configured_primary = os.getenv("LLM_PROVIDER", LLM_PROVIDER).strip().lower()
    explicit_model = os.getenv("LLM_MODEL_NAME") or LLM_MODEL_NAME
    model = (explicit_model if eff_provider == configured_primary else None) or _DEFAULT_MODELS.get(eff_provider)
    if not model:
        # Fail closed with a clear, actionable error instead of silently
        # constructing a provider client with model=None, which each SDK
        # surfaces as its own confusing, provider-specific error deep
        # inside the first request rather than at configuration time.
        raise ValueError(
            f"No model configured for LLM provider '{eff_provider}' - add it to "
            f"_DEFAULT_MODELS in backend/services/llm.py or set LLM_MODEL_NAME "
            f"(only applies when '{eff_provider}' is the configured primary provider)."
        )

    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)
    groq_key = os.getenv("GROQ_API_KEY", GROQ_API_KEY)

    if eff_provider == "nvidia":
        from langchain_nvidia_ai_endpoints import ChatNVIDIA
        # Historical note (model since migrated away from, 2026-09-24 - see
        # _DEFAULT_MODELS above): openai/gpt-oss-* previously received a
        # `reasoning_effort="low"` constructor kwarg, on the theory that it
        # bounds the model's hidden "thinking" budget. Removed (2026-09-27)
        # after empirical verification proved it does nothing on NVIDIA's
        # hosted endpoint for that model:
        #   1. The installed langchain-nvidia-ai-endpoints==1.4.3 package's
        #      own static model registry (_statics.py) marks
        #      "openai/gpt-oss-20b" with the default `supports_thinking =
        #      False` - it is not a recognized thinking-controllable model
        #      via the SDK's own first-class mechanism
        #      (thinking_param_enable/disable). Passing reasoning_effort as
        #      a constructor kwarg is therefore an unmanaged pydantic
        #      "unknown field" passthrough into `model_kwargs` (which
        #      itself prints a UserWarning on every client construction),
        #      not a supported control.
        #   2. Three direct HTTPS requests to
        #      integrate.api.nvidia.com/v1/chat/completions for this exact
        #      model - with no reasoning_effort, with it at the top level
        #      (what model_kwargs sends), and with it nested under
        #      chat_template_kwargs (the vLLM/NIM convention some hosted
        #      OSS models use) - all three timed out identically at ~90-92s
        #      with zero measurable difference. The parameter has no effect
        #      on this endpoint's latency in any shape.
        # The real, current bottleneck is NVIDIA's own hosted response time
        # for this model, which is external and not addressable by tuning a
        # client-side request parameter - the existing LLM_TIMEOUT
        # classification and provider fallback (to Gemini) are what
        # actually handle this, unchanged.
        client = ChatNVIDIA(model=model, api_key=nvidia_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "nvidia")
        setattr(client, "_timeout", eff_timeout)
        return client

    if eff_provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        client = ChatGoogleGenerativeAI(model=model, google_api_key=google_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "gemini")
        setattr(client, "_timeout", eff_timeout)
        return client

    if eff_provider == "openai":
        from langchain_openai import ChatOpenAI
        client = ChatOpenAI(model=model, api_key=openai_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "openai")
        setattr(client, "_timeout", eff_timeout)
        return client

    if eff_provider == "groq":
        from langchain_groq import ChatGroq
        client = ChatGroq(model=model, api_key=groq_key, temperature=0, timeout=eff_timeout)
        setattr(client, "_provider", "groq")
        setattr(client, "_timeout", eff_timeout)
        return client

    raise ValueError(
        f"Unsupported LLM_PROVIDER: '{eff_provider}'. Supported: 'nvidia', 'gemini', 'openai', 'groq'."
    )


def _has_provider_credentials(provider: str) -> bool:
    """
    Centralizes the NVIDIA/Gemini/OpenAI credential-presence check that used
    to be duplicated (and, for fallback selection, only partially applied)
    across get_llm(), get_fallback_provider(), and validate_config(). A
    provider with no key configured is never "available", regardless of
    where in the codebase that's being decided.
    """
    provider = (provider or "").strip().lower()
    if provider == "nvidia":
        return bool(os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY))
    if provider == "gemini":
        return bool(os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY))
    if provider == "openai":
        return bool(os.getenv("OPENAI_API_KEY", OPENAI_API_KEY))
    if provider == "groq":
        return bool(os.getenv("GROQ_API_KEY", GROQ_API_KEY))
    return False


# Deterministic fallback preference order per primary provider - unchanged
# from the original hard-coded per-branch order, just centralized into one
# table so get_fallback_provider() can walk it uniformly instead of
# hand-rolling three near-duplicate if/else ladders.
_FALLBACK_ORDER = {
    "nvidia": ("groq", "gemini", "openai"),
    "gemini": ("nvidia", "openai"),
    "openai": ("nvidia", "gemini"),
    "groq": ("nvidia", "gemini", "openai"),
}


def get_fallback_provider(
    primary: Optional[str] = None,
    exclude: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """
    Resolves the healthy, eligible fallback provider for bounded fallback.
    Returns None if fallback is disabled, or if no *credentialed*
    alternative provider is configured - a provider with no API key is
    never selectable, not even as a last-resort default (previously, the
    final branch of each preference ladder returned a hardcoded provider
    name unconditionally, so a completely uncredentialed provider could be
    selected and would only fail later, deep inside fallback initialization).

    `exclude` additionally rules out providers already attempted earlier in
    the SAME invocation (e.g. a fallback that just failed with a rate-limit
    error) - callers use this to find the NEXT credentialed alternative
    instead of re-selecting one already known to be exhausted. `primary` is
    always excluded regardless of `exclude`.
    """
    if os.getenv("LLM_FALLBACK_ENABLED", "true").strip().lower() in ("0", "false", "no"):
        return None

    primary_clean = (primary or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    excluded = {primary_clean} | {(p or "").strip().lower() for p in (exclude or ())}

    explicit_fallback = os.getenv("LLM_FALLBACK_PROVIDER")
    if explicit_fallback and not exclude:
        # The explicit single-provider override only applies to the FIRST
        # fallback selection - once that provider has already been tried
        # and excluded, honoring it again would just re-select the same
        # exhausted provider, so subsequent selections fall through to the
        # normal credential-checked preference order below instead.
        candidate = explicit_fallback.strip().lower()
        if (
            candidate in ("nvidia", "gemini", "openai")
            and candidate not in excluded
            and _has_provider_credentials(candidate)
        ):
            return candidate

    order = _FALLBACK_ORDER.get(primary_clean, ("gemini", "nvidia", "openai"))
    for candidate in order:
        if candidate not in excluded and _has_provider_credentials(candidate):
            return candidate

    return None
