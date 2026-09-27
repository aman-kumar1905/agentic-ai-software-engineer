import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia").strip().lower()
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME")  # provider-specific default applied in services/llm.py

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Groq is a credentialed FALLBACK provider only (see backend/services/llm.py's
# _FALLBACK_ORDER) - it is never selected as LLM_PROVIDER (primary) by any
# code in this repo, but nothing prevents an operator from setting
# LLM_PROVIDER=groq explicitly if they choose to.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# 30s (reduced from 75s, 2026-09-24, alongside the migration off
# openai/gpt-oss-20b - see _DEFAULT_MODELS in backend/services/llm.py):
# that prior 75s bound was sized around openai/gpt-oss-20b's own observed
# ~37-49s single-round-trip latency, which was itself only a workaround for
# that model's endpoint being stalled. Raw minimal HTTPS requests against
# the exact same NVIDIA endpoint and credentials confirmed other hosted
# models respond in ~1-7s - 30s keeps a real margin over that normal
# latency without waiting anywhere near as long to detect a genuinely
# stalled request; a request slower than this is handled by the existing
# safe provider fallback, not by raising this further. Still fully
# configurable via the LLM_REQUEST_TIMEOUT_SECONDS environment variable.
LLM_REQUEST_TIMEOUT_SECONDS = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30.0"))

# NVIDIA capacity resilience (backend/services/nvidia_resilience.py). NVIDIA's
# hosted endpoint is a shared pool that can return 503 ResourceExhausted under
# contention unrelated to this application's own request rate - these bound
# how hard this process leans on it before giving up and falling back to a
# different credentialed provider. All process-local, never distributed.
NVIDIA_MAX_CONCURRENT_REQUESTS = int(os.getenv("NVIDIA_MAX_CONCURRENT_REQUESTS", "2"))
NVIDIA_MAX_RETRIES = int(os.getenv("NVIDIA_MAX_RETRIES", "3"))
NVIDIA_CIRCUIT_FAILURE_THRESHOLD = int(os.getenv("NVIDIA_CIRCUIT_FAILURE_THRESHOLD", "3"))
NVIDIA_CIRCUIT_COOLDOWN_SECONDS = float(os.getenv("NVIDIA_CIRCUIT_COOLDOWN_SECONDS", "30.0"))

WORKSPACE_LOCK_TIMEOUT_SECONDS = float(os.getenv("WORKSPACE_LOCK_TIMEOUT_SECONDS", "30.0"))
SHUTDOWN_DRAIN_TIMEOUT_SECONDS = float(os.getenv("SHUTDOWN_DRAIN_TIMEOUT_SECONDS", "30.0"))


def get_frontend_origins() -> list[str]:
    """
    Parses allowed frontend origins for CORS.
    Supports comma-separated list via FRONTEND_ORIGIN or FRONTEND_ORIGINS.
    In production mode (AUTH_MODE=production or ENVIRONMENT=production),
    fail-closed: do NOT silently fall back to development localhost origins.
    """
    raw = os.getenv("FRONTEND_ORIGIN") or os.getenv("FRONTEND_ORIGINS")
    if raw:
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        if origins:
            return origins

    auth_mode = os.getenv("AUTH_MODE", "").strip().lower()
    env_name = os.getenv("ENVIRONMENT", "").strip().lower()
    if auth_mode == "production" or env_name == "production":
        return []

    return [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]


FRONTEND_ORIGINS = get_frontend_origins()


def validate_config(provider: str | None = None):
    """Validates that necessary provider keys are present prior to LLM invocation."""
    eff_provider = (provider or os.getenv("LLM_PROVIDER", LLM_PROVIDER)).strip().lower()
    nvidia_key = os.getenv("NVIDIA_API_KEY", NVIDIA_API_KEY)
    google_key = os.getenv("GOOGLE_API_KEY", GOOGLE_API_KEY)
    openai_key = os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

    if eff_provider == "nvidia" and not nvidia_key:
        raise ValueError(
            "LLM_PROVIDER is 'nvidia' but NVIDIA_API_KEY is missing. Add it to environment variables."
        )
    if eff_provider == "gemini" and not google_key:
        raise ValueError(
            "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing. Add it to environment variables."
        )
    if eff_provider == "openai" and not openai_key:
        raise ValueError(
            "LLM_PROVIDER is 'openai' but OPENAI_API_KEY is missing. Add it to environment variables."
        )
    if eff_provider == "groq" and not os.getenv("GROQ_API_KEY", GROQ_API_KEY):
        raise ValueError(
            "LLM_PROVIDER is 'groq' but GROQ_API_KEY is missing. Add it to environment variables."
        )
