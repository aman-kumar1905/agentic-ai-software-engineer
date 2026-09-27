from types import SimpleNamespace

import pytest

from backend.observability.telemetry import (
    collect_usage,
    estimate_cost_usd,
    extract_usage,
    invoke_structured,
    merge_usage,
)
from backend.services.errors import LLMMalformedResponseError


class TestEstimateCostUsd:
    def test_known_model_uses_its_rate(self):
        cost = estimate_cost_usd("claude-sonnet-4", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.003 + 0.015

    def test_unknown_model_uses_fallback_rate(self):
        cost = estimate_cost_usd("some-mystery-model", prompt_tokens=1000, completion_tokens=1000)
        assert cost == 0.0005 + 0.0015

    def test_zero_tokens_is_zero_cost(self):
        assert estimate_cost_usd("gemini-2.0-flash", 0, 0) == 0.0


class TestExtractUsage:
    def test_reads_standard_usage_metadata(self):
        message = SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
        )
        usage = extract_usage(message, "meta/llama-3.1-8b-instruct")
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150
        assert usage["estimated_cost_usd"] > 0
        assert usage["tracked"] is True

    def test_falls_back_to_response_metadata_token_usage(self):
        message = SimpleNamespace(
            usage_metadata=None,
            response_metadata={"token_usage": {"prompt_tokens": 20, "completion_tokens": 10}},
        )
        usage = extract_usage(message, "gpt-4o")
        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10
        assert usage["total_tokens"] == 30
        assert usage["tracked"] is True

    def test_missing_usage_defaults_to_zero_and_untracked_without_raising(self):
        message = SimpleNamespace()
        usage = extract_usage(message, "unknown-model")
        assert usage == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "estimated_cost_usd": 0.0,
            "tracked": False,
        }

    def test_malformed_message_never_raises(self):
        # response_metadata is not a dict; extraction should still degrade to zero/untracked.
        message = SimpleNamespace(usage_metadata=None, response_metadata="not-a-dict")
        usage = extract_usage(message, "unknown-model")
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["tracked"] is False


class TestMergeUsage:
    def test_adds_two_usage_dicts(self):
        a = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15, "estimated_cost_usd": 0.01, "tracked": True}
        b = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "estimated_cost_usd": 0.002, "tracked": True}
        merged = merge_usage(a, b)
        assert merged == {
            "prompt_tokens": 13,
            "completion_tokens": 7,
            "total_tokens": 20,
            "estimated_cost_usd": 0.012,
            "tracked": True,
        }

    def test_treats_none_as_zero(self):
        b = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5, "estimated_cost_usd": 0.002, "tracked": True}
        assert merge_usage(None, b) == b
        assert merge_usage(b, None) == b

    def test_tracked_is_true_if_either_side_measured_usage(self):
        untracked = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0, estimated_cost_usd=0.0, tracked=False)
        tracked = dict(prompt_tokens=1, completion_tokens=1, total_tokens=2, estimated_cost_usd=0.001, tracked=True)
        assert merge_usage(untracked, tracked)["tracked"] is True
        assert merge_usage(untracked, untracked)["tracked"] is False


class FakeStructuredRunnable:
    """Stands in for `llm.with_structured_output(schema, include_raw=True)`."""

    def __init__(self, parsed, raw, parsing_error=None):
        self._result = {"parsed": parsed, "raw": raw, "parsing_error": parsing_error}

    def invoke(self, prompt):
        return self._result


class FakeLLM:
    def __init__(
        self,
        parsed,
        raw,
        parsing_error=None,
        model="meta/llama-3.1-8b-instruct",
        direct_content=None,
    ):
        self.model = model
        self._runnable = FakeStructuredRunnable(parsed, raw, parsing_error)
        self._direct_content = direct_content

    def with_structured_output(self, schema, include_raw=False):
        assert include_raw is True
        return self._runnable

    def invoke(self, prompt):
        """Stands in for the "direct fallback" `llm.invoke(prompt)` call in
        _invoke_single_provider, reached when with_structured_output() produced
        nothing usable (parsed is None and there was no parsing_error)."""
        return SimpleNamespace(content=self._direct_content, usage_metadata=None)


class TestInvokeStructured:
    def test_returns_parsed_result(self):
        raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        llm = FakeLLM(parsed={"ok": True}, raw=raw)
        result = invoke_structured(llm, dict, "prompt")
        assert result == {"ok": True}

    def test_records_usage_into_active_collector(self):
        raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        llm = FakeLLM(parsed={"ok": True}, raw=raw)

        with collect_usage() as usage:
            invoke_structured(llm, dict, "prompt")

        assert usage["prompt_tokens"] == 10
        assert usage["completion_tokens"] == 5
        assert usage["total_tokens"] == 15
        assert usage["estimated_cost_usd"] > 0

    def test_accumulates_across_multiple_calls_in_one_block(self):
        raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        llm = FakeLLM(parsed={"ok": True}, raw=raw)

        with collect_usage() as usage:
            invoke_structured(llm, dict, "prompt")
            invoke_structured(llm, dict, "prompt")

        assert usage["prompt_tokens"] == 20
        assert usage["completion_tokens"] == 10

    def test_no_active_collector_does_not_raise(self):
        raw = SimpleNamespace(usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15})
        llm = FakeLLM(parsed={"ok": True}, raw=raw)
        # No collect_usage() block active - should simply not record anywhere.
        result = invoke_structured(llm, dict, "prompt")
        assert result == {"ok": True}

    def test_parsing_error_is_raised(self):
        llm = FakeLLM(parsed=None, raw=None, parsing_error=ValueError("bad schema"))
        try:
            invoke_structured(llm, dict, "prompt")
            assert False, "expected ValueError to propagate"
        except ValueError as e:
            assert "bad schema" in str(e)

    def test_empty_completion_is_malformed_not_invalid_request(self, monkeypatch):
        """
        An empty completion body (parsed=None, no parsing_error, direct
        llm.invoke() returns empty content) must be detected as
        LLMMalformedResponseError before it can reach model_validate_json("")
        and be misclassified as a non-retryable LLMInvalidRequestError.

        Clears every fallback-eligible provider's credential so this test's
        outcome depends only on the primary's own classification, not on
        which fallback credentials happen to be present in whatever real
        .env this suite runs against (malformed responses are fallback-
        eligible, so an incidentally-credentialed fallback would otherwise
        attempt a real call here with schema=dict, unrelated to this test).
        """
        import backend.services.llm as llm_module
        for env_name in ("GOOGLE_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
            monkeypatch.delenv(env_name, raising=False)
            monkeypatch.setattr(llm_module, env_name, None)

        llm = FakeLLM(parsed=None, raw=None, direct_content="")
        with pytest.raises(LLMMalformedResponseError):
            invoke_structured(llm, dict, "prompt")
