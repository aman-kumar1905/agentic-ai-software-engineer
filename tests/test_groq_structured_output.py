"""
Regression tests for the confirmed real EC2 E2E failure:

    [GROQ] LLM invalid request: Error code: 400 - {'error': {'message':
    'Tool choice is required, but model did not call a tool', 'type':
    'invalid_request_error', 'code': 'tool_use_failed', ...}}

Root cause: backend/observability/telemetry.py's _invoke_single_provider
called llm.with_structured_output(schema, include_raw=True) identically for
every non-NVIDIA provider. For ChatGroq (langchain-groq), this defaults to
method="function_calling" (tool calling) - openai/gpt-oss-20b sometimes
responds with ordinary text instead of invoking the tool, which Groq then
rejects outright with the above 400. This happened on a real E2E run using
the fully repository-aware Developer patch-generation prompt (not an
underspecified one) - the tool-choice failure mode is a property of Groq's
tool-calling path itself, not insufficient context.

Fix: Groq is now routed through method="json_schema" (Groq's own dedicated
Structured Output API using constrained decoding, not tool calling) with
strict=True - confirmed live against the real Groq API (10/10 trials with
the exact previously-failing prompt, plus the real nested
PatchResponse/FilePatch schema and DeveloperResult) before implementing.
strict=True is a documented no-op for any Groq model outside
langchain_groq's own strict-mode allowlist, so no model-name gating is
needed. This returns the identical {"raw","parsed","parsing_error"} dict
shape the existing code already handles - no other logic changed.
"""
from types import SimpleNamespace

import pytest

from backend.observability.telemetry import _invoke_single_provider, invoke_structured
from backend.schemas.routing import RoutingDecision, TaskType
from backend.developer.models import FilePatch
from backend.services.errors import LLMInvalidRequestError


class FakeGroqStructuredLLM:
    """
    Mimics the real ChatGroq contract relevant to this fix:
    with_structured_output()'s default method="function_calling" (tool
    calling) fails with the exact real "Tool choice is required" 400 -
    only method="json_schema" (with strict=True) succeeds. This proves the
    application code actually requests method="json_schema" for Groq,
    rather than merely asserting on a mocked-to-always-succeed fake.
    """

    def __init__(self, parsed_result, model_name="openai/gpt-oss-20b"):
        self._provider = "groq"
        self.provider = "groq"
        self.model = model_name
        self.model_name = model_name
        self.parsed_result = parsed_result
        self.calls: list[dict] = []

    def with_structured_output(self, schema, method="function_calling", strict=None, include_raw=False):
        self.calls.append({"schema": schema, "method": method, "strict": strict, "include_raw": include_raw})
        if method != "json_schema":
            # The exact real Groq 400 error shape from the production
            # failure, raised at the SAME point a real tool-calling
            # rejection would surface (inside .invoke(), not construction).
            return _RaisesToolChoiceError()

        parsed = self.parsed_result

        class _Runnable:
            def invoke(_self, prompt):
                return {"raw": SimpleNamespace(content="{}"), "parsed": parsed, "parsing_error": None}

        return _Runnable()

    def invoke(self, prompt):
        raise AssertionError("Direct .invoke() should not be reached when with_structured_output succeeds")


class _GroqBadRequestError(Exception):
    """Mirrors the real groq.BadRequestError shape relevant to
    classify_llm_exception: an openai-SDK-style client error exposes its
    HTTP status via a `.status_code` attribute, not just message text."""

    def __init__(self, message):
        super().__init__(message)
        self.status_code = 400


class _RaisesToolChoiceError:
    def invoke(self, prompt):
        raise _GroqBadRequestError(
            "Error code: 400 - {'error': {'message': 'Tool choice is required, but "
            "model did not call a tool', 'type': 'invalid_request_error', 'code': "
            "'tool_use_failed', 'failed_generation': 'I would be happy to help...'}}"
        )


class TestGroqUsesJsonSchemaNotToolCalling:
    def test_groq_structured_output_requests_json_schema_method(self):
        """The application must call with_structured_output(method='json_schema',
        strict=True, ...) for Groq - never the tool-calling default."""
        decision = RoutingDecision(
            task_type=TaskType.BUG_FIX, requires_planning=False,
            requires_knowledge=False, reasoning="ok",
        )
        llm = FakeGroqStructuredLLM(parsed_result=decision)

        result = _invoke_single_provider(llm, RoutingDecision, "prompt")

        assert result == decision
        assert len(llm.calls) == 1
        assert llm.calls[0]["method"] == "json_schema"
        assert llm.calls[0]["strict"] is True
        assert llm.calls[0]["include_raw"] is True

    def test_groq_tool_calling_default_would_have_failed_this_exact_way(self):
        """Sanity check on the fake itself: proves the fake genuinely
        reproduces the real failure when NOT given method='json_schema' -
        so the passing test above is verifying a real behavioral branch,
        not a fake that always succeeds regardless of how it's called."""
        decision = RoutingDecision(
            task_type=TaskType.BUG_FIX, requires_planning=False,
            requires_knowledge=False, reasoning="ok",
        )
        llm = FakeGroqStructuredLLM(parsed_result=decision)

        with pytest.raises(Exception, match="Tool choice is required"):
            llm.with_structured_output(RoutingDecision, method="function_calling").invoke("prompt")

    def test_valid_groq_json_parsed_into_requested_pydantic_model(self):
        """A successful Groq json_schema response is returned as the exact
        requested Pydantic instance, unchanged from what invoke_structured's
        existing dict-unwrapping logic already does for every provider."""
        decision = RoutingDecision(
            task_type=TaskType.DOCUMENTATION, requires_planning=True,
            requires_knowledge=True, reasoning="Needs README context.",
        )
        llm = FakeGroqStructuredLLM(parsed_result=decision)

        result = invoke_structured(llm, RoutingDecision, "prompt")

        assert isinstance(result, RoutingDecision)
        assert result == decision

    def test_malformed_groq_json_schema_response_fails_safely(self, monkeypatch):
        """A Groq json_schema call that comes back with a genuine JSON
        parsing failure must still be classified as LLMMalformedResponseError
        through the existing, unchanged pipeline - never silently accepted,
        never a fabricated result."""
        import json
        from backend.services.errors import LLMMalformedResponseError

        class _FailingRunnable:
            def invoke(self, prompt):
                return {
                    "raw": SimpleNamespace(content="not valid json"),
                    "parsed": None,
                    "parsing_error": json.JSONDecodeError("Expecting value", "not valid json", 0),
                }

        class _FailingGroqLLM:
            _provider = "groq"
            provider = "groq"
            model = "openai/gpt-oss-20b"
            model_name = "openai/gpt-oss-20b"

            def with_structured_output(self, schema, method="function_calling", strict=None, include_raw=False):
                assert method == "json_schema"
                return _FailingRunnable()

            def invoke(self, prompt):
                return SimpleNamespace(content="")

        with pytest.raises(LLMMalformedResponseError):
            _invoke_single_provider(_FailingGroqLLM(), RoutingDecision, "prompt")

    def test_groq_400_schema_rejection_classified_not_fabricated(self):
        """If Groq's own API rejects the request outright (e.g. a schema it
        cannot represent), the existing generic classification must produce
        a structured LLMInvalidRequestError - never free-form text accepted
        as a result."""
        llm = FakeGroqStructuredLLM(parsed_result="unused")
        # Force the tool-calling failure path directly to exercise the same
        # classification a genuine Groq-side schema rejection would hit.
        llm.with_structured_output = lambda schema, method="function_calling", strict=None, include_raw=False: _RaisesToolChoiceError()

        with pytest.raises(LLMInvalidRequestError):
            _invoke_single_provider(llm, RoutingDecision, "prompt")


class TestNvidiaStructuredOutputUnaffected:
    def test_nvidia_path_does_not_pass_groq_kwargs(self):
        """NVIDIA's with_structured_output() call must remain exactly as
        before this fix - no method/strict kwargs, since those are Groq-
        specific and irrelevant to NVIDIA's own direct guided_json path."""
        calls = []

        class FakeNvidiaLLMSpy:
            _provider = "nvidia"
            provider = "nvidia"
            model = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
            model_name = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"

            def with_structured_output(self, schema, include_raw=False, **kwargs):
                calls.append(kwargs)
                raise NotImplementedError("include_raw not supported")

            def bind(self, **kwargs):
                decision = RoutingDecision(
                    task_type=TaskType.GENERAL, requires_planning=False,
                    requires_knowledge=False, reasoning="ok",
                )
                return SimpleNamespace(invoke=lambda prompt: SimpleNamespace(content=decision.model_dump_json()))

        result = _invoke_single_provider(FakeNvidiaLLMSpy(), RoutingDecision, "prompt")

        assert result.task_type == TaskType.GENERAL
        # No method/strict kwargs ever passed for NVIDIA.
        assert all("method" not in c and "strict" not in c for c in calls)


class TestDeveloperPatchResponseViaGroqJsonSchema:
    def test_real_patch_response_schema_produces_filepatch_without_tool_call(self):
        """The actual schema used by developer_node's context-aware patch
        path (a nested list of FilePatch objects) must work through Groq's
        json_schema method - proving the fix covers the real production
        schema that triggered the original EC2 failure, not just a toy one."""
        from pydantic import BaseModel, Field

        class PatchResponse(BaseModel):
            patches: list[FilePatch] = Field(description="List of proposed file patches.")

        patch_response = PatchResponse(
            patches=[
                FilePatch(
                    file_path="README.md",
                    original_code_snippet="# agentic-ai-test-repo",
                    updated_code_snippet="# agentic-ai-test-repo\n\n## E2E Test\n",
                    explanation="Add E2E Test section",
                )
            ]
        )
        llm = FakeGroqStructuredLLM(parsed_result=patch_response, model_name="openai/gpt-oss-20b")

        result = _invoke_single_provider(llm, PatchResponse, "prompt")

        assert len(result.patches) == 1
        assert result.patches[0].file_path == "README.md"
        assert llm.calls[0]["method"] == "json_schema"
        assert llm.calls[0]["strict"] is True
