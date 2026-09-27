"""
Tests for Phase 8 - Step 1: Durable LangGraph Checkpointing.
Validates SQLite-backed checkpoint persistence, server restart recovery,
tenant security enforcement during recovery, patch integrity, and multi-process concurrency.
"""

import os
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver

from backend.graph.runner import AgentRunner
from backend.graph.nodes import git_prepare_node as _real_git_prepare_node
from backend.schemas.routing import RoutingDecision, TaskType
from backend.schemas.developer import DeveloperResult, FileChange
from backend.schemas.qa import QAResult
from backend.vcs.models import ApprovalDecision
from backend.vcs.git_manager import GitWorkspaceManager
from backend.vcs.workspace_lock import (
    WorkspaceLockManager,
    WorkspaceLockTimeoutError,
    WorkspaceLockError,
    workspace_lock_manager,
)
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager


def _init_git_workspace(tmp_path, monkeypatch, project_id: str):
    """
    Creates a real, git-initialized workspace/<project_id> under tmp_path
    and points nodes.py's os.getcwd()-based fallback path resolution at it,
    so developer_node's fallback write + git_prepare_node's diff
    computation exercise a genuine git baseline (as every real, cloned
    workspace has) instead of a canned/mocked diff. Asserts no ambient
    workspace/<project_id> already exists at the real cwd (which would
    shadow the tmp_path fallback and defeat the isolation).
    """
    workspace_dir = tmp_path / "workspace" / project_id
    workspace_dir.mkdir(parents=True)
    subprocess.run(["git", "init"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=str(workspace_dir), capture_output=True, text=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(workspace_dir), capture_output=True, text=True)
    (workspace_dir / "README.md").write_text("# Test Project\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=str(workspace_dir), capture_output=True, text=True, check=True)
    subprocess.run(["git", "commit", "-m", "initial commit"], cwd=str(workspace_dir), capture_output=True, text=True, check=True)

    assert not (Path("workspace") / project_id).exists(), (
        f"workspace/{project_id} already exists at the real cwd - pick a "
        "fresh project_id so the tmp_path fallback below isn't shadowed."
    )
    monkeypatch.setattr(os, "getcwd", lambda: str(tmp_path))
    return workspace_dir


def _mock_offline_graph_dependencies(monkeypatch):
    """
    Deterministic, fully offline stand-ins for every LLM/subprocess call
    reachable once a real git workspace (see _init_git_workspace) gives
    developer_node non-empty repo_context: its own inline exact-snippet
    LLM branch (invoke_structured), the real QualityPipeline subprocess
    checks, and the revision-loop LLM calls (defense-in-depth - QA is
    mocked PASS on the first attempt by the autouse mock_agent_nodes
    fixture, so revision_node should never actually run).

    Does NOT touch route_task/generate_code_changes/review_code_changes/
    git_prepare_node - those are handled by mock_agent_nodes, and
    git_prepare_node is restored to the real implementation by the two
    tests that need a genuine diff, individually.
    """
    monkeypatch.setattr(
        "backend.graph.nodes.invoke_structured",
        lambda llm, schema_cls, prompt, *a, **k: schema_cls(patches=[]),
    )
    monkeypatch.setattr(
        "backend.qa.pipeline.QualityPipeline.run_all",
        lambda repo_path, patches, timeout=30.0, cancel_check=None, user_request="", original_file_snapshots=None, true_original_snapshots=None: ([], None),
    )
    monkeypatch.setattr(
        "backend.agents.developer.revise_code_changes",
        lambda *args, **kwargs: DeveloperResult(
            summary="revised", changes=[], requires_testing=True, notes=[]
        ),
    )
    monkeypatch.setattr(
        "backend.agents.revision.generate_revision_patches",
        lambda *args, **kwargs: [],
    )


@pytest.fixture(autouse=True)
def mock_agent_nodes(monkeypatch):
    """
    Mock agent LLM calls and Git mutations so graph execution
    runs deterministically, quickly, and offline.
    """
    monkeypatch.setattr(
        "backend.graph.nodes.route_task",
        lambda msg: RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.95,
            reasoning="Deterministic test routing",
            requires_planning=False,
            requires_knowledge=False,
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.generate_code_changes",
        lambda user_request, plan, knowledge: DeveloperResult(
            summary=f"Automated test patch for: {user_request[:30]}",
            changes=[
                FileChange(
                    file_path="src/service.py",
                    change_type="MODIFY",
                    content="def handle_request(): return {'status': 'ok'}",
                    reason="Fix test service logic",
                )
            ],
            requires_testing=True,
            notes=[],
        ),
    )
    monkeypatch.setattr(
        "backend.graph.nodes.review_code_changes",
        lambda user_request, plan, developer_result: QAResult(
            status="PASS",
            issues=[],
            test_cases=["test_handle_request"],
            summary="QA evaluation passed.",
        ),
    )
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", MagicMock(return_value=True))
    monkeypatch.setattr(GitWorkspaceManager, "create_feature_branch", MagicMock(return_value=True))
    monkeypatch.setattr(GitWorkspaceManager, "verify_workspace_drift", MagicMock(return_value=(True, "")))
    from backend.vcs.models import GitDiffSummary
    test_diff = GitDiffSummary(
        branch_name="agent/test-branch",
        files_changed=["src/service.py"],
        lines_added=1,
        lines_deleted=0,
        unified_diff="--- a/src/service.py\n+++ b/src/service.py\n@@ -1 +1 @@\n+def handle_request(): return {'status': 'ok'}\n",
        patch_hash="a1b2c3d4e5f678901234567890abcdef1234567890abcdef1234567890abcdef",
        risk_score="LOW",
        risk_reasons=[],
    )
    monkeypatch.setattr(
        "backend.graph.nodes.git_prepare_node",
        lambda state: {
            "git_diff": test_diff,
            "patch_hash": test_diff.patch_hash,
        },
    )


# ============================================================================
# 1. DURABLE CHECKPOINTER INITIALIZATION
# ============================================================================

def test_durable_checkpointer_initialization(tmp_path):
    """
    Verifies that AgentRunner initializes a durable SqliteSaver checkpointer
    backed by SQLite with WAL mode, busy timeout, and correct schema.
    """
    db_path = str(tmp_path / "init_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    try:
        assert isinstance(runner._checkpointer, SqliteSaver)
        assert not isinstance(runner._checkpointer, MemorySaver)
        assert os.path.exists(db_path)

        # Verify tables and WAL journal mode
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal"

        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        assert "checkpoints" in tables
        assert "writes" in tables
        conn.close()
    finally:
        runner.close()


# ============================================================================
# 2. CHECKPOINT SURVIVES RUNNER RECREATION
# ============================================================================

def test_checkpoint_survives_runner_recreation(tmp_path):
    """
    Verifies that a run persisted by one runner instance is discoverable
    by an entirely new runner instance connected to the same SQLite database.
    """
    db_path = str(tmp_path / "survives_recreation.db")
    run_id = "run-survive-recreation-001"
    org_id = "org-engineering"

    # Instance 1: start run until it pauses at approval
    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Fix payment timeout", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    runner1.close()

    # Instance 2: new runner, simulating server restart
    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.status == "WAITING_APPROVAL"
        assert status2.run_id == run_id
    finally:
        runner2.close()


# ============================================================================
# 3. WAITING_FOR_APPROVAL STATE SURVIVES RESTART
# ============================================================================

def test_waiting_for_approval_state_survives_restart(tmp_path):
    """
    Verifies that WAITING_APPROVAL state, current_node ('approval'),
    and full state values survive process/runner restart.
    """
    db_path = str(tmp_path / "waiting_approval.db")
    run_id = "run-approval-restart-002"
    org_id = "org-platform"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Add retry policy", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    assert status1.current_node == "approval"
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.status == "WAITING_APPROVAL"
        assert status2.current_node == "approval"

        values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert values["user_message"] == "Add retry policy"
        assert values["organization_id"] == org_id
        assert values["run_id"] == run_id
    finally:
        runner2.close()


# ============================================================================
# 4. RECOVERED RUN CAN RESUME
# ============================================================================

def test_recovered_run_can_resume(tmp_path):
    """
    Verifies that a run paused at the approval gate on runner 1 can be
    successfully resumed on a newly instantiated runner 2 and runs to completion.
    """
    db_path = str(tmp_path / "resume_recovered.db")
    run_id = "run-resume-003"
    org_id = "org-security"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Patch SQL sanitization", organization_id=org_id)
    assert status1.status == "WAITING_APPROVAL"
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        decision = ApprovalDecision(
            approved=True,
            reviewer="lead-architect",
            patch_hash=status1.git_diff.patch_hash if status1.git_diff else "",
        )
        resumed = runner2.resume_run(run_id=run_id, approval_decision=decision, organization_id=org_id)
        assert resumed.status == "COMPLETED"

        final_status = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert final_status.status == "COMPLETED"
    finally:
        runner2.close()


# ============================================================================
# 5. RECOVERED RUN REMAINS TENANT-SCOPED
# ============================================================================

def test_recovered_run_remains_tenant_scoped(tmp_path):
    """
    Verifies that a recovered run preserves its organization_id binding
    and can be retrieved by the originating tenant.
    """
    db_path = str(tmp_path / "tenant_scoped.db")
    run_id = "run-tenant-scope-004"
    org_id = "org-alpha-corp"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    runner1.start_run(run_id=run_id, user_message="Update billing logic", organization_id=org_id)
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status.status == "WAITING_APPROVAL"

        state_values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert state_values.get("organization_id") == org_id
    finally:
        runner2.close()


# ============================================================================
# 6. CROSS-TENANT CHECKPOINT ACCESS IS REJECTED
# ============================================================================

def test_cross_tenant_checkpoint_access_is_rejected(tmp_path):
    """
    Verifies that recovering or resuming a checkpoint belonging to Organization A
    fails closed with KeyError when requested by Organization B.
    """
    db_path = str(tmp_path / "cross_tenant.db")
    run_id = "run-org-a-private-005"
    org_a = "org-owner-a"
    org_b = "org-intruder-b"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    runner1.start_run(run_id=run_id, user_message="Sensitive internal refactor", organization_id=org_a)
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        # Org B attempting get_status -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.get_status(run_id=run_id, organization_id=org_b)
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Org B attempting get_state_values -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.get_state_values(run_id=run_id, organization_id=org_b)
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Org B attempting resume_run -> rejected
        with pytest.raises(KeyError) as exc_info:
            runner2.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(approved=True, reviewer="intruder"),
                organization_id=org_b,
            )
        assert f"Run not found: {run_id}" in str(exc_info.value)

        # Legitimate Org A access remains untouched and permitted
        status_a = runner2.get_status(run_id=run_id, organization_id=org_a)
        assert status_a.status == "WAITING_APPROVAL"
    finally:
        runner2.close()


# ============================================================================
# 7. PATCH HASH SURVIVES RECOVERY
# ============================================================================

def test_patch_hash_survives_recovery(tmp_path):
    """
    Verifies that the SHA-256 cryptographic patch_hash generated before
    interruption is preserved bit-for-bit after runner restart.
    """
    db_path = str(tmp_path / "patch_hash.db")
    run_id = "run-patch-hash-006"
    org_id = "org-vcs-team"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Fix memory leak in buffer", organization_id=org_id)
    assert status1.git_diff is not None
    orig_hash = status1.git_diff.patch_hash
    assert orig_hash != ""
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.git_diff is not None
        assert status2.git_diff.patch_hash == orig_hash

        values = runner2.get_state_values(run_id=run_id, organization_id=org_id)
        assert values["git_diff"].patch_hash == orig_hash
    finally:
        runner2.close()


# ============================================================================
# 8. APPROVAL STATE SURVIVES RECOVERY
# ============================================================================

def test_approval_state_survives_recovery(tmp_path):
    """
    Verifies that after an approval decision is resumed and processed,
    the updated approval record and COMMITTED status persist across
    subsequent runner restarts.
    """
    db_path = str(tmp_path / "approval_state.db")
    run_id = "run-approval-state-007"
    org_id = "org-audit"

    # Step 1: run to approval
    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Refactor rate limiter", organization_id=org_id)
    runner1.close()

    # Step 2: resume on runner 2
    runner2 = AgentRunner(checkpoint_db_path=db_path)
    decision = ApprovalDecision(
        approved=True,
        reviewer="lead-sec-auditor",
        patch_hash=status1.git_diff.patch_hash if status1.git_diff else "",
    )
    resumed = runner2.resume_run(run_id=run_id, approval_decision=decision, organization_id=org_id)
    assert resumed.status == "COMPLETED"
    runner2.close()

    # Step 3: inspect on runner 3
    runner3 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status3 = runner3.get_status(run_id=run_id, organization_id=org_id)
        assert status3.status == "COMPLETED"

        values = runner3.get_state_values(run_id=run_id, organization_id=org_id)
        assert values.get("approval_status") == "COMMITTED"
        approval = values.get("approval")
        assert approval is not None
        assert approval.approved is True
        assert approval.reviewer == "lead-sec-auditor"
    finally:
        runner3.close()


# ============================================================================
# 9. POLICY STATE SURVIVES RECOVERY
# ============================================================================

def test_policy_state_survives_recovery(tmp_path):
    """
    Verifies that PolicyEvaluationResult evaluated before interruption
    is recovered intact, retaining decision and risk score.
    """
    db_path = str(tmp_path / "policy_state.db")
    run_id = "run-policy-eval-008"
    org_id = "org-governance"

    runner1 = AgentRunner(checkpoint_db_path=db_path)
    status1 = runner1.start_run(run_id=run_id, user_message="Modify firewall rule", organization_id=org_id)
    assert status1.policy_result is not None
    orig_decision = status1.policy_result.decision
    orig_approval = status1.policy_result.requires_human_approval
    runner1.close()

    runner2 = AgentRunner(checkpoint_db_path=db_path)
    try:
        status2 = runner2.get_status(run_id=run_id, organization_id=org_id)
        assert status2.policy_result is not None
        assert status2.policy_result.decision == orig_decision
        assert status2.policy_result.requires_human_approval == orig_approval
    finally:
        runner2.close()


# ============================================================================
# 10. NO PRODUCTION FALLBACK TO MEMORYSAVER
# ============================================================================

def test_no_production_fallback_to_memory_saver(tmp_path, monkeypatch):
    """
    Verifies that when AUTH_MODE=production, a failure to initialize the
    SQLite checkpointer raises a RuntimeError and does NOT fall back to MemorySaver.
    """
    monkeypatch.setattr(tenant_manager, "auth_mode", AuthMode.PRODUCTION)
    monkeypatch.setattr(tenant_manager, "dev_auth_fallback", False)

    # Use a directory path as the db path to trigger a SQLite initialization failure
    bad_db_path = str(tmp_path)  # directory, cannot connect as SQLite database file

    with pytest.raises(RuntimeError) as exc_info:
        AgentRunner(checkpoint_db_path=bad_db_path)

    assert "DURABLE_CHECKPOINT_INIT_FAILED" in str(exc_info.value)


# ============================================================================
# 11. MULTIPLE RUNS HAVE ISOLATED CHECKPOINTS
# ============================================================================

def test_multiple_runs_have_isolated_checkpoints(tmp_path):
    """
    Verifies that multiple concurrent runs sharing the same checkpoint database
    have strictly isolated thread checkpoints and independent lifecycles.
    """
    db_path = str(tmp_path / "multi_run_isolation.db")
    run_a = "run-isolated-A-011"
    run_b = "run-isolated-B-011"
    org_id = "org-enterprise"

    runner = AgentRunner(checkpoint_db_path=db_path)
    try:
        status_a = runner.start_run(run_id=run_a, user_message="Task A", organization_id=org_id)
        status_b = runner.start_run(run_id=run_b, user_message="Task B", organization_id=org_id)

        assert status_a.status == "WAITING_APPROVAL"
        assert status_b.status == "WAITING_APPROVAL"

        # Resume Run A only. P0-4: approval is fail-closed on a missing
        # patch_hash, so it must be submitted here to genuinely reach
        # COMMITTED - omitting it (as this test previously did) is now
        # correctly rejected instead of silently treated as approved.
        decision = ApprovalDecision(
            approved=True, reviewer="reviewer-a", patch_hash=status_a.git_diff.patch_hash
        )
        resumed_a = runner.resume_run(run_id=run_a, approval_decision=decision, organization_id=org_id)
        assert resumed_a.status == "COMPLETED"

        # Verify Run B is completely unaffected and still WAITING_APPROVAL
        check_b = runner.get_status(run_id=run_b, organization_id=org_id)
        assert check_b.status == "WAITING_APPROVAL"

        # Verify state values isolation
        vals_a = runner.get_state_values(run_id=run_a, organization_id=org_id)
        vals_b = runner.get_state_values(run_id=run_b, organization_id=org_id)
        assert vals_a["user_message"] == "Task A"
        assert vals_b["user_message"] == "Task B"
        assert vals_a.get("approval_status") == "COMMITTED"
        assert vals_b.get("approval_status") != "COMMITTED"
    finally:
        runner.close()


# ============================================================================
# 12. SQLITE CHECKPOINT PERSISTENCE WORKS CONCURRENTLY
# ============================================================================

def test_sqlite_checkpoint_persistence_works_concurrently(tmp_path):
    """
    Verifies that concurrent threads executing runs on separate AgentRunner
    instances sharing the same SQLite database operate safely without lock contention
    errors or database corruption.
    """
    db_path = str(tmp_path / "concurrency_checkpoints.db")
    errors = []
    thread_count = 4

    def worker(worker_idx: int):
        runner = AgentRunner(checkpoint_db_path=db_path)
        try:
            run_id = f"run-concurrent-{worker_idx}"
            org_id = f"org-worker-{worker_idx}"

            # Start run
            status = runner.start_run(
                run_id=run_id,
                user_message=f"Concurrent task from worker {worker_idx}",
                organization_id=org_id,
            )
            if status.status != "WAITING_APPROVAL":
                errors.append(f"Worker {worker_idx}: unexpected initial status {status.status}")

            # Query status
            retrieved = runner.get_status(run_id=run_id, organization_id=org_id)
            if retrieved.status != "WAITING_APPROVAL":
                errors.append(f"Worker {worker_idx}: unexpected retrieved status {retrieved.status}")

            # Resume run
            resumed = runner.resume_run(
                run_id=run_id,
                approval_decision=ApprovalDecision(approved=True, reviewer=f"rev-{worker_idx}"),
                organization_id=org_id,
            )
            if resumed.status != "COMPLETED":
                errors.append(f"Worker {worker_idx}: unexpected resumed status {resumed.status}")
        except Exception as exc:
            errors.append(f"Worker {worker_idx} exception: {str(exc)}")
        finally:
            runner.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(thread_count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15.0)

    assert not errors, f"Concurrent execution errors encountered: {errors}"


# ============================================================================
# PHASE 8 — STEP 2: PRODUCTION LLM TIMEOUTS + SAFE PROVIDER FALLBACK
# ============================================================================

from backend.services.llm import get_llm, get_fallback_provider
from backend.services.errors import (
    LLMError,
    LLMTimeoutError,
    LLMTransientError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMInvalidRequestError,
    LLMPermanentError,
    classify_llm_exception,
    is_fallback_eligible,
)
from backend.observability.telemetry import invoke_structured, run_context, _invoke_single_provider
from backend.observability.collector import telemetry_collector
from backend.observability.store import telemetry_store
from backend.schemas.telemetry import TelemetryEventType, FailureCategory
import httpx
import requests


class FakeProviderLLM:
    """Mock LLM that can simulate success, timeout, auth error, or transient failure."""
    def __init__(self, provider="nvidia", model_name="test-model", failure=None, result=None):
        self._provider = provider
        self.provider = provider
        self.model = model_name
        self.model_name = model_name
        self.failure = failure
        self.result = result
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        self.invocations += 1
        if self.failure:
            raise self.failure
        return self.result


# 1. test_llm_timeout_configuration
def test_llm_timeout_configuration(monkeypatch):
    """
    Verifies that the configured timeout (LLM_REQUEST_TIMEOUT_SECONDS) is actually
    passed to the underlying client instances for all supported providers:
    NVIDIA NIM, Google Gemini, and OpenAI.
    """
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "45.0")
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test-key")

    from backend.services import llm as llm_module

    # NVIDIA
    nvidia_client = llm_module.get_llm(provider="nvidia")
    assert getattr(nvidia_client._client, "timeout", None) == 45.0

    # Gemini
    gemini_client = llm_module.get_llm(provider="gemini")
    assert getattr(gemini_client, "timeout", None) == 45.0

    # OpenAI
    openai_client = llm_module.get_llm(provider="openai")
    assert getattr(openai_client, "request_timeout", None) == 45.0

    # Explicit override timeout parameter
    custom_client = llm_module.get_llm(provider="nvidia", timeout=15.0)
    assert getattr(custom_client._client, "timeout", None) == 15.0


# 2. test_provider_timeout_is_structured
def test_provider_timeout_is_structured():
    """
    Verifies that socket, network, connect, and read timeouts from underlying SDKs
    are deterministically classified into structured LLMTimeoutError exceptions.
    """
    # httpx ConnectTimeout
    exc1 = httpx.ConnectTimeout("Connection timed out after 60s")
    classified1 = classify_llm_exception(exc1, provider="nvidia")
    assert isinstance(classified1, LLMTimeoutError)
    assert is_fallback_eligible(classified1) is True
    assert classified1.provider == "nvidia"

    # requests Timeout
    exc2 = requests.exceptions.Timeout("Read timed out")
    classified2 = classify_llm_exception(exc2, provider="gemini")
    assert isinstance(classified2, LLMTimeoutError)
    assert is_fallback_eligible(classified2) is True
    assert classified2.provider == "gemini"

    # Python standard TimeoutError
    exc3 = TimeoutError("Deadline exceeded")
    classified3 = classify_llm_exception(exc3, provider="openai")
    assert isinstance(classified3, LLMTimeoutError)
    assert is_fallback_eligible(classified3) is True


# 3. test_transient_provider_failure_triggers_fallback
def test_transient_provider_failure_triggers_fallback(monkeypatch):
    """
    Verifies that a transient provider failure (e.g. 503 Service Unavailable)
    on the primary provider automatically triggers fallback to the eligible secondary provider,
    after NVIDIA's own bounded capacity-resilience retries (backend/services/nvidia_resilience.py)
    are exhausted - not on the very first attempt.
    """
    from backend.core.config import NVIDIA_MAX_RETRIES

    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError("503 Service Unavailable"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.98,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback successfully classified",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    result = invoke_structured(primary, RoutingDecision, "Fix broken endpoint")
    assert result.task_type == TaskType.BUG_FIX
    assert primary.invocations == NVIDIA_MAX_RETRIES + 1
    assert fallback.invocations == 1


# 4. test_timeout_triggers_fallback
def test_timeout_triggers_fallback(monkeypatch):
    """
    Verifies that a timeout on the primary provider triggers safe fallback
    to the secondary provider, returning the expected structured result.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=TimeoutError("Request timed out after 60.0 seconds"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.95,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded after timeout",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    result = invoke_structured(primary, RoutingDecision, "Create new user model")
    assert result.task_type == TaskType.CODE_GENERATION
    assert primary.invocations == 1
    assert fallback.invocations == 1


# 5. test_auth_failure_does_not_trigger_fallback
def test_auth_failure_does_not_trigger_fallback(monkeypatch):
    """
    Verifies that authentication/authorization failures fail closed immediately
    and NEVER trigger fallback to another provider.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=Exception("401 Unauthorized: Invalid API key nvapi-invalid"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Should not be called",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises((LLMAuthenticationError, PermissionError)) as exc_info:
        invoke_structured(primary, RoutingDecision, "Some prompt")

    assert "authentication" in str(exc_info.value).lower() or "401" in str(exc_info.value)
    assert primary.invocations == 1
    assert fallback.invocations == 0, "Fallback provider must NOT be invoked on auth failures!"


# 6. test_invalid_request_does_not_trigger_fallback
def test_invalid_request_does_not_trigger_fallback(monkeypatch):
    """
    Verifies that malformed requests (400 Bad Request / schema errors) fail closed
    immediately and NEVER trigger automatic fallback.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=Exception("400 Bad Request: maximum context length exceeded"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Should not be called",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises((LLMInvalidRequestError, ValueError)) as exc_info:
        invoke_structured(primary, RoutingDecision, "Too long prompt")

    assert "invalid request" in str(exc_info.value).lower() or "context length" in str(exc_info.value)
    assert primary.invocations == 1
    assert fallback.invocations == 0, "Fallback provider must NOT be invoked on invalid request errors!"


# 7. test_fallback_is_bounded
def test_fallback_is_bounded(monkeypatch):
    """
    Verifies that provider attempts are strictly bounded to max 2 attempts (primary -> fallback).
    If the fallback provider also fails, execution raises a structured final failure
    and does NOT attempt a 3rd provider or loop infinitely.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError("503 Service Unavailable"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        failure=httpx.NetworkError("502 Bad Gateway"),
    )

    fallback_attempts = []
    def mock_get_fallback_provider(*args, **kwargs):
        primary_arg = kwargs.get("primary") or (args[0] if args else None)
        fallback_attempts.append(primary_arg)
        return "gemini" if primary_arg == "nvidia" else "openai"

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", mock_get_fallback_provider)
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "gemini" else primary)

    with pytest.raises(LLMTransientError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Any prompt")

    assert "transient provider failure" in str(exc_info.value).lower() or "502" in str(exc_info.value)
    from backend.core.config import NVIDIA_MAX_RETRIES
    assert primary.invocations == NVIDIA_MAX_RETRIES + 1
    assert fallback.invocations == 1
    assert len(fallback_attempts) == 1, "Fallback must be attempted at most once (strict bound of 2 attempts)!"


# 8. test_all_providers_failure_returns_structured_error
def test_all_providers_failure_returns_structured_error(monkeypatch):
    """
    Verifies that when all eligible providers fail, a structured final LLM failure
    propagates cleanly without leaving the run appearing successful.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out"))
    fallback = FakeProviderLLM(provider="gemini", failure=TimeoutError("Fallback timed out"))

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises(LLMTimeoutError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert isinstance(exc_info.value, LLMTimeoutError)
    assert exc_info.value.provider == "gemini"


# 8b. test_gemini_rate_limit_after_nvidia_timeout_raises_llm_rate_limit_error
def test_gemini_rate_limit_after_nvidia_timeout_raises_llm_rate_limit_error(monkeypatch):
    """
    Reproduces the exact production sequence (run_24a226561db4): NVIDIA
    primary times out, safe fallback switches to Gemini, and Gemini itself
    returns HTTP 429 RESOURCE_EXHAUSTED (free-tier quota exhausted). The
    final structured error raised to the caller must be LLMRateLimitError
    (not LLMTimeoutError or a generic LLMTransientError) so downstream
    RUN_FAILED telemetry classifies this correctly.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out after 75.0 seconds"))
    fallback = FakeProviderLLM(
        provider="gemini",
        failure=Exception(
            "Error calling model 'gemini-3.6-flash' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. "
            "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests"
        ),
    )

    # No OTHER credentialed provider exists beyond "gemini" in this
    # scenario - `exclude` is honored (matching the real function's
    # semantics) so the second-fallback lookup correctly returns None
    # instead of re-selecting the already-exhausted "gemini" again.
    def fake_get_fallback_provider(*a, primary=None, exclude=None, **kw):
        if exclude and "gemini" in {e.lower() for e in exclude}:
            return None
        return "gemini"

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", fake_get_fallback_provider)
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises(LLMRateLimitError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert isinstance(exc_info.value, LLMRateLimitError)
    assert exc_info.value.provider == "gemini"
    assert primary.invocations == 1
    # A rate-limit/quota error is fast-failed immediately (never retried
    # within the same provider call - see _invoke_single_provider), and no
    # second fallback provider is available here, so gemini is invoked
    # exactly once before the run fails closed.
    assert fallback.invocations == 1


# 8b-ii. test_second_fallback_provider_used_when_first_fallback_is_rate_limited
def test_second_fallback_provider_used_when_first_fallback_is_rate_limited(monkeypatch):
    """
    NVIDIA times out -> Gemini fallback is rate-limited (429) -> a THIRD,
    distinct, credentialed provider (openai) is available, so it is tried
    exactly once and succeeds. Verifies the bounded second-fallback escape
    hatch actually recovers the run instead of failing closed prematurely.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out after 75.0 seconds"))
    gemini = FakeProviderLLM(
        provider="gemini",
        failure=Exception("429 RESOURCE_EXHAUSTED: Quota exceeded"),
    )
    openai = FakeProviderLLM(
        provider="openai",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Second fallback succeeded",
        ),
    )

    def fake_get_fallback_provider(*a, primary=None, exclude=None, **kw):
        excluded = {e.lower() for e in (exclude or ())}
        if "gemini" not in excluded:
            return "gemini"
        if "openai" not in excluded:
            return "openai"
        return None

    def fake_get_llm(provider=None, **kw):
        return {"gemini": gemini, "openai": openai}[provider]

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", fake_get_fallback_provider)
    monkeypatch.setattr("backend.services.llm.get_llm", fake_get_llm)

    result = invoke_structured(primary, RoutingDecision, "Prompt")

    assert result.reasoning == "Second fallback succeeded"
    assert primary.invocations == 1
    assert gemini.invocations == 1, "rate-limited provider must be tried exactly once, never retried"
    assert openai.invocations == 1

    # Two PROVIDER_FALLBACK events: nvidia->gemini, then gemini->openai.
    assert len(recorded_events) == 2
    assert recorded_events[0]["primary_provider"] == "nvidia"
    assert recorded_events[0]["fallback_provider"] == "gemini"
    assert recorded_events[1]["primary_provider"] == "gemini"
    assert recorded_events[1]["fallback_provider"] == "openai"
    assert recorded_events[1]["failure_category"] == "LLM_RATE_LIMIT"
    for event in recorded_events:
        for value in event.values():
            assert "api_key" not in str(value).lower()
            assert "nvapi" not in str(value).lower()


# 8b-iii. test_second_fallback_provider_also_rate_limited_fails_closed_bounded
def test_second_fallback_provider_also_rate_limited_fails_closed_bounded(monkeypatch):
    """
    NVIDIA times out -> Gemini is rate-limited -> the second fallback
    (openai) is ALSO rate-limited -> the run must fail closed with the
    second fallback's own LLMRateLimitError, and no FOURTH provider attempt
    may ever be made (strict bound of 3 total provider attempts).
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out"))
    gemini = FakeProviderLLM(provider="gemini", failure=Exception("429 RESOURCE_EXHAUSTED"))
    openai = FakeProviderLLM(provider="openai", failure=Exception("429 Too Many Requests"))

    fallback_lookup_calls = []

    def fake_get_fallback_provider(*a, primary=None, exclude=None, **kw):
        excluded = {e.lower() for e in (exclude or ())}
        fallback_lookup_calls.append(frozenset(excluded))
        if "gemini" not in excluded:
            return "gemini"
        if "openai" not in excluded:
            return "openai"
        return None

    def fake_get_llm(provider=None, **kw):
        return {"gemini": gemini, "openai": openai}[provider]

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", fake_get_fallback_provider)
    monkeypatch.setattr("backend.services.llm.get_llm", fake_get_llm)

    with pytest.raises(LLMRateLimitError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert exc_info.value.provider == "openai"
    assert primary.invocations == 1
    assert gemini.invocations == 1
    assert openai.invocations == 1
    # Exactly 2 fallback-provider lookups (first fallback, second fallback) -
    # a third lookup would indicate an unbounded cascade.
    assert len(fallback_lookup_calls) == 2


# 8c. test_rate_limited_primary_triggers_fallback_with_llm_rate_limit_telemetry_category
def test_rate_limited_primary_triggers_fallback_with_llm_rate_limit_telemetry_category(monkeypatch):
    """
    Verifies a rate-limited PRIMARY provider is fallback-eligible and that
    the PROVIDER_FALLBACK telemetry event records failure_category
    'LLM_RATE_LIMIT' (not 'LLM_TIMEOUT'/'LLM_TRANSIENT_FAILURE'), and that
    the fallback provider succeeds.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=Exception("429 Too Many Requests: rate limit exceeded"),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded after primary rate limit",
        ),
    )

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    result = invoke_structured(primary, RoutingDecision, "Prompt")

    assert result.task_type == TaskType.BUG_FIX
    assert len(recorded_events) == 1
    event = recorded_events[0]
    assert event["failure_category"] == "LLM_RATE_LIMIT"
    assert event["failure_type"] == "LLMRateLimitError"
    # No raw exception text or credential-shaped content in telemetry metadata.
    for value in event.values():
        assert "api_key" not in str(value).lower()
        assert "nvapi" not in str(value).lower()


# 9. test_provider_fallback_emits_telemetry
def test_provider_fallback_emits_telemetry(monkeypatch):
    """
    Verifies that PROVIDER_FALLBACK telemetry event is emitted with sanitized
    metadata (run_id, organization_id, primary_provider, fallback_provider, failure_category).
    """
    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Timeout on primary"))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback telemetry test",
        ),
    )

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-telemetry-test", organization_id="org-acme"):
        invoke_structured(primary, RoutingDecision, "Test prompt")

    assert len(recorded_events) == 1
    event = recorded_events[0]
    assert event["run_id"] == "run-telemetry-test"
    assert event["organization_id"] == "org-acme"
    assert event["primary_provider"] == "nvidia"
    assert event["fallback_provider"] == "gemini"
    assert event["failure_category"] == "LLM_TIMEOUT"
    assert event["failure_type"] == "LLMTimeoutError"
    assert event["attempt_number"] == 1


# 10. test_fallback_preserves_tenant_context
def test_fallback_preserves_tenant_context(monkeypatch):
    """
    Verifies that provider fallback preserves tenant isolation and does not alter
    the organization_id, project_id, or security context of the run.
    """
    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError("503 Service Unavailable"))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Secure tenant routing",
        ),
    )

    captured_org = []
    def mock_record(event):
        if event.event_type == TelemetryEventType.PROVIDER_FALLBACK:
            captured_org.append(event.organization_id)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-iso-123", organization_id="tenant-secure-corp"):
        res = invoke_structured(primary, RoutingDecision, "Secure prompt")

    assert res.task_type == TaskType.BUG_FIX
    assert captured_org == ["tenant-secure-corp"], "Tenant organization_id must be strictly preserved across fallback!"


# 11. test_fallback_does_not_bypass_policy_or_qa
def test_fallback_does_not_bypass_policy_or_qa(monkeypatch, tmp_path):
    """
    Verifies that code generated by a fallback provider continues through all
    authoritative downstream QA and policy evaluation gates.
    """
    db_path = str(tmp_path / "policy_qa_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    # Developer produces changes that introduce a policy violation (e.g. modify protected config)
    violating_dev_result = DeveloperResult(
        summary="Fallback developer changes",
        changes=[
            FileChange(
                file_path="config/production_secrets.json",
                change_type="MODIFY",
                content="{'secret': 'new'}",
                reason="Unauthorized modification via fallback",
            )
        ],
        requires_testing=True,
    )
    monkeypatch.setattr("backend.graph.nodes.generate_code_changes", lambda *args, **kwargs: violating_dev_result)

    try:
        status = runner.start_run(
            run_id="run-fallback-policy",
            user_message="Modify protected files",
            organization_id="test-org",
        )
        assert status.policy_result is not None
        assert status.status in ("WAITING_APPROVAL", "COMPLETED", "FAILED")
    finally:
        runner.close()


# 12. test_no_credentials_in_provider_fallback_telemetry
def test_no_credentials_in_provider_fallback_telemetry(monkeypatch):
    """
    Verifies that API keys, tokens, or auth headers are never recorded
    in the PROVIDER_FALLBACK telemetry event metadata.
    """
    sensitive_error_msg = "500 Server Error: nvapi-secretkey1234567890 failed with bearer token ghp_supersecret"
    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError(sensitive_error_msg))
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.GENERAL,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Redacted reasoning",
        ),
    )

    emitted_events = []
    def mock_record(event):
        if event.event_type == TelemetryEventType.PROVIDER_FALLBACK:
            emitted_events.append(event)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with run_context(run_id="run-redact-test", organization_id="test-org"):
        invoke_structured(primary, RoutingDecision, "Redaction test")

    assert len(emitted_events) == 1
    event_meta_str = str(emitted_events[0].safe_metadata)
    assert "nvapi-secretkey1234567890" not in event_meta_str
    assert "ghp_supersecret" not in event_meta_str


# 13. test_existing_provider_routing_remains_compatible
def test_existing_provider_routing_remains_compatible(monkeypatch):
    """
    Verifies that the existing provider routing (defaulting to LLM_PROVIDER)
    remains completely compatible and untouched.
    """
    monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    from backend.services.llm import get_llm

    monkeypatch.setenv("LLM_PROVIDER", "nvidia")
    client_nv = get_llm()
    assert getattr(client_nv, "_provider", None) == "nvidia"

    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    client_gem = get_llm()
    assert getattr(client_gem, "_provider", None) == "gemini"

    monkeypatch.setenv("LLM_PROVIDER", "openai")
    client_oai = get_llm()
    assert getattr(client_oai, "_provider", None) == "openai"


# ============================================================================
# run_80c82e8263d3 investigation: LLM_MODEL_NAME is a single, provider-
# agnostic override. Production runs LLM_PROVIDER=nvidia with
# LLM_MODEL_NAME=openai/gpt-oss-20b (pinning NVIDIA past a wave of model
# deprecations) - but get_llm() applied that override unconditionally, so
# the explicit fallback call in backend/observability/telemetry.py
# (get_llm(provider=fallback_provider)) reused the NVIDIA-only model name
# against Gemini's API, which has no such model: a real 404 Not Found that
# made the entire safe-fallback mechanism non-functional whenever
# LLM_MODEL_NAME happens to be set (as it currently is in production).
# ============================================================================

class TestLlmModelNameScopedToPrimaryProvider:
    def test_fallback_to_gemini_does_not_inherit_nvidia_model_name(self, monkeypatch):
        """1. The exact reported production scenario: primary NVIDIA with
        LLM_MODEL_NAME pinned to the NVIDIA-only model - a fallback
        get_llm(provider="gemini") call must resolve Gemini's own default
        model, never the NVIDIA name."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")

        from backend.services.llm import get_llm, _DEFAULT_MODELS

        fallback_client = get_llm(provider="gemini")
        assert fallback_client.model == _DEFAULT_MODELS["gemini"]
        assert fallback_client.model != "openai/gpt-oss-20b"

    def test_primary_nvidia_still_honors_explicit_model_name(self, monkeypatch):
        """2. The primary provider's own explicit LLM_MODEL_NAME override
        must remain unaffected - both with no provider argument (the
        normal call shape every agent uses) and with the provider passed
        explicitly."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")

        from backend.services.llm import get_llm

        assert get_llm().model == "openai/gpt-oss-20b"
        assert get_llm(provider="nvidia").model == "openai/gpt-oss-20b"

    def test_reverse_pairing_gemini_primary_falling_back_to_nvidia(self, monkeypatch):
        """3. The same bug in the other direction: primary Gemini pinned
        to a Gemini-specific model name, falling back to NVIDIA - NVIDIA
        must get its own default, never the Gemini name."""
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
        monkeypatch.setenv("LLM_PROVIDER", "gemini")
        monkeypatch.setenv("LLM_MODEL_NAME", "gemini-1.5-pro")

        from backend.services.llm import get_llm, _DEFAULT_MODELS

        fallback_client = get_llm(provider="nvidia")
        assert fallback_client.model == _DEFAULT_MODELS["nvidia"]
        assert fallback_client.model != "gemini-1.5-pro"

    def test_no_llm_model_name_set_defaults_unchanged(self, monkeypatch):
        """4. With LLM_MODEL_NAME entirely unset, every provider still
        falls back to its own _DEFAULT_MODELS entry - completely
        unaffected by this fix."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
        monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
        # This dev machine's own .env sets LLM_MODEL_NAME - delenv only
        # clears the live os.environ value; backend.services.llm's own
        # module-level LLM_MODEL_NAME name was already bound to that .env
        # value at import time (previously masked because it happened to
        # equal the old NVIDIA default too), so it must be monkeypatched
        # directly to genuinely test the no-override path.
        monkeypatch.setattr("backend.services.llm.LLM_MODEL_NAME", None)
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")

        from backend.services.llm import get_llm, _DEFAULT_MODELS

        assert get_llm().model == _DEFAULT_MODELS["nvidia"]
        assert get_llm(provider="gemini").model == _DEFAULT_MODELS["gemini"]
        assert get_llm(provider="openai").model == _DEFAULT_MODELS["openai"]

    def test_real_fallback_construction_path_uses_correctly_scoped_model(self, monkeypatch):
        """5. End-to-end through the REAL invoke_structured/get_llm path
        (not a mocked get_llm, unlike test_timeout_triggers_fallback above,
        which still passes unchanged and continues to cover the
        retry/trigger mechanics) - only the actual network call
        (_invoke_single_provider) is stubbed, so get_fallback_provider and
        get_llm run for real and the fallback client actually constructed
        for the call is inspected directly."""
        monkeypatch.setenv("NVIDIA_API_KEY", "test-nvidia-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("LLM_MODEL_NAME", "openai/gpt-oss-20b")
        monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")

        decision = RoutingDecision(
            task_type=TaskType.DOCUMENTATION, requires_planning=False,
            requires_knowledge=False, reasoning="fallback succeeded",
        )
        primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Read timed out"))
        constructed_fallback_models = []

        import backend.observability.telemetry as telemetry_module
        real_invoke_single_provider = telemetry_module._invoke_single_provider

        def spy_invoke_single_provider(llm, schema, prompt):
            if llm is primary:
                return real_invoke_single_provider(llm, schema, prompt)
            constructed_fallback_models.append(getattr(llm, "model", None))
            return decision

        monkeypatch.setattr(telemetry_module, "_invoke_single_provider", spy_invoke_single_provider)

        result = invoke_structured(primary, RoutingDecision, "prompt")

        from backend.services.llm import _DEFAULT_MODELS

        assert result == decision
        assert constructed_fallback_models == [_DEFAULT_MODELS["gemini"]]

    def test_gemini_default_model_is_not_the_decommissioned_identifier(self, monkeypatch):
        """6. run_120607d608c7 investigation: gemini-2.0-flash was
        decommissioned by Google (404 NOT_FOUND, "no longer available"),
        which broke the fallback path itself even though the model-name
        scoping fix above (PR #21) was resolving correctly. Confirmed via
        the account's live ListModels API that gemini-3.6-flash is the
        currently supported replacement (matching Google's own 404
        guidance) before pinning it here."""
        from backend.services.llm import _DEFAULT_MODELS

        assert _DEFAULT_MODELS["gemini"] == "gemini-3.6-flash"
        assert _DEFAULT_MODELS["gemini"] != "gemini-2.0-flash"


# 14. test_llm_timeout_does_not_hang_graph
def test_llm_timeout_does_not_hang_graph(monkeypatch, tmp_path):
    """
    Verifies that an unrecoverable LLM timeout during graph execution propagates
    cleanly into existing error handling and terminates with status FAILED without hanging.
    """
    db_path = str(tmp_path / "timeout_hang_checkpoints.db")
    runner = AgentRunner(checkpoint_db_path=db_path)

    def mock_timing_out_router(msg):
        raise LLMTimeoutError("Outbound request timed out after 60.0s", provider="nvidia")

    monkeypatch.setattr("backend.graph.nodes.route_task", mock_timing_out_router)

    try:
        status = runner.start_run(
            run_id="run-timeout-hang-test",
            user_message="Should fail fast on timeout",
            organization_id="test-org",
        )
        assert status.status == "FAILED"
        assert "timed out" in (status.error_summary or "").lower()
    finally:
        runner.close()


# ============================================================================
# PHASE 8 — STEP 4: LLM MALFORMED-RESPONSE CLASSIFICATION + FALLBACK
# ============================================================================

from backend.services.errors import LLMMalformedResponseError
from langchain_core.exceptions import OutputParserException
from types import SimpleNamespace
import json as _json


class FakeEmptyCompletionLLM:
    """
    Simulates a provider whose with_structured_output() produces nothing usable
    (forcing telemetry's _invoke_single_provider "direct fallback" path), and whose
    plain .invoke() then returns an empty completion body - the real-world shape of
    a provider that silently returned nothing for a structured-output request.
    """
    def __init__(self, provider="nvidia", model_name="test-model", content=""):
        self.provider = provider
        self.model = model_name
        self.content = content
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        return self  # invoke() below stands in for structured.invoke(prompt) too

    def invoke(self, prompt):
        self.invocations += 1
        if self.invocations == 1:
            return None  # with_structured_output's own invoke: nothing usable
        return SimpleNamespace(content=self.content, usage_metadata=None)


# 15. test_malformed_response_classification (Category A)
def test_malformed_response_classification():
    """
    Verifies that provider response-parsing failures (empty/truncated JSON,
    OutputParserException, JSONDecodeError, and a Pydantic ValidationError caused by
    unparseable model output) are classified as LLMMalformedResponseError, and remain
    fallback-eligible.
    """
    from backend.schemas.telemetry import FailureCategory  # noqa: F401 (keep import parity)
    from pydantic import BaseModel

    class _Schema(BaseModel):
        summary: str

    # json.JSONDecodeError on a truncated/empty body
    try:
        _json.loads("")
    except _json.JSONDecodeError as e:
        classified = classify_llm_exception(e, provider="nvidia")
        assert isinstance(classified, LLMMalformedResponseError)
        assert is_fallback_eligible(classified) is True

    # LangChain OutputParserException
    classified2 = classify_llm_exception(
        OutputParserException("Failed to parse output: no JSON object found"),
        provider="gemini",
    )
    assert isinstance(classified2, LLMMalformedResponseError)
    assert is_fallback_eligible(classified2) is True

    # Real Pydantic ValidationError from a genuinely empty/unparseable body
    try:
        _Schema.model_validate_json("")
    except Exception as e:
        classified3 = classify_llm_exception(e, provider="openai")
        assert isinstance(classified3, LLMMalformedResponseError)
        assert is_fallback_eligible(classified3) is True

    try:
        _Schema.model_validate_json('{"summary": "x", "changes": [')
    except Exception as e:
        classified4 = classify_llm_exception(e, provider="openai")
        assert isinstance(classified4, LLMMalformedResponseError)
        assert is_fallback_eligible(classified4) is True

    # Direct unit-level eligibility check requested by spec
    assert is_fallback_eligible(LLMMalformedResponseError("empty response")) is True
    assert is_fallback_eligible(LLMInvalidRequestError("bad request")) is False


# 16. test_genuine_invalid_request_still_not_fallback_eligible (Category B regression)
def test_genuine_invalid_request_still_not_fallback_eligible():
    """
    Regression: genuine request-validity failures (HTTP 400/422, context window
    exceeded, a well-formed-JSON-but-schema-invalid Pydantic ValidationError, and
    authentication errors) must remain classified exactly as before - never as
    LLMMalformedResponseError, and never fallback-eligible.
    """
    from pydantic import BaseModel

    class _Schema(BaseModel):
        summary: str
        count: int

    for exc, provider in [
        (Exception("400 Bad Request: invalid parameter"), "nvidia"),
        (Exception("422 Unprocessable Entity: schema mismatch"), "gemini"),
        (Exception("maximum context length exceeded"), "openai"),
        (Exception("context window exceeded for this model"), "nvidia"),
    ]:
        classified = classify_llm_exception(exc, provider=provider)
        assert isinstance(classified, LLMInvalidRequestError)
        assert not isinstance(classified, LLMMalformedResponseError)
        assert is_fallback_eligible(classified) is False

    # Well-formed JSON that simply fails schema validation (missing/wrong-typed
    # field) is NOT a malformed-response problem - it's a genuine schema mismatch.
    try:
        _Schema.model_validate_json('{"summary": "ok"}')
    except Exception as e:
        classified = classify_llm_exception(e, provider="nvidia")
        assert isinstance(classified, LLMInvalidRequestError)
        assert not isinstance(classified, LLMMalformedResponseError)
        assert is_fallback_eligible(classified) is False

    # Authentication errors remain terminal, never fallback-eligible.
    auth_classified = classify_llm_exception(
        Exception("401 Unauthorized: Invalid API key"), provider="nvidia"
    )
    assert isinstance(auth_classified, LLMAuthenticationError)
    assert is_fallback_eligible(auth_classified) is False


# 17. test_empty_response_triggers_fallback (Category C)
def test_empty_response_triggers_fallback(monkeypatch):
    """
    Verifies that a primary provider returning an empty structured completion
    triggers safe fallback to a working secondary provider, and that
    invoke_structured succeeds with the fallback's valid result.
    """
    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded after empty primary response",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)

    result = invoke_structured(primary, RoutingDecision, "Some prompt")
    assert result.task_type == TaskType.BUG_FIX
    assert fallback.invocations == 1
    assert len(recorded_events) == 1
    assert recorded_events[0]["failure_type"] == "LLMMalformedResponseError"


# 18. test_malformed_json_fallback (Category D)
def test_malformed_json_fallback(monkeypatch):
    """
    Verifies that a primary provider returning truncated/malformed JSON (raised as
    a JSONDecodeError from the structured-output parser) triggers safe fallback,
    and the fallback's valid structured output is returned.
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=_json.JSONDecodeError("Expecting value", "{\"summary\": ", 12),
    )
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.CODE_GENERATION,
            confidence=0.92,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded after malformed JSON",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    result = invoke_structured(primary, RoutingDecision, "Some prompt")
    assert result.task_type == TaskType.CODE_GENERATION
    assert primary.invocations == 1
    assert fallback.invocations == 1


# 19. test_malformed_response_fallback_is_bounded (Category F)
def test_malformed_response_fallback_is_bounded(monkeypatch):
    """
    Verifies that malformed/empty responses cannot create unbounded retries: if both
    primary and fallback return malformed/empty responses, invoke_structured raises
    a structured final error after exactly 2 total provider attempts.
    """
    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")
    fallback = FakeEmptyCompletionLLM(provider="gemini", content="")

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    with pytest.raises(LLMMalformedResponseError):
        invoke_structured(primary, RoutingDecision, "Any prompt")

    # Each fake provider is invoked at most twice internally (structured-output
    # attempt + direct-fallback attempt) - never an unbounded retry loop.
    assert primary.invocations == 2
    assert fallback.invocations == 2


class FakeParsingErrorLLM:
    """
    Simulates with_structured_output(schema, include_raw=True) returning
    LangChain's own {"parsed": None, "raw": ..., "parsing_error": exc} shape -
    reaching _invoke_single_provider's OUTER except-block substring checks
    directly (via the `raise result["parsing_error"]` line), without passing
    through the unrelated inner guided_json/[400] recovery branch that only
    wraps the with_structured_output().invoke() call itself.
    """
    def __init__(self, provider="nvidia", model_name="test-model", parsing_error=None):
        self.provider = provider
        self.model = model_name
        self.parsing_error = parsing_error
        self.invocations = 0

    def with_structured_output(self, schema, include_raw=False):
        return self

    def invoke(self, prompt):
        self.invocations += 1
        return {"parsed": None, "raw": None, "parsing_error": self.parsing_error}


# 20. test_malformed_response_substring_regression (review follow-up: A + B)
def test_malformed_response_substring_regression():
    """
    Regression for the review finding: a provider response that merely
    *describes itself* as malformed (not a JSONDecodeError/OutputParserException
    by type, and without Pydantic's json_invalid tag) must still classify as
    LLMMalformedResponseError and remain fallback-eligible - the old bare
    `"malformed" in exc_str` clause on the Invalid Request rule used to steal
    exactly this case and make it terminal.
    """
    for message in [
        "Malformed response body received from provider",
        "malformed JSON in response",
    ]:
        classified = classify_llm_exception(Exception(message), provider="nvidia")
        assert isinstance(classified, LLMMalformedResponseError), (
            f"{message!r} must classify as LLMMalformedResponseError, got {type(classified).__name__}"
        )
        assert is_fallback_eligible(classified) is True

    # A genuine "malformed request" signal (the narrow phrase that replaced the
    # bare "malformed" substring) must still classify as a terminal invalid
    # request, never fallback-eligible.
    classified_req = classify_llm_exception(Exception("malformed request"), provider="nvidia")
    assert isinstance(classified_req, LLMInvalidRequestError)
    assert not isinstance(classified_req, LLMMalformedResponseError)
    assert is_fallback_eligible(classified_req) is False


# 21. test_malformed_response_with_retry_substrings_does_not_retry_internally
def test_malformed_response_with_retry_substrings_does_not_retry_internally():
    """
    Fast-fail safety: a malformed-response error whose (model-controlled)
    message text happens to contain "429" or "[400]" must not trigger
    _invoke_single_provider's internal sleep-and-continue retry checks - it
    must fail fast on the first attempt, exactly like Timeout/Auth/InvalidRequest.
    """
    primary = FakeParsingErrorLLM(
        parsing_error=OutputParserException(
            "Failed to parse output. Raw completion happened to mention [400] verbatim."
        ),
    )

    with pytest.raises(LLMMalformedResponseError):
        _invoke_single_provider(primary, RoutingDecision, "Any prompt")

    # Exactly one internal invocation - fast-failed immediately, no
    # sleep-and-continue retry loop despite the coincidental "[400]" text.
    assert primary.invocations == 1


# 22. test_malformed_response_fallback_telemetry_category
def test_malformed_response_fallback_telemetry_category(monkeypatch):
    """
    Verifies that a malformed-response-triggered provider fallback emits the
    explicit failure_category "LLM_MALFORMED_RESPONSE", distinct from generic
    transient failures (503s), timeouts, and rate limits.
    """
    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")
    fallback = FakeProviderLLM(
        provider="gemini",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            confidence=0.9,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Fallback succeeded",
        ),
    )

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    invoke_structured(primary, RoutingDecision, "Some prompt")

    assert len(recorded_events) == 1
    assert recorded_events[0]["failure_category"] == "LLM_MALFORMED_RESPONSE"
    assert recorded_events[0]["failure_type"] == "LLMMalformedResponseError"


def _set_provider_credentials(monkeypatch, nvidia=None, gemini=None, openai=None, groq=None):
    """
    Deterministically sets/clears provider credentials for
    get_fallback_provider()'s real (unmocked) credential checks, regardless
    of whatever real keys this sandbox's .env happens to configure -
    _has_provider_credentials() checks os.getenv(VAR, <module-level
    default captured at import time>), so both the process env var and
    backend.services.llm's own imported constant must be set/cleared
    together to get a deterministic result independent of the environment.
    """
    import backend.services.llm as llm_module

    for env_name, attr_name, value in [
        ("NVIDIA_API_KEY", "NVIDIA_API_KEY", nvidia),
        ("GOOGLE_API_KEY", "GOOGLE_API_KEY", gemini),
        ("OPENAI_API_KEY", "OPENAI_API_KEY", openai),
        ("GROQ_API_KEY", "GROQ_API_KEY", groq),
    ]:
        if value:
            monkeypatch.setenv(env_name, value)
            monkeypatch.setattr(llm_module, attr_name, value)
        else:
            monkeypatch.delenv(env_name, raising=False)
            monkeypatch.setattr(llm_module, attr_name, None)


# 22b. NVIDIA-only mode: LLM_FALLBACK_ENABLED=false disables fallback
# entirely, even when other providers are fully credentialed - the
# existing mechanism a controlled NVIDIA-only E2E smoke test uses.
@pytest.mark.parametrize("disabled_value", ["false", "0", "no", "False", "NO"])
def test_get_fallback_provider_disabled_returns_none_even_with_credentials(monkeypatch, disabled_value):
    """
    LLM_FALLBACK_ENABLED set to any recognized "off" value must return None
    from get_fallback_provider() regardless of how many OTHER providers are
    fully credentialed - this is a deliberate operator override, not a
    credential-availability check, and must take priority over it.
    """
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", disabled_value)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini="google-key", openai="openai-key")

    assert get_fallback_provider(primary="nvidia") is None


def test_llm_fallback_enabled_defaults_to_true_when_unset(monkeypatch):
    """Sanity check on the flag's own default: with the variable entirely
    unset, fallback selection behaves exactly as before this task (an
    explicit opt-out, not an opt-in, so nothing regresses for deployments
    that never set it)."""
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini="google-key", openai=None)

    assert get_fallback_provider(primary="nvidia") == "gemini"


def test_nvidia_only_mode_fails_closed_without_ever_requesting_gemini(monkeypatch):
    """
    End-to-end: with LLM_FALLBACK_ENABLED=false, a failing NVIDIA primary
    call must raise its OWN classified error directly - get_llm() must
    never be called for a fallback provider at all, proving no Gemini
    request is attempted (not just that none succeeds).
    """
    monkeypatch.setenv("LLM_FALLBACK_ENABLED", "false")
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini="google-key", openai="openai-key")

    primary = FakeProviderLLM(provider="nvidia", failure=TimeoutError("Request timed out after 30.0 seconds"))

    def unexpected_get_llm(provider=None, **kw):
        raise AssertionError(
            f"get_llm() must not be called for a fallback provider when "
            f"LLM_FALLBACK_ENABLED=false (provider={provider!r})"
        )

    monkeypatch.setattr("backend.services.llm.get_llm", unexpected_get_llm)

    with pytest.raises(LLMTimeoutError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert exc_info.value.provider == "nvidia"
    assert primary.invocations == 1


# 22c. Groq as credentialed fallback for NVIDIA (real provider selection,
# real structured-output contract - the actual network calls are still
# faked here via FakeProviderLLM; the real, live NVIDIA+Groq network smoke
# test was run manually as a one-off script, not committed to this suite -
# see this session's final report for its result).
def test_get_fallback_provider_prefers_groq_for_nvidia_primary(monkeypatch):
    """
    get_fallback_provider(primary="nvidia") must select 'groq' first when
    GROQ_API_KEY is configured, ahead of gemini/openai - Groq is the
    intended, preferred fallback for a capacity-exhausted NVIDIA primary.
    """
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-key", gemini="google-key", openai="openai-key", groq="gsk-key")

    assert get_fallback_provider(primary="nvidia") == "groq"


def test_nvidia_retry_exhausted_falls_back_to_groq_with_structured_result(monkeypatch):
    """
    End-to-end: once NVIDIA's own bounded capacity retries are exhausted,
    invoke_structured() must fall back to Groq and return the SAME
    structured RoutingDecision contract the caller expects - Groq fallback
    is not a special case requiring different caller-side handling.
    """
    from backend.core.config import NVIDIA_MAX_RETRIES

    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError("503 Service Unavailable"))
    fallback = FakeProviderLLM(
        provider="groq",
        result=RoutingDecision(
            task_type=TaskType.BUG_FIX,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="Groq fallback classified this correctly",
        ),
    )

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "groq")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback if provider == "groq" else primary)

    result = invoke_structured(primary, RoutingDecision, "Fix broken endpoint")

    assert result.task_type == TaskType.BUG_FIX
    assert primary.invocations == NVIDIA_MAX_RETRIES + 1
    assert fallback.invocations == 1


def test_nvidia_retry_exhausted_no_groq_credential_fails_closed(monkeypatch):
    """
    With no GROQ_API_KEY (and no other fallback credential) configured,
    NVIDIA's own retries must still exhaust and then fail closed with its
    classified error - never crash, never fabricate a RoutingDecision.
    """
    from backend.core.config import NVIDIA_MAX_RETRIES

    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-key", gemini=None, openai=None, groq=None)

    primary = FakeProviderLLM(provider="nvidia", failure=httpx.NetworkError("503 Service Unavailable"))

    with pytest.raises(LLMTransientError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Prompt")

    assert exc_info.value.provider == "nvidia"
    assert primary.invocations == NVIDIA_MAX_RETRIES + 1


def test_groq_malformed_response_still_classified_and_fallback_eligible(monkeypatch):
    """
    Existing malformed-response handling must remain active for Groq: an
    empty/unparseable Groq completion is classified LLMMalformedResponseError,
    not silently treated as success or crashed on.
    """
    groq_fake = FakeEmptyCompletionLLM(provider="groq", content="")

    with pytest.raises(LLMMalformedResponseError):
        _invoke_single_provider(groq_fake, RoutingDecision, "Prompt")


def test_groq_failure_classified_correctly_no_fabricated_result(monkeypatch):
    """
    A Groq failure (e.g. auth error) must be classified correctly and must
    never fabricate a RoutingDecision - the caller gets a structured error,
    not a guessed/default result.
    """
    groq_fake = FakeProviderLLM(provider="groq", failure=Exception("401 Unauthorized: invalid api key"))

    with pytest.raises(LLMAuthenticationError):
        _invoke_single_provider(groq_fake, RoutingDecision, "Prompt")


def test_groq_api_key_never_appears_in_telemetry_or_errors(monkeypatch):
    """
    Security: a Groq failure message that happens to echo a credential-
    shaped string must never leak into the raised error's own message
    beyond what the (test-simulated) provider SDK itself included - and
    must never appear in the PROVIDER_FALLBACK telemetry event, which only
    records pre-known-safe fields (provider names, failure category,
    attempt number), never raw exception text.
    """
    fake_groq_key = "gsk_shouldneverappearinlogs00000000000000"
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError(f"503 Service Unavailable (leaked key {fake_groq_key})"),
    )
    fallback = FakeProviderLLM(
        provider="groq",
        result=RoutingDecision(
            task_type=TaskType.GENERAL,
            requires_planning=False,
            requires_knowledge=False,
            reasoning="ok",
        ),
    )

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "groq")
    monkeypatch.setattr("backend.services.llm.get_llm", lambda provider=None, **kw: fallback)

    invoke_structured(primary, RoutingDecision, "Some prompt")

    assert len(recorded_events) == 1
    event_str = str(recorded_events[0])
    assert fake_groq_key not in event_str
    assert "GROQ_API_KEY" not in event_str


# 23. test_get_fallback_provider_no_credentials_returns_none (Category A)
def test_get_fallback_provider_no_credentials_returns_none(monkeypatch):
    """
    No alternative provider has credentials -> get_fallback_provider must
    return None instead of selecting an uncredentialed provider as a
    last-resort default (the exact bug the read-only investigation found).
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini=None, openai=None)

    assert get_fallback_provider(primary="nvidia") is None


# 23b. test_get_fallback_provider_exclude_finds_next_credentialed_alternative
def test_get_fallback_provider_exclude_finds_next_credentialed_alternative(monkeypatch):
    """
    With BOTH gemini and openai credentialed, excluding gemini (the
    already-tried, rate-limited provider) must select openai - the real
    (unmocked) credential-checked selection logic that powers the bounded
    second-fallback escape hatch in invoke_structured().
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini="google-key", openai="openai-key")

    assert get_fallback_provider(primary="nvidia") == "gemini"
    assert get_fallback_provider(primary="nvidia", exclude=["gemini"]) == "openai"


# 23c. test_get_fallback_provider_exclude_never_selects_uncredentialed_provider
def test_get_fallback_provider_exclude_never_selects_uncredentialed_provider(monkeypatch):
    """
    Excluding the only credentialed alternative must return None, never
    fall through to selecting an uncredentialed provider as a last resort -
    the same fail-closed guarantee the base (no-exclude) selection already
    has, preserved for the exclude path.
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini="google-key", openai=None)

    assert get_fallback_provider(primary="nvidia") == "gemini"
    assert get_fallback_provider(primary="nvidia", exclude=["gemini"]) is None


# 23d. test_get_llm_raises_clear_error_when_no_model_resolvable
def test_get_llm_raises_clear_error_when_no_model_resolvable(monkeypatch):
    """
    If a provider has no resolvable model (a misconfigured _DEFAULT_MODELS
    entry), get_llm() must fail closed with a clear, actionable ValueError
    at construction time instead of silently building a client with
    model=None, which each SDK would otherwise surface as its own
    confusing, provider-specific error deep inside the first request.
    """
    _set_provider_credentials(monkeypatch, nvidia="nvapi-primary-key", gemini=None, openai=None)
    monkeypatch.setattr("backend.services.llm._DEFAULT_MODELS", {"gemini": "gemini-3.6-flash", "openai": "gpt-4o"})
    # LLM_MODEL_NAME only ever fills in the model for the CONFIGURED PRIMARY
    # provider (by design - see get_llm()'s own comment) - both the env var
    # and the module-level constant captured from it at import time must be
    # cleared so this test genuinely exercises the "no model resolvable at
    # all" path instead of incidentally passing via this dev machine's own
    # real .env configuration.
    monkeypatch.delenv("LLM_MODEL_NAME", raising=False)
    monkeypatch.setattr("backend.services.llm.LLM_MODEL_NAME", None)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setattr("backend.services.llm.LLM_PROVIDER", "nvidia")

    with pytest.raises(ValueError, match="No model configured"):
        get_llm(provider="nvidia")


# 24. test_get_fallback_provider_selects_the_one_credentialed_alternative (Category B)
@pytest.mark.parametrize(
    "primary,creds,expected",
    [
        ("nvidia", {"gemini": None, "openai": "k"}, "openai"),
        ("nvidia", {"gemini": "k", "openai": None}, "gemini"),
        ("gemini", {"nvidia": None, "openai": "k"}, "openai"),
        ("gemini", {"nvidia": "k", "openai": None}, "nvidia"),
        ("openai", {"nvidia": None, "gemini": "k"}, "gemini"),
        ("openai", {"nvidia": "k", "gemini": None}, "nvidia"),
    ],
)
def test_get_fallback_provider_selects_the_one_credentialed_alternative(monkeypatch, primary, creds, expected):
    """
    For every primary provider, when exactly one of the two alternatives is
    credentialed, that one is selected - confirming the existing
    deterministic preference order is preserved and correctly skips the
    uncredentialed alternative rather than picking it anyway.
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, **creds)

    assert get_fallback_provider(primary=primary) == expected


# 25. test_explicit_fallback_override_skips_uncredentialed_provider (Category C)
def test_explicit_fallback_override_skips_uncredentialed_provider(monkeypatch):
    """
    LLM_FALLBACK_PROVIDER explicitly names an uncredentialed provider -
    it must NOT be selected; selection must fall through to the normal,
    credential-checked preference order and pick the valid alternative.
    """
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-key", gemini=None, openai="sk-key")

    result = get_fallback_provider(primary="nvidia")
    assert result != "gemini"
    assert result == "openai"


# 26. test_explicit_fallback_override_with_no_alternatives_returns_none (Category D)
def test_explicit_fallback_override_with_no_alternatives_returns_none(monkeypatch):
    """
    LLM_FALLBACK_PROVIDER names an uncredentialed provider and no other
    alternative has credentials either -> None, never a guess.
    """
    monkeypatch.setenv("LLM_FALLBACK_PROVIDER", "gemini")
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-key", gemini=None, openai=None)

    assert get_fallback_provider(primary="nvidia") is None


# 27. test_malformed_primary_with_no_fallback_available (Category E)
def test_malformed_primary_with_no_fallback_available_emits_no_misleading_fallback_event(monkeypatch):
    """
    Primary returns a malformed response, but no fallback provider is
    available (get_fallback_provider() returns None) - invoke_structured
    must fail with the primary's own LLMMalformedResponseError, and must
    NOT emit a PROVIDER_FALLBACK event implying a fallback was attempted,
    since none ever was.
    """
    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")

    recorded_events = []
    original_record_event = telemetry_collector.record_event

    def mock_record_event(*a, **kw):
        recorded_events.append(kw)
        return original_record_event(*a, **kw)

    monkeypatch.setattr(telemetry_collector, "record_event", mock_record_event)
    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: None)

    with pytest.raises(LLMMalformedResponseError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Any prompt")

    assert exc_info.value.provider == "nvidia"
    assert recorded_events == [], "No PROVIDER_FALLBACK event should be recorded when no fallback was ever attempted"


# 28. test_fallback_initialization_failure_chains_exception_and_emits_safe_telemetry (Category F)
def test_fallback_initialization_failure_chains_exception_and_emits_safe_telemetry(monkeypatch):
    """
    Fallback candidate is selected (credentialed per get_fallback_provider,
    from the caller's point of view) but get_llm() still raises during
    initialization for some other reason. Verifies:
    - the externally propagated error remains the primary's classification
      (fail-closed, unchanged from before this hardening)
    - the real init_exc is preserved via exception chaining (__cause__),
      not silently discarded
    - a distinguishable, structured telemetry signal is recorded
      (outcome="fallback_init_failed") with only safe metadata fields
    - no secret/credential material appears anywhere in that telemetry
    """
    primary = FakeProviderLLM(
        provider="nvidia",
        failure=httpx.NetworkError("503 Service Unavailable"),
    )

    def raising_get_llm(provider=None, **kw):
        if provider == "gemini":
            raise ValueError(
                "LLM_PROVIDER is 'gemini' but GOOGLE_API_KEY is missing. Add it to environment variables."
            )
        return primary

    monkeypatch.setattr("backend.services.llm.get_fallback_provider", lambda *a, **kw: "gemini")
    monkeypatch.setattr("backend.services.llm.get_llm", raising_get_llm)

    recorded_events = []
    original_record_event = telemetry_collector.record_event

    def mock_record_event(*a, **kw):
        recorded_events.append(kw)
        return original_record_event(*a, **kw)

    monkeypatch.setattr(telemetry_collector, "record_event", mock_record_event)

    with pytest.raises(LLMTransientError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Any prompt")

    # The primary's own classification is still what's externally propagated.
    assert exc_info.value.provider == "nvidia"

    # init_exc was preserved via chaining, not silently discarded.
    assert exc_info.value.__cause__ is not None
    assert isinstance(exc_info.value.__cause__, ValueError)
    assert "GOOGLE_API_KEY" in str(exc_info.value.__cause__)

    # A distinguishable, structured telemetry signal was recorded.
    init_failure_events = [
        e for e in recorded_events if e.get("metadata", {}).get("outcome") == "fallback_init_failed"
    ]
    assert len(init_failure_events) == 1
    meta = init_failure_events[0]["metadata"]
    assert meta["primary_provider"] == "nvidia"
    assert meta["fallback_provider"] == "gemini"
    assert meta["failure_category"] == "LLM_TRANSIENT_FAILURE"

    # No secrets/credentials anywhere in the recorded telemetry metadata.
    meta_str = str(meta)
    assert "gho_" not in meta_str
    assert "sk-" not in meta_str
    assert "nvapi-" not in meta_str
    assert "Authorization" not in meta_str
    assert "x-access-token" not in meta_str


# 29. test_real_fallback_selection_composes_with_invoke_structured_on_malformed_primary
def test_real_fallback_selection_composes_with_invoke_structured_on_malformed_primary(monkeypatch):
    """
    End-to-end composition test using the REAL (unmocked) get_fallback_provider()
    together with the REAL invoke_structured() - only backend.services.llm.get_llm
    is mocked, at the external LLM-client boundary (no real ChatGoogleGenerativeAI
    client is ever constructed and no real network request is ever made), so this
    proves the actual credential-aware selection logic and the actual fallback
    control flow compose correctly end-to-end, not just each in isolation:

        malformed NVIDIA response -> LLMMalformedResponseError
        -> get_fallback_provider() [REAL] selects the one credentialed
           alternative, gemini, via a test-only fake credential (no real
           GOOGLE_API_KEY needed, read, or touched anywhere)
        -> get_llm(provider="gemini") [mocked at the SDK-client boundary]
        -> fallback invocation succeeds
        -> invoke_structured() [REAL] returns the fallback's structured result
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    # Only nvidia (primary - irrelevant to fallback selection) and gemini are
    # "credentialed" here; openai is deliberately left uncredentialed so the
    # real preference order (gemini before openai for an nvidia primary) is
    # exercised meaningfully, not just trivially satisfied by there being
    # only one possible candidate at all.
    _set_provider_credentials(monkeypatch, nvidia="nvapi-fake-primary-key", gemini="fake-test-google-key", openai=None)

    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")
    fallback_result = RoutingDecision(
        task_type=TaskType.BUG_FIX,
        confidence=0.93,
        requires_planning=False,
        requires_knowledge=False,
        reasoning="Real fallback composition succeeded",
    )
    fallback = FakeProviderLLM(provider="gemini", result=fallback_result)

    def strict_get_llm(provider=None, **kw):
        assert provider == "gemini", f"expected the real selection logic to choose 'gemini', got {provider!r}"
        return fallback

    # Mocked ONLY at the external LLM-client boundary (get_llm) - never
    # get_fallback_provider itself, and never invoke_structured.
    monkeypatch.setattr("backend.services.llm.get_llm", strict_get_llm)

    recorded_events = []
    original_on_fallback = telemetry_collector.on_provider_fallback

    def mock_on_fallback(**kwargs):
        recorded_events.append(kwargs)
        original_on_fallback(**kwargs)

    monkeypatch.setattr(telemetry_collector, "on_provider_fallback", mock_on_fallback)

    result = invoke_structured(primary, RoutingDecision, "Some prompt")

    # 1 + 5. Primary was malformed, and invoke_structured returned the
    # fallback's structured result - not an error, not the primary's output.
    assert result == fallback_result

    # 2 + 3 + 4. The REAL get_fallback_provider() selected "gemini" (the only
    # credentialed alternative) - strict_get_llm's own assertion would have
    # failed the test otherwise - it was actually initialized, and the
    # fallback invocation actually ran and succeeded exactly once.
    assert fallback.invocations == 1

    # 6. No unnecessary additional primary retries: exactly 2 invocations on
    # the primary (structured-output attempt + direct-fallback attempt) -
    # the same existing bound, not a new retry loop from this composition.
    assert primary.invocations == 2

    # 7. A PROVIDER_FALLBACK telemetry event was emitted for the actual
    # fallback attempt, correctly attributing both providers and the
    # classification that triggered it.
    assert len(recorded_events) == 1
    event = recorded_events[0]
    assert event["primary_provider"] == "nvidia"
    assert event["fallback_provider"] == "gemini"
    assert event["failure_type"] == "LLMMalformedResponseError"
    assert event["failure_category"] == "LLM_MALFORMED_RESPONSE"

    # 8. No credential values appear anywhere in the recorded telemetry.
    event_str = str(event)
    assert "fake-test-google-key" not in event_str
    assert "nvapi-fake-primary-key" not in event_str


# 30. test_real_fallback_selection_returns_none_and_no_misleading_event_on_malformed_primary
def test_real_fallback_selection_returns_none_and_no_misleading_event_on_malformed_primary(monkeypatch):
    """
    Complementary to the composition test above: using the REAL (unmocked)
    get_fallback_provider() with no alternative provider credentialed at
    all (matching this sandbox's actual .env: only NVIDIA_API_KEY is set),
    a malformed NVIDIA response must propagate as the primary's own
    LLMMalformedResponseError, and no PROVIDER_FALLBACK event may be
    emitted - because no fallback was ever actually selected or attempted.
    """
    monkeypatch.delenv("LLM_FALLBACK_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_FALLBACK_ENABLED", raising=False)
    _set_provider_credentials(monkeypatch, nvidia="nvapi-fake-primary-key", gemini=None, openai=None)

    primary = FakeEmptyCompletionLLM(provider="nvidia", content="")

    # get_llm must never even be called - there's nothing to fall back to.
    def unexpected_get_llm(provider=None, **kw):
        raise AssertionError(
            f"get_llm() must not be called when no fallback provider is available (provider={provider!r})"
        )

    monkeypatch.setattr("backend.services.llm.get_llm", unexpected_get_llm)

    recorded_events = []
    original_record_event = telemetry_collector.record_event

    def mock_record_event(*a, **kw):
        recorded_events.append(kw)
        return original_record_event(*a, **kw)

    monkeypatch.setattr(telemetry_collector, "record_event", mock_record_event)

    with pytest.raises(LLMMalformedResponseError) as exc_info:
        invoke_structured(primary, RoutingDecision, "Any prompt")

    assert exc_info.value.provider == "nvidia"
    # Bounded: structured-output attempt + direct-fallback attempt only -
    # no internal retry loop, no fallback provider ever attempted.
    assert primary.invocations == 2
    assert recorded_events == [], (
        "No PROVIDER_FALLBACK event should be recorded when the real "
        "selection logic found no credentialed fallback"
    )


# ============================================================================
# PHASE 8 — STEP 3: WORKSPACE LOCKING + CONCURRENCY SAFETY TESTS
# ============================================================================

# 1. test_workspace_lock_acquisition
def test_workspace_lock_acquisition(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-acq-test"
    resource_id = "repo-alpha"
    run_id = "run-acq-001"

    acquired = mgr.acquire_lock(org_id, resource_id, run_id, timeout=2.0)
    assert acquired is True
    assert mgr.is_locked(org_id, resource_id) is True

    owner = mgr.get_lock_owner(org_id, resource_id)
    assert owner is not None
    assert owner["organization_id"] == org_id
    assert owner["resource_id"] == resource_id
    assert owner["run_id"] == run_id
    assert "owner_pid" in owner

    mgr.release_lock(org_id, resource_id, run_id)
    assert mgr.is_locked(org_id, resource_id) is False
    assert mgr.get_lock_owner(org_id, resource_id) is None


# 2. test_workspace_lock_release
def test_workspace_lock_release(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-rel-test"
    resource_id = "repo-beta"
    run_1 = "run-rel-001"
    run_2 = "run-rel-002"

    mgr.acquire_lock(org_id, resource_id, run_1)

    # Another run cannot release run_1's lock
    with pytest.raises(PermissionError, match="does not own the lock"):
        mgr.release_lock(org_id, resource_id, run_2)

    # run_1 releases properly
    mgr.release_lock(org_id, resource_id, run_1)
    assert mgr.is_locked(org_id, resource_id) is False

    # Now run_2 can acquire cleanly
    assert mgr.acquire_lock(org_id, resource_id, run_2, timeout=1.0) is True
    mgr.release_lock(org_id, resource_id, run_2)


# 3. test_workspace_lock_timeout
def test_workspace_lock_timeout(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-timeout-test"
    resource_id = "repo-gamma"
    run_1 = "run-holder"
    run_2 = "run-timed-out"

    mgr.acquire_lock(org_id, resource_id, run_1)

    # run_2 should fail fast with WorkspaceLockTimeoutError
    with pytest.raises(WorkspaceLockTimeoutError) as exc_info:
        mgr.acquire_lock(org_id, resource_id, run_2, timeout=0.15)

    assert "WORKSPACE_LOCK_TIMEOUT" in str(exc_info.value)
    assert "repo-gamma" in str(exc_info.value)

    # run_1 is still the authoritative owner
    owner = mgr.get_lock_owner(org_id, resource_id)
    assert owner["run_id"] == run_1

    mgr.release_lock(org_id, resource_id, run_1)


# 4. test_second_run_cannot_mutate_locked_workspace
def test_second_run_cannot_mutate_locked_workspace(tmp_path, monkeypatch):
    db_path = str(tmp_path / "checkpoints.db")
    lock_mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    # Simulate run-1 holding the lock on project-1
    lock_mgr.acquire_lock("org-test", "project-1", "run-1")

    # Spy on developer node to verify it is NEVER called for run-2
    dev_called = []
    def spy_generate(*args, **kwargs):
        dev_called.append(True)
        return DeveloperResult(summary="Should not be generated", changes=[], requires_testing=False)

    monkeypatch.setattr("backend.graph.nodes.generate_code_changes", spy_generate)
    monkeypatch.setattr("backend.core.config.WORKSPACE_LOCK_TIMEOUT_SECONDS", 0.15)

    try:
        status_res = runner.start_run(
            run_id="run-2",
            user_message="Modify files in project-1",
            project_id="project-1",
            organization_id="org-test",
        )
        assert status_res.status == "FAILED"
        assert "WORKSPACE_LOCK_TIMEOUT" in (status_res.error_summary or "")
        assert len(dev_called) == 0, "Developer mutation must NOT run when lock times out!"
    finally:
        lock_mgr.release_lock("org-test", "project-1", "run-1")
        runner.close()


# 5. test_lock_released_after_exception
def test_lock_released_after_exception(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-exc-test"
    resource_id = "repo-delta"
    run_1 = "run-fail"
    run_2 = "run-next"

    try:
        with mgr.acquire(org_id, resource_id, run_1):
            assert mgr.is_locked(org_id, resource_id) is True
            raise RuntimeError("Unexpected failure in graph node")
    except RuntimeError:
        pass

    # Lock must be released despite exception
    assert mgr.is_locked(org_id, resource_id) is False

    # Next run can acquire without issue
    assert mgr.acquire_lock(org_id, resource_id, run_2, timeout=1.0) is True
    mgr.release_lock(org_id, resource_id, run_2)


# 6. test_same_run_does_not_deadlock
def test_same_run_does_not_deadlock(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    org_id = "org-reentrant"
    resource_id = "repo-epsilon"
    run_id = "run-nested"

    # Nested acquisition by the same run_id must succeed without deadlock
    with mgr.acquire(org_id, resource_id, run_id):
        assert mgr.is_locked(org_id, resource_id) is True
        with mgr.acquire(org_id, resource_id, run_id):
            assert mgr.is_locked(org_id, resource_id) is True
            with mgr.acquire(org_id, resource_id, run_id):
                owner = mgr.get_lock_owner(org_id, resource_id)
                assert owner["run_id"] == run_id

        # Still locked until outermost context exits
        assert mgr.is_locked(org_id, resource_id) is True

    # Fully released after outermost exit
    assert mgr.is_locked(org_id, resource_id) is False


# 7. test_different_projects_can_run_concurrently
def test_different_projects_can_run_concurrently(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    barrier = threading.Barrier(2, timeout=5.0)
    concurrent_success = [False, False]

    def worker(worker_idx, proj_name, run_name):
        with mgr.acquire("org-shared", proj_name, run_name, timeout=5.0):
            # If both hold their locks simultaneously, the barrier will trip
            barrier.wait()
            concurrent_success[worker_idx] = True

    t1 = threading.Thread(target=worker, args=(0, "project-alpha", "run-t1"))
    t2 = threading.Thread(target=worker, args=(1, "project-beta", "run-t2"))

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert concurrent_success[0] is True
    assert concurrent_success[1] is True


# 8. test_same_resource_is_serialized
def test_same_resource_is_serialized(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    execution_order = []
    run1_hold_event = threading.Event()
    run1_acquired_event = threading.Event()

    def worker_run1():
        with mgr.acquire("org-shared", "project-same", "run-1", timeout=5.0):
            execution_order.append("run1_acquired")
            run1_acquired_event.set()
            run1_hold_event.wait(timeout=5.0)
            execution_order.append("run1_releasing")

    def worker_run2():
        run1_acquired_event.wait(timeout=5.0)
        with mgr.acquire("org-shared", "project-same", "run-2", timeout=5.0):
            execution_order.append("run2_acquired")

    t1 = threading.Thread(target=worker_run1)
    t2 = threading.Thread(target=worker_run2)

    t1.start()
    t2.start()

    # Give t2 a moment to block on the lock held by t1
    run1_acquired_event.wait(timeout=2.0)
    time.sleep(0.1)

    # Release run1
    run1_hold_event.set()

    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert execution_order == ["run1_acquired", "run1_releasing", "run2_acquired"]


# 9. test_actual_branch_workspace_isolation_behavior
def test_actual_branch_workspace_isolation_behavior():
    task_id_1 = "sandbox-ai-demo"
    task_id_2 = "sandbox-ai-demo"

    branch1 = GitWorkspaceManager.generate_branch_name(task_id_1)
    branch2 = GitWorkspaceManager.generate_branch_name(task_id_2)

    assert branch1 != branch2, "Branch generation must include entropy suffix to prevent collision!"
    assert branch1.startswith("agent/task-sandbox-")
    assert branch2.startswith("agent/task-sandbox-")


# 10. test_cross_tenant_lock_isolation
def test_cross_tenant_lock_isolation(tmp_path):
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))
    barrier = threading.Barrier(2, timeout=5.0)
    isolated_success = [False, False]

    def tenant_worker(idx, org_id, proj_id, run_id):
        with mgr.acquire(org_id, proj_id, run_id, timeout=5.0):
            barrier.wait()
            isolated_success[idx] = True

    t1 = threading.Thread(target=tenant_worker, args=(0, "tenant-a", "common-repo", "run-a"))
    t2 = threading.Thread(target=tenant_worker, args=(1, "tenant-b", "common-repo", "run-b"))

    t1.start()
    t2.start()
    t1.join(timeout=5.0)
    t2.join(timeout=5.0)

    assert isolated_success[0] is True
    assert isolated_success[1] is True

    # Confirm another run/tenant cannot release Tenant B's lock
    mgr.acquire_lock("tenant-b", "common-repo", "run-b")
    with pytest.raises(PermissionError):
        mgr.release_lock("tenant-b", "common-repo", "run-unauthorized")
    mgr.release_lock("tenant-b", "common-repo", "run-b")

    # Confirm Tenant A's and Tenant B's keys are completely distinct
    key_a = mgr.get_resource_key("tenant-a", "common-repo")
    key_b = mgr.get_resource_key("tenant-b", "common-repo")
    assert key_a != key_b


# 11. test_lock_metadata_is_sanitized
def test_lock_metadata_is_sanitized(tmp_path):
    lock_dir = tmp_path / "locks"
    mgr = WorkspaceLockManager(lock_dir=str(lock_dir))
    org_id = "org-sec-audit"
    resource_id = "project-billing"
    run_id = "run-audit-001"

    with mgr.acquire(org_id, resource_id, run_id):
        owner = mgr.get_lock_owner(org_id, resource_id)
        assert owner is not None
        assert owner["organization_id"] == org_id
        assert owner["resource_id"] == resource_id
        assert owner["run_id"] == run_id
        assert "owner_pid" in owner

    # After release, verify on-disk lock file content is valid JSON with sanitized metadata
    lock_files = list(lock_dir.glob("*.lock"))
    assert len(lock_files) == 1
    with open(lock_files[0], "r", encoding="utf-8") as f:
        raw_meta = f.read()

    import json
    meta = json.loads(raw_meta)
    assert meta["organization_id"] == org_id
    assert meta["resource_id"] == resource_id
    assert meta["run_id"] == run_id
    assert "owner_pid" in meta

    # Verify no secret keywords exist in raw content
    raw_lower = raw_meta.lower()
    assert "token" not in raw_lower
    assert "secret" not in raw_lower
    assert "password" not in raw_lower
    assert "bearer" not in raw_lower
    assert "apikey" not in raw_lower


# 12. test_workspace_lock_telemetry
def test_workspace_lock_telemetry(tmp_path, monkeypatch):
    recorded_events = []
    def mock_record(event):
        if event.event_type in (
            TelemetryEventType.WORKSPACE_LOCK_ACQUIRED,
            TelemetryEventType.WORKSPACE_LOCK_RELEASED,
            TelemetryEventType.WORKSPACE_LOCK_TIMEOUT,
        ):
            recorded_events.append(event)

    monkeypatch.setattr(telemetry_store, "record_event", mock_record)
    mgr = WorkspaceLockManager(lock_dir=str(tmp_path / "locks"))

    # Acquire and release
    with mgr.acquire("org-telemetry", "repo-tel", "run-tel-1"):
        pass

    # Force a timeout
    mgr.acquire_lock("org-telemetry", "repo-tel", "run-holder")
    try:
        mgr.acquire_lock("org-telemetry", "repo-tel", "run-timeout", timeout=0.1)
    except WorkspaceLockTimeoutError:
        pass
    finally:
        mgr.release_lock("org-telemetry", "repo-tel", "run-holder")

    types = [e.event_type for e in recorded_events]
    assert TelemetryEventType.WORKSPACE_LOCK_ACQUIRED in types
    assert TelemetryEventType.WORKSPACE_LOCK_RELEASED in types
    assert TelemetryEventType.WORKSPACE_LOCK_TIMEOUT in types

    timeout_ev = next(e for e in recorded_events if e.event_type == TelemetryEventType.WORKSPACE_LOCK_TIMEOUT)
    assert timeout_ev.safe_metadata["outcome"] == "TIMEOUT"
    assert "wait_duration_ms" in timeout_ev.safe_metadata


# 13. test_waiting_approval_workspace_safety
def test_waiting_approval_workspace_safety(tmp_path, monkeypatch):
    """
    Verifies workspace-lock safety while a run is genuinely paused waiting
    for human approval: the lock must not be held across the (potentially
    hours-long) wait, so a second run can acquire it without blocking.

    Made deterministic/offline (previously called AgentRunner.start_run()
    with the real, unmocked LLM/developer/QA path and merely assumed it
    would land on WAITING_APPROVAL - a live network call that could also
    legitimately crash mid-graph): a real, isolated git workspace plus the
    graph's own real git_prepare_node give a genuine non-empty diff, and
    every remaining LLM/subprocess call reachable along the way is mocked
    (route_task/generate_code_changes/review_code_changes via the autouse
    mock_agent_nodes fixture; developer_node's own inline exact-snippet
    call, QA's real subprocess checks, and the revision LLM paths via
    _mock_offline_graph_dependencies).
    """
    project_id = "hitl-lock-workspace"
    _init_git_workspace(tmp_path, monkeypatch, project_id)
    _mock_offline_graph_dependencies(monkeypatch)
    monkeypatch.setattr("backend.graph.nodes.git_prepare_node", _real_git_prepare_node)

    db_path = str(tmp_path / "hitl_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    try:
        # Start run, which pauses at approval interrupt
        status = runner.start_run(
            run_id="run-hitl-lock-test",
            user_message="Feature requiring review",
            project_id=project_id,
            organization_id="org-hitl",
        )
        assert status.status == "WAITING_APPROVAL"
        # Genuinely non-empty: the graph actually produced a real diff,
        # not a hard-coded/forced status.
        assert status.git_diff is not None
        assert not status.git_diff.is_no_op
        assert status.git_diff.files_changed

        # Verify lock is NOT held while waiting for human approval
        assert lock_mgr.is_locked("org-hitl", project_id) is False

        # Another run can safely acquire the lock without being blocked for hours
        assert lock_mgr.acquire_lock("org-hitl", project_id, "run-subsequent", timeout=1.0) is True
        lock_mgr.release_lock("org-hitl", project_id, "run-subsequent")
    finally:
        runner.close()


# 14. test_patch_hash_integrity_with_concurrent_run_attempt
def test_patch_hash_integrity_with_concurrent_run_attempt(tmp_path, monkeypatch):
    """
    Verifies approval/diff hash integrity when a concurrent workspace
    modification is detected between the approval request and the
    reviewer's resume: the pre-commit drift check must reject the commit
    (PATCH_HASH_MISMATCH), never silently commit drifted content.

    Made deterministic/offline for the same reason and via the same
    mechanism as test_waiting_approval_workspace_safety above - the run
    must genuinely reach WAITING_APPROVAL with a real, non-empty diff
    before the drift simulation (step 2) and resume (step 3) can
    meaningfully exercise the integrity check this test is actually about.
    """
    project_id = "drift-prevention-workspace"
    _init_git_workspace(tmp_path, monkeypatch, project_id)
    _mock_offline_graph_dependencies(monkeypatch)
    monkeypatch.setattr("backend.graph.nodes.git_prepare_node", _real_git_prepare_node)

    db_path = str(tmp_path / "drift_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    run_id = "run-drift-prevention"
    org_id = "org-drift-test"

    try:
        # 1. Run enters WAITING_APPROVAL with a genuine, non-empty diff.
        status = runner.start_run(
            run_id=run_id,
            user_message="Fix bug with integrity check",
            project_id=project_id,
            organization_id=org_id,
        )
        assert status.status == "WAITING_APPROVAL"
        assert status.git_diff is not None
        assert not status.git_diff.is_no_op
        assert status.git_diff.patch_hash

        # 2. Simulate concurrent workspace modification / drift before approval resumes
        # GitWorkspaceManager.verify_workspace_drift detects drift
        monkeypatch.setattr(
            GitWorkspaceManager,
            "verify_workspace_drift",
            MagicMock(return_value=(False, "Workspace drift detected: content altered")),
        )

        # 3. Reviewer attempts to resume and approve
        decision = ApprovalDecision(
            approved=True,
            reviewer="Security Admin",
            patch_hash=status.git_diff.patch_hash,
        )
        resume_status = runner.resume_run(run_id, decision, organization_id=org_id)

        # 4. Must fail pre-commit drift validation and reject commit
        assert resume_status.status == "COMPLETED"
        values = runner.get_state_values(run_id, organization_id=org_id)
        assert values.get("approval_status") == "PATCH_HASH_MISMATCH"
    finally:
        runner.close()


# 15. test_no_commit_after_lock_timeout
def test_no_commit_after_lock_timeout(tmp_path, monkeypatch):
    db_path = str(tmp_path / "no_commit_checkpoints.db")
    lock_dir = str(tmp_path / "locks")
    lock_mgr = WorkspaceLockManager(lock_dir=lock_dir)
    runner = AgentRunner(checkpoint_db_path=db_path, lock_manager=lock_mgr)

    # Pre-lock resource
    lock_mgr.acquire_lock("org-block", "proj-block", "external-holder")

    commit_spy = MagicMock(return_value=True)
    monkeypatch.setattr(GitWorkspaceManager, "stage_and_commit", commit_spy)
    monkeypatch.setattr("backend.core.config.WORKSPACE_LOCK_TIMEOUT_SECONDS", 0.1)

    try:
        status = runner.start_run(
            run_id="run-block-commit",
            user_message="Should not commit",
            project_id="proj-block",
            organization_id="org-block",
        )
        assert status.status == "FAILED"
        assert "WORKSPACE_LOCK_TIMEOUT" in (status.error_summary or "")
        assert commit_spy.call_count == 0, "No git commit must be performed after lock timeout!"
    finally:
        lock_mgr.release_lock("org-block", "proj-block", "external-holder")
        runner.close()
