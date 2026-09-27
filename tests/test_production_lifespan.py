"""
Tests for Phase 8 Step 6: Production FastAPI Lifespan + Graceful Shutdown + Readiness.

Covers:
1. Lifespan management and clean startup/shutdown
2. Active-run draining and safe 503 rejection
3. Waiting-approval preservation across drain/shutdown
4. Cooperative cancellation on drain timeout and lock release
5. Granular readiness probes (/health/ready) vs lightweight liveness (/health)
6. Secret-free diagnostics
7. Shutdown lifecycle telemetry events
"""

import os
import shutil
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.api.app import create_app
from backend.api.lifecycle import AppLifecycleState, LifecycleManager
from backend.api.models import CreateRunRequest, ResumeRunRequest, RunStatusResponse
from backend.core.config import SHUTDOWN_DRAIN_TIMEOUT_SECONDS
from backend.graph.runner import AgentRunner
from backend.observability.collector import TelemetryCollector, telemetry_collector
from backend.observability.store import TelemetryStore, telemetry_store
from backend.schemas.telemetry import RunRecord, TelemetryEventType
from backend.schemas.tenant import Role
from backend.security.auth import AuthMode
from backend.security.tenant import tenant_manager
from backend.vcs.workspace_lock import WorkspaceLockManager


def _grant(user_id: str, org_id: str, role: Role, org_name: str = "Test Org"):
    tenant_manager.create_organization(org_id, org_name)
    tenant_manager.create_user(user_id, f"{user_id}@test.local", user_id)
    tenant_manager.add_membership(org_id, user_id, role)


@pytest.fixture(autouse=True)
def ensure_telemetry_store_open():
    """Ensure module-level telemetry store and tenant manager are ready before and after every test."""
    tenant_manager.reset()
    tenant_manager.set_mode(AuthMode.DEVELOPMENT, fallback=True)
    if hasattr(telemetry_store, "reopen"):
        telemetry_store.reopen()
    yield
    if hasattr(telemetry_store, "reopen"):
        telemetry_store.reopen()


@pytest.fixture
def temp_workspace(tmp_path):
    """Provides an isolated workspace directory with subdirectories."""
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    checkpoints_db = str(ws / "checkpoints.db")
    telemetry_db = str(ws / "telemetry.db")
    locks_dir = str(ws / "locks")
    return {
        "root": ws,
        "checkpoints_db": checkpoints_db,
        "telemetry_db": telemetry_db,
        "locks_dir": locks_dir,
    }


# =============================================================================
# Suite 1: Lifespan Management & Startup
# =============================================================================
class TestLifespanStartup:

    def test_lifespan_initializes_ready_state(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        assert app.state.lifecycle.state == AppLifecycleState.READY
        assert app.state.lifecycle.is_ready() is True
        assert app.state.lifecycle.is_accepting_work() is True
        runner.close()

    def test_lifespan_starts_and_stops_cleanly(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner, drain_timeout_seconds=2.0)

        with TestClient(app) as client:
            assert app.state.lifecycle.state == AppLifecycleState.READY
            resp = client.get("/health")
            assert resp.status_code == 200

        # After exiting context manager, lifespan teardown has executed
        assert app.state.lifecycle.state == AppLifecycleState.STOPPED
        assert app.state.lifecycle.is_accepting_work() is False

    def test_lifespan_rearms_telemetry_store(self, temp_workspace):
        store = TelemetryStore(db_path=temp_workspace["telemetry_db"])
        store.close()
        assert store.is_ready() is False

        store.reopen()
        assert store.is_ready() is True
        store.close()

    def test_lifespan_startup_readiness_validation(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        mgr = LifecycleManager()
        report = mgr.check_readiness(runner=runner, store=telemetry_store)
        assert report["is_ready"] is True
        assert report["checks"]["lifecycle"]["ready"] is True
        assert report["checks"]["checkpointer"]["ready"] is True
        assert report["checks"]["telemetry_store"]["ready"] is True
        runner.close()


# =============================================================================
# Suite 2: Draining & Safe Active-Run Protection
# =============================================================================
class TestShutdownDraining:

    def test_draining_rejects_new_runs(self, temp_workspace):
        _grant("user-a", "test-org", Role.ENGINEER)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        # Transition manually to DRAINING
        app.state.lifecycle.set_state(AppLifecycleState.DRAINING)

        resp = client.post(
            "/api/v1/runs",
            json={"user_message": "Fix login bug", "project_id": "proj_1"},
            headers={"X-User-ID": "user-a", "X-Organization-ID": "test-org"},
        )
        assert resp.status_code == 503
        assert "Server is shutting down" in resp.json()["detail"]
        assert resp.headers.get("Retry-After") == "10"
        runner.close()

    def test_draining_rejects_run_resumes(self, temp_workspace):
        _grant("user-a", "test-org", Role.ENGINEER)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        # Transition to DRAINING
        app.state.lifecycle.set_state(AppLifecycleState.DRAINING)

        resp = client.post(
            "/api/v1/runs/run_test123/resume",
            json={"approved": True, "reviewer": "user-a"},
            headers={"X-User-ID": "user-a", "X-Organization-ID": "test-org"},
        )
        assert resp.status_code == 503
        assert "Server is shutting down" in resp.json()["detail"]
        assert resp.headers.get("Retry-After") == "10"
        runner.close()

    def test_draining_allows_run_cancellations(self, temp_workspace):
        _grant("user-a", "test-org", Role.ENGINEER)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        # Seed a run in the store
        rec = RunRecord(run_id="run_cancel_test", organization_id="test-org", status="RUNNING")
        telemetry_store.create_or_update_run(rec)

        # Mock cancel_run on runner to verify it's reachable
        with patch.object(runner, "cancel_run", return_value=RunStatusResponse(run_id="run_cancel_test", status="CANCELLED")):
            app.state.lifecycle.set_state(AppLifecycleState.DRAINING)

            resp = client.post(
                "/api/v1/runs/run_cancel_test/cancel",
                json={"reason": "Operator shutdown"},
                headers={"X-User-ID": "user-a", "X-Organization-ID": "test-org"},
            )
            # Not 503 - cancellations remain available
            assert resp.status_code == 200
            assert resp.json()["status"] == "CANCELLED"
        runner.close()

    def test_draining_allows_read_operations(self, temp_workspace):
        _grant("user-a", "test-org", Role.ENGINEER)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        app.state.lifecycle.set_state(AppLifecycleState.DRAINING)

        # Health endpoint remains 200
        resp_h = client.get("/health")
        assert resp_h.status_code == 200

        # Tenant context remains readable
        resp_ctx = client.get("/api/v1/tenant/context", headers={"X-User-ID": "user-a", "X-Organization-ID": "test-org"})
        assert resp_ctx.status_code == 200
        runner.close()

    def test_drain_waits_for_active_runs_to_complete(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        with runner._lock:
            runner._active_runs["run_slow"] = "RUNNING"

        # Background thread completes the run after 0.2s
        def complete_run():
            time.sleep(0.2)
            with runner._lock:
                runner._active_runs.pop("run_slow", None)

        t = threading.Thread(target=complete_run)
        t.start()

        res = runner.drain(timeout_seconds=2.0, poll_interval_seconds=0.05)
        t.join()

        assert res["drained"] is True
        assert len(res["active_runs_remaining"]) == 0
        assert len(res["cancelled_runs"]) == 0
        runner.close()

    def test_drain_timeout_cancels_remaining_runs(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        with runner._lock:
            runner._active_runs["run_stuck_1"] = "RUNNING"
            runner._run_tenants["run_stuck_1"] = "test-org"

        # Mock cancel_run to simulate node-level cooperative cancel
        def fake_cancel(run_id, **kwargs):
            with runner._lock:
                runner._active_runs.pop(run_id, None)
            return MagicMock(status="CANCELLED")

        with patch.object(runner, "cancel_run", side_effect=fake_cancel) as mock_cancel:
            res = runner.drain(timeout_seconds=0.1, poll_interval_seconds=0.02)
            assert mock_cancel.called
            assert "run_stuck_1" in res["cancelled_runs"]
            assert res["drained"] is True
        runner.close()

    def test_drain_grace_period_unwinds_cancellation(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        with runner._lock:
            runner._active_runs["run_grace"] = "RUNNING"

        def slow_cancel(run_id, **kwargs):
            # Schedules popping active run slightly later to simulate unwinding
            def unwind():
                time.sleep(0.1)
                with runner._lock:
                    runner._active_runs.pop(run_id, None)
            threading.Thread(target=unwind).start()
            return MagicMock(status="CANCELLED")

        with patch.object(runner, "cancel_run", side_effect=slow_cancel):
            res = runner.drain(timeout_seconds=0.05, poll_interval_seconds=0.02)
            assert res["drained"] is True
        runner.close()

    def test_drain_with_zero_active_runs_exits_immediately(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        t0 = time.time()
        res = runner.drain(timeout_seconds=10.0, poll_interval_seconds=0.1)
        duration = time.time() - t0

        assert res["drained"] is True
        assert duration < 0.2
        runner.close()

    def test_waiting_approval_runs_are_not_cancelled_by_drain(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        # WAITING_APPROVAL runs are not in self._active_runs
        with runner._lock:
            assert "run_hitl" not in runner._active_runs

        with patch.object(runner, "cancel_run") as mock_cancel:
            res = runner.drain(timeout_seconds=0.1, poll_interval_seconds=0.02)
            assert res["drained"] is True
            mock_cancel.assert_not_called()
        runner.close()


# =============================================================================
# Suite 3: Resource Cleanup & Concurrency
# =============================================================================
class TestResourceCleanup:

    def test_shutdown_closes_runner_checkpointer(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        assert runner.is_ready() is True
        runner.close()
        assert runner._conn is None
        assert runner.is_ready() is False

    def test_shutdown_closes_telemetry_store(self, temp_workspace):
        store = TelemetryStore(db_path=temp_workspace["telemetry_db"])
        assert store.is_ready() is True
        store.close()
        assert store.is_ready() is False

    def test_shutdown_is_idempotent(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        runner.close()
        runner.close()  # No error
        assert runner._conn is None

        store = TelemetryStore(db_path=temp_workspace["telemetry_db"])
        store.close()
        store.close()  # No error
        assert store._closed is True

    def test_shutdown_releases_workspace_locks(self, temp_workspace):
        lock_mgr = WorkspaceLockManager(lock_dir=temp_workspace["locks_dir"])

        # Simulate lock acquisition
        with lock_mgr.acquire("test-org", "repo_1", "run_101"):
            assert lock_mgr.is_locked("test-org", "repo_1") is True

        # After block, lock is released
        assert lock_mgr.is_locked("test-org", "repo_1") is False

    def test_concurrent_drain_calls_are_safe(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        with runner._lock:
            runner._active_runs["run_c1"] = "RUNNING"

        def finish():
            time.sleep(0.1)
            with runner._lock:
                runner._active_runs.pop("run_c1", None)

        threading.Thread(target=finish).start()

        results = []
        def call_drain():
            results.append(runner.drain(timeout_seconds=1.0, poll_interval_seconds=0.05))

        threads = [threading.Thread(target=call_drain) for _ in range(3)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()

        assert len(results) == 3
        for r in results:
            assert r["drained"] is True
        runner.close()


# =============================================================================
# Suite 4: Production Readiness Endpoint
# =============================================================================
class TestReadinessEndpoint:

    def test_readiness_returns_200_when_all_healthy(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["is_ready"] is True
        assert data["lifecycle_state"] == "READY"
        assert data["checks"]["checkpointer"]["ready"] is True
        assert data["checks"]["telemetry_store"]["ready"] is True
        assert data["checks"]["workspace_lock"]["ready"] is True
        assert data["checks"]["git"]["ready"] is True
        runner.close()

    def test_readiness_returns_503_during_draining(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        app.state.lifecycle.set_state(AppLifecycleState.DRAINING)
        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "not_ready"
        assert data["is_ready"] is False
        assert data["lifecycle_state"] == "DRAINING"
        runner.close()

    def test_readiness_returns_503_on_checkpointer_failure(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        # Close runner connection to simulate DB failure
        runner.close()

        resp = client.get("/health/ready")
        assert resp.status_code == 503
        data = resp.json()
        assert data["is_ready"] is False
        assert data["checks"]["checkpointer"]["ready"] is False

    def test_readiness_returns_503_on_telemetry_failure(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        with patch.object(telemetry_store, "is_ready", return_value=False):
            resp = client.get("/health/ready")
            assert resp.status_code == 503
            data = resp.json()
            assert data["is_ready"] is False
            assert data["checks"]["telemetry_store"]["ready"] is False
        runner.close()

    def test_readiness_returns_503_on_workspace_lock_failure(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        mgr = LifecycleManager()

        with patch("backend.api.lifecycle.os_access_writable", return_value=False):
            report = mgr.check_readiness(runner=runner, store=telemetry_store)
            assert report["is_ready"] is False
            assert report["checks"]["workspace_lock"]["ready"] is False
        runner.close()

    def test_readiness_reports_fallback_configured_when_credential_present(self, temp_workspace, monkeypatch):
        """The new llm_provider diagnostic must make a configured fallback
        visible - and must never gate overall readiness on it, since a
        fallback credential is always optional by design."""
        import backend.services.llm as llm_module
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-test-key")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # groq is nvidia's preferred fallback when credentialed (see
        # backend/services/llm.py's _FALLBACK_ORDER) - cleared here (env AND
        # the module's own imported constant, since whatever real .env this
        # suite runs against may have a real GROQ_API_KEY captured at import
        # time) so this test's target of "gemini specifically" is unaffected
        # by whichever fallback credentials happen to exist in that .env.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(llm_module, "GROQ_API_KEY", None)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_ready"] is True
        llm_check = data["checks"]["llm_provider"]
        assert llm_check["primary"] == "nvidia"
        assert llm_check["primary_credentialed"] is True
        assert llm_check["fallback_configured"] is True
        assert llm_check["fallback_provider"] == "gemini"
        runner.close()

    def test_readiness_reports_no_fallback_without_weakening_readiness(self, temp_workspace, monkeypatch):
        """Missing GOOGLE_API_KEY/OPENAI_API_KEY must be visible in the
        readiness report - the application must never silently pretend it
        has a fallback - but must NOT flip is_ready/status: an
        unconfigured, optional fallback is not a service outage."""
        import backend.services.llm as llm_module
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test-key")
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Also clear groq (env AND the module's own imported constant) -
        # otherwise a real GROQ_API_KEY in whatever .env this suite runs
        # against would make it "configured" despite this test asserting
        # no fallback is configured at all. See the sibling test above.
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        monkeypatch.setattr(llm_module, "GROQ_API_KEY", None)
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_ready"] is True
        assert data["status"] == "ready"
        llm_check = data["checks"]["llm_provider"]
        assert llm_check["fallback_configured"] is False
        assert llm_check["fallback_provider"] is None
        runner.close()

    def test_readiness_never_exposes_secrets(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/health/ready")
        content = resp.text.lower()

        # Guarantee no secrets or keys leaked
        assert "api_key" not in content
        assert "token" not in content
        assert "password" not in content
        assert "secret" not in content
        runner.close()

    def test_readiness_llm_provider_check_never_leaks_actual_credential_values(
        self, temp_workspace, monkeypatch
    ):
        """The new llm_provider diagnostic reports booleans and provider
        NAMES only - a real (fake, test-only) credential VALUE set in the
        environment must never appear verbatim in the readiness response."""
        monkeypatch.setenv("LLM_PROVIDER", "nvidia")
        monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-SENTINEL-DO-NOT-LEAK-abc123")
        monkeypatch.setenv("GOOGLE_API_KEY", "google-SENTINEL-DO-NOT-LEAK-xyz789")
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        resp = client.get("/health/ready")
        assert "SENTINEL-DO-NOT-LEAK" not in resp.text
        runner.close()

    def test_liveness_remains_200_during_draining(self, temp_workspace):
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        app = create_app(runner=runner)
        client = TestClient(app)

        app.state.lifecycle.set_state(AppLifecycleState.DRAINING)

        # Liveness must stay 200 so orchestrator doesn't abruptly SIGKILL
        resp_live = client.get("/health")
        assert resp_live.status_code == 200
        assert resp_live.json()["status"] == "healthy"

        # Readiness is 503 to drop from load balancer
        resp_ready = client.get("/health/ready")
        assert resp_ready.status_code == 503
        runner.close()


# =============================================================================
# Suite 5: Shutdown Telemetry & Audit
# =============================================================================
class TestShutdownTelemetry:

    def test_shutdown_emits_shutdown_started_event(self, temp_workspace):
        mock_collector = MagicMock()
        with patch("backend.api.app.telemetry_collector", mock_collector):
            runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
            app = create_app(runner=runner, drain_timeout_seconds=5.0)

            with TestClient(app) as client:
                pass

            assert mock_collector.on_shutdown_started.called
            _, kwargs = mock_collector.on_shutdown_started.call_args
            assert kwargs["drain_timeout_seconds"] == 5.0
            runner.close()

    def test_shutdown_emits_shutdown_completed_event(self, temp_workspace):
        mock_collector = MagicMock()
        with patch("backend.api.app.telemetry_collector", mock_collector):
            runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
            app = create_app(runner=runner, drain_timeout_seconds=5.0)

            with TestClient(app) as client:
                pass

            assert mock_collector.on_shutdown_completed.called
            runner.close()

    def test_shutdown_emits_shutdown_interrupted_on_error(self, temp_workspace):
        mock_collector = MagicMock()
        runner = AgentRunner(checkpoint_db_path=temp_workspace["checkpoints_db"])
        with patch.object(runner, "drain", side_effect=RuntimeError("Simulated drain failure")):
            with patch("backend.api.app.telemetry_collector", mock_collector):
                app = create_app(runner=runner, drain_timeout_seconds=5.0)
                with TestClient(app) as client:
                    pass

                assert mock_collector.on_shutdown_interrupted.called
                _, kwargs = mock_collector.on_shutdown_interrupted.call_args
                assert "Simulated drain failure" in kwargs["reason"]
        runner.close()
