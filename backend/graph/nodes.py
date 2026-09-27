from typing import Optional

from backend.graph.state import AgentState
from backend.schemas.routing import TaskType
from backend.schemas.knowledge import KnowledgeAnswer
from backend.schemas.planning import ExecutionPlan
from backend.schemas.qa import QualityCheck, QualityCheckStatus, FailureCategory
from backend.observability.telemetry import collect_usage, invoke_structured, merge_usage
from backend.observability.collector import telemetry_collector
from backend.schemas.telemetry import TelemetryEventType
from backend.graph.cancellation import check_cancelled, mark_activity
from backend.vcs.workspace_paths import resolve_workspace_path

from backend.agents.router import route_task
from backend.agents.planner import create_plan
from backend.agents.knowledge import answer_from_project
from backend.agents.developer import generate_code_changes
from backend.agents.qa import review_code_changes

from langgraph.types import interrupt


def _resolve_project_path(state: AgentState):
    """
    Resolves the tenant-namespaced workspace path (workspace/<organization_id>/
    <project_id>) for this run's state, or None when project_id is absent
    or the (organization_id, project_id) pair doesn't resolve to a safe
    path. Every node in this module that needs the run's own workspace
    directory goes through this instead of constructing
    Path("workspace") / project_id independently - callers already treat
    a None/missing workspace path as "no repo context available", so this
    preserves that exact existing fallback behavior for the read-only
    context-lookup call sites; write sites (developer_node's patch write,
    _materialize_developer_changes) additionally fail closed with an
    explicit error when this returns None for a project_id that IS
    present, since a write must never silently target nowhere.
    """
    project_id = state.get("project_id")
    if not project_id:
        return None
    return resolve_workspace_path(state.get("organization_id", "default-org"), project_id)


def router_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "router")
    routing = route_task(state["user_message"])

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.ROUTING_COMPLETED,
            metadata={
                "task_type": getattr(routing.task_type, "value", str(routing.task_type)),
                "requires_planning": getattr(routing, "requires_planning", False),
                "requires_knowledge": getattr(routing, "requires_knowledge", False),
            },
        )

    return {
        "routing": routing
    }


def format_categorized_context(chunks: list) -> str:
    """
    Groups retrieved CodeChunks into 5 structured sections:
    PRIMARY IMPLEMENTATION, RELATED CODE, TESTS, DEPENDENCIES, CONFIGURATION.
    """
    from backend.indexer.ast_chunker import is_test_path, is_config_path

    primary = []
    related = []
    tests = []
    dependencies = []
    configs = []

    def emit_chunk(c, lines: list) -> None:
        # A "whole_file" chunk (build_patch_context_chunks /
        # whole_file_chunk_for_patch_context) is the file's complete,
        # verbatim content, not one fragment among possibly several -
        # flagged explicitly so the exact-snippet patch-generation prompt
        # knows original_code_snippet must be copied character-for-character
        # from this block, with nothing added, removed, or assumed.
        if getattr(c, "chunk_type", None) == "whole_file":
            lines.append("[COMPLETE FILE CONTENT - verbatim, nothing omitted]")
        sym = f" | SYMBOL: {c.symbol_name}" if c.symbol_name else ""
        lines.append(f"FILE: {c.file_path} (Lines {c.start_line}-{c.end_line}){sym}")
        lines.append(c.content)
        lines.append("---")

    for chunk in chunks:
        fp = chunk.file_path.replace("\\", "/").lower()
        st = getattr(chunk, "symbol_type", "") or ""

        # Collect dependencies from imports
        if getattr(chunk, "imports", None):
            for imp in chunk.imports:
                if imp not in dependencies:
                    dependencies.append(imp)

        if st == "test" or is_test_path(fp):
            tests.append(chunk)
        elif st == "config" or is_config_path(fp):
            configs.append(chunk)
        elif chunk.symbol_name and st in {"class", "function", "method"}:
            primary.append(chunk)
        else:
            related.append(chunk)

    sections = []

    if primary:
        lines = ["[PRIMARY IMPLEMENTATION]"]
        for c in primary:
            emit_chunk(c, lines)
        sections.append("\n".join(lines))

    if related:
        lines = ["[RELATED CODE]"]
        for c in related:
            emit_chunk(c, lines)
        sections.append("\n".join(lines))

    if tests:
        lines = ["[TESTS]"]
        for c in tests:
            emit_chunk(c, lines)
        sections.append("\n".join(lines))

    if dependencies:
        lines = ["[DEPENDENCIES]"]
        lines.extend(dependencies[:20])
        lines.append("---")
        sections.append("\n".join(lines))

    if configs:
        lines = ["[CONFIGURATION]"]
        for c in configs:
            emit_chunk(c, lines)
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def planner_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "planner")
    repo_context = state.get("repo_context")
    repo_evidence = None
    if repo_context:
        repo_evidence = {
            "files": list(dict.fromkeys(c.file_path for c in repo_context)),
            "symbols": list(dict.fromkeys(c.symbol_name for c in repo_context if c.symbol_name)),
            "tests": list(dict.fromkeys(c.file_path for c in repo_context if getattr(c, "symbol_type", "") == "test" or "test" in c.file_path.lower())),
        }
    elif state.get("project_id"):
        project_path = _resolve_project_path(state)
        if project_path is not None and project_path.exists():
            try:
                from backend.indexer.scanner import scan_repository
                scanned = scan_repository(str(project_path))
                files = [sf.relative_path for sf in scanned]
                tests = [f for f in files if "test" in f.lower()]
                repo_evidence = {"files": files[:10], "symbols": [], "tests": tests[:5]}
            except Exception:
                pass

    with collect_usage() as usage:
        plan = create_plan(
            state["user_message"],
            state["routing"],
            repo_evidence=repo_evidence,
        )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.PLAN_CREATED,
            metadata={"goal": plan.goal, "steps_count": len(plan.steps)},
        )

    return {
        "plan": plan,
        "metrics": merge_usage(state.get("metrics"), usage),
    }


def knowledge_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "knowledge")
    import time
    start_time = time.time()

    project_id = state.get("project_id")

    if not project_id:
        return {
            "knowledge": KnowledgeAnswer(
                answer="No project_id was provided to retrieve project context.",
                sources=[],
                sufficient_context=False,
            ),
            "repo_context": [],
            "rag_status": "RAG_INSUFFICIENT_CONTEXT",
        }

    from backend.indexer.scanner import scan_repository
    from backend.indexer.ast_chunker import chunk_file
    from backend.indexer.retriever import SimpleBM25Index, HybridRetriever
    from backend.indexer.models import CodeChunk
    from backend.rag.evaluator import RetrievalEvaluator, QueryRewriter
    from backend.schemas.rag import RAGTelemetry, RAGStatus

    sparse_candidates = []
    project_path = _resolve_project_path(state)

    all_chunks = []
    if project_path is not None and project_path.exists():
        try:
            scanned_files = scan_repository(str(project_path))
            for sf in scanned_files:
                chunks = chunk_file(sf.absolute_path, sf.relative_path)
                all_chunks.extend(chunks)

            if all_chunks:
                bm25 = SimpleBM25Index(all_chunks)
                sparse_pairs = bm25.search(state["user_message"], top_n=20)
                sparse_candidates = [pair[0] for pair in sparse_pairs]
        except Exception as e:
            print(f"BM25 retrieval error: {e}")

    # Dense retrieval is run once here (k=20, for hybrid BM25+dense fusion
    # below) and its top results are handed to answer_from_project instead
    # of it loading the same FAISS index and re-running the same query a
    # second time. `docs` stays None (not just empty) if this attempt
    # raises before assignment, so answer_from_project can tell "retrieval
    # genuinely failed" apart from "retrieval succeeded with zero matches"
    # and fall back to its own retrieval/error-handling accordingly.
    #
    # organization_id is deliberately read with NO default here (unlike
    # other nodes' workspace-path resolution): defaulting a missing value
    # to "default-org" would silently query/read that tenant's own vector
    # store for a run that never actually authenticated as it - defeating
    # get_vector_store_path's fail-closed ValueError on a genuinely missing
    # organization_id. A real None/missing value here correctly raises
    # inside load_project_index/answer_from_project below, both of which
    # are already wrapped in this function's own exception handling, so
    # the graph degrades to "no context found" exactly as it does for a
    # missing project_id - never a silent cross-tenant read.
    organization_id = state.get("organization_id")
    docs = None
    dense_candidates = []
    try:
        from backend.rag.retriever import load_project_index
        vector_store = load_project_index(project_id, organization_id=organization_id)
        docs = vector_store.similarity_search(state["user_message"], k=20)
        for doc in docs:
            dense_candidates.append(
                CodeChunk(
                    file_path=doc.metadata.get("file") or doc.metadata.get("source", "unknown"),
                    chunk_type=doc.metadata.get("symbol_type") or "dense",
                    symbol_name=doc.metadata.get("symbol"),
                    content=doc.page_content,
                    start_line=doc.metadata.get("line_start") or doc.metadata.get("start_line", 1),
                    end_line=doc.metadata.get("line_end") or doc.metadata.get("end_line", 1),
                    docstring=None,
                    decorators=[],
                    module=doc.metadata.get("module"),
                    symbol_type=doc.metadata.get("symbol_type"),
                    source_hash=doc.metadata.get("source_hash"),
                )
            )
    except Exception as e:
        print(f"Vector search retrieval error: {e}")

    try:
        knowledge = answer_from_project(
            project_id=project_id,
            question=state["user_message"],
            documents=docs,
            organization_id=organization_id,
        )
    except Exception as e:
        knowledge = KnowledgeAnswer(
            answer=f"Vector store not indexed yet: {e}",
            sources=[],
            sufficient_context=False,
        )

    retriever = HybridRetriever()
    search_results = retriever.retrieve(
        sparse_candidates=sparse_candidates,
        dense_candidates=dense_candidates,
        top_k=4,
    )
    repo_context = [res.chunk for res in search_results]
    scores = [res.score for res in search_results]

    current_query = state["user_message"]
    eval_result = RetrievalEvaluator.evaluate(
        issue_text=state["user_message"],
        query=current_query,
        chunks=repo_context,
        scores=scores,
    )

    query_rewrite = None
    attempt = 1

    # Bounded query rewriting if retrieval evaluation is INSUFFICIENT
    if eval_result.status == "INSUFFICIENT":
        rewritten = QueryRewriter.rewrite(
            issue_text=state["user_message"],
            current_query=current_query,
            missing_context=eval_result.missing_context,
            attempt=attempt,
        )
        if rewritten and rewritten != current_query:
            query_rewrite = rewritten
            attempt += 1
            if all_chunks:
                try:
                    bm25 = SimpleBM25Index(all_chunks)
                    sparse_pairs = bm25.search(rewritten, top_n=20)
                    sparse_candidates = [pair[0] for pair in sparse_pairs]
                except Exception:
                    pass

            try:
                if 'vector_store' in locals():
                    docs = vector_store.similarity_search(rewritten, k=20)
                    dense_candidates = [
                        CodeChunk(
                            file_path=doc.metadata.get("file") or doc.metadata.get("source", "unknown"),
                            chunk_type=doc.metadata.get("symbol_type") or "dense",
                            symbol_name=doc.metadata.get("symbol"),
                            content=doc.page_content,
                            start_line=doc.metadata.get("line_start") or doc.metadata.get("start_line", 1),
                            end_line=doc.metadata.get("line_end") or doc.metadata.get("end_line", 1),
                            docstring=None,
                            decorators=[],
                        ) for doc in docs
                    ]
            except Exception:
                pass

            new_search_results = retriever.retrieve(
                sparse_candidates=sparse_candidates,
                dense_candidates=dense_candidates,
                top_k=4,
            )
            if new_search_results:
                repo_context = [res.chunk for res in new_search_results]
                scores = [res.score for res in new_search_results]
                eval_result = RetrievalEvaluator.evaluate(
                    issue_text=state["user_message"],
                    query=rewritten,
                    chunks=repo_context,
                    scores=scores,
                )

    rag_status = (
        RAGStatus.RAG_INSUFFICIENT_CONTEXT.value
        if eval_result.status == "INSUFFICIENT"
        else RAGStatus.RAG_RETRIEVAL_SUCCESS.value
    )

    duration = round(time.time() - start_time, 3)
    telemetry = RAGTelemetry(
        retrieval_query=state["user_message"],
        retrieval_attempt=attempt,
        documents_retrieved=len(sparse_candidates) + len(dense_candidates),
        scores=[round(s, 4) for s in scores],
        selected_documents=[c.file_path for c in repo_context],
        evaluation_status=eval_result.status,
        evaluation_confidence=eval_result.confidence,
        query_rewrite=query_rewrite,
        missing_context=eval_result.missing_context,
        retrieval_duration_seconds=duration,
    )

    is_sufficient = eval_result.status != "INSUFFICIENT"
    knowledge = KnowledgeAnswer(
        answer=knowledge.answer,
        sources=list(dict.fromkeys(knowledge.sources + [c.file_path for c in repo_context])),
        sufficient_context=is_sufficient,
    )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.RAG_COMPLETED,
            metadata={"rag_status": rag_status, "docs_count": len(repo_context)},
        )

    return {
        "knowledge": knowledge,
        "repo_context": repo_context,
        "rag_evaluation": eval_result,
        "rag_telemetry": telemetry,
        "rag_status": rag_status,
    }


def _safe_repo_relative_target(project_path, file_path: str):
    """
    Resolves an LLM-generated, repository-relative `file_path` to an
    absolute path strictly inside `project_path`, or returns None when it
    is unsafe: empty, a '..' traversal, absolute, drive-qualified, or a
    symlink that resolves outside `project_path`.

    Thin wrapper around the single shared containment check
    (backend.policy.path_filter.safe_repo_relative_path) every LLM-
    controlled file_path read/write site in this codebase must use -
    kept here under its original name since call sites throughout this
    module already reference it, but no longer duplicates the logic
    itself.
    """
    from backend.policy.path_filter import safe_repo_relative_path

    return safe_repo_relative_path(project_path, file_path)


def _materialize_developer_changes(
    changes, project_id: str, organization_id: str = "default-org", snapshot_sink: Optional[dict] = None,
    true_original_sink: Optional[dict] = None,
) -> list:
    """
    Writes each non-empty FileChange directly to disk under
    workspace/<organization_id>/<project_id> and returns the corresponding
    FilePatch list, using the same full-file-write convention as the
    exact-snippet patch path (empty original_code_snippet - see
    backend/developer/patcher.py).

    Shared by developer_node's own no-repo-context fallback and
    revision_node's blind-revision fallback (when generate_revision_patches
    either didn't run or produced nothing), so a revised DeveloperResult
    from revise_code_changes is materialized exactly the same way an
    initial one is, instead of a second, divergent implementation.

    Invariant preserved: an effective (non-empty-content) change must
    either become a FilePatch or this call fails explicitly (ValueError) -
    it must never silently vanish. The same applies if organization_id/
    project_id don't resolve to a safe workspace path at all.

    If `snapshot_sink` is given, records each file's on-disk content from
    immediately BEFORE this write into it (file_path -> content, "" for a
    genuinely new file). This is the file's true pre-patch state - qa_node
    reads it right back off disk moments later, after this write already
    happened, so anything that needs to compare against "before this
    patch" (see QualityPipeline.check_patch_scope) cannot rely on a fresh
    disk read at that point - it would just see this write's own result.

    If `true_original_sink` is also given, records the SAME pre-write
    content into it too, but only the first time a given file_path is
    seen (setdefault, never overwritten) - unlike snapshot_sink, which is
    meant to always reflect the immediate pre-write state, this one must
    keep reflecting the file's content from before this RUN ever touched
    it, across every later write to the same file within the same run
    (e.g. a subsequent, still-blind revision attempt on the same file).
    """
    from backend.developer.models import FilePatch

    project_path = resolve_workspace_path(organization_id, project_id)
    if project_path is None:
        raise ValueError(
            f"Refusing to materialize generated changes: organization_id/"
            f"project_id did not resolve to a safe workspace path for "
            f"project '{project_id}'."
        )

    patches = []
    for ch in changes:
        # DELETE changes (and any change with no content) have nothing to
        # materialize via this write path - a legitimate no-op, not a
        # dropped "effective" change.
        if not ch.content:
            continue

        abs_f = _safe_repo_relative_target(project_path, ch.file_path)
        if abs_f is None:
            raise ValueError(
                f"Refusing to materialize generated file change: "
                f"'{ch.file_path}' is not a safe repository-relative path."
            )

        if snapshot_sink is not None or true_original_sink is not None:
            before_content = ""
            if abs_f.exists():
                try:
                    with open(abs_f, "r", encoding="utf-8", errors="ignore") as f:
                        before_content = f.read()
                except Exception:
                    pass
            if snapshot_sink is not None:
                snapshot_sink[ch.file_path] = before_content
            if true_original_sink is not None:
                true_original_sink.setdefault(ch.file_path, before_content)

        from backend.developer.patch_scope import (
            PatchWrapperArtifactError,
            detect_patch_wrapper_artifacts,
        )

        wrapper_reason = detect_patch_wrapper_artifacts(ch.content)
        if wrapper_reason:
            # Never write a hallucinated patch-tool wrapper (e.g. "*** Begin
            # Patch") to disk as if it were real file content - see
            # PatchWrapperArtifactError's docstring for how callers route
            # this into the existing bounded revision loop instead.
            raise PatchWrapperArtifactError(ch.file_path, wrapper_reason)

        try:
            abs_f.parent.mkdir(parents=True, exist_ok=True)
            with open(abs_f, "w", encoding="utf-8") as f:
                f.write(ch.content)
        except OSError as e:
            raise ValueError(
                f"Failed to materialize generated file change for "
                f"'{ch.file_path}': {e}"
            ) from e

        patches.append(
            FilePatch(
                file_path=ch.file_path,
                original_code_snippet="",
                updated_code_snippet=ch.content,
                explanation=ch.reason,
            )
        )
    return patches


def _advisory_developer_result(generated_patches, fallback):
    """
    Builds the DeveloperResult shown to the advisory QA reviewer
    (review_code_changes) and the revision agent (revise_code_changes) so
    they judge/revise the file(s) actually validated and staged
    (generated_patches) - not the separate, context-blind DeveloperResult
    produced by generate_code_changes/revise_code_changes, which never
    receives repo_context or file content and is otherwise the only
    representation of "the proposed implementation" those two prompts see.

    Falls back to `fallback` unchanged when there are no generated_patches
    (e.g. a task with nothing to patch at all) - existing blind-
    developer_result behavior for that case is untouched. Does not alter
    generated_patches, SafePatcher validation, or objective QA checks -
    those already operate on generated_patches directly and are
    unaffected by this advisory-only, LLM-facing representation.
    """
    if not generated_patches:
        return fallback

    from backend.schemas.developer import DeveloperResult, FileChange

    changes = [
        FileChange(
            file_path=p.file_path,
            change_type="MODIFY",
            content=p.updated_code_snippet,
            reason=(
                f"{p.explanation}\n\n"
                f"--- BEFORE ---\n{p.original_code_snippet or '(new file / whole-file replacement)'}\n"
                f"--- AFTER ---\n{p.updated_code_snippet}"
            ),
        )
        for p in generated_patches
    ]
    return DeveloperResult(
        summary=(
            f"Applied {len(changes)} validated patch(es): "
            f"{', '.join(c.file_path for c in changes)}."
        ),
        changes=changes,
        requires_testing=True,
        notes=[],
    )


def developer_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "developer")
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    knowledge = state.get("knowledge")
    if knowledge is None:
        knowledge = KnowledgeAnswer(
            answer="No external project context required.",
            sources=[],
            sufficient_context=True,
        )

    with collect_usage() as usage:
        # generate_code_changes() (the context-blind path - see
        # backend/agents/developer.py) is deliberately NOT called here
        # unconditionally. It only runs further below, as a genuine
        # fallback, when the context-aware exact-snippet/whole-file patch
        # path (below) produced no patches at all - calling it upfront on
        # every run wasted a second LLM round trip whenever the
        # context-aware path succeeded, and its own prompt (plan + knowledge
        # only, no repo_context/file content) is so underspecified that a
        # real Groq fallback call against it fails with "Tool choice is
        # required, but model did not call a tool" - the model has nothing
        # concrete to work with.
        developer_result = None

        repo_context = state.get("repo_context")

        # Resolved only when a real project_id is present, matching the
        # self-scan gate below - never inferred/defaulted, so consolidation
        # can't accidentally read a different project's workspace when
        # repo_context was supplied without a project_id.
        project_path = _resolve_project_path(state)

        if not repo_context and project_path is not None and project_path.exists():
            try:
                from backend.indexer.scanner import scan_repository
                from backend.indexer.ast_chunker import chunk_file
                scanned_files = scan_repository(str(project_path))
                all_chunks = []
                for sf in scanned_files:
                    all_chunks.extend(chunk_file(sf.absolute_path, sf.relative_path))
                repo_context = all_chunks
            except Exception as e:
                print(f"Developer node context scan notice: {e}")

        generated_patches = []
        # Each file's true content immediately BEFORE this node writes any
        # patch to it - captured here because qa_node reads the same path
        # back off disk moments later, by which point this node has
        # already overwritten it with the patched result. Without this,
        # any later check that needs "the original" (e.g.
        # QualityPipeline.check_patch_scope's additive-deletion guard)
        # would silently compare the patched file against itself.
        pre_patch_snapshots: dict = {}
        # Carried forward from a prior node call only for a resumed/looped
        # state (developer_node itself only ever runs once per run - see
        # route_after_developer - so in practice this starts empty and is
        # populated below, once per file, and never touched again).
        true_original_snapshots: dict = dict(state.get("true_original_snapshots") or {})
        recoverable_patch_failure_check = None
        developer_qa_result = None

        if repo_context:
            # Give the exact-snippet patch-generation prompt below one
            # authoritative, contiguous view of small non-Python files
            # (README.md, config, etc.) instead of fallback_chunk()'s
            # overlapping 100-line fragments - the LLM was generating
            # original_code_snippet anchors from fragmented/duplicated
            # context that didn't byte-match the real file. RAG/indexing
            # (chunk_file/fallback_chunk themselves) are untouched.
            if project_path is not None and project_path.exists():
                try:
                    from backend.indexer.ast_chunker import build_patch_context_chunks
                    repo_context = build_patch_context_chunks(repo_context, str(project_path))
                except Exception as e:
                    print(f"Patch context consolidation notice: {e}")
            context_str = format_categorized_context(repo_context)
            rag_eval = state.get("rag_evaluation")
            context_warning = ""
            if rag_eval and getattr(rag_eval, "status", None) == "INSUFFICIENT":
                context_warning = "\nWARNING: Repository context is evaluated as INSUFFICIENT. Do not invent non-existent files or functions. Only modify verified code.\n"

            prompt = f"""
You are the Developer Agent Patch Generator.
Your task is to generate precise and safe code patches for the user request and execution plan.
{context_warning}
USER REQUEST:
{state["user_message"]}

EXECUTION PLAN:
{plan.model_dump_json(indent=2)}

STRUCTURED REPOSITORY CONTEXT:
{context_str}

Return a list of precise FilePatches. For each patch, provide the file path, the exact original code snippet to be replaced, and the updated code snippet.

Modify ONLY what the user's request actually asks for. Preserve all unrelated existing content exactly as shown - do not rewrite, reformat, reorder, or paraphrase any part of a file the user did not ask you to change, and do not remove existing sections.

For any file shown above marked [COMPLETE FILE CONTENT - verbatim, nothing omitted], that block IS the file's entire current content.

- If the request is additive (e.g. "add", "append", "insert", "add a section", "update section") and does NOT explicitly ask for a rewrite, replacement, restructuring, or regeneration of the file: do NOT use the whole-file convention below. Instead, quote a short, exact anchor copied verbatim from the shown content (e.g. the file's last few lines, or an existing heading) as original_code_snippet, and set updated_code_snippet to that same anchor plus only the new content - so every other part of the file is left byte-for-byte untouched.
- Only when the user explicitly asks to rewrite, replace, completely restructure, or regenerate the file should you use the whole-file convention: set original_code_snippet to an empty string ("") and put the file's complete new content - the whole file, not just the changed part - in updated_code_snippet. An empty original_code_snippet always means "replace the entire file with updated_code_snippet" - use it deliberately, never as a default just because the complete file was shown to you.
"""
            from backend.services.llm import get_llm
            from backend.developer.models import FilePatch
            from pydantic import BaseModel, Field

            class PatchResponse(BaseModel):
                patches: list[FilePatch] = Field(description="List of proposed file patches.")

            llm = get_llm()
            patch_result = invoke_structured(llm, PatchResponse, prompt)
            patches = patch_result.patches

            # Perform AST pre-flight validation
            from backend.developer.patcher import SafePatcher

            for patch in patches:
                project_id = state.get("project_id", "test_project")
                project_path = resolve_workspace_path(
                    state.get("organization_id", "default-org"), project_id
                )
                if project_path is None:
                    raise ValueError(
                        f"Refusing to apply generated patch: organization_id/"
                        f"project_id did not resolve to a safe workspace path "
                        f"for project '{project_id}'."
                    )

                # SECURITY: patch.file_path is LLM-controlled. Every
                # filesystem access it drives must be contained inside
                # project_path - never a raw project_path / patch.file_path
                # join, which a '../' or absolute path (or a symlink inside
                # the cloned repo) could escape. An unsafe path fails this
                # patch explicitly rather than silently skipping it or
                # falling back to writing anywhere.
                abs_file_path = _safe_repo_relative_target(project_path, patch.file_path)
                if abs_file_path is None:
                    raise ValueError(
                        f"Refusing to apply generated patch: "
                        f"'{patch.file_path}' is not a safe repository-relative path."
                    )

                source_content = ""
                if abs_file_path.exists():
                    try:
                        with open(abs_file_path, "r", encoding="utf-8", errors="ignore") as f:
                            source_content = f.read()
                    except Exception:
                        pass
                pre_patch_snapshots[patch.file_path] = source_content
                true_original_snapshots.setdefault(patch.file_path, source_content)

                val_result = SafePatcher.apply_patch(source_content, patch)
                if not val_result.is_valid:
                    # Mirror QualityPipeline.check_ast's own distinction
                    # (backend/qa/pipeline.py): a non-Python file's mismatch
                    # is exclusively "the anchor didn't match the real file"
                    # - never a Python syntax/AST violation, since SafePatcher
                    # only runs its Python-syntax check after a successful
                    # snippet replacement. That is a recoverable
                    # patch-generation problem (the LLM's anchor was stale or
                    # fragmented), not a security or code-safety violation,
                    # so it is routed to the existing bounded revision loop
                    # instead of hard-failing the run. A .py file's failure
                    # keeps hard-failing unchanged, since it may reflect a
                    # genuine AST/syntax problem that must not be bypassed.
                    if patch.file_path.lower().endswith(".py"):
                        raise ValueError(
                            f"AST pre-flight validation failed for {patch.file_path}: {val_result.syntax_errors}"
                        )
                    recoverable_patch_failure_check = QualityCheck(
                        name="ast",
                        status=QualityCheckStatus.FAIL.value,
                        exit_code=1,
                        stderr_summary=f"{patch.file_path}: {', '.join(val_result.syntax_errors or ['Target snippet not found'])}",
                        reason="Patch pre-flight validation failed (target snippet not found; not a Python syntax error).",
                        category=FailureCategory.PATCH_APPLICATION_FAILURE.value,
                    )
                    # Nothing from this batch is trusted once one patch's
                    # anchor didn't match - never apply the rest partially,
                    # and never let a failed batch look like a success.
                    generated_patches = []
                    break
                if val_result.applied_content is not None:
                    from backend.developer.patch_scope import detect_patch_wrapper_artifacts

                    wrapper_reason = detect_patch_wrapper_artifacts(val_result.applied_content)
                    if wrapper_reason:
                        # The LLM hallucinated a patch-editor tool's wrapper
                        # syntax (e.g. "*** Begin Patch") instead of real
                        # file content. Never write that to disk - treat it
                        # exactly like a SafePatcher anchor mismatch above
                        # (a recoverable patch-generation problem, not a
                        # security violation), so the run gets a chance to
                        # regenerate instead of corrupting the workspace
                        # file with tool syntax.
                        recoverable_patch_failure_check = QualityCheck(
                            name="ast",
                            status=QualityCheckStatus.FAIL.value,
                            exit_code=1,
                            stderr_summary=f"{patch.file_path}: {wrapper_reason}",
                            reason="Patch pre-flight validation failed (hallucinated patch-tool wrapper syntax; not real file content).",
                            category=FailureCategory.PATCH_APPLICATION_FAILURE.value,
                        )
                        generated_patches = []
                        break
                    try:
                        abs_file_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(abs_file_path, "w", encoding="utf-8") as f:
                            f.write(val_result.applied_content)
                    except Exception as e:
                        print(f"Notice: could not write patched file: {e}")
                generated_patches.append(patch)

        # Skipped when the exact-snippet path hit a recoverable mismatch
        # above: that failure is real feedback for the revision loop, and
        # silently replacing it with a fresh blind patch here would erase
        # the classification before route_after_developer ever sees it.
        # This is also the ONLY place generate_code_changes() (the
        # context-blind path) is ever called - lazily, only when the
        # context-aware path above produced nothing to fall back on.
        if not generated_patches and not recoverable_patch_failure_check:
            developer_result = generate_code_changes(
                user_request=state["user_message"],
                plan=plan,
                knowledge=knowledge,
            )

        if not generated_patches and not recoverable_patch_failure_check and developer_result and developer_result.changes:
            from backend.developer.patch_scope import PatchWrapperArtifactError

            try:
                generated_patches = _materialize_developer_changes(
                    developer_result.changes,
                    state.get("project_id", "test_project"),
                    state.get("organization_id", "default-org"),
                    snapshot_sink=pre_patch_snapshots,
                    true_original_sink=true_original_snapshots,
                )
            except PatchWrapperArtifactError as e:
                recoverable_patch_failure_check = QualityCheck(
                    name="ast",
                    status=QualityCheckStatus.FAIL.value,
                    exit_code=1,
                    stderr_summary=str(e),
                    reason="Patch pre-flight validation failed (hallucinated patch-tool wrapper syntax; not real file content).",
                    category=FailureCategory.PATCH_APPLICATION_FAILURE.value,
                )
                generated_patches = []

        if recoverable_patch_failure_check is not None:
            from backend.qa.judge import StructuredQAJudge
            developer_qa_result = StructuredQAJudge.evaluate(checks=[recoverable_patch_failure_check])

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.DEVELOPMENT_COMPLETED,
            metadata={"patches_count": len(generated_patches)},
        )

    result = {
        # Derived from the real, validated generated_patches whenever they
        # exist (see _advisory_developer_result) rather than the separate
        # context-blind developer_result - this is what the HITL approval
        # payload and the git commit message actually show/use downstream,
        # so it must describe the patch that was really applied, not an
        # independent, potentially-unrelated blind guess. Falls back to the
        # blind developer_result unchanged when there are no generated_patches.
        "developer_result": _advisory_developer_result(generated_patches, developer_result),
        "plan": plan,
        "generated_patches": generated_patches,
        # Persisted (mirroring knowledge_node's existing behavior) so
        # revision_node's own context-aware patch path
        # (generate_revision_patches) can actually run on a knowledge-skip
        # task - previously this was a local variable, discarded at the end
        # of every developer_node call. Contains only already-scanned
        # source file chunks (backend/indexer/scanner.py already excludes
        # .env/*.pem/*.key/secrets.json); no credentials or repository
        # tokens ever pass through this value.
        "repo_context": repo_context,
        "metrics": merge_usage(state.get("metrics"), usage),
        "pre_patch_snapshots": pre_patch_snapshots,
        "true_original_snapshots": true_original_snapshots,
    }
    if developer_qa_result is not None:
        # Set only for the narrow recoverable patch-application mismatch
        # above (never for a normal successful attempt) so
        # route_after_developer can send this run through the existing
        # bounded revision loop instead of qa_node. revision_node already
        # knows how to consume qa_result.summary via ErrorTraceAnalyzer -
        # reusing the same field means no new consumption path is needed.
        result["qa_result"] = developer_qa_result
    return result


def qa_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "qa")
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    # 1. LLM Semantic Review - judged against the ACTUAL validated patch
    # (generated_patches) whenever one exists, not the separate,
    # context-blind DeveloperResult from generate_code_changes (see
    # _advisory_developer_result). The reviewer remains fully advisory -
    # its FAIL still participates in StructuredQAJudge's existing
    # aggregation policy (backend/qa/judge.py) unchanged.
    llm_qa_result = review_code_changes(
        user_request=state["user_message"],
        plan=plan,
        developer_result=_advisory_developer_result(
            state.get("generated_patches"), state["developer_result"]
        ),
    )

    project_id = state.get("project_id", "test_project")
    from backend.qa.pipeline import QualityPipeline
    from backend.qa.judge import StructuredQAJudge

    project_path = resolve_workspace_path(state.get("organization_id", "default-org"), project_id)
    if project_path is None:
        raise ValueError(
            f"Refusing to run QA: organization_id/project_id did not "
            f"resolve to a safe workspace path for project '{project_id}'."
        )

    patches = state.get("generated_patches") or []

    # 2. Execute Multi-Check Quality Pipeline (AST, pytest, security, lint, typecheck)
    run_id = state.get("run_id")
    org_id = state.get("organization_id")
    cancel_check = None
    if run_id:
        from backend.observability.store import telemetry_store as _telemetry_store
        cancel_check = lambda: _telemetry_store.is_cancel_requested(run_id, org_id)
    checks, test_result = QualityPipeline.run_all(
        repo_path=str(project_path),
        patches=patches,
        timeout=30.0,
        cancel_check=cancel_check,
        user_request=state["user_message"],
        original_file_snapshots=state.get("pre_patch_snapshots"),
        true_original_snapshots=state.get("true_original_snapshots"),
    )

    # 3. Structured QA Judge Evaluation with Strict Objective Priority
    qa_result = StructuredQAJudge.evaluate(
        checks=checks,
        test_result=test_result,
        patches=patches,
        llm_qa_result=llm_qa_result,
    )

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.QA_COMPLETED,
            metadata={
                "qa_status": getattr(qa_result, "status", None),
                "summary": (getattr(qa_result, "summary", "") or "")[:100],
            },
        )

    return {
        "qa_result": qa_result,
        "test_result": test_result,
    }


def route_after_router(state: AgentState) -> str:
    routing = state["routing"]

    if routing.task_type == TaskType.GENERAL:
        return "end"

    if routing.requires_planning:
        return "planner"

    if routing.task_type == TaskType.KNOWLEDGE_SEARCH or routing.requires_knowledge:
        return "knowledge"

    return "developer"


def route_after_planner(state: AgentState) -> str:
    routing = state.get("routing")
    if routing and routing.requires_knowledge and state.get("project_id"):
        return "knowledge"

    return "developer"


def route_after_knowledge(state: AgentState) -> str:
    routing = state.get("routing")
    if routing and routing.task_type == TaskType.KNOWLEDGE_SEARCH and not routing.requires_planning:
        return "end"

    return "developer"


MAX_REVISIONS = 3


def route_after_developer(state: AgentState) -> str:
    """
    Sends a run to the existing bounded revision loop instead of qa_node
    only for the narrow recoverable failure developer_node itself can
    detect and classify (FailureCategory.PATCH_APPLICATION_FAILURE - a
    generated patch's anchor didn't match the real file, on a non-Python
    file). developer_node runs exactly once per run (nothing loops back
    to it), so revision_count is always 0 here; a bounded-retries branch
    is unnecessary; MAX_REVISIONS is still enforced normally afterwards,
    by qa_router, for every subsequent revision -> qa cycle.
    Every other outcome - success, or any hard-fail ValueError raised
    above (unsafe workspace path, unsafe file path, genuine AST/syntax
    violation) - is unaffected and continues to qa_node exactly as before.
    """
    qa_result = state.get("qa_result")
    if (
        qa_result is not None
        and (qa_result.status or "").strip().upper() == "FAIL"
        and getattr(qa_result, "failure_category", None) == FailureCategory.PATCH_APPLICATION_FAILURE.value
    ):
        return "revision"
    return "qa"


def qa_router(state: AgentState) -> str:
    qa_result = state.get("qa_result")
    revision_count = state.get("revision_count", 0)

    if qa_result is None:
        return "max_retries" if revision_count >= MAX_REVISIONS else "fail"

    status = (qa_result.status or "").strip().upper()

    if status == "PASS":
        return "pass"

    if revision_count >= MAX_REVISIONS:
        return "max_retries"

    return "fail"


def revision_node(state: AgentState) -> dict:
    check_cancelled(state)
    mark_activity(state, "revision")
    import time
    start_rev = time.time()

    from backend.agents.developer import revise_code_changes
    from backend.agents.revision import generate_revision_patches
    from backend.revision.models import RevisionAttempt, RevisionHistory
    from backend.revision.analyzer import ErrorTraceAnalyzer
    from backend.schemas.developer import DeveloperResult
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.core.config import LLM_MODEL_NAME, LLM_PROVIDER

    revision_count = state.get("revision_count", 0)

    # 1. Fallback plan persistence
    plan = state.get("plan")
    if plan is None:
        plan = ExecutionPlan(
            goal=state["user_message"],
            steps=[],
            success_criteria="Complete requested task directly.",
        )

    # 2. Extract and analyze error traces from sandbox execution or QA summary
    test_result = state.get("test_result")
    qa_result = state.get("qa_result")

    if test_result is not None:
        analysis = ErrorTraceAnalyzer.analyze_test_result(test_result)
    elif qa_result is not None and getattr(qa_result, "summary", None):
        analysis = ErrorTraceAnalyzer.analyze(qa_result.summary)
    else:
        analysis = ErrorTraceAnalyzer.analyze("")

    # 3. Retrieve or initialize revision history
    existing_history = state.get("revision_history")
    if existing_history is not None:
        revision_history = existing_history
    else:
        revision_history = RevisionHistory(max_retries=MAX_REVISIONS)

    # 4. Fallback previous developer result
    prev_result = state.get("developer_result")
    if prev_result is None:
        prev_result = DeveloperResult(
            summary="Initial implementation attempt.",
            changes=[],
            requires_testing=True,
            notes=[],
        )

    # Show the revision agent the ACTUAL previous patch it needs to fix
    # (see _advisory_developer_result) - not the separate, context-blind
    # developer_result, which has no file content and would otherwise
    # correctly (but unhelpfully) report that it lacks the context to
    # revise anything.
    advisory_prev_result = _advisory_developer_result(
        state.get("generated_patches"), prev_result
    )

    # 5. Revise code changes
    with collect_usage() as usage:
        revised_result = revise_code_changes(
            user_request=state["user_message"],
            plan=plan,
            previous_result=advisory_prev_result,
            qa_result=qa_result,
        )

        # 6. Generate revised patches and perform AST pre-flight validation if repo context is available
        repo_context = state.get("repo_context")
        generated_patches = state.get("generated_patches") or []
        context_aware_patch_produced = False

        if repo_context:
            try:
                revised_patches = generate_revision_patches(
                    user_request=state["user_message"],
                    plan=plan,
                    error_analysis=analysis,
                    repo_context=repo_context,
                    revision_history=revision_history,
                    project_id=state.get("project_id", "test_project"),
                    organization_id=state.get("organization_id", "default-org"),
                )
                if revised_patches:
                    generated_patches = revised_patches
                    context_aware_patch_produced = True
            except Exception as e:
                print(f"Revision patch generation notice: {e}")

        # Blind-path fallback: if the context-aware regeneration above
        # didn't run (no repo_context) or produced nothing THIS attempt,
        # materialize revise_code_changes's own output the same way
        # developer_node's fallback does (_materialize_developer_changes),
        # so a genuinely improved blind revision still becomes the
        # candidate the next QA cycle actually evaluates. Checked against
        # context_aware_patch_produced, not "generated_patches is empty" -
        # generated_patches already holds the PREVIOUS (rejected) attempt
        # on every revision past the first, so that would otherwise never
        # be empty and this fallback would never run.
        pre_patch_snapshots = dict(state.get("pre_patch_snapshots") or {})
        # Unlike pre_patch_snapshots, never popped/replaced below - always
        # the file's content from before this run ever touched it (see
        # AgentState.true_original_snapshots and developer_node). Carried
        # forward as-is regardless of which regeneration path this cycle
        # takes.
        true_original_snapshots = dict(state.get("true_original_snapshots") or {})
        if not context_aware_patch_produced and revised_result and revised_result.changes:
            from backend.developer.patch_scope import PatchWrapperArtifactError

            try:
                generated_patches = _materialize_developer_changes(
                    revised_result.changes,
                    state.get("project_id", "test_project"),
                    state.get("organization_id", "default-org"),
                    snapshot_sink=pre_patch_snapshots,
                    true_original_sink=true_original_snapshots,
                )
            except PatchWrapperArtifactError as e:
                # Never write a hallucinated patch-tool wrapper to disk.
                # Leave generated_patches as the previous (already-rejected)
                # candidate this cycle failed to improve on - qa_node will
                # re-evaluate it, correctly fail it again, and the bounded
                # revision loop continues (or exhausts MAX_REVISIONS)
                # exactly as it already does when generate_revision_patches
                # itself raises, above.
                print(f"Revision patch materialization notice: {e}")
        elif context_aware_patch_produced:
            # generate_revision_patches (backend/agents/revision.py) never
            # writes to disk - it validates each candidate's
            # original_code_snippet anchor against CURRENT live disk
            # content and stops there. Carrying forward an earlier cycle's
            # snapshot for these same files would hand qa_node's
            # check_patch_scope a STALE "original" that this candidate was
            # never generated or validated against: if the anchor doesn't
            # happen to also exist verbatim in that stale snapshot (e.g. a
            # prior attempt already changed the file), SafePatcher reports
            # the patch invalid against it, and check_patch_scope silently
            # skips the patch as "check_ast's concern" - while check_ast
            # (which always reads live disk) validates it just fine,
            # letting a destructive patch through as an unremarked PASS on
            # both checks. Dropping the stale entry here makes
            # check_patch_scope fall back to the same live disk read
            # check_ast and generate_revision_patches already used, which
            # is the correct "before" baseline for a patch that was never
            # written anywhere.
            for p in generated_patches:
                pre_patch_snapshots.pop(p.file_path, None)

    # 7. Record this revision attempt with structured telemetry
    new_revision_count = revision_count + 1
    applied_patch = generated_patches[0] if generated_patches else None
    diagnosis_summary = analysis.diagnosis or (
        qa_result.summary if qa_result else "Revision performed based on test feedback."
    )

    # Identify failed check & category
    failed_check = None
    failure_category = getattr(qa_result, "failure_category", None) or "TEST_FAILURE"
    if qa_result and getattr(qa_result, "checks", None):
        failed_obj = next((c for c in qa_result.checks if c.status == "FAIL"), None)
        if failed_obj:
            failed_check = failed_obj.name

    existing_patch = state.get("generated_patches", [None])[0] if state.get("generated_patches") else None
    previous_patch_hash = state.get("patch_hash") or (
        state.get("git_diff").patch_hash if state.get("git_diff") else None
    ) or (
        GitWorkspaceManager.compute_patch_hash(existing_patch.updated_code_snippet)
        if existing_patch and existing_patch.updated_code_snippet else None
    )

    new_patch_hash = None
    if applied_patch and applied_patch.updated_code_snippet:
        new_patch_hash = GitWorkspaceManager.compute_patch_hash(applied_patch.updated_code_snippet)

    attempt = RevisionAttempt(
        attempt_number=new_revision_count,
        failing_tests=analysis.failing_tests,
        error_traceback=analysis.error_traceback,
        applied_patch=applied_patch,
        diagnosis=diagnosis_summary,
        failed_check=failed_check,
        failure_category=failure_category,
        previous_patch_hash=previous_patch_hash,
        new_patch_hash=new_patch_hash,
        model=LLM_MODEL_NAME or LLM_PROVIDER,
        provider=LLM_PROVIDER,
        duration_seconds=round(time.time() - start_rev, 2),
    )
    revision_history.add_attempt(attempt)

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.REVISION_COMPLETED,
            metadata={"revision_count": new_revision_count, "failure_category": failure_category},
        )

    return {
        "developer_result": revised_result,
        "plan": plan,
        "revision_count": new_revision_count,
        "revision_history": revision_history,
        "generated_patches": generated_patches,
        "metrics": merge_usage(state.get("metrics"), usage),
        "pre_patch_snapshots": pre_patch_snapshots,
        "true_original_snapshots": true_original_snapshots,
    }


def git_prepare_node(state: AgentState) -> dict:
    """
    Prepares a Git diff summary by applying validated patches to the workspace,
    computing unified diffs, and evaluating risk.
    """
    check_cancelled(state)
    mark_activity(state, "git_prepare")
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.vcs.models import GitDiffSummary

    patches = state.get("generated_patches") or []
    project_id = state.get("project_id", "test_project")
    task_id = state.get("run_id") or project_id

    project_path = resolve_workspace_path(state.get("organization_id", "default-org"), project_id)

    if not patches or project_path is None or not project_path.exists():
        # No patches to stage or no workspace; produce an empty diff summary
        branch_name = GitWorkspaceManager.generate_branch_name(task_id)
        empty_hash = GitWorkspaceManager.compute_patch_hash("")
        return {
            "git_diff": GitDiffSummary(
                branch_name=branch_name,
                files_changed=[],
                lines_added=0,
                lines_deleted=0,
                unified_diff="",
                patch_hash=empty_hash,
                risk_score="LOW",
                risk_reasons=["No file patches to apply."],
            ),
            "patch_hash": empty_hash,
        }

    try:
        diff_summary = GitWorkspaceManager.prepare_diff_summary(
            repo_path=str(project_path),
            patches=patches,
            task_id=task_id,
        )
    except Exception as e:
        branch_name = GitWorkspaceManager.generate_branch_name(task_id)
        empty_hash = GitWorkspaceManager.compute_patch_hash("")
        diff_summary = GitDiffSummary(
            branch_name=branch_name,
            files_changed=[p.file_path for p in patches],
            lines_added=0,
            lines_deleted=0,
            unified_diff="",
            patch_hash=empty_hash,
            risk_score="MEDIUM",
            risk_reasons=[f"Diff preparation encountered an error: {e}"],
        )

    return {
        "git_diff": diff_summary,
        "patch_hash": diff_summary.patch_hash,
    }


def policy_node(state: AgentState) -> dict:
    """
    Evaluates organization policy on staged changes and test results
    prior to Human-in-the-Loop approval gate and Git mutation.
    """
    check_cancelled(state)
    mark_activity(state, "policy")
    from backend.policy.evaluator import PolicyEvaluator
    from backend.schemas.policy import PolicyConfig, PolicyDecision

    policy_config = state.get("policy_config")
    if policy_config is None:
        policy_config = PolicyConfig()

    git_diff = state.get("git_diff")
    qa_result = state.get("qa_result")
    project_id = state.get("project_id")

    changed_files = git_diff.files_changed if git_diff else []
    lines_added = git_diff.lines_added if git_diff else 0
    lines_deleted = git_diff.lines_deleted if git_diff else 0
    risk_score = git_diff.risk_score if git_diff else "LOW"
    risk_reasons = git_diff.risk_reasons if git_diff else []
    branch_name = git_diff.branch_name if git_diff else None

    result = PolicyEvaluator.evaluate(
        policy=policy_config,
        repository=project_id,
        branch=branch_name,
        changed_files=changed_files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        risk_score=risk_score,
        risk_reasons=risk_reasons,
        quality_results=qa_result,
        revision_count=state.get("revision_count", 0),
    )

    failed_checks = []
    if qa_result and getattr(qa_result, "checks", None):
        failed_checks = [c.name for c in qa_result.checks if c.status == "FAIL"]

    telemetry = PolicyEvaluator.create_telemetry(
        result=result,
        risk_score=risk_score,
        repository=project_id,
        branch=branch_name,
        changed_files=changed_files,
        lines_added=lines_added,
        lines_deleted=lines_deleted,
        required_checks=policy_config.required_quality_checks,
        failed_checks=failed_checks,
    )

    approval_status = state.get("approval_status")
    if result.decision == PolicyDecision.BLOCK:
        approval_status = "POLICY_BLOCKED"

    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.POLICY_EVALUATED,
            metadata={
                "decision": getattr(result.decision, "value", str(result.decision)),
                "risk_score": str(risk_score),
            },
        )

    return {
        "policy_result": result,
        "policy_telemetry": telemetry,
        "approval_status": approval_status,
    }


def route_after_policy(state: AgentState) -> str:
    """
    Routes after policy evaluation:
    - If the staged diff is a no-op (no patches were ever generated, or the
      generated patch produced no actual content change), route straight to
      cleanup - never to approval. This is the deterministic, graph-level
      guard that stops an empty diff from ever reaching interrupt(): a
      human must never be asked to approve/reject a change that doesn't
      exist. cleanup_node distinguishes this from a real human rejection or
      a policy BLOCK via state["approval"] being unset and git_diff.is_no_op,
      and reports it as NO_CHANGES_NEEDED rather than REJECTED_AND_CLEANED.
      git_commit_node's own is_no_op check (backend/graph/nodes.py) remains
      as a second, independent backstop - this is not the only place that
      enforces the invariant.
    - If policy decision is BLOCK, route immediately to cleanup (prevents Git mutation).
    - If policy decision is ALLOW or REVIEW, route to approval gate.
    """
    from backend.schemas.policy import PolicyDecision

    git_diff = state.get("git_diff")
    if git_diff is not None and git_diff.is_no_op:
        return "cleanup"

    policy_result = state.get("policy_result")
    if policy_result and policy_result.decision == PolicyDecision.BLOCK:
        return "cleanup"

    return "approval"


def approval_node(state: AgentState) -> dict:
    """
    Human-in-the-Loop approval gate. Pauses execution via LangGraph interrupt()
    presenting the Git diff summary, cryptographic patch_hash, policy evaluation,
    and risk assessment, then resumes with an ApprovalDecision bound to the patch hash.
    """
    check_cancelled(state)
    mark_activity(state, "approval")
    from backend.vcs.models import ApprovalDecision

    git_diff = state.get("git_diff")
    developer_result = state.get("developer_result")
    policy_result = state.get("policy_result")
    patch_hash = git_diff.patch_hash if git_diff else ""

    interrupt_payload = {
        "task": "approval_required",
        "diff": git_diff.model_dump() if git_diff else None,
        "patch_hash": patch_hash,
        "developer_result": developer_result.model_dump() if developer_result else None,
        "policy": policy_result.model_dump() if policy_result else None,
        "policy_decision": policy_result.decision.value if policy_result else None,
        "message": "Review the proposed code changes, diff, policy compliance, and risk assessment.",
    }

    decision = interrupt(interrupt_payload)

    # Handle structured ApprovalDecision or simple boolean
    if isinstance(decision, dict):
        approval = ApprovalDecision(**decision)
    elif isinstance(decision, ApprovalDecision):
        approval = decision
    elif decision is True:
        approval = ApprovalDecision(approved=True)
    else:
        rejection_reason = decision if isinstance(decision, str) else "Rejected by reviewer."
        approval = ApprovalDecision(approved=False, rejection_reason=rejection_reason)

    # Cryptographic Approval Integrity Check (fail-closed):
    # An approval can only proceed when there IS a real, non-no-op diff,
    # that diff has a hash, the reviewer's decision carries a hash, and the
    # two match exactly. Any missing piece rejects the approval outright -
    # previously, a decision that simply omitted patch_hash skipped this
    # entire check (`approval.patch_hash and patch_hash` were both
    # required to be truthy just to *enter* the comparison), silently
    # treating "no hash submitted" the same as "hash verified". That is
    # the opposite of fail-closed and is deliberately not preserved.
    def _reject_for_hash_integrity(reason_code: str, detail: str) -> dict:
        rejected = ApprovalDecision(
            approved=False,
            reviewer=approval.reviewer,
            rejection_reason=f"{reason_code}: {detail}",
            patch_hash=approval.patch_hash,
            timestamp=approval.timestamp,
            reviewer_role=approval.reviewer_role,
            user_id=approval.user_id,
        )
        return {
            "approval": rejected,
            "approval_status": reason_code,
        }

    if approval.approved:
        if git_diff is None or git_diff.is_no_op:
            return _reject_for_hash_integrity(
                "NO_OP_OR_MISSING_DIFF",
                "There is no non-empty diff staged for this run to approve.",
            )
        if not patch_hash:
            return _reject_for_hash_integrity(
                "MISSING_DIFF_HASH",
                "The staged diff has no patch_hash to bind this approval to.",
            )
        if not approval.patch_hash:
            return _reject_for_hash_integrity(
                "MISSING_APPROVAL_PATCH_HASH",
                "The approval decision did not submit a patch_hash to verify "
                "against the staged diff.",
            )
        if approval.patch_hash != patch_hash:
            return _reject_for_hash_integrity(
                "PATCH_HASH_MISMATCH",
                f"Approved diff hash '{approval.patch_hash}' does not match "
                f"staged diff hash '{patch_hash}'.",
            )

    # RBAC Authorization Check:
    if approval.approved and approval.reviewer_role:
        from backend.schemas.tenant import Role
        from backend.security.rbac import can_approve_changes

        try:
            role = Role(approval.reviewer_role)
        except Exception:
            role = None

        if role is not None:
            risk_val = None
            if policy_result and hasattr(policy_result, "risk_score"):
                risk_val = float(policy_result.risk_score)
            elif git_diff and git_diff.risk_score:
                if git_diff.risk_score.upper() == "HIGH":
                    risk_val = 80.0
                elif git_diff.risk_score.upper() == "MEDIUM":
                    risk_val = 50.0
                else:
                    risk_val = 20.0

            authorized, auth_reason = can_approve_changes(
                role=role,
                risk_score=risk_val,
                is_security_sensitive=bool(
                    policy_result and getattr(policy_result, "decision", None) == "REQUIRE_APPROVAL"
                ),
            )
            if not authorized:
                approval = ApprovalDecision(
                    approved=False,
                    reviewer=approval.reviewer,
                    rejection_reason=f"APPROVAL_UNAUTHORIZED: {auth_reason}",
                    patch_hash=approval.patch_hash,
                    timestamp=approval.timestamp,
                    reviewer_role=approval.reviewer_role,
                    user_id=approval.user_id,
                )
                return {
                    "approval": approval,
                    "approval_status": "APPROVAL_UNAUTHORIZED",
                }

    status_label = "APPROVED" if approval.approved else "REJECTED"
    return {
        "approval": approval,
        "approval_status": status_label,
    }


def route_after_approval(state: AgentState) -> str:
    """
    Routes after human approval decision:
    - approved -> git_commit (provided policy decision is not BLOCK)
    - rejected or policy blocked -> cleanup
    """
    policy_result = state.get("policy_result")
    if policy_result and getattr(policy_result, "decision", None) == "BLOCK":
        return "cleanup"

    approval = state.get("approval")
    if approval is not None and approval.approved:
        return "git_commit"
    return "cleanup"


def git_commit_node(state: AgentState) -> dict:
    """
    Stages and commits approved changes in the feature branch.
    Skips committing entirely when the diff is a no-op (no files actually
    changed) - e.g. the developer agent found nothing to fix - so an empty
    or junk commit is never pushed or turned into a PR.

    Cryptographically validates workspace drift prior to commit.
    """
    check_cancelled(state)
    mark_activity(state, "git_commit")
    from backend.vcs.git_manager import GitWorkspaceManager

    git_diff = state.get("git_diff")
    if git_diff is None or git_diff.is_no_op:
        return {
            "approval_status": "NO_CHANGES_NEEDED",
        }

    project_id = state.get("project_id", "test_project")
    project_path = resolve_workspace_path(state.get("organization_id", "default-org"), project_id)
    if project_path is None:
        raise ValueError(
            f"Refusing to commit: organization_id/project_id did not "
            f"resolve to a safe workspace path for project '{project_id}'."
        )

    # Pre-commit drift and tamper verification:
    # Ensure working tree diff still matches the approved patch_hash
    if git_diff.patch_hash:
        is_valid, drift_err = GitWorkspaceManager.verify_workspace_drift(
            repo_path=str(project_path),
            expected_diff=git_diff.unified_diff,
            expected_hash=git_diff.patch_hash,
            files_changed=git_diff.files_changed,
        )
        if not is_valid:
            return {
                "approval_status": "PATCH_HASH_MISMATCH",
            }

    commit_message = f"agent: apply approved changes for {project_id}"
    developer_result = state.get("developer_result")
    if developer_result:
        commit_message = f"agent: {developer_result.summary[:80]}"

    # Final check immediately before the irreversible mutation: a
    # cancellation that arrived while drift verification ran above must
    # still stop the commit, not just be noticed at the top of the node.
    check_cancelled(state)

    if git_diff.branch_name:
        GitWorkspaceManager.create_feature_branch(str(project_path), git_diff.branch_name)

    success = GitWorkspaceManager.stage_and_commit(
        repo_path=str(project_path),
        message=commit_message,
    )

    status_val = "COMMITTED" if success else "COMMIT_FAILED"
    run_id = state.get("run_id")
    if run_id:
        telemetry_collector.record_event(
            run_id=run_id,
            organization_id=state.get("organization_id", "default-org"),
            event_type=TelemetryEventType.COMMIT_COMPLETED,
            metadata={"status": status_val, "branch": git_diff.branch_name},
        )

    return {
        "approval_status": status_val,
    }



def cleanup_node(state: AgentState) -> dict:
    """
    Cleans up the feature branch after rejection, a policy BLOCK, or a
    no-op diff (route_after_policy routes all three here - the latter two
    without ever reaching approval_node). Checks out the previous branch,
    discards any uncommitted working-tree changes, and deletes the feature
    branch.
    """
    from backend.vcs.git_manager import GitWorkspaceManager
    from backend.schemas.policy import PolicyDecision

    git_diff = state.get("git_diff")
    project_id = state.get("project_id", "test_project")
    project_path = resolve_workspace_path(state.get("organization_id", "default-org"), project_id)

    branch_name = git_diff.branch_name if git_diff else "agent/task-unknown"

    if project_path is not None:
        GitWorkspaceManager.cleanup_branch(
            repo_path=str(project_path),
            branch_name=branch_name,
        )

    approval = state.get("approval")
    policy_result = state.get("policy_result")
    policy_blocked = bool(policy_result and policy_result.decision == PolicyDecision.BLOCK)

    # A no-op diff is routed here by route_after_policy before approval_node
    # ever runs, so no ApprovalDecision exists - distinguish it from a
    # policy BLOCK (also arrives with no ApprovalDecision, but for a
    # different, more specific reason) so the status accurately reflects
    # "nothing to review" rather than "reviewed and rejected".
    if approval is None and not policy_blocked and git_diff is not None and git_diff.is_no_op:
        return {
            "approval_status": "NO_CHANGES_NEEDED",
        }

    rejection_reason = ""
    if approval and approval.rejection_reason:
        rejection_reason = f" Reason: {approval.rejection_reason}"

    return {
        "approval_status": f"REJECTED_AND_CLEANED{rejection_reason}",
    }