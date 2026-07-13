import json
import math
import socket
import sqlite3
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import run_agent as run_agent_module

from agent.conversation_loop import _record_accepted_provider_response
from agent.session_telemetry import (
    apply_telemetry_result_fields,
    begin_pending_request,
    clear_pending_request,
    record_call_without_usage,
    record_canonical_usage,
)
from agent.transports.codex_app_server_session import (
    CodexAppServerSession,
    TurnResult,
)
from agent.usage_pricing import CanonicalUsage
from hermes_constants import PARTIAL_STREAM_STUB_ID
from hermes_state import SessionDB
from run_agent import AIAgent


NUMERIC_TELEMETRY_FIELDS = {
    "api_calls",
    "api_call_count",
    "input_tokens",
    "output_tokens",
    "session_input_tokens",
    "session_output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "reasoning_tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "last_prompt_tokens",
    "last_real_prompt_tokens",
    "last_completion_tokens",
    "pending_prompt_tokens",
    "pending_generation",
    "pending_started_at",
    "compression_count",
    "context_length",
    "estimated_cost_usd",
}


def _assert_stable_numeric_telemetry(result):
    assert NUMERIC_TELEMETRY_FIELDS <= set(result)
    for field in NUMERIC_TELEMETRY_FIELDS:
        assert isinstance(result[field], (int, float))
        assert not isinstance(result[field], bool)
        assert math.isfinite(result[field])
        assert result[field] >= 0


def _stream_chunk(*, content=None, finish_reason=None, usage=None):
    delta = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning_content=None,
        reasoning=None,
    )
    choice = SimpleNamespace(
        index=0,
        delta=delta,
        finish_reason=finish_reason,
    )
    return SimpleNamespace(
        choices=[] if usage is not None else [choice],
        model="test/model",
        usage=usage,
    )


def _chat_response(
    *,
    content="done",
    finish_reason="stop",
    usage=None,
    response_id="response-1",
    refusal=None,
):
    message = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        refusal=refusal,
    )
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                index=0,
                message=message,
                finish_reason=finish_reason,
            )
        ],
        model="test/model",
        usage=usage,
        id=response_id,
    )


def _make_agent(
    session_db,
    session_id="telemetry-session",
    *,
    context_length=None,
    api_mode="chat_completions",
):
    config = (
        {"model": {"context_length": context_length}}
        if context_length is not None
        else {}
    )
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
        patch("hermes_cli.config.load_config", return_value=config),
    ):
        agent = AIAgent(
            model="test/model",
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            provider="openrouter",
            api_mode=api_mode,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            session_db=session_db,
            session_id=session_id,
            platform="cli",
        )
    return agent


def test_one_streamed_request_persists_live_and_terminal_telemetry(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, context_length=65_536)
        assert agent.context_compressor.context_length == 65_536
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path

        # A non-Mock facade keeps run_conversation on the streaming path. The
        # request-scoped client below supplies an actual chunk iterator.
        agent.client = SimpleNamespace()
        request_client = MagicMock()
        observed_pending = []
        observed_json_pending = []

        def create_stream(**kwargs):
            observed_pending.append(
                database.get_session(agent.session_id)["pending_prompt_tokens"]
            )
            live_snapshot = json.loads(
                (tmp_path / f"session_{agent.session_id}.json").read_text(
                    encoding="utf-8"
                )
            )
            observed_json_pending.append(live_snapshot["pending_prompt_tokens"])
            return iter(
                [
                    _stream_chunk(content="done"),
                    _stream_chunk(finish_reason="stop"),
                    _stream_chunk(
                        usage=SimpleNamespace(
                            prompt_tokens=11,
                            completion_tokens=7,
                            total_tokens=18,
                        )
                    ),
                ]
            )

        request_client.chat.completions.create.side_effect = create_stream
        agent._create_request_openai_client = MagicMock(
            return_value=request_client
        )
        agent._close_request_openai_client = MagicMock()

        result = agent.run_conversation("hello")

        assert result["final_response"] == "done"
        assert request_client.chat.completions.create.call_count == 1
        request_kwargs = request_client.chat.completions.create.call_args.kwargs
        assert request_kwargs["stream"] is True
        assert observed_pending and observed_pending[0] > 0
        assert observed_json_pending and observed_json_pending[0] > 0

        session = database.get_session(agent.session_id)
        assert session["api_call_count"] == 1
        assert session["last_prompt_tokens"] == 11
        assert session["last_completion_tokens"] == 7
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] >= 1
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert session["compression_count"] == 0
        assert session["context_length"] == 65_536

        assert result["api_calls"] == 1
        assert result["api_call_count"] == 1
        # The legacy result key is the monotonic live-display estimate. The
        # exact provider snapshot is exposed separately and persisted below.
        assert result["last_prompt_tokens"] >= 11
        assert result["last_real_prompt_tokens"] == 11
        assert result["last_completion_tokens"] == 7
        assert result["pending_prompt_tokens"] == 0
        assert result["context_length"] == 65_536
        _assert_stable_numeric_telemetry(result)

        snapshot_path = tmp_path / f"session_{agent.session_id}.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        assert snapshot["api_call_count"] == 1
        assert snapshot["last_prompt_tokens"] == 11
        assert snapshot["last_completion_tokens"] == 7
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_generation"] >= 1
        assert snapshot["pending_owner"] is None
        assert snapshot["pending_started_at"] == 0
        assert snapshot["compression_count"] == 0
        assert snapshot["context_length"] == 65_536
    finally:
        database.close()


def test_json_snapshot_uses_success_count_and_keeps_secrets_redacted(tmp_path):
    agent = _make_agent(None, session_id="json-telemetry")
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path
    agent._api_call_count = 9
    agent.session_api_calls = 2
    agent.session_input_tokens = 1_234
    agent.session_output_tokens = 56
    agent.pending_prompt_tokens = 789
    agent.context_compressor.last_prompt_tokens = 7_000
    agent.context_compressor.last_real_prompt_tokens = 7_000
    agent.context_compressor.last_completion_tokens = 300
    agent.context_compressor.compression_count = 2
    agent.context_compressor.context_length = 65_536
    secret = "sk-proj-abc123def456ghi789jkl012mno"

    with patch("agent.redact._REDACT_ENABLED", True):
        agent._save_session_log(
            [{"role": "user", "content": f"Authorization: Bearer {secret}"}]
        )

    snapshot_path = tmp_path / "session_json-telemetry.json"
    raw_snapshot = snapshot_path.read_text(encoding="utf-8")
    snapshot = json.loads(raw_snapshot)
    assert secret not in raw_snapshot
    assert snapshot["api_call_count"] == 2
    assert snapshot["session_input_tokens"] == 1_234
    assert snapshot["session_output_tokens"] == 56
    assert snapshot["last_prompt_tokens"] == 7_000
    assert snapshot["last_completion_tokens"] == 300
    assert snapshot["pending_prompt_tokens"] == 789
    assert snapshot["compression_count"] == 2
    assert snapshot["context_length"] == 65_536


def test_restart_hydrates_cumulative_truth_before_json_and_clears_dead_owner(
    tmp_path,
):
    """Restart/resume hydration precedes early compatibility projection."""
    database = SessionDB(tmp_path / "state.db")
    try:
        database.create_session("resume-me", "cli")
        database.update_token_counts(
            "resume-me",
            input_tokens=1_000,
            output_tokens=250,
            cache_read_tokens=500,
            cache_write_tokens=20,
            reasoning_tokens=75,
            api_call_count=4,
            last_prompt_tokens=12_345,
            last_completion_tokens=678,
            context_length=65_536,
            compression_count=3,
        )
        dead_owner = (
            f"{socket.gethostname().replace(':', '_')}:2147483647:1.0:dead"
        )
        stale_generation = database.begin_pending_request(
            "resume-me",
            tokens=44_000,
            owner=dead_owner,
            started_at=time.time(),
        )

        agent = _make_agent(
            database,
            session_id="resume-me",
            context_length=131_072,
        )
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        agent._session_messages = [
            {"role": "assistant", "content": "prior terminal"}
        ]

        agent._ensure_db_session()
        agent._save_session_log()

        assert agent.session_api_calls == 4
        assert agent.session_input_tokens == 1_000
        assert agent.session_output_tokens == 250
        assert agent.session_prompt_tokens == 1_520
        assert agent.context_compressor.last_real_prompt_tokens == 12_345
        assert agent.context_compressor.last_completion_tokens == 678
        assert agent.context_compressor.context_length == 131_072
        assert agent.context_compressor.compression_count == 3

        session = database.get_session("resume-me")
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] == stale_generation
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert session["context_length"] == 131_072

        snapshot = json.loads(
            (tmp_path / "session_resume-me.json").read_text(encoding="utf-8")
        )
        assert snapshot["api_call_count"] == 4
        assert snapshot["session_input_tokens"] == 1_000
        assert snapshot["session_output_tokens"] == 250
        assert snapshot["last_prompt_tokens"] == 12_345
        assert snapshot["last_completion_tokens"] == 678
        assert snapshot["compression_count"] == 3
        assert snapshot["context_length"] == 131_072
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_generation"] == stale_generation
    finally:
        database.close()


@pytest.mark.parametrize(
    ("persisted_context_length", "resolved_context_length"),
    [
        (65_536, 131_072),
        (131_072, 65_536),
    ],
)
def test_restart_keeps_current_resolved_context_cap_authoritative(
    tmp_path,
    persisted_context_length,
    resolved_context_length,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        database.create_session("resume-cap", "cli", model="old/model")
        database.set_context_length(
            "resume-cap", persisted_context_length
        )

        agent = _make_agent(
            database,
            session_id="resume-cap",
            context_length=resolved_context_length,
        )
        assert (
            agent.context_compressor.context_length
            == resolved_context_length
        )

        agent._ensure_db_session()

        assert (
            agent.context_compressor.context_length
            == resolved_context_length
        )
        assert (
            database.get_session("resume-cap")["context_length"]
            == resolved_context_length
        )
    finally:
        database.close()


def test_restart_clears_dead_zero_estimate_pending_owner(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        database.create_session("zero-pending", "cli")
        generation = database.begin_pending_request(
            "zero-pending",
            tokens=0,
            owner="dead-host:999999:1.0:dead",
            started_at=1.0,
        )
        agent = _make_agent(database, session_id="zero-pending")

        agent._ensure_db_session()

        session = database.get_session("zero-pending")
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] == generation
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert agent.pending_prompt_tokens == 0
        assert agent._pending_generation == generation
        assert agent._pending_owner is None
        assert agent._pending_started_at == 0
    finally:
        database.close()


def test_pending_marker_clears_when_execution_unwinds_unexpectedly(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="terminal-error")
        agent.client = SimpleNamespace()

        with (
            patch(
                "hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent.run_conversation("hello")

        session = database.get_session(agent.session_id)
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] >= 1
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert agent.pending_prompt_tokens == 0
        assert agent._pending_owner is None
        assert agent._pending_started_at == 0
    finally:
        database.close()


def test_baseexception_during_pending_projection_rolls_back_published_marker(
    tmp_path,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="pending-begin-baseexception")
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        agent.client = SimpleNamespace()

        # Turn-start persistence is the first call. The second is the pending
        # projection after SQLite publication but before provider execution.
        with (
            patch.object(
                agent,
                "_save_session_log",
                side_effect=[None, KeyboardInterrupt],
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent.run_conversation("hello")

        session = database.get_session(agent.session_id)
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] >= 1
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert agent.pending_prompt_tokens == 0
        assert agent._pending_owner is None
    finally:
        database.close()


def _json_telemetry_agent(database, tmp_path, session_id):
    agent = _make_agent(database, session_id=session_id)
    agent._ensure_db_session()
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path
    agent._session_messages = [{"role": "user", "content": "hello"}]
    return agent


def test_pending_begin_recovers_interrupt_after_sqlite_commit(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _json_telemetry_agent(database, tmp_path, "post-commit")
        real_begin = database.begin_pending_request
        interrupt = KeyboardInterrupt("after SQLite commit")

        def commit_then_interrupt(*args, **kwargs):
            real_begin(*args, **kwargs)
            raise interrupt

        with (
            patch.object(
                database,
                "begin_pending_request",
                side_effect=commit_then_interrupt,
            ),
            pytest.raises(KeyboardInterrupt) as caught,
        ):
            begin_pending_request(agent, 9_000)

        assert caught.value is interrupt
        row = database.get_session(agent.session_id)
        assert row["pending_prompt_tokens"] == 0
        assert row["pending_owner"] is None
        assert row["pending_started_at"] is None
        assert agent.pending_prompt_tokens == 0
        assert agent._pending_owner is None
        snapshot = json.loads(
            (tmp_path / "session_post-commit.json").read_text(encoding="utf-8")
        )
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_owner"] is None
    finally:
        database.close()


def test_pending_begin_interrupt_before_commit_preserves_prior_same_owner(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _json_telemetry_agent(database, tmp_path, "pre-commit")
        prior_generation = begin_pending_request(agent, 4_000, started_at=1.0)
        interrupt = KeyboardInterrupt("before SQLite commit")

        with (
            patch.object(
                database,
                "begin_pending_request",
                side_effect=interrupt,
            ),
            pytest.raises(KeyboardInterrupt) as caught,
        ):
            begin_pending_request(agent, 9_000, started_at=2.0)

        assert caught.value is interrupt
        row = database.get_session(agent.session_id)
        snapshot = json.loads(
            (tmp_path / "session_pre-commit.json").read_text(encoding="utf-8")
        )
        assert row["pending_generation"] == prior_generation
        assert snapshot["pending_generation"] == prior_generation
        assert agent._pending_generation == prior_generation
        assert row["pending_prompt_tokens"] == 4_000
        assert snapshot["pending_prompt_tokens"] == 4_000
        assert agent.pending_prompt_tokens == 4_000
        assert row["pending_owner"] == agent._session_telemetry_owner
        assert snapshot["pending_owner"] == agent._session_telemetry_owner
        assert agent._pending_owner == agent._session_telemetry_owner
    finally:
        database.close()


def test_pending_begin_reprojects_interrupt_after_atomic_json_replace(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _json_telemetry_agent(database, tmp_path, "post-json")
        real_atomic_json_write = run_agent_module.atomic_json_write
        interrupt = KeyboardInterrupt("after atomic JSON replace")
        interrupted = False

        def write_then_interrupt(path, payload, **kwargs):
            nonlocal interrupted
            real_atomic_json_write(path, payload, **kwargs)
            if not interrupted and payload["pending_prompt_tokens"] == 9_000:
                interrupted = True
                raise interrupt

        with (
            patch.object(
                run_agent_module,
                "atomic_json_write",
                side_effect=write_then_interrupt,
            ),
            pytest.raises(KeyboardInterrupt) as caught,
        ):
            begin_pending_request(agent, 9_000)

        assert caught.value is interrupt
        row = database.get_session(agent.session_id)
        snapshot = json.loads(
            (tmp_path / "session_post-json.json").read_text(encoding="utf-8")
        )
        assert row["pending_prompt_tokens"] == 0
        assert snapshot["pending_prompt_tokens"] == 0
        assert agent.pending_prompt_tokens == 0
        assert row["pending_owner"] is None
        assert snapshot["pending_owner"] is None
        assert agent._pending_owner is None
    finally:
        database.close()


def test_older_runtime_finally_cannot_clear_newer_projection(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="overlapping-pending")
        agent._ensure_db_session()
        first = begin_pending_request(agent, 4_000, started_at=1.0)
        second = begin_pending_request(agent, 9_000, started_at=2.0)

        assert second == first + 1
        assert clear_pending_request(agent, first) is False
        assert agent.pending_prompt_tokens == 9_000
        pending = database.get_session(agent.session_id)
        assert pending["pending_generation"] == second
        assert pending["pending_prompt_tokens"] == 9_000

        assert clear_pending_request(agent, second) is True
        assert agent.pending_prompt_tokens == 0
        assert database.get_session(agent.session_id)["pending_prompt_tokens"] == 0
    finally:
        database.close()


def test_clear_pending_persistent_db_failure_preserves_truth_and_reports_false(
    tmp_path,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="clear-persistent-failure")
        agent._ensure_db_session()
        generation = begin_pending_request(agent, 123, started_at=1.0)

        with patch.object(
            database,
            "clear_pending_request",
            side_effect=sqlite3.OperationalError("injected clear failure"),
        ):
            assert clear_pending_request(agent, generation) is False

        canonical = database.get_session(agent.session_id)
        assert canonical["pending_prompt_tokens"] == 123
        assert canonical["pending_owner"] == agent._session_telemetry_owner
        assert agent.pending_prompt_tokens == 123
        assert agent._pending_owner == agent._session_telemetry_owner
    finally:
        database.close()


def test_clear_pending_retries_one_transient_db_failure_to_terminal_zero(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="clear-transient-failure")
        agent._ensure_db_session()
        generation = begin_pending_request(agent, 456, started_at=1.0)
        real_clear = database.clear_pending_request
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_clear(*args, **kwargs)

        with patch.object(
            database,
            "clear_pending_request",
            side_effect=fail_once,
        ):
            assert clear_pending_request(agent, generation) is True

        canonical = database.get_session(agent.session_id)
        assert attempts == 2
        assert canonical["pending_prompt_tokens"] == 0
        assert canonical["pending_owner"] is None
        assert agent.pending_prompt_tokens == 0
        assert agent._pending_owner is None
    finally:
        database.close()


def test_provider_baseexception_survives_post_commit_cleanup_baseexception(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="provider-original-exception")
        agent.client = SimpleNamespace()
        original_interrupt = KeyboardInterrupt("original provider interrupt")
        cleanup_exit = SystemExit("post-commit cleanup exit")
        real_clear = database.clear_pending_request

        def commit_then_exit(*args, **kwargs):
            assert real_clear(*args, **kwargs)
            raise cleanup_exit

        with (
            patch(
                "hermes_cli.middleware.run_llm_execution_middleware",
                side_effect=original_interrupt,
            ),
            patch.object(
                database,
                "clear_pending_request",
                side_effect=commit_then_exit,
            ),
            pytest.raises(KeyboardInterrupt) as caught,
        ):
            agent.run_conversation("hello")

        assert caught.value is original_interrupt
        canonical = database.get_session(agent.session_id)
        assert canonical["pending_prompt_tokens"] == 0
        assert canonical["pending_owner"] is None
        assert agent.pending_prompt_tokens == 0
    finally:
        database.close()


def test_older_process_finally_adopts_newer_canonical_projection(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="cross-process-pending")
        agent._ensure_db_session()
        first = begin_pending_request(agent, 4_000, started_at=1.0)
        newer = database.begin_pending_request(
            agent.session_id,
            tokens=9_000,
            owner="other-process-owner",
            started_at=2.0,
        )

        assert newer == first + 1
        assert clear_pending_request(agent, first) is False
        assert agent.pending_prompt_tokens == 9_000
        assert agent._pending_generation == newer
        assert agent._pending_owner == "other-process-owner"
        assert agent._pending_started_at == 2.0
        pending = database.get_session(agent.session_id)
        assert pending["pending_generation"] == newer
        assert pending["pending_prompt_tokens"] == 9_000
        assert pending["pending_owner"] == "other-process-owner"
    finally:
        database.close()


def test_stale_process_json_write_cannot_resurrect_after_newer_clear(tmp_path):
    db_path = tmp_path / "state.db"
    stale_db = SessionDB(db_path)
    current_db = SessionDB(db_path)
    try:
        stale_agent = _make_agent(stale_db, session_id="serialized-pending")
        current_agent = _make_agent(current_db, session_id="serialized-pending")
        for agent in (stale_agent, current_agent):
            agent._session_json_enabled = True
            agent.logs_dir = tmp_path
            agent._session_messages = [{"role": "user", "content": "hello"}]
            agent._ensure_db_session()

        stale_generation = begin_pending_request(stale_agent, 4_000)
        current_generation = begin_pending_request(current_agent, 9_000)
        assert current_generation == stale_generation + 1

        stale_projection_ready = threading.Event()
        allow_stale_projection = threading.Event()
        current_clear_started = threading.Event()
        current_clear_finished = threading.Event()
        errors = []
        results = {}
        original_stale_save = stale_agent._save_session_log

        def paused_stale_save(*args, **kwargs):
            stale_projection_ready.set()
            if not allow_stale_projection.wait(5):
                raise AssertionError("timed out waiting to release stale projection")
            return original_stale_save(*args, **kwargs)

        stale_agent._save_session_log = paused_stale_save

        def clear_stale():
            try:
                results["stale"] = clear_pending_request(
                    stale_agent, stale_generation
                )
            except BaseException as exc:  # pragma: no cover - assertion aid
                errors.append(exc)

        def clear_current():
            current_clear_started.set()
            try:
                results["current"] = clear_pending_request(
                    current_agent, current_generation
                )
            except BaseException as exc:  # pragma: no cover - assertion aid
                errors.append(exc)
            finally:
                current_clear_finished.set()

        stale_thread = threading.Thread(target=clear_stale)
        stale_thread.start()
        assert stale_projection_ready.wait(5)

        current_thread = threading.Thread(target=clear_current)
        current_thread.start()
        assert current_clear_started.wait(5)
        # The newer canonical clear must wait for the stale DB-read + JSON
        # projection unit. Without serialization this is the forced
        # canonical=0 / JSON=9000 resurrection interleaving.
        assert not current_clear_finished.wait(0.2)
        assert current_db.get_session(stale_agent.session_id)[
            "pending_prompt_tokens"
        ] == 9_000

        allow_stale_projection.set()
        stale_thread.join(5)
        current_thread.join(5)
        assert not stale_thread.is_alive()
        assert not current_thread.is_alive()
        assert errors == []
        assert results == {"stale": False, "current": True}

        canonical = current_db.get_session(stale_agent.session_id)
        assert canonical["pending_prompt_tokens"] == 0
        assert canonical["pending_generation"] == current_generation
        snapshot = json.loads(
            (tmp_path / "session_serialized-pending.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_generation"] == current_generation
        assert snapshot["pending_owner"] is None
    finally:
        stale_db.close()
        current_db.close()


def test_stale_process_json_write_projects_all_canonical_telemetry(tmp_path):
    db_path = tmp_path / "state.db"
    current_db = SessionDB(db_path)
    stale_db = SessionDB(db_path)
    try:
        current = _make_agent(current_db, session_id="shared-telemetry")
        stale = _make_agent(stale_db, session_id="shared-telemetry")
        for agent in (current, stale):
            agent._ensure_db_session()
            agent._session_json_enabled = True
            agent.logs_dir = tmp_path
            agent._session_messages = [
                {"role": "user", "content": "hello"}
            ]

        current_db.update_token_counts(
            "shared-telemetry",
            input_tokens=200,
            output_tokens=50,
            api_call_count=2,
            absolute=True,
            last_prompt_tokens=200,
            last_completion_tokens=50,
        )
        current.session_api_calls = 2
        current.session_input_tokens = 200
        current.session_output_tokens = 50
        current.context_compressor.last_prompt_tokens = 200
        current.context_compressor.last_real_prompt_tokens = 200
        current.context_compressor.last_completion_tokens = 50
        current._save_session_log()

        stale._save_session_log()

        snapshot = json.loads(
            (tmp_path / "session_shared-telemetry.json").read_text(
                encoding="utf-8"
            )
        )
        canonical = current_db.get_session("shared-telemetry")
        assert snapshot["api_call_count"] == canonical["api_call_count"] == 2
        assert (
            snapshot["session_input_tokens"]
            == canonical["input_tokens"]
            == 200
        )
        assert (
            snapshot["session_output_tokens"]
            == canonical["output_tokens"]
            == 50
        )
        assert (
            snapshot["last_prompt_tokens"]
            == canonical["last_prompt_tokens"]
            == 200
        )
        assert (
            snapshot["last_completion_tokens"]
            == canonical["last_completion_tokens"]
            == 50
        )
    finally:
        stale_db.close()
        current_db.close()


def test_forced_canonical_zero_json_nine_thousand_is_repaired(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        stale_agent = _make_agent(database, session_id="forced-interleaving")
        current_agent = _make_agent(database, session_id="forced-interleaving")
        for agent in (stale_agent, current_agent):
            agent._session_json_enabled = True
            agent.logs_dir = tmp_path
            agent._session_messages = [{"role": "user", "content": "hello"}]
            agent._ensure_db_session()

        stale_generation = begin_pending_request(stale_agent, 4_000)
        current_generation = begin_pending_request(current_agent, 9_000)
        real_atomic_json_write = run_agent_module.atomic_json_write
        forced_pairs = []

        def force_clear_between_fence_and_write(path, payload, **kwargs):
            if not forced_pairs and payload["pending_prompt_tokens"] == 9_000:
                assert database.clear_pending_request(
                    stale_agent.session_id,
                    generation=current_generation,
                    owner=current_agent._session_telemetry_owner,
                )
                forced_pairs.append(
                    (
                        database.get_session(stale_agent.session_id)[
                            "pending_prompt_tokens"
                        ],
                        payload["pending_prompt_tokens"],
                    )
                )
            return real_atomic_json_write(path, payload, **kwargs)

        with patch.object(
            run_agent_module,
            "atomic_json_write",
            side_effect=force_clear_between_fence_and_write,
        ):
            assert clear_pending_request(stale_agent, stale_generation) is False

        assert forced_pairs == [(0, 9_000)]
        canonical = database.get_session(stale_agent.session_id)
        assert canonical["pending_prompt_tokens"] == 0
        snapshot = json.loads(
            (tmp_path / "session_forced-interleaving.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_generation"] == current_generation
        assert snapshot["pending_owner"] is None
    finally:
        database.close()


def test_valid_stream_without_usage_still_counts_as_success(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(
            database,
            session_id="no-usage",
            context_length=65_536,
        )
        assert agent.context_compressor.context_length == 65_536
        agent.client = SimpleNamespace()
        request_client = MagicMock()
        request_client.chat.completions.create.return_value = iter(
            [
                _stream_chunk(content="done"),
                _stream_chunk(finish_reason="stop"),
            ]
        )
        agent._create_request_openai_client = MagicMock(
            return_value=request_client
        )
        agent._close_request_openai_client = MagicMock()

        result = agent.run_conversation("hello")

        session = database.get_session(agent.session_id)
        assert result["final_response"] == "done"
        assert result["api_calls"] == 1
        assert result["api_call_count"] == 1
        assert session["api_call_count"] == 1
        assert session["context_length"] == 65_536
        assert session["last_prompt_tokens"] == 0
        assert session["last_completion_tokens"] == 0
        assert session["pending_prompt_tokens"] == 0
    finally:
        database.close()


def test_late_db_bind_hydrates_before_recording_call_without_usage(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        database.update_token_counts("late-no-usage", api_call_count=4)
        agent = _make_agent(database, session_id="late-no-usage")
        real_create = database.create_session
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_create(*args, **kwargs)

        with patch.object(database, "create_session", side_effect=fail_once):
            agent._ensure_db_session()
            assert agent._session_db_created is False
            record_call_without_usage(agent)

        row = database.get_session(agent.session_id)
        assert agent.session_api_calls == 5
        assert row["api_call_count"] == 5
    finally:
        database.close()


def test_late_db_bind_hydrates_before_recording_canonical_usage(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        database.update_token_counts(
            "late-usage",
            input_tokens=80,
            output_tokens=25,
            api_call_count=4,
            estimated_cost_usd=0.5,
            last_prompt_tokens=80,
            last_completion_tokens=25,
        )
        agent = _make_agent(database, session_id="late-usage")
        real_create = database.create_session
        attempts = 0

        def fail_once(*args, **kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_create(*args, **kwargs)

        with (
            patch.object(database, "create_session", side_effect=fail_once),
            patch(
                "agent.usage_pricing.estimate_usage_cost",
                return_value=SimpleNamespace(
                    amount_usd=0.25,
                    status="estimated",
                    source="official_docs_snapshot",
                ),
            ),
        ):
            agent._ensure_db_session()
            assert agent._session_db_created is False
            usage = record_canonical_usage(
                agent,
                CanonicalUsage(input_tokens=4, output_tokens=1),
            )

        row = database.get_session(agent.session_id)
        assert usage["total_tokens"] == 5
        assert usage["estimated_cost_usd"] == 0.25
        assert agent.session_api_calls == 5
        assert agent.session_input_tokens == 84
        assert agent.session_output_tokens == 26
        assert agent.session_total_tokens == 110
        assert agent.session_estimated_cost_usd == pytest.approx(0.75)
        assert agent.context_compressor.last_prompt_tokens == 4
        assert agent.context_compressor.last_completion_tokens == 1
        assert row["api_call_count"] == 5
        assert row["input_tokens"] == 84
        assert row["output_tokens"] == 26
        assert row["total_tokens"] == 110
        assert row["estimated_cost_usd"] == pytest.approx(0.75)
    finally:
        database.close()


@pytest.mark.parametrize(
    "invalid_extra_cost",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
def test_invalid_extra_cost_does_not_erase_valid_base_cost(
    tmp_path,
    invalid_extra_cost,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="valid-base-cost")
        with patch(
            "agent.usage_pricing.estimate_usage_cost",
            return_value=SimpleNamespace(
                amount_usd=0.25,
                status="estimated",
                source="official_docs_snapshot",
            ),
        ):
            usage = record_canonical_usage(
                agent,
                CanonicalUsage(input_tokens=4, output_tokens=1),
                extra_cost_usd=invalid_extra_cost,
            )

        row = database.get_session(agent.session_id)
        assert usage["estimated_cost_usd"] == 0.25
        assert agent.session_estimated_cost_usd == 0.25
        assert row["estimated_cost_usd"] == 0.25
    finally:
        database.close()


def test_cumulative_finite_cost_saturates_in_memory_and_sqlite(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="finite-cost-overflow")
        maximum = sys.float_info.max
        agent.session_estimated_cost_usd = maximum
        database.update_token_counts(
            agent.session_id,
            estimated_cost_usd=maximum,
            absolute=True,
        )

        with patch(
            "agent.usage_pricing.estimate_usage_cost",
            return_value=SimpleNamespace(
                amount_usd=maximum,
                status="estimated",
                source="official_docs_snapshot",
            ),
        ):
            usage = record_canonical_usage(
                agent,
                CanonicalUsage(input_tokens=1, output_tokens=1),
            )

        row = database.get_session(agent.session_id)
        assert usage["estimated_cost_usd"] == maximum
        assert agent.session_estimated_cost_usd == maximum
        assert row["estimated_cost_usd"] == maximum
    finally:
        database.close()


def test_http_200_refusal_with_usage_is_accounted_before_terminal_return(
    tmp_path,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="refusal-with-usage")
        agent.client = MagicMock()
        agent.client.chat.completions.create.return_value = _chat_response(
            content=None,
            finish_reason="stop",
            refusal="I cannot help with that.",
            usage=SimpleNamespace(
                prompt_tokens=13,
                completion_tokens=2,
                total_tokens=15,
            ),
        )

        result = agent.run_conversation("disallowed request")

        assert result["completed"] is False
        assert result["failed"] is True
        assert "content_policy_blocked" in result["error"]
        _assert_stable_numeric_telemetry(result)
        session = database.get_session(agent.session_id)
        assert session["api_call_count"] == 1
        assert session["input_tokens"] == 13
        assert session["output_tokens"] == 2
        assert session["last_prompt_tokens"] == 13
        assert session["last_completion_tokens"] == 2
    finally:
        database.close()


def test_real_length_response_usage_counts_before_continuation(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="length-with-usage")
        agent.client = MagicMock()
        agent.client.chat.completions.create.side_effect = [
            _chat_response(
                content="partial ",
                finish_reason="length",
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=2,
                    total_tokens=12,
                ),
                response_id="real-length",
            ),
            _chat_response(
                content="terminal",
                finish_reason="stop",
                usage=SimpleNamespace(
                    prompt_tokens=12,
                    completion_tokens=3,
                    total_tokens=15,
                ),
                response_id="real-stop",
            ),
        ]

        result = agent.run_conversation("please answer")

        assert result["completed"] is True
        _assert_stable_numeric_telemetry(result)
        session = database.get_session(agent.session_id)
        assert session["api_call_count"] == 2
        assert session["input_tokens"] == 22
        assert session["output_tokens"] == 5
        assert session["last_prompt_tokens"] == 12
        assert session["last_completion_tokens"] == 3
    finally:
        database.close()


def test_partial_stream_stub_is_not_accepted_or_accounted(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(database, session_id="partial-stub")
        agent._ensure_db_session()
        response = _chat_response(
            content="partial",
            finish_reason="length",
            usage=SimpleNamespace(
                prompt_tokens=99,
                completion_tokens=8,
                total_tokens=107,
            ),
            response_id=PARTIAL_STREAM_STUB_ID,
        )

        _record_accepted_provider_response(
            agent,
            response,
            [{"role": "user", "content": "hello"}],
            api_duration=1.0,
        )

        session = database.get_session(agent.session_id)
        assert agent.session_api_calls == 0
        assert agent.session_total_tokens == 0
        assert session["api_call_count"] == 0
        assert session["input_tokens"] == 0
        assert session["output_tokens"] == 0
    finally:
        database.close()


def test_completed_compression_is_persisted_at_the_boundary():
    session_db = MagicMock()
    agent = _make_agent(session_db, session_id="before-compression")
    agent.compression_in_place = False
    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 3
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor

    agent._compress_context(
        [{"role": "user", "content": "old"}],
        "system",
        approx_tokens=10_000,
    )

    session_db.update_token_counts.assert_called_once()
    call_args = session_db.update_token_counts.call_args
    assert call_args.args == (agent.session_id,)
    assert call_args.kwargs["absolute"] is True
    assert call_args.kwargs["compression_count"] == 3


def test_compression_epoch_shrinks_json_then_exposes_new_terminal_turn(tmp_path):
    agent = _make_agent(
        None,
        session_id="compression-json",
        context_length=65_536,
    )
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path
    original_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer two"},
    ]
    agent._save_session_log(original_messages)

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 2
    compressor.last_prompt_tokens = 0
    compressor.last_real_prompt_tokens = 0
    compressor.last_completion_tokens = 300
    compressor.context_length = 65_536
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor

    agent._compress_context(
        original_messages,
        "system",
        approx_tokens=10_000,
    )

    snapshot_path = tmp_path / "session_compression-json.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    compressed_messages = compressor.compress.return_value
    assert snapshot["message_count"] == len(compressed_messages)
    assert snapshot["messages"] == compressed_messages
    assert snapshot["compression_count"] == 2
    assert snapshot["compression_epoch"] == 2
    assert snapshot["last_completion_tokens"] == 300
    assert snapshot["context_length"] == 65_536

    terminal_messages = compressed_messages + [
        {"role": "assistant", "content": "post-compression terminal"}
    ]
    agent._save_session_log(terminal_messages)
    terminal_snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert terminal_snapshot["message_count"] == len(terminal_messages)
    assert terminal_snapshot["messages"][-1] == {
        "role": "assistant",
        "content": "post-compression terminal",
    }


def test_stale_process_cannot_resurrect_pre_compaction_transcript(tmp_path):
    db_path = tmp_path / "state.db"
    current_db = SessionDB(db_path)
    stale_db = SessionDB(db_path)
    try:
        current = _make_agent(
            current_db,
            session_id="shared-compaction",
        )
        stale = _make_agent(
            stale_db,
            session_id="shared-compaction",
        )
        original = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "b"},
        ]
        compressed = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "current"},
        ]
        for agent in (current, stale):
            agent._ensure_db_session()
            agent._session_json_enabled = True
            agent.logs_dir = tmp_path
            agent._session_messages = list(original)

        current.context_compressor.compression_count = 2
        current_db.set_compression_count("shared-compaction", 2)
        current._save_session_log(
            compressed,
            allow_compaction_shrink=True,
        )

        stale._save_session_log(original)

        snapshot = json.loads(
            (tmp_path / "session_shared-compaction.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshot["messages"] == compressed
        assert snapshot["message_count"] == len(compressed)
        assert snapshot["compression_epoch"] == 2
        assert current_db.get_session("shared-compaction")[
            "compression_count"
        ] == 2
    finally:
        stale_db.close()
        current_db.close()


def test_legacy_compression_rotation_keeps_db_json_telemetry_equal(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        old_session_id = "before-rotation"
        agent = _make_agent(
            database,
            session_id=old_session_id,
            context_length=65_536,
        )
        agent._ensure_db_session()
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        agent.compression_in_place = False
        agent._compression_feasibility_checked = True
        messages = [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "b"},
            {"role": "assistant", "content": "b"},
        ]
        compressed = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "assistant", "content": "recent"},
        ]
        agent._session_messages = messages
        agent.session_api_calls = 1
        agent.session_input_tokens = 100
        agent.session_output_tokens = 20
        agent.session_total_tokens = 120
        agent.context_compressor.last_prompt_tokens = 100
        agent.context_compressor.last_real_prompt_tokens = 100
        agent.context_compressor.last_completion_tokens = 20
        agent.context_compressor.compression_count = 1
        database.update_token_counts(
            old_session_id,
            input_tokens=100,
            output_tokens=20,
            total_tokens=120,
            api_call_count=1,
            absolute=True,
            last_prompt_tokens=100,
            last_completion_tokens=20,
            context_length=65_536,
            compression_count=0,
        )

        with (
            patch.object(
                agent.context_compressor,
                "compress",
                return_value=list(compressed),
            ),
            patch.object(
                agent,
                "_build_system_prompt",
                return_value="new system",
            ),
        ):
            agent._compress_context(
                messages,
                "old system",
                approx_tokens=100_000,
            )

        new_session_id = agent.session_id
        assert new_session_id != old_session_id
        row = database.get_session(new_session_id)
        snapshot = json.loads(
            (tmp_path / f"session_{new_session_id}.json").read_text(
                encoding="utf-8"
            )
        )
        assert row["api_call_count"] == snapshot["api_call_count"] == 1
        assert (
            row["input_tokens"]
            == snapshot["session_input_tokens"]
            == 100
        )
        assert (
            row["output_tokens"]
            == snapshot["session_output_tokens"]
            == 20
        )
        assert row["total_tokens"] == 120
        # The compression boundary intentionally invalidates the previous
        # real-prompt snapshot until the child receives fresh provider usage.
        assert row["last_prompt_tokens"] == snapshot["last_prompt_tokens"] == 0
        assert (
            row["last_completion_tokens"]
            == snapshot["last_completion_tokens"]
            == 20
        )
        assert (
            row["compression_count"]
            == snapshot["compression_count"]
            == 1
        )
        assert row["context_length"] == snapshot["context_length"] == 65_536
        assert agent._session_telemetry_hydrated_session_id == new_session_id
    finally:
        database.close()


def test_transient_compression_json_failure_preserves_later_shrink_permission(
    tmp_path,
):
    agent = _make_agent(
        None,
        session_id="compression-json-retry",
        context_length=65_536,
    )
    agent._session_json_enabled = True
    agent.logs_dir = tmp_path
    original_messages = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer two"},
    ]
    agent._save_session_log(original_messages)

    compressed_messages = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor = MagicMock()
    compressor.compress.return_value = compressed_messages
    compressor.compression_count = 2
    compressor.last_prompt_tokens = 0
    compressor.last_real_prompt_tokens = 0
    compressor.last_completion_tokens = 300
    compressor.context_length = 65_536
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    agent.context_compressor = compressor

    with patch(
        "run_agent.atomic_json_write",
        side_effect=OSError("transient boundary write failure"),
    ):
        agent._compress_context(
            original_messages,
            "system",
            approx_tokens=10_000,
        )

    snapshot_path = tmp_path / "session_compression-json-retry.json"
    unchanged = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert unchanged["messages"] == original_messages
    assert agent._session_json_pending_compaction_epoch == 2

    terminal_messages = compressed_messages + [
        {"role": "assistant", "content": "terminal after retry"}
    ]
    agent._save_session_log(terminal_messages)

    recovered = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert recovered["message_count"] == len(terminal_messages)
    assert recovered["messages"] == terminal_messages
    assert recovered["compression_epoch"] == 2
    assert agent._session_json_pending_compaction_epoch == 0


def test_codex_complete_turn_projects_pending_usage_compaction_and_terminal_json(
    tmp_path,
):
    database = SessionDB(tmp_path / "state.db")
    observed = {}

    def fake_run_turn(_session, user_input, **_kwargs):
        observed["db_pending"] = database.get_session(
            "codex-complete"
        )["pending_prompt_tokens"]
        observed["json_pending"] = json.loads(
            (tmp_path / "session_codex-complete.json").read_text(
                encoding="utf-8"
            )
        )["pending_prompt_tokens"]
        return TurnResult(
            final_text="codex done",
            projected_messages=[
                {"role": "assistant", "content": "codex done"}
            ],
            turn_id="turn-complete",
            thread_id="thread-complete",
            token_usage_last={
                "totalTokens": 130,
                "inputTokens": 80,
                "cachedInputTokens": 20,
                "outputTokens": 25,
                "reasoningOutputTokens": 5,
            },
            model_context_window=200_000,
            compacted=True,
        )

    try:
        agent = _make_agent(
            database,
            session_id="codex-complete",
            api_mode="codex_app_server",
        )
        agent._session_json_enabled = True
        agent.logs_dir = tmp_path
        with (
            patch.object(CodexAppServerSession, "run_turn", fake_run_turn),
            patch.object(
                CodexAppServerSession,
                "ensure_started",
                return_value="thread-complete",
            ),
            patch.object(agent, "_spawn_background_review", return_value=None),
        ):
            result = agent.run_conversation("hello codex")

        assert observed["db_pending"] > 0
        assert observed["json_pending"] > 0
        assert result["completed"] is True
        assert result["api_call_count"] == 1
        assert result["total_tokens"] == 130
        assert result["compression_count"] == 1
        assert result["last_real_prompt_tokens"] == 100
        assert result["last_completion_tokens"] == 25
        assert result["context_length"] == 200_000
        _assert_stable_numeric_telemetry(result)

        session = database.get_session("codex-complete")
        assert session["api_call_count"] == 1
        assert session["last_prompt_tokens"] == 100
        assert session["last_completion_tokens"] == 25
        assert session["total_tokens"] == 130
        assert session["compression_count"] == 1
        assert session["context_length"] == 200_000
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None

        resumed_agent = _make_agent(
            database,
            session_id="codex-complete",
            api_mode="codex_app_server",
        )
        resumed_agent._ensure_db_session()
        assert resumed_agent.session_total_tokens == 130

        snapshot = json.loads(
            (tmp_path / "session_codex-complete.json").read_text(
                encoding="utf-8"
            )
        )
        assert snapshot["messages"][-1] == {
            "role": "assistant",
            "content": "codex done",
        }
        assert snapshot["api_call_count"] == 1
        assert snapshot["compression_count"] == 1
        assert snapshot["pending_prompt_tokens"] == 0
        assert snapshot["pending_owner"] is None
        assert snapshot["pending_started_at"] == 0
    finally:
        database.close()


@pytest.mark.parametrize(
    ("interrupted", "error"),
    [
        (False, "provider terminal error"),
        (True, None),
    ],
)
def test_codex_error_or_interrupt_persists_usage_without_success_count(
    tmp_path,
    interrupted,
    error,
):
    database = SessionDB(tmp_path / "state.db")

    def fake_run_turn(_session, user_input, **_kwargs):
        return TurnResult(
            final_text="partial codex text",
            projected_messages=[
                {"role": "assistant", "content": "partial codex text"}
            ],
            interrupted=interrupted,
            error=error,
            turn_id="turn-terminal",
            thread_id="thread-terminal",
            token_usage_last={
                "totalTokens": 12,
                "inputTokens": 8,
                "cachedInputTokens": 0,
                "outputTokens": 4,
                "reasoningOutputTokens": 0,
            },
        )

    try:
        agent = _make_agent(
            database,
            session_id=f"codex-terminal-{interrupted}",
            api_mode="codex_app_server",
        )
        with (
            patch.object(CodexAppServerSession, "run_turn", fake_run_turn),
            patch.object(
                CodexAppServerSession,
                "ensure_started",
                return_value="thread-terminal",
            ),
            patch.object(agent, "_spawn_background_review", return_value=None),
        ):
            result = agent.run_conversation("hello")

        assert result["completed"] is False
        assert result["partial"] is True
        assert result["api_call_count"] == 0
        assert result["total_tokens"] == 12
        _assert_stable_numeric_telemetry(result)
        session = database.get_session(agent.session_id)
        assert session["api_call_count"] == 0
        assert session["input_tokens"] == 8
        assert session["output_tokens"] == 4
        assert session["pending_prompt_tokens"] == 0
    finally:
        database.close()


def test_codex_baseexception_still_clears_generation_fenced_pending(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(
            database,
            session_id="codex-baseexception",
            api_mode="codex_app_server",
        )
        with (
            patch.object(
                CodexAppServerSession,
                "run_turn",
                side_effect=KeyboardInterrupt,
            ),
            patch.object(
                CodexAppServerSession,
                "ensure_started",
                return_value="thread-baseexception",
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            agent.run_conversation("hello")

        session = database.get_session(agent.session_id)
        assert session["pending_prompt_tokens"] == 0
        assert session["pending_generation"] >= 1
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
        assert agent.pending_prompt_tokens == 0
    finally:
        database.close()


def test_codex_baseexception_survives_post_commit_cleanup_baseexception(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _make_agent(
            database,
            session_id="codex-original-exception",
            api_mode="codex_app_server",
        )
        original_interrupt = KeyboardInterrupt("original Codex interrupt")
        cleanup_exit = SystemExit("post-commit Codex cleanup exit")
        real_clear = database.clear_pending_request

        def commit_then_exit(*args, **kwargs):
            assert real_clear(*args, **kwargs)
            raise cleanup_exit

        with (
            patch.object(
                CodexAppServerSession,
                "run_turn",
                side_effect=original_interrupt,
            ),
            patch.object(
                CodexAppServerSession,
                "ensure_started",
                return_value="thread-original-exception",
            ),
            patch.object(
                database,
                "clear_pending_request",
                side_effect=commit_then_exit,
            ),
            pytest.raises(KeyboardInterrupt) as caught,
        ):
            agent.run_conversation("hello")

        assert caught.value is original_interrupt
        canonical = database.get_session(agent.session_id)
        assert canonical["pending_prompt_tokens"] == 0
        assert canonical["pending_owner"] is None
        assert agent.pending_prompt_tokens == 0
    finally:
        database.close()


@pytest.mark.parametrize(
    "early_result",
    [
        {
            "final_response": "refused",
            "messages": [],
            "completed": False,
            "failed": True,
        },
        {
            "final_response": "truncated",
            "messages": [],
            "completed": False,
            "partial": True,
        },
        {
            "final_response": "interrupted",
            "messages": [],
            "completed": False,
            "interrupted": True,
        },
        {
            "final_response": "terminal error",
            "messages": [],
            "completed": False,
            "error": "boom",
        },
    ],
)
def test_every_forwarded_terminal_shape_gets_stable_numeric_fields(early_result):
    agent = _make_agent(None, session_id="result-contract")
    with patch(
        "agent.conversation_loop.run_conversation",
        return_value=dict(early_result),
    ):
        result = agent.run_conversation("hello")

    _assert_stable_numeric_telemetry(result)


@pytest.mark.parametrize(
    "invalid_cost",
    [float("nan"), float("inf"), float("-inf"), -1.0],
)
def test_terminal_numeric_contract_normalizes_none_and_invalid_cost(
    invalid_cost,
):
    agent = _make_agent(None, session_id="invalid-numeric-contract")
    agent.session_estimated_cost_usd = invalid_cost
    with patch(
        "agent.conversation_loop.run_conversation",
        return_value={
            "final_response": "terminal",
            "messages": [],
            "completed": False,
            "api_calls": None,
        },
    ):
        result = agent.run_conversation("hello")

    assert result["api_calls"] == 0
    assert result["estimated_cost_usd"] == 0.0
    _assert_stable_numeric_telemetry(result)


@pytest.mark.parametrize(
    "invalid_cost",
    [-1.0, float("nan"), float("inf"), float("-inf")],
)
@pytest.mark.parametrize("absolute", [False, True])
def test_invalid_cost_is_canonical_on_memory_json_sqlite_and_result(
    tmp_path,
    invalid_cost,
    absolute,
):
    database = SessionDB(tmp_path / "state.db")
    try:
        agent = _json_telemetry_agent(database, tmp_path, "invalid-cost")
        agent.session_api_calls = None
        agent.session_estimated_cost_usd = invalid_cost

        result = apply_telemetry_result_fields({"api_calls": None}, agent)
        agent._save_session_log()
        database.update_token_counts(
            agent.session_id,
            estimated_cost_usd=invalid_cost,
            actual_cost_usd=invalid_cost,
            absolute=absolute,
        )

        snapshot = json.loads(
            (tmp_path / "session_invalid-cost.json").read_text(encoding="utf-8")
        )
        row = database.get_session(agent.session_id)
        assert result["api_calls"] == 0
        assert agent.session_api_calls == 0
        for value in (
            result["estimated_cost_usd"],
            agent.session_estimated_cost_usd,
            snapshot["estimated_cost_usd"],
            row["estimated_cost_usd"],
            row["actual_cost_usd"],
        ):
            assert isinstance(value, (int, float))
            assert not isinstance(value, bool)
            assert math.isfinite(value)
            assert value >= 0
    finally:
        database.close()
