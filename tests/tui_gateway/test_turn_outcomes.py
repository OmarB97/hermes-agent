import contextlib
import queue
import sys
import threading
import types

import pytest

from tui_gateway import server


REAL_DRAIN_QUEUED_PROMPT = server._drain_queued_prompt
REAL_THREAD = threading.Thread


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True
        try:
            self._target()
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive


class _OutcomeDb:
    def __init__(self):
        self.rows = []
        self._ids = set()

    def record_turn_outcome(self, session_id, turn_id, **row):
        if turn_id in self._ids:
            return False
        self._ids.add(turn_id)
        self.rows.append({"session_id": session_id, "turn_id": turn_id, **row})
        return True


class _Agent:
    def __init__(self, result=None, *, error=None, on_run=None):
        self.model = "deepseek-v4-flash"
        self.provider = "local-vllm"
        self._fallback_activated = False
        self._result = result
        self._error = error
        self._on_run = on_run

    def clear_interrupt(self):
        return None

    def run_conversation(self, *_args, **_kwargs):
        if self._on_run is not None:
            self._on_run()
        if self._error is not None:
            raise self._error
        return self._result


def _session(agent):
    return {
        "agent": agent,
        "attached_images": [],
        "cols": 80,
        "cwd": "/tmp",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "running": True,
        "session_key": "stored-session",
        "transport": None,
    }


@pytest.fixture()
def turn_harness(monkeypatch):
    emitted = []
    outcome_db = _OutcomeDb()

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(outcome_db),
    )
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)

    def run(agent, *, queued_prompt=None, text="run the turn"):
        session = _session(agent)
        if queued_prompt is not None:
            session["queued_prompt"] = queued_prompt
        server._start_inflight_turn(session, text)
        server._run_prompt_submit("request-1", "runtime-session", session, text)
        outcomes = [args[2] for args in emitted if args[0] == "turn.outcome"]
        return session, outcomes

    return types.SimpleNamespace(db=outcome_db, emitted=emitted, run=run)


@pytest.mark.parametrize(
    ("result", "expected_status", "reason_fragment"),
    [
        (
            {
                "completed": True,
                "final_response": "done",
                "messages": [
                    {"role": "user", "content": "run"},
                    {"role": "assistant", "content": "done"},
                ],
            },
            "completed",
            "response delivered",
        ),
        (
            {
                "completed": True,
                "final_response": "",
                "messages": [{"role": "user", "content": "run"}],
            },
            "failed",
            "without a visible response",
        ),
        (
            {
                "completed": False,
                "failed": True,
                "failure_reason": "timeout",
                "error": "ReadTimeout: request timed out after 494.5s",
                "final_response": "API call timed out",
                "messages": [{"role": "user", "content": "run"}],
            },
            "timed_out",
            "timed out",
        ),
        (
            {
                "completed": False,
                "failed": True,
                "error": "HTTP 502: all backends failed",
                "final_response": "API call failed after retries",
                "messages": [{"role": "user", "content": "run"}],
            },
            "failed",
            "HTTP 502",
        ),
        (
            {
                "completed": False,
                "interrupted": True,
                "interrupt_message": "user cancelled while waiting for model response",
                "final_response": "",
                "messages": [{"role": "user", "content": "run"}],
            },
            "cancelled",
            "user cancelled",
        ),
    ],
)
def test_forced_terminal_results_emit_and_persist_exactly_once(
    turn_harness, result, expected_status, reason_fragment
):
    _session_row, outcomes = turn_harness.run(_Agent(result))

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == expected_status
    assert reason_fragment.lower() in outcomes[0]["reason"].lower()
    assert len(turn_harness.db.rows) == 1
    assert turn_harness.db.rows[0]["turn_id"] == outcomes[0]["id"]
    assert [args[0] for args in turn_harness.emitted].count("turn.outcome") == 1


def test_http_502_and_exception_payloads_are_force_redacted(turn_harness):
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    result = {
        "completed": False,
        "failed": True,
        "error": f"HTTP 502 after retries; Authorization: Bearer {secret}",
        "final_response": "",
        "messages": [{"role": "user", "content": "run"}],
    }
    agent = _Agent(result)
    agent.provider = f"Authorization: Bearer {secret}"

    _session_row, outcomes = turn_harness.run(agent)

    serialized = str(outcomes[0]) + str(turn_harness.db.rows[0])
    assert "HTTP 502" in serialized
    assert secret not in serialized
    assert "***" in serialized


def test_fallback_success_and_exhaustion_have_one_terminal_outcome(turn_harness):
    success_agent = _Agent(
        {
            "completed": True,
            "final_response": "fallback answer",
            "messages": [{"role": "assistant", "content": "fallback answer"}],
        }
    )
    success_agent._fallback_activated = True

    _success_session, success = turn_harness.run(success_agent)
    assert success[0]["status"] == "fallback"

    turn_harness.emitted.clear()
    turn_harness.db.rows.clear()
    turn_harness.db._ids.clear()
    exhausted_session = None

    def mark_exhausted():
        exhausted_session["_turn_fallback_notice"] = (
            "primary and fallback routes exhausted: all backends failed"
        )

    exhausted_agent = _Agent(
        {
            "completed": False,
            "failed": True,
            "error": "HTTP 502: all backends failed",
            "final_response": "",
            "messages": [],
        },
        on_run=mark_exhausted,
    )
    exhausted_session = _session(exhausted_agent)
    server._start_inflight_turn(exhausted_session, "run")
    server._run_prompt_submit(
        "request-2", "runtime-session", exhausted_session, "run"
    )
    exhausted = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]

    assert len(exhausted) == 1
    assert exhausted[0]["status"] == "failed"
    assert "fallback routes exhausted" in exhausted[0]["reason"]


def test_unexpected_exception_has_no_raw_error_event_and_no_duplicate(turn_harness):
    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    agent = _Agent(error=RuntimeError(f"provider exploded with {secret}"))

    _session_row, outcomes = turn_harness.run(agent)

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "failed"
    assert secret not in str(outcomes[0])
    assert not [args for args in turn_harness.emitted if args[0] == "error"]
    assert len(turn_harness.db.rows) == 1


def test_leftover_steer_precedes_prompt_queued_later(turn_harness):
    result = {
        "completed": True,
        "final_response": "done",
        "messages": [{"role": "assistant", "content": "done"}],
        "pending_steer": "accepted late steer",
    }
    session, outcomes = turn_harness.run(
        _Agent(result),
        queued_prompt={"text": "later queued prompt", "transport": "later-client"},
    )

    assert outcomes[0]["status"] == "completed"
    assert session["queued_prompt"] == {
        "text": "accepted late steer\n\nlater queued prompt",
        "transport": "later-client",
    }


def test_context_reference_refusal_uses_terminal_outcome_not_error_event(
    turn_harness, monkeypatch
):
    import agent.context_references as context_references
    import agent.model_metadata as model_metadata

    secret = "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"
    agent = _Agent(
        {
            "completed": True,
            "final_response": "must not run",
            "messages": [],
        }
    )
    monkeypatch.setattr(model_metadata, "get_model_context_length", lambda *_args, **_kwargs: 8192)
    monkeypatch.setattr(
        context_references,
        "preprocess_context_references",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            blocked=True,
            message="",
            warnings=[f"Context injection refused; Authorization: Bearer {secret}"],
        ),
    )

    _session_row, outcomes = turn_harness.run(agent, text="inspect @file:outside")

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "failed"
    assert "Context injection refused" in outcomes[0]["reason"]
    assert secret not in str(outcomes[0])
    assert not [args for args in turn_harness.emitted if args[0] == "error"]


def test_agent_initialization_failure_uses_terminal_outcome(
    turn_harness, monkeypatch
):
    session = _session(_Agent(None))
    session["running"] = False
    server._sessions["runtime-session"] = session
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda _session: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda _session: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "_wait_agent",
        lambda *_args: {
            "error": {
                "message": "agent initialization failed: no usable backend"
            }
        },
    )

    try:
        response = server.handle_request(
            {
                "id": "request-init",
                "method": "prompt.submit",
                "params": {
                    "session_id": "runtime-session",
                    "text": "hello",
                },
            }
        )
    finally:
        server._sessions.pop("runtime-session", None)

    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert response["result"]["status"] == "streaming"
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "failed"
    assert "initialization failed" in outcomes[0]["reason"]
    assert not [args for args in turn_harness.emitted if args[0] == "error"]


def test_interrupt_finalizes_a_stranded_inflight_turn(turn_harness, monkeypatch):
    agent = _Agent(None)
    agent.interrupt = lambda: None
    session = _session(agent)
    server._start_inflight_turn(session, "hello")
    server._sessions["runtime-session"] = session
    monkeypatch.setattr(server, "_clear_pending", lambda *_args: None)

    try:
        response = server.handle_request(
            {
                "id": "request-cancel",
                "method": "session.interrupt",
                "params": {"session_id": "runtime-session"},
            }
        )
    finally:
        server._sessions.pop("runtime-session", None)

    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert response["result"]["status"] == "interrupted"
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "cancelled"
    assert session["inflight_turn"] is None


def test_duplicate_and_late_finalize_are_idempotent(turn_harness):
    agent = _Agent({"completed": True, "final_response": "done"})
    session = _session(agent)
    server._start_inflight_turn(session, "run")

    first = server._finalize_turn_outcome(
        "runtime-session", session, result=agent._result
    )
    second = server._finalize_turn_outcome(
        "runtime-session",
        session,
        result={"completed": False, "error": "late failure"},
    )

    assert first == second
    assert first["status"] == "completed"
    assert len(turn_harness.db.rows) == 1
    assert [args[0] for args in turn_harness.emitted].count("turn.outcome") == 1


def test_duplicate_finalize_retries_failed_persistence_without_reemitting(
    turn_harness, monkeypatch
):
    class _FlakyDb(_OutcomeDb):
        def __init__(self):
            super().__init__()
            self.attempts = 0

        def record_turn_outcome(self, *args, **kwargs):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("database is locked")
            return super().record_turn_outcome(*args, **kwargs)

    flaky = _FlakyDb()
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(flaky),
    )
    agent = _Agent({"completed": True, "final_response": "done"})
    session = _session(agent)
    server._start_inflight_turn(session, "run")

    server._finalize_turn_outcome("runtime-session", session, result=agent._result)
    server._finalize_turn_outcome("runtime-session", session, result=agent._result)

    assert flaky.attempts == 2
    assert len(flaky.rows) == 1
    assert [args[0] for args in turn_harness.emitted].count("turn.outcome") == 1


def test_queued_dispatch_sync_failure_restores_prompt_and_fails_loudly_once(
    turn_harness, monkeypatch
):
    session = _session(_Agent(None))
    session["running"] = False
    session["queued_prompt"] = {
        "text": "do not lose this prompt",
        "transport": "desktop-client",
    }
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatch exploded")),
    )

    assert REAL_DRAIN_QUEUED_PROMPT(
        "request-queued", "runtime-session", session
    )

    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert session["queued_prompt"] == {
        "text": "do not lose this prompt",
        "transport": "desktop-client",
    }
    assert session["running"] is False
    assert session["inflight_turn"] is None
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "failed"
    assert "dispatch exploded" in outcomes[0]["reason"]
    assert len(turn_harness.db.rows) == 1


def test_notification_sync_failure_restores_current_and_entire_batch_tail(
    turn_harness, monkeypatch
):
    session = _session(_Agent(None))
    session["running"] = False
    notifications = [
        ({"id": "event-1"}, "first"),
        ({"id": "event-2"}, "second"),
        ({"id": "event-3"}, "third"),
    ]
    completion_queue = queue.Queue()
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatch exploded")),
    )

    server._dispatch_notification_batch(
        "request-notification",
        "runtime-session",
        session,
        notifications,
        completion_queue,
    )

    restored = [completion_queue.get_nowait() for _ in range(3)]
    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert restored == [event for event, _text in notifications]
    assert completion_queue.empty()
    assert session["running"] is False
    assert session["inflight_turn"] is None
    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "failed"
    assert len(turn_harness.db.rows) == 1


def test_notification_batch_requeues_full_tail_after_first_dispatch_starts(
    turn_harness, monkeypatch
):
    session = _session(_Agent(None))
    session["running"] = False
    notifications = [
        ({"id": "event-1"}, "first"),
        ({"id": "event-2"}, "second"),
        ({"id": "event-3"}, "third"),
    ]
    completion_queue = queue.Queue()
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_args, **_kwargs: None)

    server._dispatch_notification_batch(
        "request-notification",
        "runtime-session",
        session,
        notifications,
        completion_queue,
    )

    assert session["running"] is True
    assert [completion_queue.get_nowait(), completion_queue.get_nowait()] == [
        notifications[1][0],
        notifications[2][0],
    ]
    assert completion_queue.empty()


def test_goal_dispatch_sync_failure_restores_behind_queued_user_work(
    turn_harness, monkeypatch
):
    session = _session(_Agent(None))
    session["running"] = False
    session["queued_prompt"] = {
        "text": "new user work",
        "transport": "desktop-client",
    }
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("dispatch exploded")),
    )

    assert not server._dispatch_goal_continuation(
        "request-goal", "runtime-session", session, "continue the goal"
    )

    assert session["queued_prompt"] == {
        "text": "new user work\n\ncontinue the goal",
        "transport": "desktop-client",
    }
    assert session["running"] is False
    assert session["inflight_turn"] is None
    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert [outcome["status"] for outcome in outcomes] == ["failed"]
    assert len(turn_harness.db.rows) == 1


def test_completion_freezes_before_delayed_message_complete_delivery(
    turn_harness, monkeypatch
):
    complete_blocked = threading.Event()
    release_complete = threading.Event()
    agent = _Agent(
        {
            "completed": True,
            "final_response": "complete answer",
            "messages": [{"role": "assistant", "content": "complete answer"}],
        }
    )
    agent.interrupt = lambda: None
    session = _session(agent)
    server._start_inflight_turn(session, "run")
    server._sessions["runtime-session"] = session
    monkeypatch.setattr(server.threading, "Thread", REAL_THREAD)
    monkeypatch.setattr(server, "_clear_pending", lambda *_args: None)

    def emit(*args):
        turn_harness.emitted.append(args)
        if args[0] == "message.complete":
            complete_blocked.set()
            assert release_complete.wait(2)

    monkeypatch.setattr(server, "_emit", emit)
    try:
        server._run_prompt_submit(
            "request-race", "runtime-session", session, "run"
        )
        run_thread = session["_run_thread"]
        assert complete_blocked.wait(2)
        response = server.handle_request(
            {
                "id": "request-stop",
                "method": "session.interrupt",
                "params": {"session_id": "runtime-session"},
            }
        )
        release_complete.set()
        run_thread.join(2)
    finally:
        release_complete.set()
        server._sessions.pop("runtime-session", None)

    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert response["result"]["status"] == "interrupted"
    assert not run_thread.is_alive()
    assert [outcome["status"] for outcome in outcomes] == ["completed"]
    completes = [
        args[2] for args in turn_harness.emitted if args[0] == "message.complete"
    ]
    assert [payload["status"] for payload in completes] == ["complete"]
    assert len(turn_harness.db.rows) == 1


def test_stop_freezes_cancellation_before_agent_completion(
    turn_harness, monkeypatch
):
    agent_entered = threading.Event()
    release_agent = threading.Event()

    def wait_for_release():
        agent_entered.set()
        assert release_agent.wait(2)

    agent = _Agent(
        {
            "completed": True,
            "final_response": "late answer",
            "messages": [{"role": "assistant", "content": "late answer"}],
        },
        on_run=wait_for_release,
    )
    agent.interrupt = lambda: None
    session = _session(agent)
    session["pending_title"] = "must not be applied"
    server._start_inflight_turn(session, "run")
    server._sessions["runtime-session"] = session
    monkeypatch.setattr(server.threading, "Thread", REAL_THREAD)
    monkeypatch.setattr(server, "_clear_pending", lambda *_args: None)
    side_effects = []

    class _GoalManager:
        def __init__(self, **_kwargs):
            pass

        def is_active(self):
            return True

        def evaluate_after_turn(self, *_args, **_kwargs):
            side_effects.append("goal")
            return {"should_continue": False}

    class _TitleDb:
        def set_session_title(self, *_args, **_kwargs):
            side_effects.append("title")
            return True

    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.goals",
        types.SimpleNamespace(
            GoalManager=_GoalManager,
            gather_background_processes=lambda: [],
        ),
    )
    monkeypatch.setattr(server, "_get_db", lambda: _TitleDb())
    monkeypatch.setattr(
        server,
        "_voice_tts_enabled",
        lambda: side_effects.append("tts-check") or False,
    )
    try:
        server._run_prompt_submit(
            "request-race", "runtime-session", session, "run"
        )
        run_thread = session["_run_thread"]
        assert agent_entered.wait(2)
        response = server.handle_request(
            {
                "id": "request-stop",
                "method": "session.interrupt",
                "params": {"session_id": "runtime-session"},
            }
        )
        release_agent.set()
        run_thread.join(2)
    finally:
        release_agent.set()
        server._sessions.pop("runtime-session", None)

    outcomes = [
        args[2] for args in turn_harness.emitted if args[0] == "turn.outcome"
    ]
    assert response["result"]["status"] == "interrupted"
    assert not run_thread.is_alive()
    assert [outcome["status"] for outcome in outcomes] == ["cancelled"]
    completes = [
        args[2] for args in turn_harness.emitted if args[0] == "message.complete"
    ]
    assert [payload["status"] for payload in completes] == ["interrupted"]
    assert len(turn_harness.db.rows) == 1
    assert side_effects == []
    assert session["pending_title"] == "must not be applied"


def test_inflight_snapshot_carries_turn_identity():
    session = _session(_Agent(None))
    server._start_inflight_turn(session, "resume me")

    snapshot = server._inflight_snapshot(session)

    assert snapshot["turn_id"] == session["inflight_turn"]["id"]


def test_display_projection_inserts_outcomes_without_mutating_prompt_history():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]
    original = [dict(row) for row in history]
    outcomes = [
        {
            "completed_at": 3.0,
            "model": "m",
            "provider": "p",
            "reason": "response delivered",
            "started_at": 1.0,
            "status": "completed",
            "text": "turn:completed · p/m · response delivered",
            "turn_id": "turn-1",
            "user_ordinal": 0,
        }
    ]

    projected = server._history_to_messages(history, outcomes)

    assert history == original
    assert [row["role"] for row in projected] == [
        "user",
        "assistant",
        "system",
        "user",
    ]
    assert projected[2]["turn_outcome"]["id"] == "turn-1"


def test_outcome_without_persisted_prompt_still_projects_as_visible_system_row():
    projected = server._history_to_messages(
        [],
        [
            {
                "completed_at": 2.0,
                "status": "failed",
                "text": "turn:failed · unknown provider/unknown model · init failed",
                "turn_id": "turn-init",
                "user_ordinal": 0,
            }
        ],
    )

    assert projected == [
        {
            "role": "system",
            "text": "turn:failed · unknown provider/unknown model · init failed",
            "timestamp": 2.0,
            "turn_outcome": {
                "completed_at": 2.0,
                "id": "turn-init",
                "status": "failed",
                "text": "turn:failed · unknown provider/unknown model · init failed",
                "turn_id": "turn-init",
                "user_ordinal": 0,
            },
        }
    ]
