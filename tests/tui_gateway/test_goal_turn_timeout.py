"""A turn that times out mid-goal must not strand the goal loop.

Observed 2026-08-02: a delegated build session (deepseek-v4-flash-0731-ds4)
stopped dead when one turn ran past the provider stale-timeout. The turn was
classified ``timed_out`` and rendered ``turn:timed out · …``
(``_derive_turn_outcome`` / ``_format_turn_outcome``), and ``message.complete``
remapped that to ``status="error"``. The post-turn goal hook only ran for
``status == "complete"``, so the goal was never told the turn had ended: it
stayed ``active`` with no continuation queued and nothing driving it. The
session looked alive. Nothing was happening.

What the fix has to get right, and what each test here pins:

1. A timed-out turn reaches the goal as a FAILED CONTINUATION, and is retried.
2. The second consecutive one stalls the goal visibly instead of retrying
   forever — an operator watching must be able to see it stopped.
3. A turn the USER cancelled is not a failure and must not be retried, or Stop
   would mean nothing.
4. A completed turn still goes to the judge, exactly as before.
"""

from __future__ import annotations

import contextlib
import threading
import types
from pathlib import Path

import pytest

from tui_gateway import server


REAL_THREAD = threading.Thread
# Captured before any fixture stubs it, so the chaining test below can put the
# genuine dispatcher back and prove turns really do fire themselves.
REAL_DISPATCH = server._dispatch_goal_continuation

TIMED_OUT_RESULT = {
    "completed": False,
    "failed": True,
    "error": "Request timed out after 180s",
    "failure_reason": "timeout",
    "messages": [{"role": "user", "content": "run"}],
}

FAILED_RESULT = {
    "completed": False,
    "failed": True,
    "error": "provider returned 500",
    "messages": [{"role": "user", "content": "run"}],
}

COMPLETED_RESULT = {
    "completed": True,
    "final_response": "did a step",
    "messages": [
        {"role": "user", "content": "run"},
        {"role": "assistant", "content": "did a step"},
    ],
}


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


class _Agent:
    def __init__(self, result):
        self.model = "deepseek-v4-flash-0731-ds4"
        self.provider = "ai-router"
        self._fallback_activated = False
        # A list means "one result per turn", so a test can drive a session
        # across several turns and see each one separately.
        self._results = list(result) if isinstance(result, list) else None
        self._result = None if self._results else result
        self.prompts: list[str] = []

    def clear_interrupt(self):
        return None

    def run_conversation(self, user_input=None, *_args, **_kwargs):
        self.prompts.append(str(user_input or ""))
        if self._results is not None:
            return self._results.pop(0) if self._results else COMPLETED_RESULT
        return self._result


class _OutcomeDb:
    def __init__(self):
        self._ids = set()

    def record_turn_outcome(self, _session_id, turn_id, **_row):
        if turn_id in self._ids:
            return False
        self._ids.add(turn_id)
        return True


SESSION_KEY = "goal-timeout-session"


@pytest.fixture()
def goal_harness(monkeypatch, tmp_path):
    """Drive a real turn through _run_prompt_submit against a real GoalManager.

    Only the agent and the follow-up dispatch are faked: the goal state under
    test is the one that actually gets persisted, so these assertions exercise
    the same code the desktop runs.
    """
    emitted = []
    continuations = []

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals
    from hermes_state import SessionDB

    # Point the goal store at this test's own state.db explicitly. Setting
    # HERMES_HOME alone is NOT enough: hermes_state resolves DEFAULT_DB_PATH
    # once at import, so a bare SessionDB() keeps writing to whichever home was
    # current when the module first loaded — and goals would leak between tests.
    goals._DB_CACHE.clear()
    test_db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: test_db)

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server, "_session_db", lambda _session: contextlib.nullcontext(_OutcomeDb())
    )
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_a: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_a: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_a: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_a, **_k: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_a: None)
    monkeypatch.setattr(server, "_session_info", lambda *_a, **_k: {})
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _t, _c: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_a: False)
    monkeypatch.setattr(
        server,
        "_dispatch_goal_continuation",
        lambda _rid, _sid, _session, prompt: continuations.append(prompt),
    )

    def run(result, *, cancelled=False):
        session = {
            "agent": _Agent(result),
            "attached_images": [],
            "cols": 80,
            "cwd": str(tmp_path),
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "inflight_turn": None,
            "running": True,
            "session_key": SESSION_KEY,
            "transport": None,
        }
        server._start_inflight_turn(session, "run the turn")
        if cancelled:
            session["_turn_cancel_requested"] = True
            session["_turn_cancel_reason"] = "user cancelled the turn"
        server._run_prompt_submit("request-1", "runtime-session", session, "run the turn")
        return session

    def goal_messages():
        return [
            args[2]["text"]
            for args in emitted
            if args[0] == "status.update" and args[2].get("kind") == "goal"
        ]

    yield types.SimpleNamespace(
        continuations=continuations,
        emitted=emitted,
        goal_messages=goal_messages,
        run=run,
    )

    goals._DB_CACHE.clear()


def _set_goal(max_turns=10):
    from hermes_cli.goals import GoalManager

    mgr = GoalManager(session_id=SESSION_KEY, default_max_turns=max_turns)
    mgr.set("ship the parser")
    return mgr


def _reload():
    from hermes_cli.goals import GoalManager

    return GoalManager(session_id=SESSION_KEY).state


class TestTimedOutTurnInsideAGoal:
    def test_timeout_retries_instead_of_stranding_the_loop(self, goal_harness) -> None:
        """The regression: this turn used to reach the goal not at all."""
        _set_goal()

        goal_harness.run(TIMED_OUT_RESULT)

        state = _reload()
        assert state.status == "active"
        assert state.consecutive_turn_failures == 1
        # A continuation was queued — the loop is still moving.
        assert len(goal_harness.continuations) == 1
        assert "ship the parser" in goal_harness.continuations[0]
        assert any("timed out" in m for m in goal_harness.goal_messages())

    def test_the_reason_survives_into_the_visible_message(self, goal_harness) -> None:
        _set_goal()

        goal_harness.run(TIMED_OUT_RESULT)

        assert any("180s" in m for m in goal_harness.goal_messages())

    def test_second_consecutive_timeout_stalls_the_goal_visibly(
        self, goal_harness
    ) -> None:
        _set_goal()

        goal_harness.run(TIMED_OUT_RESULT)
        goal_harness.run(TIMED_OUT_RESULT)

        state = _reload()
        assert state.status == "stalled"
        # No third attempt was queued.
        assert len(goal_harness.continuations) == 1
        stalled = [m for m in goal_harness.goal_messages() if "stalled" in m]
        assert stalled, goal_harness.goal_messages()
        assert "/goal resume" in stalled[-1]

    def test_a_plain_failure_is_handled_the_same_way(self, goal_harness) -> None:
        """Nothing to judge is nothing to judge, timeout or not."""
        _set_goal()

        goal_harness.run(FAILED_RESULT)

        state = _reload()
        assert state.status == "active"
        assert state.consecutive_turn_failures == 1
        assert any("failed" in m for m in goal_harness.goal_messages())

    def test_a_good_turn_after_a_timeout_clears_the_streak(
        self, goal_harness, monkeypatch
    ) -> None:
        from hermes_cli import goals

        _set_goal()
        monkeypatch.setattr(
            goals, "judge_goal", lambda *_a, **_k: ("continue", "more work", False, None, False)
        )

        goal_harness.run(TIMED_OUT_RESULT)
        assert _reload().consecutive_turn_failures == 1

        goal_harness.run(COMPLETED_RESULT)

        state = _reload()
        assert state.consecutive_turn_failures == 0
        assert state.status == "active"


class TestTurnsThatMustNotTouchTheGoal:
    def test_a_user_cancelled_turn_is_not_a_failed_continuation(
        self, goal_harness
    ) -> None:
        """Stop has to mean stop. Re-poking the agent here would undo it."""
        _set_goal()

        goal_harness.run(COMPLETED_RESULT, cancelled=True)

        state = _reload()
        assert state.status == "active"
        assert state.turns_used == 0
        assert state.consecutive_turn_failures == 0
        assert goal_harness.continuations == []

    def test_a_timeout_with_no_goal_set_changes_nothing(self, goal_harness) -> None:
        goal_harness.run(TIMED_OUT_RESULT)

        assert _reload() is None
        assert goal_harness.continuations == []
        assert goal_harness.goal_messages() == []

    def test_a_timeout_on_a_paused_goal_does_not_resurrect_it(
        self, goal_harness
    ) -> None:
        mgr = _set_goal()
        mgr.pause()

        goal_harness.run(TIMED_OUT_RESULT)

        state = _reload()
        assert state.status == "paused"
        assert state.turns_used == 0
        assert goal_harness.continuations == []


class TestCompletedTurnsStillGoToTheJudge:
    def test_done_verdict_ends_auto_continuation(self, goal_harness, monkeypatch) -> None:
        from hermes_cli import goals

        _set_goal()
        monkeypatch.setattr(
            goals, "judge_goal", lambda *_a, **_k: ("done", "shipped", False, None, False)
        )

        goal_harness.run(COMPLETED_RESULT)

        assert _reload().status == "done"
        assert goal_harness.continuations == []
        assert any("achieved" in m for m in goal_harness.goal_messages())

    def test_budget_cap_ends_auto_continuation_visibly(
        self, goal_harness, monkeypatch
    ) -> None:
        from hermes_cli import goals

        _set_goal(max_turns=1)
        monkeypatch.setattr(
            goals, "judge_goal", lambda *_a, **_k: ("continue", "more work", False, None, False)
        )

        goal_harness.run(COMPLETED_RESULT)

        state = _reload()
        assert state.status == "paused"
        assert "budget" in (state.paused_reason or "").lower()
        assert goal_harness.continuations == []
        assert any("turns used" in m for m in goal_harness.goal_messages())


class TestTheLoopActuallyChainsItself:
    """The whole point of a goal: nobody has to type between turns.

    Everything above stubs `_dispatch_goal_continuation` to observe the
    decision. These tests put the REAL dispatcher back, so a continuation
    genuinely re-enters `_run_prompt_submit` — which is the only way to prove
    the loop runs without a human, rather than that it intended to.
    """

    def test_a_three_turn_goal_finishes_with_one_submit(
        self, goal_harness, monkeypatch
    ) -> None:
        from hermes_cli import goals

        monkeypatch.setattr(server, "_dispatch_goal_continuation", REAL_DISPATCH)
        _set_goal(max_turns=20)

        verdicts = iter(
            [
                ("continue", "step 1 done, more to do", False, None, False),
                ("continue", "step 2 done, more to do", False, None, False),
                ("done", "parser ships", False, None, False),
            ]
        )
        monkeypatch.setattr(goals, "judge_goal", lambda *_a, **_k: next(verdicts))

        session = goal_harness.run([COMPLETED_RESULT, COMPLETED_RESULT, COMPLETED_RESULT])

        agent = session["agent"]
        # One human submit; the other two turns were the loop's own doing.
        assert len(agent.prompts) == 3, agent.prompts
        assert agent.prompts[0] == "run the turn"
        assert all("ship the parser" in p for p in agent.prompts[1:])

        state = _reload()
        assert state.status == "done"
        assert state.turns_used == 3

    def test_a_timeout_mid_goal_is_retried_and_the_goal_still_finishes(
        self, goal_harness, monkeypatch
    ) -> None:
        """The exact shape of the 2026-08-02 failure, end to end: turn two dies
        on the per-turn timeout. Before the fix the loop stopped there with the
        goal still `active`. Now it retries and reaches the objective."""
        from hermes_cli import goals

        monkeypatch.setattr(server, "_dispatch_goal_continuation", REAL_DISPATCH)
        _set_goal(max_turns=20)

        verdicts = iter(
            [
                ("continue", "step 1 done", False, None, False),
                ("done", "parser ships", False, None, False),
            ]
        )
        monkeypatch.setattr(goals, "judge_goal", lambda *_a, **_k: next(verdicts))

        session = goal_harness.run(
            [COMPLETED_RESULT, TIMED_OUT_RESULT, COMPLETED_RESULT]
        )

        agent = session["agent"]
        # Turn 1 good, turn 2 timed out, turn 3 is the retry — all unattended.
        assert len(agent.prompts) == 3, agent.prompts

        state = _reload()
        assert state.status == "done"
        # The timed-out turn counted, and the good turn after it cleared the streak.
        assert state.turns_used == 3
        assert state.consecutive_turn_failures == 0
        assert any("timed out" in m for m in goal_harness.goal_messages())
        assert any("achieved" in m for m in goal_harness.goal_messages())

    def test_the_cap_stops_the_loop_and_says_so(self, goal_harness, monkeypatch) -> None:
        """A judge that never says done must still terminate — visibly."""
        from hermes_cli import goals

        monkeypatch.setattr(server, "_dispatch_goal_continuation", REAL_DISPATCH)
        _set_goal(max_turns=3)
        monkeypatch.setattr(
            goals, "judge_goal", lambda *_a, **_k: ("continue", "never satisfied", False, None, False)
        )

        session = goal_harness.run([COMPLETED_RESULT] * 10)

        agent = session["agent"]
        assert len(agent.prompts) == 3, agent.prompts

        state = _reload()
        assert state.status == "paused"
        assert state.turns_used == 3
        assert "budget" in (state.paused_reason or "").lower()
        capped = [m for m in goal_harness.goal_messages() if "3/3" in m]
        assert capped, goal_harness.goal_messages()

    def test_repeated_timeouts_stall_the_loop_instead_of_spinning(
        self, goal_harness, monkeypatch
    ) -> None:
        """The runaway guard: a goal whose every turn dies must stop after the
        one retry, not burn the whole budget on turns that report nothing."""
        monkeypatch.setattr(server, "_dispatch_goal_continuation", REAL_DISPATCH)
        _set_goal(max_turns=20)

        session = goal_harness.run([TIMED_OUT_RESULT] * 10)

        agent = session["agent"]
        # The original turn plus exactly one retry.
        assert len(agent.prompts) == 2, agent.prompts

        state = _reload()
        assert state.status == "stalled"
        assert state.turns_used == 2
