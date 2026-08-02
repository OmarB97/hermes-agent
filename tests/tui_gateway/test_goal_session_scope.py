"""Standing goals bound at ``session.create`` (desktop backend).

Sibling of test_toolsets_session_scope.py. The Ralph goal loop in
``hermes_cli/goals.py`` has always existed; what did not exist was a way to
START a session with a goal. The only route in was the ``/goal <text>`` slash
command, which sets the goal and re-sends the text as the kickoff turn — so a
programmatic caller (``hermes desktop spawn``) had to smuggle its objective in
as ``"/goal <objective>"`` and rely on the composer's command vocabulary. That
worked, but it made the goal a property of the first *message* rather than of
the session, and it only existed after a turn had already been submitted.

Contract under test:

1. A ``goal`` on session.create is stored against the session's own key — the
   same key ``_run_prompt_submit``'s post-turn hook reads — so the FIRST turn
   is already judged against it.
2. ``goal_max_turns`` sets the auto-continuation cap for that goal alone.
3. Bad shapes fail the create rather than opening a chat whose goal silently
   is not there.
4. Omitting it leaves a session with no goal — the unchanged path.
5. A SessionDB failure does NOT fail the create: a chat with no loop still
   beats no chat at all. It is logged instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import tui_gateway.server as server


@pytest.fixture
def created_sessions(monkeypatch, tmp_path):
    """Call session.create for real, minus the background agent build.

    HERMES_HOME is redirected so the goal rows land in a throwaway SessionDB
    instead of the developer's own.
    """
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))

    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals
    from hermes_state import SessionDB

    # Explicit db_path, not just HERMES_HOME: hermes_state freezes
    # DEFAULT_DB_PATH at import, so a bare SessionDB() would write to whichever
    # home was current when the module first loaded.
    goals._DB_CACHE.clear()
    test_db = SessionDB(db_path=home / "state.db")
    monkeypatch.setattr(goals, "_get_session_db", lambda: test_db)

    made: list[str] = []

    def create(params: dict) -> dict:
        resp = server._methods["session.create"]("rid-1", params)
        if "result" in resp:
            made.append(resp["result"]["session_id"])
        return resp

    yield create

    for sid in made:
        server._sessions.pop(sid, None)
    goals._DB_CACHE.clear()


def _goal_of(resp: dict):
    """Load the goal the way the post-turn hook does — by session_key."""
    from hermes_cli.goals import GoalManager

    key = server._sessions[resp["result"]["session_id"]]["session_key"]
    return GoalManager(session_id=key).state


class TestSessionCreateAcceptsGoal:
    def test_goal_is_bound_to_the_session_key(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "goal": "ship the parser"})

        state = _goal_of(resp)
        assert state is not None
        assert state.goal == "ship the parser"
        assert state.status == "active"
        # Nothing has run yet, so the first turn still gets judged.
        assert state.turns_used == 0

    def test_goal_text_is_stripped(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "goal": "  ship the parser  "})

        assert _goal_of(resp).goal == "ship the parser"

    def test_goal_max_turns_sets_the_cap(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "goal": "ship it", "goal_max_turns": 40})

        assert _goal_of(resp).max_turns == 40

    def test_omitted_goal_leaves_the_session_without_one(self, created_sessions) -> None:
        """The unchanged path every typed chat takes."""
        resp = created_sessions({"cols": 80})

        assert _goal_of(resp) is None

    def test_goal_default_cap_comes_from_config(self, created_sessions, monkeypatch) -> None:
        monkeypatch.setattr(server, "_load_cfg", lambda: {"goals": {"max_turns": 7}})

        resp = created_sessions({"cols": 80, "goal": "ship it"})

        assert _goal_of(resp).max_turns == 7


class TestSessionCreateRejectsBadGoal:
    """A goal that cannot be honored must fail the create, not be dropped."""

    def test_non_string_goal_is_refused(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "goal": ["ship it"]})

        assert "error" in resp
        assert "goal must be a string" in resp["error"]["message"]

    def test_blank_goal_is_refused(self, created_sessions) -> None:
        """An all-whitespace goal is a typo, not a request for no goal."""
        resp = created_sessions({"cols": 80, "goal": "   "})

        assert "error" in resp
        assert "empty" in resp["error"]["message"]

    def test_a_refused_goal_leaves_no_session_behind(self, created_sessions) -> None:
        before = set(server._sessions)

        created_sessions({"cols": 80, "goal": ""})

        assert set(server._sessions) == before

    @pytest.mark.parametrize("bad", ["40", 0, -1, True])
    def test_bad_goal_max_turns_is_refused(self, created_sessions, bad) -> None:
        resp = created_sessions({"cols": 80, "goal": "ship it", "goal_max_turns": bad})

        assert "error" in resp
        assert "goal_max_turns" in resp["error"]["message"]

    def test_goal_max_turns_without_a_goal_is_refused(self, created_sessions) -> None:
        """A budget with no goal would be read by nobody."""
        resp = created_sessions({"cols": 80, "goal_max_turns": 40})

        assert "error" in resp
        assert "requires goal" in resp["error"]["message"]

    def test_null_goal_is_treated_as_omitted(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "goal": None})

        assert "result" in resp
        assert _goal_of(resp) is None


class TestGoalStorageFailureDoesNotFailTheCreate:
    def test_session_still_opens_when_the_goal_cannot_be_stored(
        self, created_sessions, monkeypatch, capsys
    ) -> None:
        """A chat with no loop beats no chat at all — but say so loudly."""
        from hermes_cli import goals

        def boom(*_a, **_k):
            raise RuntimeError("state_meta is unwritable")

        monkeypatch.setattr(goals.GoalManager, "set", boom)

        resp = created_sessions({"cols": 80, "goal": "ship it"})

        assert "result" in resp
        assert "could not set goal" in capsys.readouterr().err
