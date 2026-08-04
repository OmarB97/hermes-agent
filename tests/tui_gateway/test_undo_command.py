"""Tests for /undo handling in tui_gateway.

The TUI routes ``/undo`` through ``command.dispatch`` (it's in
``_PENDING_INPUT_COMMANDS`` because the CLI handler queues input the
slash-worker subprocess can't read). The server handles it directly,
mutates SessionDB to soft-delete rows, refreshes the in-memory session
history, fires the memory-provider hook with ``rewound=True``, and
returns ``{"type": "prefill", "message": <text>, "notice": ...}`` so
the Ink client drops the message into the composer for editing.

``/undo N`` backs up N user turns at once (default 1). See issue #21910.
"""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from hermes_state import SessionDB


@pytest.fixture()
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    yield home


@pytest.fixture()
def server(hermes_home):
    # Mocks are scoped to the initial import only (see
    # tests/tui_gateway/test_protocol.py for the rationale).
    with patch.dict(
        "sys.modules",
        {
            "hermes_cli.env_loader": MagicMock(),
            "hermes_cli.banner": MagicMock(),
        },
    ):
        mod = importlib.import_module("tui_gateway.server")

    methods = dict(mod._methods)
    yield mod
    # Restore in place instead of clear+reload: importlib.reload
    # re-registers atexit hooks (duplicate ThreadPoolExecutor shutdowns
    # race the stderr buffer at interpreter exit — same class as PR #34217)
    # and re-captures module-level paths like _hermes_home against this
    # test's soon-deleted tmpdir, breaking later files in the same process.
    mod._methods.clear()
    mod._methods.update(methods)
    mod._sessions.clear()
    mod._pending.clear()
    mod._answers.clear()
    mod._db = None


@pytest.fixture()
def db(hermes_home):
    return SessionDB(db_path=hermes_home / "state.db")


@pytest.fixture()
def session_with_history(server, db):
    """Build a session with 3 user turns + assistant replies persisted in DB."""
    sid = "sid-undo"
    session_key = "tui-undo-1"
    db.create_session(session_key, source="tui")
    for i in range(1, 4):
        db.append_message(session_key, "user", f"question {i}")
        db.append_message(session_key, "assistant", f"answer {i}")
    history = db.get_messages_as_conversation(session_key)
    agent = MagicMock()
    agent._memory_manager = MagicMock()
    agent._last_flushed_db_idx = len(history)
    s = {
        "session_key": session_key,
        "history": list(history),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "agent": agent,
        "attached_images": [],
        "cols": 120,
    }
    server._sessions[sid] = s
    # Wire the DB cache so _get_db() returns our fixture.
    server._db = db
    return sid, session_key, s, agent


def _call(server, method, **params):
    return server._methods[method](1, params)


def test_undo_returns_prefill_with_target_text(server, session_with_history):
    sid, session_key, s, agent = session_with_history
    resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
    result = resp["result"]
    assert result["type"] == "prefill"
    # Default /undo backs up one user turn — "question 3"
    assert result["message"] == "question 3"
    assert "Undid" in result["notice"]


class TestUndoFollowsTheSessionProfile:
    """/undo must rewind the session's OWN profile db, not the launch home's.

    In app-global remote mode one backend serves every profile, so a session
    carries ``profile_home`` and its rows live in that profile's ``state.db``.
    ``_get_db()`` is pinned to the gateway's launch home — ``hermes_state``
    freezes ``DEFAULT_DB_PATH`` at import — so reading through it found no
    rows for the session and /undo dead-ended on "no user messages to undo".
    ``_session_db`` is the profile-aware handle (same one ``_ensure_session_db_row``
    uses); these tests pin /undo to it.
    """

    @pytest.fixture()
    def profile_session(self, server, tmp_path, db):
        """A session whose rows live in a profile db, with a decoy launch db.

        ``db`` (the launch-home fixture) is wired into ``server._db`` holding a
        DIFFERENT session, so anything reaching for ``_get_db()`` sees a db that
        simply has no rows under this session's key.
        """
        profile_home = tmp_path / "profiles" / "work"
        profile_home.mkdir(parents=True)
        pdb = SessionDB(db_path=profile_home / "state.db")
        session_key = "tui-undo-profile-1"
        pdb.create_session(session_key, source="tui")
        for i in range(1, 4):
            pdb.append_message(session_key, "user", f"profile question {i}")
            pdb.append_message(session_key, "assistant", f"profile answer {i}")

        # Decoy: the launch db is live but knows nothing about this session.
        db.create_session("tui-launch-other", source="tui")
        server._db = db

        agent = MagicMock()
        agent._memory_manager = MagicMock()
        sid = "sid-undo-profile"
        server._sessions[sid] = {
            "session_key": session_key,
            "profile_home": str(profile_home),
            "history": list(pdb.get_messages_as_conversation(session_key)),
            "history_lock": threading.Lock(),
            "history_version": 0,
            "running": False,
            "agent": agent,
            "attached_images": [],
            "cols": 120,
        }
        return sid, session_key, pdb

    def test_undo_rewinds_the_profile_db(self, server, profile_session):
        sid, session_key, pdb = profile_session
        resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
        assert "error" not in resp, resp
        assert resp["result"]["message"] == "profile question 3"

        rows = pdb.get_messages(session_key, include_inactive=True)
        assert len(rows) == 6
        assert len([r for r in rows if r["active"] == 1]) == 4
        assert pdb.get_session(session_key)["rewind_count"] == 1

    def test_undo_leaves_the_launch_db_untouched(self, server, profile_session, db):
        sid, session_key, _ = profile_session
        _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
        # The rewind must not have created or touched a row over in the
        # launch home under this session's key.
        assert db.get_session(session_key) is None

    def test_launchless_session_still_uses_the_shared_handle(
        self, server, session_with_history, db
    ):
        """No profile_home (the ordinary local case) keeps the _get_db() path."""
        sid, session_key, _, _ = session_with_history
        assert "profile_home" not in server._sessions[sid]
        resp = _call(server, "command.dispatch", session_id=sid, name="undo", arg="")
        assert "error" not in resp, resp
        assert db.get_session(session_key)["rewind_count"] == 1
