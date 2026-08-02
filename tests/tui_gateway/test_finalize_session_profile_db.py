"""``_finalize_session`` must close a session's row in ITS OWN profile db.

Real ``SessionDB`` files under a temp HERMES_HOME with two profiles — no db
mocks — because a mocked handle hides the defect entirely: the finalize write
resolved its db with ``_get_db()``, the launch profile's cached ``SessionDB``
(bound to ``DEFAULT_DB_PATH``, evaluated at import time), while every
neighbouring write in ``tui_gateway/server.py`` goes through the profile-aware
``_session_db(session)``.

A session created with ``profile: "<other>"`` keeps its row in
``<root>/profiles/<other>/state.db``, so that had two consequences:

1. ``get_session`` returned None → ``source`` was "" → the #60609 gateway-owned
   guard never fired. The TUI treated every profile session as one it owns,
   including a Telegram/Discord session it is only a viewer of.
2. ``end_session`` ran ``UPDATE ... WHERE id = ? AND ended_at IS NULL`` against
   the launch db and matched 0 rows — a silent no-op. The row was closed later
   as ``agent_close`` by agent teardown, a reason
   ``find_latest_gateway_session_for_peer`` treats as *recoverable*, so a
   cleanly-closed session stayed stale-routable.

Both profile dbs seed a row under the SAME id so the write target is
unambiguous: before the fix it was the launch row that moved.
"""

import contextlib
import threading
import types

import pytest

import tools.async_delegation as async_delegation
from hermes_state import SessionDB
from tui_gateway import server

SESSION_ID = "sess-profile-1"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A real HERMES_HOME root: launch ``state.db`` + one extra profile's."""
    root = tmp_path / "hermes"
    worker_home = root / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    launch = SessionDB(db_path=root / "state.db")
    worker = SessionDB(db_path=worker_home / "state.db")
    # Never let the finalize path fall through to the host's real state.db.
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    try:
        yield types.SimpleNamespace(
            launch=launch, worker=worker, worker_home=worker_home
        )
    finally:
        for db in (launch, worker):
            with contextlib.suppress(Exception):
                db.close()


def _session(profile_home=None, *, sid="tab1"):
    return {
        "agent": types.SimpleNamespace(session_id=SESSION_ID),
        "history": [],
        "history_lock": threading.Lock(),
        "session_key": SESSION_ID,
        "profile_home": str(profile_home) if profile_home else None,
        "_sid": sid,
    }


def _seed(homes, *, profile_source, launch_source="desktop"):
    homes.worker.create_session(SESSION_ID, source=profile_source)
    homes.launch.create_session(SESSION_ID, source=launch_source)


def test_profile_session_is_ended_in_its_own_profile_db(homes):
    _seed(homes, profile_source="desktop")

    server._finalize_session(_session(homes.worker_home), end_reason="tui_close")

    row = homes.worker.get_session(SESSION_ID)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "tui_close"
    # The launch profile's own row is untouched — it was never this session's.
    assert homes.launch.get_session(SESSION_ID)["ended_at"] is None


def test_gateway_owned_profile_session_is_not_ended(homes):
    """#60609's guard has to read the source from the session's OWN db.

    Reading the launch db returned no row (source=""), so the TUI ended a
    gateway-owned profile session as ``tui_close``/``ws_orphan_reap`` — the
    write the guard exists to prevent.
    """
    _seed(homes, profile_source="telegram")

    server._finalize_session(_session(homes.worker_home), end_reason="ws_orphan_reap")

    assert homes.worker.get_session(SESSION_ID)["ended_at"] is None
    assert homes.launch.get_session(SESSION_ID)["ended_at"] is None


def test_launch_profile_session_still_ends_in_the_launch_db(homes):
    """Control: a session with no profile binding is unchanged."""
    homes.launch.create_session(SESSION_ID, source="tui")

    server._finalize_session(_session(None), end_reason="tui_close")

    row = homes.launch.get_session(SESSION_ID)
    assert row["ended_at"] is not None
    assert row["end_reason"] == "tui_close"


def test_profile_session_delegations_are_interrupted_by_key(homes, monkeypatch):
    """The TUI owns a desktop/TUI profile session — its background subagents
    end with it (#55578)."""
    _seed(homes, profile_source="desktop")
    captured = {}
    monkeypatch.setattr(
        async_delegation,
        "interrupt_for_session",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    server._finalize_session(
        _session(homes.worker_home, sid="tab9"), end_reason="tui_close"
    )

    assert captured["session_key"] == SESSION_ID
    assert captured["origin_ui_session_id"] == "tab9"


def test_gateway_owned_profile_session_keeps_gateway_delegations(homes, monkeypatch):
    """The other half of a live guard: closing a viewer tab on a gateway-owned
    profile session must not kill the gateway's own background work. Only this
    tab's own dispatches (origin id) are interrupted."""
    _seed(homes, profile_source="telegram")
    captured = {}
    monkeypatch.setattr(
        async_delegation,
        "interrupt_for_session",
        lambda **kwargs: captured.update(kwargs) or 0,
    )

    server._finalize_session(
        _session(homes.worker_home, sid="tab9"), end_reason="ws_orphan_reap"
    )

    assert captured["session_key"] == ""
    assert captured["origin_ui_session_id"] == "tab9"
