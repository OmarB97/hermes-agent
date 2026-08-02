"""Every live-session read/write in ``tui_gateway/server.py`` must use the db
that owns that session's row.

``_get_db()`` is a cached module global bound to ``hermes_state.DEFAULT_DB_PATH``
at *import* time — the LAUNCH profile's ``state.db``. A session created with
``profile: "<other>"`` keeps its rows in ``<root>/profiles/<other>/state.db``,
reached through the profile-aware ``_session_db(session)`` helper. Each call site
below held a live session dict and read the launch handle anyway.

Real ``SessionDB`` files under a temp home with two profiles, no db mocks: a
mocked handle hides this defect completely (both dbs answer identically), and
each of these failures is specifically "the row is not in the db you asked".
Both dbs are seeded under the SAME session id wherever the write target would
otherwise be ambiguous.
"""

import contextlib
import threading
import types

import pytest

from hermes_state import SessionDB
from tui_gateway import server

KEY = "sess-profile-1"


@pytest.fixture
def homes(tmp_path, monkeypatch):
    """A real HERMES_HOME root: launch ``state.db`` + one extra profile's."""
    root = tmp_path / "hermes"
    worker_home = root / "profiles" / "worker"
    worker_home.mkdir(parents=True)
    launch = SessionDB(db_path=root / "state.db")
    worker = SessionDB(db_path=worker_home / "state.db")
    # Never let a site under test fall through to the host's real state.db.
    monkeypatch.setattr(server, "_get_db", lambda: launch)
    try:
        yield types.SimpleNamespace(
            launch=launch, worker=worker, worker_home=worker_home, root=root
        )
    finally:
        for db in (launch, worker):
            with contextlib.suppress(Exception):
                db.close()


def _session(profile_home=None, *, key=KEY, agent=None, history=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(session_id=key),
        "session_key": key,
        "history": list(history or []),
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        "profile_home": str(profile_home) if profile_home else None,
        **extra,
    }


@contextlib.contextmanager
def _live(sid, session):
    server._sessions[sid] = session
    try:
        yield session
    finally:
        server._sessions.pop(sid, None)


# ── /history and /context (live slash commands) ───────────────────────────────


def test_live_history_reads_the_profile_transcript(homes):
    """``/history`` overwrote the live window with an EMPTY launch-db read."""
    homes.worker.create_session(KEY, source="desktop")
    homes.worker.append_message(session_id=KEY, role="user", content="hello there")
    homes.worker.append_message(session_id=KEY, role="assistant", content="hi back")

    out = server._format_live_history_output(
        _session(homes.worker_home, history=[{"role": "user", "content": "hello there"}])
    )

    assert "No conversation history yet." not in out
    assert "hello there" in out
    assert "hi back" in out


def test_live_context_counts_the_profile_transcript(homes):
    """``/context`` silently degraded to the in-memory window, so a compressed
    session under-reported every message that only exists in the db."""
    homes.worker.create_session(KEY, source="desktop")
    for i in range(4):
        homes.worker.append_message(session_id=KEY, role="user", content=f"m{i}")

    out = server._format_live_context_output(
        _session(homes.worker_home, history=[{"role": "user", "content": "m3"}])
    )

    assert "Conversation: 4 messages" in out


def test_live_history_still_reads_the_launch_db_without_a_profile(homes):
    """Control: a session with no profile binding is unchanged."""
    homes.launch.create_session(KEY, source="tui")
    homes.launch.append_message(session_id=KEY, role="user", content="launch msg")

    out = server._format_live_history_output(_session(None))

    assert "launch msg" in out


# ── titles ────────────────────────────────────────────────────────────────────


def test_live_title_reads_the_profile_row(homes):
    """``session.info``/the sidebar reported "" for every profile session."""
    homes.worker.create_session(KEY, source="desktop")
    homes.worker.set_session_title(KEY, "Profile Title")

    assert server._session_live_title(_session(homes.worker_home), KEY) == "Profile Title"


def test_session_title_read_returns_the_persisted_profile_title(homes):
    """The set path already reached the profile db (via its missing-row
    recovery); the read path had no such fallback and answered ""."""
    homes.worker.create_session(KEY, source="desktop")
    session = _session(homes.worker_home)

    with _live("tab1", session):
        wrote = server._methods["session.title"]("r1", {"session_id": "tab1", "title": "Renamed"})
        read = server._methods["session.title"]("r2", {"session_id": "tab1"})

    assert wrote["result"] == {"pending": False, "title": "Renamed"}
    assert homes.worker.get_session_title(KEY) == "Renamed"
    assert read["result"]["title"] == "Renamed"
    # The launch profile never gained a row for this session.
    assert homes.launch.get_session(KEY) is None


def test_session_title_on_a_fresh_profile_draft_creates_the_row_there(homes, monkeypatch):
    """#47987's create-the-row-up-front path, under a profile. The row is
    deferred to the first prompt, so /title must land it in the profile db and
    read it straight back through the same handle."""
    monkeypatch.setattr(server, "_resolve_model", lambda *a, **k: "test/model")
    session = _session(homes.worker_home, source="desktop")

    with _live("tab1", session):
        wrote = server._methods["session.title"]("r1", {"session_id": "tab1", "title": "Draft"})
        read = server._methods["session.title"]("r2", {"session_id": "tab1"})

    assert wrote["result"] == {"pending": False, "title": "Draft"}
    assert read["result"]["title"] == "Draft"
    assert homes.worker.get_session_title(KEY) == "Draft"
    assert homes.launch.get_session(KEY) is None


def test_session_title_set_still_writes_the_launch_db_without_a_profile(homes):
    """Control: no profile binding → the shared launch handle, as before."""
    homes.launch.create_session(KEY, source="tui")
    session = _session(None)

    with _live("tab1", session):
        resp = server._methods["session.title"]("r1", {"session_id": "tab1", "title": "Plain"})

    assert resp["result"] == {"pending": False, "title": "Plain"}
    assert homes.launch.get_session_title(KEY) == "Plain"


# ── /status ───────────────────────────────────────────────────────────────────


def test_session_status_reads_the_profile_row(homes):
    """No row found meant no Title line and Created/Last Activity = "now"."""
    homes.worker.create_session(KEY, source="desktop")
    homes.worker.set_session_title(KEY, "Profile Title")
    started = homes.worker.get_session(KEY)["started_at"]

    with _live("tab1", _session(homes.worker_home)):
        out = server._methods["session.status"]("r1", {"session_id": "tab1"})["result"]["output"]

    assert "Title: Profile Title" in out
    from datetime import datetime

    assert f"Created: {datetime.fromtimestamp(float(started)).strftime('%Y-%m-%d %H:%M')}" in out


# ── notification ownership (compression lineage) ──────────────────────────────


def _seed_compression_chain(db, parent="sess-parent", child="sess-child"):
    db.create_session(parent, source="desktop")
    db.append_message(session_id=parent, role="user", content="a")
    db.end_session(parent, end_reason="compression")
    db.create_session(child, source="desktop", parent_session_id=parent)
    db.append_message(session_id=child, role="user", content="b")
    return parent, child


def test_owns_notification_event_follows_the_profile_compression_chain(homes):
    """The fail-closed gate of #55578 dropped a profile session's own
    delegation completion: the launch db can't map the pre-compression key to
    the continuation tip, so ownership came back False and the poller logged
    "Dropping unowned async_delegation notification"."""
    parent, child = _seed_compression_chain(homes.worker)
    assert homes.worker.resolve_resume_session_id(parent) == child
    assert homes.launch.resolve_resume_session_id(parent) == parent  # the wrong answer

    session = _session(homes.worker_home, key=child)
    evt = {"type": "async_delegation", "session_key": parent}

    assert server._session_owns_notification_event("tab1", session, evt) is True


def test_owns_notification_event_prefers_the_agents_own_handle(homes, tmp_path):
    """A live session's agent already holds a profile-scoped handle; the poll
    loop uses it rather than opening a read-write db per event.

    ``profile_home`` points at an empty profile whose ``state.db`` knows nothing
    about the chain, so only the agent's handle can produce the right answer.
    """
    parent, child = _seed_compression_chain(homes.worker)
    empty_home = tmp_path / "hermes" / "profiles" / "empty"
    empty_home.mkdir(parents=True)
    agent = types.SimpleNamespace(session_id=child, _session_db=homes.worker)
    session = _session(empty_home, key=child, agent=agent)

    assert server._session_owns_notification_event("tab1", session, {
        "type": "async_delegation", "session_key": parent,
    }) is True


def test_belongs_elsewhere_sees_the_profile_continuation(homes):
    """Mirror check: an event captured before compression must be recognised as
    the live continuation's, not offered to the stale parent tab."""
    parent, child = _seed_compression_chain(homes.worker)
    stale = _session(homes.worker_home, key=parent)
    live_child = _session(homes.worker_home, key=child)
    evt = {"type": "async_delegation", "session_key": parent}

    with _live("tab-parent", stale), _live("tab-child", live_child):
        assert server._notification_event_belongs_elsewhere("tab-parent", stale, evt) is True


def test_owns_notification_event_unchanged_without_a_profile(homes):
    """Control: launch-profile lineage still resolves through the shared db."""
    parent, child = _seed_compression_chain(homes.launch)
    session = _session(None, key=child)

    assert server._session_owns_notification_event("tab1", session, {
        "type": "async_delegation", "session_key": parent,
    }) is True


# ── /branch ───────────────────────────────────────────────────────────────────


@pytest.fixture
def stub_branch_build(monkeypatch):
    """Stand in for the child's agent build, recording what it was handed."""
    captured = {}

    def fake_make_agent(sid, key, **kwargs):
        captured["agent_db"] = kwargs.get("session_db")
        return types.SimpleNamespace(model="test/model", session_id=key)

    def fake_init_session(sid, key, agent, history, **kwargs):
        captured["init_db"] = kwargs.get("session_db")
        server._sessions[sid] = _session(None, key=key, agent=agent, history=history)

    monkeypatch.setattr(server, "_make_agent", fake_make_agent)
    monkeypatch.setattr(server, "_init_session", fake_init_session)
    return captured


def test_branch_of_a_profile_session_writes_the_child_to_the_profile_db(
    homes, stub_branch_build
):
    """The branch row is an FK CHILD of the parent's row. Created against the
    launch db (which has no parent row) the INSERT failed outright and the
    handler returned "branch failed: FOREIGN KEY constraint failed"."""
    homes.worker.create_session(KEY, source="desktop")
    homes.worker.set_session_title(KEY, "Parent")
    history = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    parent = _session(homes.worker_home, history=history)

    with _live("tab1", parent):
        resp = server._methods["session.branch"]("r1", {"session_id": "tab1"})
        assert "result" in resp, f"got error: {resp.get('error')}"
        new_sid = resp["result"]["session_id"]
        child_key = server._sessions[new_sid]["session_key"]
        # The child keeps the parent's profile: its rows are there, so every
        # later turn has to resolve its home and its db there too.
        assert server._sessions[new_sid]["profile_home"] == str(homes.worker_home)
        server._sessions.pop(new_sid, None)

    row = homes.worker.get_session(child_key)
    assert row is not None
    assert row["parent_session_id"] == KEY
    assert [m["content"] for m in homes.worker.get_messages_as_conversation(child_key)] == [
        "hello",
        "hi",
    ]
    assert homes.worker.get_session_title(child_key) == resp["result"]["title"]
    # Nothing was written to the launch profile.
    assert homes.launch.get_session(child_key) is None
    # The child agent persists into the same db that holds its row.
    assert str(stub_branch_build["agent_db"].db_path) == str(homes.worker_home / "state.db")
    assert stub_branch_build["init_db"] is stub_branch_build["agent_db"]


def test_branch_never_touches_the_launch_handle_for_a_profile_session(
    homes, stub_branch_build, monkeypatch
):
    """Mock-shaped contract: with a profile bound, the launch db is not a
    participant at all — no read, no write, no handle handed to the agent."""
    homes.worker.create_session(KEY, source="desktop")
    captured = {}

    class LaunchDB:
        def __getattr__(self, name):
            captured.setdefault("launch_calls", []).append(name)
            raise AssertionError(f"launch db used for a profile session: {name}")

    monkeypatch.setattr(server, "_get_db", lambda: LaunchDB())
    parent = _session(homes.worker_home, history=[{"role": "user", "content": "x"}])

    with _live("tab1", parent):
        resp = server._methods["session.branch"]("r1", {"session_id": "tab1"})
        server._sessions.pop(resp["result"]["session_id"], None)

    assert "result" in resp, f"got error: {resp.get('error')}"
    assert "launch_calls" not in captured


def test_branch_of_a_launch_session_still_uses_the_launch_db(homes, stub_branch_build):
    """No profile binding → the shared launch db, exactly as before. The handle
    is now passed to the child agent explicitly; it is the same object
    ``_make_agent`` defaulted to on its own."""
    homes.launch.create_session(KEY, source="tui")
    parent = _session(None, history=[{"role": "user", "content": "hello"}])

    with _live("tab1", parent):
        resp = server._methods["session.branch"]("r1", {"session_id": "tab1"})
        assert "result" in resp, f"got error: {resp.get('error')}"
        child_key = server._sessions[resp["result"]["session_id"]]["session_key"]
        assert server._sessions[resp["result"]["session_id"]].get("profile_home") is None
        server._sessions.pop(resp["result"]["session_id"], None)

    assert homes.launch.get_session(child_key)["parent_session_id"] == KEY
    assert stub_branch_build["agent_db"] is homes.launch


# ── prompt.submit: history truncation + title persistence ─────────────────────


class _ImmediateThread:
    """Run the turn inline so the whole submit path is exercised."""

    def __init__(self, target=None, daemon=None, name=None, args=(), kwargs=None):
        self._target, self._args, self._kwargs = target, args, kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)

    def is_alive(self):
        return False


@pytest.fixture
def inline_turn(monkeypatch):
    class _Agent:
        model = "test/model"
        provider = "test"

        def __init__(self, key):
            self.session_id = key

        def run_conversation(self, prompt, conversation_history=None, stream_callback=None):
            return {
                "final_response": "edited reply",
                "messages": [
                    *(conversation_history or []),
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "edited reply"},
                ],
            }

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_get_usage", lambda _a: {})
    monkeypatch.setattr(server, "render_message", lambda _t, _c: "")
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    return _Agent


def test_prompt_submit_truncation_rewrites_the_profile_transcript(homes, inline_turn):
    """Editing a user message rewrites the stored transcript. Against the launch
    db that raised "FOREIGN KEY constraint failed" — which also skipped the
    turn-outcome deactivation — and left the profile db holding the FULL
    pre-edit history, so resuming brought the edited-away turns back."""
    homes.worker.create_session(KEY, source="desktop")
    original = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "second reply"},
    ]
    for msg in original:
        homes.worker.append_message(session_id=KEY, role=msg["role"], content=msg["content"])

    session = _session(homes.worker_home, agent=inline_turn(KEY), history=list(original))
    with _live("sid", session):
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {
                "session_id": "sid",
                "text": "edited second",
                "truncate_before_user_ordinal": 1,
            },
        })
    assert resp.get("result"), f"got error: {resp.get('error')}"

    stored = [m["content"] for m in homes.worker.get_messages_as_conversation(KEY)]
    assert stored[:2] == ["first", "first reply"]
    assert "second" not in stored
    assert "second reply" not in stored


def test_pending_title_is_applied_to_the_profile_db(homes, inline_turn):
    """A title given at session.create is applied after the first turn. The
    launch UPDATE matched no row, returned False and left ``pending_title``
    set — so it was retried every turn and never persisted."""
    session = _session(
        homes.worker_home,
        agent=inline_turn(KEY),
        pending_title="Named At Create",
    )
    with _live("sid", session):
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        })
        assert resp.get("result"), f"got error: {resp.get('error')}"
        assert session["pending_title"] is None

    assert homes.worker.get_session_title(KEY) == "Named At Create"
    assert homes.launch.get_session(KEY) is None


def test_auto_title_is_handed_the_agents_own_db(homes, inline_turn, monkeypatch):
    """The auto-titler writes from its own daemon thread, so it needs a handle
    that outlives the turn — the agent's, which is profile-scoped. Given the
    launch handle its UPDATE found no row and profile sessions stayed
    permanently untitled."""
    captured = {}
    monkeypatch.setattr(
        "agent.title_generator.maybe_auto_title",
        lambda session_db, *a, **k: captured.update(db=session_db),
    )
    agent = inline_turn(KEY)
    agent._session_db = homes.worker

    with _live("sid", _session(homes.worker_home, agent=agent)):
        resp = server.handle_request({
            "id": "1",
            "method": "prompt.submit",
            "params": {"session_id": "sid", "text": "hello"},
        })
    assert resp.get("result"), f"got error: {resp.get('error')}"
    assert captured["db"] is homes.worker


# ── background / preview subagents ────────────────────────────────────────────


def test_background_agent_inherits_the_parent_agents_db(homes):
    """Every other field in these kwargs is inherited from the parent agent.
    The launch handle wrote a profile session's background transcript into the
    LAUNCHER's state.db — wrong profile's session list, missing from its own."""
    agent = types.SimpleNamespace(session_id=KEY, _session_db=homes.worker, model="m")

    kwargs = server._background_agent_kwargs(agent, "bg_123")

    assert kwargs["session_db"] is homes.worker


def test_background_agent_falls_back_to_the_launch_db(homes):
    """Control: an agent with no db of its own still gets the shared handle."""
    agent = types.SimpleNamespace(session_id=KEY, model="m")

    assert server._background_agent_kwargs(agent, "bg_123")["session_db"] is homes.launch
