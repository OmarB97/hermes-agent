"""Profile-isolation regression tests for single-process multi-profile runtimes.

In runtimes that serve every profile from one OS process (the desktop
``tui_gateway``), the profile boundary is the context-local
``_HERMES_HOME_OVERRIDE`` ContextVar, not the process environment.  State that
escapes the request call stack — import-time-frozen path constants, direct
``os.environ`` reads, or worker threads that don't inherit the request context —
silently reverts to the launch/default profile and leaks one profile's data
into another.

These tests drive each previously-leaking site under override A then override B
with real temp HERMES_HOME directories (no mocks) and assert the *active*
profile's path is used.  They are the productionized form of the manual smoke
probes used to confirm the bug class.
"""

import threading
from pathlib import Path

import pytest

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)


@pytest.fixture
def two_profiles(tmp_path):
    """Two distinct profile HERMES_HOME dirs with the dir skeleton created."""
    prof_a = tmp_path / "profA"
    prof_b = tmp_path / "profB"
    for p in (prof_a, prof_b):
        (p / "skills").mkdir(parents=True, exist_ok=True)
        (p / "state").mkdir(parents=True, exist_ok=True)
        (p / "cache").mkdir(parents=True, exist_ok=True)
    return prof_a, prof_b


def _under_override(home: Path, fn):
    """Run ``fn`` with the profile override set to ``home`` and reset after."""
    token = set_hermes_home_override(str(home))
    try:
        return fn()
    finally:
        reset_hermes_home_override(token)


# ---------------------------------------------------------------------------
# M1 — import-time path globals / direct os.environ reads
# ---------------------------------------------------------------------------

class TestSkillsHubPathResolution:
    """tools/skills_hub.py path constants must reflect the active profile."""

    def test_skills_dir_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.skills_hub as sh

        # Importing/touching under A must NOT pin the path for B.
        a_seen = _under_override(prof_a, lambda: Path(sh.SKILLS_DIR))
        b_seen = _under_override(prof_b, lambda: Path(sh.SKILLS_DIR))

        assert a_seen == prof_a / "skills"
        assert b_seen == prof_b / "skills"
        assert a_seen != b_seen

    def test_hub_derived_paths_follow_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import tools.skills_hub as sh

        b_lock = _under_override(prof_b, lambda: Path(sh.LOCK_FILE))
        b_audit = _under_override(prof_b, lambda: Path(sh.AUDIT_LOG))
        b_index = _under_override(prof_b, lambda: Path(sh.INDEX_CACHE_DIR))

        assert b_lock == prof_b / "skills" / ".hub" / "lock.json"
        assert b_audit == prof_b / "skills" / ".hub" / "audit.log"
        assert b_index == prof_b / "skills" / ".hub" / "index-cache"

    def test_lockfile_default_arg_resolves_active_profile(self, two_profiles):
        prof_a, prof_b = two_profiles
        from tools.skills_hub import HubLockFile, TapsManager

        lock_b = _under_override(prof_b, lambda: HubLockFile())
        taps_b = _under_override(prof_b, lambda: TapsManager())

        assert lock_b.path == prof_b / "skills" / ".hub" / "lock.json"
        assert taps_b.path == prof_b / "skills" / ".hub" / "taps.json"


class TestGatewayCacheDirResolution:
    """gateway/platforms/base.py cache getters must follow the active profile."""

    def test_image_cache_dir_follows_override(self, two_profiles):
        prof_a, prof_b = two_profiles
        import gateway.platforms.base as gb

        a_seen = _under_override(prof_a, lambda: gb.get_image_cache_dir())
        b_seen = _under_override(prof_b, lambda: gb.get_image_cache_dir())

        assert str(a_seen).startswith(str(prof_a))
        assert str(b_seen).startswith(str(prof_b))
        assert a_seen != b_seen

    def test_all_cache_getters_follow_override(self, two_profiles):
        _prof_a, prof_b = two_profiles
        import gateway.platforms.base as gb

        getters = (
            gb.get_image_cache_dir,
            gb.get_audio_cache_dir,
            gb.get_video_cache_dir,
            gb.get_document_cache_dir,
        )
        for getter in getters:
            seen = _under_override(prof_b, getter)
            assert str(seen).startswith(str(prof_b)), f"{getter.__name__} leaked: {seen}"

    def test_monkeypatched_constant_still_wins(self, two_profiles, monkeypatch, tmp_path):
        """The existing test seam (monkeypatch the module constant) is preserved."""
        _prof_a, _prof_b = two_profiles
        import gateway.platforms.base as gb

        forced = tmp_path / "forced_img"
        monkeypatch.setattr("gateway.platforms.base.IMAGE_CACHE_DIR", forced)
        # Even with an active override, an explicit monkeypatch takes precedence.
        seen = _under_override(_prof_b, lambda: gb.get_image_cache_dir())
        assert seen == forced


class TestRichSentStorePathResolution:
    """gateway/rich_sent_store.py must honor the override, not read os.environ."""

    def test_store_path_follows_override(self, two_profiles, monkeypatch):
        prof_a, prof_b = two_profiles
        # Ensure no ambient HERMES_HOME env masks the test.
        monkeypatch.delenv("HERMES_HOME", raising=False)
        import gateway.rich_sent_store as rss

        b_seen = _under_override(prof_b, lambda: rss._store_path())
        assert b_seen.startswith(str(prof_b))
        assert b_seen.endswith("state/rich_sent_index.json")


# ---------------------------------------------------------------------------
# M2 — thread / executor context propagation
# ---------------------------------------------------------------------------

class TestThreadContextPropagation:
    """Worker threads must inherit the spawning turn's profile override."""

    def test_raw_thread_loses_override(self, two_profiles):
        """Document the underlying hazard: a bare thread does NOT inherit it."""
        _prof_a, prof_b = two_profiles
        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        def run():
            t = threading.Thread(target=worker)
            t.start()
            t.join()

        _under_override(prof_b, run)
        # A bare thread falls back to the process default — this is WHY the fix
        # primitive is needed.  (Asserted as the hazard, not the desired state.)
        assert seen["home"] != str(prof_b)

    def test_propagate_primitive_preserves_override(self, two_profiles):
        _prof_a, prof_b = two_profiles
        from tools.thread_context import propagate_context_to_thread

        seen = {}

        def worker():
            seen["home"] = str(get_hermes_home())

        def run():
            t = threading.Thread(target=propagate_context_to_thread(worker))
            t.start()
            t.join()

        _under_override(prof_b, run)
        assert seen["home"] == str(prof_b)

    def test_run_async_worker_preserves_override(self, two_profiles):
        """model_tools._run_async's worker-thread branch must keep the override.

        This is the generic sync->async bridge for every async tool; if it
        leaks, every async tool that resolves get_hermes_home() leaks.
        """
        import asyncio

        _prof_a, prof_b = two_profiles
        import model_tools

        async def reads_home():
            return str(get_hermes_home())

        async def driver():
            # Inside a running loop, _run_async spawns a worker thread + loop.
            return model_tools._run_async(reads_home())

        seen = _under_override(prof_b, lambda: asyncio.run(driver()))
        assert seen == str(prof_b)


# ---------------------------------------------------------------------------
# M3 — RPC handlers that resolve HERMES_HOME before the per-turn binding
# ---------------------------------------------------------------------------

@pytest.fixture
def gateway_two_profiles(tmp_path, monkeypatch):
    """A real profile root: launch profile "launcher" + foreign profile "worker".

    Their ``config.yaml`` files name DIFFERENT default models, so any value that
    leaks from the launch profile is visible in the assertion rather than
    coincidentally equal (the reason the field-reported rows were ambiguous —
    both real profiles happened to name the same default).

    Yields ``(server, launcher_home, worker_home)`` with the gateway module
    posed as a backend launched under "launcher": ``HERMES_HOME`` is what the
    profile-name resolver reads, ``server._hermes_home`` the import-frozen path
    ``_load_cfg`` falls back to when no override is bound.
    """
    root = tmp_path / "hermes-root"
    launcher = root / "profiles" / "launcher"
    worker = root / "profiles" / "worker"
    for home, model in ((launcher, "launcher/model-A"), (worker, "worker/model-B")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f'model:\n  default: "{model}"\n', encoding="utf-8"
        )

    monkeypatch.setenv("HERMES_HOME", str(launcher))
    # _resolve_model checks these first; the host's environment must not decide
    # the answer for a test about which config.yaml gets read.
    monkeypatch.delenv("HERMES_MODEL", raising=False)
    monkeypatch.delenv("HERMES_INFERENCE_MODEL", raising=False)

    from tui_gateway import server

    monkeypatch.setattr(server, "_hermes_home", launcher)
    # _load_cfg caches on the resolved path; start from cold and restore after.
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_cfg_mtime", None)
    return server, launcher, worker


def _row(home: Path, session_key: str) -> dict | None:
    from hermes_state import SessionDB

    db = SessionDB(db_path=home / "state.db")
    try:
        return db.get_session(session_key)
    finally:
        db.close()


class TestSessionRowIdentityUsesOwnProfile:
    """``_ensure_session_db_row`` runs on the RPC thread, before the turn thread
    binds HERMES_HOME — so it must bind the session's home itself.

    This is a permanent corruption, not a transient one: ``_insert_session_row``
    upserts under ``model = COALESCE(sessions.model, excluded.model)``, so the
    agent's own later (correct) lazy-create cannot repair a wrong first write.
    """

    def test_row_model_falls_back_to_the_sessions_own_profile_default(
        self, gateway_two_profiles
    ):
        server, _launcher, worker = gateway_two_profiles

        server._ensure_session_db_row(
            {
                "session_key": "s-worker",
                "profile_home": str(worker),
                "model_override": None,
            }
        )

        row = _row(worker, "s-worker")
        assert row is not None, "row must land in the worker profile's state.db"
        assert row["model"] == "worker/model-B"
        assert row["model"] != "launcher/model-A", "launch profile's default leaked"

    def test_row_is_attributed_to_its_profile(self, gateway_two_profiles):
        """A session whose first turn never runs keeps whatever this write left,
        so the row has to name its profile up front rather than rely on the
        agent's backfill."""
        server, _launcher, worker = gateway_two_profiles

        server._ensure_session_db_row(
            {"session_key": "s-worker", "profile_home": str(worker)}
        )

        assert _row(worker, "s-worker")["profile_name"] == "worker"

    def test_launch_profile_session_keeps_its_own_default(self, gateway_two_profiles):
        """Control: binding the session's home must not mean "always foreign"."""
        server, launcher, _worker = gateway_two_profiles

        server._ensure_session_db_row(
            {"session_key": "s-launch", "profile_home": str(launcher)}
        )

        row = _row(launcher, "s-launch")
        assert row["model"] == "launcher/model-A"
        assert row["profile_name"] == "launcher"

    def test_explicit_composer_pick_still_wins(self, gateway_two_profiles):
        """The anti-race intent is unchanged: an explicit pick is never
        overwritten by any profile's default."""
        server, _launcher, worker = gateway_two_profiles

        server._ensure_session_db_row(
            {
                "session_key": "s-picked",
                "profile_home": str(worker),
                "model_override": {"model": "picked/model-C", "provider": "openrouter"},
            }
        )

        row = _row(worker, "s-picked")
        assert row["model"] == "picked/model-C"


class TestSessionProfileNameDoesNotFlip:
    """``session.create``/lazy-resume answered with the LAUNCH profile's name
    while the deferred build — the one caller that DOES bind the home — answered
    with the session's, so the client watched ``profile_name`` change under it.
    """

    def test_session_create_reports_the_requested_profile(
        self, gateway_two_profiles, monkeypatch
    ):
        server, _launcher, _worker = gateway_two_profiles
        # Building the agent is a separate (network-touching) concern; what is
        # under test is the response the client paints from, sent before it.
        monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
        monkeypatch.setattr(
            server, "_schedule_session_cap_enforcement", lambda *a, **k: None
        )

        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.create",
                "params": {"cols": 80, "profile": "worker"},
            }
        )
        sid = resp["result"]["session_id"]
        try:
            info = resp["result"]["info"]
            assert info["profile_name"] == "worker"
            assert info["model"] == "worker/model-B"
            # ...and the deferred build's session.info agrees, so nothing flips.
            assert (
                server._session_info(None, server._sessions[sid])["profile_name"]
                == "worker"
            )
        finally:
            server._sessions.pop(sid, None)

    def test_lazy_resume_info_reports_the_sessions_profile(self, gateway_two_profiles):
        server, _launcher, worker = gateway_two_profiles

        info = server._lazy_resume_info(str(worker), profile_home=str(worker))

        assert info["profile_name"] == "worker"
        assert info["model"] == "worker/model-B"
