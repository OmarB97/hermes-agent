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

# A port nothing listens on. Provider validation may try to reach the endpoint;
# it must fail as "unreachable" (a warning) rather than resolve to a real
# service, and it must never make the test depend on the network.
_DEAD_ENDPOINT = "http://127.0.0.1:9/v1"


@pytest.fixture
def gateway_two_profiles(tmp_path, monkeypatch):
    """A real profile root: launch profile "launcher" + foreign profile "worker".

    Their ``config.yaml`` files name DIFFERENT default models AND different
    ``providers:`` entries, so any value that leaks from the launch profile is
    visible in the assertion rather than coincidentally equal (the reason the
    field-reported rows were ambiguous — both real profiles happened to name the
    same default). Provider names are the sharper signal: they are per-profile
    vocabulary, so a leaked one is not merely wrong, it is unresolvable.

    Yields ``(server, launcher_home, worker_home)`` with the gateway module
    posed as a backend launched under "launcher": ``HERMES_HOME`` is what the
    profile-name resolver reads, ``server._hermes_home`` the import-frozen path
    ``_load_cfg`` falls back to when no override is bound.
    """
    root = tmp_path / "hermes-root"
    launcher = root / "profiles" / "launcher"
    worker = root / "profiles" / "worker"
    for home, model, provider in (
        (launcher, "launcher/model-A", "launch-router"),
        (worker, "worker/model-B", "worker-local"),
    ):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(
            f"model:\n"
            f"  provider: {provider}\n"
            f'  default: "{model}"\n'
            f'  base_url: "{_DEAD_ENDPOINT}"\n'
            f"providers:\n"
            f"  {provider}:\n"
            f'    api: "{_DEAD_ENDPOINT}"\n'
            f'    api_key: "sk-test"\n'
            # Keep the picker offline: no catalog fetch, no pricing round-trip.
            f"model_catalog:\n  enabled: false\n",
            encoding="utf-8",
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


class TestComposerProviderPickUsesTargetProfile:
    """The composer's provider pick — the catalog it picks FROM, and applying
    the pick — must resolve under the session's profile.

    ``providers:`` / ``custom_providers:`` are per-profile vocabulary, so a name
    resolved against the launch profile is not merely a different choice: it is
    a name the target profile cannot resolve, and the turn dies in agent init on
    "Unknown provider '<name>'" (the husk-leaving failure #321 fixed for
    ``hermes desktop spawn --provider``; this is the desktop composer's path to
    the same dead end).
    """

    def _worker_session(self, server, worker):
        session = {
            "session_key": "s1",
            "profile_home": str(worker),
            "agent": None,
            "cwd": str(worker),
        }
        server._sessions["S"] = session
        return session

    def test_model_options_offers_the_sessions_own_providers(
        self, gateway_two_profiles
    ):
        server, _launcher, worker = gateway_two_profiles
        self._worker_session(server, worker)
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "model.options",
                    "params": {"session_id": "S", "explicit_only": True},
                }
            )
        finally:
            server._sessions.pop("S", None)

        result = resp["result"]
        slugs = {p.get("slug", "") for p in (result.get("providers") or [])}
        # Asserted as membership, not an exact list: the payload also carries
        # whatever the host has credentials for, which is not this test's
        # business. What matters is which profile's vocabulary is present.
        assert "worker-local" in slugs
        assert "launch-router" not in slugs, "launch profile's provider leaked"
        assert result.get("provider") == "worker-local"
        assert result.get("model") == "worker/model-B"

    def test_applying_the_pick_resolves_the_sessions_own_provider(
        self, gateway_two_profiles
    ):
        """The mirror failure, and the sharper one: before the fix a session
        could not even switch to its OWN profile's provider — the launch
        profile's config was consulted, so the valid name was rejected."""
        server, _launcher, worker = gateway_two_profiles
        self._worker_session(server, worker)
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "config.set",
                    "params": {
                        "session_id": "S",
                        "key": "model",
                        "value": "worker/model-B --provider worker-local --session",
                    },
                }
            )
        finally:
            server._sessions.pop("S", None)

        assert "error" not in resp, resp.get("error")
        assert resp["result"]["value"] == "worker/model-B"
        assert resp["result"]["scope"] == "session"

    def test_applying_a_foreign_providers_pick_is_still_rejected(
        self, gateway_two_profiles
    ):
        """Binding the target profile is not the same as accepting anything: a
        name only the LAUNCH profile defines must still fail, and now it fails
        at the switch (reported to the client) instead of silently at turn
        start."""
        server, _launcher, worker = gateway_two_profiles
        self._worker_session(server, worker)
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "config.set",
                    "params": {
                        "session_id": "S",
                        "key": "model",
                        "value": "launcher/model-A --provider launch-router --session",
                    },
                }
            )
        finally:
            server._sessions.pop("S", None)

        assert "error" in resp
        assert "launch-router" in resp["error"]["message"]

    def test_launch_profile_session_is_unaffected(self, gateway_two_profiles):
        """Control: a session with no profile_home takes the unbound path it
        always did."""
        server, _launcher, _worker = gateway_two_profiles
        server._sessions["S"] = {"session_key": "s1", "agent": None}
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "model.options",
                    "params": {"session_id": "S", "explicit_only": True},
                }
            )
        finally:
            server._sessions.pop("S", None)

        slugs = {p.get("slug", "") for p in (resp["result"].get("providers") or [])}
        assert "launch-router" in slugs
        assert "worker-local" not in slugs

    def test_save_key_authenticates_the_sessions_own_profile(
        self, gateway_two_profiles, monkeypatch
    ):
        """The picker's "connect" action writes to the profile whose providers
        it is showing. ``.env`` is per-profile, so saving to the launcher would
        leave the row the user just connected still unauthenticated."""
        server, launcher, worker = gateway_two_profiles
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        self._worker_session(server, worker)
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "model.save_key",
                    "params": {
                        "session_id": "S",
                        "slug": "deepseek",
                        "api_key": "sk-worker-test-key",
                    },
                }
            )
        finally:
            server._sessions.pop("S", None)

        assert "error" not in resp, resp.get("error")
        assert "sk-worker-test-key" in (worker / ".env").read_text(encoding="utf-8")
        assert not (launcher / ".env").exists(), "key landed in the launch profile"

    def test_profile_param_scopes_the_picker_with_no_session(
        self, gateway_two_profiles
    ):
        """The composer's other opening: the picker for a chat that does not
        exist yet. There is no session to derive a profile from, so the client
        names it — the same scoping the REST twin /api/model/options has always
        had, which is why the two surfaces used to disagree."""
        server, _launcher, _worker = gateway_two_profiles

        resp = server.handle_request(
            {
                "id": "1",
                "method": "model.options",
                "params": {"profile": "worker", "explicit_only": True},
            }
        )

        result = resp["result"]
        slugs = {p.get("slug", "") for p in (result.get("providers") or [])}
        assert "worker-local" in slugs
        assert "launch-router" not in slugs
        assert result.get("model") == "worker/model-B"

    def test_live_sessions_own_home_beats_the_profile_param(
        self, gateway_two_profiles
    ):
        """When both arrive and disagree, the session wins: its home is where
        its turn will actually run, so that is the vocabulary that has to
        resolve."""
        server, _launcher, worker = gateway_two_profiles
        self._worker_session(server, worker)
        try:
            resp = server.handle_request(
                {
                    "id": "1",
                    "method": "model.options",
                    "params": {
                        "session_id": "S",
                        "profile": "launcher",
                        "explicit_only": True,
                    },
                }
            )
        finally:
            server._sessions.pop("S", None)

        slugs = {p.get("slug", "") for p in (resp["result"].get("providers") or [])}
        assert "worker-local" in slugs
        assert "launch-router" not in slugs
