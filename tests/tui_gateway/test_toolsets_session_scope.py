"""Per-session toolset pinning on ``session.create`` (desktop backend).

Sibling of test_fast_session_scope.py. ``model`` / ``reasoning_effort`` /
``fast`` are all per-session overrides precisely so steering ONE chat cannot
mutate what the user's next chat gets; ``toolsets`` is the fourth, and arrives
the same way — as a ``session.create`` parameter, never a config write (that is
``tools.configure``'s job, and it is global on purpose).

History worth keeping: ``hermes desktop spawn --toolsets`` shipped in #298
carrying this value all the way to the renderer, which dropped it because there
was nothing on the backend to bind it to. #315 removed the flag rather than
leave one that lied. This is the binding that lets it come back.

Contract under test:

1. A valid ``toolsets`` list pins ``create_toolsets_override`` on the session
   and writes no config.
2. ``all`` / ``*`` resolves to None — *every* toolset — which is a VALUE, not
   "no pin". The sentinel is what keeps those apart.
3. A name that does not resolve fails the create outright. Handing back a
   session whose tools are not the ones asked for is the failure mode the pin
   exists to prevent.
4. ``_make_agent`` hands the pin to AIAgent as ``enabled_toolsets``, and only
   falls back to the process-wide resolver when there is no pin.
5. The pin dies at a ``/new`` boundary, like every other session-scoped
   override — which is also what stops it silently overruling the tools UI.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tui_gateway.server as server

# Real built-ins (see toolsets.get_all_toolsets); "git" deliberately is NOT a
# toolset, which makes it a truthful stand-in for a typo.
VALID = ["file", "terminal"]
NOT_A_TOOLSET = "git"


@pytest.fixture
def created_sessions(monkeypatch, tmp_path):
    """Call session.create for real, minus the background agent build.

    The deferred build fires a threading.Timer that would construct a real
    agent; every test here cares only about what session.create *recorded*.
    """
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (None, None))
    monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))

    made: list[str] = []

    def create(params: dict) -> dict:
        resp = server._methods["session.create"]("rid-1", params)
        if "result" in resp:
            made.append(resp["result"]["session_id"])
        return resp

    yield create

    for sid in made:
        server._sessions.pop(sid, None)


def _session_of(resp: dict) -> dict:
    return server._sessions[resp["result"]["session_id"]]


class TestSessionCreateAcceptsToolsets:
    def test_valid_names_pin_the_session_without_touching_config(
        self, created_sessions
    ) -> None:
        with patch.object(server, "_write_config_key") as write_key:
            resp = created_sessions({"cols": 80, "toolsets": VALID})

        assert "result" in resp
        assert _session_of(resp)["create_toolsets_override"] == VALID
        # tools.configure is the global writer; a spawn must never be one.
        write_key.assert_not_called()

    def test_names_are_stripped(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "toolsets": ["  file  ", "terminal"]})

        assert _session_of(resp)["create_toolsets_override"] == VALID

    def test_all_pins_every_toolset_as_none(self, created_sessions) -> None:
        """``all`` resolves to None, and None here means EVERY toolset.

        This is the trap the sentinel exists for: None must not be read back as
        "no pin given", or `all` would silently degrade to the process default.
        """
        resp = created_sessions({"cols": 80, "toolsets": ["all"]})

        session = _session_of(resp)
        assert session["create_toolsets_override"] is None
        assert session["create_toolsets_override"] is not server._NO_TOOLSET_OVERRIDE

    def test_omitted_leaves_no_pin_at_all(self, created_sessions) -> None:
        """The unchanged path every typed chat takes."""
        resp = created_sessions({"cols": 80})

        assert _session_of(resp)["create_toolsets_override"] is server._NO_TOOLSET_OVERRIDE


class TestSessionCreateRejectsBadToolsets:
    """A pin that cannot be honored must fail the create, not be dropped."""

    def test_unknown_name_is_refused_and_named(self, created_sessions) -> None:
        resp = created_sessions({"cols": 80, "toolsets": ["file", NOT_A_TOOLSET]})

        assert "error" in resp
        assert resp["error"]["code"] == 4026
        assert NOT_A_TOOLSET in resp["error"]["message"]

    def test_a_refused_create_leaves_no_session_behind(self, created_sessions) -> None:
        before = set(server._sessions)
        created_sessions({"cols": 80, "toolsets": [NOT_A_TOOLSET]})

        assert set(server._sessions) == before

    def test_a_refused_create_claims_no_active_session_slot(
        self, created_sessions, monkeypatch
    ) -> None:
        """Validate BEFORE claiming the slot, or a typo burns a lease.

        The concurrency lease is released by session.close, which never runs
        for a create that returned an error — so a slot claimed here would be
        gone until restart. Ordering is the whole guard.
        """
        claim = MagicMock(return_value=(None, None))
        monkeypatch.setattr(server, "_claim_active_session_slot", claim)

        created_sessions({"cols": 80, "toolsets": [NOT_A_TOOLSET]})

        claim.assert_not_called()

    @pytest.mark.parametrize(
        "value",
        [
            "file,terminal",  # the raw CLI string; the wire contract is a list
            {"file": True},
            ["file", 7],
            [["file"]],
        ],
    )
    def test_wrong_shape_is_refused(self, created_sessions, value) -> None:
        resp = created_sessions({"cols": 80, "toolsets": value})

        assert "error" in resp
        assert resp["error"]["code"] == 4025

    @pytest.mark.parametrize("value", [[], ["", "   "]])
    def test_naming_nothing_is_refused_rather_than_inherited(
        self, created_sessions, value
    ) -> None:
        """An empty pin is a typo, not "use the current toolsets"."""
        resp = created_sessions({"cols": 80, "toolsets": value})

        assert "error" in resp
        assert resp["error"]["code"] == 4025

    def test_null_is_treated_as_omitted(self, created_sessions) -> None:
        """JSON null is how a client says "no opinion", not an empty pin."""
        resp = created_sessions({"cols": 80, "toolsets": None})

        assert "result" in resp
        assert _session_of(resp)["create_toolsets_override"] is server._NO_TOOLSET_OVERRIDE


def _build_agent(**make_agent_kwargs):
    """Run _make_agent with the world stubbed, return the AIAgent kwargs."""
    with (
        patch("tui_gateway.server._load_cfg", return_value={"model": {"default": "m"}}),
        patch("tui_gateway.server._get_db", return_value=MagicMock()),
        patch("tui_gateway.server._load_tool_progress_mode", return_value="compact"),
        patch("tui_gateway.server._load_reasoning_config", return_value=None),
        patch("tui_gateway.server._load_service_tier", return_value=None),
        patch(
            "tui_gateway.server._load_enabled_toolsets", return_value=["process-wide"]
        ) as process_wide,
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value={
                "provider": "openai",
                "base_url": "https://example.invalid",
                "api_key": "sk-test",
                "api_mode": "chat_completions",
                "command": None,
                "args": None,
                "credential_pool": None,
            },
        ),
        patch("run_agent.AIAgent") as agent,
    ):
        server._make_agent("sid-1", "key-1", **make_agent_kwargs)

        return agent.call_args.kwargs, process_wide


class TestMakeAgentHonorsThePin:
    def test_pin_becomes_enabled_toolsets(self) -> None:
        kwargs, process_wide = _build_agent(toolsets_override=VALID)

        assert kwargs["enabled_toolsets"] == VALID
        # The pin REPLACES the process-wide resolution rather than merging with
        # it — a session asked for exactly these.
        process_wide.assert_not_called()

    def test_all_pin_reaches_the_agent_as_none(self) -> None:
        """None is forwarded as a value, not swallowed as "nothing given"."""
        kwargs, process_wide = _build_agent(toolsets_override=None)

        assert kwargs["enabled_toolsets"] is None
        process_wide.assert_not_called()

    def test_no_pin_falls_back_to_the_process_resolution(self) -> None:
        kwargs, process_wide = _build_agent()

        assert kwargs["enabled_toolsets"] == ["process-wide"]
        process_wide.assert_called_once()


def test_reset_clears_the_pin() -> None:
    """/new is a conversation boundary: session-scoped pins do not survive it.

    This is also what keeps the tools UI honest — `tools.configure` rebuilds
    through here, so toggling a toolset re-derives from config instead of being
    silently overruled by a spawn's pin the user can neither see nor clear.
    """
    session = {
        "session_key": "k1",
        "agent": None,
        "create_toolsets_override": VALID,
        "history_lock": __import__("threading").Lock(),
        "history": [],
        "history_version": 0,
    }

    with (
        patch.dict(server._sessions, {"s1": session}, clear=False),
        patch.object(server, "_make_agent", return_value=MagicMock()),
        patch.object(server, "_session_info", return_value={}),
        patch.object(server, "_emit"),
        patch.object(server, "_restart_slash_worker"),
        patch.object(server, "_load_show_reasoning", return_value=True),
        patch.object(server, "_load_tool_progress_mode", return_value="all"),
        patch.object(server, "_config_model_target", return_value=None),
    ):
        server._reset_session_agent("s1", session)

    assert "create_toolsets_override" not in session
