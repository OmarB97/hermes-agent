"""Per-session declared command allowlists on ``session.create``.

Sibling of test_toolsets_session_scope.py, and it arrives the same way: as a
``session.create`` parameter, never a config write. The reason is sharper here
than for toolsets — this parameter WIDENS what a session may do without a
human, so leaking it into config would widen the user's next chat too.

What it buys: ``tools/approval.py::_smart_approve`` fails CLOSED, so any error
reaching the approval classifier means "ask a human". In a `--delegated` run
nobody is there, and each flagged command burns ``approvals.timeout`` before
being blocked. A session that declared an allowlist degrades to that
declaration instead of stalling.

Contract under test:

1. A valid declaration binds to the session's approval key, so the guards read
   it back, and writes no config.
2. The two halves are mandatory together — commands without a root, or a root
   without commands, fail the create.
3. A declaration that cannot be honored (only program-bearing names, or a
   relative root) fails the create rather than handing back a session that
   silently has no allowlist — which is the stall the caller was avoiding.
4. A refused create leaves no session and burns no concurrency lease.
5. Omitting it leaves the session fully fail-closed: the unchanged path.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import tools.approval as approval
import tui_gateway.server as server


@pytest.fixture
def worktree(tmp_path):
    root = tmp_path / "game"
    root.mkdir()
    return str(root)


@pytest.fixture
def created_sessions(monkeypatch, tmp_path):
    """Call session.create for real, minus the background agent build."""
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
        session = server._sessions.pop(sid, None)
        if session and (key := session.get("session_key")):
            approval.clear_session(key)


def _session_key_of(resp: dict) -> str:
    return server._sessions[resp["result"]["session_id"]]["session_key"]


def _declared_for(resp: dict):
    return approval._declared_allowlist_for_session(_session_key_of(resp))


class TestSessionCreateAcceptsADeclaration:
    def test_it_binds_to_the_approval_key_without_touching_config(
        self, created_sessions, worktree, monkeypatch
    ) -> None:
        # No config default, so anything found came from this create alone.
        monkeypatch.setattr(approval, "_config_declared_allowlist", lambda: None)

        with patch.object(server, "_write_config_key") as write_key:
            resp = created_sessions({
                "cols": 80,
                "allowed_commands": ["godot", "  git  "],
                "allowed_command_root": worktree,
            })

        assert "result" in resp
        declared = _declared_for(resp)
        assert declared is not None
        assert declared.commands == frozenset({"godot", "git"})
        assert declared.root == worktree
        write_key.assert_not_called()

    def test_omitted_leaves_the_session_fail_closed(
        self, created_sessions, monkeypatch
    ) -> None:
        """The unchanged path every typed chat takes."""
        monkeypatch.setattr(approval, "_config_declared_allowlist", lambda: None)

        resp = created_sessions({"cols": 80})

        assert _declared_for(resp) is None

    def test_it_is_dropped_when_the_session_is_torn_down(
        self, created_sessions, worktree, monkeypatch
    ) -> None:
        monkeypatch.setattr(approval, "_config_declared_allowlist", lambda: None)
        monkeypatch.setattr(server, "_finalize_session", lambda *a, **k: None)

        resp = created_sessions({
            "cols": 80,
            "allowed_commands": ["godot"],
            "allowed_command_root": worktree,
        })
        session = server._sessions[resp["result"]["session_id"]]
        assert _declared_for(resp) is not None

        server._teardown_session(session, end_reason="test")

        assert _declared_for(resp) is None


class TestSessionCreateRejectsABadDeclaration:
    """A declaration that cannot be honored fails the create, not silently."""

    @pytest.mark.parametrize("params, needle", [
        ({"allowed_commands": ["git"]}, "allowed_command_root"),
        ({"allowed_command_root": "/tmp/x"}, "allowed_commands"),
        ({"allowed_commands": [], "allowed_command_root": "/tmp/x"}, "empty"),
        ({"allowed_commands": "git", "allowed_command_root": "/tmp/x"}, "array of strings"),
        ({"allowed_commands": ["git"], "allowed_command_root": 7}, "must be a string"),
        ({"allowed_commands": ["git"], "allowed_command_root": "game"}, "absolute path"),
    ])
    def test_shape_and_pairing_errors_are_named(
        self, created_sessions, params, needle
    ) -> None:
        resp = created_sessions({"cols": 80, **params})

        assert "error" in resp
        assert resp["error"]["code"] == 4029
        assert needle in resp["error"]["message"]

    @pytest.mark.parametrize("name", ["sh", "python3", "sudo", "env", "ssh", "xargs"])
    def test_a_program_bearing_command_is_refused(
        self, created_sessions, worktree, name
    ) -> None:
        """Allowlisting one of these would allowlist everything it runs."""
        resp = created_sessions({
            "cols": 80,
            "allowed_commands": [name],
            "allowed_command_root": worktree,
        })

        assert "error" in resp
        assert resp["error"]["code"] == 4029

    def test_a_refused_create_leaves_no_session_behind(
        self, created_sessions, worktree
    ) -> None:
        before = set(server._sessions)
        created_sessions({
            "cols": 80,
            "allowed_commands": ["sh"],
            "allowed_command_root": worktree,
        })

        assert set(server._sessions) == before

    def test_a_refused_create_claims_no_active_session_slot(
        self, created_sessions, worktree, monkeypatch
    ) -> None:
        """Validate BEFORE claiming the slot, or a typo burns a lease.

        The lease is released by session.close, which never runs for a create
        that returned an error — so a slot claimed here would be gone until
        restart. Ordering is the whole guard.
        """
        claim = MagicMock(return_value=(None, None))
        monkeypatch.setattr(server, "_claim_active_session_slot", claim)

        created_sessions({
            "cols": 80,
            "allowed_commands": ["sh"],
            "allowed_command_root": worktree,
        })

        claim.assert_not_called()
