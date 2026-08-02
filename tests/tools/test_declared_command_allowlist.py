"""Declared command allowlists: the policy an unattended run degrades to.

``tools/approval.py::_smart_approve`` fails CLOSED — every exception out of its
``call_llm(task="approval")`` becomes ``"escalate"``, i.e. ask a human. With
nobody there to ask, that is a stall rather than a decision: in a gateway/ask
surface the command either burns ``approvals.timeout`` and is BLOCKED, or comes
straight back as ``status: "pending_approval"`` and halts the operation.

These tests pin the three things that matter about closing that gap:

1. **Nothing changes by default.** With no declaration, a classifier failure
   escalates exactly as it does today.
2. **A declaration only helps where it was declared.** Classifier unreachable +
   a command on the list, inside the declared root → approved by policy, with
   the reason on the record. Anything else still escalates.
3. **The matcher is not fooled by shell text.** Chaining, substitution,
   redirection, an argument outside the root, a symlink that leaves it, and an
   environment-assignment prefix are all refused.

The classifier is made unavailable the way it really failed on 2026-08-02 — by
having ``agent.auxiliary_client.call_llm`` raise — rather than by stubbing
``_smart_approve``, so these tests exercise the real fail-closed site.
"""

import logging
from unittest.mock import patch

import pytest

import tools.approval as approval_module
from tools.approval import (
    check_all_command_guards,
    clear_session,
    set_current_session_key,
    set_session_command_allowlist,
)
from tools.declared_allowlist import build_declared_allowlist, match_declared_command


SESSION_KEY = "test:session:declared-allowlist"

# Captured before any fixture stubs it out, so the one test that cares about the
# profile-wide config surface can exercise the real resolver.
REAL_CONFIG_RESOLVER = approval_module._config_declared_allowlist

# Genuinely flagged by detect_dangerous_command ("git clean with force"), so it
# reaches the smart-approval gate without any detector stubbing — and it is
# exactly the kind of command a delegated worktree session legitimately runs.
FLAGGED_GIT = "git clean -fdx"


@pytest.fixture
def worktree(tmp_path):
    """A declared worktree, a sibling directory outside it, and a link out."""
    root = tmp_path / "worktree"
    (root / "sub").mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("not yours")
    # A name that lives inside the worktree but leads out of it. Judging the
    # name would allow it; judging where it leads must not.
    (root / "escape").symlink_to(outside)
    return root


@pytest.fixture
def unattended(monkeypatch):
    """An unattended smart-approval surface with clean approval state.

    ``HERMES_EXEC_ASK`` with no notify callback registered is the cheap,
    non-blocking shape of "a human is being asked": the guard queues a pending
    approval and returns ``status: "pending_approval"`` instead of sitting on
    ``approvals.timeout``. That return IS the stall this feature exists to
    remove, so it is what the escalation assertions look for.
    """
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.setenv("HERMES_EXEC_ASK", "1")
    monkeypatch.setenv("HERMES_SESSION_KEY", SESSION_KEY)
    monkeypatch.setattr(approval_module, "_YOLO_MODE_FROZEN", False)
    monkeypatch.setattr(approval_module, "_get_approval_mode", lambda: "smart")
    # No config-level declaration unless a test asks for one.
    monkeypatch.setattr(approval_module, "_config_declared_allowlist", lambda: None)

    token = set_current_session_key(SESSION_KEY)
    saved_permanent = approval_module._permanent_approved.copy()
    approval_module._permanent_approved.clear()
    approval_module._session_approved.clear()
    try:
        yield SESSION_KEY
    finally:
        approval_module._permanent_approved.update(saved_permanent)
        try:
            approval_module._approval_session_key.reset(token)
        except Exception:
            pass
        clear_session(SESSION_KEY)


@pytest.fixture
def tirith_warns(monkeypatch):
    """Make every command reach the smart-approval gate.

    Tirith flags on content rather than on a command pattern, so this is how a
    perfectly ordinary ``godot`` invocation ends up in front of the classifier
    in real life. Tests that need a command the pattern detector ignores use
    this instead of inventing a fake dangerous command.
    """
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _cmd: {
            "action": "warn",
            "findings": [{"rule_id": "test-rule", "severity": "LOW",
                          "title": "flagged for test", "description": "flagged"}],
            "summary": "flagged for test",
        },
    )


@pytest.fixture
def tirith_quiet(monkeypatch):
    monkeypatch.setattr(
        "tools.tirith_security.check_command_security",
        lambda _cmd: {"action": "allow", "findings": [], "summary": ""},
    )


def classifier_unavailable():
    """Patch context where the approval classifier cannot be reached at all."""
    return patch(
        "agent.auxiliary_client.call_llm",
        side_effect=RuntimeError("no API key configured for the approval lane"),
    )


def classifier_says(answer: str):
    """Patch context where the classifier is up and returns *answer*."""
    class _Msg:
        content = answer

    class _Choice:
        message = _Msg()

    class _Response:
        choices = [_Choice()]

    return patch("agent.auxiliary_client.call_llm", return_value=_Response())


def declare(commands, root):
    return set_session_command_allowlist(SESSION_KEY, commands, str(root))


def guard(command, cwd, env_type="local"):
    return check_all_command_guards(command, env_type, cwd=str(cwd))


def assert_escalated(result):
    """The command was NOT run and a human is being asked — today's behaviour."""
    assert result["approved"] is False
    assert result.get("status") == "pending_approval"
    assert "policy_approved" not in result


def assert_policy_approved(result):
    assert result["approved"] is True
    assert result.get("policy_approved") is True
    assert result.get("policy_reason")


# ── The default: an absent declaration changes nothing ────────────────────


class TestNoDeclaration:
    def test_classifier_failure_still_escalates(self, unattended, tirith_quiet, worktree):
        with classifier_unavailable():
            result = guard(FLAGGED_GIT, worktree)

        assert_escalated(result)
        # The human really was queued, not just refused.
        assert approval_module._pending.get(SESSION_KEY, {}).get("command") == FLAGGED_GIT

    def test_an_empty_declaration_is_no_declaration(self, unattended, tirith_quiet, worktree):
        assert declare([], worktree) is False
        assert declare(["git"], "") is False

        with classifier_unavailable():
            assert_escalated(guard(FLAGGED_GIT, worktree))

    def test_config_default_is_empty_out_of_the_box(self, monkeypatch):
        """The shipped default must produce no allowlist at all."""
        from hermes_cli.config import DEFAULT_CONFIG

        declared = DEFAULT_CONFIG["approvals"]["delegated_allowlist"]
        assert declared == {"commands": [], "root": ""}
        assert build_declared_allowlist(
            declared["commands"], declared["root"], source="test"
        ) is None


# ── The degradation: declared + unreachable classifier ────────────────────


class TestDeclaredAndUnavailable:
    def test_declared_command_is_approved_by_policy(
        self, unattended, tirith_quiet, worktree, caplog
    ):
        assert declare(["godot", "git"], worktree) is True

        with caplog.at_level(logging.WARNING, logger="tools.approval"):
            with classifier_unavailable():
                result = guard(FLAGGED_GIT, worktree)

        assert_policy_approved(result)
        assert str(worktree.resolve()) in result["policy_reason"]
        # The reason has to be on the record: nobody saw this approval happen.
        assert "AUTO-APPROVED by declared command allowlist" in caplog.text
        assert FLAGGED_GIT in caplog.text

    def test_a_subdirectory_of_the_declared_root_is_inside_it(
        self, unattended, tirith_quiet, worktree
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_policy_approved(guard(FLAGGED_GIT, worktree / "sub"))

    def test_config_default_covers_sessions_that_declare_nothing(
        self, unattended, tirith_quiet, worktree, monkeypatch
    ):
        """The profile-wide surface is live, not just the per-session one.

        Restores the real resolver (the fixture stubs it out) and feeds it a
        real config dict, so this exercises the actual
        ``approvals.delegated_allowlist`` read rather than a fake.
        """
        monkeypatch.setattr(
            approval_module, "_config_declared_allowlist", REAL_CONFIG_RESOLVER
        )
        monkeypatch.setattr(
            "hermes_cli.config.load_config",
            lambda *a, **k: {
                "approvals": {
                    "delegated_allowlist": {
                        "commands": ["git"],
                        "root": str(worktree),
                    }
                }
            },
        )

        with classifier_unavailable():
            assert_policy_approved(guard(FLAGGED_GIT, worktree))

    def test_a_session_declaration_replaces_the_config_default(
        self, unattended, tirith_quiet, worktree, monkeypatch
    ):
        """A brief that names godot must not inherit config's git."""
        monkeypatch.setattr(
            approval_module,
            "_config_declared_allowlist",
            lambda: build_declared_allowlist(["git"], str(worktree), source="test"),
        )
        declare(["godot"], worktree)

        with classifier_unavailable():
            assert_escalated(guard(FLAGGED_GIT, worktree))


# ── Declared, but this command is not covered ─────────────────────────────


class TestDeclaredButNotCovered:
    def test_undeclared_command_still_escalates(self, unattended, tirith_quiet, worktree):
        declare(["godot"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(FLAGGED_GIT, worktree))

    def test_a_working_classifier_that_escalates_is_not_overridden(
        self, unattended, tirith_quiet, worktree
    ):
        """Uncertainty from a REACHABLE classifier still wants a human."""
        declare(["git"], worktree)
        with classifier_says("ESCALATE"):
            assert_escalated(guard(FLAGGED_GIT, worktree))

    def test_a_working_classifier_that_denies_is_not_overridden(
        self, unattended, tirith_quiet, worktree
    ):
        declare(["git"], worktree)
        with classifier_says("DENY"):
            result = guard(FLAGGED_GIT, worktree)
        assert result["approved"] is False
        assert "policy_approved" not in result

    def test_interactive_cli_still_prompts(self, unattended, tirith_quiet, worktree, monkeypatch):
        """A human at a terminal answers immediately; nothing to degrade from."""
        monkeypatch.setenv("HERMES_INTERACTIVE", "1")
        monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
        declare(["git"], worktree)
        asked = []

        def cb(command, description, *, allow_permanent=True, smart_denied=False):
            asked.append(command)
            return "deny"

        with classifier_unavailable():
            result = check_all_command_guards(
                FLAGGED_GIT, "local", approval_callback=cb, cwd=str(worktree),
            )

        assert asked == [FLAGGED_GIT]
        assert result["approved"] is False
        assert "policy_approved" not in result

    def test_non_local_backend_is_never_covered(self, unattended, tirith_quiet, worktree):
        """A declared root names a path on THIS machine, not inside a sandbox."""
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(FLAGGED_GIT, worktree, env_type="ssh"))

    def test_unknown_working_directory_is_not_covered(self, unattended, tirith_quiet, worktree):
        declare(["git"], worktree)
        with classifier_unavailable():
            result = check_all_command_guards(FLAGGED_GIT, "local")
        assert_escalated(result)


# ── Evasion: the matcher must not be fooled by shell text ─────────────────


class TestEvasion:
    # Payloads are destructive but deliberately NOT hardline (`rm -rf /` is
    # blocked unconditionally long before any of this), so what refuses them
    # here is the matcher and nothing else.
    @pytest.mark.parametrize("command", [
        "git status ; rm -rf /tmp/gone",
        "git status && rm -rf /tmp/gone",
        "git status || rm -rf /tmp/gone",
        "git status | tee /etc/passwd",
        "git status > /etc/passwd",
        "git status $(rm -rf /tmp/gone)",
        "git status `rm -rf /tmp/gone`",
        "git status\nrm -rf /tmp/gone",
        "git status ${IFS}foo",
        "git clean -fdx *",
        "git clean -fdx ~",
        "git clean -fdx \\; rm -rf /tmp/gone",
    ])
    def test_shell_syntax_is_refused(
        self, unattended, tirith_warns, worktree, command
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(command, worktree))

    @pytest.mark.parametrize("command", [
        "git -C /etc status",                  # explicit path out
        "git clean -fdx /tmp",                 # absolute argument out
        "git clean -fdx ../outside",           # traversal out
        "git clean -fdx escape",               # symlink whose target is out
        "git clean -fdx escape/secret.txt",    # through the symlink
        "git --git-dir=/etc/git clean -fdx",   # out via an option value
    ])
    def test_reaching_outside_the_declared_root_is_refused(
        self, unattended, tirith_warns, worktree, command
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(command, worktree))

    def test_running_outside_the_declared_root_is_refused(
        self, unattended, tirith_quiet, worktree
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(FLAGGED_GIT, worktree.parent))

    # Found by probing the matcher, not by reasoning about it: `-c` is a bare
    # flag so it is skipped, and `core.pager=/bin/sh` does not start with `-`,
    # so checking the WHOLE token resolved it to an innocent relative name
    # under the worktree — while the half git actually runs is an absolute
    # path to a shell. Both halves of a `key=value` argument are checked now,
    # and a value naming a program-bearing command is refused wherever it
    # appears, not just in the executable slot.
    @pytest.mark.parametrize("command", [
        "git -c core.pager=/bin/sh status",
        "git -c core.pager=sh status",
        "git -c core.sshCommand=/bin/bash status",
        "git -c diff.external=python3 diff",
        "git -c core.editor=/usr/bin/env log",
        "git -c alias.x=sh x",
        "git --pager=sh log",
    ])
    def test_an_argument_that_names_an_interpreter_is_refused(
        self, unattended, tirith_warns, worktree, command
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard(command, worktree))

    def test_environment_assignment_prefix_is_refused(
        self, unattended, tirith_warns, worktree
    ):
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard("GIT_DIR=/etc/git git clean -fdx", worktree))

    def test_a_different_path_to_the_same_name_is_not_the_declared_command(
        self, unattended, tirith_warns, worktree
    ):
        """`git` declares `git`, not any binary that happens to be called git."""
        declare(["git"], worktree)
        with classifier_unavailable():
            assert_escalated(guard("/usr/bin/git clean -fdx", worktree))
            assert_escalated(guard("./git clean -fdx", worktree))


# ── The matcher and the declaration parser, on their own ──────────────────


class TestMatcher:
    def test_matches_a_declared_command_in_its_root(self, worktree):
        allowlist = build_declared_allowlist(["godot"], str(worktree), source="test")
        decision = match_declared_command(
            "godot --headless --export-release Linux build/game", allowlist, str(worktree)
        )
        assert decision.allowed is True
        assert "godot" in decision.reason

    def test_no_allowlist_is_never_a_match(self, worktree):
        assert match_declared_command("godot", None, str(worktree)).allowed is False

    def test_a_relative_working_directory_cannot_be_enforced(self, worktree):
        allowlist = build_declared_allowlist(["git"], str(worktree), source="test")
        decision = match_declared_command("git clean -fdx", allowlist, "worktree")
        assert decision.allowed is False
        assert "absolute" in decision.reason

    def test_unbalanced_quoting_is_refused(self, worktree):
        allowlist = build_declared_allowlist(["git"], str(worktree), source="test")
        decision = match_declared_command("git commit -m 'oops", allowlist, str(worktree))
        assert decision.allowed is False

    @pytest.mark.parametrize("command", [
        "git status",
        "git clean -fdx",
        "git commit -m 'ship it'",
        "git add sub/file.txt",
        "git log --format=%H",
        "git config user.name=me",
        "git status --porcelain=v2",
        "godot --headless --export-release Linux build/game",
    ])
    def test_ordinary_invocations_are_not_refused(self, worktree, command):
        """The strictness must not cost the commands people actually run."""
        allowlist = build_declared_allowlist(
            ["git", "godot"], str(worktree), source="test"
        )
        decision = match_declared_command(command, allowlist, str(worktree))
        assert decision.allowed is True, decision.reason


class TestTerminalToolEndToEnd:
    """The wiring, not the policy: does a real terminal() call reach this?

    ``check_all_command_guards`` learns the run directory from its caller, and
    the caller resolves it the same way execution does — including an explicit
    ``workdir=``, which overrides the session cwd. Judging the session default
    while the command runs somewhere else would enforce the declared root
    against the wrong directory, which is how a scoped allowlist quietly stops
    being scoped.
    """

    @pytest.fixture
    def fake_terminal(self, monkeypatch, worktree):
        import tools.terminal_tool as terminal_tool

        executed = []

        class FakeEnv:
            env = {}

            def execute(self, command, **kwargs):
                executed.append((command, kwargs))
                return {"output": "ok", "returncode": 0}

        monkeypatch.setattr(terminal_tool, "_active_environments", {SESSION_KEY: FakeEnv()})
        monkeypatch.setattr(terminal_tool, "_last_activity", {})
        monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
        monkeypatch.setattr(
            terminal_tool,
            "_get_env_config",
            lambda: {
                "env_type": "local",
                "cwd": str(worktree),
                "timeout": 60,
                "lifetime_seconds": 3600,
            },
        )
        return terminal_tool, executed

    def test_a_declared_command_runs_without_a_human(
        self, unattended, tirith_quiet, worktree, fake_terminal
    ):
        import json

        terminal_tool, executed = fake_terminal
        declare(["git"], worktree)

        with classifier_unavailable():
            result = json.loads(
                terminal_tool.terminal_tool(command=FLAGGED_GIT, task_id=SESSION_KEY)
            )

        assert result["exit_code"] == 0
        assert [cmd for cmd, _ in executed] == [FLAGGED_GIT]
        # The note is the only trace a person will ever see of this approval.
        assert "declared command allowlist" in result["approval"]

    def test_an_explicit_workdir_outside_the_root_is_judged_not_ignored(
        self, unattended, tirith_quiet, worktree, fake_terminal
    ):
        import json

        terminal_tool, executed = fake_terminal
        declare(["git"], worktree)

        with classifier_unavailable():
            result = json.loads(
                terminal_tool.terminal_tool(
                    command=FLAGGED_GIT,
                    task_id=SESSION_KEY,
                    workdir=str(worktree.parent),
                )
            )

        assert result["status"] == "pending_approval"
        assert executed == []


class TestDeclarationParsing:
    def test_a_root_is_required(self, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.declared_allowlist"):
            assert build_declared_allowlist(["git"], "", source="test") is None
        assert "no root directory" in caplog.text

    def test_the_filesystem_root_is_not_a_scope(self):
        assert build_declared_allowlist(["git"], "/", source="test") is None

    def test_a_relative_root_is_refused(self):
        assert build_declared_allowlist(["git"], "worktree", source="test") is None

    @pytest.mark.parametrize(
        "name", ["sh", "bash", "python3", "sudo", "env", "ssh", "docker", "xargs",
                 "timeout", "/usr/bin/env"],
    )
    def test_program_bearing_commands_are_refused(self, tmp_path, name, caplog):
        with caplog.at_level(logging.WARNING, logger="tools.declared_allowlist"):
            assert build_declared_allowlist([name], str(tmp_path), source="test") is None
        assert "allowlist every command" in caplog.text

    @pytest.mark.parametrize("name", ["./godot", "../tools/godot", "bin/godot"])
    def test_a_relative_executable_path_is_refused(self, tmp_path, name):
        """It would name a different binary in every directory the run visits."""
        assert build_declared_allowlist([name], str(tmp_path), source="test") is None

    def test_an_absolute_executable_path_is_allowed(self, tmp_path):
        allowlist = build_declared_allowlist(
            ["/opt/godot/bin/godot"], str(tmp_path), source="test"
        )
        assert allowlist is not None
        assert allowlist.commands == frozenset({"/opt/godot/bin/godot"})

    def test_a_command_line_is_not_a_command_name(self, tmp_path):
        assert build_declared_allowlist(
            ["git clean -fdx"], str(tmp_path), source="test"
        ) is None
        assert build_declared_allowlist(
            ["git; rm -rf /"], str(tmp_path), source="test"
        ) is None

    def test_junk_entries_are_dropped_not_fatal(self, tmp_path):
        allowlist = build_declared_allowlist(
            ["git", "", None, 7, "  godot  "], str(tmp_path), source="test"
        )
        assert allowlist is not None
        assert allowlist.commands == frozenset({"git", "godot"})
