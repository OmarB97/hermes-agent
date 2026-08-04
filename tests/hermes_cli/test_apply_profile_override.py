"""Regression tests for _apply_profile_override HERMES_HOME guard (issue #22502).

When HERMES_HOME is set to the hermes root (e.g. systemd hardcodes
HERMES_HOME=/root/.hermes), _apply_profile_override must still read
active_profile and update HERMES_HOME to the profile directory.

When HERMES_HOME is already a profile directory (.../profiles/<name>),
_apply_profile_override must trust it and return without re-reading
active_profile (child-process inheritance contract).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _run_apply_profile_override(
    tmp_path, monkeypatch, *, hermes_home: str | None, active_profile: str | None,
    argv: list[str] | None = None,
):
    """Run _apply_profile_override in isolation.

    Returns the value of os.environ["HERMES_HOME"] after the call,
    or None if unset.
    """
    hermes_root = tmp_path / ".hermes"
    hermes_root.mkdir(parents=True, exist_ok=True)

    if active_profile is not None:
        (hermes_root / "active_profile").write_text(active_profile)

    if active_profile and active_profile != "default":
        (hermes_root / "profiles" / active_profile).mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    if hermes_home is not None:
        monkeypatch.setenv("HERMES_HOME", hermes_home)
    else:
        monkeypatch.delenv("HERMES_HOME", raising=False)

    monkeypatch.setattr(sys, "argv", argv or ["hermes", "gateway", "start"])

    from hermes_cli.main import _apply_profile_override
    _apply_profile_override()

    return os.environ.get("HERMES_HOME")


class TestApplyProfileOverrideHermesHomeGuard:
    """Regression guard for issue #22502.

    Verifies that HERMES_HOME pointing to the hermes root does NOT suppress
    the active_profile check, while HERMES_HOME already pointing to a
    profile directory IS trusted as-is.
    """

    def test_hermes_home_at_root_with_active_profile_is_redirected(
        self, tmp_path, monkeypatch
    ):
        """HERMES_HOME=/root/.hermes + active_profile=coder must redirect
        HERMES_HOME to .../profiles/coder.

        Bug scenario from #22502: systemd sets HERMES_HOME to the hermes root
        and the user switches to a profile via `hermes profile use`.
        Before the fix, the guard returned early and active_profile was ignored.
        """
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)

        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=str(hermes_root),
            active_profile="coder",
        )

        assert result is not None, "HERMES_HOME must be set after profile redirect"
        assert "profiles" in result, (
            f"Expected HERMES_HOME to point into profiles/ dir, got: {result!r}"
        )
        assert result.endswith("coder"), (
            f"Expected HERMES_HOME to end with 'coder', got: {result!r}"
        )


    def test_sudo_explicit_profile_resolves_invoking_users_profile(self, tmp_path, monkeypatch):
        """sudo elias ... should resolve `-p elias` under SUDO_USER, not root."""
        root_home = tmp_path / "root"
        user_home = tmp_path / "home" / "hermes"
        profile_dir = user_home / ".hermes" / "profiles" / "elias"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (root_home / ".hermes").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: root_home)
        monkeypatch.setenv("SUDO_USER", "hermes")
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
        monkeypatch.setattr(sys, "argv", ["hermes", "-p", "elias", "gateway", "install", "--system"])

        import pwd

        monkeypatch.setattr(pwd, "getpwnam", lambda name: SimpleNamespace(pw_dir=str(user_home)))

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(profile_dir)
        assert sys.argv == ["hermes", "gateway", "install", "--system"]



    def test_skip_profile_override_preserves_launcher_sandbox(
        self, tmp_path, monkeypatch
    ):
        """An explicit launcher sandbox must win over sticky active_profile."""
        hermes_root = tmp_path / ".hermes"
        profile_dir = hermes_root / "profiles" / "glm"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (hermes_root / "active_profile").write_text("glm")

        sandbox = tmp_path / "meshboard-dispatch-sandbox"
        sandbox.mkdir()

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.setenv("HERMES_HOME", str(sandbox))
        monkeypatch.setenv("HERMES_SKIP_PROFILE_OVERRIDE", "1")
        monkeypatch.setattr(sys, "argv", ["hermes", "-z", "review task"])

        from hermes_cli.main import _apply_profile_override

        _apply_profile_override()

        assert os.environ.get("HERMES_HOME") == str(sandbox)
        assert sys.argv == ["hermes", "-z", "review task"]


class TestSupervisedChildIgnoresStickyProfile:
    """The reserved default gateway s6 slot must not follow active_profile.

    Inside the Docker s6 image the ``gateway-default`` service slot runs a
    bare ``hermes gateway run`` (no ``-p``) to mean "the root HERMES_HOME
    profile". The run-script exports ``HERMES_S6_SUPERVISED_CHILD=1``.
    Without a guard, ``_apply_profile_override`` would read the sticky
    ``active_profile`` file (set by e.g. the dashboard profile switcher) and
    redirect the reserved default gateway into that profile — producing a
    duplicate gateway for the active profile and no real default gateway.
    """


    def test_non_supervised_run_still_follows_active_profile(
        self, tmp_path, monkeypatch
    ):
        """Without the sentinel, a normal `hermes gateway run` still honors
        active_profile — the guard is scoped strictly to supervised children."""
        result = _run_apply_profile_override(
            tmp_path,
            monkeypatch,
            hermes_home=None,
            active_profile="briefer",
            argv=["hermes", "gateway", "run"],
        )

        assert result is not None
        assert result.endswith("briefer")

    def test_supervised_named_profile_flag_still_wins(self, tmp_path, monkeypatch):
        """A supervised named-profile slot passes ``-p <name>`` explicitly;
        that must still resolve (the sentinel guard only skips the sticky
        active_profile fallback, never an explicit flag)."""
        hermes_root = tmp_path / ".hermes"
        hermes_root.mkdir(parents=True, exist_ok=True)
        (hermes_root / "active_profile").write_text("briefer")
        (hermes_root / "profiles" / "briefer").mkdir(parents=True, exist_ok=True)
        (hermes_root / "profiles" / "coder").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setenv("HERMES_S6_SUPERVISED_CHILD", "1")
        monkeypatch.setattr(sys, "argv", ["hermes", "-p", "coder", "gateway", "run"])

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        result = os.environ.get("HERMES_HOME")
        assert result is not None
        assert result.endswith("coder")


class TestPytestArgvIsNotAHermesCommandLine:
    """pytest's ``-p <plugin>`` must not be read as ``--profile <name>``.

    ``_apply_profile_override`` runs at import of ``hermes_cli.main`` and scans
    raw ``sys.argv``. Under pytest that argv belongs to pytest, whose ``-p``
    selects a plugin. The step-1b regex only rejects names that cannot be
    profile ids ("no:logging"); plugin names like "anyio" or "cacheprovider"
    are valid ids, so they used to reach ``resolve_profile_env()`` and exit 1 —
    failing every test in any file importing this module, directly or
    transitively, with "Profile 'anyio' does not exist".
    """

    def test_under_pytest_is_true_during_a_test(self):
        from hermes_cli.main import _under_pytest

        assert _under_pytest() is True

    def test_under_pytest_survives_unset_current_test_env(self, monkeypatch):
        """Collection-time imports have no PYTEST_CURRENT_TEST — sys.modules
        is what makes the guard cover them."""
        from hermes_cli.main import _under_pytest

        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert _under_pytest() is True

    def test_reimport_with_pytest_plugin_flag_does_not_exit(self, monkeypatch):
        """The regression: a plugin name that is a *valid* profile id.

        "cacheprovider" matches the step-1b profile-id regex, so only the
        under-pytest guard keeps this import from calling resolve_profile_env()
        and exiting 1 on a profile nobody asked for.
        """
        import importlib

        monkeypatch.setattr(
            sys, "argv", ["pytest", "-p", "cacheprovider", "tests/x.py"]
        )
        monkeypatch.delenv("HERMES_SKIP_PROFILE_OVERRIDE", raising=False)
        monkeypatch.setitem(sys.modules, "hermes_cli.main", None)
        sys.modules.pop("hermes_cli.main", None)

        # Must not raise SystemExit.
        importlib.import_module("hermes_cli.main")

    def test_explicit_call_still_reads_a_real_hermes_argv(
        self, tmp_path, monkeypatch
    ):
        """The guard is on the import-time call only — the function itself is
        unchanged, so tests (and the CLI) still get full profile resolution
        even though this whole suite runs under pytest."""
        hermes_root = tmp_path / ".hermes"
        (hermes_root / "profiles" / "coder").mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        monkeypatch.delenv("HERMES_HOME", raising=False)
        monkeypatch.setattr(sys, "argv", ["hermes", "-p", "coder", "chat"])

        from hermes_cli.main import _apply_profile_override
        _apply_profile_override()

        result = os.environ.get("HERMES_HOME")
        assert result is not None
        assert result.endswith("coder")
