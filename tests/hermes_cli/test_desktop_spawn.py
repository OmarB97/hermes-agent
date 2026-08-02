"""Tests for ``hermes desktop spawn``.

Covers the CLI half of the desktop spawn control channel:
  - happy path: exact POST URL/headers/body, including omitted None keys
  - --toolsets: comma-separated normalization onto the wire, and the values
    argparse cannot reject
  - control file missing (desktop app not running)
  - control file present but corrupt / missing required fields
  - --delegated / --delegated-timeout: wire shape and the flag combinations
    argparse cannot express
  - profile routing: the control file is read from the ROOT home (one app
    process serves every profile), and the requested profile is recovered
    from HERMES_HOME because the global pre-parse strips the flag
  - 401 (stale token), connection refused (stale file), 503 (no window),
    a generic non-202 status, and a generic 4xx with server-supplied detail

The desktop app's control server (apps/desktop/electron/spawn-control.ts) is
never started here — every HTTP call is mocked via urllib.request.urlopen.
Per AGENTS.md ("Tests must not write to ~/.hermes/"), the control file is
written under the per-test HERMES_HOME the autouse `_hermetic_environment`
fixture (tests/conftest.py) already redirects to a tmp dir — no real
``~/.hermes`` is ever touched.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

import hermes_cli.desktop_spawn as ds


def _ns(**kw):
    defaults = dict(
        prompt="hello",
        model=None,
        provider=None,
        profile=None,
        toolsets=None,
        delegated=False,
        delegated_timeout=None,
        goal=None,
        goal_turns=None,
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _write_control_file(*, port=51234, token="tok_abc", schema_version=1):
    """Write a valid control.json where the desktop app really publishes one.

    The app resolves its own HERMES_HOME through ``normalizeHermesHomeRoot``
    (apps/desktop/electron/backend-env.ts) before writing, so the file always
    lands at the ROOT home even when a profile is active. Writing it at
    ``get_hermes_home()`` instead would let a profile-scoped test pass against
    a file no app would ever have created there.
    """
    from hermes_constants import get_default_hermes_root

    desktop_dir = get_default_hermes_root() / "desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    path = desktop_dir / "control.json"
    path.write_text(
        json.dumps(
            {
                "schemaVersion": schema_version,
                "port": port,
                "token": token,
                "pid": 4242,
                "startedAt": "2026-08-02T04:00:00.000Z",
            }
        ),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _fake_http_202():
    """Return a context-manager urlopen stub reporting HTTP 202."""
    cm = MagicMock()
    cm.__enter__.return_value.status = 202
    return cm


class TestHappyPath:
    def test_posts_exact_url_header_and_body_omitting_none_keys(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            captured["body"] = json.loads(req.data.decode())
            captured["timeout"] = timeout
            captured["method"] = req.get_method()
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = ds.cmd_desktop_spawn(_ns(prompt="hello world"))

        assert result == 0
        assert captured["url"] == "http://127.0.0.1:51234/spawn"
        assert captured["method"] == "POST"
        assert captured["headers"]["X-hermes-desktop-token"] == "tok_abc"
        assert captured["headers"]["Content-type"] == "application/json"
        assert captured["timeout"] == 10.0
        # Only `prompt` was set — model/provider/profile must be OMITTED
        # entirely, never sent as JSON null.
        assert captured["body"] == {"prompt": "hello world"}

        out = capsys.readouterr().out
        assert "✓" in out

    def test_includes_optional_fields(self):
        _write_control_file(port=9999, token="tok_xyz")
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(
                _ns(
                    prompt="do the thing",
                    model="deepseek-v4-flash-0731-ds4",
                    provider="ai-router",
                    profile="work",
                )
            )

        assert captured["body"] == {
            "prompt": "do the thing",
            "model": "deepseek-v4-flash-0731-ds4",
            "provider": "ai-router",
            "profile": "work",
        }

    def test_splits_toolsets_onto_the_wire(self):
        """The flag is comma-separated for humans; the wire contract is a list.

        Blank segments are dropped so a trailing comma or a stray space is not
        POSTed as a toolset named "" — the gateway would refuse the whole
        create for it.
        """
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(
                _ns(prompt="do the thing", toolsets="file, terminal ,, skills")
            )

        assert captured["body"] == {
            "prompt": "do the thing",
            "toolsets": ["file", "terminal", "skills"],
        }

    def test_toolsets_naming_nothing_exits_nonzero(self, capsys):
        """`--toolsets " , ,"` is a typo, not a request to inherit.

        argparse cannot express "non-empty after splitting", so this is the
        layer that has to say it. Spawning with the app's toolsets instead
        would be the exact accept-and-ignore this flag was removed for in #315.
        """
        _write_control_file()

        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(toolsets=" , ,"))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        assert "--toolsets" in capsys.readouterr().out


def _use_profile_home(monkeypatch, name="meshboard-worker"):
    """Point HERMES_HOME at ``<root>/profiles/<name>`` and return both paths.

    This is what the real command line does. ``--profile <name>`` never
    reaches argparse — ``_apply_profile_override`` (hermes_cli/main.py) scans
    raw sys.argv first, rewrites HERMES_HOME to exactly this path, and strips
    the flag. A sticky ``hermes profile use <name>`` lands the same way.
    Reproducing the rewrite is what makes these tests exercise the real
    invocation instead of an ``args.profile`` no command line can produce.
    """
    from hermes_constants import get_default_hermes_root

    root = get_default_hermes_root()
    profile_home = root / "profiles" / name
    profile_home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(profile_home))
    return root, profile_home


class TestProfileRouting:
    """One app process owns every profile, so one control channel serves all.

    Regression cover for `hermes desktop spawn --profile <p>` reporting the
    app as not running whenever <p> was not the app's active profile.
    """

    def test_reads_control_file_from_the_root_while_scoped_to_a_profile(
        self, monkeypatch
    ):
        root, profile_home = _use_profile_home(monkeypatch)
        _write_control_file(port=7001, token="tok_root")
        # The app publishes here and only here.
        assert (root / "desktop" / "control.json").exists()
        assert not (profile_home / "desktop" / "control.json").exists()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["url"] = req.full_url
            captured["headers"] = dict(req.header_items())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            result = ds.cmd_desktop_spawn(_ns())

        assert result == 0
        assert captured["url"] == "http://127.0.0.1:7001/spawn"
        assert captured["headers"]["X-hermes-desktop-token"] == "tok_root"

    def test_profile_scoped_home_supplies_the_profile_the_flag_cannot(
        self, monkeypatch
    ):
        _use_profile_home(monkeypatch, "meshboard-worker")
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        # profile=None is what argparse really hands over — the pre-parse ate
        # the flag. The name has to come back off HERMES_HOME or the session
        # silently runs under whatever profile the app happens to be on.
        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns(profile=None))

        assert captured["body"]["profile"] == "meshboard-worker"

    def test_root_home_sends_no_profile(self):
        """No profile in play means "whatever the app has selected"."""
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns())

        assert "profile" not in captured["body"]

    def test_explicit_profile_argument_wins_over_the_home(self, monkeypatch):
        _use_profile_home(monkeypatch, "meshboard-worker")
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns(profile="analyst"))

        assert captured["body"]["profile"] == "analyst"

    def test_success_line_names_the_profile(self, monkeypatch, capsys):
        _use_profile_home(monkeypatch, "meshboard-worker")
        _write_control_file()

        with patch.object(
            ds.urllib.request, "urlopen", side_effect=lambda *a, **k: _fake_http_202()
        ):
            ds.cmd_desktop_spawn(_ns())

        out = capsys.readouterr().out
        assert "profile: meshboard-worker" in out

    def test_missing_control_file_names_the_root_not_the_profile(
        self, monkeypatch, capsys
    ):
        """The old message pointed at a path nothing ever writes."""
        root, profile_home = _use_profile_home(monkeypatch)
        # No control file anywhere — the app is genuinely not running.

        with pytest.raises(SystemExit) as exc:
            ds.cmd_desktop_spawn(_ns())

        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert str(root / "desktop" / "control.json") in out
        assert str(profile_home / "desktop" / "control.json") not in out


class TestControlFileFailures:
    def test_missing_control_file_exits_nonzero_with_actionable_message(self, capsys):
        # No control.json written — the per-test HERMES_HOME has no desktop/ dir.
        with pytest.raises(SystemExit) as exc:
            ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "does not appear to be running" in out
        assert "hermes desktop" in out

    def test_corrupt_json_exits_nonzero(self, capsys):
        from hermes_constants import get_hermes_home

        desktop_dir = get_hermes_home() / "desktop"
        desktop_dir.mkdir(parents=True)
        (desktop_dir / "control.json").write_text("{not json", encoding="utf-8")

        with pytest.raises(SystemExit) as exc:
            ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "not valid JSON" in capsys.readouterr().out

    def test_missing_token_field_exits_nonzero(self, capsys):
        from hermes_constants import get_hermes_home

        desktop_dir = get_hermes_home() / "desktop"
        desktop_dir.mkdir(parents=True)
        (desktop_dir / "control.json").write_text(
            json.dumps({"schemaVersion": 1, "port": 51234, "pid": 1, "startedAt": "x"}),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc:
            ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "missing 'port' or 'token'" in capsys.readouterr().out

    def test_missing_port_field_exits_nonzero(self, capsys):
        from hermes_constants import get_hermes_home

        desktop_dir = get_hermes_home() / "desktop"
        desktop_dir.mkdir(parents=True)
        (desktop_dir / "control.json").write_text(
            json.dumps({"schemaVersion": 1, "token": "tok_abc", "pid": 1, "startedAt": "x"}),
            encoding="utf-8",
        )

        with pytest.raises(SystemExit) as exc:
            ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "missing 'port' or 'token'" in capsys.readouterr().out


class TestTransportFailures:
    def test_401_exits_nonzero_with_stale_token_message(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:51234/spawn",
            code=401,
            msg="unauthorized",
            hdrs=None,
            fp=BytesIO(json.dumps({"ok": False, "error": "unauthorized"}).encode()),
        )
        with patch.object(ds.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        out = capsys.readouterr().out.lower()
        assert "stale" in out
        assert "restart" in out

    def test_connection_refused_exits_nonzero_with_stale_file_message(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        err = urllib.error.URLError(ConnectionRefusedError(61, "Connection refused"))
        with patch.object(ds.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        out = capsys.readouterr().out
        assert "not running" in out
        assert "stale" in out.lower()

    def test_503_no_window_exits_nonzero(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:51234/spawn",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=BytesIO(json.dumps({"ok": False, "error": "no window"}).encode()),
        )
        with patch.object(ds.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "no window" in capsys.readouterr().out.lower()

    def test_400_bad_request_surfaces_server_error_detail(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        err = urllib.error.HTTPError(
            url="http://127.0.0.1:51234/spawn",
            code=400,
            msg="Bad Request",
            hdrs=None,
            fp=BytesIO(json.dumps({"ok": False, "error": "prompt is required"}).encode()),
        )
        with patch.object(ds.urllib.request, "urlopen", side_effect=err):
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "prompt is required" in capsys.readouterr().out

    def test_unexpected_non_202_status_is_treated_as_failure(self, capsys):
        _write_control_file(port=51234, token="tok_abc")
        cm = MagicMock()
        cm.__enter__.return_value.status = 200
        with patch.object(ds.urllib.request, "urlopen", return_value=cm):
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns())
        assert exc.value.code == 1
        assert "200" in capsys.readouterr().out


class TestDelegated:
    """`--delegated` — the unattended spawn.

    The CLI's only job here is the wire shape: the desktop app owns the
    contract text and the auto-answer (apps/desktop/src/lib/delegated-spawn.ts).
    """

    def _body_for(self, **kw) -> dict:
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns(**kw))
        return captured["body"]

    def test_plain_spawn_sends_no_delegated_keys(self):
        # An ordinary spawn must stay byte-identical to what it was before the
        # flag existed, or every attended chat inherits unattended behavior.
        assert self._body_for(prompt="hi") == {"prompt": "hi"}

    def test_delegated_sends_the_flag_and_no_timeout_by_default(self):
        # No timeout on the wire means "use the app's default" — the CLI does
        # not get to hardcode a second copy of that number.
        assert self._body_for(prompt="hi", delegated=True) == {
            "prompt": "hi",
            "delegated": True,
        }

    def test_timeout_travels_in_milliseconds(self):
        # Seconds on the command line because that is how people say it;
        # milliseconds on the wire because that is what the renderer's timer
        # takes. One conversion, here.
        body = self._body_for(prompt="hi", delegated=True, delegated_timeout=45)
        assert body["delegatedTimeoutMs"] == 45000

    def test_fractional_seconds_round_to_whole_milliseconds(self):
        body = self._body_for(prompt="hi", delegated=True, delegated_timeout=1.5)
        assert body["delegatedTimeoutMs"] == 1500

    def test_delegated_composes_with_the_other_overrides(self):
        body = self._body_for(
            prompt="hi",
            model="deepseek-v4-flash-0731-ds4",
            provider="ai-router",
            delegated=True,
            delegated_timeout=30,
        )
        assert body == {
            "prompt": "hi",
            "model": "deepseek-v4-flash-0731-ds4",
            "provider": "ai-router",
            "delegated": True,
            "delegatedTimeoutMs": 30000,
        }

    def test_timeout_without_delegated_fails_loud(self, capsys):
        # Accepting it would run an attended session the caller believes is
        # unattended — exactly the failure the flag exists to prevent.
        _write_control_file()
        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(delegated_timeout=30))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        out = capsys.readouterr().out
        assert "--delegated-timeout" in out
        assert "--delegated" in out

    @pytest.mark.parametrize("seconds", [0, -1])
    def test_non_positive_timeout_fails_loud(self, seconds, capsys):
        _write_control_file()
        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(delegated=True, delegated_timeout=seconds))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        assert "greater than 0" in capsys.readouterr().out

    def test_success_line_says_the_run_is_unattended(self, capsys):
        self._body_for(prompt="hi", delegated=True)
        out = capsys.readouterr().out
        assert "✓" in out
        assert "delegated" in out


class TestGoal:
    """`--goal` gives the session a standing objective.

    The loop itself is old (hermes_cli/goals.py, reachable as `/goal <text>`);
    what this flag adds is setting it at session creation. The workaround it
    replaces was spawning the prompt `"/goal <objective>"`, which worked only
    because the renderer happens to parse slash commands out of a submitted
    message — an interface nobody would have chosen.
    """

    def _body_for(self, **kw) -> dict:
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns(**kw))
        return captured["body"]

    def test_plain_spawn_sends_no_goal_keys(self):
        # An ordinary spawn stays byte-identical to what it was before the flag.
        assert self._body_for(prompt="hi") == {"prompt": "hi"}

    def test_goal_alone_becomes_both_the_goal_and_the_opening_turn(self):
        """Mirrors `/goal <text>`, which sets the goal and sends that same text
        as the kickoff turn. Making the caller repeat it would be noise."""
        assert self._body_for(prompt=None, goal="ship the parser") == {
            "prompt": "ship the parser",
            "goal": "ship the parser",
        }

    def test_an_explicit_prompt_stays_the_opening_turn(self):
        """A different first move must remain expressible."""
        assert self._body_for(prompt="read the tests first", goal="ship the parser") == {
            "prompt": "read the tests first",
            "goal": "ship the parser",
        }

    def test_goal_turns_rides_along_as_camel_case(self):
        assert self._body_for(prompt=None, goal="ship it", goal_turns=40) == {
            "prompt": "ship it",
            "goal": "ship it",
            "goalMaxTurns": 40,
        }

    def test_goal_composes_with_delegated_and_the_other_overrides(self):
        """The case this was built for: an unattended session with a standing
        objective, pinned to one profile and model."""
        body = self._body_for(
            prompt=None,
            goal="ship the parser",
            goal_turns=40,
            delegated=True,
            model="deepseek-v4-flash-0731-ds4",
            provider="ai-router",
            profile="work",
        )

        assert body == {
            "prompt": "ship the parser",
            "goal": "ship the parser",
            "goalMaxTurns": 40,
            "delegated": True,
            "model": "deepseek-v4-flash-0731-ds4",
            "provider": "ai-router",
            "profile": "work",
        }

    def test_goal_and_prompt_are_stripped(self):
        assert self._body_for(prompt="  ", goal="  ship it  ") == {
            "prompt": "ship it",
            "goal": "ship it",
        }

    def test_neither_prompt_nor_goal_fails_loud_naming_both_fixes(self, capsys):
        _write_control_file()
        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(prompt=None))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        out = capsys.readouterr().out
        assert "--goal" in out

    def test_goal_turns_without_a_goal_fails_loud(self, capsys):
        """A turn budget with no goal would be read by nobody."""
        _write_control_file()
        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(prompt="hi", goal_turns=40))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        out = capsys.readouterr().out
        assert "--goal-turns" in out
        assert "--goal" in out

    @pytest.mark.parametrize("turns", [0, -1])
    def test_non_positive_goal_turns_fails_loud(self, turns, capsys):
        _write_control_file()
        with patch.object(ds.urllib.request, "urlopen") as urlopen:
            with pytest.raises(SystemExit) as exc:
                ds.cmd_desktop_spawn(_ns(prompt=None, goal="ship it", goal_turns=turns))

        assert exc.value.code == 1
        urlopen.assert_not_called()
        assert "at least 1" in capsys.readouterr().out

    def test_success_line_says_the_session_keeps_going(self, capsys):
        self._body_for(prompt=None, goal="ship it", goal_turns=40)
        out = capsys.readouterr().out
        assert "✓" in out
        assert "goal" in out
        assert "40 turns" in out
