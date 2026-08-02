"""Tests for ``hermes desktop spawn``.

Covers the CLI half of the desktop spawn control channel:
  - happy path: exact POST URL/headers/body, including omitted None keys
    and comma-separated --toolsets normalization
  - control file missing (desktop app not running)
  - control file present but corrupt / missing required fields
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
    defaults = dict(prompt="hello", model=None, provider=None, profile=None, toolsets=None)
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def _write_control_file(*, port=51234, token="tok_abc", schema_version=1):
    """Write a valid control.json under the (already-isolated) HERMES_HOME."""
    from hermes_constants import get_hermes_home

    desktop_dir = get_hermes_home() / "desktop"
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
        # Only `prompt` was set — model/provider/profile/toolsets must be
        # OMITTED entirely, never sent as JSON null.
        assert captured["body"] == {"prompt": "hello world"}

        out = capsys.readouterr().out
        assert "✓" in out

    def test_includes_optional_fields_and_splits_toolsets(self):
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
                    toolsets="browser, terminal ,, computer_use",
                )
            )

        assert captured["body"] == {
            "prompt": "do the thing",
            "model": "deepseek-v4-flash-0731-ds4",
            "provider": "ai-router",
            "profile": "work",
            "toolsets": ["browser", "terminal", "computer_use"],
        }

    def test_blank_toolsets_flag_is_omitted(self):
        _write_control_file()
        captured: dict = {}

        def fake_urlopen(req, timeout=None):
            captured["body"] = json.loads(req.data.decode())
            return _fake_http_202()

        with patch.object(ds.urllib.request, "urlopen", side_effect=fake_urlopen):
            ds.cmd_desktop_spawn(_ns(toolsets=" , ,"))

        assert "toolsets" not in captured["body"]


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
