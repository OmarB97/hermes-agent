"""``hermes desktop spawn`` — ask the running desktop app to start a session.

The Hermes Electron app starts a small loopback HTTP control server on
launch (see ``apps/desktop/electron/spawn-control.ts``) and publishes its
port + a per-launch token in a 0600 file at
``<HERMES_HOME>/desktop/control.json`` (``get_hermes_home()`` — never a
hardcoded ``~/.hermes``, that breaks profiles). This module reads that file
and POSTs the prompt to ``/spawn``, which hands it to the renderer to start
a new chat exactly as if it had been typed in — streaming, transcript,
sidebar all go through the existing typed-chat path.

This CLI command is a one-shot fire-and-forget POST, not a client for the
resulting session: the desktop app owns it from here.
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

SPAWN_TOKEN_HEADER = "X-Hermes-Desktop-Token"
_REQUEST_TIMEOUT = 10.0


def _control_file_path() -> Path:
    """Return the desktop control file path under the active HERMES_HOME."""
    from hermes_constants import get_hermes_home

    return get_hermes_home() / "desktop" / "control.json"


def _read_control_file(path: Path) -> dict:
    """Read and validate the desktop control file.

    Raises RuntimeError with a user-facing message for every failure mode:
    the app was never launched (no file), or a corrupt/incomplete file.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "The Hermes desktop app does not appear to be running (no control "
            f"file at {path}). Start it first with `hermes desktop`."
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"Could not read the desktop control file at {path}: {exc}"
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Desktop control file at {path} is not valid JSON ({exc}). "
            "Restart the desktop app."
        ) from exc

    if (
        not isinstance(data, dict)
        or not isinstance(data.get("port"), int)
        or not data.get("token")
    ):
        raise RuntimeError(
            f"Desktop control file at {path} is missing 'port' or 'token'. "
            "Restart the desktop app."
        )
    return data


def _spawn_request_body(args) -> dict:
    """Build the JSON body for POST /spawn, omitting unset optional fields."""
    body: dict = {"prompt": args.prompt}
    for key in ("model", "provider", "profile"):
        value = getattr(args, key, None)
        if value:
            body[key] = value

    raw_toolsets = getattr(args, "toolsets", None)
    if raw_toolsets:
        toolsets = [t.strip() for t in raw_toolsets.split(",") if t.strip()]
        if toolsets:
            body["toolsets"] = toolsets
    return body


def _post_spawn(*, port: int, token: str, body: dict) -> None:
    """POST the spawn request to the desktop app's control server.

    Raises RuntimeError with a user-facing message on any transport failure
    or non-202 response.
    """
    url = f"http://127.0.0.1:{port}/spawn"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        method="POST",
        headers={
            SPAWN_TOKEN_HEADER: token,
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
            if resp.status != 202:
                raise RuntimeError(
                    f"Desktop app returned unexpected HTTP {resp.status}."
                )
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            err_body = json.loads(exc.read().decode())
            detail = err_body.get("error") or ""
        except Exception:
            pass
        if exc.code == 401:
            raise RuntimeError(
                "Desktop app rejected the request (401 unauthorized) — the "
                "control token is stale. Restart the desktop app and try again."
            ) from exc
        if exc.code == 503:
            raise RuntimeError(
                "Desktop app has no window to run this in (HTTP 503)."
                + (f" {detail}" if detail else "")
            ) from exc
        raise RuntimeError(
            f"Desktop app returned HTTP {exc.code}" + (f": {detail}" if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        if isinstance(exc.reason, ConnectionRefusedError):
            raise RuntimeError(
                f"Could not connect to the Hermes desktop app at 127.0.0.1:{port} "
                "(connection refused). The control file is stale — the app is "
                "not running. Start it first with `hermes desktop`."
            ) from exc
        raise RuntimeError(
            f"Could not reach the Hermes desktop app at 127.0.0.1:{port}: {exc.reason}"
        ) from exc


def cmd_desktop_spawn(args) -> int:
    """Ask the running Hermes desktop app to start a new chat session.

    Exits non-zero with an actionable message on any failure (control file
    missing/corrupt, stale token, connection refused, non-202 response).
    """
    control_path = _control_file_path()
    try:
        control = _read_control_file(control_path)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    body = _spawn_request_body(args)

    try:
        _post_spawn(port=control["port"], token=str(control["token"]), body=body)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    print("✓ Sent prompt to the Hermes desktop app.")
    return 0
