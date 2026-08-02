"""``hermes desktop spawn`` — ask the running desktop app to start a session.

The Hermes Electron app starts a small loopback HTTP control server on
launch (see ``apps/desktop/electron/spawn-control.ts``) and publishes its
port + a per-launch token in a 0600 file at
``<root>/desktop/control.json``. This module reads that file and POSTs the
prompt to ``/spawn``, which hands it to the renderer to start a new chat
exactly as if it had been typed in — streaming, transcript, sidebar all go
through the existing typed-chat path.

One app process serves every profile, so there is exactly one control
channel per machine and it lives at the ROOT home — see
:func:`_control_file_path`. Which profile the session runs under is a
parameter of the request, not a property of the channel: the app already
carries ``profile`` down to ``session.create``, which binds the session to
that profile's home. See :func:`_requested_profile` for why that name has
to be recovered from ``HERMES_HOME`` rather than read off ``args``.

This CLI command is a one-shot fire-and-forget POST, not a client for the
resulting session: the desktop app owns it from here.

``--delegated`` marks the spawn unattended. The behaviour that follows from
that — the contract prepended to the prompt, and answering a clarify prompt
nobody is there to answer — belongs to the app
(``apps/desktop/src/lib/delegated-spawn.ts``); this side only carries the
flag and, optionally, how long to wait.
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
    """Return the desktop control file path under the ROOT Hermes home.

    The app publishes this file at the root, never under the active profile:
    Electron main puts every ``HERMES_HOME`` through
    ``normalizeHermesHomeRoot`` (``apps/desktop/electron/backend-env.ts``),
    which maps ``<root>/profiles/<name>`` back to ``<root>``. A single app
    process owns all the profiles, so there is one control channel and it
    belongs to the machine rather than to a profile.

    Resolving this against ``get_hermes_home()`` instead pointed at
    ``<root>/profiles/<name>/desktop/control.json``, which nothing writes —
    so every profile-scoped invocation (``--profile <p>``, or any spawn at
    all once ``hermes profile use`` had made a profile sticky) reported the
    desktop app as not running. ``get_default_hermes_root()`` is the Python
    twin of ``normalizeHermesHomeRoot`` and keeps Docker and custom homes
    resolving the same way on both sides.
    """
    from hermes_constants import get_default_hermes_root

    return get_default_hermes_root() / "desktop" / "control.json"


def _requested_profile(args) -> str | None:
    """Return the profile this spawn should run under, or None for the app's.

    ``--profile`` does not survive to argparse: ``_apply_profile_override``
    in ``hermes_cli/main.py`` scans raw ``sys.argv`` before any parser runs,
    rewrites ``HERMES_HOME`` to the named profile's directory, and strips the
    flag so the parser never sees it. ``args.profile`` is therefore always
    ``None`` for a real command line, and the resolved home is where the
    requested name actually survives — recover it from there.

    A sticky ``active_profile`` (from ``hermes profile use``) resolves the
    same way, so a bare ``hermes desktop spawn`` follows the profile the user
    already selected instead of silently landing on the app's.

    ``args.profile`` still wins when a caller sets it directly, so the flag
    stays meaningful if the pre-parse ever stops consuming it.
    """
    explicit = getattr(args, "profile", None)
    if explicit:
        return str(explicit).strip() or None

    from hermes_constants import get_hermes_home

    home = get_hermes_home()

    # Mirrors normalizeHermesHomeRoot's test: a profile home is the child of
    # a directory literally named "profiles". Anything else is the root, and
    # the root means "whatever profile the app is on".
    if home.parent.name == "profiles":
        return home.name
    return None


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
    """Build the JSON body for POST /spawn, omitting unset optional fields.

    Raises RuntimeError for flag combinations argparse cannot express.
    """
    body: dict = {"prompt": args.prompt}
    for key in ("model", "provider"):
        value = getattr(args, key, None)
        if value:
            body[key] = value

    profile = _requested_profile(args)
    if profile:
        body["profile"] = profile

    raw_toolsets = getattr(args, "toolsets", None)
    if raw_toolsets:
        toolsets = [t.strip() for t in raw_toolsets.split(",") if t.strip()]
        if toolsets:
            body["toolsets"] = toolsets

    delegated = bool(getattr(args, "delegated", False))
    timeout_seconds = getattr(args, "delegated_timeout", None)

    # A timeout with no --delegated would be read by nobody. Say so rather than
    # accepting the flag and quietly running an attended session.
    if timeout_seconds is not None and not delegated:
        raise RuntimeError(
            "--delegated-timeout only applies to a delegated spawn. Add "
            "--delegated, or drop the timeout."
        )
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise RuntimeError(
            f"--delegated-timeout must be greater than 0 seconds (got {timeout_seconds})."
        )

    if delegated:
        body["delegated"] = True
        # Seconds on the command line because that is how people say it;
        # milliseconds on the wire because that is what every consumer of it
        # downstream (the renderer's timer) actually wants.
        if timeout_seconds is not None:
            body["delegatedTimeoutMs"] = int(round(timeout_seconds * 1000))
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

    try:
        body = _spawn_request_body(args)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    try:
        _post_spawn(port=control["port"], token=str(control["token"]), body=body)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    # Name the profile when one was requested: the session opens in the
    # running app's window whichever profile it runs under, so the transcript
    # is the only other place that distinction shows up.
    where = f" (profile: {body['profile']})" if body.get("profile") else ""
    if body.get("delegated"):
        print(
            f"✓ Sent prompt to the Hermes desktop app{where} "
            "(delegated — it will not stop to ask)."
        )
    else:
        print(f"✓ Sent prompt to the Hermes desktop app{where}.")
    return 0
