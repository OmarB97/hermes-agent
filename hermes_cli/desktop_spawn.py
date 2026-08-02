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
resulting session: the desktop app owns it from here. That is precisely why
the provider override is checked before the POST — see
:func:`_provider_refusal` for why a bad one cannot be reported back after it.

``--delegated`` marks the spawn unattended. The behaviour that follows from
that — the contract prepended to the prompt, and answering a clarify prompt
nobody is there to answer — belongs to the app
(``apps/desktop/src/lib/delegated-spawn.ts``); this side only carries the
flag and, optionally, how long to wait.

``--goal`` gives the session a standing objective instead of a single errand.
The loop that pursues it already exists (``hermes_cli/goals.py``) and has
always been reachable as ``/goal <text>``; what this flag adds is setting it at
session creation, so the goal is a property of the session rather than of its
first message. The gateway binds it in ``session.create``, exactly as it binds
model and toolsets. Composes with ``--delegated`` and ``--profile``: an
unattended goal session is the case this was built for.
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


def _unresolvable_provider(provider: str) -> str | None:
    """Return why ``provider`` cannot run here, or None if it resolves.

    Provider names are per-profile vocabulary, not global: ``providers:`` and
    ``custom_providers:`` live in each profile's own ``config.yaml``. Observed
    2026-08-02: ``--provider ai-router`` (the root profile's own provider)
    against ``meshboard-game-dev``, whose only provider is
    ``meshboard-qualified-local``, died in agent init with *"Unknown provider
    'ai-router'"* and left a 0-message session behind.

    Asks the backend's own question with the backend's own resolver: this
    process already runs under the target profile's ``HERMES_HOME`` (the
    global pre-parse rewrote it before argparse), and
    ``resolve_runtime_provider`` is what the agent calls at turn start.

    Only an ``invalid_provider`` failure counts. "Known provider, missing
    credentials" (``missing_api_key``, or the untyped "No <vendor> credentials
    found") is a different problem with its own in-app onboarding prompt, and
    this process may not see credentials the backend can — treating those as
    unresolvable would break spawns that work today. A resolver bug likewise
    reports nothing: the turn still gets its own chance to resolve for real.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider

    try:
        resolve_runtime_provider(requested=provider)
    except AuthError as exc:
        if getattr(exc, "code", None) != "invalid_provider":
            return None
        return str(exc)
    except Exception:
        return None
    return None


def _provider_refusal(provider: str, profile: str, detail: str) -> str:
    """Message for a PINNED spawn whose provider cannot run in that profile.

    Hard-fails because this is the one case the check is sound for: the body
    pins ``profile``, so the session provably runs under the very config that
    was just read.

    It has to be caught before the POST because the control channel cannot
    report it afterwards. ``POST /spawn`` answers 202 as soon as Electron main
    holds the request — ``deliverSpawnToRenderer`` reports "delivered" for one
    it has merely parked — and there is no reply channel from the renderer
    back to the CLI. So the CLI's ``✓`` only ever meant "the app accepted
    this", never "the turn ran".

    What the doomed spawn left behind is a 0-message row: ``session.create``
    deliberately persists nothing (``tui_gateway/server.py``), so the row is
    written by the first ``prompt.submit`` — a row with no messages *is* a
    turn that started and failed. Refusing here creates no session at all.
    """
    return (
        f"--provider {provider!r} is not a provider in profile {profile!r}, "
        f"so the turn would fail in agent init:\n\n  {detail}\n\n"
        "Provider names come from that profile's own config.yaml (providers: "
        "/ custom_providers:), so a name that works in another profile can be "
        "unknown here. Nothing was sent — no session was created."
    )


def _provider_caveat(provider: str, detail: str) -> str:
    """Message for an UNPINNED spawn whose provider looks unresolvable.

    Names ``default`` as the profile it read, because an unpinned spawn is the
    ROOT home by definition (see :func:`_requested_profile`) and the root home
    *is* the default profile — ``get_profile_dir("default")`` returns it —
    whatever that root path happens to be.

    Warns instead of refusing, because here the check is NOT sound in either
    direction. An unpinned spawn is routed by the app, and the renderer
    resolves an omitted profile against its own live gateway selection — which
    moves whenever an earlier spawn pinned one. So this process read the wrong
    config, and it cannot tell which way:

    * measured in the dev sandbox 2026-08-02 — with the app launched on
      ``default``, an earlier ``--profile labtwo`` spawn left the live gateway
      on ``labtwo``; a later unpinned ``--provider lab-router`` (valid in
      ``default``, so it checked clean) then ran under ``labtwo`` and died on
      *"Unknown provider 'lab-router'"*.
    * the mirror case is a provider declared ONLY in the profile the app is
      on. Refusing that would block a spawn that would have worked — a
      false refusal is worse than the husk, because the husk at least ran.

    So say what was read and what could not be known, and still send it.
    """
    return (
        f"--provider {provider!r} is not a provider in profile 'default', "
        f"which is the config this command could read:\n"
        f"\n  {detail}\n\n"
        "This spawn pins no profile, so the app runs it under whichever "
        "profile it is on — if that is a profile where the name IS defined, "
        "it will work. Pass --profile to check and pin the same one."
    )


def _spawn_destination_note(profile: str | None) -> str:
    """Return the ``(profile: …)`` fragment for the success line.

    Always says something. A pinned spawn names its profile; an unpinned one
    is routed by the app, and saying so is the only way the CLI can flag the
    second seam the operator hit — the app's window showing one profile while
    an unpinned spawn ran under another. Claiming a profile this side merely
    guessed would be worse than admitting the app decides.
    """
    if profile:
        return f" (profile: {profile})"
    return " (profile: the app's active one — pass --profile to pin it)"


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

    Everything here is bound per-session by ``session.create`` on the gateway:
    model/provider/profile/toolsets steer this one chat and never write config,
    so a spawn cannot change what the user's next chat gets.

    Raises RuntimeError for flag combinations argparse cannot express, and for
    a ``--toolsets`` value that resolves to nothing.
    """
    goal = (getattr(args, "goal", None) or "").strip()
    prompt = (getattr(args, "prompt", None) or "").strip()

    # The objective is already a statement of what to do, so a goal spawn does
    # not need a separate opening prompt — this mirrors `/goal <text>`, which
    # sets the goal and then submits that same text as the kickoff turn. A
    # caller who wants a different first move can still pass both.
    if goal and not prompt:
        prompt = goal
    if not prompt:
        raise RuntimeError(
            "nothing to send: pass a prompt, or --goal with the objective to "
            "work toward."
        )

    body: dict = {"prompt": prompt}
    if goal:
        body["goal"] = goal

    goal_turns = getattr(args, "goal_turns", None)
    # A turn budget with no goal would be read by nobody. Say so rather than
    # accepting the flag and quietly running an ordinary one-shot session.
    if goal_turns is not None and not goal:
        raise RuntimeError(
            "--goal-turns only applies to a goal spawn. Add --goal, or drop "
            "the turn budget."
        )
    if goal_turns is not None:
        if goal_turns < 1:
            raise RuntimeError(
                f"--goal-turns must be at least 1 (got {goal_turns})."
            )
        body["goalMaxTurns"] = goal_turns

    for key in ("model", "provider"):
        value = getattr(args, key, None)
        if value:
            body[key] = value

    profile = _requested_profile(args)
    if profile:
        body["profile"] = profile

    # Split here rather than shipping the raw string: the wire contract is a
    # list of names, and the gateway validates them against the same vocabulary
    # HERMES_TUI_TOOLSETS accepts. A value that is all separators is a typo, not
    # a request to inherit — say so instead of silently spawning with the app's
    # toolsets, which is the failure this flag already had once (#315).
    raw_toolsets = getattr(args, "toolsets", None)
    if raw_toolsets is not None:
        toolsets = [t.strip() for t in str(raw_toolsets).split(",") if t.strip()]
        if not toolsets:
            raise RuntimeError(
                f"--toolsets got no usable names (got {raw_toolsets!r}). Pass a "
                "comma-separated list like 'file,terminal', or drop the flag to "
                "inherit the app's toolsets."
            )
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

    # Check the provider override BEFORE the POST, never after: the 202 is
    # unconditional and there is no channel to retract it on. A pinned spawn
    # is refused outright, because only then does the config just read provably
    # govern the turn; an unpinned one can only be warned about.
    provider = str(body.get("provider") or "")
    pinned = body.get("profile")
    if provider and (detail := _unresolvable_provider(provider)):
        if pinned:
            print(f"✗ {_provider_refusal(provider, str(pinned), detail)}")
            sys.exit(1)
        print(f"⚠ {_provider_caveat(provider, detail)}")

    try:
        _post_spawn(port=control["port"], token=str(control["token"]), body=body)
    except RuntimeError as exc:
        print(f"✗ {exc}")
        sys.exit(1)

    # The session opens in the running app's window whichever profile it runs
    # under, so this line is the only place that distinction is visible up front.
    where = _spawn_destination_note(body.get("profile"))
    notes = []
    if body.get("delegated"):
        notes.append("delegated — it will not stop to ask")
    if body.get("goal"):
        budget = body.get("goalMaxTurns")
        notes.append(
            "goal — it keeps taking the next step on its own"
            + (f", up to {budget} turns" if budget else "")
        )
    suffix = f" ({'; '.join(notes)})" if notes else ""
    print(f"✓ Sent prompt to the Hermes desktop app{where}{suffix}.")
    return 0
