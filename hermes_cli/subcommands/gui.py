"""``hermes gui`` subcommand parser.

Extracted verbatim from ``hermes_cli/main.py:main()`` (god-file Phase 2).
Handler injected to avoid importing ``main``.
"""

from __future__ import annotations

from typing import Callable


def build_gui_parser(
    subparsers, *, cmd_gui: Callable, cmd_desktop_spawn: Callable
) -> None:
    """Attach the ``gui`` subcommand to ``subparsers``."""
    # =========================================================================
    gui_parser = subparsers.add_parser(
        "desktop",
        aliases=["gui"],
        help="Build and launch the native desktop app",
        description=(
            "Launch the Hermes Electron desktop app. By default this installs "
            "workspace Node dependencies, builds the current OS's unpacked "
            "Electron app, then launches that packaged artifact."
        ),
    )
    gui_parser.add_argument(
        "--source",
        action="store_true",
        help="Launch via `electron .` against apps/desktop/dist instead of the packaged app",
    )
    gui_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build the desktop app but do not launch it (used by the installer's --update flow)",
    )
    gui_parser.add_argument(
        "--fake-boot",
        action="store_true",
        help="Enable deterministic desktop boot delays for validating startup UI",
    )
    gui_parser.add_argument(
        "--ignore-existing",
        action="store_true",
        help="Force Desktop to ignore any hermes CLI already on PATH during backend resolution",
    )
    gui_parser.add_argument(
        "--hermes-root",
        help="Override the Hermes source root used by Desktop (sets HERMES_DESKTOP_HERMES_ROOT)",
    )
    gui_parser.add_argument(
        "--cwd",
        help="Initial project directory for Desktop chat sessions (sets HERMES_DESKTOP_CWD)",
    )
    gui_parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip npm install/package and launch the existing unpacked app from apps/desktop/release",
    )
    gui_parser.add_argument(
        "--force-build",
        action="store_true",
        help="Force a full rebuild even if the content stamp matches",
    )
    gui_parser.set_defaults(func=cmd_gui)

    # `hermes desktop spawn "<prompt>"` — ask an already-running desktop app
    # to start a new chat session, via its loopback control server. Nested
    # subparser so bare `hermes desktop` (and its --build-only / --force-build
    # / etc. flags) keeps building/launching as before (set_defaults(func=
    # cmd_gui) above remains the default).
    desktop_subparsers = gui_parser.add_subparsers(dest="desktop_subcommand")
    spawn_parser = desktop_subparsers.add_parser(
        "spawn",
        help="Ask the running desktop app to start a new chat session",
        description=(
            "Send a prompt to the already-running Hermes desktop app over its "
            "local control channel. The app starts a new chat exactly as if "
            "the prompt had been typed in — streaming, transcript, and sidebar "
            "all go through the normal typed-chat path. Requires the desktop "
            "app to already be running (`hermes desktop`)."
        ),
    )
    # Optional so `--goal` can stand on its own: a goal spawn's objective is
    # already the thing to work on, and making the caller repeat it as a
    # positional would be noise. One of the two must be present — checked in
    # _spawn_request_body, which can say which one is missing and why.
    spawn_parser.add_argument(
        "prompt",
        nargs="?",
        default=None,
        help=(
            "Prompt to run in a new desktop chat session. Optional when --goal "
            "is given, in which case the objective is used as the opening turn."
        ),
    )
    spawn_parser.add_argument(
        "--goal",
        default=None,
        metavar="OBJECTIVE",
        help=(
            "Run the session against a standing objective: after each turn a "
            "judge decides whether the objective is met, and if not the app "
            "submits the next step itself. The session keeps going without "
            "anyone typing, and stops when the objective is reached, the turn "
            "budget runs out, or it stalls. Same machinery as the /goal "
            "command, set before the first turn instead of during it."
        ),
    )
    spawn_parser.add_argument(
        "--goal-turns",
        type=int,
        default=None,
        metavar="N",
        help=(
            "How many turns the goal may spend before it pauses for you "
            "(default 20, or goals.max_turns in config.yaml). Only meaningful "
            "with --goal."
        ),
    )
    spawn_parser.add_argument(
        "-m",
        "--model",
        default=None,
        help=(
            "Model for the new session, e.g. deepseek-v4-flash-0731-ds4 (used "
            "together with --provider ai-router). Omit to use whatever model "
            "the desktop app currently has selected."
        ),
    )
    spawn_parser.add_argument(
        "--provider",
        default=None,
        help="Provider for --model, e.g. ai-router. Omit to use the app's current provider.",
    )
    spawn_parser.add_argument(
        "--profile",
        default=None,
        help="Hermes profile to run the session under. Omit to use the app's active profile.",
    )
    spawn_parser.add_argument(
        "--toolsets",
        default=None,
        help=(
            "Comma-separated toolsets to enable for this session only, e.g. "
            "'file,terminal' or 'all'. Accepts built-in toolsets, plugin "
            "toolsets and enabled MCP server names. Scoped to this one chat — "
            "it does not change what your next chat gets (use `hermes tools` "
            "for that). Omit to inherit the app's current toolsets."
        ),
    )
    spawn_parser.add_argument(
        "--delegated",
        action="store_true",
        help=(
            "Run unattended: tell the session up front not to ask questions, "
            "to pick the safest reversible option when the brief leaves a "
            "choice open, and to deliver its result in the chat. If it asks "
            "anyway, the app answers the prompt itself after "
            "--delegated-timeout and notes that in the transcript, so the "
            "session cannot sit waiting for someone who is not there."
        ),
    )
    spawn_parser.add_argument(
        "--allow-command",
        action="append",
        default=None,
        metavar="NAME",
        help=(
            "Let this session run NAME without asking a human when the "
            "approval classifier is unreachable — the failure that otherwise "
            "leaves an unattended run waiting on an approval nobody answers. "
            "Repeatable. Requires --allow-command-root, which is the only "
            "directory the command may act in. Everything else keeps asking. "
            "Commands that run whatever their arguments name (sh, python, "
            "sudo, env, ssh, docker, xargs, ...) are refused."
        ),
    )
    spawn_parser.add_argument(
        "--allow-command-root",
        default=None,
        metavar="PATH",
        help=(
            "Absolute directory the --allow-command entries are scoped to, "
            "e.g. the worktree this session works in. A command that would "
            "run elsewhere, or that names a path (or a symlink) leading "
            "outside it, is not covered."
        ),
    )
    spawn_parser.add_argument(
        "--delegated-timeout",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "How long a delegated session waits for a person before answering "
            "its own question (default 120). Only meaningful with --delegated."
        ),
    )
    spawn_parser.set_defaults(func=cmd_desktop_spawn)
