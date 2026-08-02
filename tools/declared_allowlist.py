"""Declared command allowlists — what an unattended run may do without a human.

``tools/approval.py::_smart_approve`` fails CLOSED: any exception out of its
``call_llm(task="approval")`` becomes ``"escalate"``, which means "ask a
human".  That is the right default when a human is reachable.  It is the wrong
one for a delegated (unattended) run, where "ask a human" is not a decision but
a stall — in a gateway session every flagged command first burns
``approvals.timeout`` (60s by default) and is then BLOCKED, or comes straight
back as ``status: "pending_approval"`` and halts the operation.

This module holds the policy an operator can declare ahead of time so that
classifier UNAVAILABILITY degrades to a rule instead of to a stall: *in this
worktree, these commands are fine.*  Nothing here widens anything on its own.
An absent or empty declaration produces no allowlist at all, and the
fail-closed path is byte-for-byte what it was.

Matching is deliberately paranoid, because a declared allowlist is a security
boundary and the command text is written by a model that may itself be
prompt-injected:

* The raw command must first survive a CHARACTER allowlist.  Anything that
  could chain, substitute, redirect, expand or glob (``; & | < > $`` backtick
  ``( ) { } * ? [ ] ~ ! #``, backslash, newline) is refused outright, so
  ``godot --headless ; rm -rf ~`` never reaches the argv stage at all.
* What survives is parsed into argv with :mod:`shlex`, and the decision is made
  on ``argv[0]`` plus the RESOLVED working directory — never on a substring or
  regex over the raw shell text, which quoting alone defeats.
* Every remaining token is resolved against that working directory and must
  land inside the declared root — both the whole token and, for a
  ``key=value`` argument, the value half.  :meth:`pathlib.Path.resolve`
  follows symlinks, so a link that sits inside the worktree but points at
  ``/etc`` is judged on its target rather than on its name.

Refusing to allowlist a program-bearing command (``sudo``, ``env``, ``sh``,
``python`` …) is the other half of the boundary: those take the program to run
*from their own arguments*, so declaring one would silently declare every
command there is.  The same names are refused as ARGUMENTS wherever they
appear, which is what stops ``git -c core.pager=/bin/sh status`` — a declared
``git`` that runs an undeclared shell.

What none of this can constrain is a program's own escape hatches: a build tool
that runs scripts from the repo, a config flag that writes outside the tree.
That code is inside what the operator declared, and declaring any command is
always trusting that command.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, NamedTuple, Optional

logger = logging.getLogger(__name__)


# Characters a declared-allowlist candidate may contain.  An ALLOWLIST rather
# than a denylist, for the same reason ``terminal_tool._WORKDIR_SAFE_RE`` is
# one: a novel metacharacter must fail closed, not sail through.  Quotes are
# permitted because ``$`` and backtick are not — with those gone there is
# nothing left for a quoted span to hide.
_SAFE_COMMAND_CHARS_RE = re.compile(r"^[A-Za-z0-9_@%+=:,./'\"\t \-]*$")

# ``FOO=bar git status`` — a leading assignment changes the environment the
# program runs in (PATH, GIT_DIR, LD_PRELOAD …), so it is not the command that
# was declared even though argv[1] would look like it.
_ENV_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# A declared name is an executable, not a command line.
_SAFE_COMMAND_NAME_RE = re.compile(r"^[A-Za-z0-9_@%+:,./-]+$")

# Longer than any real invocation of an allowlisted tool; a giant string is a
# parser-exhaustion attempt, not a `godot` call.
_MAX_CANDIDATE_CHARS = 4096

# Commands whose entire job is to run the program named in their arguments.
# Allowlisting one of these would allowlist everything, because the decision
# this module makes (argv[0] is declared) stops meaning anything.  Mirrors the
# spirit of ``approval._COMMAND_WRAPPER_WORDS`` / ``_INTERPRETER_EXEC_FLAGS``,
# kept local and explicit so the refusal set is auditable in one place.
_PROGRAM_BEARING_COMMANDS = frozenset({
    # Wrappers: exec whatever follows.
    "builtin", "chroot", "command", "doas", "env", "eval", "exec", "ionice",
    "nice", "nohup", "nsenter", "script", "setsid", "stdbuf", "su", "sudo",
    "time", "timeout", "unshare", "watch", "xargs",
    # Shells.
    "ash", "bash", "busybox", "cmd", "csh", "dash", "fish", "ksh", "powershell",
    "pwsh", "sh", "tcsh", "zsh",
    # Interpreters with an inline-source flag.
    "bun", "deno", "lua", "node", "osascript", "perl", "php", "python",
    "python2", "python3", "rscript", "ruby", "tclsh",
    # Execution somewhere this root cannot describe.
    "docker", "kubectl", "podman", "rsync", "scp", "ssh",
})


class AllowlistDecision(NamedTuple):
    """Whether a command is covered, and the sentence explaining why.

    ``reason`` is written to be logged either way: on a match it names what was
    approved and under which root, and on a refusal it names the specific thing
    that failed, so an operator debugging a stalled unattended run can see
    which half of their declaration was wrong.
    """

    allowed: bool
    reason: str


@dataclass(frozen=True)
class DeclaredAllowlist:
    """A validated declaration: these executables, inside this directory.

    ``root`` is always a non-empty, absolute, symlink-resolved path — a
    declaration without one is refused at build time (see
    :func:`build_declared_allowlist`), because an unscoped ``git`` is a licence
    to act on any repository on the machine.
    """

    commands: frozenset[str]
    root: str

    def describe(self) -> str:
        return f"{', '.join(sorted(self.commands))} under {self.root}"


def build_declared_allowlist(
    commands: Optional[Iterable[str]],
    root: Optional[str],
    *,
    source: str = "config",
) -> Optional[DeclaredAllowlist]:
    """Validate a declaration and return it, or ``None`` if nothing is usable.

    ``None`` is the untouched-behaviour answer: no allowlist, so the caller
    keeps failing closed.  Every rejection is logged at WARNING rather than
    swallowed — a declaration that silently evaporates would look exactly like
    a working one right up until the run stalled anyway.

    Args:
        commands: Executable names the session may run.
        root: Absolute directory the commands are scoped to. Required.
        source: Where the declaration came from, for log lines
            (``"config"``, ``"session <key>"``, …).
    """
    names = []
    for raw in commands or ():
        if not isinstance(raw, str):
            logger.warning(
                "Declared command allowlist (%s): ignoring non-string entry %r.",
                source, raw,
            )
            continue
        name = raw.strip()
        if not name:
            continue
        if not _SAFE_COMMAND_NAME_RE.match(name):
            logger.warning(
                "Declared command allowlist (%s): ignoring %r — a declaration "
                "names one executable, not a command line.",
                source, name,
            )
            continue
        if "/" in name and not os.path.isabs(name):
            # A bare name is resolved by PATH; an absolute path names one
            # binary. A relative one ("./godot", "../tools/godot") means a
            # different binary in every working directory the session visits,
            # which is not something anyone can have meant to declare.
            logger.warning(
                "Declared command allowlist (%s): ignoring %r — name the "
                "executable (godot) or its absolute path (/opt/godot/bin/godot); "
                "a relative path means a different binary in every directory.",
                source, name,
            )
            continue
        if os.path.basename(name).lower() in _PROGRAM_BEARING_COMMANDS:
            logger.warning(
                "Declared command allowlist (%s): refusing %r — it runs whatever "
                "program its arguments name, so allowlisting it would allowlist "
                "every command.",
                source, name,
            )
            continue
        names.append(name)

    if not names:
        return None

    root_text = (root or "").strip() if isinstance(root, str) else ""
    if not root_text:
        logger.warning(
            "Declared command allowlist (%s): ignoring %s — no root directory "
            "was declared, and an unscoped allowlist would let these commands "
            "act anywhere on this machine.",
            source, ", ".join(names),
        )
        return None

    # Absoluteness is judged BEFORE resolving, not after: ``resolve()`` would
    # happily anchor "worktree" to whatever directory the gateway happened to
    # start in and hand back an absolute path, which is a declaration whose
    # meaning depends on where the process was launched.
    expanded_root = os.path.expanduser(root_text)
    if not os.path.isabs(expanded_root):
        logger.warning(
            "Declared command allowlist (%s): ignoring root %r — a scope must "
            "be an absolute path.", source, root_text,
        )
        return None

    try:
        resolved_root = Path(expanded_root).resolve()
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning(
            "Declared command allowlist (%s): ignoring root %r — it could not "
            "be resolved (%s).", source, root_text, exc,
        )
        return None

    if resolved_root == Path(resolved_root.anchor):
        logger.warning(
            "Declared command allowlist (%s): ignoring root %r — the filesystem "
            "root is not a scope.", source, root_text,
        )
        return None

    return DeclaredAllowlist(commands=frozenset(names), root=str(resolved_root))


def _is_within(candidate: Path, root: Path) -> bool:
    """True when *candidate* is *root* itself or lives under it."""
    return candidate == root or root in candidate.parents


def _resolve(value: str) -> Optional[Path]:
    """Fully resolve *value*, or ``None`` if it cannot be resolved at all.

    ``resolve()`` walks symlinks, which is the point: a name inside the
    worktree that leads somewhere else must be judged on where it leads.
    ``None`` is always a refusal for the caller.
    """
    try:
        return Path(value).resolve()
    except (OSError, RuntimeError, ValueError):
        return None


def _resolve_argument(value: str, cwd: Path) -> Optional[Path]:
    """Resolve *value* the way the command's child process would see it."""
    try:
        candidate = Path(value)
    except ValueError:
        return None
    if not candidate.is_absolute():
        candidate = cwd / candidate
    return _resolve(str(candidate))


def _argument_values(token: str) -> list[str]:
    """Return every part of *token* a program might read as a path or a program.

    A bare flag (``--headless``) names nothing.  Anything else can name
    something twice over, because ``key=value`` arguments are everywhere and
    the value half is what the program acts on:

    * ``--git-dir=/etc/git`` → ``/etc/git``
    * ``core.pager=/bin/sh`` → BOTH the whole token (it could be a filename)
      and ``/bin/sh``.  Checking only the whole token is what let
      ``git -c core.pager=/bin/sh status`` through: as a relative name it sits
      innocently under the working directory, while the half git actually runs
      is an absolute path to a shell.
    """
    if token.startswith("-"):
        if "=" not in token:
            return []
        tail = token.split("=", 1)[1]
        return [tail] if tail else []

    values = [token] if token else []
    if "=" in token:
        tail = token.split("=", 1)[1]
        if tail:
            values.append(tail)
    return values


def match_declared_command(
    command: str,
    allowlist: Optional[DeclaredAllowlist],
    cwd: Optional[str],
) -> AllowlistDecision:
    """Decide whether *command*, run in *cwd*, is covered by *allowlist*.

    Every ``False`` here means "keep doing what we do today" — escalate to a
    human — so a refusal is never worse than the current behaviour.  The order
    of the checks matters: the raw-text screen runs before parsing, so a
    command that chains or substitutes is rejected on its syntax and never gets
    the chance to be judged on a benign-looking ``argv[0]``.
    """
    if allowlist is None:
        return AllowlistDecision(False, "no command allowlist is declared")

    text = command or ""
    if not text.strip():
        return AllowlistDecision(False, "the command is empty")
    if len(text) > _MAX_CANDIDATE_CHARS:
        return AllowlistDecision(
            False,
            f"the command is longer than {_MAX_CANDIDATE_CHARS} characters",
        )
    if not _SAFE_COMMAND_CHARS_RE.match(text):
        return AllowlistDecision(
            False,
            "it contains shell syntax that can change what runs — chaining, "
            "command substitution, redirection, expansion or globbing",
        )

    try:
        argv = shlex.split(text, posix=True)
    except ValueError:
        return AllowlistDecision(
            False, "it could not be parsed into arguments (unbalanced quoting)"
        )
    if not argv:
        return AllowlistDecision(False, "it parsed to no arguments at all")

    executable = argv[0]
    if _ENV_ASSIGNMENT_RE.match(executable):
        return AllowlistDecision(
            False,
            "it begins with an environment assignment, which changes what the "
            "declared program would actually do",
        )
    if executable not in allowlist.commands:
        return AllowlistDecision(
            False, f"{executable!r} is not one of the declared commands"
        )

    root = Path(allowlist.root)

    cwd_text = (cwd or "").strip()
    if not cwd_text:
        return AllowlistDecision(
            False,
            "no working directory could be resolved, so the declared root "
            f"{allowlist.root} cannot be enforced",
        )
    if not os.path.isabs(cwd_text):
        # Anchoring a relative working directory to anything would be a guess,
        # and guessing it into the root is the one guess that wrongly ALLOWS.
        return AllowlistDecision(
            False,
            f"the working directory {cwd_text!r} is not an absolute path, so "
            f"the declared root {allowlist.root} cannot be enforced",
        )
    resolved_cwd = _resolve(cwd_text)
    if resolved_cwd is None:
        return AllowlistDecision(
            False, f"the working directory {cwd!r} could not be resolved"
        )
    if not _is_within(resolved_cwd, root):
        return AllowlistDecision(
            False,
            f"it would run in {resolved_cwd}, which is outside the declared "
            f"root {allowlist.root}",
        )

    for token in argv[1:]:
        for value in _argument_values(token):
            if os.path.basename(value).lower() in _PROGRAM_BEARING_COMMANDS:
                # An argument that NAMES a shell or interpreter is how a
                # declared command becomes an undeclared one:
                # ``git -c core.pager=/bin/sh status`` runs sh, and the
                # allowlist said git. Refused wherever it appears, for the same
                # reason such a name cannot be declared in the first place.
                return AllowlistDecision(
                    False,
                    f"the argument {token[:120]!r} names {value!r}, a program "
                    "that runs whatever it is handed",
                )
            # Every remaining value is treated as a possible path, including
            # bare names: ``git add link`` where ``link`` is a symlink out of
            # the worktree is exactly the case a "does it look like a path?"
            # test would wave through. A value that is genuinely not a path
            # resolves under the working directory, which is already inside
            # the root.
            resolved = _resolve_argument(value, resolved_cwd)
            if resolved is None or not _is_within(resolved, root):
                return AllowlistDecision(
                    False,
                    f"the argument {token[:120]!r} resolves outside the declared "
                    f"root {allowlist.root}",
                )

    return AllowlistDecision(
        True,
        f"{executable!r} is declared for this session and everything it names "
        f"stays inside {allowlist.root}",
    )
