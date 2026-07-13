"""Profile distributions — shareable, packaged Hermes profiles via git.

A distribution is a Hermes profile published as a git repository (or
installed from a local directory for development). Install with one command
from a git URL, update in place, and keep your local memories / sessions /
credentials untouched.

Where this fits relative to the existing pieces:

* ``hermes profile export/import`` — local backup / restore for a profile
  on your own machine. NOT a distribution format. Stays as-is.
* ``hermes skills install <url>`` — the URL install pattern we're mirroring,
  but at the profile granularity.

Subcommands (all live under ``hermes profile``, not a parallel tree):

    hermes profile install <source> [--name N] [--alias] [--force] [--yes]
    hermes profile update  <name>  [--force-config] [--yes]
    hermes profile info    <name>

``<source>`` is one of:

* A git URL (``github.com/user/repo``, ``https://github.com/...``, ``git@...``,
  ``ssh://``, ``git://``), optionally with ``#<ref>`` to pin a tag / branch /
  commit SHA.
* A local directory that already contains ``distribution.yaml`` — used
  during profile development before the first push.

Manifest format (``distribution.yaml`` at the profile root)::

    name: telemetry
    version: 0.1.0
    description: "Compliance monitoring harness"
    hermes_requires: ">=0.18.2"
    hermes_capabilities:
      - explicit-fallback-policy-v1
      - transactional-profile-distribution-v1
    author: "..."
    license: "..."
    env_requires:
      - name: OPENAI_API_KEY
        description: "OpenAI API key"
        required: true
      - name: GRAPHITI_MCP_URL
        description: "Memory graph URL"
        required: false
        default: "http://127.0.0.1:8000/sse"
    distribution_owned:      # optional; sensible defaults apply
      - SOUL.md
      - skills/
      - cron/
      - mcp.json

Update semantics:

* Distribution-owned paths (SOUL.md, mcp.json, skills/, cron/,
  distribution.yaml) are replaced from the new source.
* ``config.yaml`` is distribution-owned but preserved on update unless
  ``--force-config`` is passed (user overrides typically live here).
* User-owned paths (memories/, sessions/, state.db, auth.json, .env,
  logs/, workspace/, home/, plans/, *_cache/, and anything under
  ``local/``) are never touched.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Dict, List, Optional, Tuple

from agent.skill_utils import is_excluded_skill_path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MANIFEST_FILENAME = "distribution.yaml"
ENV_TEMPLATE_FILENAME = ".env.template"
ENV_EXAMPLE_FILENAME = ".env.EXAMPLE"
RECEIPT_FILENAME = ".distribution-receipt.json"
ROLLBACK_RECEIPT_FILENAME = "rollback-receipt.json"
LOCK_DIRNAME = ".distribution-locks"
BACKUP_DIRNAME = ".distribution-backups"

# Distribution manifests may require behavioral contracts that cannot be
# represented honestly by the package version alone.  Keep these stable and
# additive: a manifest declaring an unknown capability must fail before any
# profile payload is written.
SUPPORTED_DISTRIBUTION_CAPABILITIES: frozenset[str] = frozenset({
    "explicit-fallback-policy-v1",
    "transactional-profile-distribution-v1",
})

# Default distribution-owned paths (relative to profile root).  Authors may
# override via ``distribution_owned:`` in the manifest.  config.yaml is
# distribution-owned but treated specially on update (see _is_config_like).
DEFAULT_DIST_OWNED: Tuple[str, ...] = (
    "SOUL.md",
    "config.yaml",
    "mcp.json",
    "skills",
    "cron",
    MANIFEST_FILENAME,
)

# Paths that are NEVER part of a distribution. These are user-owned and are
# protected on update. Must stay consistent with
# ``profiles.py::_DEFAULT_EXPORT_EXCLUDE_ROOT`` plus the ``local/``
# convention for user customizations.
USER_OWNED_EXCLUDE: frozenset = frozenset({
    # Credentials & runtime secrets
    "auth.json", ".env",
    # Databases & runtime state
    "state.db", "state.db-shm", "state.db-wal",
    "hermes_state.db", "response_store.db",
    "response_store.db-shm", "response_store.db-wal",
    "gateway.pid", "gateway_state.json", "processes.json",
    "auth.lock", "active_profile", ".update_check",
    "errors.log", ".hermes_history",
    # User data
    "memories", "sessions", "logs", "plans", "workspace", "home",
    "image_cache", "audio_cache", "document_cache",
    "browser_screenshots", "checkpoints", "sandboxes",
    "backups", "cache",
    # Infrastructure
    "hermes-agent", ".worktrees", "profiles", "bin", "node_modules",
    # User customization namespace
    "local",
})

_INTERNAL_DISTRIBUTION_PATHS: frozenset = frozenset({
    RECEIPT_FILENAME,
    ROLLBACK_RECEIPT_FILENAME,
    LOCK_DIRNAME,
    BACKUP_DIRNAME,
})

_WINDOWS_RESERVED_NAMES: frozenset = frozenset({
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})


def _normalize_owned_path(raw: Any) -> str:
    """Validate and normalize one manifest-owned relative path.

    Distribution manifests are untrusted input.  Refuse absolute paths,
    traversal, Windows separators, and any path rooted in Hermes-owned user
    state.  The old parser stripped leading slashes, which could silently turn
    ``/etc`` into the apparently-safe relative path ``etc``.
    """
    text = str(raw).strip()
    if not text:
        raise DistributionError("distribution_owned entries cannot be empty")
    windows_path = PureWindowsPath(text)
    if "\\" in text or windows_path.is_absolute() or windows_path.drive:
        raise DistributionError(
            f"distribution_owned path must use a safe relative POSIX path: {text!r}"
        )
    if text.startswith("/"):
        raise DistributionError(
            f"distribution_owned path must be relative: {text!r}"
        )
    text = text.rstrip("/")
    parts = text.split("/")
    if not text or any(part in {"", ".", ".."} for part in parts):
        raise DistributionError(
            f"distribution_owned path contains traversal or empty segments: {raw!r}"
        )
    if any(
        part.endswith((".", " "))
        or ":" in part
        or part.split(".", 1)[0].casefold() in _WINDOWS_RESERVED_NAMES
        for part in parts
    ):
        raise DistributionError(
            f"distribution_owned path has a platform-ambiguous segment: {text!r}"
        )
    path = PurePosixPath(text)
    if path.is_absolute():
        raise DistributionError(
            f"distribution_owned path must be relative: {text!r}"
        )
    root = path.parts[0]
    root_folded = root.casefold()
    if root_folded in {entry.casefold() for entry in USER_OWNED_EXCLUDE}:
        raise DistributionError(
            f"distribution_owned path targets user-owned state: {text!r}"
        )
    if root_folded in {
        entry.casefold() for entry in _INTERNAL_DISTRIBUTION_PATHS
    }:
        raise DistributionError(
            f"distribution_owned path is reserved for Hermes: {text!r}"
        )
    return path.as_posix()


def _dedupe_owned_paths(paths: List[str]) -> List[str]:
    """Return deterministic roots, dropping entries already covered by a parent."""
    result: List[str] = []
    for rel in sorted(set(paths), key=lambda value: (value.count("/"), value)):
        if any(rel == parent or rel.startswith(parent + "/") for parent in result):
            continue
        result.append(rel)
    return sorted(result)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DistributionError(Exception):
    """Raised for distribution install/update failures."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class EnvRequirement:
    name: str
    description: str = ""
    required: bool = True
    default: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Any) -> "EnvRequirement":
        if not isinstance(data, dict):
            raise DistributionError(
                f"env_requires entry must be a mapping, got {type(data).__name__}"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError("env_requires entry missing 'name'")
        return cls(
            name=name,
            description=str(data.get("description") or ""),
            required=bool(data.get("required", True)),
            default=data.get("default"),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"name": self.name, "description": self.description}
        if not self.required:
            out["required"] = False
        if self.default is not None:
            out["default"] = self.default
        return out


@dataclass
class DistributionManifest:
    name: str
    version: str = "0.1.0"
    description: str = ""
    hermes_requires: str = ""
    hermes_capabilities: List[str] = field(default_factory=list)
    author: str = ""
    license: str = ""
    env_requires: List[EnvRequirement] = field(default_factory=list)
    distribution_owned: List[str] = field(default_factory=list)
    # Tracked after install — where we pulled from, so ``update`` can re-pull.
    source: str = ""
    # ISO-8601 UTC timestamp written on install / update, so ``info`` and
    # ``list`` can show when a distribution landed on disk.  Empty for
    # manifests that ship in a repo (authors don't populate this).
    installed_at: str = ""

    @classmethod
    def from_dict(cls, data: Any) -> "DistributionManifest":
        if not isinstance(data, dict):
            raise DistributionError(
                f"{MANIFEST_FILENAME} must be a mapping, got {type(data).__name__}"
            )
        name = str(data.get("name") or "").strip()
        if not name:
            raise DistributionError(f"{MANIFEST_FILENAME} missing 'name'")
        env_raw = data.get("env_requires") or []
        if not isinstance(env_raw, list):
            raise DistributionError("env_requires must be a list")
        env_requires = [EnvRequirement.from_dict(e) for e in env_raw]
        capabilities_raw = data.get("hermes_capabilities")
        if capabilities_raw is None:
            capabilities_raw = []
        if not isinstance(capabilities_raw, list):
            raise DistributionError("hermes_capabilities must be a list")
        hermes_capabilities: List[str] = []
        for raw_capability in capabilities_raw:
            if not isinstance(raw_capability, str) or not raw_capability.strip():
                raise DistributionError(
                    "hermes_capabilities entries must be non-empty strings"
                )
            capability = raw_capability.strip()
            if capability in hermes_capabilities:
                raise DistributionError(
                    f"hermes_capabilities contains duplicate entry: {capability!r}"
                )
            hermes_capabilities.append(capability)
        dist_owned_raw = data.get("distribution_owned") or []
        if dist_owned_raw and not isinstance(dist_owned_raw, list):
            raise DistributionError("distribution_owned must be a list")
        distribution_owned = [
            _normalize_owned_path(path)
            for path in dist_owned_raw
            if str(path).strip()
        ]
        return cls(
            name=name,
            version=str(data.get("version") or "0.1.0"),
            description=str(data.get("description") or ""),
            hermes_requires=str(data.get("hermes_requires") or ""),
            hermes_capabilities=hermes_capabilities,
            author=str(data.get("author") or ""),
            license=str(data.get("license") or ""),
            env_requires=env_requires,
            distribution_owned=distribution_owned,
            source=str(data.get("source") or ""),
            installed_at=str(data.get("installed_at") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
        }
        if self.description:
            out["description"] = self.description
        if self.hermes_requires:
            out["hermes_requires"] = self.hermes_requires
        if self.hermes_capabilities:
            out["hermes_capabilities"] = self.hermes_capabilities
        if self.author:
            out["author"] = self.author
        if self.license:
            out["license"] = self.license
        if self.env_requires:
            out["env_requires"] = [e.to_dict() for e in self.env_requires]
        if self.distribution_owned:
            out["distribution_owned"] = self.distribution_owned
        if self.source:
            out["source"] = self.source
        if self.installed_at:
            out["installed_at"] = self.installed_at
        return out

    def owned_paths(self) -> List[str]:
        """Resolve which paths count as distribution-owned."""
        if self.distribution_owned:
            return [_normalize_owned_path(path) for path in self.distribution_owned]
        return [_normalize_owned_path(path) for path in DEFAULT_DIST_OWNED]


def _load_yaml(text: str) -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover — pyyaml is a hard dep
        raise DistributionError("PyYAML is required for distribution manifests") from exc
    return yaml.safe_load(text)


def _dump_yaml(data: Any) -> str:
    import yaml

    return yaml.safe_dump(data, sort_keys=False, default_flow_style=False)


def read_manifest(profile_dir: Path) -> Optional[DistributionManifest]:
    """Return the manifest for *profile_dir*, or None if it isn't a distribution."""
    mf_path = profile_dir / MANIFEST_FILENAME
    if not mf_path.is_file():
        return None
    try:
        data = _load_yaml(mf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise DistributionError(f"Failed to parse {mf_path}: {exc}") from exc
    return DistributionManifest.from_dict(data or {})


def write_manifest(profile_dir: Path, manifest: DistributionManifest) -> Path:
    mf_path = profile_dir / MANIFEST_FILENAME
    mf_path.write_text(_dump_yaml(manifest.to_dict()), encoding="utf-8")
    return mf_path


# ---------------------------------------------------------------------------
# Version check
# ---------------------------------------------------------------------------


_VERSION_OP_RE = re.compile(r"^\s*(>=|<=|==|!=|>|<)\s*(.+?)\s*$")


def _parse_semver(v: str) -> Tuple[int, int, int]:
    """Very small semver parser — major.minor.patch only.  Extra labels stripped."""
    s = str(v).strip().lstrip("v")
    # Strip any pre-release / build metadata (e.g. "0.12.0-rc1+abc")
    s = re.split(r"[-+]", s, 1)[0]
    parts = s.split(".")
    while len(parts) < 3:
        parts.append("0")
    try:
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError as exc:
        raise DistributionError(f"Unparseable version: {v!r}") from exc


def check_hermes_requires(spec: str, current_version: str) -> None:
    """Raise DistributionError if ``current_version`` does not satisfy ``spec``.

    ``spec`` accepts a single comparator (``>=0.12.0``, ``==0.12.0``, etc.).
    Empty or blank spec is a no-op — no requirement.
    """
    if not spec or not spec.strip():
        return
    m = _VERSION_OP_RE.match(spec)
    if not m:
        # Bare version → treat as ``>=``
        op, target = ">=", spec.strip()
    else:
        op, target = m.group(1), m.group(2)
    cur = _parse_semver(current_version)
    tgt = _parse_semver(target)
    ok = {
        ">=": cur >= tgt,
        "<=": cur <= tgt,
        "==": cur == tgt,
        "!=": cur != tgt,
        ">":  cur > tgt,
        "<":  cur < tgt,
    }[op]
    if not ok:
        raise DistributionError(
            f"This distribution requires Hermes {op}{target}, "
            f"but you have {current_version}."
        )


def check_hermes_capabilities(required: List[str]) -> None:
    """Reject manifests that require behavioral contracts this build lacks."""
    unsupported = sorted(
        set(required).difference(SUPPORTED_DISTRIBUTION_CAPABILITIES)
    )
    if unsupported:
        supported = ", ".join(sorted(SUPPORTED_DISTRIBUTION_CAPABILITIES))
        raise DistributionError(
            "This distribution requires unsupported Hermes capabilities: "
            f"{', '.join(unsupported)}. Supported by this build: {supported}."
        )


# ---------------------------------------------------------------------------
# Env var template helper
# ---------------------------------------------------------------------------


def _env_template_from_manifest(manifest: DistributionManifest) -> str:
    """Generate a ``.env.template`` body from env_requires."""
    lines = [
        "# Environment variables required by this Hermes distribution.",
        "# Copy to `.env` and fill in your own values before running.",
        "",
    ]
    for req in manifest.env_requires:
        if req.description:
            lines.append(f"# {req.description}")
        status = "required" if req.required else "optional"
        lines.append(f"# ({status})")
        default_val = req.default if req.default is not None else ""
        prefix = "" if req.required else "# "
        lines.append(f"{prefix}{req.name}={default_val}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Source staging — git clone or local directory
# ---------------------------------------------------------------------------


def _looks_like_git_url(s: str) -> bool:
    s = s.strip()
    if s.endswith(".git"):
        return True
    if s.startswith(("git@", "ssh://", "git://")):
        return True
    if s.startswith(("http://", "https://")):
        # Any http(s) URL is treated as a git repo.  We no longer accept
        # tar.gz URLs — git is the only remote transport.
        return True
    # Bare github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", s):
        return True
    return False


def _git_clone(url: str, dest: Path) -> None:
    # Normalize github.com/user/repo shorthand
    if re.match(r"^github\.com/[\w.-]+/[\w.-]+/?$", url):
        url = f"https://{url.rstrip('/')}"
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(dest)],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise DistributionError("git is required for git-URL installs") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise DistributionError(f"git clone failed: {stderr.strip()}") from exc


def _stage_source(source: str, workdir: Path) -> Tuple[Path, str]:
    """Resolve *source* to a local directory containing distribution.yaml.

    Returns ``(staged_dir, provenance)`` where ``provenance`` is stored in the
    installed manifest's ``source:`` field so ``hermes profile update`` can
    re-pull from the same place.

    Accepts:
      * A git URL (https / ssh / git@ / bare github.com shorthand) — cloned
        into a temp directory; ``.git`` removed after clone.
      * A local directory already containing ``distribution.yaml``.
    """
    src_str = source.strip()

    # Git URL
    if _looks_like_git_url(src_str):
        cloned = workdir / "clone"
        _git_clone(src_str, cloned)
        # Remove .git to keep the staged tree clean
        shutil.rmtree(cloned / ".git", ignore_errors=True)
        if not (cloned / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} at the root of {src_str!r}. "
                "This repository is not a Hermes profile distribution."
            )
        return cloned, src_str

    # Local directory
    path_guess = Path(src_str).expanduser()
    if path_guess.is_dir():
        if not (path_guess / MANIFEST_FILENAME).is_file():
            raise DistributionError(
                f"No {MANIFEST_FILENAME} in {path_guess}. "
                "A local-directory source must contain a distribution.yaml at its root."
            )
        return path_guess.resolve(), str(path_guess.resolve())

    raise DistributionError(
        f"Cannot resolve distribution source: {source!r}. "
        "Expected a git URL (e.g. github.com/user/repo) or a local directory."
    )


def _reject_distribution_symlinks(staged: Path) -> None:
    """Reject symlinks before reading or copying distribution files."""
    for entry in staged.rglob("*"):
        if not entry.is_symlink():
            continue
        try:
            rel = entry.relative_to(staged)
        except ValueError:
            rel = entry
        raise DistributionError(
            f"Profile distributions cannot contain symlinks: {rel}"
        )


# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------


@dataclass
class InstallPlan:
    """Summary of what an install will do, surfaced for user confirmation."""
    manifest: DistributionManifest
    staged_dir: Path
    provenance: str
    target_dir: Path
    existing: bool  # True if target profile already exists (update path)
    preserves_config: bool = True
    has_cron: bool = False
    has_skills: bool = False
    receipt_path: Optional[Path] = None
    receipt: Dict[str, Any] = field(default_factory=dict)


def _has_cron_jobs(staged: Path) -> bool:
    cron_dir = staged / "cron"
    if not cron_dir.is_dir():
        return False
    for _ in cron_dir.rglob("*.json"):
        return True
    for _ in cron_dir.rglob("*.yaml"):
        return True
    return False


def _count_skills(staged: Path) -> int:
    skills_dir = staged / "skills"
    if not skills_dir.is_dir():
        return 0
    return sum(
        1 for p in skills_dir.rglob("SKILL.md") if not is_excluded_skill_path(p)
    )


def plan_install(
    source: str,
    workdir: Path,
    override_name: Optional[str] = None,
) -> InstallPlan:
    """Stage *source* and produce a plan describing what install would do."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )
    from hermes_cli import __version__ as hermes_version

    staged, provenance = _stage_source(source, workdir)
    _reject_distribution_symlinks(staged)
    manifest = read_manifest(staged)
    if manifest is None:
        raise DistributionError(
            f"No {MANIFEST_FILENAME} found at the distribution root — "
            "this source is not a Hermes distribution."
        )

    # Version check up-front so we fail fast
    check_hermes_requires(manifest.hermes_requires, hermes_version)
    check_hermes_capabilities(manifest.hermes_capabilities)

    # Resolve target profile name
    target_name = override_name or manifest.name
    canon = normalize_profile_name(target_name)
    validate_profile_name(canon)
    if canon == "default":
        raise DistributionError(
            "Cannot install a distribution as 'default' — that is the built-in "
            "root profile (~/.hermes).  Pass --name <name> to install under a "
            "new profile."
        )
    manifest.name = canon
    manifest.source = provenance
    # Stamped once here so plan_install() callers (both fresh install and
    # update) propagate a freshly-minted timestamp through _copy_dist_payload.
    manifest.installed_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    target_dir = get_profile_dir(canon)
    existing = target_dir.is_dir()
    has_cron = _has_cron_jobs(staged)
    skill_count = _count_skills(staged)

    return InstallPlan(
        manifest=manifest,
        staged_dir=staged,
        provenance=provenance,
        target_dir=target_dir,
        existing=existing,
        preserves_config=existing,
        has_cron=has_cron,
        has_skills=skill_count > 0,
    )


def _copy_dist_payload(
    staged: Path,
    target: Path,
    manifest: DistributionManifest,
    preserve_config: bool,
) -> Tuple[List[str], List[str]]:
    """Copy distribution-owned files from *staged* into *target*.

    User-owned paths are never touched.  ``config.yaml`` is replaced only when
    ``preserve_config`` is False (fresh install or ``--force-config`` update).
    ``.env.template`` is renamed to ``.env.EXAMPLE`` in the target to avoid
    shadowing a real ``.env``.  Returns ``(source_paths, installed_paths)`` for
    the transaction receipt and integrity digest.
    """
    target.mkdir(parents=True, exist_ok=True)

    specs: List[Tuple[str, str]] = []
    for source_rel in manifest.owned_paths():
        target_rel = (
            ENV_EXAMPLE_FILENAME
            if source_rel == ENV_TEMPLATE_FILENAME
            else source_rel
        )
        if target_rel == "config.yaml" and preserve_config:
            continue
        specs.append((source_rel, target_rel))

    # The installed manifest is always Hermes-owned, even if an explicit
    # distribution_owned list omitted it.  Keep the historical env-template
    # convenience too: shipping .env.template never writes a live secret file.
    specs.append((MANIFEST_FILENAME, MANIFEST_FILENAME))
    if (staged / ENV_TEMPLATE_FILENAME).is_file() or manifest.env_requires:
        specs.append((ENV_TEMPLATE_FILENAME, ENV_EXAMPLE_FILENAME))

    # A parent entry owns its entire subtree.  Dedupe before copying so a
    # manifest cannot make the same destination participate twice.
    by_target: Dict[str, str] = {}
    for source_rel, target_rel in specs:
        source_rel = _normalize_owned_path(source_rel)
        target_rel = _normalize_owned_path(target_rel)
        by_target.setdefault(target_rel, source_rel)
    installed_paths = _dedupe_owned_paths(list(by_target))
    source_paths: List[str] = []

    for target_rel in installed_paths:
        source_rel = by_target[target_rel]
        source_paths.append(source_rel)
        entry = staged / source_rel
        dest = target / target_rel
        if not entry.exists():
            # Absence is meaningful on update: the transaction removes an old
            # owned path and the digest records the expected missing marker.
            continue
        if entry.is_symlink():
            raise DistributionError(
                f"Profile distributions cannot contain symlinks: {source_rel}"
            )
        dest.parent.mkdir(parents=True, exist_ok=True)
        if entry.is_dir():
            shutil.copytree(entry, dest)
        elif entry.is_file():
            shutil.copy2(entry, dest)
        else:
            raise DistributionError(
                f"Unsupported distribution entry type: {source_rel}"
            )

    # Emit .env.EXAMPLE from manifest if the staged tree didn't ship one
    if manifest.env_requires and not (target / ENV_EXAMPLE_FILENAME).exists():
        (target / ENV_EXAMPLE_FILENAME).write_text(
            _env_template_from_manifest(manifest), encoding="utf-8"
        )

    # Make sure the manifest on disk reflects resolved name + source
    write_manifest(target, manifest)
    return _dedupe_owned_paths(source_paths), installed_paths


def _remove_path(path: Path) -> None:
    """Remove one file or directory without following symlinks."""
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


def _assert_safe_owned_path(root: Path, rel: str) -> Path:
    """Reject symlinks anywhere from *root* through an owned path."""
    rel = _normalize_owned_path(rel)
    if root.is_symlink():
        raise DistributionError(f"Profile root cannot be a symlink: {root}")
    current = root
    for part in PurePosixPath(rel).parts:
        current = current / part
        if current.is_symlink():
            raise DistributionError(
                f"Owned payload path cannot traverse a symlink: {rel} ({current})"
            )
    return current


def _ensure_safe_owned_parent(root: Path, rel: str) -> Path:
    """Create missing parents without traversing a pre-existing symlink."""
    rel = _normalize_owned_path(rel)
    if root.is_symlink():
        raise DistributionError(f"Profile root cannot be a symlink: {root}")
    current = root
    for part in PurePosixPath(rel).parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise DistributionError(
                f"Owned payload parent cannot be a symlink: {rel} ({current})"
            )
        if current.exists() and not current.is_dir():
            raise DistributionError(
                f"Owned payload parent is not a directory: {rel} ({current})"
            )
        current.mkdir(exist_ok=True)
        if current.is_symlink():
            raise DistributionError(
                f"Owned payload parent became a symlink: {rel} ({current})"
            )
    return root / rel


def _tree_digest(root: Path, owned_paths: List[str]) -> str:
    """Hash owned paths deterministically, including absence and empty dirs."""
    digest = hashlib.sha256(b"hermes-distribution-payload-v1\0")

    def _record(path: Path, rel: str) -> None:
        if path.is_symlink():
            raise DistributionError(f"Owned payload contains a symlink: {rel}")
        if not path.exists():
            digest.update(b"M\0" + rel.encode("utf-8") + b"\0")
            return
        if path.is_dir():
            mode = stat.S_IMODE(path.stat().st_mode)
            digest.update(
                b"D\0"
                + rel.encode("utf-8")
                + b"\0"
                + f"{mode:o}".encode("ascii")
                + b"\0"
            )
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                child_rel = f"{rel}/{child.name}"
                _record(child, child_rel)
            return
        if not path.is_file():
            raise DistributionError(f"Unsupported owned payload entry: {rel}")
        mode = stat.S_IMODE(path.stat().st_mode)
        digest.update(
            b"F\0"
            + rel.encode("utf-8")
            + b"\0"
            + f"{mode:o}".encode("ascii")
            + b"\0"
        )
        digest.update(str(path.stat().st_size).encode("ascii") + b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest.update(b"\0")

    for rel in _dedupe_owned_paths(
        [_normalize_owned_path(path) for path in owned_paths]
    ):
        _record(_assert_safe_owned_path(root, rel), rel)
    return digest.hexdigest()


def _ignored_source_paths(staged: Path, source_paths: List[str]) -> List[str]:
    """List top-level source entries outside the manifest's owned boundary."""
    owned = [_normalize_owned_path(path) for path in source_paths]
    ignored: List[str] = []
    for entry in sorted(staged.iterdir(), key=lambda item: item.name):
        rel = entry.name
        if rel == ".git":
            continue
        covered = any(
            rel == path
            or path.startswith(rel + "/")
            or rel.startswith(path + "/")
            for path in owned
        )
        if not covered:
            ignored.append(rel)
    return ignored


def _durable_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Durably replace a JSON file on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        tmp.unlink(missing_ok=True)


def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    """Commit-marker writer kept separate for fault-injection tests."""
    _durable_write_json(path, data)


def _write_rollback_receipt(
    directory: Path,
    plan: InstallPlan,
    *,
    transaction_id: str,
    operation: str,
    error: BaseException,
    restored: bool,
    recovery_errors: Optional[List[str]] = None,
    pre_transaction_sha256: Optional[str] = None,
    restored_sha256: Optional[str] = None,
) -> Optional[Path]:
    """Persist terminal rollback evidence outside the restored profile."""
    path = directory / ROLLBACK_RECEIPT_FILENAME
    payload = {
        "schema_version": 1,
        "status": "rolled_back",
        "transaction_id": transaction_id,
        "operation": operation,
        "profile": plan.manifest.name,
        "source": plan.provenance,
        "failed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "restored": restored,
        "error": str(error),
        "recovery_errors": recovery_errors or [],
        "pre_transaction_sha256": pre_transaction_sha256,
        "restored_sha256": restored_sha256,
    }
    try:
        _durable_write_json(path, payload)
    except BaseException:
        return None
    return path


def _lock_path(target_dir: Path) -> Path:
    return target_dir.parent / LOCK_DIRNAME / f"{target_dir.name}.lock"


@contextmanager
def _distribution_lock(target_dir: Path):
    """Serialize installs/updates for one profile and fail on stale locks."""
    lock_path = _lock_path(target_dir)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        detail = ""
        try:
            detail = lock_path.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        suffix = f" ({detail})" if detail else ""
        raise DistributionError(
            f"Profile '{target_dir.name}' has an active or stale distribution "
            f"transaction lock at {lock_path}{suffix}. Run `hermes profile "
            f"doctor {target_dir.name}` before retrying."
        ) from exc
    try:
        payload = {
            "pid": os.getpid(),
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield lock_path
    finally:
        lock_path.unlink(missing_ok=True)


def _active_profile_snapshot() -> Dict[str, Any]:
    """Return a content-only snapshot of the sticky active-profile pointer."""
    from hermes_cli.profiles import get_profile_dir

    path = get_profile_dir("default") / "active_profile"
    try:
        content = path.read_bytes()
    except FileNotFoundError:
        return {
            "path": str(path),
            "exists": False,
            "sha256": None,
            "size": 0,
            "profile": "default",
        }
    except OSError as exc:
        raise DistributionError(f"Cannot read active-profile pointer {path}: {exc}") from exc
    return {
        "path": str(path),
        "exists": True,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
        "profile": content.decode("utf-8", errors="replace").strip() or "default",
    }


def _build_receipt(
    plan: InstallPlan,
    *,
    transaction_id: str,
    operation: str,
    source_paths: List[str],
    installed_paths: List[str],
    source_sha256: str,
    installed_sha256: str,
    active_before: Dict[str, Any],
    active_after: Dict[str, Any],
    backup_path: Optional[Path],
    preserves_config: bool,
) -> Dict[str, Any]:
    ignored_source_paths = _ignored_source_paths(plan.staged_dir, source_paths)
    if preserves_config:
        ignored_source_paths = [
            path for path in ignored_source_paths if path != "config.yaml"
        ]
    unowned_collisions = [
        rel
        for rel in ignored_source_paths
        if (plan.target_dir / rel).exists() or (plan.target_dir / rel).is_symlink()
    ]
    return {
        "schema_version": 1,
        "status": "committed",
        "transaction_id": transaction_id,
        "operation": operation,
        "profile": plan.manifest.name,
        "manifest_version": plan.manifest.version,
        "hermes_capabilities": plan.manifest.hermes_capabilities,
        "source": plan.provenance,
        "committed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "distribution_owned": plan.manifest.owned_paths(),
        "source_paths": source_paths,
        "ignored_source_paths": ignored_source_paths,
        "unowned_collisions": unowned_collisions,
        "verified_paths": installed_paths,
        "preserved_paths": ["config.yaml"] if preserves_config else [],
        "source_sha256": source_sha256,
        "installed_sha256": installed_sha256,
        "active_profile": {
            "before": active_before,
            "after": active_after,
            "unchanged": active_before == active_after,
        },
        "backup_path": str(backup_path) if backup_path else None,
    }


def _bootstrap_user_dirs(target: Path) -> None:
    """Create the bootstrap dirs a fresh profile expects."""
    for d in ("memories", "sessions", "skills", "skins", "logs",
              "plans", "workspace", "cron", "home"):
        (target / d).mkdir(parents=True, exist_ok=True)


def _prepare_candidate(
    plan: InstallPlan,
    *,
    preserve_config: bool,
    bootstrap_user_dirs: bool,
) -> Tuple[Path, List[str], List[str], str, str]:
    """Build and hash the candidate tree on the target filesystem."""
    parent = plan.target_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    candidate = Path(
        tempfile.mkdtemp(
            prefix=f".{plan.manifest.name}.distribution-candidate.",
            dir=parent,
        )
    )
    try:
        source_paths, installed_paths = _copy_dist_payload(
            plan.staged_dir,
            candidate,
            plan.manifest,
            preserve_config=preserve_config,
        )
        if bootstrap_user_dirs:
            _bootstrap_user_dirs(candidate)
        source_sha256 = _tree_digest(plan.staged_dir, source_paths)
        installed_sha256 = _tree_digest(candidate, installed_paths)
        return (
            candidate,
            source_paths,
            installed_paths,
            source_sha256,
            installed_sha256,
        )
    except BaseException:
        shutil.rmtree(candidate, ignore_errors=True)
        raise


def _commit_fresh_candidate(
    plan: InstallPlan,
    candidate: Path,
    *,
    source_paths: List[str],
    installed_paths: List[str],
    source_sha256: str,
    expected_sha256: str,
) -> None:
    """Publish a new profile with one same-filesystem atomic rename."""
    target = plan.target_dir
    if target.exists():
        raise DistributionError(
            f"Profile '{plan.manifest.name}' appeared while install was being prepared. "
            "Retry, or pass --force if replacing it is intentional."
        )
    transaction_id = uuid.uuid4().hex
    active_before = _active_profile_snapshot()
    committed = False
    retain_committed = False
    try:
        os.replace(candidate, target)
        committed = True
        active_after = _active_profile_snapshot()
        if active_before != active_after:
            actual_sha256 = _tree_digest(target, installed_paths)
            receipt = _build_receipt(
                plan,
                transaction_id=transaction_id,
                operation="install",
                source_paths=source_paths,
                installed_paths=installed_paths,
                source_sha256=source_sha256,
                installed_sha256=actual_sha256,
                active_before=active_before,
                active_after=active_after,
                backup_path=None,
                preserves_config=False,
            )
            receipt["status"] = "committed_with_concurrent_activation"
            receipt_path = target / RECEIPT_FILENAME
            _atomic_write_json(receipt_path, receipt)
            plan.receipt_path = receipt_path
            plan.receipt = receipt
            retain_committed = True
            raise DistributionError(
                "The sticky active-profile pointer changed concurrently during install; "
                f"the committed profile was retained at {target} so an activation "
                f"cannot point to a deleted profile. Receipt: {receipt_path}."
            )
        actual_sha256 = _tree_digest(target, installed_paths)
        if actual_sha256 != expected_sha256:
            raise DistributionError(
                "Installed profile payload did not match its prepared digest; "
                "the new profile was rolled back."
            )
        receipt = _build_receipt(
            plan,
            transaction_id=transaction_id,
            operation="install",
            source_paths=source_paths,
            installed_paths=installed_paths,
            source_sha256=source_sha256,
            installed_sha256=actual_sha256,
            active_before=active_before,
            active_after=active_after,
            backup_path=None,
            preserves_config=False,
        )
        receipt_path = target / RECEIPT_FILENAME
        _atomic_write_json(receipt_path, receipt)
        plan.receipt_path = receipt_path
        plan.receipt = receipt
    except BaseException as exc:
        recovery_errors: List[str] = []
        if committed and not retain_committed:
            try:
                retain_committed = (
                    _active_profile_snapshot().get("profile")
                    == plan.manifest.name
                )
            except BaseException as recovery_exc:
                recovery_errors.append(
                    f"active-profile recheck failed: {recovery_exc}"
                )
        if committed and not retain_committed:
            try:
                _remove_path(target)
            except BaseException as recovery_exc:
                recovery_errors.append(f"target removal failed: {recovery_exc}")
        if retain_committed:
            note = (
                f"Fresh profile remains committed at {target}; it was not removed "
                "because it became the active profile."
            )
            if hasattr(exc, "add_note"):
                exc.add_note(note)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, DistributionError):
                raise
            raise DistributionError(f"{exc}. {note}") from exc
        rollback_dir = (
            target.parent
            / BACKUP_DIRNAME
            / plan.manifest.name
            / transaction_id
        )
        restored = not target.exists() and not recovery_errors
        rollback_receipt = _write_rollback_receipt(
            rollback_dir,
            plan,
            transaction_id=transaction_id,
            operation="install",
            error=exc,
            restored=restored,
            recovery_errors=recovery_errors,
        )
        receipt_note = (
            f" Rollback receipt: {rollback_receipt}."
            if rollback_receipt
            else " Rollback receipt could not be written."
        )
        outcome = "restored" if restored else "INCOMPLETE"
        note = (
            f"Fresh distribution install failed; rollback {outcome}: "
            f"{exc}.{receipt_note}"
        )
        if hasattr(exc, "add_note"):
            exc.add_note(note)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise DistributionError(note) from exc


def _commit_existing_candidate(
    plan: InstallPlan,
    candidate: Path,
    *,
    operation: str,
    source_paths: List[str],
    installed_paths: List[str],
    source_sha256: str,
    expected_sha256: str,
    preserves_config: bool,
) -> None:
    """Swap owned paths with durable backups and rollback on any failure."""
    target = plan.target_dir
    if target.is_symlink():
        raise DistributionError(f"Profile root cannot be a symlink: {target}")
    if not target.is_dir():
        raise DistributionError(
            f"Profile '{plan.manifest.name}' disappeared during the transaction."
        )

    for rel in installed_paths:
        _assert_safe_owned_path(target, rel)
    pre_transaction_sha256 = _tree_digest(target, installed_paths)

    transaction_id = uuid.uuid4().hex
    backup = (
        target.parent
        / BACKUP_DIRNAME
        / plan.manifest.name
        / transaction_id
    )
    backup.mkdir(parents=True, exist_ok=False)
    moved_existing: List[Tuple[Path, Path]] = []
    installed_new: List[Path] = []
    receipt_path = target / RECEIPT_FILENAME
    backup_receipt = backup / RECEIPT_FILENAME
    active_before = _active_profile_snapshot()

    try:
        if receipt_path.is_symlink():
            raise DistributionError(
                f"Distribution receipt cannot be a symlink: {receipt_path}"
            )
        if receipt_path.exists():
            backup_receipt.parent.mkdir(parents=True, exist_ok=True)
            os.replace(receipt_path, backup_receipt)

        for rel in installed_paths:
            source = candidate / rel
            destination = _assert_safe_owned_path(target, rel)
            prior = backup / rel
            if destination.exists():
                prior.parent.mkdir(parents=True, exist_ok=True)
                _assert_safe_owned_path(target, rel)
                os.replace(destination, prior)
                moved_existing.append((destination, prior))
            if source.is_symlink():
                raise DistributionError(
                    f"Candidate payload cannot contain a symlink: {source}"
                )
            if source.exists():
                destination = _ensure_safe_owned_parent(target, rel)
                _assert_safe_owned_path(target, rel)
                os.replace(source, destination)
                installed_new.append(destination)

        actual_sha256 = _tree_digest(target, installed_paths)
        if actual_sha256 != expected_sha256:
            raise DistributionError(
                "Installed profile payload did not match its prepared digest."
            )
        active_after = _active_profile_snapshot()
        if active_before != active_after:
            raise DistributionError(
                "The sticky active-profile pointer changed concurrently; owned-path "
                "changes were rolled back and the pointer was left untouched."
            )

        has_backup = bool(moved_existing) or backup_receipt.exists()
        receipt = _build_receipt(
            plan,
            transaction_id=transaction_id,
            operation=operation,
            source_paths=source_paths,
            installed_paths=installed_paths,
            source_sha256=source_sha256,
            installed_sha256=actual_sha256,
            active_before=active_before,
            active_after=active_after,
            backup_path=backup if has_backup else None,
            preserves_config=preserves_config,
        )
        _atomic_write_json(receipt_path, receipt)
        plan.receipt_path = receipt_path
        plan.receipt = receipt
        if not has_backup:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException as exc:
        recovery_errors: List[str] = []
        try:
            if receipt_path.is_symlink():
                raise DistributionError(
                    f"Receipt became a symlink during rollback: {receipt_path}"
                )
            _remove_path(receipt_path)
        except BaseException as recovery_exc:
            recovery_errors.append(f"new receipt removal failed: {recovery_exc}")
        for destination in reversed(installed_new):
            try:
                rel = destination.relative_to(target).as_posix()
                _assert_safe_owned_path(target, rel)
                _remove_path(destination)
            except BaseException as recovery_exc:
                recovery_errors.append(
                    f"new payload removal failed for {destination}: {recovery_exc}"
                )
        for destination, prior in reversed(moved_existing):
            try:
                rel = destination.relative_to(target).as_posix()
                destination = _ensure_safe_owned_parent(target, rel)
                _assert_safe_owned_path(target, rel)
                if destination.exists():
                    raise DistributionError(
                        f"Rollback destination unexpectedly exists: {destination}"
                    )
                os.replace(prior, destination)
            except BaseException as recovery_exc:
                recovery_errors.append(
                    f"prior payload restore failed for {destination}: {recovery_exc}"
                )
        try:
            if backup_receipt.exists() or backup_receipt.is_symlink():
                if receipt_path.exists() or receipt_path.is_symlink():
                    raise DistributionError(
                        f"Rollback receipt destination unexpectedly exists: {receipt_path}"
                    )
                os.replace(backup_receipt, receipt_path)
        except BaseException as recovery_exc:
            recovery_errors.append(f"prior receipt restore failed: {recovery_exc}")

        restored_sha256: Optional[str] = None
        try:
            restored_sha256 = _tree_digest(target, installed_paths)
            if restored_sha256 != pre_transaction_sha256:
                recovery_errors.append(
                    "post-rollback digest mismatch: "
                    f"expected {pre_transaction_sha256}, got {restored_sha256}"
                )
        except BaseException as recovery_exc:
            recovery_errors.append(f"post-rollback verification failed: {recovery_exc}")
        restored = not recovery_errors
        rollback_receipt = _write_rollback_receipt(
            backup,
            plan,
            transaction_id=transaction_id,
            operation=operation,
            error=exc,
            restored=restored,
            recovery_errors=recovery_errors,
            pre_transaction_sha256=pre_transaction_sha256,
            restored_sha256=restored_sha256,
        )
        receipt_note = (
            f" Rollback receipt: {rollback_receipt}."
            if rollback_receipt
            else " Rollback receipt could not be written."
        )
        outcome = "restored" if restored else "INCOMPLETE"
        note = (
            f"Distribution transaction failed; rollback {outcome}: "
            f"{exc}.{receipt_note}"
        )
        if hasattr(exc, "add_note"):
            exc.add_note(note)
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise DistributionError(note) from exc


def install_distribution(
    source: str,
    name: Optional[str] = None,
    force: bool = False,
    create_alias: bool = False,
) -> InstallPlan:
    """Install a distribution from *source* into a new profile.

    Returns the resolved :class:`InstallPlan`.  Use :func:`plan_install`
    first if you want to preview + prompt the user before calling this.
    """
    from hermes_cli.profiles import (
        check_alias_collision,
        create_wrapper_script,
    )

    with tempfile.TemporaryDirectory(prefix="hermes_dist_install_") as tmp:
        plan = plan_install(source, Path(tmp), override_name=name)

        with _distribution_lock(plan.target_dir):
            plan.existing = plan.target_dir.is_dir()
            if plan.existing and not force:
                raise DistributionError(
                    f"Profile '{plan.manifest.name}' already exists at {plan.target_dir}. "
                    "Use `hermes profile update` to upgrade in place, "
                    "or pass --force to overwrite."
                )

            candidate: Optional[Path] = None
            try:
                (
                    candidate,
                    source_paths,
                    installed_paths,
                    source_sha256,
                    expected_sha256,
                ) = _prepare_candidate(
                    plan,
                    preserve_config=False,
                    bootstrap_user_dirs=not plan.existing,
                )
                if plan.existing:
                    _commit_existing_candidate(
                        plan,
                        candidate,
                        operation="force-install",
                        source_paths=source_paths,
                        installed_paths=installed_paths,
                        source_sha256=source_sha256,
                        expected_sha256=expected_sha256,
                        preserves_config=False,
                    )
                else:
                    _commit_fresh_candidate(
                        plan,
                        candidate,
                        source_paths=source_paths,
                        installed_paths=installed_paths,
                        source_sha256=source_sha256,
                        expected_sha256=expected_sha256,
                    )
            finally:
                if candidate is not None:
                    shutil.rmtree(candidate, ignore_errors=True)

        if create_alias:
            collision = check_alias_collision(plan.manifest.name)
            if collision is None:
                create_wrapper_script(plan.manifest.name)

        return plan


def update_distribution(
    profile_name: str,
    force_config: bool = False,
) -> InstallPlan:
    """Re-pull the distribution for an existing profile and apply updates.

    The source is read from the installed profile's ``distribution.yaml``
    ``source:`` field.  Distribution-owned files are overwritten; user-owned
    data (memories, sessions, auth) is never touched.  ``config.yaml`` is
    preserved unless ``force_config`` is True.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")

    with _distribution_lock(target):
        existing_manifest = read_manifest(target)
        if existing_manifest is None:
            raise DistributionError(
                f"Profile '{canon}' is not a distribution (no {MANIFEST_FILENAME}). "
                "Only profiles installed via `hermes profile install` can be updated."
            )
        if not existing_manifest.source:
            raise DistributionError(
                f"Profile '{canon}' has no recorded source.  Re-install with "
                "`hermes profile install <source> --name {canon} --force`."
            )

        with tempfile.TemporaryDirectory(prefix="hermes_dist_update_") as tmp:
            plan = plan_install(
                existing_manifest.source,
                Path(tmp),
                override_name=canon,
            )
            plan.preserves_config = not force_config
            candidate: Optional[Path] = None
            try:
                (
                    candidate,
                    source_paths,
                    installed_paths,
                    source_sha256,
                    expected_sha256,
                ) = _prepare_candidate(
                    plan,
                    preserve_config=plan.preserves_config,
                    bootstrap_user_dirs=False,
                )
                _commit_existing_candidate(
                    plan,
                    candidate,
                    operation="update",
                    source_paths=source_paths,
                    installed_paths=installed_paths,
                    source_sha256=source_sha256,
                    expected_sha256=expected_sha256,
                    preserves_config=plan.preserves_config,
                )
                return plan
            finally:
                if candidate is not None:
                    shutil.rmtree(candidate, ignore_errors=True)


# ---------------------------------------------------------------------------
# Info — render a manifest summary
# ---------------------------------------------------------------------------


def describe_distribution(profile_name: str) -> Dict[str, Any]:
    """Return a structured view of a profile's distribution metadata.

    Returns an empty dict if the profile exists but has no manifest.
    Raises DistributionError if the profile itself doesn't exist.
    """
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")
    manifest = read_manifest(target)
    if manifest is None:
        return {}
    return manifest.to_dict()


def doctor_distribution(profile_name: str) -> Dict[str, Any]:
    """Verify a distribution receipt, owned payload, and transaction state."""
    from hermes_cli.profiles import (
        get_profile_dir,
        normalize_profile_name,
        validate_profile_name,
    )

    canon = normalize_profile_name(profile_name)
    validate_profile_name(canon)
    target = get_profile_dir(canon)
    if not target.is_dir():
        raise DistributionError(f"Profile '{canon}' does not exist.")

    issues: List[str] = []
    receipt_path = target / RECEIPT_FILENAME
    manifest: Optional[DistributionManifest] = None
    receipt: Dict[str, Any] = {}
    actual_sha256: Optional[str] = None
    actual_source_sha256: Optional[str] = None
    source_available = False

    try:
        manifest = read_manifest(target)
    except DistributionError as exc:
        issues.append(str(exc))
    if manifest is None:
        issues.append(f"Missing {MANIFEST_FILENAME}; profile is not a distribution")

    if not receipt_path.is_file():
        issues.append(
            f"Missing {RECEIPT_FILENAME}; reinstall or update with a receipt-aware Hermes"
        )
    else:
        try:
            loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("receipt root is not an object")
            receipt = loaded
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            issues.append(f"Unreadable distribution receipt: {exc}")

    if receipt:
        if receipt.get("schema_version") != 1:
            issues.append(
                f"Unsupported receipt schema: {receipt.get('schema_version')!r}"
            )
        if receipt.get("status") != "committed":
            issues.append(
                f"Transaction receipt is not committed: {receipt.get('status')!r}"
            )
        if receipt.get("profile") != canon:
            issues.append(
                f"Receipt profile {receipt.get('profile')!r} does not match {canon!r}"
            )
        active = receipt.get("active_profile")
        if not isinstance(active, dict) or active.get("unchanged") is not True:
            issues.append("Install/update did not prove the active-profile pointer unchanged")

        raw_paths = receipt.get("verified_paths")
        if not isinstance(raw_paths, list) or not raw_paths:
            issues.append("Receipt has no verified_paths")
        else:
            try:
                verified_paths = [
                    _normalize_owned_path(path) for path in raw_paths
                ]
                actual_sha256 = _tree_digest(target, verified_paths)
                expected_sha256 = receipt.get("installed_sha256")
                if not isinstance(expected_sha256, str) or not expected_sha256:
                    issues.append("Receipt has no installed_sha256")
                elif actual_sha256 != expected_sha256:
                    issues.append(
                        "Owned payload digest mismatch: "
                        f"expected {expected_sha256}, got {actual_sha256}"
                    )
            except DistributionError as exc:
                issues.append(str(exc))

        source = receipt.get("source")
        raw_source_paths = receipt.get("source_paths")
        if isinstance(source, str) and isinstance(raw_source_paths, list):
            source_dir = Path(source).expanduser()
            if source_dir.is_dir():
                source_available = True
                try:
                    actual_source_sha256 = _tree_digest(
                        source_dir,
                        [_normalize_owned_path(path) for path in raw_source_paths],
                    )
                    expected_source_sha256 = receipt.get("source_sha256")
                    if actual_source_sha256 != expected_source_sha256:
                        issues.append(
                            "Source payload digest mismatch: "
                            f"expected {expected_source_sha256}, "
                            f"got {actual_source_sha256}"
                        )
                except DistributionError as exc:
                    issues.append(str(exc))

        unowned_collisions = receipt.get("unowned_collisions") or []
        if not isinstance(unowned_collisions, list):
            issues.append("Receipt unowned_collisions field is not a list")
        elif unowned_collisions:
            issues.append(
                "Unowned source paths collided with installed profile paths: "
                + ", ".join(str(path) for path in unowned_collisions)
            )

        if manifest is not None:
            if manifest.name != canon:
                issues.append(
                    f"Manifest profile {manifest.name!r} does not match {canon!r}"
                )
            if receipt.get("manifest_version") != manifest.version:
                issues.append(
                    "Manifest version does not match receipt: "
                    f"{manifest.version!r} != {receipt.get('manifest_version')!r}"
                )

    lock = _lock_path(target)
    if lock.exists():
        issues.append(f"Active or stale transaction lock present: {lock}")

    return {
        "ok": not issues,
        "profile": canon,
        "profile_path": str(target),
        "receipt_path": str(receipt_path),
        "transaction_id": receipt.get("transaction_id") if receipt else None,
        "operation": receipt.get("operation") if receipt else None,
        "installed_sha256": (
            receipt.get("installed_sha256") if receipt else None
        ),
        "actual_sha256": actual_sha256,
        "source_sha256": receipt.get("source_sha256") if receipt else None,
        "actual_source_sha256": actual_source_sha256,
        "source_available": source_available,
        "active_profile_unchanged": (
            (receipt.get("active_profile") or {}).get("unchanged")
            if receipt
            else None
        ),
        "backup_path": receipt.get("backup_path") if receipt else None,
        "issues": issues,
    }
