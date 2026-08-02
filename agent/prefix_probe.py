"""Per-turn request-prefix hashing — a diagnostic for prompt-cache misses.

A provider's prompt cache (and a local server's KV cache) can only reuse work
up to the first byte where this request differs from the last one. When a long
session starts re-prefilling from scratch every turn, the question is always
the same and is otherwise unanswerable from the client: *did WE change the
prefix, or did something else evict the server's cache?*

This module answers it. With ``HERMES_PREFIX_PROBE=1`` set, every outbound API
call records a rolling hash over the prefix elements of the request, compares
it against the previous call in the same session, and logs where the two first
diverge — by element index, role, and character offset.

Reading the output:

* ``prefix stable through 64/64 elements`` — the client sent a byte-identical
  prefix. A cache miss here is NOT the client's doing; look at the server
  (another session on the same slot, an eviction, a restart).
* ``diverged at element 0 (tools)`` — the tool schema changed. A late-connecting
  MCP server or a check_fn flip rewrote the toolset.
* ``diverged at element 1 (system)`` — the system prompt was rebuilt.
* ``diverged at element N (role=tool)`` — history was mutated in place at N;
  everything from there on has to be re-prefilled.

State is file-backed under the session log dir rather than held on the agent,
because the cases worth catching are exactly the ones where the agent object
does not survive: a compute-host crash, an app update, a fresh per-turn agent
on the gateway path.

Diagnostic only. Enabled by an env var alongside its sibling
``HERMES_DUMP_REQUESTS``, ships off, and never raises into the request path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)

_ENV_FLAG = "HERMES_PREFIX_PROBE"
_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9_.-]")


def prefix_probe_enabled() -> bool:
    """Return whether the prefix probe is switched on for this process.

    Defers to ``env_var_enabled`` so this flag reads exactly like every other
    Hermes env toggle — the call site in ``conversation_loop`` gates on the
    same helper, and two different notions of "truthy" would mean a value that
    enables one and not the other.
    """
    from utils import env_var_enabled

    return env_var_enabled(_ENV_FLAG)


def _canonical(value: Any) -> str:
    """Serialize one prefix element the way a prefix comparison needs it.

    ``sort_keys`` so that dict ordering — which no provider's chat template
    depends on, and which Python does not guarantee across rebuilds — cannot
    masquerade as a real divergence.
    """
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    except Exception:
        return repr(value)


def prefix_elements(api_kwargs: dict) -> list[tuple[str, str]]:
    """Split a request into ordered ``(label, serialized)`` prefix elements.

    Tools come first: chat templates render tool definitions ahead of the
    conversation, so a toolset change invalidates everything after it. Then one
    element per message, in wire order.

    Sampling parameters are deliberately excluded — they do not participate in
    the cached prefix.
    """
    elements: list[tuple[str, str]] = []

    tools = api_kwargs.get("tools")
    if tools:
        elements.append(("tools", _canonical(tools)))

    messages = api_kwargs.get("messages")
    if not isinstance(messages, list):
        # Responses-style payloads carry the conversation under "input".
        messages = api_kwargs.get("input")
    if isinstance(messages, list):
        for index, message in enumerate(messages):
            role = ""
            if isinstance(message, dict):
                role = str(message.get("role") or "")
            label = role or f"msg{index}"
            elements.append((label, _canonical(message)))

    return elements


def rolling_hashes(elements: list[tuple[str, str]]) -> list[str]:
    """Return the cumulative hash after each element.

    Cumulative rather than per-element so that comparing two runs yields the
    first *prefix* divergence directly: the earliest index where the rolling
    hashes differ is the point past which no cached work can be reused.
    """
    hashes: list[str] = []
    running = hashlib.blake2b(digest_size=16)
    for _label, serialized in elements:
        running.update(serialized.encode("utf-8", "replace"))
        hashes.append(running.hexdigest())
    return hashes


def divergence_index(previous: list[str], current: list[str]) -> int | None:
    """Return the first index whose rolling hash differs, or None if none does.

    A pure extension of the previous request — the append-only case a healthy
    agentic turn should produce — returns None: every hash the two share is
    equal and the new request merely continues past the end of the old one.
    """
    for index in range(min(len(previous), len(current))):
        if previous[index] != current[index]:
            return index
    return None


def _state_path(agent: Any) -> str | None:
    logs_dir = getattr(agent, "logs_dir", None)
    session_id = str(getattr(agent, "session_id", "") or "unknown")
    if not logs_dir:
        return None
    safe = _UNSAFE_NAME.sub("_", session_id)[:120] or "unknown"
    return os.path.join(str(logs_dir), f"prefix_probe_{safe}.json")


def record_request_prefix(agent: Any, api_kwargs: dict) -> dict | None:
    """Hash this request's prefix, log its divergence from the previous one.

    Returns the comparison result (also useful to tests), or ``None`` when the
    probe is off or state could not be read/written. Never raises.
    """
    if not prefix_probe_enabled():
        return None
    try:
        elements = prefix_elements(api_kwargs)
        if not elements:
            return None
        current = rolling_hashes(elements)
        lengths = [len(serialized) for _label, serialized in elements]
        labels = [label for label, _serialized in elements]

        path = _state_path(agent)
        previous: list[str] = []
        previous_call = 0
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    stored = json.load(handle)
                if isinstance(stored, dict):
                    previous = list(stored.get("hashes") or [])
                    previous_call = int(stored.get("call") or 0)
            except Exception:
                previous = []

        index = divergence_index(previous, current) if previous else None
        call = previous_call + 1
        result = {
            "call": call,
            "elements": len(current),
            "previous_elements": len(previous),
            "divergence_index": index,
            "divergence_label": labels[index] if index is not None else None,
            "stable_chars": sum(lengths[:index]) if index is not None else sum(lengths),
        }

        if not previous:
            logger.info(
                "prefix-probe call #%d: baseline, %d elements (%d chars)",
                call, len(current), sum(lengths),
            )
        elif index is None:
            logger.info(
                "prefix-probe call #%d: prefix STABLE through %d/%d elements "
                "(%d chars reusable); request is append-only vs the previous call",
                call, len(previous), len(current), sum(lengths[: len(previous)]),
            )
        else:
            logger.warning(
                "prefix-probe call #%d: prefix DIVERGED at element %d/%d "
                "(label=%s) — only %d chars reusable, everything after is "
                "re-prefilled. Previous request had %d elements.",
                call, index, len(current), labels[index],
                result["stable_chars"], len(previous),
            )

        if path:
            try:
                tmp = f"{path}.tmp"
                with open(tmp, "w", encoding="utf-8") as handle:
                    json.dump({"call": call, "hashes": current, "labels": labels}, handle)
                os.replace(tmp, path)
            except Exception:
                pass
        return result
    except Exception:
        # A diagnostic must never be able to fail a turn.
        return None
