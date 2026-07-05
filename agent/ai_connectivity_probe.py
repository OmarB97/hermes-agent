"""Reusable "test my AI" connectivity probe (BYO-frontier capture, TASK 2).

A cheap, end-to-end, *real* inference call that tells a user whether their own
Anthropic / OpenAI credential actually works. It deliberately does a 1-token
inference call (``max_tokens=1``) rather than a ``/models`` listing or an
``auth status`` boolean, so it catches expired tokens, revoked keys, wrong
region, and zero-quota accounts that a listing call would pass.

Design reference: ``~/Workspaces/.mesh/handoffs/BYO-FRONTIER-CAPTURE-DESIGN.md``
§B ("The 'test my AI' probe"). This module is the shared library called out in
TASK 2 — callable from the engine, from ``hermes auth probe``, and (via a
shell-out) from the Rust ``meshboard-agent test-ai`` device subcommand.

Contract (``ProbeResult``):
    {provider, mode: "api"|"subscription", ok: bool, latency_ms: int,
     model: str, error_class?: invalid_key|expired_login|no_quota|network|unknown,
     hint?: str, status?: int, secret_fingerprint?: str}

No secret is ever returned in the result; only a sha256 fingerprint (matching
``agent.credential_persistence._fingerprint_value``) so a caller can show
"key ending …, last validated 2m ago" without holding the secret.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional


# Cheapest currently-available models for a 1-token validity ping. Kept here so
# the probe is self-contained on a customer device that has hermes but not the
# meshboard engine. Override via ``model=`` for testing / future model bumps.
DEFAULT_ANTHROPIC_PROBE_MODEL = "claude-haiku-4-5"
DEFAULT_OPENAI_PROBE_MODEL = "gpt-5.4-mini"

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_VERSION = "2023-06-01"

PROBE_TIMEOUT_SECS = 20.0

# Error-class enum returned to the user. Stable strings — the cloud row and the
# UI key off these.
ERROR_INVALID_KEY = "invalid_key"
ERROR_EXPIRED_LOGIN = "expired_login"
ERROR_NO_QUOTA = "no_quota"
ERROR_NETWORK = "network"
ERROR_UNKNOWN = "unknown"

# Auth-expiry signatures. Mirrors the intent of the engine's
# ``_PROVIDER_AUTH_EXPIRED_PATTERNS``
# (meshboard tools/meshctl_launcher_frontier_cli_implementer.py) but is kept as
# a local copy: this module must work on a customer device that has hermes but
# not the meshboard engine checkout. Keep the two lists conceptually in sync;
# do NOT import across repos.
_AUTH_EXPIRED_PATTERNS = (
    "failed to refresh token",
    "refresh token has expired",
    "refresh token has already been used",
    "refresh_token_reused",
    "please sign in again",
    "please run `codex login`",
    "please run codex login",
    "not logged in",
    "please re-authenticate",
    "expired",
    "revoked",
    "oauth token",
    "session expired",
)

# Substrings that mark a 4xx as "bad key" vs "expired login". An invalid/
# malformed/revoked-at-creation key reads as authentication, not refresh.
_INVALID_KEY_PATTERNS = (
    "invalid api key",
    "invalid x-api-key",
    "incorrect api key",
    "invalid_api_key",
    "authentication_error",
    "invalid bearer token",
    "no api key",
    "missing api key",
    "could not be authenticated",
)

# Substrings that mark a 429 / 4xx as "out of quota / billing" rather than a key
# problem — so the user is told to add credit, not paste a new key.
_NO_QUOTA_PATTERNS = (
    "insufficient_quota",
    "quota",
    "billing",
    "exceeded your current",
    "credit balance is too low",
    "out of credits",
    "payment",
)


def _fingerprint_value(value: Any) -> Optional[str]:
    """sha256:<first-16-hex> of a secret. Matches
    ``agent.credential_persistence._fingerprint_value`` so the cloud fingerprint
    is identical regardless of which path computed it."""
    if value is None:
        return None
    text = str(value)
    if not text:
        return None
    digest = hashlib.sha256(text.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"sha256:{digest[:16]}"


@dataclass
class ProbeResult:
    provider: str
    mode: str  # "api" | "subscription"
    ok: bool
    model: str
    latency_ms: int = 0
    error_class: Optional[str] = None
    hint: Optional[str] = None
    status: Optional[int] = None
    secret_fingerprint: Optional[str] = None
    # Free-form detail for logs/debug; never contains the secret.
    detail: Optional[str] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}

    def one_line(self) -> str:
        """The single user-visible line, e.g.
        'Anthropic (API key): ✓ connected (412ms)' /
        'OpenAI (API key): ✗ invalid key'."""
        label = _provider_label(self.provider)
        mode = "API key" if self.mode == "api" else "subscription"
        if self.ok:
            return f"{label} ({mode}): ✓ connected ({self.latency_ms}ms)"
        reason = {
            ERROR_INVALID_KEY: "✗ invalid key",
            ERROR_EXPIRED_LOGIN: "✗ login expired — re-authenticate",
            ERROR_NO_QUOTA: "✗ no quota / billing — add credit",
            ERROR_NETWORK: "✗ network error — could not reach provider",
        }.get(self.error_class, "✗ failed")
        if self.hint:
            return f"{label} ({mode}): {reason} — {self.hint}"
        return f"{label} ({mode}): {reason}"


def _provider_label(provider: str) -> str:
    p = (provider or "").strip().lower()
    if p in {"anthropic", "claude"}:
        return "Anthropic"
    if p in {"openai", "openai-api"}:
        return "OpenAI"
    return provider or "provider"


def _classify_4xx(status: int, body_text: str) -> str:
    """Map a non-2xx HTTP response onto an ``error_class``."""
    blob = (body_text or "").lower()
    if any(m in blob for m in _NO_QUOTA_PATTERNS) or status == 402:
        return ERROR_NO_QUOTA
    if status == 429:
        # 429 without an explicit quota marker is a transient rate limit; treat
        # it as no_quota-ish but distinguishable. Most "out of credit" answers
        # carry a quota marker handled above.
        return ERROR_NO_QUOTA
    if any(m in blob for m in _AUTH_EXPIRED_PATTERNS):
        return ERROR_EXPIRED_LOGIN
    if any(m in blob for m in _INVALID_KEY_PATTERNS):
        return ERROR_INVALID_KEY
    if status in (401, 403):
        # A bare 401/403 with no refresh phrasing is an invalid/forbidden key.
        return ERROR_INVALID_KEY
    return ERROR_UNKNOWN


# An HTTP transport is ``(method, url, headers, body) -> (status, text)``.
# Injectable so tests mock provider responses without a network.
Transport = Callable[[str, str, dict, bytes], "tuple[int, str]"]


def _urllib_transport(method: str, url: str, headers: dict, body: bytes) -> "tuple[int, str]":
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=PROBE_TIMEOUT_SECS) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:  # 4xx/5xx with a body
        try:
            text = exc.read().decode("utf-8", errors="replace")
        except Exception:
            text = str(exc)
        return exc.code, text
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        # Network-level failure: no HTTP status at all.
        raise _NetworkError(str(exc)) from exc


class _NetworkError(Exception):
    pass


def probe_anthropic(
    api_key: str,
    *,
    base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
    model: str = DEFAULT_ANTHROPIC_PROBE_MODEL,
    is_oauth: bool = False,
    transport: Optional[Transport] = None,
) -> ProbeResult:
    """1-token POST {base}/v1/messages. ``x-api-key`` for API keys,
    ``Authorization: Bearer`` for subscription OAuth tokens."""
    transport = transport or _urllib_transport
    url = base_url.rstrip("/") + "/v1/messages"
    headers = {
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
    }
    if is_oauth:
        headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["x-api-key"] = api_key
    payload = json.dumps(
        {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    ).encode("utf-8")
    return _run(
        provider="anthropic",
        mode="subscription" if is_oauth else "api",
        model=model,
        secret=api_key,
        method="POST",
        url=url,
        headers=headers,
        body=payload,
        transport=transport,
    )


def probe_openai(
    api_key: str,
    *,
    base_url: str = DEFAULT_OPENAI_BASE_URL,
    model: str = DEFAULT_OPENAI_PROBE_MODEL,
    transport: Optional[Transport] = None,
) -> ProbeResult:
    """1-token POST {base}/chat/completions, ``Authorization: Bearer``."""
    transport = transport or _urllib_transport
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "content-type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = json.dumps(
        {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]}
    ).encode("utf-8")
    return _run(
        provider="openai-api",
        mode="api",
        model=model,
        secret=api_key,
        method="POST",
        url=url,
        headers=headers,
        body=payload,
        transport=transport,
    )


def _run(
    *,
    provider: str,
    mode: str,
    model: str,
    secret: str,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    transport: Transport,
) -> ProbeResult:
    fingerprint = _fingerprint_value(secret)
    started = time.monotonic()
    try:
        status, text = transport(method, url, headers, body)
    except _NetworkError as exc:
        return ProbeResult(
            provider=provider,
            mode=mode,
            ok=False,
            model=model,
            latency_ms=int((time.monotonic() - started) * 1000),
            error_class=ERROR_NETWORK,
            hint="could not reach the provider",
            secret_fingerprint=fingerprint,
            detail=str(exc)[:200],
        )
    latency_ms = int((time.monotonic() - started) * 1000)
    if 200 <= status < 300:
        return ProbeResult(
            provider=provider,
            mode=mode,
            ok=True,
            model=model,
            latency_ms=latency_ms,
            status=status,
            secret_fingerprint=fingerprint,
        )
    if status >= 500:
        return ProbeResult(
            provider=provider,
            mode=mode,
            ok=False,
            model=model,
            latency_ms=latency_ms,
            status=status,
            error_class=ERROR_NETWORK,
            hint="provider returned a server error — try again",
            secret_fingerprint=fingerprint,
            detail=text[:200],
        )
    error_class = _classify_4xx(status, text)
    hint = {
        ERROR_INVALID_KEY: "paste a new key",
        ERROR_EXPIRED_LOGIN: "re-run your provider login",
        ERROR_NO_QUOTA: "add credit / check billing",
    }.get(error_class)
    return ProbeResult(
        provider=provider,
        mode=mode,
        ok=False,
        model=model,
        latency_ms=latency_ms,
        status=status,
        error_class=error_class,
        hint=hint,
        secret_fingerprint=fingerprint,
        detail=text[:200],
    )


# Provider-name aliases accepted by the user-facing surfaces. The Rust agent and
# CLI accept friendly names ("anthropic", "openai"); map onto probe functions.
_PROVIDER_ALIASES = {
    "anthropic": "anthropic",
    "claude": "anthropic",
    "openai": "openai-api",
    "openai-api": "openai-api",
}


def normalize_provider(provider: str) -> Optional[str]:
    return _PROVIDER_ALIASES.get((provider or "").strip().lower())


def probe(
    provider: str,
    api_key: str,
    *,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
    is_oauth: bool = False,
    transport: Optional[Transport] = None,
) -> ProbeResult:
    """Dispatch a probe by friendly provider name. Raises ``ValueError`` for an
    unknown provider so callers can map it to a clean exit."""
    canonical = normalize_provider(provider)
    if canonical == "anthropic":
        return probe_anthropic(
            api_key,
            base_url=base_url or DEFAULT_ANTHROPIC_BASE_URL,
            model=model or DEFAULT_ANTHROPIC_PROBE_MODEL,
            is_oauth=is_oauth,
            transport=transport,
        )
    if canonical == "openai-api":
        return probe_openai(
            api_key,
            base_url=base_url or DEFAULT_OPENAI_BASE_URL,
            model=model or DEFAULT_OPENAI_PROBE_MODEL,
            transport=transport,
        )
    raise ValueError(f"unknown provider for probe: {provider!r}")
