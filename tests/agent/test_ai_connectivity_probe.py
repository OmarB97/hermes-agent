"""Unit tests for the BYO-frontier "test my AI" probe (TASK 2).

Each ``error_class`` (invalid_key, expired_login, no_quota, network) is covered
with a MOCKED provider response — no real credential, no network. Also asserts
the success path, the secret fingerprint, the one-line rendering, and that the
secret is never echoed back in the result.
"""

from __future__ import annotations

import json

import pytest

from agent import ai_connectivity_probe as probe_mod


def _transport_returning(status: int, body: str):
    """Build a mock transport that records the request and returns a canned
    (status, body). Lets each test assert the real 1-token call shape."""
    seen = {}

    def _t(method, url, headers, body_bytes):
        seen["method"] = method
        seen["url"] = url
        seen["headers"] = headers
        seen["body"] = json.loads(body_bytes.decode("utf-8"))
        return status, body

    _t.seen = seen
    return _t


def _network_transport():
    def _t(method, url, headers, body_bytes):
        raise probe_mod._NetworkError("getaddrinfo failed")

    return _t


# ── success ────────────────────────────────────────────────────────────────

def test_anthropic_success_does_a_real_1_token_messages_call():
    t = _transport_returning(200, json.dumps({"content": [{"text": "p"}]}))
    res = probe_mod.probe("anthropic", "sk-ant-fake", transport=t)
    assert res.ok is True
    assert res.error_class is None
    assert res.mode == "api"
    assert res.model == probe_mod.DEFAULT_ANTHROPIC_PROBE_MODEL
    # Real inference call: POST /v1/messages with max_tokens=1.
    assert t.seen["method"] == "POST"
    assert t.seen["url"].endswith("/v1/messages")
    assert t.seen["body"]["max_tokens"] == 1
    assert t.seen["headers"]["x-api-key"] == "sk-ant-fake"
    assert t.seen["headers"]["anthropic-version"] == probe_mod.ANTHROPIC_VERSION
    # ✓ line and no secret echoed anywhere in the structured result.
    assert "✓ connected" in res.one_line()
    assert "sk-ant-fake" not in json.dumps(res.to_dict())


def test_openai_success_uses_chat_completions_bearer():
    t = _transport_returning(200, json.dumps({"choices": [{"message": {"content": "p"}}]}))
    res = probe_mod.probe("openai", "sk-openai-fake", transport=t)
    assert res.ok is True
    assert res.provider == "openai-api"
    assert t.seen["url"].endswith("/chat/completions")
    assert t.seen["body"]["max_tokens"] == 1
    assert t.seen["headers"]["Authorization"] == "Bearer sk-openai-fake"


def test_subscription_oauth_uses_bearer_not_x_api_key():
    t = _transport_returning(200, "{}")
    res = probe_mod.probe("anthropic", "oauth-tok", is_oauth=True, transport=t)
    assert res.ok is True
    assert res.mode == "subscription"
    assert t.seen["headers"]["Authorization"] == "Bearer oauth-tok"
    assert "x-api-key" not in t.seen["headers"]


# ── error_class: invalid_key ─────────────────────────────────────────────────

def test_invalid_key_401_maps_to_invalid_key():
    body = json.dumps({"error": {"type": "authentication_error", "message": "invalid x-api-key"}})
    t = _transport_returning(401, body)
    res = probe_mod.probe("anthropic", "sk-ant-bad", transport=t)
    assert res.ok is False
    assert res.error_class == probe_mod.ERROR_INVALID_KEY
    assert "✗ invalid key" in res.one_line()


def test_bare_403_with_no_refresh_phrasing_is_invalid_key():
    t = _transport_returning(403, "Forbidden")
    res = probe_mod.probe("openai", "sk-bad", transport=t)
    assert res.error_class == probe_mod.ERROR_INVALID_KEY


# ── error_class: expired_login ──────────────────────────────────────────────

def test_expired_oauth_login_maps_to_expired_login():
    body = json.dumps({"error": {"message": "OAuth token has expired, please sign in again"}})
    t = _transport_returning(401, body)
    res = probe_mod.probe("anthropic", "oauth-stale", is_oauth=True, transport=t)
    assert res.ok is False
    assert res.error_class == probe_mod.ERROR_EXPIRED_LOGIN
    assert "expired" in res.one_line().lower()


# ── error_class: no_quota ───────────────────────────────────────────────────

def test_insufficient_quota_429_maps_to_no_quota():
    body = json.dumps({"error": {"type": "insufficient_quota", "message": "You exceeded your current quota"}})
    t = _transport_returning(429, body)
    res = probe_mod.probe("openai", "sk-real-but-broke", transport=t)
    assert res.ok is False
    assert res.error_class == probe_mod.ERROR_NO_QUOTA


def test_402_payment_required_maps_to_no_quota():
    body = json.dumps({"error": {"message": "Your credit balance is too low"}})
    t = _transport_returning(402, body)
    res = probe_mod.probe("anthropic", "sk-ant-broke", transport=t)
    assert res.error_class == probe_mod.ERROR_NO_QUOTA


# ── error_class: network ────────────────────────────────────────────────────

def test_network_failure_maps_to_network():
    res = probe_mod.probe("anthropic", "sk-ant-x", transport=_network_transport())
    assert res.ok is False
    assert res.error_class == probe_mod.ERROR_NETWORK
    assert "network error" in res.one_line().lower()


def test_5xx_server_error_maps_to_network():
    t = _transport_returning(503, "upstream unavailable")
    res = probe_mod.probe("openai", "sk-x", transport=t)
    assert res.ok is False
    assert res.error_class == probe_mod.ERROR_NETWORK
    assert res.status == 503


# ── fingerprint + contract ──────────────────────────────────────────────────

def test_fingerprint_is_sha256_first16_and_secret_absent():
    t = _transport_returning(200, "{}")
    res = probe_mod.probe("anthropic", "sk-ant-fingerprint-me", transport=t)
    fp = res.secret_fingerprint
    assert fp is not None
    assert fp.startswith("sha256:")
    assert len(fp) == len("sha256:") + 16
    # Identical scheme to credential_persistence._fingerprint_value.
    import hashlib

    expected = "sha256:" + hashlib.sha256(b"sk-ant-fingerprint-me").hexdigest()[:16]
    assert fp == expected
    # The raw key never appears in the serialized contract.
    assert "sk-ant-fingerprint-me" not in json.dumps(res.to_dict())


def test_unknown_provider_raises():
    with pytest.raises(ValueError):
        probe_mod.probe("gemini", "key", transport=_transport_returning(200, "{}"))


def test_one_line_examples_match_design_contract():
    ok = probe_mod.ProbeResult(provider="anthropic", mode="api", ok=True, model="m", latency_ms=412)
    assert ok.one_line() == "Anthropic (API key): ✓ connected (412ms)"
    bad = probe_mod.ProbeResult(
        provider="openai-api", mode="api", ok=False, model="m", error_class=probe_mod.ERROR_INVALID_KEY
    )
    assert bad.one_line().startswith("OpenAI (API key): ✗ invalid key")
