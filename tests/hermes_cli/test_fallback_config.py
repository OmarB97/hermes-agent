"""Tests for fallback entry credentials, ordering, and policy enforcement."""

from hermes_cli.fallback_config import (
    filter_fallback_chain_for_policy,
    get_configured_fallback_chain,
    get_fallback_chain,
    get_fallback_policy,
    resolve_entry_api_key,
)


class TestResolveEntryApiKey:
    def test_inline_api_key_wins(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        entry = {"provider": "custom", "api_key": "inline-key", "key_env": "FB_KEY"}
        assert resolve_entry_api_key(entry) == "inline-key"

    def test_key_env_resolves_from_environment(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"key_env": "FB_KEY"}) == "env-key"

    def test_api_key_env_alias(self, monkeypatch):
        monkeypatch.setenv("FB_ALIAS_KEY", "alias-key")
        assert resolve_entry_api_key({"api_key_env": "FB_ALIAS_KEY"}) == "alias-key"

    def test_unset_env_var_returns_none(self, monkeypatch):
        monkeypatch.delenv("FB_MISSING", raising=False)
        assert resolve_entry_api_key({"key_env": "FB_MISSING"}) is None

    def test_empty_env_var_returns_none(self, monkeypatch):
        monkeypatch.setenv("FB_EMPTY", "   ")
        assert resolve_entry_api_key({"key_env": "FB_EMPTY"}) is None

    def test_no_key_fields_returns_none(self):
        assert resolve_entry_api_key({"provider": "openrouter", "model": "glm"}) is None

    def test_non_dict_returns_none(self):
        assert resolve_entry_api_key(None) is None
        assert resolve_entry_api_key("nope") is None  # type: ignore[arg-type]

    def test_whitespace_inline_key_falls_through_to_env(self, monkeypatch):
        monkeypatch.setenv("FB_KEY", "env-key")
        assert resolve_entry_api_key({"api_key": "   ", "key_env": "FB_KEY"}) == "env-key"


def test_missing_policy_preserves_legacy_any_behavior():
    entries = [
        {"provider": "openrouter", "model": "remote"},
        {"provider": "custom", "model": "local", "base_url": "http://127.0.0.1:8000/v1"},
    ]
    cfg = {"fallback_providers": entries}
    assert get_fallback_policy(cfg) == "any"
    assert get_fallback_chain(cfg) == entries


def test_off_keeps_configured_order_but_has_no_eligible_routes():
    entries = [
        {"provider": "openrouter", "model": "first"},
        {"provider": "anthropic", "model": "second"},
    ]
    cfg = {"fallback_policy": "off", "fallback_providers": entries}
    assert get_configured_fallback_chain(cfg) == entries
    assert get_fallback_chain(cfg) == []


def test_local_only_filters_by_endpoint_not_model_name():
    entries = [
        {"provider": "openrouter", "model": "local-looking-name"},
        {"provider": "custom", "model": "cloud-looking-name", "base_url": "http://127.0.0.1:8000/v1"},
    ]
    cfg = {"fallback_policy": "local-only", "fallback_providers": entries}
    assert get_fallback_chain(cfg) == [entries[1]]


def test_local_only_rejects_builtin_cloud_provider_with_explicit_local_url():
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "anthropic", "model": "claude", "base_url": "http://localhost:9000/v1"}
        ],
    }
    assert get_fallback_chain(cfg) == []


def test_invalid_explicit_policy_fails_closed():
    cfg = {
        "fallback_policy": "ANY",
        "fallback_providers": [{"provider": "openrouter", "model": "remote"}],
    }
    assert get_fallback_policy(cfg) == "off"
    assert get_fallback_chain(cfg) == []


def test_supplied_init_chain_is_filtered_by_policy():
    entries = [
        {"provider": "openrouter", "model": "remote"},
        {"provider": "custom", "model": "local", "base_url": "http://127.0.0.1:8000/v1"},
    ]
    assert filter_fallback_chain_for_policy(entries, {"fallback_policy": "off"}) == []
    assert filter_fallback_chain_for_policy(entries, {"fallback_policy": "local-only"}) == [entries[1]]
    assert filter_fallback_chain_for_policy(entries, {"fallback_policy": "any"}) == entries
