from __future__ import annotations

from hermes_cli import fallback_config
from hermes_cli import models as models_mod
from hermes_cli.fallback_config import (
    filter_fallback_chain_for_policy,
    get_configured_fallback_chain,
    get_fallback_chain,
    get_fallback_policy,
)


def _promotions_config(**overrides):
    cfg = {
        "fallback_promotions": {
            "enabled": True,
            "providers": ["nous"],
            "position": "prepend",
        }
    }
    cfg["fallback_promotions"].update(overrides)
    return cfg


def test_missing_promotions_config_preserves_static_only_behavior(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
            }
        ],
    )

    assert get_fallback_chain({}) == []


def test_free_nous_promotion_entries_read_current_portal_recommendations(monkeypatch):
    calls = []

    def _fake_fetch(portal_base_url, timeout=5.0, **kwargs):
        calls.append((portal_base_url, timeout, kwargs))
        return {
            "freeRecommendedModels": [
                {"modelName": "stepfun/step-3.7-flash:free"},
                {"modelName": " stepfun/step-3.7-flash:free "},
                {"modelName": ""},
                {},
            ]
        }

    monkeypatch.setattr(models_mod, "_resolve_nous_portal_url", lambda: "https://portal.example")
    monkeypatch.setattr(models_mod, "fetch_nous_recommended_models", _fake_fetch)

    assert fallback_config._free_nous_promotion_entries() == [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": True,
            "source": "dynamic-free-promotion",
        }
    ]
    assert calls == [("https://portal.example", 1.5, {})]


def test_nous_free_promotions_prepend_to_static_fallbacks(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        ],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    ]

    chain = get_fallback_chain(cfg)

    assert chain == [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": True,
            "source": "dynamic-free-promotion",
        },
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"},
        {"provider": "openrouter", "model": "anthropic/claude-sonnet-4.6"},
    ]


def test_explicit_fallback_entry_wins_over_dynamic_duplicate(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [
            {
                "provider": "nous",
                "model": "stepfun/step-3.7-flash:free",
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        ],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [
        {
            "provider": "nous",
            "model": "stepfun/step-3.7-flash:free",
            "supports_tools": False,
            "note": "operator override",
        }
    ]

    assert get_fallback_chain(cfg) == cfg["fallback_providers"]


def test_promotions_can_append_after_static_chain(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    cfg = _promotions_config(position="append")
    cfg["fallback_providers"] = [{"provider": "taro", "model": "qwen3.6-27b-256k"}]

    assert get_fallback_chain(cfg) == [
        {"provider": "taro", "model": "qwen3.6-27b-256k"},
        {"provider": "nous", "model": "stepfun/step-3.7-flash:free"},
    ]


def test_promotions_skip_when_provider_is_not_authenticated(monkeypatch):
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: False)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    cfg = _promotions_config()
    cfg["fallback_providers"] = [{"provider": "opencode-zen", "model": "deepseek-v4-flash-free"}]

    assert get_fallback_chain(cfg) == [
        {"provider": "opencode-zen", "model": "deepseek-v4-flash-free"}
    ]


def test_env_override_can_disable_promotions(monkeypatch):
    monkeypatch.setenv("HERMES_FALLBACK_PROMOTIONS", "0")
    monkeypatch.setattr(fallback_config, "_provider_has_auth", lambda provider: True)
    monkeypatch.setattr(
        fallback_config,
        "_free_nous_promotion_entries",
        lambda: [{"provider": "nous", "model": "stepfun/step-3.7-flash:free"}],
    )

    assert get_fallback_chain(_promotions_config()) == []


def test_missing_policy_preserves_legacy_any_behavior():
    cfg = {
        "fallback_providers": [
            {"provider": "openrouter", "model": "remote"},
            {
                "provider": "custom",
                "model": "local",
                "base_url": "http://127.0.0.1:8000/v1",
            },
        ]
    }

    assert get_fallback_policy(cfg) == "any"
    assert get_fallback_chain(cfg) == cfg["fallback_providers"]


def test_off_keeps_configured_order_but_has_no_eligible_routes():
    cfg = {
        "fallback_policy": "off",
        "fallback_providers": [
            {"provider": "openrouter", "model": "first"},
            {"provider": "anthropic", "model": "second"},
        ],
    }

    assert get_configured_fallback_chain(cfg) == cfg["fallback_providers"]
    assert get_fallback_chain(cfg) == []


def test_local_only_uses_endpoint_metadata_not_model_names(monkeypatch):
    monkeypatch.setenv("LM_BASE_URL", "http://10.55.0.3:1234/v1")
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "opencode-zen", "model": "local-looking-name"},
            {"provider": "lmstudio", "model": "cloud-looking-name"},
            {"provider": "mystery", "model": "definitely-local"},
        ],
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "lmstudio", "model": "cloud-looking-name"}
    ]


def test_local_only_does_not_reclassify_builtin_cloud_provider_via_env(monkeypatch):
    monkeypatch.setenv("OPENCODE_ZEN_BASE_URL", "http://127.0.0.1:9999/v1")
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {"provider": "opencode-zen", "model": "remote-model"},
        ],
    }

    assert get_fallback_chain(cfg) == []


def test_local_only_rejects_builtin_anthropic_even_with_explicit_local_url():
    cfg = {
        "fallback_policy": "local-only",
        "fallback_providers": [
            {
                "provider": "anthropic",
                "model": "claude",
                "base_url": "http://localhost:9000/v1",
            }
        ],
    }

    assert get_fallback_chain(cfg) == []


def test_local_only_allows_explicit_user_provider_redefinition():
    cfg = {
        "fallback_policy": "local-only",
        "providers": {
            "anthropic": {
                "base_url": "http://10.55.0.3:9000/v1",
            }
        },
        "fallback_providers": [
            {"provider": "anthropic", "model": "local-compatible"}
        ],
    }

    assert get_fallback_chain(cfg) == [
        {"provider": "anthropic", "model": "local-compatible"}
    ]


def test_invalid_explicit_policy_fails_closed():
    cfg = {
        "fallback_policy": "ANY",
        "fallback_providers": [{"provider": "openrouter", "model": "remote"}],
    }

    assert get_fallback_policy(cfg) == "off"
    assert get_fallback_chain(cfg) == []


def test_local_only_never_fetches_remote_promotions(monkeypatch):
    monkeypatch.setattr(
        fallback_config,
        "_dynamic_free_promotion_entries",
        lambda _cfg: (_ for _ in ()).throw(AssertionError("promotion lookup ran")),
    )
    cfg = {
        **_promotions_config(),
        "fallback_policy": "local-only",
        "fallback_providers": [
            {
                "provider": "custom",
                "model": "local",
                "base_url": "http://localhost:8000/v1",
            }
        ],
    }

    assert get_fallback_chain(cfg) == cfg["fallback_providers"]


def test_supplied_init_chain_is_filtered_by_policy():
    entries = [
        {"provider": "openrouter", "model": "remote"},
        {
            "provider": "custom",
            "model": "local",
            "base_url": "http://127.0.0.1:8000/v1",
        },
    ]

    assert filter_fallback_chain_for_policy(
        entries,
        {"fallback_policy": "off"},
    ) == []
    assert filter_fallback_chain_for_policy(
        entries,
        {"fallback_policy": "local-only"},
    ) == [entries[1]]
    assert filter_fallback_chain_for_policy(
        entries,
        {"fallback_policy": "any"},
    ) == entries
