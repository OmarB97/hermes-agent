from __future__ import annotations

import textwrap

from hermes_cli.timeouts import (
    get_provider_first_chunk_timeout,
    get_provider_request_timeout,
    get_provider_stale_timeout,
)


def _write_config(tmp_path, body: str) -> None:
    (tmp_path / "config.yaml").write_text(textwrap.dedent(body), encoding="utf-8")


def test_model_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: 30
            models:
              claude-opus-4.6:
                timeout_seconds: 120
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") == 120.0


def test_provider_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ollama-local:
            request_timeout_seconds: 300
        """,
    )

    assert get_provider_request_timeout("ollama-local", "qwen3:32b") == 300.0


def test_model_stale_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: 600
            models:
              gpt-5.4:
                stale_timeout_seconds: 1800
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") == 1800.0


def test_provider_stale_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: 900
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") == 900.0


def test_missing_timeout_returns_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            models:
              claude-opus-4.6:
                context_length: 200000
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("missing-provider", "claude-opus-4.6") is None


def test_invalid_timeout_values_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: "fast"
            models:
              claude-opus-4.6:
                timeout_seconds: -5
          ollama-local:
            request_timeout_seconds: -1
        """,
    )

    assert get_provider_request_timeout("anthropic", "claude-opus-4.6") is None
    assert get_provider_request_timeout("anthropic", "claude-sonnet-4.5") is None
    assert get_provider_request_timeout("ollama-local") is None


def test_invalid_stale_timeout_values_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          openai-codex:
            stale_timeout_seconds: "slow"
            models:
              gpt-5.4:
                stale_timeout_seconds: -1
        """,
    )

    assert get_provider_stale_timeout("openai-codex", "gpt-5.4") is None
    assert get_provider_stale_timeout("openai-codex", "gpt-5.5") is None


def test_model_first_chunk_timeout_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ds4:
            first_chunk_timeout_seconds: 1200
            models:
              deepseek-v4-flash-0731-ds4:
                first_chunk_timeout_seconds: 1800
        """,
    )

    assert get_provider_first_chunk_timeout(
        "ds4", "deepseek-v4-flash-0731-ds4"
    ) == 1800.0


def test_provider_first_chunk_timeout_used_when_no_model_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ds4:
            first_chunk_timeout_seconds: 1200
            models:
              deepseek-v4-flash-0731-ds4:
                stale_timeout_seconds: 300
        """,
    )

    assert get_provider_first_chunk_timeout(
        "ds4", "deepseek-v4-flash-0731-ds4"
    ) == 1200.0
    assert get_provider_first_chunk_timeout("ds4", "some-other-model") == 1200.0


def test_first_chunk_timeout_is_independent_of_stale_timeout(monkeypatch, tmp_path):
    """Two phases, two knobs. Declaring one must not imply the other.

    ``stale_timeout_seconds`` measures the gap BETWEEN chunks;
    ``first_chunk_timeout_seconds`` measures queue admission + model load +
    prefill of the whole prompt before any chunk exists. A lane can want
    minutes for one and seconds for the other.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ds4:
            stale_timeout_seconds: 300
        """,
    )

    assert get_provider_stale_timeout("ds4") == 300.0
    assert get_provider_first_chunk_timeout("ds4") is None


def test_missing_and_invalid_first_chunk_timeouts_return_none(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ds4:
            first_chunk_timeout_seconds: "soon"
            models:
              deepseek-v4-flash-0731-ds4:
                first_chunk_timeout_seconds: -5
          taro: {}
        """,
    )

    assert get_provider_first_chunk_timeout("ds4", "deepseek-v4-flash-0731-ds4") is None
    assert get_provider_first_chunk_timeout("taro") is None
    assert get_provider_first_chunk_timeout("missing-provider") is None


def test_bare_custom_first_chunk_timeout_resolves_via_base_url_attribution(monkeypatch, tmp_path):
    """The new knob inherits the bare-"custom" attribution fix for free.

    ``resolve_runtime_provider()`` reports every user-declared endpoint as the
    bare billing class ``"custom"``, so a getter that only did
    ``providers["custom"]`` would silently report "nothing configured" on
    exactly the local lanes this knob exists for.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ds4:
            api: "http://10.55.0.3:8000/v1"
            first_chunk_timeout_seconds: 1200
            models:
              deepseek-v4-flash-0731-ds4:
                first_chunk_timeout_seconds: 1800
        """,
    )

    assert get_provider_first_chunk_timeout(
        "custom", base_url="http://10.55.0.3:8000/v1"
    ) == 1200.0
    assert get_provider_first_chunk_timeout(
        "custom", "deepseek-v4-flash-0731-ds4", base_url="http://10.55.0.3:8000/v1"
    ) == 1800.0
    # "custom:<name>" already carries the entry name — no base_url needed.
    assert get_provider_first_chunk_timeout("custom:ds4") == 1200.0
    # An endpoint no configured entry owns stays unattributed.
    assert get_provider_first_chunk_timeout(
        "custom", base_url="http://10.99.99.99:9999/v1"
    ) is None


def test_bare_custom_provider_resolves_entry_timeouts_via_base_url_regression(monkeypatch, tmp_path):
    """Regression test: resolve_runtime_provider() reports every user-declared
    endpoint as the bare string "custom" — the resolved billing class, not a
    routable identity. Before the fix, get_provider_stale_timeout("custom", ...)
    looked up providers["custom"] (a key that never exists) and silently
    returned None, so providers.ai-router.stale_timeout_seconds was ignored
    and a deep-context local turn was killed by the default watchdog instead.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            request_timeout_seconds: 120
            stale_timeout_seconds: 900
        """,
    )

    assert get_provider_request_timeout(
        "custom", base_url="http://10.55.0.3:8000/v1"
    ) == 120.0
    assert get_provider_stale_timeout(
        "custom", base_url="http://10.55.0.3:8000/v1"
    ) == 900.0


def test_bare_custom_provider_model_override_wins_via_base_url_attribution(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            request_timeout_seconds: 120
            stale_timeout_seconds: 900
            models:
              deepseek-v4:
                timeout_seconds: 60
                stale_timeout_seconds: 1800
        """,
    )

    assert get_provider_request_timeout(
        "custom", "deepseek-v4", base_url="http://10.55.0.3:8000/v1"
    ) == 60.0
    assert get_provider_stale_timeout(
        "custom", "deepseek-v4", base_url="http://10.55.0.3:8000/v1"
    ) == 1800.0


def test_bare_custom_provider_unmatched_base_url_does_not_fall_back_to_config_model_provider(monkeypatch, tmp_path):
    """Mis-attribution guard: when base_url is present but owns no configured
    entry, the bare "custom" provider must stay unattributed — it must NOT
    inherit config.model.provider's timeouts, even though that names a real
    providers: entry. A genuinely ad-hoc endpoint has no configured timeouts.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            request_timeout_seconds: 120
            stale_timeout_seconds: 900
        model:
          provider: ai-router
          default: deepseek-v4
        """,
    )

    assert get_provider_request_timeout(
        "custom", base_url="http://10.99.99.99:9999/v1"
    ) is None
    assert get_provider_stale_timeout(
        "custom", base_url="http://10.99.99.99:9999/v1"
    ) is None


def test_custom_prefixed_provider_id_resolves_name_directly_with_and_without_base_url(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            stale_timeout_seconds: 900
        """,
    )

    # "custom:<name>" already carries the entry name — no base_url needed.
    assert get_provider_stale_timeout("custom:ai-router") == 900.0
    # An unrelated/mismatched base_url does not change the outcome, proving
    # this path never consults base_url matching at all.
    assert get_provider_stale_timeout(
        "custom:ai-router", base_url="http://unrelated.example:1234/v1"
    ) == 900.0


def test_bare_custom_provider_without_base_url_falls_back_to_config_model_provider(monkeypatch, tmp_path):
    """Documented behavior: the Desktop/TUI path where base_url was lost.
    With no base_url at all to reverse-lookup, a bare "custom" provider falls
    back to config.model.provider when that names a real providers: entry.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            stale_timeout_seconds: 900
        model:
          provider: ai-router
          default: deepseek-v4
        """,
    )

    assert get_provider_stale_timeout("custom") == 900.0


def test_named_provider_lookup_bypasses_custom_recovery_path(monkeypatch, tmp_path):
    """providers.anthropic resolves via the direct dict lookup and never
    touches the bare-"custom" recovery path added for the base_url
    attribution fix — named/built-in providers are unaffected by it."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          anthropic:
            request_timeout_seconds: 30
            stale_timeout_seconds: 45
        """,
    )

    import hermes_cli.timeouts as timeouts_mod

    def _fail_if_called(provider_id, base_url):
        raise AssertionError(
            f"_recover_custom_provider_key should not run for named provider {provider_id!r}"
        )

    monkeypatch.setattr(timeouts_mod, "_recover_custom_provider_key", _fail_if_called)

    assert get_provider_request_timeout("anthropic") == 30.0
    assert get_provider_stale_timeout("anthropic") == 45.0


def test_custom_provider_base_url_match_normalizes_trailing_slash_and_case(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        providers:
          ai-router:
            api: "http://10.55.0.3:8000/v1"
            stale_timeout_seconds: 900
        """,
    )

    assert get_provider_stale_timeout(
        "custom", base_url="HTTP://10.55.0.3:8000/V1/"
    ) == 900.0


def test_custom_providers_legacy_list_entry_resolves_via_base_url_attribution(monkeypatch, tmp_path):
    """find_custom_provider_identity() also matches the legacy custom_providers:
    LIST shape ({name, base_url, ...}), not just providers: dict entries keyed
    by api:. providers.ai-router carries no url field here, forcing the
    reverse lookup to fall through to the custom_providers: list match before
    _provider_config finds the timeout under providers.ai-router.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    _write_config(
        tmp_path,
        """\
        custom_providers:
          - name: ai-router
            base_url: "http://10.55.0.3:8000/v1"
        providers:
          ai-router:
            stale_timeout_seconds: 900
        """,
    )

    assert get_provider_stale_timeout(
        "custom", base_url="http://10.55.0.3:8000/v1"
    ) == 900.0


def test_anthropic_adapter_honors_timeout_kwarg():
    """build_anthropic_client(timeout=X) overrides the 900s default read timeout."""
    pytest = __import__("pytest")
    anthropic = pytest.importorskip("anthropic")  # skip if optional SDK missing
    from agent.anthropic_adapter import build_anthropic_client

    c_default = build_anthropic_client("sk-ant-dummy", None)
    c_custom = build_anthropic_client("sk-ant-dummy", None, timeout=45.0)
    c_invalid = build_anthropic_client("sk-ant-dummy", None, timeout=-1)

    # Default stays at 900s; custom overrides; invalid falls back to default
    assert c_default.timeout.read == 900.0
    assert c_custom.timeout.read == 45.0
    assert c_invalid.timeout.read == 900.0
    # Connect timeout always stays at 10s regardless
    assert c_default.timeout.connect == 10.0
    assert c_custom.timeout.connect == 10.0


def test_resolved_api_call_timeout_priority(monkeypatch, tmp_path):
    """AIAgent._resolved_api_call_timeout() honors config > env > default priority."""
    # Isolate HERMES_HOME
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    # Case A: config wins over env var
    _write_config(tmp_path, """\
        providers:
          openrouter:
            request_timeout_seconds: 77
            models:
              openai/gpt-4o-mini:
                timeout_seconds: 42
        """)
    monkeypatch.setenv("HERMES_API_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="openai/gpt-4o-mini",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    # Per-model override wins
    assert agent._resolved_api_call_timeout() == 42.0

    # Provider-level (different model, no per-model override)
    agent.model = "some/other-model"
    assert agent._resolved_api_call_timeout() == 77.0

    # Case B: no config → env wins
    _write_config(tmp_path, "")
    # Clear the cached config load
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="some/model",
        provider="openrouter",
        api_key="sk-dummy",
        base_url="https://openrouter.ai/api/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_timeout() == 999.0

    # Case C: no config, no env → 1800.0 default
    monkeypatch.delenv("HERMES_API_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_timeout() == 1800.0


def test_resolved_api_call_stale_timeout_priority(monkeypatch, tmp_path):
    """AIAgent stale timeout honors config > env > default priority."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")

    _write_config(tmp_path, """\
        providers:
          openai-codex:
            stale_timeout_seconds: 600
            models:
              gpt-5.4:
                stale_timeout_seconds: 1800
        """)
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "999")

    from run_agent import AIAgent
    agent = AIAgent(
        model="gpt-5.4",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent._resolved_api_call_stale_timeout_base() == (1800.0, False)

    agent.model = "gpt-5.5"
    assert agent._resolved_api_call_stale_timeout_base() == (600.0, False)

    _write_config(tmp_path, "")
    import importlib
    from hermes_cli import config as cfg_mod
    importlib.reload(cfg_mod)
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)
    import run_agent as ra_mod
    importlib.reload(ra_mod)

    agent2 = ra_mod.AIAgent(
        model="gpt-5.4",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    assert agent2._resolved_api_call_stale_timeout_base() == (999.0, False)

    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    assert agent2._resolved_api_call_stale_timeout_base() == (90.0, True)


def test_default_non_stream_stale_timeout_auto_disables_for_local_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    from run_agent import AIAgent
    agent = AIAgent(
        model="qwen3:32b",
        provider="ollama-local",
        api_key="sk-dummy",
        base_url="http://127.0.0.1:11434/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert agent._compute_non_stream_stale_timeout([]) == float("inf")


def test_explicit_non_stream_stale_timeout_is_honored_for_local_endpoints(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.setenv("HERMES_API_CALL_STALE_TIMEOUT", "300")

    from run_agent import AIAgent
    agent = AIAgent(
        model="qwen3:32b",
        provider="ollama-local",
        api_key="sk-dummy",
        base_url="http://127.0.0.1:11434/v1",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )

    assert agent._compute_non_stream_stale_timeout([]) == 300.0
