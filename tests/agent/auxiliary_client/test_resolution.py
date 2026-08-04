"""Auxiliary-client provider/model resolution and client construction."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import logging
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from agent.auxiliary_client import (
    _NOUS_MODEL,
    get_text_auxiliary_client,
    get_available_vision_backends,
    resolve_vision_provider_client,
    resolve_provider_client,
    _normalize_aux_provider,
    _try_openrouter,
    _OPENROUTER_MODEL,
    OPENROUTER_BASE_URL,
    _resolve_auto,
    _resolve_task_provider_model,
    _pool_runtime_base_url,
)

from tests.agent.auxiliary_client.conftest import (
    _FakeAnthropicStream,
)


class TestResolveTaskProviderModel:
    @pytest.mark.parametrize(
        "provider",
        [
            "anthropic",
            "minimax-oauth",
            "nous",
            "openai-codex",
            "qwen-oauth",
            "xai-oauth",
        ],
    )
    def test_explicit_base_url_preserves_first_class_provider_identity(self, provider):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="moa_reference",
            provider=provider,
            model="test-model",
            base_url="https://provider.example/v1",
            api_key="resolved-token",
        )

        assert resolved_provider == provider
        assert model == "test-model"
        assert base_url == "https://provider.example/v1"
        assert api_key == "resolved-token"
        assert api_mode is None

    @pytest.mark.parametrize("provider", ["", "auto", "custom", "custom:local", "unknown-provider"])
    def test_explicit_base_url_without_first_class_provider_routes_as_custom(self, provider):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="moa_reference",
            provider=provider,
            model="test-model",
            base_url="https://provider.example/v1",
            api_key="resolved-token",
        )

        assert resolved_provider == "custom"
        assert model == "test-model"
        assert base_url == "https://provider.example/v1"
        assert api_key == "resolved-token"
        assert api_mode is None

    def test_direct_openai_alias_with_base_url_still_routes_as_custom(self):
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="vision",
            provider="openai",
            model="gpt-4o-mini",
            base_url="https://proxy.example/v1",
            api_key="sk-test",
        )

        assert resolved_provider == "custom"
        assert model == "gpt-4o-mini"
        assert base_url == "https://proxy.example/v1"
        assert api_key == "sk-test"
        assert api_mode is None

    def test_explicit_provider_adopts_configured_task_endpoint(self):
        """Explicit provider matching the configured one must not bypass
        auxiliary.<task>.base_url/api_key (#58515)."""
        task_config = {
            "provider": "custom",
            "model": "meta/llama-3.2-11b-vision-instruct",
            "base_url": "https://integrate.api.nvidia.com/v1",
            "api_key": "nvapi-secret",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
                model="meta/llama-3.2-11b-vision-instruct",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://integrate.api.nvidia.com/v1"
        assert api_key == "nvapi-secret"
        assert model == "meta/llama-3.2-11b-vision-instruct"
        assert api_mode is None

    def test_explicit_provider_adopts_endpoint_when_config_names_no_provider(self):
        task_config = {
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://nim.example/v1"
        assert api_key == "cfg-key"

    def test_explicit_first_class_provider_with_matching_config_keeps_identity(self):
        task_config = {
            "provider": "anthropic",
            "base_url": "https://anthropic-proxy.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="compression",
                provider="anthropic",
            )

        assert resolved_provider == "anthropic"
        assert base_url == "https://anthropic-proxy.example/v1"
        assert api_key == "cfg-key"

    def test_explicit_auto_provider_keeps_auto_resolution(self):
        """provider="auto" is a sentinel for "inherit / auto-detect" and must
        not adopt the configured endpoint — the auto chain owns resolution."""
        task_config = {
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="auto",
            )

        assert resolved_provider == "auto"
        assert base_url is None
        assert api_key is None

    def test_explicit_provider_differing_from_config_ignores_config_endpoint(self):
        """A caller forcing a different provider keeps full explicit-arg
        priority — the configured endpoint belongs to cfg_provider only."""
        task_config = {
            "provider": "custom",
            "base_url": "https://nim.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="nous",
            )

        assert resolved_provider == "nous"
        assert base_url is None
        assert api_key is None

    def test_explicit_provider_and_base_url_still_win_over_config(self):
        task_config = {
            "provider": "custom",
            "base_url": "https://configured.example/v1",
            "api_key": "cfg-key",
        }
        with patch("agent.auxiliary_client._get_auxiliary_task_config", return_value=task_config):
            resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
                task="vision",
                provider="custom",
                base_url="https://explicit.example/v1",
                api_key="explicit-key",
            )

        assert resolved_provider == "custom"
        assert base_url == "https://explicit.example/v1"
        assert api_key == "explicit-key"

    def test_explicit_provider_moa_unwraps_to_aggregator(self, monkeypatch):
        """An *explicit* `provider="moa"` arg (e.g. a per-task model override
        naming a MoA preset) must resolve to the preset's aggregator, not the
        literal "moa" string — mirrors #53827's fix for the implicit
        "main provider is moa" case in _resolve_auto(), which this function
        never went through."""
        preset = {
            "aggregator": {"provider": "openrouter", "model": "anthropic/claude-opus-4.8"},
        }
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: preset,
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
            provider="moa",
            model="opus-gpt",
            base_url="moa://local",
            api_key="moa-virtual-provider",
        )

        assert resolved_provider == "openrouter"
        assert model == "anthropic/claude-opus-4.8"
        # The virtual moa:// endpoint must not be forwarded to the aggregator.
        assert base_url is None
        assert api_key is None

    def test_config_provider_moa_unwraps_to_aggregator(self, monkeypatch):
        """`auxiliary.<task>.provider: moa` in config.yaml — the same crash,
        reached via the config path instead of an explicit call-time arg.
        Before the fix this returned ("moa", ...) verbatim, and
        resolve_provider_client() would then look up "moa" in
        PROVIDER_REGISTRY (which has no such entry, it's not a real HTTP
        provider), fail, and surface a "MOA_API_KEY environment variable"
        error for a provider that was never meant to be reached over the wire."""
        preset = {
            "aggregator": {"provider": "anthropic", "model": "claude-opus-4.8"},
        }
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"provider": "moa", "model": "opus-gpt"} if task == "title_generation" else {},
        )
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: preset,
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
        )

        assert resolved_provider == "anthropic"
        assert model == "claude-opus-4.8"
        assert base_url is None
        assert api_key is None

    def test_provider_moa_falls_back_to_literal_when_preset_resolution_fails(self, monkeypatch):
        """If the MoA preset can't be resolved (e.g. renamed/deleted), the
        function must not raise — it degrades to the pre-fix behavior
        (literal "moa") rather than crash resolve_provider_client() harder."""
        monkeypatch.setattr("agent.auxiliary_client._get_auxiliary_task_config", lambda task: {})
        monkeypatch.setattr(
            "hermes_cli.moa_config.resolve_moa_preset",
            lambda cfg, name: (_ for _ in ()).throw(KeyError("gone-preset")),
        )
        monkeypatch.setattr("hermes_cli.config.load_config", lambda: {"moa": {}})
        monkeypatch.setattr("hermes_cli.config.load_config_readonly", lambda: {"moa": {}})

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
            provider="moa",
            model="gone-preset",
        )

        assert resolved_provider == "moa"
        assert model == "gone-preset"

    def test_explicit_model_auto_sentinel_is_normalized(self):
        """MoA slots (agent/moa_loop.py's _slot_runtime) forward a preset's
        `model:` field as the explicit `model` kwarg here, not through
        auxiliary.<task> config. Only cfg_model was normalized before, so a
        MoA reference/aggregator slot configured with `model: auto` sent the
        literal string "auto" to the wire as a model id."""
        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            provider="anthropic",
            model="auto",
        )

        assert resolved_provider == "anthropic"
        assert model is None


class TestNormalizeAuxProvider:
    def test_maps_github_copilot_aliases(self):
        assert _normalize_aux_provider("github") == "copilot"
        assert _normalize_aux_provider("github-copilot") == "copilot"
        assert _normalize_aux_provider("github-models") == "copilot"

    def test_maps_github_copilot_acp_aliases(self):
        assert _normalize_aux_provider("github-copilot-acp") == "copilot-acp"
        assert _normalize_aux_provider("copilot-acp-agent") == "copilot-acp"


class TestBuildCodexClient:
    def test_pool_without_selected_entry_falls_back_to_auth_store(self):
        with (
            patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)),
            patch("agent.auxiliary_client._read_codex_access_token", return_value="codex-auth-token"),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.call_args.kwargs["api_key"] == "codex-auth-token"
        assert mock_openai.call_args.kwargs["base_url"] == "https://chatgpt.com/backend-api/codex"

    def test_rejects_missing_model(self):
        """Callers must pass an explicit model; no hardcoded default."""
        from agent.auxiliary_client import _build_codex_client

        client, model = _build_codex_client("")
        assert client is None
        assert model is None

    def test_cached_codex_client_rebuilds_when_pool_entry_changes(self):
        import agent.auxiliary_client as aux

        class _Entry:
            def __init__(self, entry_id, token):
                self.id = entry_id
                self.runtime_api_key = token
                self.runtime_base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def __init__(self):
                self.entry = _Entry("cred-a", "tok-a")

            def has_credentials(self):
                return True

            def current(self):
                return self.entry

            def peek(self):
                return self.entry

            def select(self):
                return self.entry

        pool = _Pool()
        client_a = MagicMock(name="codex-client-a")
        client_b = MagicMock(name="codex-client-b")

        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client.OpenAI", side_effect=[client_a, client_b]) as mock_openai,
        ):
            aux.shutdown_cached_clients()
            try:
                first_client, first_model = aux._get_cached_client("openai-codex", "gpt-5.4")
                pool.entry = _Entry("cred-b", "tok-b")
                second_client, second_model = aux._get_cached_client("openai-codex", "gpt-5.4")
            finally:
                aux.shutdown_cached_clients()

        assert first_client is not second_client
        assert first_model == "gpt-5.4"
        assert second_model == "gpt-5.4"
        assert mock_openai.call_count == 2


class TestResolveProviderClientUniversalModelFallback:
    """resolve_provider_client() picks a sensible model when callers pass none (#31845).

    Aux tasks (title generation, vision, session search, etc.) routinely
    reach this function without an explicit model — the user's main
    provider was picked via ``hermes model``, no per-task override is
    set, and the expectation is "just use my main model for side tasks
    too."  The resolver fills in ``model`` from a 3-step universal
    fallback before any provider branch runs:

        1. ``model`` argument           (caller knew what they wanted)
        2. provider's catalog default   (cheap aux model, if registered)
        3. user's main model            (``model.model`` in config.yaml)

    Pre-fix the OAuth providers (xai-oauth, openai-codex) returned
    ``(None, None)`` on an empty model — both lack a catalog default
    because their accepted-model lists drift on the backend.  That
    silent failure caused ``_resolve_auto`` to drop to its Step-2
    fallback chain (OpenRouter / Nous / etc.), so aux tasks billed
    against the wrong subscription.
    """

    def test_empty_model_for_oauth_provider_falls_back_to_main_model(self):
        """xai-oauth: no catalog default → uses main model."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="grok-4.3",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",  # xai-oauth has no catalog default
            ),
            patch(
                "agent.auxiliary_client._build_xai_oauth_aux_client",
                return_value=(MagicMock(), "grok-4.3"),
            ) as mock_build,
        ):
            client, model = resolve_provider_client("xai-oauth", "")

        assert client is not None, (
            "should not fall through when main model is set"
        )
        assert model == "grok-4.3"
        # The builder receives the main-model fallback, never the empty
        # string the caller passed.
        assert mock_build.call_args.args[0] == "grok-4.3"

    def test_empty_model_for_codex_also_uses_main_model(self):
        """openai-codex: symmetric with xai-oauth — same universal fallback."""
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                return_value="gpt-5.4",
            ),
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="",  # openai-codex has no catalog default either
            ),
            patch(
                "agent.auxiliary_client._build_codex_client",
                return_value=(MagicMock(), "gpt-5.4"),
            ) as mock_build,
            patch(
                "agent.auxiliary_client._select_pool_entry",
                return_value=(True, None),
            ),
        ):
            client, model = resolve_provider_client("openai-codex", "")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_build.call_args.args[0] == "gpt-5.4"

    def test_empty_model_for_catalog_provider_uses_catalog_default(self):
        """anthropic / nous / openrouter / etc.: catalog default wins
        over main model when no explicit model is passed.

        This preserves the original \"cheap aux model for direct API
        providers\" behaviour — users on anthropic for their main chat
        still get claude-haiku-4-5 for title generation, NOT their
        expensive chat model.  Step 2 of the universal fallback chain.
        """
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch(
                "agent.auxiliary_client._read_main_model",
                # Main model is the expensive opus; if this leaks into
                # aux it costs real money.
                return_value="claude-opus-4-6",
            ) as mock_read_main,
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="claude-haiku-4-5-20251001",
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                return_value=MagicMock(),
            ),
            patch(
                "agent.anthropic_adapter.resolve_anthropic_token",
                return_value="sk-ant-***",
            ),
            patch(
                "agent.auxiliary_client._read_nous_auth", return_value=None
            ),
        ):
            client, model = resolve_provider_client("anthropic", "")

        # Catalog default takes precedence — main_model was a no-op
        # because step 2 of the fallback chain already produced a model.
        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        mock_read_main.assert_not_called()

    def test_explicit_model_takes_precedence_over_fallbacks(self):
        """Step 1: caller-passed model wins.  Per-task config
        (``auxiliary.<task>.model``) routes here — when the user
        explicitly picks gemini-3-flash for title generation, that's
        what runs, not their main model.
        """
        from agent.auxiliary_client import resolve_provider_client

        with (
            patch("agent.auxiliary_client._read_main_model") as mock_read_main,
            patch(
                "agent.auxiliary_client._get_aux_model_for_provider",
                return_value="catalog-default-should-not-be-used",
            ),
            patch(
                "agent.auxiliary_client._build_xai_oauth_aux_client",
                return_value=(MagicMock(), "grok-4.20-multi-agent"),
            ) as mock_build,
        ):
            client, model = resolve_provider_client(
                "xai-oauth", "grok-4.20-multi-agent",
            )

        assert client is not None
        assert model == "grok-4.20-multi-agent"
        mock_read_main.assert_not_called()
        assert mock_build.call_args.args[0] == "grok-4.20-multi-agent"


class TestExplicitProviderRouting:
    """Test explicit provider selection bypasses auto chain correctly."""

    def test_explicit_anthropic_api_key(self, monkeypatch):
        """provider='anthropic' + regular API key should work with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api-regular-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client("anthropic")
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_explicit_openrouter_pool_exhausted_logs_precise_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "credential pool has no usable entries" in record.message
            for record in caplog.records
        )
        assert not any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

    def test_explicit_openrouter_missing_env_keeps_not_set_warning(self, monkeypatch, caplog):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = resolve_provider_client("openrouter")
        assert client is None
        assert model is None
        assert any(
            "OPENROUTER_API_KEY not set" in record.message
            for record in caplog.records
        )

    def test_try_openrouter_pool_exhausted_falls_back_to_env(self, monkeypatch):
        """Pool present but exhausted → fall through to OPENROUTER_API_KEY env var."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-fallback")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client

            client, model = _try_openrouter()

        assert client is mock_client
        assert model == _OPENROUTER_MODEL
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "sk-or-env-fallback"
        assert mock_openai.call_args.kwargs["base_url"] == OPENROUTER_BASE_URL

    def test_try_openrouter_pool_exhausted_no_env_marks_unhealthy(self, monkeypatch):
        """Pool exhausted AND no env var → final failure marks provider unhealthy."""
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("agent.auxiliary_client._mark_provider_unhealthy") as mock_mark, \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter()

        assert client is None
        assert model is None
        mock_openai.assert_not_called()
        mock_mark.assert_called_once_with("openrouter", ttl=60)


class TestGetTextAuxiliaryClient:
    """Test the full resolution chain for get_text_auxiliary_client."""

    def test_codex_pool_entry_takes_priority_over_auth_store(self):
        class _Entry:
            access_token = "pooled-codex-token"
            base_url = "https://chatgpt.com/backend-api/codex"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI"),
            patch("hermes_cli.auth._read_codex_tokens", side_effect=AssertionError("legacy codex store should not run")),
        ):
            from agent.auxiliary_client import _build_codex_client

            client, model = _build_codex_client("gpt-5.4")

        from agent.auxiliary_client import CodexAuxiliaryClient

        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.4"

    def test_returns_none_when_nothing_available(self, monkeypatch):
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        with patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._read_codex_access_token", return_value=None), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = get_text_auxiliary_client()
        assert client is None
        assert model is None

    def test_custom_endpoint_uses_codex_wrapper_when_runtime_requests_responses_api(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("https://api.openai.com/v1", "sk-test", "codex_responses")), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None), \
             patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=None), \
             patch("agent.auxiliary_client._read_main_model", return_value="gpt-5.3-codex"), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = get_text_auxiliary_client()

        from agent.auxiliary_client import CodexAuxiliaryClient
        assert isinstance(client, CodexAuxiliaryClient)
        assert model == "gpt-5.3-codex"
        assert mock_openai.call_args.kwargs["base_url"] == "https://api.openai.com/v1"
        assert mock_openai.call_args.kwargs["api_key"] == "sk-test"


class TestVisionClientFallback:
    """Vision client auto mode resolves known-good multimodal backends."""

    def test_vision_auto_includes_active_provider_when_configured(self, monkeypatch):
        """Active provider appears in available backends when credentials exist."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.auxiliary_client._read_main_provider", return_value="anthropic"),
            patch("agent.auxiliary_client._read_main_model", return_value="claude-sonnet-4"),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            backends = get_available_vision_backends()

        assert "anthropic" in backends

    def test_resolve_provider_client_returns_native_anthropic_wrapper(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "***")
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="***"),
        ):
            client, model = resolve_provider_client("anthropic")

        assert client is not None
        assert client.__class__.__name__ == "AnthropicAuxiliaryClient"
        assert model == "claude-haiku-4-5-20251001"

    def test_anthropic_auxiliary_client_aggregates_stream_response(self):
        from agent.auxiliary_client import AnthropicAuxiliaryClient

        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="streamed aux response")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
        messages_api = SimpleNamespace(
            stream=MagicMock(return_value=_FakeAnthropicStream(final_message)),
            create=MagicMock(return_value="raw event-stream text"),
        )
        real_client = SimpleNamespace(messages=messages_api)
        client = AnthropicAuxiliaryClient(
            real_client,
            "claude-sonnet-4-20250514",
            "sk-test",
            "https://sse-only.example/v1",
        )

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": "summarize"}],
            max_tokens=16,
        )

        messages_api.stream.assert_called_once()
        messages_api.create.assert_not_called()
        assert response.choices[0].message.content == "streamed aux response"
        assert response.usage.prompt_tokens == 3
        assert response.usage.completion_tokens == 4

    def test_anthropic_auxiliary_client_uses_model_output_limit_by_default(self):
        from agent.auxiliary_client import AnthropicAuxiliaryClient

        final_message = SimpleNamespace(
            content=[SimpleNamespace(type="text", text="aux response")],
            stop_reason="end_turn",
            usage=SimpleNamespace(input_tokens=3, output_tokens=4),
        )
        messages_api = SimpleNamespace(create=MagicMock())
        real_client = SimpleNamespace(messages=messages_api)
        captured_kwargs = {}

        def fake_create_anthropic_message(_client, kwargs, **_options):
            # ``**_options`` absorbs transport-level extras the aux client now
            # forwards (on_stream_event, ...); this test is about the resolved
            # max_tokens, not the call's option surface.
            captured_kwargs.update(kwargs)
            return final_message

        client = AnthropicAuxiliaryClient(
            real_client,
            "claude-opus-4-8",
            "sk-test",
            "https://api.anthropic.com",
        )

        with patch(
            "agent.anthropic_adapter.create_anthropic_message",
            side_effect=fake_create_anthropic_message,
        ):
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert response.choices[0].message.content == "aux response"
        # Behavior contract, not a frozen literal: a capless native-Anthropic
        # aux call must default to the model's native output ceiling (resolved
        # via _get_anthropic_max_output) rather than the old hidden 2000 cap.
        # Asserting against the resolver keeps this test alive across
        # model-table churn while still catching a regression to `or 2000`.
        from agent.anthropic_adapter import _get_anthropic_max_output

        expected_ceiling = _get_anthropic_max_output("claude-opus-4-8")
        assert expected_ceiling > 2000
        assert captured_kwargs["max_tokens"] == expected_ceiling


class TestStaleBaseUrlWarning:
    """_resolve_auto() warns when OPENAI_BASE_URL conflicts with config provider (#5161)."""

    def test_warns_when_openai_base_url_set_with_named_provider(self, monkeypatch, caplog):
        """Warning fires when OPENAI_BASE_URL is set but provider is a named provider."""
        import agent.auxiliary_client as mod
        # Reset the module-level flag so the warning fires
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Expected a warning about stale OPENAI_BASE_URL"
        assert mod._stale_base_url_warned is True


class TestVisionAutoSkipsKimiCoding:
    """_resolve_auto vision branch skips providers that have no vision on
    their main endpoint (e.g. Kimi Coding Plan /coding) and falls through
    to the aggregator chain instead of handing back a client that will 404
    on every request (#17076).
    """

    def test_kimi_coding_skipped_falls_through_to_openrouter(self, monkeypatch):
        """kimi-coding as main + vision auto → OpenRouter (not kimi)."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        # Guard: if the skip doesn't fire, _resolve_strict_vision_backend
        # and resolve_provider_client both would try kimi-coding — detect
        # either via the main-provider call and fail loud.
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding "
            "on the vision auto path"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )

        def fake_strict(provider, model=None):
            if provider == "openrouter":
                return fake_or_client, _NOUS_MODEL
            if provider == "nous":
                return None, None
            raise AssertionError(
                f"strict vision backend should not be called for {provider!r} "
                "when main provider is kimi-coding"
            )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            fake_strict,
        )

        provider, client, model = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client
        assert model == _NOUS_MODEL

    def test_kimi_coding_cn_skipped_too(self, monkeypatch):
        """Same skip applies to the CN variant."""
        fake_or_client = MagicMock(name="openrouter_client")

        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "kimi-coding-cn",
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_model", lambda: "kimi-code",
        )
        rpc_mock = MagicMock(side_effect=AssertionError(
            "resolve_provider_client should NOT be called for kimi-coding-cn"))
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client", rpc_mock,
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._resolve_strict_vision_backend",
            lambda p, m=None: (fake_or_client, "gemini")
            if p == "openrouter"
            else (None, None),
        )

        provider, client, _ = resolve_vision_provider_client()
        assert provider == "openrouter"
        assert client is fake_or_client

    def test_explicit_override_to_kimi_coding_still_honored(self, monkeypatch):
        """When a user *explicitly* requests kimi-coding for vision (e.g.
        they know what they're doing, or are running a future build that
        adds image_in capability to Kimi Code), the explicit path still
        routes to kimi-coding — only the auto branch applies the skip.
        """
        monkeypatch.setattr(
            "agent.auxiliary_client._read_main_provider", lambda: "openrouter",
        )
        fake_kimi_client = MagicMock(name="kimi_client")
        gcc_mock = MagicMock(return_value=(fake_kimi_client, "kimi-code"))
        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client", gcc_mock,
        )

        provider, client, model = resolve_vision_provider_client(
            provider="kimi-coding",
        )
        assert provider == "kimi-coding"
        assert client is fake_kimi_client
        gcc_mock.assert_called_once()

    def test_skip_set_covers_exactly_known_entries(self):
        """Guard against accidental widening of the skip list."""
        from agent.auxiliary_client import _PROVIDERS_WITHOUT_VISION
        assert _PROVIDERS_WITHOUT_VISION == frozenset({
            "kimi-coding",
            "kimi-coding-cn",
        })


def test_resolve_api_key_provider_skips_unconfigured_anthropic(monkeypatch):
    """_resolve_api_key_provider must not try anthropic when user never configured it."""
    from collections import OrderedDict
    from hermes_cli.auth import ProviderConfig

    # Build a minimal registry with only "anthropic" so the loop is guaranteed
    # to reach it without being short-circuited by earlier providers.
    fake_registry = OrderedDict({
        "anthropic": ProviderConfig(
            id="anthropic",
            name="Anthropic",
            auth_type="api_key",
            inference_base_url="https://api.anthropic.com",
            api_key_env_vars=("ANTHROPIC_API_KEY",),
        ),
    })

    called = []

    def mock_try_anthropic():
        called.append("anthropic")
        return None, None

    monkeypatch.setattr("agent.auxiliary_client._try_anthropic", mock_try_anthropic)
    monkeypatch.setattr("hermes_cli.auth.PROVIDER_REGISTRY", fake_registry)
    monkeypatch.setattr(
        "hermes_cli.auth.is_provider_explicitly_configured",
        lambda pid: False,
    )

    from agent.auxiliary_client import _resolve_api_key_provider
    _resolve_api_key_provider()

    assert "anthropic" not in called, \
        "_try_anthropic() should not be called when anthropic is not explicitly configured"


def test_pool_runtime_base_url_uses_nous_env_override(monkeypatch):
    entry = SimpleNamespace(
        provider="nous",
        runtime_base_url="https://inference-api.nousresearch.com/v1",
        inference_base_url="https://inference-api.nousresearch.com/v1",
        base_url="https://inference-api.nousresearch.com/v1",
    )
    monkeypatch.setenv("NOUS_INFERENCE_BASE_URL", "https://ai.wildebeest-newton.ts.net/v1")

    assert _pool_runtime_base_url(entry) == "https://ai.wildebeest-newton.ts.net/v1"


class TestOpenRouterExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_openrouter()."""

    def test_resolve_provider_client_passes_explicit_api_key_to_openrouter(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called with explicit_api_key for OpenRouter,
        the explicit key should be passed to the OpenAI client instead of falling back
        to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should NOT be used when explicit_api_key is provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key="explicit-pool-key",
            )

            # Verify a client was created
            assert client is not None
            # Verify the explicit key was used, not the env var fallback
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "explicit-pool-key", (
                f"Expected explicit_api_key to be passed, got: {call_kwargs['api_key']}"
            )
            assert call_kwargs["api_key"] != "env-fallback-key", (
                "Should NOT fall back to OPENROUTER_API_KEY when explicit_api_key is provided"
            )

    def test_resolve_provider_client_without_explicit_api_key_falls_back_to_env(
        self, monkeypatch
    ):
        """
        When resolve_provider_client() is called WITHOUT explicit_api_key for OpenRouter,
        it should fall back to OPENROUTER_API_KEY env var.
        """
        # Set up env var as fallback (should be used when explicit_api_key is NOT provided)
        monkeypatch.setenv("OPENROUTER_API_KEY", "env-fallback-key")

        # Mock OpenAI to capture the api_key used
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="openrouter-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="openrouter",
                explicit_api_key=None,
            )

            # Verify a client was created
            assert client is not None
            # Verify the env var fallback was used
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "env-fallback-key", (
                f"Expected env fallback key to be used when explicit_api_key is None, got: {call_kwargs['api_key']}"
            )


class TestAnthropicExplicitApiKey:
    """Test that explicit_api_key is correctly propagated to _try_anthropic().

    Parity with the OpenRouter fix in #18768: resolve_provider_client() passes
    explicit_api_key to _try_openrouter(), but the anthropic branch was not
    updated — _try_anthropic() always fell back to resolve_anthropic_token()
    even when an explicit key was supplied (e.g. from a fallback_model entry).
    """

    def test_try_anthropic_uses_explicit_api_key_over_env(self):
        """_try_anthropic(explicit_api_key) must use the supplied key, not the env fallback."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic("explicit-pool-key")
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-pool-key", (
            f"Expected explicit_api_key to be passed, got: {mock_build.call_args.args[0]}"
        )
        assert mock_build.call_args.args[0] != "env-fallback-key"

    def test_try_anthropic_without_explicit_key_falls_back_to_resolve(self):
        """Without explicit_api_key, _try_anthropic falls back to resolve_anthropic_token."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-fallback-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
        assert client is not None
        assert mock_build.call_args.args[0] == "env-fallback-key"

    def test_resolve_provider_client_passes_explicit_api_key_to_anthropic(self):
        """resolve_provider_client(provider='anthropic', explicit_api_key=...) must propagate the key."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="env-key"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            client, model = resolve_provider_client(
                provider="anthropic",
                explicit_api_key="explicit-fallback-key",
            )
        assert client is not None
        assert mock_build.call_args.args[0] == "explicit-fallback-key", (
            "resolve_provider_client must forward explicit_api_key to _try_anthropic()"
        )


class TestCustomEndpointApiKeyInheritance:
    """Issue #9318: when an auxiliary task uses provider=custom with an
    explicit base_url but empty api_key, the custom_key fallback chain must
    inherit ``model.api_key`` from config.yaml before falling to the
    ``no-key-required`` placeholder.

    Without this fix, users on self-hosted gateways who share the same
    endpoint+credentials for both the main model and auxiliary tasks get 401
    auth errors because the placeholder key is sent instead of the real one.

    Inheritance is host-gated: the main key is only inherited when the aux
    base_url points at the same host as the main model's base_url, so a
    misconfigured aux endpoint cannot leak the main credential cross-host.
    """

    def test_inherits_main_api_key_when_aux_key_empty(self, monkeypatch):
        """RED→GREEN: explicit_api_key is None, OPENAI_API_KEY unset →
        model.api_key from config.yaml must be used (same-host gateway)."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        fake_config = {
            "model": {
                "api_key": "sk-main-config-key",
                "base_url": "https://gw.example.com/v1",
                "default": "main-model",
            }
        }
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "sk-main-config-key", (
            "Custom endpoint with empty api_key should inherit "
            "model.api_key from config, got: "
            + repr(captured.get("api_key"))
        )

    def test_explicit_api_key_takes_precedence(self, monkeypatch):
        """explicit_api_key wins over config model.api_key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {"api_key": "sk-main-config-key"}}
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key="sk-explicit",
            )

        assert captured.get("api_key") == "sk-explicit"

    def test_local_server_falls_to_no_key_required(self, monkeypatch):
        """When no key is available anywhere (explicit, env, config), fall
        back to ``no-key-required`` for local servers (Ollama, etc.)."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {}}  # no api_key configured
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="http://localhost:11434/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"

    def test_runtime_override_key_is_used(self, monkeypatch):
        """When _RUNTIME_MAIN_API_KEY is set (by set_runtime_main), it takes
        precedence over config.yaml for the custom endpoint key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch.object(ac, "_RUNTIME_MAIN_API_KEY", "sk-runtime-key"), \
             patch.object(ac, "_RUNTIME_MAIN_BASE_URL", "https://gw.example.com/v1"), \
             patch("hermes_cli.config.load_config", return_value={"model": {}}), patch("hermes_cli.config.load_config_readonly", return_value={"model": {}}), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "sk-runtime-key"

    def test_cross_host_aux_endpoint_does_not_inherit_main_key(self, monkeypatch):
        """An aux base_url on a DIFFERENT host than the main model must NOT
        inherit model.api_key — that would leak the main credential to
        whatever host a misconfigured aux endpoint names. Falls back to the
        fail-safe no-key-required placeholder instead."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {
            "model": {
                "api_key": "sk-main-config-key",
                "base_url": "https://gw.example.com/v1",
            }
        }
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://other-host.example.net/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"

    def test_no_main_base_url_does_not_inherit_main_key(self, monkeypatch):
        """When the main model has no base_url (e.g. a first-class provider),
        there is no 'same gateway' to match — do not inherit the key."""
        import agent.auxiliary_client as ac

        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        fake_config = {"model": {"api_key": "sk-main-config-key"}}
        captured: dict = {}

        def _capture_create(**kwargs):
            captured.update(kwargs)
            return MagicMock()

        with patch("hermes_cli.config.load_config", return_value=fake_config), patch("hermes_cli.config.load_config_readonly", return_value=fake_config), \
             patch.object(ac, "_create_openai_client", side_effect=_capture_create):
            client, model = resolve_provider_client(
                "custom",
                model="test-model",
                explicit_base_url="https://gw.example.com/v1",
                explicit_api_key=None,
            )

        assert captured.get("api_key") == "no-key-required"


class TestMoaAggregatorSharedResolution:
    """The shared MoA→aggregator helper and the layers that consume it.

    Real-config tests: write an actual config.yaml under a temp HERMES_HOME
    and exercise the genuine load_config() → resolve_moa_preset() boundary —
    no mocking of the configuration-resolution chain.
    """

    @staticmethod
    def _write_moa_config(tmp_path, monkeypatch, default_preset="opus-gpt"):
        import yaml

        home = tmp_path / ".hermes"
        home.mkdir(exist_ok=True)
        (home / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "moa": {
                        "default_preset": default_preset,
                        "presets": {
                            "opus-gpt": {
                                "enabled": True,
                                "reference_models": [
                                    {"provider": "openrouter", "model": "openai/gpt-5.5"}
                                ],
                                "aggregator": {
                                    "provider": "openrouter",
                                    "model": "anthropic/claude-opus-4.8",
                                },
                            },
                            "nous-mix": {
                                "enabled": True,
                                "reference_models": [
                                    {"provider": "nous", "model": "hermes-4-70b"}
                                ],
                                "aggregator": {
                                    "provider": "nous",
                                    "model": "hermes-4-405b",
                                },
                            },
                        },
                    }
                }
            )
        )
        monkeypatch.setenv("HERMES_HOME", str(home))
        return home

    def test_real_config_explicit_task_provider_moa(self, tmp_path, monkeypatch):
        """auxiliary.<task>.provider: moa in a REAL config.yaml resolves to the
        aggregator through the genuine load_config()/resolve_moa_preset() path."""
        import yaml

        home = self._write_moa_config(tmp_path, monkeypatch)
        cfg = yaml.safe_load((home / "config.yaml").read_text())
        cfg["auxiliary"] = {"title_generation": {"provider": "moa", "model": "opus-gpt"}}
        (home / "config.yaml").write_text(yaml.safe_dump(cfg))

        resolved_provider, model, base_url, api_key, api_mode = _resolve_task_provider_model(
            task="title_generation",
        )

        assert resolved_provider == "openrouter"
        assert model == "anthropic/claude-opus-4.8"
        assert base_url is None
        assert api_key is None






    def test_main_agent_fallback_uses_aggregator_for_moa_main(self, tmp_path, monkeypatch):
        """_try_main_agent_model_fallback with a moa main resolves the
        aggregator instead of asking for a literal "moa" client."""
        from agent.auxiliary_client import _try_main_agent_model_fallback

        self._write_moa_config(tmp_path, monkeypatch)
        with patch("agent.auxiliary_client._read_main_provider", return_value="moa"), \
             patch("agent.auxiliary_client._read_main_model", return_value="opus-gpt"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            mock_client = MagicMock()
            mock_resolve.return_value = (mock_client, "anthropic/claude-opus-4.8")

            client, model, label = _try_main_agent_model_fallback("anthropic", task="compression")

        assert client is mock_client
        assert model == "anthropic/claude-opus-4.8"
        assert label == "main-agent(openrouter)"
        assert mock_resolve.call_args.kwargs["provider"] == "openrouter"
        assert mock_resolve.call_args.kwargs["model"] == "anthropic/claude-opus-4.8"
