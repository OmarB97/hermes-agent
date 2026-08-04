"""Auxiliary-client credential resolution, OAuth, pool rotation and health caches."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import json
import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    resolve_provider_client,
    _NOUS_MODEL,
    _OPENROUTER_MODEL,
    _read_codex_access_token,
    _resolve_xai_oauth_for_aux,
    _try_openrouter,
)

from tests.agent.auxiliary_client.conftest import (
    _jwt_with_claims,
    _AuxAuth401,
    _DummyResponse,
    _FailingThenSuccessCompletions,
    _AsyncFailingThenSuccessCompletions,
)


class TestReadCodexAccessToken:
    def test_valid_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "tok-123", "refresh_token": "r-456"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "tok-123"

    def test_pool_without_selected_entry_falls_back_to_auth_store(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        valid_jwt = "eyJhbGciOiJSUzI1NiJ9.eyJleHAiOjk5OTk5OTk5OTl9.sig"
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(True, None)), \
             patch("hermes_cli.auth._read_codex_tokens", return_value={
                 "tokens": {"access_token": valid_jwt, "refresh_token": "refresh"}
             }):
            result = _read_codex_access_token()

        assert result == valid_jwt

    def test_missing_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({"version": 1, "providers": {}}))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            result = _read_codex_access_token()
        assert result is None

    def test_empty_token_returns_none(self, tmp_path, monkeypatch):
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "  ", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result is None

    def test_malformed_json_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text("{bad json")
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None

    def test_missing_tokens_key_returns_none(self, tmp_path):
        codex_dir = tmp_path / ".codex"
        codex_dir.mkdir()
        (codex_dir / "auth.json").write_text(json.dumps({"other": "data"}))
        with patch("agent.auxiliary_client.Path.home", return_value=tmp_path):
            result = _read_codex_access_token()
        assert result is None


    def test_expired_jwt_returns_none(self, tmp_path, monkeypatch):
        """Expired JWT tokens should be skipped so auto chain continues."""
        import base64
        import time as _time

        # Build a JWT with exp in the past
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            result = _read_codex_access_token()
        assert result is None, "Expired JWT should return None"

    def test_valid_jwt_returns_token(self, tmp_path, monkeypatch):
        """Non-expired JWT tokens should be returned."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) + 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        valid_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": valid_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == valid_jwt

    def test_non_jwt_token_passes_through(self, tmp_path, monkeypatch):
        """Non-JWT tokens (no dots) should be returned as-is."""
        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": "plain-token-no-jwt", "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == "plain-token-no-jwt"


class TestResolveXaiOAuthForAux:
    def test_uses_pool_backed_credentials_without_singleton(self, tmp_path, monkeypatch):
        """Auxiliary xAI OAuth must see pool-only credentials.

        ``hermes auth status`` already reports these as logged in; compression
        should not fall through to "no auxiliary provider configured" just
        because the singleton auth-store entry is absent.
        """
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.delenv("HERMES_XAI_BASE_URL", raising=False)
        monkeypatch.delenv("XAI_BASE_URL", raising=False)

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai123",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            DEFAULT_XAI_OAUTH_BASE_URL,
        )

    def test_pool_backed_credentials_honor_base_url_env_override(self, tmp_path, monkeypatch):
        from agent.credential_pool import AUTH_TYPE_OAUTH, PooledCredential, load_pool
        from hermes_cli.auth import DEFAULT_XAI_OAUTH_BASE_URL

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {},
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("HERMES_XAI_BASE_URL", "https://example.x.ai/v1/")

        pool = load_pool("xai-oauth")
        pool.add_entry(PooledCredential(
            provider="xai-oauth",
            id="xai456",
            label="pool-only",
            auth_type=AUTH_TYPE_OAUTH,
            priority=0,
            source="manual:xai_pkce",
            access_token="pool-access-token",
            refresh_token="pool-refresh-token",
            base_url=DEFAULT_XAI_OAUTH_BASE_URL,
        ))

        assert _resolve_xai_oauth_for_aux() == (
            "pool-access-token",
            "https://example.x.ai/v1",
        )


class TestAnthropicOAuthFlag:
    """Test that OAuth tokens get is_oauth=True in auxiliary Anthropic client."""

    def test_oauth_token_sets_flag(self, monkeypatch):
        """OAuth tokens (sk-ant-oat01-*) should create client with is_oauth=True."""
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-token")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            # The adapter inside should have is_oauth=True
            adapter = client.chat.completions
            assert adapter._is_oauth is True

    def test_api_key_no_oauth_flag(self, monkeypatch):
        """Regular API keys (sk-ant-api-*) should create client with is_oauth=False."""
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-api03-testkey1234"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic, AnthropicAuxiliaryClient
            client, model = _try_anthropic()
            assert client is not None
            assert isinstance(client, AnthropicAuxiliaryClient)
            adapter = client.chat.completions
            assert adapter._is_oauth is False

    def test_pool_entry_takes_priority_over_legacy_resolution(self):
        class _Entry:
            access_token = "sk-ant-oat01-pooled"
            base_url = "https://api.anthropic.com"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.anthropic_adapter.resolve_anthropic_token", side_effect=AssertionError("legacy path should not run")),
            patch("agent.anthropic_adapter.build_anthropic_client", return_value=MagicMock()) as mock_build,
        ):
            from agent.auxiliary_client import _try_anthropic

            client, model = _try_anthropic()

        assert client is not None
        assert model == "claude-haiku-4-5-20251001"
        assert mock_build.call_args.args[0] == "sk-ant-oat01-pooled"


class TestExpiredCodexFallback:
    """Test that expired Codex tokens don't block the auto chain."""

    def test_expired_codex_falls_through_to_next(self, tmp_path, monkeypatch):
        """When Codex token is expired, auto chain should skip it and try next provider."""
        import base64
        import time as _time

        # Expired Codex JWT
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Set up Anthropic as fallback
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-test-fallback")
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
            # Should NOT be Codex, should be Anthropic (or another available provider)
            assert not isinstance(client, type(None)), "Should find a provider after expired Codex"


    def test_expired_codex_openrouter_wins(self, tmp_path, monkeypatch):
        """With expired Codex + OpenRouter key, OpenRouter should win (1st in chain)."""
        import base64
        import time as _time

        # Belt-and-suspenders: _try_openrouter marks openrouter unhealthy
        # when OPENROUTER_API_KEY is absent (which the preceding test in
        # this class exercises).  The file-level _clean_env autouse fixture
        # clears the cache, but fixture ordering with the conftest
        # _hermetic_environment autouse can leave a narrow window where
        # the mark reappears.  Explicitly clear here so this test is
        # independent of run order.
        import agent.auxiliary_client as _aux_mod
        _aux_mod._aux_unhealthy_until.clear()
        _aux_mod._aux_unhealthy_logged_at.clear()

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-test-key")

        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import _resolve_auto
            client, model = _resolve_auto()
            assert client is not None
            # OpenRouter is 1st in chain, should win
            mock_openai.assert_called()

    def test_expired_codex_custom_endpoint_wins(self, tmp_path, monkeypatch):
        """With expired Codex + custom endpoint (Ollama), custom should win (3rd in chain)."""
        import base64
        import time as _time

        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"exp": int(_time.time()) - 3600}).encode()
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        expired_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": expired_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))

        # Simulate Ollama or custom endpoint
        with patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "sk-dummy")):
            with patch("agent.auxiliary_client.OpenAI") as mock_openai:
                mock_openai.return_value = MagicMock()
                from agent.auxiliary_client import _resolve_auto
                client, model = _resolve_auto()
                assert client is not None


    def test_hermes_oauth_file_sets_oauth_flag(self, monkeypatch):
        """OAuth-style tokens should get is_oauth=*** (token is not sk-ant-api-*)."""
        # Mock resolve_anthropic_token to return an OAuth-style token
        with patch("agent.anthropic_adapter.resolve_anthropic_token", return_value="sk-ant-oat-hermes-token"), \
             patch("agent.anthropic_adapter.build_anthropic_client") as mock_build, \
             patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)):
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
            assert client is not None, "Should resolve token"
            adapter = client.chat.completions
            assert adapter._is_oauth is True, "Non-sk-ant-api token should set is_oauth=True"

    def test_jwt_missing_exp_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid JSON but no exp claim should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256","typ":"JWT"}').rstrip(b"=").decode()
        payload_data = json.dumps({"sub": "user123"}).encode()  # no exp
        payload = base64.urlsafe_b64encode(payload_data).rstrip(b"=").decode()
        no_exp_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": no_exp_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == no_exp_jwt, "JWT without exp should pass through"

    def test_jwt_invalid_json_payload_passes_through(self, tmp_path, monkeypatch):
        """JWT with valid base64 but invalid JSON payload should pass through."""
        import base64
        header = base64.urlsafe_b64encode(b'{"alg":"RS256"}').rstrip(b"=").decode()
        payload = base64.urlsafe_b64encode(b"not-json-content").rstrip(b"=").decode()
        bad_jwt = f"{header}.{payload}.fakesig"

        hermes_home = tmp_path / "hermes"
        hermes_home.mkdir(parents=True, exist_ok=True)
        (hermes_home / "auth.json").write_text(json.dumps({
            "version": 1,
            "providers": {
                "openai-codex": {
                    "tokens": {"access_token": bad_jwt, "refresh_token": "r"},
                },
            },
        }))
        monkeypatch.setenv("HERMES_HOME", str(hermes_home))
        result = _read_codex_access_token()
        assert result == bad_jwt, "JWT with invalid JSON payload should pass through"

    def test_claude_code_oauth_env_sets_flag(self, monkeypatch):
        """CLAUDE_CODE_OAUTH_TOKEN env var should get is_oauth=True."""
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "sk-ant-oat-cc-test-token")
        monkeypatch.delenv("ANTHROPIC_TOKEN", raising=False)
        with patch("agent.anthropic_adapter.build_anthropic_client") as mock_build:
            mock_build.return_value = MagicMock()
            from agent.auxiliary_client import _try_anthropic
            client, model = _try_anthropic()
            assert client is not None
            adapter = client.chat.completions
            assert adapter._is_oauth is True


class TestAuxiliaryPoolAwareness:
    def test_try_nous_uses_pool_entry(self):
        pooled_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() + 3600),
        })

        class _Entry:
            access_token = "pooled-access-token"
            agent_key = pooled_token
            agent_key_expires_at = "2099-01-01T00:00:00+00:00"
            scope = "inference:invoke"
            inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None),
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert client is not None
        assert model == _NOUS_MODEL
        assert mock_openai.call_args.kwargs["api_key"] == pooled_token
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_try_nous_refreshes_stale_pool_entry(self):
        stale_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() - 60),
        })
        fresh_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() + 3600),
        })

        class _Entry:
            def __init__(self, token):
                self.access_token = "pooled-access-token"
                self.agent_key = token
                self.agent_key_expires_at = "2099-01-01T00:00:00+00:00"
                self.scope = "inference:invoke"
                self.inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            refreshed = False

            def has_credentials(self):
                return True

            def select(self):
                return _Entry(stale_token)

            def try_refresh_current(self):
                self.refreshed = True
                return _Entry(fresh_token)

        pool = _Pool()
        with (
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=None),
        ):
            from agent.auxiliary_client import _try_nous

            client, model = _try_nous()

        assert pool.refreshed is True
        assert client is not None
        assert model == _NOUS_MODEL
        assert mock_openai.call_args.kwargs["api_key"] == fresh_token
        assert mock_openai.call_args.kwargs["base_url"] == "https://inference.pool.example/v1"

    def test_resolve_nous_runtime_api_rejects_stale_pool_entry_when_refresh_fails(self):
        stale_token = _jwt_with_claims({
            "scope": "inference:invoke",
            "exp": int(time.time() - 60),
        })

        class _Entry:
            access_token = "pooled-access-token"
            agent_key = stale_token
            agent_key_expires_at = "2099-01-01T00:00:00+00:00"
            scope = "inference:invoke"
            inference_base_url = "https://inference.pool.example/v1"

        class _Pool:
            def has_credentials(self):
                return True

            def select(self):
                return _Entry()

            def try_refresh_current(self):
                return None

        with (
            patch("agent.auxiliary_client.load_pool", return_value=_Pool()),
            patch(
                "hermes_cli.auth.resolve_nous_runtime_credentials",
                side_effect=RuntimeError("no singleton auth"),
            ),
        ):
            from agent.auxiliary_client import _resolve_nous_runtime_api

            runtime = _resolve_nous_runtime_api()

        assert runtime is None

    def test_try_nous_uses_portal_recommendation_for_text(self):
        """When the Portal recommends a compaction model, _try_nous honors it."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value="minimax/minimax-m2.7") as mock_rec,
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            from agent.auxiliary_client import _try_nous

            mock_openai.return_value = MagicMock()
            client, model = _try_nous(vision=False)

        assert client is not None
        assert model == "minimax/minimax-m2.7"
        assert mock_rec.call_args.kwargs["vision"] is False

    def test_try_nous_uses_portal_recommendation_for_vision(self):
        """Vision tasks should ask for the vision-specific recommendation."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", return_value=_NOUS_MODEL) as mock_rec,
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous(vision=True)

        assert client is not None
        assert model == _NOUS_MODEL
        assert mock_rec.call_args.kwargs["vision"] is True

    def test_try_nous_falls_back_when_recommendation_lookup_raises(self):
        """If the Portal lookup throws, we must still return a usable model."""
        fresh_base = "https://inference-api.nousresearch.com/v1"
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value={"access_token": "***"}),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", fresh_base)),
            patch("hermes_cli.models.get_nous_recommended_aux_model", side_effect=RuntimeError("portal down")),
            patch("agent.auxiliary_client.OpenAI"),
        ):
            from agent.auxiliary_client import _try_nous
            client, model = _try_nous()

        assert client is not None
        assert model == _NOUS_MODEL

    def test_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create.side_effect = _Auth401("stale nous key")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_client.chat.completions.create.return_value = {"ok": True}

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client.OpenAI", return_value=fresh_client),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_nous_after_free_tier_block_when_account_paid(self):
        from hermes_cli.nous_account import NousPortalAccountInfo

        class _Payment404(Exception):
            status_code = 404

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create.side_effect = _Payment404(
            "model_not_supported_on_free_tier: model is not available on the free tier"
        )

        fresh_client = MagicMock()
        fresh_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_client.chat.completions.create.return_value = {"ok": True}

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client.OpenAI", return_value=fresh_client),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
            patch(
                "hermes_cli.nous_account.get_nous_portal_account_info",
                return_value=NousPortalAccountInfo(
                    logged_in=True,
                    source="account_api",
                    fresh=True,
                    paid_service_access=True,
                ),
            ),
        ):
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_retries_nous_after_401(self):
        class _Auth401(Exception):
            status_code = 401

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create = AsyncMock(side_effect=_Auth401("stale nous key"))

        fresh_async_client = MagicMock()
        fresh_async_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_async_client.chat.completions.create = AsyncMock(return_value={"ok": True})

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client._to_async_client", return_value=(fresh_async_client, "nous-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_async_client.chat.completions.create.await_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_nous_after_free_tier_block_when_account_paid(self):
        from hermes_cli.nous_account import NousPortalAccountInfo

        class _Payment404(Exception):
            status_code = 404

        stale_client = MagicMock()
        stale_client.base_url = "https://inference-api.nousresearch.com/v1"
        stale_client.chat.completions.create = AsyncMock(side_effect=_Payment404(
            "model_not_supported_on_free_tier: model is not available on the free tier"
        ))

        fresh_async_client = MagicMock()
        fresh_async_client.base_url = "https://inference-api.nousresearch.com/v1"
        fresh_async_client.chat.completions.create = AsyncMock(return_value={"ok": True})

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("nous", "nous-model", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", return_value=(stale_client, "nous-model")),
            patch("agent.auxiliary_client._to_async_client", return_value=(fresh_async_client, "nous-model")),
            patch("agent.auxiliary_client._validate_llm_response", side_effect=lambda resp, _task, **_kw: resp),
            patch("agent.auxiliary_client._resolve_nous_runtime_api", return_value=("fresh-agent-key", "https://inference-api.nousresearch.com/v1")),
            patch(
                "hermes_cli.nous_account.get_nous_portal_account_info",
                return_value=NousPortalAccountInfo(
                    logged_in=True,
                    source="account_api",
                    fresh=True,
                    paid_service_access=True,
                ),
            ),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert result == {"ok": True}
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_async_client.chat.completions.create.await_count == 1

    def test_cached_gmi_client_keeps_explicit_slash_model_override(self):
        import agent.auxiliary_client as aux

        fake_client = MagicMock()

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(fake_client, "google/gemini-3.1-flash-lite-preview"),
        ) as mock_resolve:
            aux.shutdown_cached_clients()
            try:
                client, model = aux._get_cached_client(
                    "gmi",
                    "google/gemini-3.1-flash-lite-preview",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
                assert client is fake_client
                assert model == "google/gemini-3.1-flash-lite-preview"

                client, model = aux._get_cached_client(
                    "gmi",
                    "openai/gpt-5.4-mini",
                    base_url="https://api.gmi-serving.com/v1",
                    api_key="gmi-key",
                )
            finally:
                aux.shutdown_cached_clients()

        assert client is fake_client
        assert model == "openai/gpt-5.4-mini"
        # A DIFFERENT model resolves its own client (model participates in the
        # cache key). This isolation is what stops two concurrent advisors on
        # the same provider/base_url/key (e.g. a MoA fan-out) from sharing — and
        # racing the lifecycle of — one cached client. Same-model reuse is still
        # a single resolve (verified elsewhere); distinct models => distinct
        # resolves.
        assert mock_resolve.call_count == 2


class TestAuxiliaryAuthRefreshRetry:
    def test_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _FailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-sync")

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-sync"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_call_llm_refreshes_codex_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("stale codex token")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-non-vision")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-non-vision"
        mock_refresh.assert_called_once_with("openai-codex")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_copilot_when_auto_routes_to_copilot_on_401(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.githubcopilot.com"
        stale_client.chat.completions.create.side_effect = _AuxAuth401(
            "IDE token expired: unauthorized: token expired"
        )

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.githubcopilot.com"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-auto-copilot")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.5"), (fresh_client, "gpt-5.5")]) as mock_get_client,
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
            patch("agent.auxiliary_client._evict_cached_clients") as mock_evict,
        ):
            resp = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hi"}],
                main_runtime={"provider": "copilot", "model": "gpt-5.5"},
            )

        assert resp.choices[0].message.content == "fresh-auto-copilot"
        mock_refresh.assert_called_once_with("copilot")
        mock_evict.assert_called_once_with("auto")
        assert mock_get_client.call_args_list[0].args[0] == "auto"
        assert mock_get_client.call_args_list[1].args[0] == "copilot"
        assert mock_get_client.call_args_list[1].args[1] == "gpt-5.5"
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_codex_when_auto_routes_to_codex_on_401(self):
        # Preflight compression's exact failure (#23670): provider auto →
        # Codex OAuth backend 401s → before the fix, no refresh was attempted
        # because resolved_provider stayed "auto".
        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("User not found.")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-auto-codex")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("auto", None, None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.5"), (fresh_client, "gpt-5.5")]) as mock_get_client,
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
            patch("agent.auxiliary_client._evict_cached_clients") as mock_evict,
        ):
            resp = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
                main_runtime={"provider": "openai-codex", "model": "gpt-5.5"},
            )

        assert resp.choices[0].message.content == "fresh-auto-codex"
        mock_refresh.assert_called_once_with("openai-codex")
        mock_evict.assert_called_once_with("auto")
        assert mock_get_client.call_args_list[1].args[0] == "openai-codex"
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    def test_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create.side_effect = _AuxAuth401("anthropic token expired")

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create.return_value = _DummyResponse("fresh-anthropic")

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.call_count == 1
        assert fresh_client.chat.completions.create.call_count == 1

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_codex_on_401_for_vision(self):
        failing_client = MagicMock()
        failing_client.base_url = "https://chatgpt.com/backend-api/codex"
        failing_client.chat.completions = _AsyncFailingThenSuccessCompletions()

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async"))

        with (
            patch(
                "agent.auxiliary_client.resolve_vision_provider_client",
                side_effect=[("openai-codex", failing_client, "gpt-5.4"), ("openai-codex", fresh_client, "gpt-5.4")],
            ),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="vision",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async"
        mock_refresh.assert_called_once_with("openai-codex")

    def test_refresh_provider_credentials_force_refreshes_anthropic_oauth_and_evicts_cache(self, monkeypatch):
        stale_client = MagicMock()
        cache_key = ("anthropic", False, None, None, None)

        monkeypatch.setenv("ANTHROPIC_TOKEN", "")
        monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "")

        with (
            patch("agent.auxiliary_client._client_cache", {cache_key: (stale_client, "claude-haiku-4-5-20251001", None)}),
            patch("agent.anthropic_adapter.read_claude_code_credentials", return_value={
                "accessToken": "expired-token",
                "refreshToken": "refresh-token",
                "expiresAt": 0,
            }),
            patch("agent.anthropic_adapter.refresh_anthropic_oauth_pure", return_value={
                "access_token": "fresh-token",
                "refresh_token": "refresh-token-2",
                "expires_at_ms": 9999999999999,
            }) as mock_refresh_oauth,
            patch("agent.anthropic_adapter._write_claude_code_credentials") as mock_write,
        ):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("anthropic") is True

        mock_refresh_oauth.assert_called_once_with("refresh-token", use_json=False)
        mock_write.assert_called_once_with("fresh-token", "refresh-token-2", 9999999999999)
        stale_client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_call_llm_refreshes_anthropic_on_401_for_non_vision(self):
        stale_client = MagicMock()
        stale_client.base_url = "https://api.anthropic.com"
        stale_client.chat.completions.create = AsyncMock(side_effect=_AuxAuth401("anthropic token expired"))

        fresh_client = MagicMock()
        fresh_client.base_url = "https://api.anthropic.com"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("fresh-async-anthropic"))

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("anthropic", "claude-haiku-4-5-20251001", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "claude-haiku-4-5-20251001"), (fresh_client, "claude-haiku-4-5-20251001")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=True) as mock_refresh,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="anthropic",
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "fresh-async-anthropic"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_client.chat.completions.create.await_count == 1
        assert fresh_client.chat.completions.create.await_count == 1

    def test_refresh_provider_credentials_remints_vertex_token_and_evicts_cache(self):
        """Vertex tokens live ~1h; on a long-running gateway the cached
        auxiliary client's bearer token expires mid-session and 401s.
        _refresh_provider_credentials("vertex") must re-mint the token via
        the adapter (which refreshes in place when near expiry) and evict
        the stale cached client so the next call rebuilds with a fresh one —
        previously there was no "vertex" branch here at all, so this fell
        through to the final `return False` and the stale client (and its
        dead token) stayed cached until process restart."""
        stale_client = MagicMock()
        cache_key = ("vertex", False, None, None, None)

        with (
            patch("agent.auxiliary_client._client_cache", {cache_key: (stale_client, _NOUS_MODEL, None)}),
            patch(
                "agent.vertex_adapter.get_vertex_config",
                return_value=("ya29.FRESH", "https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi"),
            ) as mock_get_config,
        ):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("vertex") is True

        mock_get_config.assert_called_once()
        stale_client.close.assert_called_once()

    def test_refresh_provider_credentials_vertex_returns_false_when_unminted(self):
        """No usable token/base_url (e.g. ADC and the service-account file
        both failed) — refresh must report failure, not silently evict and
        pretend the client is fixed."""
        with patch("agent.vertex_adapter.get_vertex_config", return_value=(None, None)):
            from agent.auxiliary_client import _refresh_provider_credentials

            assert _refresh_provider_credentials("vertex") is False

    def test_resolve_provider_client_vertex_none_when_no_credentials(self):
        with patch("agent.vertex_adapter.has_vertex_credentials", return_value=False):
            client, model = resolve_provider_client("vertex", _NOUS_MODEL)

        assert client is None
        assert model is None


class TestAuxiliaryPoolRotationRetry:
    def test_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create.side_effect = [rate_err, rate_err]

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create.return_value = _DummyResponse("rotated-sync")

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-sync"
        assert stale_client.chat.completions.create.call_count == 2
        assert fresh_client.chat.completions.create.call_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_call_llm_rotates_explicit_codex_pool_on_429(self):
        rate_err = Exception("usage limit reached")
        rate_err.status_code = 429

        stale_client = MagicMock()
        stale_client.base_url = "https://chatgpt.com/backend-api/codex"
        stale_client.chat.completions.create = AsyncMock(side_effect=[rate_err, rate_err])

        fresh_client = MagicMock()
        fresh_client.base_url = "https://chatgpt.com/backend-api/codex"
        fresh_client.chat.completions.create = AsyncMock(return_value=_DummyResponse("rotated-async"))

        class _Pool:
            def __init__(self):
                self.rotate_calls = []

            def has_credentials(self):
                return True

            def try_refresh_current(self):
                return None

            def mark_exhausted_and_rotate(self, **kwargs):
                self.rotate_calls.append(kwargs)
                return SimpleNamespace(id="cred-b")

        pool = _Pool()

        with (
            patch("agent.auxiliary_client._resolve_task_provider_model", return_value=("openai-codex", "gpt-5.4", None, None, None)),
            patch("agent.auxiliary_client._get_cached_client", side_effect=[(stale_client, "gpt-5.4"), (fresh_client, "gpt-5.4")]),
            patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False),
            patch("agent.auxiliary_client.load_pool", return_value=pool),
            patch("agent.auxiliary_client._try_payment_fallback") as mock_fallback,
        ):
            resp = await async_call_llm(
                task="compression",
                provider="openai-codex",
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hi"}],
            )

        assert resp.choices[0].message.content == "rotated-async"
        assert stale_client.chat.completions.create.await_count == 2
        assert fresh_client.chat.completions.create.await_count == 1
        assert len(pool.rotate_calls) == 1
        assert pool.rotate_calls[0]["status_code"] == 429
        mock_fallback.assert_not_called()


class TestAuxUnhealthyCache:
    """Recently-402'd providers are skipped on subsequent aux calls.

    Without this, every compression / title-gen / session-search call on a
    long session retries a depleted OpenRouter (~1 RTT to 402) before
    falling back to the next provider. The TTL cache hides the unhealthy
    provider for ``_AUX_UNHEALTHY_TTL_SECONDS`` so the chain skips it.
    """

    def setup_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def teardown_method(self):
        from agent.auxiliary_client import _reset_aux_unhealthy_cache
        _reset_aux_unhealthy_cache()

    def test_mark_then_skip(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        assert _is_provider_unhealthy("openrouter") is False
        _mark_provider_unhealthy("openrouter")
        assert _is_provider_unhealthy("openrouter") is True

    def test_ttl_expiry_evicts(self):
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
            _aux_unhealthy_until,
        )
        _mark_provider_unhealthy("openrouter", ttl=0.01)
        assert _is_provider_unhealthy("openrouter") is True
        import time
        time.sleep(0.02)
        # Lazy eviction: first lookup after expiry returns False AND removes the entry.
        assert _is_provider_unhealthy("openrouter") is False
        assert "openrouter" not in _aux_unhealthy_until

    def test_alias_normalization(self):
        """'codex' should normalize to 'openai-codex' so the cache lookup
        matches the chain label."""
        from agent.auxiliary_client import (
            _mark_provider_unhealthy,
            _is_provider_unhealthy,
        )
        _mark_provider_unhealthy("codex")
        assert _is_provider_unhealthy("openai-codex") is True

    def test_resolve_auto_skips_unhealthy_step2(self):
        """_resolve_auto Step-2 chain skips unhealthy providers."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark OpenRouter unhealthy → chain should skip it and pick nous.
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value=""), \
             patch("agent.auxiliary_client._read_main_model", return_value=""), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "nous-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        assert client is nous_client
        assert model == "nous-model"
        # The skipped provider's _try_* should NOT have been called at all.
        or_try.assert_not_called()

    def test_resolve_auto_skips_unhealthy_main_in_step1(self):
        """Step-1 also consults the unhealthy cache so a depleted main
        provider doesn't burn a 402 RTT every aux call. Falls through to
        Step-2 chain (which also respects the cache)."""
        from agent.auxiliary_client import (
            _resolve_auto,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        _mark_provider_unhealthy("openrouter")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4.6"), \
             patch("agent.auxiliary_client.resolve_provider_client") as step1, \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model = _resolve_auto()
        # Step-1 was bypassed — resolve_provider_client never invoked
        step1.assert_not_called()
        # Step-2 also skipped openrouter and landed on nous
        or_try.assert_not_called()
        assert client is nous_client

    def test_payment_fallback_skips_unhealthy(self):
        """_try_payment_fallback also consults the unhealthy cache so a 402
        on OpenRouter doesn't cause a second OR call within the same chain
        iteration if it gets re-entered."""
        from agent.auxiliary_client import (
            _try_payment_fallback,
            _mark_provider_unhealthy,
        )
        nous_client = MagicMock()
        # Mark BOTH the failed provider (openrouter) and a sibling (custom)
        # unhealthy. The chain should still find nous.
        _mark_provider_unhealthy("local/custom")
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._try_openrouter") as or_try, \
             patch("agent.auxiliary_client._try_nous", return_value=(nous_client, "n-model")), \
             patch("agent.auxiliary_client._try_custom_endpoint") as custom_try, \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is nous_client
        assert label == "nous"
        # OR is skipped via skip_chain_labels (failed provider), custom via unhealthy cache.
        or_try.assert_not_called()
        custom_try.assert_not_called()

    def test_call_llm_marks_provider_unhealthy_on_402(self, monkeypatch):
        """A 402 from call_llm causes the provider to be marked unhealthy
        so the next call skips it instead of re-trying the same depleted
        endpoint."""
        from agent.auxiliary_client import (
            call_llm,
            _is_provider_unhealthy,
        )
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        # base_url tells _recoverable_pool_provider() that this is OpenRouter
        # (resolved_provider="auto" doesn't carry that information by itself).
        primary_client.base_url = "https://openrouter.ai/api/v1/"
        err = Exception("Payment Required: insufficient credits")
        err.status_code = 402
        primary_client.chat.completions.create.side_effect = err

        nous_client = MagicMock()
        nous_resp = MagicMock()
        nous_resp.choices = [MagicMock(message=MagicMock(content="ok"))]
        nous_client.chat.completions.create.return_value = nous_resp

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, _NOUS_MODEL)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", _NOUS_MODEL, None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(nous_client, "n-model", "nous")), \
             patch("agent.auxiliary_client._build_call_kwargs",
                    return_value={"model": "n-model", "messages": [{"role": "user", "content": "hi"}]}):
            assert _is_provider_unhealthy("openrouter") is False
            call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hi"}],
            )
            # After the 402, OpenRouter is in the unhealthy cache.
            assert _is_provider_unhealthy("openrouter") is True


class TestAuxiliaryClientPoisonedCacheEviction:
    """Connection/timeout errors must evict the cached aux client.

    Otherwise the next auxiliary call (compression retry, memory flush,
    background review) reuses the closed httpx transport and fails with
    ``Connection error`` even though the main provider route is healthy.
    See https://github.com/NousResearch/hermes-agent/issues/23432.
    """

    def test_evict_cached_client_instance_drops_direct_match(self):
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
        )

        target = MagicMock(name="target_client")
        other = MagicMock(name="other_client")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openrouter", False, None, None, None)] = (target, "x", None)
            _client_cache[("anthropic", False, None, None, None)] = (other, "y", None)
        try:
            assert _evict_cached_client_instance(target) is True
            assert ("openrouter", False, None, None, None) not in _client_cache
            assert ("anthropic", False, None, None, None) in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_walks_codex_wrapper(self):
        """Closing the underlying OpenAI client must evict the Codex shim."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (wrapper, "gpt-5.5", None)
        try:
            # Eviction by the inner OpenAI client must remove the wrapper entry.
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_evict_cached_client_instance_handles_none_and_misses(self):
        from agent.auxiliary_client import _evict_cached_client_instance

        assert _evict_cached_client_instance(None) is False
        assert _evict_cached_client_instance(MagicMock()) is False

    def test_evict_cached_client_instance_walks_async_wrapper(self):
        """async_mode is part of the cache key so sync and async share the same
        underlying OpenAI client across two distinct cache entries. A single
        timeout that closes the leaf must evict BOTH — otherwise the async
        entry survives, keeps reusing the dead transport, and every async
        aux call (compression, vision, session_search) fails fast with
        'Connection error' until gateway restart even while the sync route
        recovers.

        Regression for the async-side gap left by #23482, which fixed the
        sync wrapper's _real_client walk but missed the async wrappers.
        """
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock, _evict_cached_client_instance,
            CodexAuxiliaryClient, AsyncCodexAuxiliaryClient,
        )

        real = SimpleNamespace(api_key="k", base_url="https://chatgpt.com/backend-api/codex",
                               responses=SimpleNamespace(stream=lambda **k: None),
                               close=lambda: None)
        sync_wrapper = CodexAuxiliaryClient(real, "gpt-5.5")
        async_wrapper = AsyncCodexAuxiliaryClient(sync_wrapper)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[("openai-codex", False, None, None, None)] = (sync_wrapper, "gpt-5.5", None)
            _client_cache[("openai-codex", True, None, None, None)] = (async_wrapper, "gpt-5.5", None)
        try:
            assert _evict_cached_client_instance(real) is True
            assert ("openai-codex", False, None, None, None) not in _client_cache
            assert ("openai-codex", True, None, None, None) not in _client_cache, (
                "async cache entry survived eviction — wrapper is missing _real_client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_codex_timeout_evicts_cached_wrapper(self):
        """The timeout closer evicts the cache entry that wraps the closed client."""
        from agent.auxiliary_client import (
            _client_cache, _client_cache_lock,
            _CodexCompletionsAdapter, CodexAuxiliaryClient,
        )

        class _SlowAliveCreateStream:
            def __iter__(self):
                for _ in range(20):
                    time.sleep(0.01)
                    yield SimpleNamespace(type="response.in_progress")

            def close(self): pass

        closed = {"flag": False}

        class FakeClient:
            def __init__(self):
                self.responses = SimpleNamespace(create=lambda **k: _SlowAliveCreateStream())
                self.api_key = "k"
                self.base_url = "https://chatgpt.com/backend-api/codex"

            def close(self):
                closed["flag"] = True

        fake_real = FakeClient()
        wrapper = CodexAuxiliaryClient(fake_real, "gpt-5.5")
        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (wrapper, "gpt-5.5", None)
        try:
            adapter = _CodexCompletionsAdapter(fake_real, "gpt-5.5")
            with pytest.raises(TimeoutError):
                adapter.create(
                    messages=[{"role": "user", "content": "x"}],
                    timeout=0.05,
                )
            assert closed["flag"] is True, "timeout closer must close inner client"
            assert cache_key not in _client_cache, (
                "timeout closer must evict cache entry that wraps the closed client"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    def test_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        """Connection error on an explicit provider must drop the cached client.

        Reporter scenario: ``auxiliary.compression.provider: main`` (resolves
        to ``openai-codex``).  After #26803, capacity errors (payment/quota/
        connection) DO trigger fallback even on explicit providers — so we
        also stub ``_try_payment_fallback`` to ``(None, None, "")`` so the
        connection error re-raises after eviction instead of escaping into
        a real network call.  The contract under test is cache eviction,
        not the fallback gate.
        """
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create.side_effect = ConnectionError("transport closed")

        cache_key = ("openai-codex", False, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ), patch(
                "agent.auxiliary_client._try_payment_fallback",
                return_value=(None, None, ""),
            ), patch(
                "agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0
            ):
                with pytest.raises(ConnectionError):
                    call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache, (
                "connection error must evict cached client so the next call rebuilds"
            )
        finally:
            with _client_cache_lock:
                _client_cache.clear()

    @pytest.mark.asyncio
    async def test_async_call_llm_evicts_on_connection_error_with_explicit_provider(self):
        from agent.auxiliary_client import _client_cache, _client_cache_lock

        poisoned = MagicMock(name="poisoned_async_client")
        poisoned.base_url = "https://chatgpt.com/backend-api/codex"
        poisoned.chat.completions.create = AsyncMock(side_effect=ConnectionError("transport closed"))

        cache_key = ("openai-codex", True, None, None, None)
        with _client_cache_lock:
            _client_cache.clear()
            _client_cache[cache_key] = (poisoned, "gpt-5.5", None)

        try:
            with patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openai-codex", "gpt-5.5", None, None, None),
            ), patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(poisoned, "gpt-5.5"),
            ), patch(
                "agent.auxiliary_client._try_payment_fallback",
                return_value=(None, None, ""),
            ):
                with pytest.raises(ConnectionError):
                    await async_call_llm(
                        task="compression",
                        messages=[{"role": "user", "content": "x"}],
                    )
            assert cache_key not in _client_cache
        finally:
            with _client_cache_lock:
                _client_cache.clear()


class TestOpenRouterPaidLaneGuard:
    """Issue #75803: auxiliary auto-chain OpenRouter fallback must be
    configurable and never silently engage a PAID model."""

    def test_free_only_skips_paid_default_model(self, monkeypatch):
        """free_only=true + default (paid) model → OpenRouter skipped."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {"free_only": True}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter()
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

    def test_free_only_allows_free_model(self, monkeypatch):
        """free_only=true + :free model → OpenRouter used with that model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly",
                   return_value={"auxiliary": {"free_only": True,
                                              "openrouter_model": "nvidia/nemotron-3-ultra-550b-a55b:free"}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            client, model = _try_openrouter()
        assert client is mock_client
        assert model == "nvidia/nemotron-3-ultra-550b-a55b:free"

    def test_configured_model_overrides_hardcoded_default(self, monkeypatch):
        """auxiliary.openrouter_model replaces _OPENROUTER_MODEL."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly",
                   return_value={"auxiliary": {"openrouter_model": "some/vendor-model"}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            client, model = _try_openrouter()
        assert client is mock_client
        assert model == "some/vendor-model"

    def test_explicit_caller_model_respects_free_only(self, monkeypatch):
        """Auxiliary.<task>.model (explicit) is also gated by free_only."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {"free_only": True}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            client, model = _try_openrouter(model="google/gemini-3.6-flash")
        assert client is None
        assert model is None
        mock_openai.assert_not_called()

    def test_paid_lane_warns_once(self, monkeypatch, caplog):
        """Engaging the default paid model logs a WARNING (once per model)."""
        import logging
        from agent.auxiliary_client import _paid_lane_warned
        _paid_lane_warned.discard(_OPENROUTER_MODEL)
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_client = MagicMock(name="openrouter_client")
            mock_openai.return_value = mock_client
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                client, model = _try_openrouter()
        assert client is mock_client
        assert model == _OPENROUTER_MODEL
        assert any("PAID lane engaged" in r.getMessage() for r in caplog.records)
        # Second call logs nothing new.
        with patch("agent.auxiliary_client._select_pool_entry", return_value=(False, None)), \
             patch("hermes_cli.config.load_config_readonly", return_value={"auxiliary": {}}), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai:
            caplog.clear()
            with caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
                _try_openrouter()
        assert not any("PAID lane engaged" in r.getMessage() for r in caplog.records)
        _paid_lane_warned.discard(_OPENROUTER_MODEL)

    def test_is_free_model(self):
        from agent.auxiliary_client import _is_free_model
        assert _is_free_model("nvidia/nemotron-3-ultra-550b-a55b:free")
        assert not _is_free_model("google/gemini-3.6-flash")
        assert not _is_free_model("")
        assert not _is_free_model(None)
