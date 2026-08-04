"""Auxiliary-client fallback chain and transient-failure retry behaviour."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    _NOUS_MODEL,
    _get_provider_chain,
    _refresh_nous_recommended_model,
    _try_payment_fallback,
)

from tests.agent.auxiliary_client.conftest import (
    _AuxAuth401,
    _DummyResponse,
)


class TestRefreshNousRecommendedModel:
    """_refresh_nous_recommended_model picks a fresh model after a stale 404."""

    def test_returns_fresh_portal_recommendation(self, monkeypatch):
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: "stepfun/step-3.7-flash:free",
        )
        out = _refresh_nous_recommended_model(
            vision=True, stale_model="openai/gpt-5.4-mini")
        assert out == "stepfun/step-3.7-flash:free"

    def test_falls_back_to_default_when_portal_matches_stale(self, monkeypatch):
        """If the Portal still recommends the model that just 404'd, fall back
        to the known-good default."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: "openai/gpt-5.4-mini",
        )
        out = _refresh_nous_recommended_model(
            vision=True, stale_model="openai/gpt-5.4-mini")
        assert out == _NOUS_MODEL

    def test_falls_back_to_default_when_portal_unavailable(self, monkeypatch):
        def _boom(**kw):
            raise RuntimeError("portal down")
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model", _boom)
        out = _refresh_nous_recommended_model(
            vision=False, stale_model="some/dead-model")
        assert out == _NOUS_MODEL

    def test_returns_none_when_no_distinct_alternative(self, monkeypatch):
        """When the failed model IS the default and the Portal has nothing
        else, there's no usable alternative."""
        monkeypatch.setattr(
            "hermes_cli.models.get_nous_recommended_aux_model",
            lambda **kw: _NOUS_MODEL,
        )
        out = _refresh_nous_recommended_model(
            vision=False, stale_model=_NOUS_MODEL)
        assert out is None


class TestGetProviderChain:
    """_get_provider_chain() resolves functions at call time (testable)."""

    def test_returns_four_entries(self):
        chain = _get_provider_chain()
        assert len(chain) == 4
        labels = [label for label, _ in chain]
        assert labels == ["openrouter", "nous", "local/custom", "api-key"]
        # Codex is deliberately NOT in this chain — see _get_provider_chain
        # docstring. ChatGPT-account Codex has a shifting model allow-list;
        # guessing a model to fall back on breaks more often than it helps.
        assert "openai-codex" not in labels

    def test_picks_up_patched_functions(self):
        """Patches on _try_* functions must be visible in the chain."""
        sentinel = lambda: ("patched", "model")
        with patch("agent.auxiliary_client._try_openrouter", sentinel):
            chain = _get_provider_chain()
        assert chain[0] == ("openrouter", sentinel)


class TestTryPaymentFallback:
    """_try_payment_fallback skips the failed provider and tries alternatives."""

    @pytest.fixture(autouse=True)
    def _clear_unhealthy_cache(self):
        """Earlier tests in this file call _mark_provider_unhealthy() which
        pollutes the module-level ``_aux_unhealthy_until`` dict (10-min TTL).
        Without this cleanup the fallback chain skips providers we've patched
        to return valid clients — the patched function is never called.
        """
        from agent.auxiliary_client import _aux_unhealthy_until, _aux_unhealthy_logged_at
        _aux_unhealthy_until.clear()
        _aux_unhealthy_logged_at.clear()
        yield
        _aux_unhealthy_until.clear()
        _aux_unhealthy_logged_at.clear()

    def test_skips_failed_provider(self):
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(mock_client, "nous-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter", task="compression")
        assert client is mock_client
        assert model == "nous-model"
        assert label == "nous"

    def test_returns_none_when_no_fallback(self):
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert label == ""

    def test_codex_alias_maps_to_chain_label(self):
        """'codex' should map to 'openai-codex' in the skip set."""
        mock_client = MagicMock()
        with patch("agent.auxiliary_client._try_openrouter", return_value=(mock_client, "or-model")), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openai-codex"):
            client, model, label = _try_payment_fallback("openai-codex", task="vision")
        assert client is mock_client
        assert label == "openrouter"

    def test_codex_not_in_fallback_chain(self):
        """Codex is deliberately NOT a fallback rung (shifting model allow-list).

        When OR/Nous/custom/api-key all fail, payment-fallback returns None —
        Codex is never tried with a guessed model.
        """
        with patch("agent.auxiliary_client._try_openrouter", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_nous", return_value=(None, None)), \
             patch("agent.auxiliary_client._try_custom_endpoint", return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_api_key_provider", return_value=(None, None)), \
             patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"):
            client, model, label = _try_payment_fallback("openrouter")
        assert client is None
        assert model is None
        assert label == ""

    def test_policy_off_never_walks_builtin_remote_chain(self):
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "off"},
        ), patch("agent.auxiliary_client._get_provider_chain") as chain:
            client, model, label = _try_payment_fallback("custom", task="vision")

        chain.assert_not_called()
        assert (client, model, label) == (None, None, "")

    def test_local_only_skips_remote_and_returns_local_endpoint(self):
        remote = MagicMock(base_url="https://api.anthropic.com")
        local = MagicMock(base_url="http://10.55.0.3:8000/v1")
        remote_resolver = MagicMock(return_value=(remote, "claude"))
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "local-only"},
        ), patch(
            "agent.auxiliary_client._get_provider_chain",
            return_value=[
                ("anthropic", remote_resolver),
                ("local/custom", lambda: (local, "qwen")),
            ],
        ), patch("agent.auxiliary_client._read_main_provider", return_value="custom"):
            client, model, label = _try_payment_fallback("failed", task="compression")

        assert client is local
        assert model == "qwen"
        assert label == "local/custom"
        remote_resolver.assert_not_called()


class TestCallLlmPaymentFallback:
    """call_llm() retries with a different provider on 402 / payment / rate-limit errors."""

    def _make_402_error(self, msg="Payment Required: insufficient credits"):
        exc = Exception(msg)
        exc.status_code = 402
        return exc

    def _make_429_rate_limit_error(self, msg="Rate limit exceeded, try again in 60 seconds"):
        exc = Exception(msg)
        exc.status_code = 429
        return exc

    def test_non_payment_error_not_caught(self, monkeypatch):
        """Non-payment/non-connection errors (500) should NOT trigger fallback."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        server_err = Exception("Internal Server Error")
        server_err.status_code = 500
        primary_client.chat.completions.create.side_effect = server_err

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, _NOUS_MODEL)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", _NOUS_MODEL, None, None, None)):
            with pytest.raises(Exception, match="Internal Server Error"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

    def test_429_rate_limit_triggers_fallback(self, monkeypatch):
        """429 rate-limit errors should trigger fallback to next provider."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        rate_err = self._make_429_rate_limit_error()
        primary_client.chat.completions.create.side_effect = rate_err

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="fallback response"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                    return_value=(primary_client, "xiaomi/mimo-v2-pro")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                    return_value=("auto", "xiaomi/mimo-v2-pro", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                    return_value=(fallback_client, "fallback-model", "openrouter")):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
            )
        # Fallback client should have been used
        assert fallback_client.chat.completions.create.called

    def test_401_auth_error_triggers_fallback_in_auto_mode(self, monkeypatch):
        """401 auth errors should trigger fallback in auto mode (#21165).

        When refresh is unavailable/fails and the user is on the auto chain,
        a 401 must fall back instead of silently dropping the aux task
        (which caused compression message loss).
        """
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.base_url = "https://api.minimax.chat/v1"
        primary_client.chat.completions.create.side_effect = _AuxAuth401("expired key")

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = _DummyResponse("fallback auth response")

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimax/minimax-m2.7")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", "minimax/minimax-m2.7", None, None, None)), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(fallback_client, "fallback-model", "openrouter")) as mock_fb:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "fallback auth response"
        assert fallback_client.chat.completions.create.called
        # Labelled as an auth error, not mis-tagged as a connection error.
        assert mock_fb.call_args.kwargs.get("reason") == "auth error"

    def test_401_auth_error_no_fallback_with_explicit_provider(self, monkeypatch):
        """401 on an explicitly-configured provider must NOT silently switch.

        Auth is not a capacity error: the explicit-provider gate means a 401
        respects the user's choice and raises instead of falling back. This
        guards the deliberate design at the should_fallback/is_capacity gate.
        """
        primary_client = MagicMock()
        primary_client.base_url = "https://api.minimax.chat/v1"
        primary_client.chat.completions.create.side_effect = _AuxAuth401("expired key")

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimax/minimax-m2.7")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("minimax", "minimax/minimax-m2.7", None, None, None)), \
             patch("agent.auxiliary_client._refresh_provider_credentials", return_value=False), \
             patch("agent.auxiliary_client._try_payment_fallback") as mock_fb:
            with pytest.raises(_AuxAuth401):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )
        mock_fb.assert_not_called()


class TestStaleFallbackCandidateSkip:
    """A fallback candidate with a stale credential must not abort the task.

    Live case (mattalachia debug dump, Jul 2026): Codex compression timed out,
    the aux chain fell back to Anthropic using an expired ANTHROPIC_TOKEN, and
    the resulting 401 aborted compression with a 60s cooldown — five times in
    one session — even though refreshing or skipping the candidate would have
    let compression proceed.
    """

    def _timeout_err(self):
        # Class name carries "Timeout" — matches _is_connection_error's
        # type-name detection, like the real Codex stream-deadline error.
        class _AuxStreamTimeoutError(Exception):
            pass
        return _AuxStreamTimeoutError(
            "Codex auxiliary Responses stream exceeded 120.0s total timeout")

    def test_stale_anthropic_fallback_refreshes_and_retries(self, monkeypatch):
        """401 from the fallback candidate → refresh its creds → retry succeeds."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        stale_fb = MagicMock()
        stale_fb.base_url = "https://api.anthropic.com"
        stale_fb.chat.completions.create.side_effect = _AuxAuth401("Invalid bearer token")

        fresh_fb = MagicMock()
        fresh_fb.base_url = "https://api.anthropic.com"
        fresh_fb.chat.completions.create.return_value = _DummyResponse("fresh-fallback")

        def _cached_client(provider, model=None, **kw):
            if provider == "anthropic":
                return (fresh_fb, "claude-haiku-4-5-20251001")
            return (primary_client, "gpt-5.5")

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client", side_effect=_cached_client), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(stale_fb, "claude-haiku-4-5-20251001", "anthropic")), \
             patch("agent.auxiliary_client._refresh_provider_credentials",
                   return_value=True) as mock_refresh:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "fresh-fallback"
        mock_refresh.assert_called_once_with("anthropic")
        assert stale_fb.chat.completions.create.call_count == 1
        assert fresh_fb.chat.completions.create.call_count == 1

    def test_unrefreshable_stale_candidate_is_skipped_to_next(self, monkeypatch):
        """Refresh fails (expired setup token) → candidate quarantined, chain
        walked again, next candidate serves the request."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        stale_fb = MagicMock()
        stale_fb.base_url = "https://api.anthropic.com"
        stale_fb.chat.completions.create.side_effect = _AuxAuth401("Invalid bearer token")

        healthy_fb = MagicMock()
        healthy_fb.base_url = "https://openrouter.ai/api/v1"
        healthy_fb.chat.completions.create.return_value = _DummyResponse("openrouter-serves")

        fb_walks = [
            (stale_fb, "claude-haiku-4-5-20251001", "anthropic"),
            (healthy_fb, "fallback-model", "openrouter"),
        ]

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   side_effect=fb_walks) as mock_fb, \
             patch("agent.auxiliary_client._refresh_provider_credentials",
                   return_value=False), \
             patch("agent.auxiliary_client._mark_provider_unhealthy") as mock_mark:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "summarize"}],
            )

        assert result.choices[0].message.content == "openrouter-serves"
        assert mock_fb.call_count == 2
        assert mock_fb.call_args_list[1].kwargs.get("reason") == "stale fallback credential"
        mock_mark.assert_called_once_with("anthropic")
        assert stale_fb.chat.completions.create.call_count == 1
        assert healthy_fb.chat.completions.create.call_count == 1

    def test_non_auth_fallback_error_still_raises(self, monkeypatch):
        """A non-auth error from the fallback candidate propagates unchanged."""
        primary_client = MagicMock()
        primary_client.base_url = "https://chatgpt.com/backend-api/codex"
        primary_client.chat.completions.create.side_effect = self._timeout_err()

        broken_fb = MagicMock()
        broken_fb.base_url = "https://api.anthropic.com"
        broken_fb.chat.completions.create.side_effect = ValueError("malformed response")

        with patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_payment_fallback",
                   return_value=(broken_fb, "claude-haiku-4-5-20251001", "anthropic")):
            with pytest.raises(ValueError, match="malformed response"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "summarize"}],
                )


class TestAuxiliaryFallbackLayering:
    """Explicit-provider users get layered fallback: configured_chain → main agent → warn."""

    def _make_payment_err(self):
        exc = Exception("Payment Required: insufficient credits")
        exc.status_code = 402
        return exc

    def test_empty_choices_with_output_text_is_recovered_before_fallback(self, monkeypatch):
        """Responses-style output_text should be used before provider fallback."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            output_text="recovered title",
            model="minimaxai/minimax-m3",
        )

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain") as mock_chain:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "recovered title"
        mock_chain.assert_not_called()

    def test_empty_choices_with_output_items_is_recovered_before_fallback(self, monkeypatch):
        """Responses-style output message items should be normalized for aux callers."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = SimpleNamespace(
            choices=[],
            output=[
                SimpleNamespace(
                    type="message",
                    content=[
                        SimpleNamespace(type="output_text", text="part one"),
                        {"type": "text", "text": "part two"},
                    ],
                )
            ],
            model="minimaxai/minimax-m3",
        )

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain") as mock_chain:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "part one\npart two"
        mock_chain.assert_not_called()

    def test_invalid_empty_choices_response_triggers_fallback(self, monkeypatch):
        """HTTP-200 malformed chat completions should not abort aux fallback."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.return_value = MagicMock(choices=[])

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain, \
             patch("agent.auxiliary_client._try_main_agent_model_fallback") as mock_main:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "from fallback chain"
        mock_chain.assert_called_once_with(
            "title_generation",
            "nvidia",
            reason="invalid provider response",
            # A malformed response is model-specific, so the chain skip is
            # narrowed to the exact model that failed rather than the whole
            # provider (see _try_configured_fallback_chain's failed_model).
            failed_model="minimaxai/minimax-m3",
        )
        mock_main.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_invalid_empty_choices_response_triggers_fallback(self, monkeypatch):
        """Async aux calls use the same malformed-response fallback path."""
        primary_client = MagicMock()
        primary_client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[]))

        fallback_client = MagicMock()
        async_fallback_client = MagicMock()
        async_fallback_client.chat.completions.create = AsyncMock(return_value=MagicMock(choices=[
            MagicMock(message=MagicMock(content="from async fallback"))
        ]))

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "minimaxai/minimax-m3")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("nvidia", "minimaxai/minimax-m3", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain, \
             patch("agent.auxiliary_client._to_async_client",
                   return_value=(async_fallback_client, "gpt-5.4-mini")):
            result = await async_call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert result.choices[0].message.content == "from async fallback"
        mock_chain.assert_called_once_with(
            "compression",
            "nvidia",
            reason="invalid provider response",
            failed_model="minimaxai/minimax-m3",
        )

    def test_auto_provider_uses_task_then_main_chain_before_builtin_chain(self, monkeypatch):
        """Auto aux call failures try per-task then top-level fallback before built-ins."""
        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        main_chain_client = MagicMock()
        main_chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from main fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "qwen/qwen3.5-122b-a10b")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("auto", None, None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")) as mock_task_chain, \
             patch("agent.auxiliary_client._try_main_fallback_chain",
                   return_value=(main_chain_client, "inclusionai/ring-2.6-1t:free", "openrouter")) as mock_main_chain, \
             patch("agent.auxiliary_client._try_payment_fallback") as mock_builtin_chain:
            result = call_llm(
                task="title_generation",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert main_chain_client.chat.completions.create.called
        mock_task_chain.assert_called_once_with(
            "title_generation", "auto", reason="payment error",
            # 402 is provider-wide, so the whole provider stays skipped:
            # failed_model is deliberately None here.
            failed_model=None)
        mock_main_chain.assert_called_once_with(
            "title_generation", "auto", reason="payment error")
        mock_builtin_chain.assert_not_called()

    def test_explicit_provider_uses_configured_chain_first(self, monkeypatch, caplog):
        """When a user has fallback_chain configured, it's tried BEFORE the main agent model."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        chain_client = MagicMock()
        chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from configured chain"))
        ])

        main_called = MagicMock()

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(chain_client, "gpt-4o-mini", "fallback_chain[0](openai)")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   side_effect=main_called):
            result = call_llm(
                task="vision",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert chain_client.chat.completions.create.called
        # Main agent fallback should NOT have been consulted — chain succeeded first
        main_called.assert_not_called()

    def test_explicit_provider_falls_back_to_main_when_chain_exhausted(self, monkeypatch):
        """If configured fallback_chain returns nothing, main agent model is tried next."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        main_client = MagicMock()
        main_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from main agent"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   return_value=(main_client, "claude-sonnet-4", "main-agent(openrouter)")):
            result = call_llm(
                task="vision",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert main_client.chat.completions.create.called

    def test_explicit_provider_rate_limit_triggers_fallback(self, monkeypatch):
        """429 rate-limit on an explicit provider must trigger fallback (not be ignored).

        Regression test for #52228: rate limits were excluded from
        ``is_capacity_error``, so explicit-provider auxiliary calls never
        fell back on 429 — only auto-provider calls did.
        """
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        rate_err = Exception("Rate limit exceeded, try again in 60 seconds")
        rate_err.status_code = 429
        primary_client.chat.completions.create.side_effect = rate_err

        fallback_client = MagicMock()
        fallback_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from fallback chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "gpt-5.5")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("openai-codex", "gpt-5.5", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(fallback_client, "deepseek-v4-pro", "fallback_chain[0](opencode-go)")) as mock_chain, \
             patch("agent.auxiliary_client._try_main_agent_model_fallback") as mock_main:
            result = call_llm(
                task="kanban_decomposer",
                messages=[{"role": "user", "content": "decompose this"}],
            )

        # Fallback chain MUST be tried for rate-limit on explicit provider
        mock_chain.assert_called()
        assert fallback_client.chat.completions.create.called
        # Main agent fallback should NOT be needed when chain succeeds
        mock_main.assert_not_called()


    def test_warning_emitted_when_all_fallbacks_exhausted(self, monkeypatch, caplog):
        """When chain AND main model both fail, a user-visible warning fires before re-raise."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")

        primary_client = MagicMock()
        primary_client.chat.completions.create.side_effect = self._make_payment_err()

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(primary_client, "glm-4v-flash")), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("glm", "glm-4v-flash", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")), \
             patch("agent.auxiliary_client._try_main_agent_model_fallback",
                   return_value=(None, None, "")), \
             caplog.at_level("WARNING", logger="agent.auxiliary_client"):
            with pytest.raises(Exception, match="Payment Required"):
                call_llm(
                    task="vision",
                    messages=[{"role": "user", "content": "hello"}],
                )

        assert any(
            "all fallbacks exhausted" in r.message for r in caplog.records
        ), f"Expected exhaustion warning, got: {[r.message for r in caplog.records]}"

    def test_explicit_provider_no_client_uses_configured_chain_before_error(self, monkeypatch):
        """Missing primary credentials should still honor auxiliary fallback_chain."""
        chain_client = MagicMock()
        chain_client.chat.completions.create.return_value = MagicMock(choices=[
            MagicMock(message=MagicMock(content="from configured chain"))
        ])

        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(chain_client, "gpt-5.4-mini", "fallback_chain[0](openai-codex)")) as mock_chain:
            result = call_llm(
                task="compression",
                messages=[{"role": "user", "content": "hello"}],
            )

        assert chain_client.chat.completions.create.called
        assert result.choices[0].message.content == "from configured chain"
        mock_chain.assert_called_once_with(
            "compression",
            "ollama-cloud",
            reason="provider unavailable",
        )

    def test_explicit_provider_no_client_without_chain_keeps_clear_error(self, monkeypatch):
        """No fallback configured: keep the existing actionable missing-key error."""
        with patch("agent.auxiliary_client._get_cached_client",
                   return_value=(None, None)), \
             patch("agent.auxiliary_client._resolve_task_provider_model",
                   return_value=("ollama-cloud", "deepseek-v4-flash:cloud", None, None, None)), \
             patch("agent.auxiliary_client._try_configured_fallback_chain",
                   return_value=(None, None, "")) as mock_chain:
            with pytest.raises(RuntimeError, match="Provider 'ollama-cloud'.*no API key"):
                call_llm(
                    task="compression",
                    messages=[{"role": "user", "content": "hello"}],
                )

        mock_chain.assert_called_once_with(
            "compression",
            "ollama-cloud",
            reason="provider unavailable",
        )

    def test_fallback_entry_openai_codex_uses_oauth_pool_without_inline_key(self):
        """Configured Codex fallback resolves through Hermes auth / credential pool."""
        from agent.auxiliary_client import _resolve_fallback_entry

        pool_entry = MagicMock()
        pool_entry.id = "codex-pool-1"
        pool_entry.runtime_api_key = "codex-oauth-token"
        pool_entry.access_token = "codex-oauth-token"
        pool_entry.runtime_base_url = "https://chatgpt.com/backend-api/codex"

        real_client = MagicMock()
        real_client.api_key = "codex-oauth-token"
        real_client.base_url = "https://chatgpt.com/backend-api/codex"

        with patch("agent.auxiliary_client._select_pool_entry",
                   return_value=(True, pool_entry)), \
             patch("agent.auxiliary_client._read_codex_access_token",
                   side_effect=AssertionError("should use pool token")), \
             patch("agent.auxiliary_client.OpenAI", return_value=real_client) as mock_openai:
            client, model = _resolve_fallback_entry({
                "provider": "openai-codex",
                "model": "gpt-5.4-mini",
            })

        assert client is not None
        assert model == "gpt-5.4-mini"
        mock_openai.assert_called_once()
        assert mock_openai.call_args.kwargs["api_key"] == "codex-oauth-token"


class TestTryMainAgentModelFallback:
    """_try_main_agent_model_fallback resolves the user's main provider+model as a safety net."""

    def test_returns_none_when_main_provider_is_auto(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="auto"), \
             patch("agent.auxiliary_client._read_main_model", return_value="some-model"):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is None and model is None and label == ""

    def test_returns_none_when_failed_provider_equals_main(self):
        """If the thing that failed IS the main model, no point retrying it."""
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"):
            client, model, label = _try_main_agent_model_fallback("openrouter", task="vision")
        assert client is None and label == ""

    def test_resolves_main_provider_client(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        fake_client = MagicMock()
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=False), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(fake_client, "anthropic/claude-sonnet-4")):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is fake_client
        assert model == "anthropic/claude-sonnet-4"
        assert label == "main-agent(openrouter)"

    def test_skips_when_main_provider_is_unhealthy(self):
        from agent.auxiliary_client import _try_main_agent_model_fallback
        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="anthropic/claude-sonnet-4"), \
             patch("agent.auxiliary_client._is_provider_unhealthy", return_value=True):
            client, model, label = _try_main_agent_model_fallback("glm", task="vision")
        assert client is None


class TestTransientTransportRetry:
    """call_llm retries ONCE on the same provider for a transient transport
    blip before escalating to the fallback chain.

    Salvaged from PR #16587 (@ARegalado1). The original fixed only the
    context-compression caller; this lives in call_llm so every auxiliary
    task (compression, memory flush, title-gen, session-search, vision)
    gets the same same-target retry, and the gate reuses the canonical
    _is_connection_error detector.
    """

    def _patches(self, client):
        return (
            patch(
                "agent.auxiliary_client._resolve_task_provider_model",
                return_value=("openrouter", "some-model", None, None, None),
            ),
            patch(
                "agent.auxiliary_client._get_cached_client",
                return_value=(client, "some-model"),
            ),
            patch(
                "agent.auxiliary_client._validate_llm_response",
                side_effect=lambda resp, _task, **_kw: resp,
            ),
        )

    def test_retries_streaming_close_once_same_provider(self):
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            Exception(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            ),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        # Same client called twice — no provider fallback needed.
        assert client.chat.completions.create.call_count == 2

    def test_retries_5xx_once_same_provider(self):
        class _Err503(Exception):
            status_code = 503

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [_Err503("upstream"), {"ok": True}]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_does_not_retry_non_transient_400(self):
        class _Err400(Exception):
            status_code = 400

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = _Err400("bad request")
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3, pytest.raises(_Err400):
            call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        # Non-transient: single attempt, no same-target retry.
        assert client.chat.completions.create.call_count == 1

    def test_second_transient_failure_escalates_to_fallback(self):
        """Two transient failures in a row exhaust the same-target retry and
        fall through to the existing connection-error provider fallback."""
        primary = MagicMock()
        primary.base_url = "https://openrouter.ai/api/v1"
        primary.chat.completions.create.side_effect = Exception(
            "peer closed connection without sending complete message body"
        )

        fb_client = MagicMock()
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch("agent.auxiliary_client._transient_retry_count", return_value=1),
            patch("agent.auxiliary_client._TRANSIENT_RETRY_BACKOFF_BASE", 0.0),
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(None, None, ""),
            ),
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(fb_client, "fb-model", "openai"),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        # Primary tried twice (initial + one same-target retry), then fallback.
        assert primary.chat.completions.create.call_count == 2
        assert fb_client.chat.completions.create.call_count == 1

    def test_compression_skips_same_provider_retry_on_timeout(self):
        """A timeout on the critical compression path must NOT retry the same
        provider (that doubles the user-visible stall, issue #54465) — it
        falls straight through to the fallback chain instead.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        primary = MagicMock()
        primary.base_url = "https://openrouter.ai/api/v1"
        primary.chat.completions.create.side_effect = _Timeout("Request timed out.")

        fb_client = MagicMock()
        fb_client.base_url = "https://api.openai.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(None, None, ""),
            ),
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(fb_client, "fb-model", "openai"),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        # Primary tried ONCE only — no same-provider timeout retry — then fallback.
        assert primary.chat.completions.create.call_count == 1
        assert fb_client.chat.completions.create.call_count == 1

    def test_non_compression_still_retries_same_provider_on_timeout(self):
        """The timeout skip is scoped to compression only; other auxiliary
        tasks keep the single same-provider transient retry.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            _Timeout("Request timed out."),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="title_generation", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_compression_still_retries_streaming_close_on_timeout_path(self):
        """A fast streaming-close (not a full-budget timeout) still retries
        same-provider even for compression — only timeouts are skipped.
        """
        client = MagicMock()
        client.base_url = "https://openrouter.ai/api/v1"
        client.chat.completions.create.side_effect = [
            Exception(
                "peer closed connection without sending complete message body "
                "(incomplete chunked read)"
            ),
            {"ok": True},
        ]
        p1, p2, p3 = self._patches(client)
        with p1, p2, p3:
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"ok": True}
        assert client.chat.completions.create.call_count == 2

    def test_timeout_forwards_failed_model_to_configured_chain(self):
        """A timeout is model-specific, so call_llm must forward the failed
        model to the configured chain (failed_model=<model>, not None). This
        lets a same-provider sibling in the chain be tried instead of the
        whole provider being skipped — the exact NVIDIA NIM bug's trigger.
        """
        class _Timeout(Exception):
            pass
        _Timeout.__name__ = "APITimeoutError"

        primary = MagicMock()
        primary.base_url = "https://integrate.api.nvidia.com/v1"
        primary.chat.completions.create.side_effect = _Timeout("Request timed out.")

        fb_client = MagicMock()
        fb_client.base_url = "https://integrate.api.nvidia.com/v1"
        fb_client.chat.completions.create.return_value = {"fallback": True}

        p1, p2, p3 = self._patches(primary)
        with (
            p1, p2, p3,
            patch(
                "agent.auxiliary_client._try_configured_fallback_chain",
                return_value=(fb_client, "sibling-model", "fallback_chain[0](openrouter)"),
            ) as mock_chain,
            patch(
                "agent.auxiliary_client._try_main_agent_model_fallback",
                return_value=(None, None, ""),
            ),
        ):
            result = call_llm(task="compression", messages=[{"role": "user", "content": "hi"}])
        assert result == {"fallback": True}
        _, kwargs = mock_chain.call_args
        assert kwargs.get("failed_model") == "some-model", (
            "A timeout is model-specific — the failed model must be forwarded "
            "so a same-provider sibling can be tried, not skipped wholesale."
        )


class TestAuxClientNoSdkRetries:
    """Auxiliary OpenAI clients are constructed with SDK-internal retries
    disabled so Hermes owns the retry/timeout budget (issue #54465). The SDK
    default (max_retries=2 → 3 attempts) silently triples the effective wall
    time of every aux call against a slow/hung endpoint.
    """

    def test_sync_client_disables_sdk_retries(self):
        from agent import auxiliary_client as ac
        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(ac, "OpenAI", _FakeOpenAI), \
             patch.object(ac, "_openai_http_client_kwargs", return_value={}):
            ac._create_openai_client(api_key="k", base_url="https://x/v1")
        assert captured.get("max_retries") == 0

    def test_explicit_max_retries_override_wins(self):
        from agent import auxiliary_client as ac
        captured = {}

        class _FakeOpenAI:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        with patch.object(ac, "OpenAI", _FakeOpenAI), \
             patch.object(ac, "_openai_http_client_kwargs", return_value={}):
            ac._create_openai_client(api_key="k", base_url="https://x/v1", max_retries=5)
        assert captured.get("max_retries") == 5


class TestCompressionFallbackContextFilter:
    """Aux fallback chains must skip candidates whose context window is
    smaller than the task minimum, then continue to the next candidate.

    Layer coverage:
      L2: _try_configured_fallback_chain skips too-small candidates
      L3: _try_main_fallback_chain skips too-small candidates
      L4: candidates with unknown context (None) are passed through
      L5: backward compat — first viable candidate still wins
    """

    @staticmethod
    def _make_chain_entry(provider, model, base_url="https://example.com/v1",
                          api_key="k"):
        return {
            "provider": provider,
            "model": model,
            "base_url": base_url,
            "api_key": api_key,
        }

    def _mock_resolve(self, entry):
        """Mock _resolve_fallback_entry to return a (client, model) per entry."""
        client = MagicMock()
        client.base_url = entry.get("base_url", "")
        return client, entry["model"]

    # ── L2: configured fallback chain ─────────────────────────────────

    def test_configured_chain_skips_too_small_candidate_for_compression(self, monkeypatch):
        """When entry[0] is reachable but too small and entry[1] is large enough,
        _try_configured_fallback_chain must return entry[1], not entry[0]."""
        from agent.auxiliary_client import (
            _try_configured_fallback_chain,
        )

        small_client = MagicMock(name="small_client")
        large_client = MagicMock(name="large_client")
        entries = [
            self._make_chain_entry("small-provider", "tiny-8k"),
            self._make_chain_entry("big-provider", "huge-1m"),
        ]

        def fake_resolve(entry):
            if entry is entries[0]:
                return small_client, "tiny-8k"
            return large_client, "huge-1m"

        # tiny-8k resolves to 8K (below 64K floor); huge-1m resolves to 1M
        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"tiny-8k": 8192, "huge-1m": 1_048_576}.get(model, 256_000)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client, (
            f"Expected large_client (1M context), got {client}. "
            "L2 bug: chain returned the first reachable candidate without "
            "screening by context window.")
        assert model == "huge-1m"
        assert "big-provider" in label

    def test_configured_chain_continues_after_skipping_too_small(self, monkeypatch):
        """When all small candidates are skipped and only the last is large enough,
        the chain still returns it (does not stop after first filter)."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        small_client_a = MagicMock(name="small_a")
        small_client_b = MagicMock(name="small_b")
        large_client = MagicMock(name="large")
        entries = [
            self._make_chain_entry("p1", "small-a-32k"),
            self._make_chain_entry("p2", "small-b-48k"),
            self._make_chain_entry("p3", "large-512k"),
        ]

        def fake_resolve(entry):
            if entry is entries[0]:
                return small_client_a, "small-a-32k"
            if entry is entries[1]:
                return small_client_b, "small-b-48k"
            return large_client, "large-512k"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"small-a-32k": 32_000,
                    "small-b-48k": 48_000,
                    "large-512k": 512_000}.get(model, 256_000)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client
        assert model == "large-512k"

    # ── L3: main fallback chain ────────────────────────────────────────

    def test_main_chain_skips_too_small_candidate_for_compression(self, monkeypatch):
        """Same behaviour for the top-level main-agent fallback chain."""
        from agent.auxiliary_client import (
            _try_main_fallback_chain,
        )

        small_client = MagicMock(name="small_main")
        large_client = MagicMock(name="large_main")

        # Mock load_config + get_fallback_chain to return our controlled chain
        chain = [
            self._make_chain_entry("p-small", "tiny-16k"),
            self._make_chain_entry("p-large", "huge-1m"),
        ]

        def fake_resolve(entry):
            if entry is chain[0]:
                return small_client, "tiny-16k"
            return large_client, "huge-1m"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return {"tiny-16k": 16_384, "huge-1m": 1_048_576}.get(model, 256_000)

        monkeypatch.setattr(
            "hermes_cli.fallback_config.get_fallback_chain",
            lambda cfg: chain,
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx), \
             patch("agent.auxiliary_client._is_provider_unhealthy",
                   return_value=False):
            client, model, label = _try_main_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is large_client, (
            f"Expected large_client (1M), got {client}. "
            "L3 bug: main chain returned the first reachable candidate "
            "without screening by context window.")
        assert model == "huge-1m"

    # ── L4: unknown context passthrough ────────────────────────────────

    def test_configured_chain_passes_through_unknown_context(self, monkeypatch):
        """When get_model_context_length returns None (cannot probe),
        the candidate is NOT filtered — the existing behaviour of using
        the default 256K fallback in the resolver chain is preserved."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        unknown_client = MagicMock(name="unknown_client")
        entries = [self._make_chain_entry("unknown-provider", "unprobed-model")]

        def fake_resolve(entry):
            return unknown_client, "unprobed-model"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return None  # cannot determine context length

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="compression", failed_provider="auto")

        assert client is unknown_client, (
            "L4 bug: candidates with unknown context must be passed through, "
            "not blocked. Being unsure is not the same as being too small.")
        assert model == "unprobed-model"

    # ── L5: backward compat — non-compression tasks unchanged ──────────

    def test_non_compression_task_does_not_filter_by_context(self, monkeypatch):
        """For tasks without a context floor (e.g. title_generation, vision),
        the chain behaviour is unchanged: first reachable candidate wins."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        small_client = MagicMock(name="small")
        entries = [self._make_chain_entry("p", "tiny-4k")]

        def fake_resolve(entry):
            return small_client, "tiny-4k"

        def fake_ctx(model, base_url="", api_key="", **kwargs):
            return 4_096  # small — but title_generation has no floor

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "title_generation" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=fake_resolve), \
             patch("agent.auxiliary_client.get_model_context_length",
                   side_effect=fake_ctx):
            client, model, label = _try_configured_fallback_chain(
                task="title_generation", failed_provider="auto")

        assert client is small_client, (
            "L5 regression: non-compression tasks must not be filtered "
            "by context window. The first reachable candidate should win.")
        assert model == "tiny-4k"

    # ── End-to-end: configured chain skips too-small for vision too ──
    # vision has its own implicit context requirements; test that the
    # compression-specific filter does NOT affect vision chains.

    def test_compression_task_uses_minimum_context_constant(self):
        """The task minimum for compression must equal MINIMUM_CONTEXT_LENGTH
        so the runtime fallback stays consistent with the startup feasibility
        check in agent/conversation_compression.py."""
        from agent.auxiliary_client import _task_minimum_context_length
        from agent.model_metadata import MINIMUM_CONTEXT_LENGTH

        assert _task_minimum_context_length("compression") == MINIMUM_CONTEXT_LENGTH
        # Non-compression tasks have no minimum (None)
        assert _task_minimum_context_length("vision") is None
        assert _task_minimum_context_length("title_generation") is None
        assert _task_minimum_context_length("web_extract") is None
        assert _task_minimum_context_length("skills_hub") is None
        assert _task_minimum_context_length("mcp") is None
        assert _task_minimum_context_length("session_search") is None
        # Empty / unknown tasks have no minimum
        assert _task_minimum_context_length("") is None
        assert _task_minimum_context_length(None) is None

    def test_same_provider_same_model_still_skipped(self, monkeypatch):
        """The exact (provider, model) pair that just failed is still
        skipped — failed_model narrows the skip, it doesn't disable it."""
        from agent.auxiliary_client import _try_configured_fallback_chain

        entries = [
            self._make_chain_entry("nvidia", "deepseek-ai/deepseek-v4-pro"),
        ]

        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": entries} if task == "compression" else {},
        )

        with patch("agent.auxiliary_client._resolve_fallback_entry",
                   side_effect=AssertionError("must not be resolved")):
            client, model, label = _try_configured_fallback_chain(
                task="compression",
                failed_provider="nvidia",
                failed_model="deepseek-ai/deepseek-v4-pro",
            )

        assert client is None
        assert model is None
        assert label == ""


class TestSynchronousFallbackCachePlans:
    @staticmethod
    def _run_configured_fallback(monkeypatch, entry):
        from agent.auxiliary_client import (
            _call_fallback_candidate_sync,
            _try_configured_fallback_chain,
        )

        client = MagicMock()
        client.base_url = entry["base_url"]
        client.chat.completions.create.return_value = _DummyResponse()
        resolved_calls = []

        def resolve(provider, model=None, **kwargs):
            resolved_calls.append((provider, model, kwargs))
            return client, model

        monkeypatch.setattr("agent.auxiliary_client.resolve_provider_client", resolve)
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": [entry]},
        )
        fallback_client, fallback_model, label = _try_configured_fallback_chain(
            task="moa_aggregator",
            failed_provider="primary",
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        _call_fallback_candidate_sync(
            fallback_client,
            fallback_model,
            label,
            task="moa_aggregator",
            messages=[
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": "lookup"},
            ],
            temperature=None,
            max_tokens=None,
            tools=tools,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )
        return client, resolved_calls, tools

    def test_direct_anthropic_fallback_uses_entry_destination_for_tool_marker(self, monkeypatch):
        client, resolved_calls, tools = self._run_configured_fallback(monkeypatch, {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        })

        assert resolved_calls == [(
            "anthropic",
            "claude-sonnet-4-6",
            {
                "explicit_base_url": "https://api.anthropic.com",
                "explicit_api_key": None,
                "api_mode": "anthropic_messages",
            },
        )]
        wire_tools = client.chat.completions.create.call_args.kwargs["tools"]
        assert "cache_control" in wire_tools[-1]
        assert "cache_control" not in tools[-1]

    def test_third_party_anthropic_fallback_keeps_message_markers_without_tool_marker(self, monkeypatch):
        client, resolved_calls, tools = self._run_configured_fallback(monkeypatch, {
            "provider": "custom",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.minimax.io/anthropic",
            "api_mode": "anthropic_messages",
        })

        assert resolved_calls[0][2]["explicit_base_url"] == "https://api.minimax.io/anthropic"
        assert resolved_calls[0][2]["api_mode"] == "anthropic_messages"
        wire_request = client.chat.completions.create.call_args.kwargs
        assert "cache_control" not in wire_request["tools"][-1]
        assert "cache_control" not in tools[-1]
        assert any(
            isinstance(part, dict) and "cache_control" in part
            for message in wire_request["messages"]
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        )


class TestAsynchronousFallbackCachePlans:
    @pytest.mark.asyncio
    async def test_async_fallback_replans_cache_sections_like_sync(self, monkeypatch):
        """Async mirror parity: per-destination cache replan, not verbatim pass-through."""
        from agent.auxiliary_client import (
            _call_fallback_candidate_async,
            _try_configured_fallback_chain,
        )

        entry = {
            "provider": "anthropic",
            "model": "claude-sonnet-4-6",
            "base_url": "https://api.anthropic.com",
            "api_mode": "anthropic_messages",
        }
        client = MagicMock()
        client.base_url = entry["base_url"]

        async def _create(**kwargs):
            return _DummyResponse()

        client.chat.completions.create = MagicMock(side_effect=_create)
        monkeypatch.setattr(
            "agent.auxiliary_client.resolve_provider_client",
            lambda provider, model=None, **kwargs: (client, model),
        )
        monkeypatch.setattr(
            "agent.auxiliary_client._get_auxiliary_task_config",
            lambda task: {"fallback_chain": [entry]},
        )
        fallback_client, fallback_model, label = _try_configured_fallback_chain(
            task="moa_aggregator",
            failed_provider="primary",
        )
        tools = [{
            "type": "function",
            "function": {
                "name": "lookup",
                "parameters": {"type": "object", "properties": {}},
            },
        }]
        await _call_fallback_candidate_async(
            fallback_client,
            fallback_model,
            label,
            task="moa_aggregator",
            messages=[
                {"role": "system", "content": "stable prefix"},
                {"role": "user", "content": "lookup"},
            ],
            temperature=None,
            max_tokens=None,
            tools=tools,
            effective_timeout=30.0,
            effective_extra_body={},
            reasoning_config=None,
        )

        wire_tools = client.chat.completions.create.call_args.kwargs["tools"]
        assert "cache_control" in wire_tools[-1]
        assert "cache_control" not in tools[-1]
