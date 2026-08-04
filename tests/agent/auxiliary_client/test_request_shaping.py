"""Auxiliary-client outbound request shaping: max-tokens kwarg, tags, extra_body, headers."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import logging
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    resolve_provider_client,
    auxiliary_max_tokens_param,
    call_llm,
    async_call_llm,
    _build_call_kwargs,
    _resolve_auto,
)



class TestAuxiliaryMaxTokensParamGithubCopilot:
    """GitHub Copilot custom-base cases.

    These lived in a second ``TestAuxiliaryMaxTokensParam`` that was
    shadowed by the class below, so pytest never collected them.
    """

    def test_uses_max_completion_tokens_for_github_copilot_custom_base(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}

    def test_uses_max_completion_tokens_for_github_copilot_custom_base_path(self):
        with patch("agent.auxiliary_client._resolve_custom_runtime", return_value=("https://api.githubcopilot.com/chat/completions", "key", None)), \
             patch("agent.auxiliary_client._read_nous_auth", return_value=None):
            assert auxiliary_max_tokens_param(2048) == {"max_completion_tokens": 2048}


class TestAuxiliaryMaxTokensParam:
    """Verify the kwarg emitted by ``auxiliary_max_tokens_param`` across
    URL / provider / model-name combinations. Regression cover: a custom
    OpenAI-compatible endpoint serving ``gpt-5.x`` was silently getting
    ``max_tokens`` and 400-ing on ``unsupported_parameter``."""

    def test_direct_openai_returns_max_completion_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://api.openai.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_completion_tokens": 4096}

    def test_local_endpoint_without_model_uses_max_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="http://localhost:11434/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_tokens": 4096}

    def test_openrouter_api_key_present_keeps_max_tokens_without_model_hint(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://openrouter.ai/api/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096) == {"max_tokens": 4096}

    # Model-name fallback — this is the regression guard.

    def test_custom_endpoint_serving_gpt5_uses_max_completion_tokens(self):
        """Third-party gateway + gpt-5.x: name-based detection must kick in."""
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="gpt-5.4") == {
                "max_completion_tokens": 4096
            }

    def test_openrouter_serving_gpt4o_uses_max_completion_tokens(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://openrouter.ai/api/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="openai/gpt-4o-mini") == {
                "max_completion_tokens": 4096
            }

    def test_custom_endpoint_serving_classic_llama_keeps_max_tokens(self):
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="llama3-70b") == {
                "max_tokens": 4096
            }

    def test_empty_model_falls_back_to_url_only(self):
        """No model hint → only the URL-based rule applies."""
        with (
            patch("agent.auxiliary_client._current_custom_base_url",
                  return_value="https://my-gateway.example.com/v1"),
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
        ):
            assert auxiliary_max_tokens_param(4096, model="") == {"max_tokens": 4096}
            assert auxiliary_max_tokens_param(4096, model=None) == {"max_tokens": 4096}


class TestBuildCallKwargsMaxTokens:
    """_build_call_kwargs should not cap output by default (#34530).

    Most chat-completions providers treat an omitted max_tokens as "use the
    model max", which is what we want for auxiliary tasks. An explicit cap only
    risks truncation or a wire-format 400 (GitHub Copilot / GPT-5 reject
    max_tokens; ZAI vision rejects it entirely). The Anthropic Messages wire is
    the one exception — max_tokens is a mandatory field there.
    """

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("copilot", "gpt-5.4", "https://api.githubcopilot.com"),
            ("copilot", "gpt-5.5", "https://api.githubcopilot.com"),
            ("custom", "gpt-5", "https://api.openai.com/v1"),
            ("openrouter", "anthropic/claude-sonnet-4.6", "https://openrouter.ai/api/v1"),
            ("nous", "hermes-4", "https://inference-api.nousresearch.com/v1"),
            ("custom", "qwen", "http://localhost:8080/v1"),
            ("zai", "glm-4v-flash", "https://open.bigmodel.cn/api/paas/v4"),
        ],
    )
    def test_omits_max_tokens_for_openai_compatible(self, provider, model, base_url):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1234,
            base_url=base_url,
        )
        assert "max_tokens" not in kwargs
        assert "max_completion_tokens" not in kwargs

    @pytest.mark.parametrize(
        "provider,model,base_url",
        [
            ("minimax", "minimax-m2", "https://api.minimax.io/v1"),
            ("custom", "claude", "https://proxy.example.com/anthropic/v1"),
        ],
    )
    def test_keeps_max_tokens_on_anthropic_wire(self, provider, model, base_url):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1234,
            base_url=base_url,
        )
        assert kwargs["max_tokens"] == 1234
        assert "max_completion_tokens" not in kwargs

    def test_keeps_max_tokens_for_nvidia_nim(self):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="nvidia",
            model="minimaxai/minimax-m3",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=4096,
            base_url="https://integrate.api.nvidia.com/v1",
        )
        assert kwargs["max_tokens"] == 4096

    @pytest.mark.parametrize(
        "provider,model,base_url,expected_key",
        [
            ("zai", "glm-5.2", "https://api.z.ai/api/coding/paas/v4", "max_tokens"),
            ("openrouter", "deepseek/deepseek-v4-flash:nitro", "https://openrouter.ai/api/v1", "max_tokens"),
            ("copilot", "gpt-5.5", "https://api.githubcopilot.com", "max_completion_tokens"),
            ("nous", "hermes-4", "https://inference-api.nousresearch.com/v1", "max_tokens"),
        ],
    )
    def test_moa_task_sends_max_tokens_on_openai_compatible(self, provider, model, base_url, expected_key):
        """MoA reference tasks must honor max_tokens regardless of provider.

        The ``reference_max_tokens`` config option (PR #56756) caps advisor output
        to reduce turn latency.  Before the fix, ``_build_call_kwargs`` silently
        dropped the value for OpenAI-compatible providers (PR #34845), so the cap
        never reached the API.  With the ``task`` parameter threaded through,
        ``task == "moa_reference"`` includes the output cap in kwargs.

        Models that require ``max_completion_tokens`` (GPT-5 family, Copilot)
        get the correct parameter name via ``auxiliary_max_tokens_param()``.
        """
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider=provider,
            model=model,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=800,
            base_url=base_url,
            task="moa_reference",
        )
        assert kwargs[expected_key] == 800

    def test_moa_task_exact_match(self):
        """Only task == "moa_reference" triggers the cap — not the aggregator,
        not arbitrary 'moa_' prefixed tasks."""
        from agent.auxiliary_client import _build_call_kwargs

        # 'moa_reference' → honored
        kw = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_reference",
        )
        assert kw["max_tokens"] == 500

        # 'moa_aggregator' → dropped (aggregator is the acting model, not an advisor)
        kw2 = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_aggregator",
        )
        assert "max_tokens" not in kw2

        # 'moa_custom_future' → dropped (only moa_reference is whitelisted)
        kw3 = _build_call_kwargs(
            provider="zai", model="glm-5.2",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=500,
            base_url="https://api.z.ai/api/coding/paas/v4",
            task="moa_custom_future",
        )
        assert "max_tokens" not in kw3


class TestNousTagsScoping:
    def test_tags_injected_when_provider_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", False)

        kwargs = aux._build_call_kwargs(
            provider="nous",
            model="hermes-4",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert kwargs["extra_body"]["tags"] == aux._nous_portal_tags()

    def test_tags_not_injected_for_gemini_when_main_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", True)

        kwargs = aux._build_call_kwargs(
            provider="gemini",
            model="gemini-2.5-flash",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "extra_body" not in kwargs

    def test_tags_not_injected_for_openrouter_when_main_is_nous(self, monkeypatch):
        import agent.auxiliary_client as aux

        monkeypatch.setattr(aux, "auxiliary_is_nous", True)

        kwargs = aux._build_call_kwargs(
            provider="openrouter",
            model="openai/gpt-5.4",
            messages=[{"role": "user", "content": "hi"}],
        )

        assert "extra_body" not in kwargs


class TestKimiTemperatureOmitted:
    """Kimi/Moonshot models should have temperature OMITTED from API kwargs.

    The Kimi gateway selects the correct temperature server-side based on the
    active mode (thinking → 1.0, non-thinking → 0.6).  Sending any temperature
    value conflicts with gateway-managed defaults.
    """

    @pytest.mark.parametrize(
        "model",
        [
            "kimi-for-coding",
            "kimi-k2.5",
            "kimi-k2.6",
            "kimi-k2-turbo-preview",
            "kimi-k2-0905-preview",
            "kimi-k2-thinking",
            "kimi-k2-thinking-turbo",
            "kimi-k2-instruct",
            "kimi-k2-instruct-0905",
            "moonshotai/kimi-k2.5",
            "moonshotai/Kimi-K2-Thinking",
            "moonshotai/Kimi-K2-Instruct",
        ],
    )
    def test_kimi_models_omit_temperature(self, model):
        """No kimi model should have a temperature key in kwargs."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert "temperature" not in kwargs

    def test_kimi_for_coding_no_temperature_when_none(self):
        """When caller passes temperature=None, still no temperature key."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-for-coding",
            messages=[{"role": "user", "content": "hello"}],
            temperature=None,
        )

        assert "temperature" not in kwargs

    def test_sync_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.asyncio
    async def test_async_call_omits_temperature(self):
        client = MagicMock()
        client.base_url = "https://api.kimi.com/coding/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        with patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-for-coding"),
        ), patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=("auto", "kimi-for-coding", None, None, None),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0.1,
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["model"] == "kimi-for-coding"
        assert "temperature" not in kwargs

    @pytest.mark.parametrize(
        "model",
        [
            "anthropic/claude-sonnet-4-6",
            "gpt-5.4",
            "deepseek-chat",
        ],
    )
    def test_non_kimi_models_preserve_temperature(self, model):
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="openrouter",
            model=model,
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.3,
        )

        assert kwargs["temperature"] == 0.3

    @pytest.mark.parametrize(
        "base_url",
        [
            "https://api.moonshot.ai/v1",
            "https://api.moonshot.cn/v1",
            "https://api.kimi.com/coding/v1",
        ],
    )
    def test_kimi_k2_5_omits_temperature_regardless_of_endpoint(self, base_url):
        """Temperature is omitted regardless of which Kimi endpoint is used."""
        from agent.auxiliary_client import _build_call_kwargs

        kwargs = _build_call_kwargs(
            provider="kimi-coding",
            model="kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.1,
            base_url=base_url,
        )

        assert "temperature" not in kwargs


class TestAuxiliaryTaskExtraBody:
    def test_sync_call_merges_task_extra_body_from_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create.return_value = response

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {
                        "enable_thinking": False,
                        "reasoning": {"effort": "none"},
                    }
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"metadata": {"source": "test"}},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is False
        assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}
        assert kwargs["extra_body"]["metadata"] == {"source": "test"}

    @pytest.mark.asyncio
    async def test_async_call_explicit_extra_body_overrides_task_config(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        response = MagicMock()
        client.chat.completions.create = AsyncMock(return_value=response)

        config = {
            "auxiliary": {
                "session_search": {
                    "extra_body": {"enable_thinking": False}
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            result = await async_call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
                extra_body={"enable_thinking": True},
            )

        assert result is response
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["enable_thinking"] is True

    def test_reasoning_effort_shorthand_folds_into_extra_body(self):
        """auxiliary.<task>.reasoning_effort becomes extra_body.reasoning."""
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        client.chat.completions.create.return_value = MagicMock()

        config = {
            "auxiliary": {
                "session_search": {"reasoning_effort": "low"}
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            call_llm(
                task="session_search",
                messages=[{"role": "user", "content": "hello"}],
            )

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["reasoning"] == {"enabled": True, "effort": "low"}

    def test_reasoning_effort_none_disables(self):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        client.chat.completions.create.return_value = MagicMock()

        config = {"auxiliary": {"session_search": {"reasoning_effort": "none"}}}

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            call_llm(task="session_search", messages=[{"role": "user", "content": "hi"}])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["reasoning"] == {"enabled": False}

    def test_explicit_extra_body_reasoning_wins_over_shorthand(self):
        """config extra_body.reasoning beats the reasoning_effort shorthand."""
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        client.chat.completions.create.return_value = MagicMock()

        config = {
            "auxiliary": {
                "session_search": {
                    "reasoning_effort": "xhigh",
                    "extra_body": {"reasoning": {"effort": "none"}},
                }
            }
        }

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ):
            call_llm(task="session_search", messages=[{"role": "user", "content": "hi"}])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"]["reasoning"] == {"effort": "none"}

    def test_invalid_reasoning_effort_ignored_with_warning(self, caplog):
        client = MagicMock()
        client.base_url = "https://api.example.com/v1"
        client.chat.completions.create.return_value = MagicMock()

        config = {"auxiliary": {"session_search": {"reasoning_effort": "warp9"}}}

        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "glm-4.5-air"),
        ), caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            call_llm(task="session_search", messages=[{"role": "user", "content": "hi"}])

        kwargs = client.chat.completions.create.call_args.kwargs
        assert "reasoning" not in (kwargs.get("extra_body") or {})
        assert any("reasoning_effort" in rec.message for rec in caplog.records)

    def test_empty_reasoning_effort_is_noop(self):
        """The DEFAULT_CONFIG ships reasoning_effort: '' — must add nothing."""
        from agent.auxiliary_client import _get_task_extra_body

        config = {"auxiliary": {"session_search": {"reasoning_effort": ""}}}
        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config):
            assert _get_task_extra_body("session_search") == {}

    @pytest.mark.parametrize("moa_task", ["moa_reference", "moa_aggregator"])
    def test_moa_tasks_reject_task_level_reasoning_effort(self, moa_task, caplog):
        """MoA reasoning is per-slot in the preset — the auxiliary-task
        shorthand is ignored with a warning pointing at the preset config."""
        from agent.auxiliary_client import _get_task_extra_body

        config = {"auxiliary": {moa_task: {"reasoning_effort": "xhigh"}}}
        with patch("hermes_cli.config.load_config", return_value=config), patch("hermes_cli.config.load_config_readonly", return_value=config), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            result = _get_task_extra_body(moa_task)

        assert "reasoning" not in result
        assert any("per-slot" in rec.message for rec in caplog.records)

    @pytest.mark.parametrize("moa_task", ["moa_reference", "moa_aggregator"])
    def test_moa_default_config_has_no_reasoning_effort(self, moa_task):
        """Invariant: the shipped MoA auxiliary blocks must not grow a
        reasoning_effort key — per-slot preset config is the only surface."""
        from hermes_cli.config import DEFAULT_CONFIG

        assert "reasoning_effort" not in DEFAULT_CONFIG["auxiliary"][moa_task]

    def test_anthropic_aux_client_forwards_extra_body_reasoning(self):
        """_AnthropicCompletionsAdapter passes extra_body.reasoning into
        build_anthropic_kwargs as reasoning_config."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(MagicMock(), "claude-sonnet-4-6", is_oauth=False)

        with patch("agent.anthropic_adapter.build_anthropic_kwargs",
                   return_value={"model": "claude-sonnet-4-6", "messages": [], "max_tokens": 64}) as mock_bak, \
             patch("agent.anthropic_adapter.create_anthropic_message") as mock_create, \
             patch("agent.transports.get_transport") as mock_gt:
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok", tool_calls=None, reasoning=None, finish_reason="stop",
                usage=None, provider_data=None,
            )
            adapter.create(
                model="claude-sonnet-4-6",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=64,
                extra_body={"reasoning": {"enabled": True, "effort": "low"}},
            )

        assert mock_bak.call_args.kwargs["reasoning_config"] == {
            "enabled": True, "effort": "low",
        }
        mock_create.assert_called_once()

    def _run_anthropic_adapter(self, *, call_extra_body=None, bak_result=None):
        """Drive _AnthropicCompletionsAdapter.create() with mocked SDK layers;
        return the api_kwargs handed to create_anthropic_message."""
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        adapter = _AnthropicCompletionsAdapter(MagicMock(), "claude-sonnet-4-6", is_oauth=False)
        bak_result = bak_result or {
            "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 64,
        }
        with patch("agent.anthropic_adapter.build_anthropic_kwargs",
                   return_value=dict(bak_result)), \
             patch("agent.anthropic_adapter.create_anthropic_message") as mock_create, \
             patch("agent.transports.get_transport") as mock_gt:
            mock_gt.return_value.normalize_response.return_value = MagicMock(
                content="ok", tool_calls=None, reasoning=None, finish_reason="stop",
                usage=None, provider_data=None,
            )
            kwargs = {
                "model": "claude-sonnet-4-6",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 64,
            }
            if call_extra_body is not None:
                kwargs["extra_body"] = call_extra_body
            adapter.create(**kwargs)
        return mock_create.call_args.args[1]

    def test_anthropic_aux_extra_body_passthrough(self):
        """Bug B (#37217): vendor fields in extra_body reach the Anthropic SDK."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={"thinking": {"type": "disabled"}, "metadata": {"user_id": "u1"}},
        )
        assert api_kwargs["extra_body"] == {
            "thinking": {"type": "disabled"}, "metadata": {"user_id": "u1"},
        }

    def test_anthropic_aux_extra_body_excludes_reasoning_and_private_keys(self):
        """The OpenAI-shaped reasoning dict is translated (not forwarded), and
        private _-prefixed plumbing keys never reach the wire."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={
                "reasoning": {"enabled": True, "effort": "low"},
                "_internal": "plumbing",
                "metadata": {"user_id": "u1"},
            },
        )
        assert api_kwargs["extra_body"] == {"metadata": {"user_id": "u1"}}

    def test_anthropic_aux_extra_body_merges_over_existing(self):
        """Caller extra_body merges on top of what build_anthropic_kwargs
        already emitted (fast-mode speed) instead of clobbering it."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={"metadata": {"user_id": "u1"}},
            bak_result={
                "model": "claude-sonnet-4-6", "messages": [], "max_tokens": 64,
                "extra_body": {"speed": "fast"},
            },
        )
        assert api_kwargs["extra_body"] == {
            "speed": "fast", "metadata": {"user_id": "u1"},
        }

    def test_anthropic_aux_no_extra_body_unchanged(self):
        """Regression guard: no caller extra_body -> kwargs identical to before."""
        api_kwargs = self._run_anthropic_adapter(call_extra_body=None)
        assert "extra_body" not in api_kwargs

    def test_anthropic_aux_reasoning_only_extra_body_adds_nothing(self):
        """extra_body containing ONLY the reasoning key must not create an
        empty extra_body dict on the wire."""
        api_kwargs = self._run_anthropic_adapter(
            call_extra_body={"reasoning": {"enabled": False}},
        )
        assert "extra_body" not in api_kwargs

    def test_no_warning_when_provider_is_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom' — OPENAI_BASE_URL is expected."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client._resolve_custom_runtime",
                   return_value=("http://localhost:11434/v1", "test-key", None)), \
             patch("agent.auxiliary_client.OpenAI") as mock_openai, \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            mock_openai.return_value = MagicMock()
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom'"

    def test_no_warning_when_provider_is_named_custom(self, monkeypatch, caplog):
        """No warning when the provider is 'custom:myname' — base_url comes from config."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        with patch("agent.auxiliary_client._read_main_provider", return_value="custom:ollama-local"), \
             patch("agent.auxiliary_client._read_main_model", return_value="llama3"), \
             patch("agent.auxiliary_client.resolve_provider_client",
                   return_value=(MagicMock(), "llama3")), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when provider is 'custom:*'"

    def test_no_warning_when_openai_base_url_not_set(self, monkeypatch, caplog):
        """No warning when OPENAI_BASE_URL is absent."""
        import agent.auxiliary_client as mod
        monkeypatch.setattr(mod, "_stale_base_url_warned", False)
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

        with patch("agent.auxiliary_client._read_main_provider", return_value="openrouter"), \
             patch("agent.auxiliary_client._read_main_model", return_value="google/gemini-flash"), \
             caplog.at_level(logging.WARNING, logger="agent.auxiliary_client"):
            _resolve_auto()

        assert not any("OPENAI_BASE_URL is set" in rec.message for rec in caplog.records), \
            "Should NOT warn when OPENAI_BASE_URL is not set"


class TestBuildCallKwargsToolDedup:
    """_build_call_kwargs must deduplicate tool names before passing to API.

    Providers like Google Vertex, Azure, and Bedrock reject requests with
    duplicate tool names (HTTP 400).  This guard converts a hard failure into
    a warning log so agent turns succeed even if an upstream injection path
    regresses.  See: https://github.com/NousResearch/hermes-agent/issues/18478
    """

    def _make_tool(self, name: str) -> dict:
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"Tool {name}",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_unique_tools_pass_through_unchanged(self):
        tools = [self._make_tool("alpha"), self._make_tool("beta")]
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=tools,
        )
        assert len(kwargs["tools"]) == 2
        names = [t["function"]["name"] for t in kwargs["tools"]]
        assert names == ["alpha", "beta"]

    def test_duplicate_tool_names_are_deduplicated(self):
        """RED test — must fail until dedup guard is added."""
        tools = [
            self._make_tool("lcm_grep"),
            self._make_tool("lcm_describe"),
            self._make_tool("lcm_grep"),  # duplicate
            self._make_tool("lcm_expand"),
            self._make_tool("lcm_describe"),  # duplicate
        ]
        kwargs = _build_call_kwargs(
            provider="google", model="gemini-2.5-pro", messages=[], tools=tools,
        )
        result_tools = kwargs["tools"]
        names = [t["function"]["name"] for t in result_tools]
        # Must be deduplicated — no repeated names
        assert len(names) == len(set(names)), (
            f"Duplicate tool names found: {names}"
        )
        assert len(result_tools) == 3  # lcm_grep, lcm_describe, lcm_expand

    def test_empty_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=[],
        )
        assert kwargs.get("tools") == [] or "tools" not in kwargs

    def test_none_tools_unchanged(self):
        kwargs = _build_call_kwargs(
            provider="openai", model="gpt-4o", messages=[], tools=None,
        )
        assert "tools" not in kwargs


class TestNvidiaBillingHeaders:
    """NVIDIA NIM billing-origin headers are scoped to NVIDIA cloud."""

    def test_resolve_provider_client_cloud_adds_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs["default_headers"]
        assert headers["X-BILLING-INVOKE-ORIGIN"] == "HermesAgent"

    def test_resolve_provider_client_local_nim_skips_billing_origin_header(self, monkeypatch):
        monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-key")
        monkeypatch.setenv("NVIDIA_BASE_URL", "http://localhost:8000/v1")
        mock_openai = MagicMock()
        mock_openai.return_value = MagicMock(name="nvidia-local-client")

        with patch("agent.auxiliary_client.OpenAI", mock_openai):
            client, model = resolve_provider_client(
                provider="nvidia",
                model="nvidia/test-model",
            )

        assert client is not None
        assert model == "nvidia/test-model"
        call_kwargs = mock_openai.call_args[1]
        headers = call_kwargs.get("default_headers", {})
        assert "X-BILLING-INVOKE-ORIGIN" not in headers
