"""Auxiliary-client provider adapters: Anthropic compat and Codex completions."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import time
from types import SimpleNamespace
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from agent.auxiliary_client import (
    call_llm,
    async_call_llm,
    CodexAuxiliaryClient,
    _build_call_kwargs,
    _CodexCompletionsAdapter,
)



class TestAnthropicCompatImageConversion:
    """Tests for _is_anthropic_compat_endpoint and _convert_openai_images_to_anthropic."""

    def test_known_providers_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("minimax", "")
        assert _is_anthropic_compat_endpoint("minimax-cn", "")

    def test_openrouter_not_detected(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert not _is_anthropic_compat_endpoint("openrouter", "")
        assert not _is_anthropic_compat_endpoint("anthropic", "")

    def test_url_based_detection(self):
        from agent.auxiliary_client import _is_anthropic_compat_endpoint
        assert _is_anthropic_compat_endpoint("custom", "https://api.minimax.io/anthropic")
        assert _is_anthropic_compat_endpoint("custom", "https://example.com/anthropic/v1")
        assert not _is_anthropic_compat_endpoint("custom", "https://api.openai.com/v1")

    def test_base64_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][1]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "iVBOR="

    def test_url_image_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "https://example.com/img.jpg"}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        img_block = result[0]["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "url"
        assert img_block["source"]["url"] == "https://example.com/img.jpg"

    def test_text_only_messages_unchanged(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{"role": "user", "content": "Hello"}]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0] is messages[0]  # same object, not copied

    def test_jpeg_media_type_parsed(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/="}}
            ]
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["source"]["media_type"] == "image/jpeg"

    def test_base64_video_converted_to_video_block(self):
        # MiniMax M3's Anthropic-compatible endpoint expects type="video"
        # (not OpenAI's "video_url", not "input_video").
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "What happens in this clip?"},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        vid_block = result[0]["content"][1]
        assert vid_block["type"] == "video"
        assert vid_block["source"]["type"] == "base64"
        assert vid_block["source"]["media_type"] == "video/mp4"
        assert vid_block["source"]["data"] == "AAAA"

    def test_video_media_type_parsed_from_data_uri(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "data:video/quicktime;base64,QQ=="}}
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["source"]["media_type"] == "video/quicktime"

    def test_url_video_converted_to_video_block(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}}
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        vid_block = result[0]["content"][0]
        assert vid_block["type"] == "video"
        assert vid_block["source"] == {"type": "url", "url": "https://example.com/clip.mp4"}

    def test_mixed_image_and_video_both_converted(self):
        from agent.auxiliary_client import _convert_openai_images_to_anthropic
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBOR"}},
                {"type": "video_url", "video_url": {"url": "data:video/mp4;base64,AAAA"}},
            ],
        }]
        result = _convert_openai_images_to_anthropic(messages)
        assert result[0]["content"][0]["type"] == "image"
        assert result[0]["content"][1]["type"] == "video"


class TestAnthropicAuxiliaryReasoningTranslation:
    """Native Anthropic aux adapters must receive normalized Hermes reasoning.

    MoA slot reasoning is carried through call_llm as a Hermes
    ``reasoning_config``. The native Anthropic Messages path cannot consume the
    generic OpenAI-style ``extra_body.reasoning`` fallback, so assert the final
    ``messages.create`` kwargs contain Anthropic's provider-aware wire shape.
    """

    @staticmethod
    def _build_adapter(model="claude-fable-5"):
        from agent.auxiliary_client import _AnthropicCompletionsAdapter

        captured = {}

        class _Messages:
            def create(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="ok")],
                    stop_reason="end_turn",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                )

        real_client = SimpleNamespace(messages=_Messages())
        return _AnthropicCompletionsAdapter(real_client, model), captured

    def test_reasoning_config_reaches_native_anthropic_wire_kwargs(self):
        adapter, captured = self._build_adapter()

        adapter.create(
            model="claude-fable-5",
            messages=[{"role": "user", "content": "hi"}],
            _reasoning_config={"enabled": True, "effort": "medium"},
        )

        assert captured["thinking"] == {"type": "adaptive", "display": "summarized"}
        assert captured["output_config"] == {"effort": "medium"}
        assert "extra_body" not in captured

    def test_build_call_kwargs_private_reasoning_only_for_anthropic_messages(self):
        anthropic_kwargs = _build_call_kwargs(
            "anthropic",
            "claude-fable-5",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.anthropic.com/v1",
        )
        assert anthropic_kwargs["_reasoning_config"] == {"enabled": True, "effort": "medium"}

        proxy_kwargs = _build_call_kwargs(
            "custom",
            "claude-fable-5",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://example.test/anthropic/v1",
        )
        assert proxy_kwargs["_reasoning_config"] == {"enabled": True, "effort": "medium"}

        openai_wire_kwargs = _build_call_kwargs(
            "custom",
            "gpt-compatible",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://example.test/v1",
        )
        assert "_reasoning_config" not in openai_wire_kwargs


class TestAuxiliaryProviderProfileReasoning:
    """Auxiliary calls must reuse provider-profile reasoning wire shapes."""

    def test_kimi_reasoning_uses_top_level_effort(self):
        kwargs = _build_call_kwargs(
            "kimi-coding",
            "kimi-k2-turbo-preview",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "medium"},
            base_url="https://api.moonshot.ai/v1",
        )

        assert kwargs["reasoning_effort"] == "medium"
        assert "reasoning" not in kwargs.get("extra_body", {})
        assert "thinking" not in kwargs.get("extra_body", {})

    def test_gemini_reasoning_uses_thinking_config(self):
        kwargs = _build_call_kwargs(
            "gemini",
            "gemini-3.5-flash",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "high"},
            base_url="https://generativelanguage.googleapis.com/v1beta",
        )

        assert kwargs["extra_body"]["thinking_config"] == {
            "includeThoughts": True,
            "thinkingLevel": "high",
        }
        assert "reasoning" not in kwargs["extra_body"]

    def test_custom_openai_compatible_reasoning_uses_top_level_effort(self):
        kwargs = _build_call_kwargs(
            "custom",
            "glm-5.2",
            [{"role": "user", "content": "hi"}],
            reasoning_config={"enabled": True, "effort": "max"},
            base_url="https://example.test/v1",
        )

        assert kwargs["reasoning_effort"] == "max"
        assert "reasoning" not in kwargs.get("extra_body", {})

    @pytest.mark.asyncio
    async def test_async_call_llm_preserves_profile_reasoning_kwargs(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )
        create = AsyncMock(return_value=response)
        client = SimpleNamespace(
            base_url="https://api.moonshot.ai/v1",
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=create),
            ),
        )

        with patch(
            "agent.auxiliary_client._resolve_task_provider_model",
            return_value=(
                "kimi-coding",
                "kimi-k2-turbo-preview",
                "https://api.moonshot.ai/v1",
                "test-key",
                None,
            ),
        ), patch(
            "agent.auxiliary_client._get_cached_client",
            return_value=(client, "kimi-k2-turbo-preview"),
        ):
            result = await async_call_llm(
                provider="kimi-coding",
                model="kimi-k2-turbo-preview",
                messages=[{"role": "user", "content": "hi"}],
                reasoning_config={"enabled": True, "effort": "high"},
            )

        assert result is response
        final_kwargs = create.call_args.kwargs
        assert final_kwargs["reasoning_effort"] == "high"
        assert "reasoning" not in final_kwargs.get("extra_body", {})


class TestCodexAdapterReasoningTranslation:
    """Verify _CodexCompletionsAdapter translates extra_body.reasoning
    into the Responses API's top-level reasoning + include fields, matching
    agent/transports/codex.py::build_kwargs() behavior.

    Regression for user feedback (Apr 26): auxiliary callers that configure
    reasoning via auxiliary.<task>.extra_body.reasoning had that config
    silently dropped because the adapter only forwarded messages/model/tools.
    """

    @staticmethod
    def _build_adapter():
        """Build a _CodexCompletionsAdapter with a mocked responses.create()."""
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        # The event-driven path consumes ``responses.create(stream=True)`` as a
        # raw iterable of SSE events.  Emit a minimal stream containing one
        # ``response.output_item.done`` (message) and a ``response.completed``
        # terminal frame.
        message_item = SimpleNamespace(
            type="message",
            role="assistant",
            status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed",
                    id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.3-codex")
        return adapter, captured_kwargs

    def test_reasoning_effort_medium_translated_to_top_level(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "medium"}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_minimal_clamped_to_low(self):
        """Codex backend rejects 'minimal'; adapter clamps to 'low' per main transport."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "minimal"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_low_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "low"}},
        )
        assert captured.get("reasoning") == {"effort": "low", "summary": "auto"}

    def test_reasoning_effort_high_passed_through(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": "high"}},
        )
        assert captured.get("reasoning") == {"effort": "high", "summary": "auto"}

    def test_reasoning_disabled_omits_reasoning_and_include(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"enabled": False}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_reasoning_default_effort_when_only_enabled_flag(self):
        """extra_body={"reasoning": {}} (truthy enabled by omission) → default 'medium'."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_no_extra_body_means_no_reasoning_keys(self):
        """Baseline: without extra_body, no reasoning/include is sent (preserves
        current behavior for callers that don't opt in)."""
        adapter, captured = self._build_adapter()
        adapter.create(messages=[{"role": "user", "content": "hi"}])
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_extra_body_without_reasoning_key_is_noop(self):
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"metadata": {"source": "test"}},
        )
        assert "reasoning" not in captured
        assert "include" not in captured

    def test_non_dict_reasoning_value_is_ignored_gracefully(self):
        """Defensive: if a caller accidentally passes a string/None, we
        silently skip instead of crashing inside the adapter."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": "medium"},  # wrong shape — must not crash
        )
        assert "reasoning" not in captured

    def test_reasoning_effort_null_falls_back_to_medium(self):
        """Parity with agent/transports/codex.py::build_kwargs() — falsy
        ``effort`` (None / empty / 0) keeps the default ``medium`` instead
        of being forwarded to Codex.  Codex rejects ``{"effort": null}``
        with HTTP 400 (Invalid value for parameter `reasoning.effort`)."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": None}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_empty_string_falls_back_to_medium(self):
        """Empty-string effort (e.g. ``effort: ""`` in YAML) is falsy in
        the main-agent path's truthy check; mirror that here so the same
        config produces the same result."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": ""}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]

    def test_reasoning_effort_zero_falls_back_to_medium(self):
        """Numeric ``0`` is also falsy — the docstring lists it explicitly,
        so cover the contract.  Codex would reject ``{"effort": 0}`` the
        same way it rejects ``null``."""
        adapter, captured = self._build_adapter()
        adapter.create(
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning": {"effort": 0}},
        )
        assert captured.get("reasoning") == {"effort": "medium", "summary": "auto"}
        assert captured.get("include") == ["reasoning.encrypted_content"]


class TestCodexAdapterPromptCacheKey:
    """_CodexCompletionsAdapter emits a stable content-addressed prompt_cache_key
    on the Codex/Responses aux path, matching the main transport
    (agent/transports/codex.py). Regression for issue #53735: MoA acting-
    aggregator and other auxiliary Responses calls stayed cache-cold because
    the adapter never set prompt_cache_key.
    """

    @staticmethod
    def _build_adapter(base_url="https://chatgpt.com/backend-api/codex", model="gpt-5.5"):
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        message_item = SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed", id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.base_url = base_url
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, model)
        return adapter, captured_kwargs

    def test_cache_key_set_and_prefixed(self):
        adapter, captured = self._build_adapter()
        adapter.create(messages=[
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "hi"},
        ])
        key = captured.get("prompt_cache_key")
        assert isinstance(key, str) and key.startswith("pck_")

    def test_cache_key_stable_across_identical_prefix(self):
        """Same instructions + tools → same key (content-addressed, not per-call)."""
        a1, c1 = self._build_adapter()
        a1.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "first"},
        ])
        a2, c2 = self._build_adapter()
        a2.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "second — different user turn"},
        ])
        # User-turn content differs but the static prefix (instructions) matches,
        # so the routing key is identical → same warm cache bucket.
        assert c1["prompt_cache_key"] == c2["prompt_cache_key"]

    def test_cache_key_differs_on_different_instructions(self):
        a1, c1 = self._build_adapter()
        a1.create(messages=[{"role": "system", "content": "SYS-A"}, {"role": "user", "content": "x"}])
        a2, c2 = self._build_adapter()
        a2.create(messages=[{"role": "system", "content": "SYS-B"}, {"role": "user", "content": "x"}])
        assert c1["prompt_cache_key"] != c2["prompt_cache_key"]

    def test_cache_key_skipped_for_xai_host(self):
        """xAI Responses takes the key in extra_body, not top-level — skip here."""
        adapter, captured = self._build_adapter(base_url="https://api.x.ai/v1")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_key" not in captured

    def test_cache_key_skipped_for_github_copilot_host(self):
        """GitHub/Copilot Responses opts out of cache-key routing entirely."""
        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_key" not in captured

    @pytest.mark.parametrize("model", [
        "gpt-4.1",
        "gpt-5.1-codex-max",
        "openai.gpt-5.5-pro",
    ])
    def test_extended_cache_models_set_prompt_cache_retention(self, model):
        adapter, captured = self._build_adapter(
            base_url="https://bedrock-mantle.us-west-2.api.aws/v1",
            model=model,
        )
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert captured["prompt_cache_retention"] == "24h"

    def test_prompt_cache_retention_skipped_for_codex_backend(self):
        adapter, captured = self._build_adapter()
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com/v1",
        "https://example.services.ai.azure.com/openai/v1",
        "https://responses.example.com/v1",
    ])
    def test_prompt_cache_retention_skipped_for_other_compatible_endpoints(self, base_url):
        adapter, captured = self._build_adapter(base_url=base_url)
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

    def test_prompt_cache_retention_skipped_for_xai_and_github_hosts(self):
        adapter, captured = self._build_adapter(base_url="https://api.x.ai/v1")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured

        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=[
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "hi"},
        ])
        assert "prompt_cache_retention" not in captured


class TestCodexAdapterGithubResponsesMessageIdDrop:
    """_CodexCompletionsAdapter must drop codex_message_items ``id`` when
    talking to Copilot (githubcopilot.com), independent of the main
    transport's build_kwargs path. Auxiliary calls (context compression,
    flush_memories, MoA aggregation) route through this adapter instead of
    agent/transports/codex.py, so they need the same #32716 guard applied
    separately — Copilot binds replayed ids to a backend "connection" that
    doesn't survive credential rotation/gateway restarts, and rejects a
    stale id with HTTP 401 regardless of its length.
    """

    @staticmethod
    def _build_adapter(base_url):
        from agent.auxiliary_client import _CodexCompletionsAdapter
        from types import SimpleNamespace

        message_item = SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text="hi")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(
                type="response.completed",
                response=SimpleNamespace(
                    status="completed", id="resp_test",
                    usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
                ),
            ),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        captured_kwargs = {}

        def _create(**kwargs):
            captured_kwargs.update(kwargs)
            return _FakeCreateStream()

        real_client = MagicMock()
        real_client.base_url = base_url
        real_client.responses.create = _create
        adapter = _CodexCompletionsAdapter(real_client, "gpt-5.5")
        return adapter, captured_kwargs

    @staticmethod
    def _replay_messages():
        return [
            {"role": "system", "content": "You are helpful."},
            {
                "role": "assistant",
                "content": "pong",
                "codex_message_items": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "in_progress",
                        "content": [{"type": "output_text", "text": "pong"}],
                        "id": "msg_short_but_connection_scoped",
                        "phase": "final_answer",
                    }
                ],
            },
            {"role": "user", "content": "continue"},
        ]

    def test_drops_message_id_for_github_copilot_host(self):
        adapter, captured = self._build_adapter(base_url="https://api.githubcopilot.com")
        adapter.create(messages=self._replay_messages())
        message_item = next(
            item for item in captured["input"] if item.get("type") == "message"
        )
        assert "id" not in message_item
        assert message_item["phase"] == "final_answer"
        assert message_item["status"] == "in_progress"
        assert message_item["content"] == [{"type": "output_text", "text": "pong"}]

    def test_keeps_message_id_for_codex_backend_host(self):
        adapter, captured = self._build_adapter(
            base_url="https://chatgpt.com/backend-api/codex"
        )
        adapter.create(messages=self._replay_messages())
        message_item = next(
            item for item in captured["input"] if item.get("type") == "message"
        )
        assert message_item["id"] == "msg_short_but_connection_scoped"


class TestCodexAuxiliaryAdapterTimeout:
    def test_forwards_timeout_to_responses_create(self):
        message_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="summary")],
        )
        events = [
            SimpleNamespace(type="response.output_item.done", item=message_item),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed", id="r1", usage=None,
            )),
        ]

        class _FakeCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        response = adapter.create(
            messages=[{"role": "user", "content": "summarize this"}],
            timeout=12.5,
        )

        assert fake_client.responses.kwargs["timeout"] == 12.5
        assert fake_client.responses.kwargs["stream"] is True
        assert response.choices[0].message.content == "summary"

    def test_enforces_total_timeout_while_stream_keeps_emitting_events(self):
        class _SlowAliveCreateStream:
            def __iter__(self):
                for _ in range(5):
                    time.sleep(0.03)
                    yield SimpleNamespace(type="response.in_progress")

            def close(self): pass

        class FakeResponses:
            def create(self, **kwargs):
                return _SlowAliveCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses(), close=lambda: None)
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        started = time.monotonic()
        with pytest.raises(TimeoutError):
            adapter.create(
                messages=[{"role": "user", "content": "summarize this"}],
                timeout=0.05,
            )

        assert time.monotonic() - started < 0.14


class TestCodexAuxiliaryToolMessageConversion:
    """Regression for issue #5709.

    The auxiliary Codex adapter used to maintain its own chat->Responses
    conversion loop that forwarded every non-system message's ``role``
    verbatim into Responses ``input[]``. When ``flush_memories()`` /
    compression replayed real session history containing assistant
    ``tool_calls`` and ``role="tool"`` results, the tool messages leaked
    into the request and the Responses API rejected them with
    ``HTTP 400: Invalid value: 'tool'. Supported values are: 'assistant',
    'system', 'developer', and 'user'.``

    The fix routes the auxiliary path through the SAME shared converter the
    main agent transport uses (``_chat_messages_to_responses_input``), so
    no Responses request ever includes a raw ``role="tool"`` input item.
    """

    def _capture_input(self, messages):
        from agent.auxiliary_client import _CodexCompletionsAdapter

        class _FakeCreateStream:
            def __iter__(self):
                return iter([
                    SimpleNamespace(type="response.created"),
                    SimpleNamespace(
                        type="response.output_item.done",
                        item=SimpleNamespace(
                            type="message",
                            content=[SimpleNamespace(type="output_text", text="ok")],
                        ),
                    ),
                    SimpleNamespace(type="response.completed", response=SimpleNamespace(
                        status="completed", id="r1", usage=None,
                    )),
                ])

            def close(self):
                pass

        class FakeResponses:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return _FakeCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")
        adapter.create(messages=messages, model="gpt-5.5")
        return fake_client.responses.kwargs

    def test_tool_history_never_leaks_role_tool(self):
        messages = [
            {"role": "system", "content": "You are a memory summarizer."},
            {"role": "user", "content": "What files did I touch?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_abc123",
                    "type": "function",
                    "function": {"name": "search_files", "arguments": '{"pattern":"foo"}'},
                }],
            },
            {"role": "tool", "tool_call_id": "call_abc123", "content": "Found 3 matches"},
            {"role": "assistant", "content": "You touched bar.py."},
        ]
        kwargs = self._capture_input(messages)
        input_items = kwargs["input"]

        # No raw role="tool" item reaches the Responses API (the 400 trigger).
        assert not any(it.get("role") == "tool" for it in input_items)

        # Assistant tool call -> function_call item with a call_id.
        function_calls = [it for it in input_items if it.get("type") == "function_call"]
        assert function_calls, "assistant tool_call must become a function_call item"
        assert function_calls[0]["call_id"] == "call_abc123"
        assert function_calls[0]["name"] == "search_files"

        # Tool result -> function_call_output with the matching call_id.
        outputs = [it for it in input_items if it.get("type") == "function_call_output"]
        assert outputs, "tool result must become a function_call_output item"
        assert outputs[0]["call_id"] == "call_abc123"

        # System message is hoisted to instructions, not left in input[].
        assert kwargs["instructions"] == "You are a memory summarizer."
        assert not any(it.get("role") == "system" for it in input_items)

    def test_plain_text_history_still_works(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ]
        kwargs = self._capture_input(messages)
        input_items = kwargs["input"]
        roles = [it.get("role") for it in input_items]
        assert "user" in roles and "assistant" in roles
        assert not any(it.get("role") == "tool" for it in input_items)
        assert kwargs["instructions"] == "sys"


class TestCodexAuxiliaryAdapterNullOutputRecovery:
    def test_recovers_output_item_when_terminal_event_has_null_output(self):
        """Regression for #11179 in auxiliary calls.

        The wire shape that broke the SDK is ``response.completed`` with
        ``response.output = null``.  The event-driven path is structurally
        immune because it reconstructs from ``response.output_item.done``
        events and never reads the terminal event's ``output`` field for
        content.  Assert the auxiliary path returns the streamed item even
        when the terminal frame's output is ``null``.
        """
        output_item = SimpleNamespace(
            type="message",
            content=[SimpleNamespace(type="output_text", text="aux survived")],
        )
        events = [
            SimpleNamespace(type="response.created"),
            SimpleNamespace(type="response.output_item.done", item=output_item),
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed",
                id="resp_null_output",
                # This is the field the SDK helper would have iterated and crashed on:
                output=None,
                usage=None,
            )),
        ]

        class _NullOutputCreateStream:
            def __iter__(self): return iter(events)
            def close(self): pass

        class FakeResponses:
            def create(self, **kwargs):
                return _NullOutputCreateStream()

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

        response = adapter.create(messages=[{"role": "user", "content": "summarize"}])

        assert response.choices[0].message.content == "aux survived"

    def test_handles_final_output_is_none_after_consumer(self):
        """Regression for #33368 — defense against ``final.output`` being ``None``.

        The event-driven consumer always sets ``final.output`` to a list, so this
        shape can't come from our own path. But a mocked client / compatibility
        shim that returns a typed Response with ``output=None`` directly (or a
        future code path that wraps a different consumer) would crash on
        ``for item in getattr(final, "output", [])`` because ``getattr`` returns
        ``None`` (not the default) when the attribute exists but is ``None``.
        Coerce with ``or []`` to handle this defensively.
        """
        # Stream that returns no items but a terminal with output=None.
        # The consumer assembles an empty list. We then mock the consumer's
        # return to simulate a third-party path that returns final.output=None.
        empty_events = [
            SimpleNamespace(type="response.completed", response=SimpleNamespace(
                status="completed", id="r", output=None, usage=None,
            )),
        ]

        class _Stream:
            def __iter__(self): return iter(empty_events)
            def close(self): pass

        # Monkey-patch the consumer to return a final whose .output is None
        # (mimics third-party shim behavior the defensive guard protects against).
        from agent import codex_runtime
        original_consume = codex_runtime._consume_codex_event_stream

        def _consume_returning_none_output(*args, **kwargs):
            return SimpleNamespace(
                output=None,  # the defensive guard target
                output_text="",
                usage=None,
                status="completed",
                id="r",
                model=kwargs.get("model"),
                incomplete_details=None,
                error=None,
            )

        codex_runtime._consume_codex_event_stream = _consume_returning_none_output
        try:
            class FakeResponses:
                def create(self, **kwargs):
                    return _Stream()

            fake_client = SimpleNamespace(responses=FakeResponses())
            adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.5")

            # Should not raise TypeError: 'NoneType' object is not iterable
            response = adapter.create(messages=[{"role": "user", "content": "x"}])
            assert response.choices[0].message.content is None
            assert response.choices[0].finish_reason == "stop"
        finally:
            codex_runtime._consume_codex_event_stream = original_consume


class TestCodexAuxiliaryAdapterCompletedResponse:
    def test_accepts_completed_response_when_stream_was_requested(self):
        completed = SimpleNamespace(
            status="completed",
            id="resp_completed",
            output=[SimpleNamespace(
                type="message",
                content=[SimpleNamespace(
                    type="output_text",
                    text="completed response",
                )],
            )],
            usage=SimpleNamespace(
                input_tokens=11,
                output_tokens=3,
                total_tokens=14,
            ),
        )

        class FakeResponses:
            def create(self, **kwargs):
                assert kwargs["stream"] is True
                return completed

        fake_client = SimpleNamespace(responses=FakeResponses())
        adapter = _CodexCompletionsAdapter(fake_client, "gpt-5.6-terra")

        response = adapter.create(
            messages=[{"role": "user", "content": "review this"}],
        )

        assert response.choices[0].message.content == "completed response"
        assert response.usage.prompt_tokens == 11
        assert response.usage.completion_tokens == 3
        assert response.usage.total_tokens == 14


class TestMoaAggregatorStreamingBypass:
    def test_moa_aggregator_stream_bypasses_relay_for_codex_auxiliary_client(self, monkeypatch):
        """The MoA facade owns the streaming contract. For Codex Responses-shim
        clients (openai-codex, xai-oauth), call_llm must return the provider's
        direct create() result instead of routing through Relay's managed
        stream, which cannot iterate a completed SimpleNamespace (#74903).
        """

        completed = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))]
        )

        real_client = SimpleNamespace(
            api_key="test-key",
            base_url="https://chatgpt.com/backend-api/codex/",
            close=lambda: None,
        )
        client = CodexAuxiliaryClient(real_client, "gpt-5.6-sol")
        direct_create = MagicMock(return_value=completed)
        monkeypatch.setattr(client.chat.completions, "create", direct_create)

        monkeypatch.setattr(
            "agent.auxiliary_client._get_cached_client",
            lambda *args, **kwargs: (client, "gpt-5.6-sol"),
        )
        relay_stream = MagicMock(side_effect=AssertionError("_relay_sync_stream must not be used"))
        monkeypatch.setattr("agent.auxiliary_client._relay_sync_stream", relay_stream)

        result = call_llm(
            task="moa_aggregator",
            provider="openai-codex",
            model="gpt-5.6-sol",
            messages=[{"role": "user", "content": "只回答 OK"}],
            stream=True,
        )

        assert result is completed
        direct_create.assert_called_once()
        relay_stream.assert_not_called()
