"""Tests for ordered provider fallback chain (salvage of PR #1761).

Extends the single-fallback tests in test_fallback_model.py to cover
the new list-based ``fallback_providers`` config format and chain
advancement through multiple providers.
"""

from unittest.mock import MagicMock, patch

from agent.error_classifier import FailoverReason
from run_agent import AIAgent, _pool_may_recover_from_rate_limit


def _make_agent(
    fallback_model=None,
    fallback_chain_from_config=None,
    initial_fallback_entry=None,
):
    """Create a minimal AIAgent with optional fallback config."""
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
            fallback_model=fallback_model,
            fallback_chain_from_config=fallback_chain_from_config,
            initial_fallback_entry=initial_fallback_entry,
        )
        agent.client = MagicMock()
        return agent


def _mock_client(base_url="https://openrouter.ai/api/v1", api_key="fb-key"):
    mock = MagicMock()
    mock.base_url = base_url
    mock.api_key = api_key
    return mock


# ── Chain initialisation ──────────────────────────────────────────────────


class TestFallbackChainInit:
    def test_no_fallback(self):
        agent = _make_agent(fallback_model=None)
        assert agent._fallback_chain == []
        assert agent._fallback_index == 0
        assert agent._fallback_model is None



    def test_invalid_entries_filtered(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "", "model": "glm-4.7"},
            {"provider": "zai"},
            "not-a-dict",
        ]
        agent = _make_agent(fallback_model=fbs)
        assert len(agent._fallback_chain) == 1
        assert agent._fallback_chain[0]["provider"] == "openai"


    def test_invalid_dict_no_provider(self):
        agent = _make_agent(fallback_model={"model": "gpt-4o"})
        assert agent._fallback_chain == []


# ── Chain advancement ─────────────────────────────────────────────────────


class TestFallbackChainAdvancement:
    def test_exhausted_returns_false(self):
        agent = _make_agent(fallback_model=None)
        assert agent._try_activate_fallback() is False

    def test_advances_index(self):
        fbs = [
            {"provider": "openai", "model": "gpt-4o"},
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client",
                    return_value=(_mock_client(), "gpt-4o")):
            assert agent._try_activate_fallback() is True
            assert agent._fallback_index == 1
            assert agent.model == "gpt-4o"
            assert agent._fallback_activated is True



    def test_skips_unconfigured_provider_to_next(self):
        """If resolve_provider_client returns None, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                (None, None),                    # broken provider
                (_mock_client(), "gpt-4o"),       # fallback succeeds
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"
            assert agent._fallback_index == 2

    def test_skips_provider_that_raises_to_next(self):
        """If resolve_provider_client raises, skip to next in chain."""
        fbs = [
            {"provider": "broken", "model": "nope"},
            {"provider": "openai", "model": "gpt-4o"},
        ]
        agent = _make_agent(fallback_model=fbs)
        with patch("agent.auxiliary_client.resolve_provider_client") as mock_rpc:
            mock_rpc.side_effect = [
                RuntimeError("auth failed"),
                (_mock_client(), "gpt-4o"),
            ]
            assert agent._try_activate_fallback() is True
            assert agent.model == "gpt-4o"

    def test_resolves_key_env_for_fallback_provider(self):
        fbs = [
            {
                "provider": "custom",
                "model": "fallback-model",
                "base_url": "https://fallback.example/v1",
                "key_env": "MY_FALLBACK_KEY",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch.dict("os.environ", {"MY_FALLBACK_KEY": "env-secret"}, clear=False),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(
                        base_url="https://fallback.example/v1",
                        api_key="env-secret",
                    ),
                    "fallback-model",
                ),
            ) as mock_rpc,
        ):
            assert agent._try_activate_fallback() is True
            assert mock_rpc.call_args.kwargs["explicit_api_key"] == "env-secret"


    def test_nous_anthropic_fallback_uses_the_messages_wire(self):
        """Portal Claude fallbacks must not stay on chat_completions.

        ``resolve_provider_client`` still returns an OpenAI client for Nous;
        activation has to re-derive api_mode from the model and rebuild the
        Anthropic client — otherwise the turn POSTs /chat/completions.
        """
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [
            {
                "provider": "nous",
                "model": "anthropic/claude-opus-4.8",
            }
        ]
        agent = _make_agent(fallback_model=fbs)
        rebuilt = {"count": 0}

        def _fake_build(api_key, base_url, timeout=None, **kwargs):
            rebuilt["count"] += 1
            rebuilt["api_key"] = api_key
            rebuilt["base_url"] = base_url
            return MagicMock(name="anthropic-client")

        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "anthropic/claude-opus-4.8",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=_fake_build,
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "anthropic_messages"
        assert agent.provider == "nous"
        assert agent.model == "anthropic/claude-opus-4.8"
        assert agent.client is None
        assert rebuilt["count"] == 1
        assert rebuilt["api_key"] == "portal-jwt"
        assert rebuilt["base_url"] == portal
        assert agent._anthropic_client is not None

    def test_nous_non_anthropic_fallback_stays_on_chat_completions(self):
        portal = "https://inference-api.nousresearch.com/v1"
        fbs = [{"provider": "nous", "model": "hermes-4-405b"}]
        agent = _make_agent(fallback_model=fbs)
        with (
            patch(
                "agent.chat_completion_helpers._fallback_entry_unavailable_without_network",
                return_value=None,
            ),
            patch(
                "agent.auxiliary_client.resolve_provider_client",
                return_value=(
                    _mock_client(base_url=portal, api_key="portal-jwt"),
                    "hermes-4-405b",
                ),
            ),
            patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ),
            patch(
                "agent.anthropic_adapter.build_anthropic_client",
                side_effect=AssertionError("must not build Anthropic client"),
            ),
        ):
            assert agent._try_activate_fallback() is True

        assert agent.api_mode == "chat_completions"
        assert agent.client is not None


class TestExplicitFallbackPolicy:
    def test_cached_agent_refreshes_policy_and_chain_without_recreation(self):
        agent = _make_agent(
            fallback_model=[{"provider": "openrouter", "model": "remote"}],
            fallback_chain_from_config=True,
        )
        agent._fallback_terminal_status_emitted = True
        agent._pending_fallback_notice = "queued init decision"

        refreshed = [
            {
                "provider": "custom",
                "model": "local-b",
                "base_url": "http://127.0.0.1:9000/v1",
            },
            {
                "provider": "custom",
                "model": "local-a",
                "base_url": "http://127.0.0.1:8000/v1",
            },
        ]

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "local-only",
                "fallback_providers": refreshed,
            },
        ):
            assert agent._refresh_fallback_policy() == "local-only"

        assert agent._fallback_policy == "local-only"
        assert agent._fallback_chain == refreshed
        assert agent._fallback_model == refreshed[0]
        assert agent._fallback_index == 0
        assert agent._fallback_terminal_status_emitted is False
        assert agent._pending_fallback_notice == "queued init decision"

    def test_cached_agent_refresh_removes_deleted_routes(self):
        initial = [{"provider": "openrouter", "model": "deleted"}]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "any",
                "fallback_providers": initial,
            },
        ):
            agent = _make_agent(fallback_model=initial)

        assert agent._fallback_chain_from_config is True

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "any", "fallback_providers": []},
        ):
            agent._refresh_fallback_policy()

        assert agent._fallback_chain == []
        assert agent._fallback_model is None

    def test_programmatic_chain_survives_empty_default_config(self):
        supplied = [{"provider": "openrouter", "model": "library-backup"}]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "any", "fallback_providers": []},
        ):
            agent = _make_agent(fallback_model=supplied)
            agent._refresh_fallback_policy()

        assert agent._fallback_chain_from_config is False
        assert agent._fallback_chain == supplied

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "off", "fallback_providers": []},
        ):
            agent._refresh_fallback_policy()
        assert agent._fallback_chain == []

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={"fallback_policy": "any", "fallback_providers": []},
        ):
            agent._refresh_fallback_policy()
        assert agent._fallback_chain == supplied

    def test_programmatic_chain_is_not_replaced_by_unrelated_config_chain(self):
        supplied = [{"provider": "openrouter", "model": "library-backup"}]
        configured = [{"provider": "openrouter", "model": "profile-backup"}]
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "any",
                "fallback_providers": configured,
            },
        ):
            agent = _make_agent(
                fallback_model=supplied,
                fallback_chain_from_config=False,
            )
            agent._refresh_fallback_policy()

        assert agent._fallback_chain_from_config is False
        assert agent._fallback_chain == supplied

    def test_config_managed_chain_recovers_after_init_config_read_failure(self):
        supplied = [{"provider": "openrouter", "model": "initial-backup"}]
        recovered = [{"provider": "openrouter", "model": "recovered-backup"}]

        with patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=OSError("config temporarily unreadable"),
        ):
            agent = _make_agent(
                fallback_model=supplied,
                fallback_chain_from_config=True,
            )

        assert agent._fallback_policy == "off"
        assert agent._fallback_chain == []
        assert agent._fallback_chain_from_config is True

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "any",
                "fallback_providers": recovered,
            },
        ):
            agent._refresh_fallback_policy()

        assert agent._fallback_chain == recovered

    def test_initial_entry_without_captured_chain_infers_config_ownership(self):
        selected = {"provider": "openrouter", "model": "profile-backup"}
        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "any",
                "fallback_providers": [selected],
            },
        ):
            agent = _make_agent(
                fallback_model=None,
                initial_fallback_entry=selected,
            )
            agent._refresh_fallback_policy()

        assert agent._fallback_chain_from_config is True
        assert agent._fallback_chain == [selected]

    def test_initial_entry_recovers_config_ownership_after_read_failure(self):
        selected = {"provider": "openrouter", "model": "profile-backup"}
        with patch(
            "hermes_cli.config.load_config_readonly",
            side_effect=OSError("config temporarily unreadable"),
        ):
            agent = _make_agent(
                fallback_model=None,
                initial_fallback_entry=selected,
            )

        assert agent._fallback_chain_from_config is True
        assert agent._fallback_chain == []

        with patch(
            "hermes_cli.config.load_config_readonly",
            return_value={
                "fallback_policy": "any",
                "fallback_providers": [selected],
            },
        ):
            agent._refresh_fallback_policy()

        assert agent._fallback_chain == [selected]

    def test_off_never_resolves_or_switches_and_fails_loudly(self):
        agent = _make_agent(
            fallback_model=[{"provider": "openrouter", "model": "remote"}]
        )
        primary_provider = agent.provider
        agent._fallback_policy = "off"
        statuses = []
        agent.status_callback = lambda kind, text: statuses.append((kind, text))

        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback(FailoverReason.server_error) is False

        resolve.assert_not_called()
        assert agent.provider == primary_provider
        assert statuses and statuses[-1][0] == "fallback"
        assert "policy off" in statuses[-1][1]
        assert "No provider switch was attempted" in statuses[-1][1]

    def test_local_only_rejects_remote_metadata_before_resolution(self):
        agent = _make_agent(
            fallback_model=[{"provider": "anthropic", "model": "claude"}]
        )
        agent._fallback_policy = "local-only"
        statuses = []
        agent.status_callback = lambda kind, text: statuses.append((kind, text))

        with patch("agent.auxiliary_client.resolve_provider_client") as resolve:
            assert agent._try_activate_fallback(FailoverReason.timeout) is False

        resolve.assert_not_called()
        assert "no eligible local fallback remained" in statuses[-1][1]
        assert "Remote and unknown-endpoint fallbacks were not used" in statuses[-1][1]

    def test_local_only_switch_status_precedes_runtime_mutation(self):
        local = {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        agent = _make_agent(fallback_model=[local])
        agent._fallback_policy = "local-only"
        primary_identity = (agent.model, agent.provider)
        observed = []

        def status(kind, text):
            observed.append((kind, text, agent.model, agent.provider))

        agent.status_callback = status
        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(_mock_client(local["base_url"]), "local-model"),
        ):
            assert agent._try_activate_fallback(FailoverReason.server_error) is True

        decision = next(item for item in observed if item[0] == "fallback")
        assert decision[2:4] == primary_identity
        assert "reason: server error" in decision[1]
        assert "switching to local-model via custom" in decision[1]
        assert agent.model == "local-model"
        assert agent.provider == "custom"

    def test_local_only_rechecks_resolved_endpoint_before_switch(self):
        local = {
            "provider": "custom",
            "model": "local-model",
            "base_url": "http://127.0.0.1:8000/v1",
        }
        agent = _make_agent(fallback_model=[local])
        agent._fallback_policy = "local-only"
        remote_client = _mock_client("https://remote.example/v1")

        with patch(
            "agent.auxiliary_client.resolve_provider_client",
            return_value=(remote_client, "local-model"),
        ):
            assert agent._try_activate_fallback() is False

        remote_client.close.assert_called_once()
        assert agent.model != "local-model"

    def test_terminal_policy_status_is_emitted_once(self):
        agent = _make_agent(fallback_model=[])
        agent._fallback_policy = "any"
        statuses = []
        agent.status_callback = lambda kind, text: statuses.append((kind, text))

        assert agent._try_activate_fallback(FailoverReason.timeout) is False
        assert agent._try_activate_fallback(FailoverReason.timeout) is False

        fallback_statuses = [item for item in statuses if item[0] == "fallback"]
        assert len(fallback_statuses) == 1
        assert "policy any" in fallback_statuses[0][1]


# ── Pool-rotation vs fallback gating (#11314) ────────────────────────────


def _pool(n_entries: int, has_available: bool = True):
    """Make a minimal credential-pool stand-in for rotation-room checks."""
    pool = MagicMock()
    pool.entries.return_value = [MagicMock() for _ in range(n_entries)]
    pool.has_available.return_value = has_available
    return pool


class TestPoolRotationRoom:
    def test_none_pool_returns_false(self):
        assert _pool_may_recover_from_rate_limit(None) is False







# ── Skip-self dedup (#22548) ───────────────────────────────────────────────


class TestFallbackChainDedup:
    """A fallback chain entry that resolves to the current provider/model
    (or the same custom-provider base_url) must be skipped, not retried.
    Otherwise a misconfigured chain or two custom_providers entries pointing
    at the same shim loop the same failure. See issue #22548."""

    def test_skips_entry_matching_current_provider_and_model(self):
        """Chain has [same-as-current, real-fallback]; activate must skip
        the first and use the second."""
        fbs = [
            # First entry == current state. Should be skipped.
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
            # Second entry: real fallback.
            {"provider": "zai", "model": "glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        # Stub out resolve_provider_client so we can assert which entry was
        # actually used — return a MagicMock client tagged with the provider.
        called = []
        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(), model
        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch("hermes_cli.model_normalize.normalize_model_for_provider", side_effect=lambda m, p: m):
                ok = agent._try_activate_fallback()

        assert ok is True
        # The first entry was skipped — only the second reached resolve.
        assert called == [("zai", "glm-4.7")], (
            f"expected fallback to skip same-state entry, got call order: {called}"
        )


    def test_returns_false_when_only_self_matching_entries(self):
        """A chain with only self-matching entries exhausts to False."""
        fbs = [
            {"provider": "openrouter", "model": "z-ai/glm-4.7"},
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "openrouter"
        agent.model = "z-ai/glm-4.7"
        agent.base_url = "https://openrouter.ai/api/v1"

        with patch("agent.auxiliary_client.resolve_provider_client") as mock_resolve:
            ok = agent._try_activate_fallback()

        assert ok is False
        mock_resolve.assert_not_called()

    def test_allows_xai_api_fallback_from_xai_oauth_same_host_model(self):
        """xai-oauth and xai share api.x.ai but use different credentials.

        A spending-limit 403 on OAuth must still be able to fall over to the
        API-key provider even when both entries use the same model slug and
        base URL.  Blind base_url+model dedup incorrectly skipped that path.
        """
        fbs = [
            {
                "provider": "xai",
                "model": "grok-4.5",
                "base_url": "https://api.x.ai/v1",
            },
        ]
        agent = _make_agent(fallback_model=fbs)
        agent.provider = "xai-oauth"
        agent.model = "grok-4.5"
        agent.base_url = "https://api.x.ai/v1"

        called = []

        def _resolve(provider, model=None, raw_codex=False, **kwargs):
            called.append((provider, model))
            return _mock_client(base_url="https://api.x.ai/v1"), model

        with patch("agent.auxiliary_client.resolve_provider_client", side_effect=_resolve):
            with patch(
                "hermes_cli.model_normalize.normalize_model_for_provider",
                side_effect=lambda m, p: m,
            ):
                ok = agent._try_activate_fallback()

        assert ok is True
        assert called == [("xai", "grok-4.5")]
        assert agent.provider == "xai"
        assert agent.model == "grok-4.5"
