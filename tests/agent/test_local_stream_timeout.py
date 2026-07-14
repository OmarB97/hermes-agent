"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
import pytest
from types import SimpleNamespace
from unittest.mock import patch

from agent.model_metadata import is_local_endpoint
from agent.chat_completion_helpers import (
    _dflash_local_first_chunk_timeout,
    _dflash_local_stale_timeout,
    _is_dflash_like_model,
    interruptible_streaming_api_call,
    _dflash_context_timeout_default,
    _gate_status_url,
    _local_backend_generation_active,
    _is_managed_local_w2_route,
    resolve_dflash_local_first_chunk_timeout,
    resolve_stream_stale_timeout,
)


class TestLocalStreamReadTimeout:
    """Verify stream read timeout auto-detection logic."""

    @pytest.mark.parametrize("base_url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://host.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_local_endpoint_bumps_read_timeout(self, base_url):
        """Local endpoint + default timeout -> bumps to base_timeout."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 1800.0

    def test_user_override_respected_for_local(self):
        """User sets HERMES_STREAM_READ_TIMEOUT -> keep their value even for local."""
        with patch.dict(os.environ, {"HERMES_STREAM_READ_TIMEOUT": "300"}, clear=False):
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = "http://localhost:11434"
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 300.0

    @pytest.mark.parametrize("base_url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
    ])
    def test_remote_endpoint_keeps_default(self, base_url):
        """Remote endpoint -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0

    def test_empty_base_url_keeps_default(self):
        """No base_url set -> keep 120s default."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HERMES_STREAM_READ_TIMEOUT", None)
            _base_timeout = float(os.getenv("HERMES_API_TIMEOUT", 1800.0))
            _stream_read_timeout = float(os.getenv("HERMES_STREAM_READ_TIMEOUT", 120.0))
            base_url = ""
            if _stream_read_timeout == 120.0 and base_url and is_local_endpoint(base_url):
                _stream_read_timeout = _base_timeout
            assert _stream_read_timeout == 120.0


class TestLocalDflashStaleTimeout:
    """dflash is local, but must not be allowed to wait forever with no chunks."""

    @staticmethod
    def _payload_for_estimated_tokens(tokens: int) -> dict[str, list[str]]:
        return {"messages": ["x" * (tokens * 4)]}

    def _make_agent(
        self,
        *,
        model="dflash",
        base_url="http://10.10.20.211:8080/v1",
        provider="taro",
    ):
        from run_agent import AIAgent

        with patch("agent.context_compressor.get_model_context_length", return_value=256_000):
            return AIAgent(
                api_key="sk-dummy",
                base_url=base_url,
                provider=provider,
                model=model,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )

    @pytest.mark.parametrize(
        "model",
        [
            "dflash",
            "deepseek-v4-flash-w2",
            "deepseek-v4-flash-iq3xxs",
        ],
    )
    def test_default_local_dflash_stream_stale_timeout_is_bounded(
        self,
        monkeypatch,
        tmp_path,
        model,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model=model)

        timeout = resolve_stream_stale_timeout(
            agent,
            {"model": model, "messages": [{"role": "user", "content": "hi"}]},
        )

        assert timeout == 180.0

    @pytest.mark.parametrize(
        "model",
        [
            "dflash",
            "taro/dflash-w2",
            "deepseek-v4-flash-w2",
            "deepseek-v4-flash-iq3xxs",
            "custom/deepseek_v4_flash:iq3xxs",
        ],
    )
    def test_dflash_family_recognizes_aliases_and_production_ids(self, model):
        assert _is_dflash_like_model(model) is True

    @pytest.mark.parametrize(
        "model",
        [
            None,
            "",
            "qwen3.6-27b",
            "deepseek-v4",
            "deepseek-v3-flash-w2",
            "notdeepseek-v4-flash-w2",
            "dflashish",
        ],
    )
    def test_dflash_family_rejects_unrelated_model_ids(self, model):
        assert _is_dflash_like_model(model) is False

    @pytest.mark.parametrize(
        ("estimated_tokens", "min_expected_timeout"),
        [
            # The old ladder returned exactly (180, 240, 240, 300, 300) here. It
            # was a step function whose numbers matched no measurement, and it
            # killed healthy requests: a 90.7k-token turn needs ~370s cold on the
            # measured W2 curve but was allowed 240s. The budget is now
            # continuous, so assert the FLOOR (it must never shrink) rather than
            # pinning a magic constant that would just re-freeze the old bug.
            (50_000, 180.0),
            (50_001, 240.0),
            (100_000, 240.0),
            (100_001, 300.0),
            (120_000, 300.0),
        ],
    )
    def test_default_dflash_first_chunk_timeout_scales_with_context(
        self,
        monkeypatch,
        estimated_tokens,
        min_expected_timeout,
    ):
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_COLD_START_ALLOWANCE", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", raising=False)

        timeout = _dflash_local_first_chunk_timeout(
            self._payload_for_estimated_tokens(estimated_tokens),
            "dflash",
        )

        assert timeout >= min_expected_timeout

    @pytest.mark.parametrize(
        "model",
        ["deepseek-v4-flash-w2", "deepseek-v4-flash-iq3xxs"],
    )
    def test_production_dflash_first_chunk_default_allows_observed_cold_start(
        self,
        monkeypatch,
        model,
    ):
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        timeout = _dflash_local_first_chunk_timeout({"messages": []}, model)

        # This used to assert exactly 180.0. That floor was itself the bug this
        # test was trying to guard against: a llama-swap model load on taro takes
        # a MEASURED 138.5s, so a cold W2 turn with even a modest prompt blew the
        # 180s budget and was killed while perfectly healthy. The floor now
        # carries an explicit cold-start allowance on top of the base watchdog.
        assert timeout > 138.5, "budget must clear the measured W2 cold start"
        assert timeout >= 300.0

    def test_managed_local_w2_timeout_floor_is_route_and_model_specific(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        payload = {
            "model": "deepseek-v4-flash-w2",
            "messages": [{"role": "user", "content": "hello"}],
        }
        taro_w2 = self._make_agent(model="deepseek-v4-flash-w2")
        meshboard_w2 = self._make_agent(
            model="deepseek-v4-flash-w2",
            base_url="http://127.0.0.1:8080/v1",
            provider="meshboard-qualified-local",
        )
        other_route = self._make_agent(
            model="deepseek-v4-flash-w2",
            provider="other-lan",
        )
        remote_lookalike = self._make_agent(
            model="deepseek-v4-flash-w2",
            base_url="https://models.example.com/v1",
            provider="meshboard-qualified-local",
        )
        managed_iq3 = self._make_agent(
            model="deepseek-v4-flash-iq3xxs",
            provider="meshboard-qualified-local",
        )

        assert resolve_stream_stale_timeout(taro_w2, payload) == 180.0
        assert resolve_dflash_local_first_chunk_timeout(taro_w2, payload) == 360.0
        assert resolve_stream_stale_timeout(meshboard_w2, payload) == 180.0
        assert resolve_dflash_local_first_chunk_timeout(meshboard_w2, payload) == 360.0
        assert resolve_stream_stale_timeout(other_route, payload) == 180.0
        assert resolve_dflash_local_first_chunk_timeout(other_route, payload) == 180.0
        assert resolve_stream_stale_timeout(remote_lookalike, payload) == 180.0
        assert resolve_dflash_local_first_chunk_timeout(remote_lookalike, payload) is None
        assert resolve_dflash_local_first_chunk_timeout(
            managed_iq3,
            {**payload, "model": "deepseek-v4-flash-iq3xxs"},
        ) == 180.0

    def test_production_w2_no_first_chunk_wait_exits_at_family_deadline(
        self,
        monkeypatch,
    ):
        """Exercise the poll-loop guard without sleeping for the real deadline."""
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        class Clock:
            now = 0.0

            @classmethod
            def time(cls):
                return cls.now

        class NoResponseThread:
            joins = 0

            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon

            def start(self):
                return None

            def is_alive(self):
                return True

            def join(self, timeout=None):
                if timeout == 0.3:
                    type(self).joins += 1
                    Clock.now = 181.0 if self.joins == 1 else 361.0

        statuses = []
        replacements = []

        class Agent:
            api_mode = "chat_completions"
            base_url = "http://10.10.20.211:8080/v1"
            provider = "taro"
            model = "deepseek-v4-flash-w2"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_message):
                return None

            @staticmethod
            def _buffer_status(message):
                statuses.append(message)

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                replacements.append(reason)

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread",
            NoResponseThread,
        )
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active",
            lambda base_url, model: None,
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)

        with pytest.raises(
            TimeoutError,
            match=r"Local dflash stream produced no first chunk after 361s .*360s",
        ):
            interruptible_streaming_api_call(
                Agent(),
                {
                    "model": "deepseek-v4-flash-w2",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert len(statuses) == 1
        assert "No first stream chunk from local dflash for 361s" in statuses[0]
        assert statuses[0].endswith("Reconnecting...")
        assert replacements == ["dflash_first_chunk_pool_cleanup"]

    def test_first_chunk_timeout_cancels_the_worker_instead_of_orphaning_it(
        self,
        monkeypatch,
    ):
        """A first-chunk timeout must CANCEL the streaming worker, not orphan it.

        The poll loop force-closes the connection, then breaks with a
        TimeoutError. The worker's exception handler retries on transport
        errors *unless* ``_request_cancelled`` is set — it cannot otherwise
        distinguish our deliberate close from a network blip.

        So if the flag is not set before the close, the worker retries in the
        background against a turn the main loop has already abandoned. When a
        slow local model (deepseek-v4-flash-w2) finally answers minutes later,
        that orphan writes into the finalised turn: the desktop streams text
        into an idle composer that offers no way to interrupt it.

        Assert the flag is set on the worker's own closure.
        """
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_TTFB_TIMEOUT", raising=False)

        class Clock:
            now = 0.0

            @classmethod
            def time(cls):
                return cls.now

        spawned = []

        class NoResponseThread:
            joins = 0

            def __init__(self, *, target, daemon):
                self.target = target
                self.daemon = daemon
                spawned.append(self)

            def start(self):
                return None

            def is_alive(self):
                return True

            def join(self, timeout=None):
                if timeout == 0.3:
                    type(self).joins += 1
                    Clock.now = 181.0 if self.joins == 1 else 361.0

        class Agent:
            api_mode = "chat_completions"
            base_url = "http://10.10.20.211:8080/v1"
            provider = "taro"
            model = "deepseek-v4-flash-w2"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_message):
                return None

            @staticmethod
            def _buffer_status(_message):
                return None

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                return None

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread",
            NoResponseThread,
        )
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active",
            lambda base_url, model: None,
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)

        with pytest.raises(TimeoutError, match=r"no first chunk after 361s"):
            interruptible_streaming_api_call(
                Agent(),
                {
                    "model": "deepseek-v4-flash-w2",
                    "messages": [{"role": "user", "content": "hello"}],
                },
            )

        assert len(spawned) == 1, "expected exactly one streaming worker"
        worker = spawned[0].target
        freevars = worker.__code__.co_freevars
        assert "_request_cancelled" in freevars, (
            "the streaming worker no longer closes over _request_cancelled; "
            "this test must be updated to track the new cancellation channel"
        )
        cancelled = worker.__closure__[
            freevars.index("_request_cancelled")
        ].cell_contents
        assert cancelled["value"] is True, (
            "first-chunk timeout abandoned the turn without cancelling the "
            "worker: it will retry in the background and stream a late "
            "response into an already-finalised turn"
        )

    def test_budget_is_continuous_with_no_cliffs(self, monkeypatch):
        """The old step ladder jumped 240s -> 300s across a single token."""
        for var in (
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_COLD_START_ALLOWANCE",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

        def budget(tokens):
            return _dflash_context_timeout_default(
                {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}
            )

        # Straddle the old 100k cliff: the step must be small, not 60 seconds.
        below, above = budget(99_000), budget(101_000)
        assert above > below
        assert above - below < 30.0, "budget still has a cliff at ~100k tokens"

        # Monotonic non-decreasing across the whole range.
        sizes = [1_000, 10_000, 25_000, 50_000, 75_000, 90_700, 110_000, 131_072]
        budgets = [budget(n) for n in sizes]
        assert budgets == sorted(budgets), f"budget is not monotonic: {budgets}"

    def test_budget_clears_a_cold_model_load_at_every_context(self, monkeypatch):
        """A llama-swap eviction must never kill the next healthy turn.

        llama-swap unloads W2 whenever another model is requested on the same
        GPU, so the very next W2 turn pays a full model load AND re-prefills the
        whole context (the eviction wipes the KV cache too). The measured cold
        load on taro is 138.5s for a 4-token prompt — before any prefill.

        The old ladder's floor was 180s, which left ~40s for prefill after a cold
        load. That is why a healthy 90.7k turn died at its 240s bucket.

        Deliberately does NOT pin a prefill-rate curve: measured TTFT spans an
        order of magnitude depending on prefix-cache state (18.8s vs 62.6s for
        the same 4k prompt, cached vs unique), so any fitted constant here would
        be fiction. The real guarantee is the liveness probe; this only asserts
        the fallback budget is not absurd.
        """
        for var in (
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_COLD_START_ALLOWANCE",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

        measured_cold_load_s = 138.5

        for tokens in (0, 10_000, 50_000, 90_700, 131_072):
            budget = _dflash_context_timeout_default(
                {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}
            )
            assert budget > measured_cold_load_s, (
                f"budget {budget:.0f}s at {tokens:,} tokens does not even clear a "
                f"cold model load ({measured_cold_load_s}s) — the next turn after "
                "any llama-swap eviction would be killed while healthy"
            )

        # And the specific request that actually died: 90.7k tokens got 240s.
        budget_90k = _dflash_context_timeout_default(
            {"messages": [{"role": "user", "content": "x" * (90_700 * 4)}]}
        )
        assert budget_90k > 240.0 * 2, (
            f"90.7k-token budget is {budget_90k:.0f}s; the 240s it used to get was "
            "already too tight to survive a cold load plus prefill"
        )

    def test_budget_never_narrower_than_the_old_ladder(self, monkeypatch):
        for var in (
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_COLD_START_ALLOWANCE",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

        def old_ladder(n):
            if n > 100_000:
                return 300.0
            if n > 50_000:
                return 240.0
            return 180.0

        for tokens in (1_000, 10_000, 50_001, 90_700, 100_001, 131_072):
            budget = _dflash_context_timeout_default(
                {"messages": [{"role": "user", "content": "x" * (tokens * 4)}]}
            )
            assert budget >= old_ladder(tokens), (
                f"budget shrank at {tokens:,} tokens: {budget} < {old_ladder(tokens)}"
            )

    def test_budget_is_bounded_by_the_ceiling(self, monkeypatch):
        """A wedged backend must still be bounded — this is a watchdog."""
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_COLD_START_ALLOWANCE", raising=False)
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", "600")

        budget = _dflash_context_timeout_default(
            {"messages": [{"role": "user", "content": "x" * (131_072 * 4)}]}
        )
        assert budget == 600.0


    def test_dflash_first_chunk_timeout_has_independent_env(self, monkeypatch):
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "45")

        assert _dflash_local_first_chunk_timeout({"messages": []}, "dflash") == 45.0

    def test_configured_model_stale_timeout_is_canonical_for_both_stream_phases(
        self,
        monkeypatch,
        tmp_path,
    ):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            """providers:
  taro:
    stale_timeout_seconds: 320
    models:
      deepseek-v4-flash-w2:
        stale_timeout_seconds: 360
""",
            encoding="utf-8",
        )
        # Config is canonical even when every legacy environment surface is
        # present. The payload model must also win over the agent default.
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "12")
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "45")
        monkeypatch.setenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", "50")
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "55")
        monkeypatch.setenv("HERMES_DFLASH_TTFB_TIMEOUT", "60")

        agent = self._make_agent(model="deepseek-v4-flash-iq3xxs")
        payload = {
            "model": "deepseek-v4-flash-w2",
            **self._payload_for_estimated_tokens(120_000),
        }
        stream_timeout = resolve_stream_stale_timeout(agent, payload)
        first_chunk_timeout = resolve_dflash_local_first_chunk_timeout(
            agent,
            payload,
            resolved_stream_stale_timeout=stream_timeout,
        )

        assert stream_timeout == 360.0
        assert first_chunk_timeout == 360.0

        provider_payload = {
            "model": "deepseek-v4-flash-iq3xxs",
            **self._payload_for_estimated_tokens(120_000),
        }
        provider_stream_timeout = resolve_stream_stale_timeout(
            agent,
            provider_payload,
        )
        provider_first_chunk_timeout = resolve_dflash_local_first_chunk_timeout(
            agent,
            provider_payload,
            resolved_stream_stale_timeout=provider_stream_timeout,
        )

        assert provider_stream_timeout == 320.0
        assert provider_first_chunk_timeout == 320.0

    def test_generic_local_stream_stale_timeout_still_disables_by_default(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model="qwen3.6-27b")

        timeout = resolve_stream_stale_timeout(
            agent,
            {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert timeout == float("inf")

    def test_default_local_dflash_non_stream_stale_timeout_is_bounded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        agent = self._make_agent(model="deepseek-v4-flash-w2")

        assert agent._compute_non_stream_stale_timeout({"messages": []}) == 180.0

    def test_explicit_stream_stale_timeout_still_wins_for_dflash(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "12")
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "75")

        agent = self._make_agent(model="deepseek-v4-flash-w2")

        assert resolve_stream_stale_timeout(
            agent,
            {"model": "deepseek-v4-flash-w2", "messages": []},
        ) == 12.0

    @pytest.mark.parametrize(
        ("estimated_tokens", "expected_timeout"),
        [
            (10_000, 180.0),
            (10_001, 180.0),
            (25_000, 180.0),
            (25_001, 180.0),
            (50_000, 180.0),
            (50_001, 240.0),
            (100_000, 240.0),
            (100_001, 300.0),
        ],
    )
    def test_default_dflash_stale_timeout_threshold_boundaries(
        self,
        monkeypatch,
        estimated_tokens,
        expected_timeout,
    ):
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)

        timeout = _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(estimated_tokens),
            "deepseek-v4-flash-iq3xxs",
        )

        assert timeout == expected_timeout

    def test_non_positive_dflash_stale_timeout_disables_watchdog(self, monkeypatch):
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "0")

        timeout = _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(1),
            "deepseek-v4-flash-w2",
        )

        assert timeout == float("inf")


class TestIsLocalEndpoint:
    """Direct unit tests for is_local_endpoint."""

    @pytest.mark.parametrize("url", [
        "http://localhost:11434",
        "http://127.0.0.1:8080",
        "http://0.0.0.0:5000",
        "http://[::1]:11434",
        "http://192.168.1.100:8000",
        "http://10.0.0.5:1234",
        "http://172.17.0.1:11434",
    ])
    def test_classic_local_addresses(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://host.docker.internal:11434",
        "http://host.docker.internal:8080/v1",
        "http://gateway.docker.internal:11434",
        "http://host.containers.internal:11434",
        "http://host.lima.internal:11434",
    ])
    def test_container_dns_names(self, url):
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://ollama:11434",
        "http://litellm:4000/v1",
        "http://hermes-litellm:8080",
        "http://vllm:8000",
    ])
    def test_unqualified_docker_hostnames(self, url):
        """Unqualified hostnames (no dots) are local — Docker Compose, /etc/hosts, etc."""
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
        "https://evil.docker.internal.example.com",
    ])
    def test_remote_endpoints(self, url):
        assert is_local_endpoint(url) is False

    @pytest.mark.parametrize("url", [
        "http://100.64.0.0:11434",            # lower bound of CGNAT block
        "http://100.64.0.1:11434/v1",         # lower bound +1
        "http://100.77.243.5:11434",          # representative Tailscale host
        "https://100.100.100.100:443",        # Tailscale MagicDNS anchor
        "https://100.127.255.254:443",        # upper bound -1
        "http://100.127.255.255:11434",       # upper bound of CGNAT block
    ])
    def test_tailscale_cgnat_is_local(self, url):
        """Tailscale 100.64.0.0/10 should be treated as local for timeout bumps."""
        assert is_local_endpoint(url) is True

    @pytest.mark.parametrize("url", [
        "http://100.63.255.255:11434",        # just below CGNAT block
        "http://100.128.0.1:11434",           # just above CGNAT block
        "http://100.200.0.1:11434",           # well outside CGNAT
        "http://99.64.0.1:11434",             # first octet wrong
    ])
    def test_near_but_not_cgnat_is_remote(self, url):
        """Hosts adjacent to but outside 100.64.0.0/10 must not match."""
        assert is_local_endpoint(url) is False


class TestLocalBackendProgressSignal:
    """The watchdog must ask the backend, not just count seconds."""

    def test_active_generation_for_our_model_reports_busy(self, monkeypatch):
        payload = {
            "active_generation_count": 1,
            "active_generations": [
                {"id": "g1", "model": "deepseek-v4-flash-w2", "age_s": 210.0}
            ],
        }
        monkeypatch.setattr(
            "httpx.get",
            lambda url, timeout=None: SimpleNamespace(
                status_code=200, json=lambda: payload
            ),
        )
        assert _local_backend_generation_active(
            "http://10.10.20.211:8080/v1", "deepseek-v4-flash-w2"
        ) is True

    def test_generation_for_another_model_still_reports_busy(self, monkeypatch):
        """Single-GPU llama-swap: another model generating means OURS was evicted
        and is swapping back in. That is exactly when to be patient."""
        payload = {
            "active_generation_count": 1,
            "active_generations": [{"id": "g1", "model": "deepseek-v4-flash-iq3xxs"}],
        }
        monkeypatch.setattr(
            "httpx.get",
            lambda url, timeout=None: SimpleNamespace(
                status_code=200, json=lambda: payload
            ),
        )
        assert _local_backend_generation_active(
            "http://10.10.20.211:8080/v1", "deepseek-v4-flash-w2"
        ) is True

    def test_idle_backend_reports_not_busy(self, monkeypatch):
        payload = {"active_generation_count": 0, "active_generations": []}
        monkeypatch.setattr(
            "httpx.get",
            lambda url, timeout=None: SimpleNamespace(
                status_code=200, json=lambda: payload
            ),
        )
        assert _local_backend_generation_active(
            "http://10.10.20.211:8080/v1", "deepseek-v4-flash-w2"
        ) is False

    def test_no_gate_yields_no_signal_not_a_false_negative(self, monkeypatch):
        """A provider without ai-gate must be UNAFFECTED: None, never False.

        Returning False here would make the watchdog kill remote-API requests
        the moment their budget expired, with no chance to extend.
        """
        def boom(url, timeout=None):
            raise ConnectionError("no gate here")

        monkeypatch.setattr("httpx.get", boom)
        assert _local_backend_generation_active(
            "https://api.example.com/v1", "gpt-whatever"
        ) is None

    def test_non_200_and_garbage_bodies_yield_no_signal(self, monkeypatch):
        monkeypatch.setattr(
            "httpx.get",
            lambda url, timeout=None: SimpleNamespace(status_code=404, json=lambda: {}),
        )
        assert _local_backend_generation_active(
            "http://10.10.20.211:8080/v1", "m"
        ) is None

        monkeypatch.setattr(
            "httpx.get",
            lambda url, timeout=None: SimpleNamespace(
                status_code=200, json=lambda: {"unexpected": "shape"}
            ),
        )
        assert _local_backend_generation_active(
            "http://10.10.20.211:8080/v1", "m"
        ) is None

    def test_gate_status_url_derivation(self):
        assert (
            _gate_status_url("http://10.10.20.211:8080/v1")
            == "http://10.10.20.211:8080/_gate/status"
        )
        assert _gate_status_url("") is None
        assert _gate_status_url("not-a-url") is None


class TestProgressAwareWatchdog:
    """The watchdog must verify liveness before killing a slow local request."""

    def _drive(self, monkeypatch, *, backend_active, ceiling="600"):
        """Run the poll loop with a backend that never emits a first chunk.

        The fake clock advances 60s per poll so the ceiling is reachable.
        Returns (raised_exception_or_None, buffered_statuses).
        """
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_STALE_TIMEOUT",
            "HERMES_DFLASH_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_FIRST_CHUNK_TIMEOUT",
            "HERMES_DFLASH_TTFB_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", ceiling)
        # Pin the budget well under the ceiling. Since the runtime resolver now
        # reaches the context-scaled budget, a 90.7k prompt would otherwise
        # resolve to ~723s -- ABOVE this test's 600s ceiling -- so the loop would
        # skip the liveness probe and kill, which is not what this test is about.
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "120")

        class Clock:
            now = 0.0

            @classmethod
            def time(cls):
                return cls.now

        class NeverAnswersThread:
            def __init__(self, *, target, daemon):
                self.target = target

            def start(self):
                return None

            def is_alive(self):
                return True

            def join(self, timeout=None):
                # Advance the wall clock so both the per-attempt budget and the
                # absolute ceiling can actually be reached.
                Clock.now += 60.0

        statuses = []

        class Agent:
            api_mode = "chat_completions"
            base_url = "http://10.10.20.211:8080/v1"
            provider = "taro"
            model = "deepseek-v4-flash-w2"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_m):
                return None

            @staticmethod
            def _buffer_status(m):
                statuses.append(m)

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                return None

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread", NeverAnswersThread
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active",
            lambda base_url, model: backend_active,
        )

        raised = None
        try:
            interruptible_streaming_api_call(
                Agent(),
                {
                    "model": "deepseek-v4-flash-w2",
                    "messages": [{"role": "user", "content": "x" * (90_700 * 4)}],
                },
            )
        except BaseException as exc:  # noqa: BLE001
            raised = exc
        return raised, statuses

    def test_busy_backend_extends_instead_of_killing(self, monkeypatch):
        """THE fix: a 90.7k-token request the server is actively prefilling must
        not be killed just because a client-side stopwatch expired."""
        raised, statuses = self._drive(monkeypatch, backend_active=True)

        waited = [s for s in statuses if "Still prefilling" in s]
        assert waited, (
            "watchdog killed a request the backend reported as ACTIVE — it never "
            "extended. This is the bug: a healthy 90.7k prefill dies at the budget."
        )
        # It still stops eventually: the ceiling bounds a backend that lies.
        assert isinstance(raised, TimeoutError)

    def test_idle_backend_is_still_killed_promptly(self, monkeypatch):
        """A genuinely wedged backend must NOT be given the extension."""
        raised, statuses = self._drive(monkeypatch, backend_active=False)

        assert isinstance(raised, TimeoutError)
        assert not [s for s in statuses if "Still prefilling" in s], (
            "extended a backend that reported NO active generation"
        )

    def test_no_progress_signal_preserves_legacy_stopwatch(self, monkeypatch):
        """Providers with no gate (remote APIs) must be unaffected: pure budget."""
        raised, statuses = self._drive(monkeypatch, backend_active=None)

        assert isinstance(raised, TimeoutError)
        assert not [s for s in statuses if "Still prefilling" in s]


class TestRuntimeFirstChunkBudgetIsReachable:
    """The context-scaled budget must actually be used by the RUNTIME resolver.

    `_dflash_context_timeout_default` models the real cost of a healthy request
    (cold-start allowance + per-1k prefill). But only the legacy
    `_dflash_local_first_chunk_timeout` helper ever called it — the runtime path,
    `resolve_dflash_local_first_chunk_timeout`, fell straight through to the
    stale ladder. So the cost model existed and was never reached: a 90k-token
    turn got 240s from a step function calibrated against nothing.
    """

    class _Agent:
        # The path the desktop actually uses: ko-nas fleet router -> taro -> W2.
        provider = "ai-router"
        base_url = "http://10.10.20.199:9081/v1"
        model = "deepseek-v4-flash-w2"
        api_mode = "chat_completions"

    @staticmethod
    def _kwargs(tokens: int) -> dict:
        return {
            "model": "deepseek-v4-flash-w2",
            "messages": [{"role": "user", "content": "x" * (tokens * 4)}],
        }

    @staticmethod
    def _clear(monkeypatch) -> None:
        for var in (
            "HERMES_DFLASH_FIRST_CHUNK_TIMEOUT",
            "HERMES_DFLASH_TTFB_TIMEOUT",
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_STALE_TIMEOUT",
            "HERMES_DFLASH_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_COLD_START_ALLOWANCE",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

    def test_ai_router_is_recognised_as_a_managed_w2_route(self):
        """The provider allowlist missed the one the desktop actually uses."""
        assert _is_managed_local_w2_route(self._Agent(), "deepseek-v4-flash-w2") is True

    def test_an_arbitrary_lan_provider_is_NOT_a_managed_w2_route(self):
        """The floor is route-specific on purpose.

        A provider that merely happens to serve a W2-named model over the LAN is
        not this lane and must keep the generic deadline. Widening the check to
        "any local endpoint" would silently grant it the 360s floor.
        """
        class Agent(self._Agent):
            provider = "other-lan"

        assert _is_managed_local_w2_route(Agent(), "deepseek-v4-flash-w2") is False

    def test_remote_endpoint_is_not_a_managed_w2_route(self):
        """A W2-named model behind a REMOTE endpoint must not get local floors."""
        class Agent(self._Agent):
            provider = "openrouter"
            base_url = "https://openrouter.ai/api/v1"

        assert _is_managed_local_w2_route(Agent(), "deepseek-v4-flash-w2") is False

    def test_non_w2_model_is_not_a_managed_w2_route(self):
        assert _is_managed_local_w2_route(self._Agent(), "qwen3.6-27b") is False

    def test_runtime_budget_scales_with_context_instead_of_the_ladder(
        self, monkeypatch
    ):
        """The regression: 90.7k used to resolve to exactly 240s."""
        self._clear(monkeypatch)

        budget_90k = resolve_dflash_local_first_chunk_timeout(
            self._Agent(), self._kwargs(90_700)
        )
        assert budget_90k > 240.0 * 2, (
            f"runtime budget at 90.7k is {budget_90k:.0f}s — the stale ladder's "
            "240s is still winning, so the context cost model is unreachable"
        )

        # Continuous, not a step function.
        budgets = [
            resolve_dflash_local_first_chunk_timeout(self._Agent(), self._kwargs(n))
            for n in (10_000, 50_000, 79_800, 90_700, 131_072)
        ]
        assert budgets == sorted(budgets), f"budget is not monotonic: {budgets}"

    def test_runtime_budget_never_shrinks_below_the_old_floors(self, monkeypatch):
        self._clear(monkeypatch)

        def old_ladder(n):
            if n > 100_000:
                return 300.0
            if n > 50_000:
                return 240.0
            return 180.0

        for tokens in (1_000, 50_001, 90_700, 100_001, 131_072):
            budget = resolve_dflash_local_first_chunk_timeout(
                self._Agent(), self._kwargs(tokens)
            )
            assert budget >= old_ladder(tokens), (
                f"budget shrank at {tokens:,} tokens: {budget} < {old_ladder(tokens)}"
            )

    def test_explicit_env_override_still_wins(self, monkeypatch):
        """An operator who pinned a number must still get exactly that number."""
        self._clear(monkeypatch)
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "45")

        assert (
            resolve_dflash_local_first_chunk_timeout(
                self._Agent(), self._kwargs(90_700)
            )
            == 45.0
        )
