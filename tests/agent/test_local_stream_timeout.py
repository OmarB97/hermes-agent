"""Tests for local provider stream read timeout auto-detection.

When a local LLM provider is detected (Ollama, llama.cpp, vLLM, etc.),
the httpx stream read timeout should be automatically increased from the
default 60s to HERMES_API_TIMEOUT (1800s) to avoid premature connection
kills during long prefill phases.
"""

import os
import re
import textwrap

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


def _unwrap_thread_target(target):
    """Return the streaming worker itself, past any ContextVars wrapper.

    ``interruptible_streaming_api_call`` no longer hands the worker closure
    straight to ``threading.Thread``: it wraps it with
    ``_context_thread_target``, which returns ``lambda: context.run(callback)``
    so the worker inherits the caller's ContextVars.  The tests below reach
    into the worker's own closure cells (chunk trackers, cancellation flag),
    so peel the wrapper off first — the wrapper's freevars are exactly
    ``('callback', 'context')`` and the worker is the ``callback`` cell.

    Anything that is not that wrapper is returned untouched, so the assertions
    that follow still fail loudly if the worker stops closing over what they
    read.
    """
    for _ in range(8):  # bounded: guards against a self-referential cell
        free = target.__code__.co_freevars
        if set(free) != {"callback", "context"}:
            return target
        target = target.__closure__[free.index("callback")].cell_contents
    raise AssertionError(
        "could not unwrap the streaming worker from its thread-target "
        "wrappers; this harness must be updated"
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
        worker = _unwrap_thread_target(spawned[0].target)
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

    def test_generic_local_stream_stale_timeout_is_bounded_but_looser(
        self, monkeypatch, tmp_path
    ):
        """A non-DFlash local model is bounded too — just far more loosely.

        This test previously asserted ``inf`` under the name
        ``..._still_disables_by_default``. That was not the intended contract:
        ``agent.local_stream_stale_timeout`` (default 900) and
        ``HERMES_LOCAL_STREAM_STALE_TIMEOUT`` were shipped and documented as a
        finite ceiling replacing exactly that infinite disable, and #263 dropped
        the code that read them while leaving the default and the docs in place.
        So the old assertion pinned the regression, not the design.

        900s rather than this family's 180s because a generic local endpoint may
        be serving a much larger model than DFlash on unknown hardware; finite
        rather than ``inf`` because a wedged local server must eventually let
        reconnect/fallback run instead of parking the session forever.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_LOCAL_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_STALE_TIMEOUT",
            "HERMES_DFLASH_STREAM_STALE_TIMEOUT",
        ):
            monkeypatch.delenv(var, raising=False)

        agent = self._make_agent(model="qwen3.6-27b")

        timeout = resolve_stream_stale_timeout(
            agent,
            {"model": "qwen3.6-27b", "messages": [{"role": "user", "content": "hi"}]},
        )

        assert timeout == 900.0

        # The DFlash family keeps its own, tighter budget on the same endpoint.
        assert resolve_stream_stale_timeout(
            agent,
            {"model": "dflash", "messages": [{"role": "user", "content": "hi"}]},
        ) == 180.0

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
            # These numbers are 180s (the family base) + 4s per 1k prompt
            # tokens, rounded — NOT a re-freeze of a step function.
            #
            # They replace the old ladder (180/180/180/180/180/240/240/300 at
            # these same token counts), which this test used to pin exactly.
            # That ladder was the production defect: a local DFlash lane serving
            # ~95k tokens got its 240s bucket and had three healthy prefills
            # killed mid-flight (agent.log 2026-08-02: 240s @ 94,829 tokens,
            # 240s @ 88,614, 180s @ 32,854). Its 300s cap also meant a 200k
            # turn on a 262k-window model was budgeted the same as a 100k one.
            #
            # Why THIS number is right rather than a wider one: 4s/1k is the
            # measured uncached prefill rate for this family, so the budget now
            # tracks the one thing that actually scales with a bigger prompt.
            # The cold-start allowance is deliberately absent here — that is a
            # measurement of one specific llama-swap deployment and stays gated
            # to the managed W2 route (see
            # test_managed_local_w2_timeout_floor_is_route_and_model_specific).
            # So every value below is strictly wider than the ladder's, and
            # never wider than base + real prefill cost.
            (10_000, 220.0),
            (10_001, 220.0),
            (25_000, 280.0),
            (25_001, 280.0),
            (50_000, 380.0),
            (50_001, 380.0),
            (100_000, 580.0),
            (100_001, 580.0),
            # Above the old 300s cap the budget keeps growing instead of
            # flat-lining. This row is the cap's headstone.
            (200_000, 980.0),
        ],
    )
    def test_default_dflash_stale_timeout_scales_continuously_with_context(
        self,
        monkeypatch,
        estimated_tokens,
        expected_timeout,
    ):
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", raising=False)

        timeout = _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(estimated_tokens),
            "deepseek-v4-flash-iq3xxs",
        )

        assert timeout == expected_timeout

    def test_default_dflash_stale_timeout_has_no_cliffs(self, monkeypatch):
        """The old ladder jumped 60s across a single token at 50k and 100k."""
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)

        def budget(tokens):
            return _dflash_local_stale_timeout(
                self._payload_for_estimated_tokens(tokens),
                "deepseek-v4-flash-iq3xxs",
            )

        for cliff in (10_000, 25_000, 50_000, 100_000):
            assert budget(cliff + 1) - budget(cliff) < 1.0, (
                f"budget still has a cliff at {cliff:,} tokens"
            )

        sizes = [0, 1_000, 10_000, 50_000, 94_829, 131_072, 200_000]
        budgets = [budget(n) for n in sizes]
        assert budgets == sorted(budgets), f"budget is not monotonic: {budgets}"

    def test_dflash_stale_timeout_is_bounded_by_the_ceiling(self, monkeypatch):
        """Context scaling must not turn the watchdog off for a huge prompt."""
        monkeypatch.delenv("HERMES_DFLASH_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", "600")

        assert _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(1_000_000),
            "deepseek-v4-flash-iq3xxs",
        ) == 600.0

    def test_legacy_dflash_stale_env_is_widened_never_narrowed(self, monkeypatch):
        """A pinned legacy value is the FLOOR, matching the old ladder's max()."""
        monkeypatch.setenv("HERMES_DFLASH_STALE_TIMEOUT", "900")
        monkeypatch.delenv("HERMES_DFLASH_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_PREFILL_SECONDS_PER_1K", raising=False)
        monkeypatch.delenv("HERMES_DFLASH_FIRST_CHUNK_CEILING", raising=False)

        assert _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(0),
            "deepseek-v4-flash-iq3xxs",
        ) == 900.0
        assert _dflash_local_stale_timeout(
            self._payload_for_estimated_tokens(100_000),
            "deepseek-v4-flash-iq3xxs",
        ) == 1300.0

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
        "https://api.openai.com",
        "https://openrouter.ai/api",
        "https://api.anthropic.com",
        "https://evil.docker.internal.example.com",
    ])
    def test_remote_endpoints(self, url):
        assert is_local_endpoint(url) is False


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
        """A bare config-entry name still matches directly, without a lookup.

        This is the STUBBED shape, not the production one — see
        ``TestFirstChunkBudgetThroughRealProviderResolution`` for what
        ``agent.provider`` actually is at runtime. Stubbing it is precisely
        what hid the original bug, so treat a green here as covering the
        direct branch only.
        """
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


class TestFirstChunkBudgetThroughRealProviderResolution:
    """Resolve the provider the way runtime does, instead of stubbing it.

    ``TestRuntimeFirstChunkBudgetIsReachable`` above sets ``provider =
    "ai-router"`` straight onto a fake agent. Runtime never looks like that:
    ``resolve_runtime_provider(requested="ai-router")`` returns the bare
    billing class ``"custom"`` for EVERY user-declared endpoint, so
    ``agent.provider`` is ``"custom"`` in production. Stubbing the provider is
    what let the original bug hide — the allowlist held config-entry names and
    was compared against a value that is never one, so the managed-W2 branch
    (and the cold-start budget behind it) never ran in production on any lane,
    and a fix that "added ai-router to the allowlist" shipped as dead code.

    The contract these tests now pin: the allowlist is matched against the
    config-entry name RECOVERED from the live endpoint, so a real resolved
    agent on a declared managed lane reaches the floor — while an endpoint no
    allowlisted entry owns still does not. Building through the real resolver
    is what makes that gap unable to reopen; keep this shape.
    """

    @staticmethod
    def _agent(monkeypatch, tmp_path, *, requested, model, config_body):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            textwrap.dedent(config_body), encoding="utf-8"
        )
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_STALE_TIMEOUT",
            "HERMES_DFLASH_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_FIRST_CHUNK_TIMEOUT",
            "HERMES_DFLASH_TTFB_TIMEOUT",
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_COLD_START_ALLOWANCE",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

        from hermes_cli.runtime_provider import resolve_runtime_provider
        from run_agent import AIAgent

        resolved = resolve_runtime_provider(requested=requested)
        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=262_144,
        ):
            return AIAgent(
                api_key=resolved.get("api_key") or "sk-dummy",
                base_url=resolved["base_url"],
                provider=resolved["provider"],
                api_mode=resolved.get("api_mode"),
                model=model,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )

    @staticmethod
    def _payload(model: str, tokens: int) -> dict:
        return {
            "model": model,
            "messages": [{"role": "user", "content": "x" * (tokens * 4)}],
        }

    _AI_ROUTER_CONFIG = """\
    providers:
      ai-router:
        api: "http://10.10.20.199:9081/v1"
    """

    def test_bare_custom_billing_class_is_still_a_managed_w2_route(
        self, monkeypatch, tmp_path
    ):
        """Both halves of the fix, in one place.

        The premise the stubbed tests got wrong stays pinned as a fact:
        resolution reports the bare billing class, NOT the entry name. What
        changed is the conclusion drawn from it — the route check recovers the
        entry name from the endpoint, so the managed lane is recognised anyway.

        This assertion read ``is False`` before, documenting the bug: the 360s
        floor was unreachable on the one lane its allowlist was written for.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ai-router",
            model="deepseek-v4-flash-w2",
            config_body=self._AI_ROUTER_CONFIG,
        )

        assert agent.provider == "custom"
        assert _is_managed_local_w2_route(agent, "deepseek-v4-flash-w2") is True

    def test_real_resolution_reaches_the_managed_w2_floor(
        self, monkeypatch, tmp_path
    ):
        """A small prompt isolates the floor from the context scaling.

        180s base + ~0 prefill would be the generic DFlash answer, and was what
        this lane actually got in production. The managed branch floors it at
        360s. Asserting on a SMALL prompt is deliberate: at 90k tokens the
        context term dominates and would mask a floor that never applied.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ai-router",
            model="deepseek-v4-flash-w2",
            config_body=self._AI_ROUTER_CONFIG,
        )
        payload = self._payload("deepseek-v4-flash-w2", 8)

        assert resolve_dflash_local_first_chunk_timeout(agent, payload) == 360.0
        # The gap BETWEEN chunks is a different cost and must not move.
        assert resolve_stream_stale_timeout(agent, payload) == 180.0

    def test_real_resolution_gets_the_cold_start_inclusive_budget(
        self, monkeypatch, tmp_path
    ):
        """At scale the managed lane also gets the cold-start allowance.

        Re-derived: this asserted 543.0, the generic budget (180s base + 4s/1k
        * 90.7k), because the managed branch was unreachable. The managed
        budget adds the measured 180s llama-swap cold start on top — the very
        term that exists so the first W2 turn after an eviction is not killed
        while healthy.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ai-router",
            model="deepseek-v4-flash-w2",
            config_body=self._AI_ROUTER_CONFIG,
        )
        payload = self._payload("deepseek-v4-flash-w2", 90_700)

        generic = 543.0
        assert resolve_dflash_local_first_chunk_timeout(agent, payload) == 723.0
        assert 723.0 - generic == 180.0, "the delta IS the cold-start allowance"
        # Unchanged: the stale timeout never consulted the managed-W2 branch.
        assert resolve_stream_stale_timeout(agent, payload) == generic

    def test_custom_prefixed_runtime_id_also_matches(self, monkeypatch, tmp_path):
        """``custom:<name>`` carries the entry name and needs no lookup."""
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ai-router",
            model="deepseek-v4-flash-w2",
            config_body=self._AI_ROUTER_CONFIG,
        )
        agent.provider = "custom:ai-router"

        assert _is_managed_local_w2_route(agent, "deepseek-v4-flash-w2") is True
        assert (
            resolve_dflash_local_first_chunk_timeout(
                agent, self._payload("deepseek-v4-flash-w2", 8)
            )
            == 360.0
        )

    def test_configured_but_unlisted_provider_stays_off_the_managed_lane(
        self, monkeypatch, tmp_path
    ):
        """Recovery must not become "any declared endpoint serving W2".

        ``ko-3090`` is a real, configured, local W2-capable endpoint that is
        deliberately NOT on the managed lane. Recovering its name is the point;
        granting it the floor would be the bug the allowlist exists to prevent.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ko-3090",
            model="deepseek-v4-flash-w2",
            config_body="""\
            providers:
              ko-3090:
                api: "http://10.10.20.77:8080/v1"
            """,
        )

        assert agent.provider == "custom"
        assert _is_managed_local_w2_route(agent, "deepseek-v4-flash-w2") is False
        assert (
            resolve_dflash_local_first_chunk_timeout(
                agent, self._payload("deepseek-v4-flash-w2", 8)
            )
            == 180.0
        )

    def test_two_entries_sharing_an_endpoint_are_ambiguous(
        self, monkeypatch, tmp_path
    ):
        """Exactly-one-owner (#330): a contested endpoint has no owner.

        Rows with distinct credentials may legitimately share a base_url, and
        none of them can claim it alone — so recovery returns nothing and the
        route falls back to the generic deadline rather than guessing.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ai-router",
            model="deepseek-v4-flash-w2",
            config_body="""\
            providers:
              ai-router:
                api: "http://10.10.20.199:9081/v1"
              ai-router-standby:
                api: "http://10.10.20.199:9081/v1"
            """,
        )

        assert agent.provider == "custom"
        assert _is_managed_local_w2_route(agent, "deepseek-v4-flash-w2") is False

    def test_production_repro_ds4_lane_at_94829_tokens(self, monkeypatch, tmp_path):
        """The exact turn from agent.log 2026-08-02 09:43:11.

        ``Local dflash stream produced no first chunk for 240s (threshold 240s).
        model=deepseek-v4-flash-0731-ds4 context=~94,829 tokens. Killing
        connection.`` — a healthy prefill on a ~240 tok/s lane needs ~8 minutes
        at that size, so 240s killed it three times over. The budget must now
        comfortably clear a real 8-minute prefill.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ds4",
            model="deepseek-v4-flash-0731-ds4",
            config_body="""\
            providers:
              ds4:
                api: "http://10.10.20.211:8080/v1"
            """,
        )
        payload = self._payload("deepseek-v4-flash-0731-ds4", 94_829)

        budget = resolve_dflash_local_first_chunk_timeout(agent, payload)
        assert budget > 480.0, (
            f"first-chunk budget is {budget}s — an 8-minute healthy prefill at "
            "94,829 tokens would still be killed"
        )
        assert budget == 559.0

    def test_declared_first_chunk_timeout_wins_over_everything(
        self, monkeypatch, tmp_path
    ):
        """The dedicated knob outranks stale_timeout_seconds and the env hatch.

        Per-model beats per-provider, and both beat
        ``HERMES_DFLASH_FIRST_CHUNK_TIMEOUT``. The two phases stay independent:
        stale_timeout_seconds still governs the gap BETWEEN chunks.
        """
        agent = self._agent(
            monkeypatch,
            tmp_path,
            requested="ds4",
            model="deepseek-v4-flash-0731-ds4",
            config_body="""\
            providers:
              ds4:
                api: "http://10.10.20.211:8080/v1"
                stale_timeout_seconds: 300
                first_chunk_timeout_seconds: 1200
                models:
                  deepseek-v4-flash-0731-ds4:
                    first_chunk_timeout_seconds: 1500
            """,
        )
        monkeypatch.setenv("HERMES_DFLASH_FIRST_CHUNK_TIMEOUT", "45")
        payload = self._payload("deepseek-v4-flash-0731-ds4", 94_829)

        assert resolve_dflash_local_first_chunk_timeout(agent, payload) == 1500.0
        assert resolve_stream_stale_timeout(agent, payload) == 300.0

        other_model = self._payload("deepseek-v4-flash-iq3xxs", 94_829)
        assert (
            resolve_dflash_local_first_chunk_timeout(agent, other_model) == 1200.0
        )


class TestManagedW2RouteIdentityRecoveryBoundaries:
    """What the recovered-identity match must refuse, and what it must not cost.

    The route check is the gate on a hardcoded timing exception for one
    physical lane, so widening it is a real regression: every endpoint it
    wrongly admits gets a 360s floor and a 180s cold-start allowance measured
    on hardware that is not it. These tests fix the two boundaries — the
    refusals, and the per-turn cost of asking at all.
    """

    W2 = "deepseek-v4-flash-w2"

    @staticmethod
    def _home(monkeypatch, tmp_path, config_body: str) -> None:
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            textwrap.dedent(config_body), encoding="utf-8"
        )

    @staticmethod
    def _agent(provider: str, base_url: str):
        return SimpleNamespace(provider=provider, base_url=base_url)

    def test_no_base_url_refuses_the_config_provider_fallback(
        self, monkeypatch, tmp_path
    ):
        """Documented choice: endpoint identity only, never ``model.provider``.

        ``resolve_provider_config_key`` will fall back to
        ``config.model.provider`` when there is no endpoint to match on — right
        for recovering an operator's own credentials and timeouts, wrong here.
        This lane's floor is OUR hardcoded exception for specific hardware, and
        an agent with no base_url is an endpoint we cannot identify; handing it
        the exception on the strength of a config field would let any ad-hoc
        endpoint inherit it. The first assertion proves the fallback really
        would have fired, so this refusal stays deliberate rather than
        accidentally becoming unreachable.
        """
        from hermes_cli.timeouts import resolve_provider_config_key

        self._home(
            monkeypatch,
            tmp_path,
            """\
            model:
              provider: ai-router
            providers:
              ai-router:
                api: "http://10.10.20.199:9081/v1"
            """,
        )

        assert resolve_provider_config_key("custom", None) == "ai-router"
        assert _is_managed_local_w2_route(self._agent("custom", ""), self.W2) is False

    def test_non_w2_model_never_pays_for_identity_recovery(
        self, monkeypatch, tmp_path
    ):
        """Cost guard: recovery reads config, and this runs every API turn.

        The model test must short-circuit before the lookup, so the overhead
        lands only on the handful of turns that could possibly be on this lane.
        The second half asserts the patch targets a name that is actually
        called — otherwise the guard would pass by pointing at nothing.
        """
        import agent.chat_completion_helpers as helpers

        self._home(
            monkeypatch,
            tmp_path,
            """\
            providers:
              ai-router:
                api: "http://10.10.20.199:9081/v1"
            """,
        )
        calls = []

        def _record(provider_id, base_url=None, providers=None):
            calls.append(provider_id)
            return "ai-router"

        monkeypatch.setattr(helpers, "resolve_provider_config_key", _record)
        agent = self._agent("custom", "http://10.10.20.199:9081/v1")

        assert _is_managed_local_w2_route(agent, "qwen3.6-27b") is False
        assert _is_managed_local_w2_route(agent, "deepseek-v4-flash-iq3xxs") is False
        assert calls == [], f"config lookup ran for a non-W2 model: {calls}"

        assert _is_managed_local_w2_route(agent, self.W2) is True
        assert calls == ["custom"]

    def test_a_named_provider_never_pays_for_identity_recovery(
        self, monkeypatch, tmp_path
    ):
        """A non-``custom`` id already IS its config key — nothing to recover.

        Keeps the cost off built-in providers, and is why the remote-endpoint
        and arbitrary-LAN cases stay False without touching config at all.
        """
        self._home(monkeypatch, tmp_path, "providers: {}\n")

        def _fail(*args, **kwargs):
            raise AssertionError("recovery must not run for a named provider")

        monkeypatch.setattr(
            "hermes_cli.timeouts._recover_custom_provider_key", _fail
        )

        assert (
            _is_managed_local_w2_route(
                self._agent("other-lan", "http://10.10.20.211:8080/v1"), self.W2
            )
            is False
        )

    def test_broken_config_cannot_raise_into_the_request_path(
        self, monkeypatch, tmp_path
    ):
        """Recovery failure degrades to the generic deadline, never an error."""
        import agent.chat_completion_helpers as helpers

        self._home(monkeypatch, tmp_path, "providers: {}\n")

        def _boom(*args, **kwargs):
            raise RuntimeError("config is unreadable")

        monkeypatch.setattr(helpers, "resolve_provider_config_key", _boom)

        assert (
            _is_managed_local_w2_route(
                self._agent("custom", "http://10.10.20.199:9081/v1"), self.W2
            )
            is False
        )


class TestGenericLocalStreamStaleCeiling:
    """The generic (non-DFlash) local ceiling, end to end through config.

    ``agent.local_stream_stale_timeout`` and ``HERMES_LOCAL_STREAM_STALE_TIMEOUT``
    were defaulted in ``DEFAULT_CONFIG``, documented in
    ``website/docs/reference/environment-variables.md``, and read by NOTHING:
    #263 replaced the inline block that read them with
    ``resolve_stream_stale_timeout`` and never reinstated the ceiling. A knob
    that is defaulted and documented but inert is worse than no knob — it costs
    an investigation the time it takes to discover the bound it promises is not
    in force. These tests resolve through the real config loader so the reader
    cannot be dropped again without going red.
    """

    @staticmethod
    def _payload(model: str, tokens: int = 0) -> dict:
        # A list of raw strings estimates to EXACTLY ``tokens`` (chars // 4),
        # with none of the dict-repr overhead the message-dict form carries.
        return {"model": model, "messages": ["x" * (tokens * 4)]}

    @staticmethod
    def _agent(monkeypatch, tmp_path, *, config_body: str = "", model="qwen3.6-27b"):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        (tmp_path / "config.yaml").write_text(
            textwrap.dedent(config_body), encoding="utf-8"
        )
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_LOCAL_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
        ):
            monkeypatch.delenv(var, raising=False)

        from run_agent import AIAgent

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=256_000,
        ):
            return AIAgent(
                api_key="sk-dummy",
                base_url="http://localhost:11434/v1",
                provider="ollama",
                model=model,
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )

    def test_default_ceiling_comes_from_default_config(self, monkeypatch, tmp_path):
        """No user config at all -> the 900 in ``DEFAULT_CONFIG`` is in force."""
        agent = self._agent(monkeypatch, tmp_path)

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 900.0

    def test_config_yaml_value_is_honored(self, monkeypatch, tmp_path):
        agent = self._agent(
            monkeypatch,
            tmp_path,
            config_body="""\
            agent:
              local_stream_stale_timeout: 1200
            """,
        )

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 1200.0

    def test_env_var_overrides_config_yaml(self, monkeypatch, tmp_path):
        """The documented escape hatch outranks the canonical setting."""
        agent = self._agent(
            monkeypatch,
            tmp_path,
            config_body="""\
            agent:
              local_stream_stale_timeout: 1200
            """,
        )
        monkeypatch.setenv("HERMES_LOCAL_STREAM_STALE_TIMEOUT", "300")

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 300.0

    def test_unparseable_env_value_falls_through_to_config(
        self, monkeypatch, tmp_path
    ):
        """A typo is not an explicit override, and must not disable the bound."""
        agent = self._agent(
            monkeypatch,
            tmp_path,
            config_body="""\
            agent:
              local_stream_stale_timeout: 1200
            """,
        )
        monkeypatch.setenv("HERMES_LOCAL_STREAM_STALE_TIMEOUT", "not-a-number")

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 1200.0

    @pytest.mark.parametrize("disable_value", ["0", "-1"])
    def test_non_positive_env_value_restores_the_infinite_wait(
        self, monkeypatch, tmp_path, disable_value
    ):
        """The escape hatch for an exotically slow local model stays reachable.

        Bounding the generic local endpoint is a real behavior change, so the
        previous unbounded wait must remain available deliberately — via the
        same non-positive-disables convention every other watchdog here uses.
        """
        agent = self._agent(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_LOCAL_STREAM_STALE_TIMEOUT", disable_value)

        timeout = resolve_stream_stale_timeout(agent, self._payload(agent.model))

        assert timeout == float("inf")

    def test_zero_in_config_yaml_also_disables(self, monkeypatch, tmp_path):
        agent = self._agent(
            monkeypatch,
            tmp_path,
            config_body="""\
            agent:
              local_stream_stale_timeout: 0
            """,
        )

        assert resolve_stream_stale_timeout(
            agent, self._payload(agent.model)
        ) == float("inf")

    def test_ceiling_widens_with_prompt_size(self, monkeypatch, tmp_path):
        """A big prompt costs more to prefill, so it gets a bigger budget.

        900s + 4s per 1k prompt tokens — the same measured prefill rate the
        DFlash budget uses, and the same widen-only rule. A FLAT deadline is
        precisely how the old ladder killed healthy prefills mid-flight
        (#278, #334); repeating that mistake here would trade "hangs forever"
        for "kills work that was fine", which is the worse of the two.
        """
        agent = self._agent(monkeypatch, tmp_path)

        # 900 + 4 * 90 = 1260. Widen-only: a small prompt never goes below 900.
        assert resolve_stream_stale_timeout(
            agent, self._payload(agent.model, 90_000)
        ) == 1260.0
        assert resolve_stream_stale_timeout(
            agent, self._payload(agent.model, 1_000)
        ) == 904.0

    def test_wedged_endpoint_stays_bounded_at_any_prompt_size(
        self, monkeypatch, tmp_path
    ):
        """Context scaling must not become a back door to an unbounded wait."""
        agent = self._agent(monkeypatch, tmp_path)

        # 900 + 4 * 400 = 2500, clamped by HERMES_DFLASH_FIRST_CHUNK_CEILING.
        assert resolve_stream_stale_timeout(
            agent, self._payload(agent.model, 400_000)
        ) == 1800.0

    def test_explicit_stream_stale_timeout_still_wins(self, monkeypatch, tmp_path):
        """An operator who pins the global stale timeout gets exactly it."""
        agent = self._agent(monkeypatch, tmp_path)
        monkeypatch.setenv("HERMES_STREAM_STALE_TIMEOUT", "45")

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 45.0

    def test_provider_config_still_wins(self, monkeypatch, tmp_path):
        """``providers.<id>.stale_timeout_seconds`` remains canonical."""
        agent = self._agent(
            monkeypatch,
            tmp_path,
            config_body="""\
            agent:
              local_stream_stale_timeout: 1200
            providers:
              ollama:
                stale_timeout_seconds: 60
            """,
        )

        assert resolve_stream_stale_timeout(agent, self._payload(agent.model)) == 60.0

    def test_remote_endpoint_is_untouched(self, monkeypatch, tmp_path):
        """The ceiling is local-only; cloud providers keep the 180s default."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        monkeypatch.delenv("HERMES_STREAM_STALE_TIMEOUT", raising=False)
        monkeypatch.setenv("HERMES_LOCAL_STREAM_STALE_TIMEOUT", "300")

        from run_agent import AIAgent

        with patch(
            "agent.context_compressor.get_model_context_length",
            return_value=256_000,
        ):
            agent = AIAgent(
                api_key="sk-dummy",
                base_url="https://api.openai.com/v1",
                provider="openai",
                model="gpt-5.2",
                quiet_mode=True,
                skip_context_files=True,
                skip_memory=True,
                platform="cli",
            )

        assert resolve_stream_stale_timeout(agent, self._payload("gpt-5.2")) == 180.0


class TestGenericLocalStaleWatchdogActuallyFires:
    """The ceiling has to reach the watchdog, not just the resolver.

    The resolution tests above prove the number. This one drives the real poll
    loop in ``interruptible_streaming_api_call`` against a local endpoint that
    accepts the request and then never sends a chunk — the wedged-server shape
    the ceiling exists for — and asserts the connection is actually killed.
    Before the fix that branch was unreachable for a generic local endpoint:
    its threshold was ``inf``, so ``_stale_elapsed > _stream_stale_timeout``
    could never be true and the turn parked until the user gave up.
    """

    @staticmethod
    def _drive(monkeypatch, tmp_path, *, polls, env=None):
        """Poll ``polls`` times with a 60s-per-poll clock and no chunks ever."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_LOCAL_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_STREAM_STALE_GIVEUP",
        ):
            monkeypatch.delenv(var, raising=False)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)

        class Clock:
            now = 0.0

            @classmethod
            def time(cls):
                return cls.now

        class NeverAnswersThread:
            joins = 0

            def __init__(self, *, target, daemon):
                self.target = target

            def start(self):
                return None

            def is_alive(self):
                return NeverAnswersThread.joins < polls

            def join(self, timeout=None):
                NeverAnswersThread.joins += 1
                Clock.now += 60.0

        statuses = []

        class Agent:
            api_mode = "chat_completions"
            base_url = "http://localhost:11434/v1"
            provider = "ollama"
            model = "qwen3.6-27b"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_m):
                return None

            @staticmethod
            def _buffer_status(m):
                statuses.append(m)

            @staticmethod
            def _emit_wait_notice(_m):
                return None

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                return None

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread", NeverAnswersThread
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)
        # A bare Ollama has no `/_gate/status`, so the liveness probe yields no
        # signal and the plain stopwatch decides. Stub it rather than let the
        # watchdog make a real request to port 11434 from a unit test.
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active",
            lambda base_url, model: None,
        )

        agent = Agent()
        try:
            interruptible_streaming_api_call(
                agent,
                {
                    "model": "qwen3.6-27b",
                    "messages": [{"role": "user", "content": "hi"}],
                },
            )
        except BaseException:  # noqa: BLE001
            pass
        kills = [s for s in statuses if "No response from provider" in s]
        return agent, kills

    def test_wedged_local_endpoint_is_killed_at_the_ceiling(
        self, monkeypatch, tmp_path
    ):
        """20 polls = 1200s of silence: the 900s ceiling must trip inside it."""
        agent, kills = self._drive(monkeypatch, tmp_path, polls=20)

        assert len(kills) == 1, kills
        # First poll past 900s on a 60s-per-poll clock.
        assert "960s" in kills[0]
        # The kill counts toward the cross-turn give-up breaker, so a
        # persistently wedged endpoint escalates instead of looping forever.
        assert agent._consecutive_stale_streams == 1

    def test_explicit_disable_reproduces_the_old_unbounded_wait(
        self, monkeypatch, tmp_path
    ):
        """Also the pre-fix behavior, kept reachable on purpose.

        Same 1200s of silence, ceiling disabled: nothing is killed and nothing
        is counted. That is exactly what EVERY generic local endpoint did
        before this fix — an operator who wants it back still has it.
        """
        agent, kills = self._drive(
            monkeypatch,
            tmp_path,
            polls=20,
            env={"HERMES_LOCAL_STREAM_STALE_TIMEOUT": "0"},
        )

        assert kills == []
        assert agent._consecutive_stale_streams == 0


class TestLocalStaleWatchdogAsksTheBackend:
    """The stale branch must ask a local backend before killing its stream.

    #278 established the rule for the no-first-chunk branch: a client-side
    stopwatch cannot tell a wedged server apart from a healthily slow one, so
    ask ``/_gate/status`` and only kill a backend that is not working (or one
    that has blown the absolute ceiling). The stale branch — the one that
    actually kills a generic local stream — never asked.

    That was harmless while generic local endpoints resolved to ``inf``: the
    branch could not fire for them. #338 gave them a finite 900s ceiling, which
    is what makes the question live. A healthy-but-slow local server now has its
    connection killed and reconnected, throwing away the prefill it had already
    paid for, which is the first step of a kill -> re-prefill -> kill spiral.

    The probe is unchanged and still conservative: it returns None for any
    endpoint without a gate, so a bare Ollama or llama.cpp behaves exactly as
    before and a remote API is never even asked.
    """

    POLL_SECONDS = 60.0

    @staticmethod
    def _seconds(message: str, prefix: str) -> int:
        match = re.search(rf"{prefix} (\d+)s", message)
        assert match, f"no {prefix!r} figure in {message!r}"
        return int(match.group(1))

    @classmethod
    def _kill_after(cls, kills: list) -> int:
        return cls._seconds(kills[0], "No response from provider for")

    @classmethod
    def _drive(
        cls,
        monkeypatch,
        tmp_path,
        *,
        polls,
        backend_active,
        base_url="http://localhost:11434/v1",
        provider="ollama",
        model="qwen3.6-27b",
        saw_first_chunk=False,
        chunk_at=None,
        env=None,
    ):
        """Poll ``polls`` times on a 60s-per-poll clock, delivering no chunks.

        ``backend_active`` is what the liveness probe reports: True (busy),
        False (idle), or None (endpoint has no progress signal at all).
        ``saw_first_chunk`` starts the call in the mid-stream phase, and
        ``chunk_at`` delivers one real chunk at that poll number.

        Returns ``(agent, kills, waits, probes)`` where ``probes`` records the
        wall-clock time of every liveness question asked.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        for var in (
            "HERMES_STREAM_STALE_TIMEOUT",
            "HERMES_LOCAL_STREAM_STALE_TIMEOUT",
            "HERMES_DFLASH_PREFILL_SECONDS_PER_1K",
            "HERMES_DFLASH_FIRST_CHUNK_CEILING",
            "HERMES_DFLASH_STALE_TIMEOUT",
            "HERMES_DFLASH_STREAM_STALE_TIMEOUT",
            "HERMES_STREAM_STALE_GIVEUP",
        ):
            monkeypatch.delenv(var, raising=False)
        for key, value in (env or {}).items():
            monkeypatch.setenv(key, value)

        class Clock:
            now = 0.0

            @classmethod
            def time(cls_):
                return cls_.now

        class NeverAnswersThread:
            joins = 0

            def __init__(self, *, target, daemon):
                self.target = target
                # Play the part of the streaming worker: the chunk trackers the
                # poll loop reads live in ``_call_chat_completions``, which
                # ``_call`` delegates to, so reach them one level down.
                worker = _unwrap_thread_target(target)
                outer = worker.__code__.co_freevars
                assert "_call_chat_completions" in outer, (
                    "the streaming worker no longer delegates to "
                    "_call_chat_completions; this harness must be updated"
                )
                inner = worker.__closure__[
                    outer.index("_call_chat_completions")
                ].cell_contents
                free = inner.__code__.co_freevars
                assert {"last_chunk_time", "first_chunk_seen"} <= set(free), (
                    "the streaming worker no longer closes over the chunk "
                    "trackers; this harness must be updated to reach them"
                )
                self.last_chunk_time = inner.__closure__[
                    free.index("last_chunk_time")
                ].cell_contents
                self.first_chunk_seen = inner.__closure__[
                    free.index("first_chunk_seen")
                ].cell_contents
                if saw_first_chunk:
                    self.first_chunk_seen["yes"] = True

            def start(self):
                return None

            def is_alive(self):
                return NeverAnswersThread.joins < polls

            def join(self, timeout=None):
                NeverAnswersThread.joins += 1
                Clock.now += cls.POLL_SECONDS
                if chunk_at is not None and NeverAnswersThread.joins == chunk_at:
                    # A real chunk lands: the worker stamps the tracker, exactly
                    # as it does on every SSE delta.
                    self.last_chunk_time["t"] = Clock.now
                    self.first_chunk_seen["yes"] = True

        statuses = []
        probes = []

        def _probe(probe_base_url, probe_model):
            probes.append((Clock.now, probe_base_url, probe_model))
            return backend_active

        class Agent:
            api_mode = "chat_completions"
            _interrupt_requested = False
            _consecutive_stale_streams = 0

            @staticmethod
            def _touch_activity(_m):
                return None

            @staticmethod
            def _buffer_status(m):
                statuses.append(m)

            @staticmethod
            def _emit_wait_notice(_m):
                return None

            @staticmethod
            def _replace_primary_openai_client(*, reason):
                return None

        Agent.base_url = base_url
        Agent.provider = provider
        Agent.model = model

        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread", NeverAnswersThread
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active", _probe
        )

        agent = Agent()
        try:
            interruptible_streaming_api_call(
                agent,
                {"model": model, "messages": [{"role": "user", "content": "hi"}]},
            )
        except AssertionError:
            # A guard from the fake worker above: a harness that has gone blind
            # must fail loudly, not report an empty status list as a result.
            raise
        except BaseException:  # noqa: BLE001
            pass
        kills = [s for s in statuses if "No response from provider" in s]
        waits = [s for s in statuses if "The server reports it is working" in s]
        return agent, kills, waits, probes

    def test_busy_backend_extends_instead_of_killing(self, monkeypatch, tmp_path):
        """THE fix: a local server that says it is generating keeps its socket.

        Without it the 900s ceiling fires at the first poll past it (960s) and
        the prefill already paid for is thrown away.
        """
        agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=True
        )

        assert waits, (
            "the watchdog killed a stream the backend reported as ACTIVE — it "
            "never asked, or asked and ignored the answer"
        )
        assert kills == [], kills
        assert agent._consecutive_stale_streams == 0, (
            "an extended stream must not count toward the give-up breaker"
        )
        # It asked about the endpoint and model actually in flight.
        assert probes and probes[0][1:] == (
            "http://localhost:11434/v1",
            "qwen3.6-27b",
        )
        # And it asked at the threshold, not before it.
        assert self._seconds(waits[0], "No output for") == 960

    def test_idle_backend_is_still_killed_promptly(self, monkeypatch, tmp_path):
        """A backend that reports nothing in flight is genuinely wedged."""
        agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=False
        )

        assert waits == []
        assert len(kills) == 1, kills
        assert self._kill_after(kills) == 960
        assert probes, "killed without asking"
        assert agent._consecutive_stale_streams == 1

    def test_endpoint_with_no_gate_keeps_the_plain_stopwatch(
        self, monkeypatch, tmp_path
    ):
        """A bare Ollama/llama.cpp has no progress signal: unchanged behavior.

        This is the same outcome as before this change, and it is the common
        case — the probe only ever helps a gated lane.
        """
        agent, kills, waits, _probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=None
        )

        assert waits == []
        assert len(kills) == 1, kills
        assert self._kill_after(kills) == 960
        assert agent._consecutive_stale_streams == 1

    def test_remote_endpoint_is_never_asked(self, monkeypatch, tmp_path):
        """Only a local endpoint can have a gate — never probe a remote API.

        ``backend_active=True`` here would extend forever if the guard were
        missing, so this pins the guard and not just the probe's None default.
        """
        # Six polls = 360s: one pass of the remote 180s threshold, so exactly
        # one kill to look at (it re-arms every 180s after that).
        _agent, kills, waits, probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=6,
            backend_active=True,
            base_url="https://api.example-cloud.com/v1",
            provider="openai",
            model="acme-chat-1",
        )

        assert probes == [], "asked a remote provider for a local liveness signal"
        assert waits == []
        assert len(kills) == 1, kills

    def test_a_backend_that_never_stops_saying_busy_is_still_bounded(
        self, monkeypatch, tmp_path
    ):
        """The ceiling is the backstop for a gate that lies (or is stuck).

        The invariant: one silent stretch can never outlast the ceiling by more
        than the poll interval that discovers it.
        """
        ceiling = 1200.0
        _agent, kills, waits, _probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=40,
            backend_active=True,
            env={"HERMES_DFLASH_FIRST_CHUNK_CEILING": str(int(ceiling))},
        )

        assert waits, "never extended, so this proves nothing about the ceiling"
        assert len(kills) == 1, kills
        killed_after = self._kill_after(kills)
        assert ceiling < killed_after <= ceiling + self.POLL_SECONDS, (
            f"silent stretch ran {killed_after}s against a {ceiling:.0f}s ceiling"
        )

    def test_dflash_mid_stream_stall_is_asked_about_too(self, monkeypatch, tmp_path):
        """The DFlash stale branch gets the same treatment, not just the generic one.

        DFlash local already had the probe before its first chunk. Once that
        chunk arrived it fell back to the same unconditional kill, so the answer
        to "is the server still working?" was authoritative for one phase of a
        request and ignored for the next. The branch is shared, so both phases
        now ask.
        """
        dflash = {
            "base_url": "http://10.10.20.211:8080/v1",
            "provider": "taro",
            "model": "deepseek-v4-flash-w2",
            "saw_first_chunk": True,
        }
        # Seven polls = 420s: past the 180s DFlash stale timeout, and short of
        # the re-arm that would give the no-signal arm below a second kill.
        _agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=7, backend_active=True, **dflash
        )

        assert waits, "a mid-stream DFlash stall was killed without asking"
        assert kills == []
        assert probes[0][1:] == (
            "http://10.10.20.211:8080/v1",
            "deepseek-v4-flash-w2",
        )

        # Same stall, no progress signal: the 180s DFlash stale timeout still
        # kills, so the extension above is the probe's doing and not a widened
        # threshold.
        _agent, kills, waits, _probes = self._drive(
            monkeypatch, tmp_path, polls=7, backend_active=None, **dflash
        )
        assert waits == []
        assert len(kills) == 1, kills
        assert self._kill_after(kills) == 240

    def test_extension_does_not_carry_into_the_next_silent_stretch(
        self, monkeypatch, tmp_path
    ):
        """Each stall is judged on its own; grants must not accumulate.

        If the extension survived the chunk that ended the stall, every
        extension would raise the bar for the rest of the request and the
        watchdog would drift toward never firing.
        """
        chunk_at = 20  # a real chunk at t=1200s, after the first extension
        _agent, _kills, _waits, probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=40,
            backend_active=True,
            chunk_at=chunk_at,
        )

        chunk_at_seconds = chunk_at * self.POLL_SECONDS
        asked_after_chunk = [t for (t, _u, _m) in probes if t > chunk_at_seconds]
        assert asked_after_chunk, "the post-chunk stretch never reached a probe"
        # The fresh stretch is measured from the chunk and gets the whole 900s
        # threshold back — not 900s plus what the previous stretch was granted.
        assert asked_after_chunk[0] - chunk_at_seconds == 960
