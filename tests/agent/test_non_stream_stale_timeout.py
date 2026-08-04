"""Tests for the non-stream stale-call detector context estimator.

Covers:
- ``estimate_request_context_tokens`` for Chat Completions, Responses API,
  bare lists, and mixed-shape dicts.
- ``AIAgent._compute_non_stream_stale_timeout`` with both legacy ``messages``
  list and full ``api_kwargs`` dicts.
- The May 2026 default-base change (300s -> 90s) and the lowered
  context-tier ceilings (450/600 -> 150/240).
- The local liveness probe on the non-streaming kill path: which
  configurations can reach that kill at all, and that a local backend
  reporting an active generation is asked before its request is killed.
"""

from __future__ import annotations

import re
from pathlib import Path



def _write_config(tmp_path: Path, body: str) -> None:
    hermes_home = tmp_path
    (hermes_home / "config.yaml").write_text(body or "{}\n", encoding="utf-8")


def _make_agent(tmp_path: Path, **overrides):
    from run_agent import AIAgent
    kwargs = dict(
        model="gpt-5.5",
        provider="openai-codex",
        api_key="sk-dummy",
        base_url="https://chatgpt.com/backend-api/codex",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )
    kwargs.update(overrides)
    return AIAgent(**kwargs)


# ── estimator ──────────────────────────────────────────────────────────────




def test_estimator_responses_api_input():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    payload = {
        "model": "gpt-5.5",
        "instructions": "i" * 1000,
        "input": "x" * 4000,
        "tools": [{"name": "t", "description": "d" * 200}],
    }
    # input(4000) + instructions(1000) + tools (~stringified) -> well over 1000 tokens
    tokens = estimate_request_context_tokens(payload)
    assert tokens >= 1200, f"Responses API estimator returned {tokens}"






def test_estimator_empty_inputs():
    from agent.chat_completion_helpers import estimate_request_context_tokens
    assert estimate_request_context_tokens({}) == 0
    assert estimate_request_context_tokens([]) == 0
    assert estimate_request_context_tokens(None) == 0




# ── default base + tier scaling ────────────────────────────────────────────


def test_default_base_is_90s(monkeypatch, tmp_path):
    """Default base stale timeout dropped from 300s to 90s (May 2026)."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)
    _write_config(tmp_path, "")

    agent = _make_agent(tmp_path)
    base, implicit = agent._resolved_api_call_stale_timeout_base()
    assert base == 90.0
    assert implicit is True










def test_explicit_user_config_overrides_default(monkeypatch, tmp_path):
    """If the user explicitly sets a stale_timeout, the new defaults don't apply."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / ".env").write_text("", encoding="utf-8")
    _write_config(tmp_path, """\
providers:
  openai-codex:
    stale_timeout_seconds: 1800
""")
    monkeypatch.delenv("HERMES_API_CALL_STALE_TIMEOUT", raising=False)

    import importlib
    from hermes_cli import timeouts as to_mod
    importlib.reload(to_mod)

    agent = _make_agent(tmp_path)
    assert agent._compute_non_stream_stale_timeout({"input": "hi"}) == 1800.0


# ── openai-codex gateway-scale stale floor ────────────────────────────────




def test_openai_codex_stale_floor_tiers():
    from agent.chat_completion_helpers import openai_codex_stale_timeout_floor

    assert openai_codex_stale_timeout_floor(55_000) == 900.0
    assert openai_codex_stale_timeout_floor(120_000) == 1200.0


# ── the local liveness probe on the non-streaming kill path ───────────────


class TestLocalNonStreamStaleWatchdogAsksTheBackend:
    """The non-streaming stale detector must ask a local backend before killing.

    Both streaming watchdogs already do: the DFlash pre-first-chunk wait
    (#278) and the shared stale-stream branch (#343). A client-side stopwatch
    cannot tell a wedged server apart from a healthily slow one; the gate can.
    The non-streaming detector was the last one still deciding on the clock
    alone, and it is the one where a wrong answer costs the most — nothing is
    delivered until everything is, so a kill throws away the whole prefill AND
    the whole generation, and the retry restarts from zero.

    The reachable population is narrower than the streaming case, and the
    first test below pins why: a generic local endpoint on the implicit
    default resolves to ``inf`` and cannot reach the kill at all, so extending
    it would be dead code. What CAN reach it is covered here — a local DFlash
    model, a local model in the reasoning-floor allowlist, and an
    operator-pinned threshold.

    The probe itself is unchanged and still conservative: it returns None for
    any endpoint without a gate, so a bare Ollama or llama.cpp is decided by
    the timer exactly as before and a remote API is never asked at all.
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

    @staticmethod
    def _stale_streak(agent) -> int:
        """Read the give-up breaker the way production does.

        ``_consecutive_stale_streams`` is set lazily — only a bump or a reset
        creates it — so a fresh agent has no attribute at all and reading it
        directly is an AttributeError, not a zero.
        """
        from agent.chat_completion_helpers import _stale_streak

        return _stale_streak(agent)

    @classmethod
    def _drive(
        cls,
        monkeypatch,
        tmp_path,
        *,
        polls,
        backend_active,
        base_url="http://10.10.20.211:8080/v1",
        provider="taro",
        model="deepseek-v4-flash-w2",
        env=None,
    ):
        """Poll ``polls`` times on a 60s-per-poll clock, never answering.

        A non-streaming call delivers nothing until it delivers everything, so
        the worker simply never returns — which is exactly the shape of the
        failure being judged.  ``backend_active`` is what the liveness probe
        reports: True (busy), False (idle) or None (no progress signal at all).

        Returns ``(agent, kills, waits, probes)`` where ``probes`` records the
        wall-clock time of every liveness question asked.
        """
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        (tmp_path / ".env").write_text("", encoding="utf-8")
        _write_config(tmp_path, "")
        for var in (
            "HERMES_API_CALL_STALE_TIMEOUT",
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

            def start(self):
                return None

            def is_alive(self):
                return NeverAnswersThread.joins < polls

            def join(self, timeout=None):
                NeverAnswersThread.joins += 1
                Clock.now += cls.POLL_SECONDS

        statuses = []
        probes = []

        def _probe(probe_base_url, probe_model):
            probes.append((Clock.now, probe_base_url, probe_model))
            return backend_active

        agent = _make_agent(
            tmp_path, provider=provider, base_url=base_url, model=model
        )
        # Harness guards. Without these a blind harness reports an empty
        # status list as if it were evidence of unchanged behavior:
        #   - a Codex api_mode would kill on the watchdogs ABOVE the stale
        #     branch, on their own cutoffs;
        #   - should_use_direct_api_call would route the whole call to
        #     direct_api_call, which has no poll loop and no watchdog at all.
        from agent.chat_completion_helpers import should_use_direct_api_call

        assert agent.api_mode == "chat_completions", agent.api_mode
        assert not should_use_direct_api_call(agent)

        monkeypatch.setattr(agent, "_buffer_status", statuses.append)
        monkeypatch.setattr(agent, "_touch_activity", lambda _m: None)
        monkeypatch.setattr(agent, "_emit_wait_notice", lambda _m: None)
        monkeypatch.setattr(
            "agent.chat_completion_helpers.threading.Thread", NeverAnswersThread
        )
        monkeypatch.setattr("agent.chat_completion_helpers.time.time", Clock.time)
        monkeypatch.setattr(
            "agent.chat_completion_helpers._local_backend_generation_active", _probe
        )

        from agent.chat_completion_helpers import interruptible_api_call

        try:
            interruptible_api_call(
                agent,
                {"model": model, "messages": [{"role": "user", "content": "hi"}]},
            )
        except AssertionError:
            # A harness guard, not a result: fail loudly rather than report an
            # empty status list as evidence of anything.
            raise
        except BaseException:  # noqa: BLE001
            pass
        kills = [s for s in statuses if "No response from provider" in s]
        waits = [s for s in statuses if "The server reports it is working" in s]
        return agent, kills, waits, probes

    def test_generic_local_endpoint_cannot_reach_the_kill_at_all(
        self, monkeypatch, tmp_path
    ):
        """Why this fix is narrower than the streaming one.

        A local endpoint on the implicit default resolves to ``inf`` in
        ``_compute_non_stream_stale_timeout``, so the branch never fires and
        there is nothing to ask about. If this ever stops being true, the
        probe below starts covering a much larger population — which is fine,
        but it should be a decision and not a surprise.
        """
        agent, kills, waits, probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=20,
            backend_active=True,
            base_url="http://localhost:11434/v1",
            provider="ollama",
            model="llama3.3-70b",
        )

        assert agent._compute_non_stream_stale_timeout(
            {"messages": [{"role": "user", "content": "hi"}]}
        ) == float("inf")
        assert kills == []
        assert waits == []
        assert probes == [], "asked about a request that could never be killed"

    def test_busy_dflash_backend_extends_instead_of_killing(
        self, monkeypatch, tmp_path
    ):
        """THE fix: a local server that says it is generating keeps its socket.

        Without it the DFlash 180s budget fires at the first poll past it
        (240s) and the prefill already paid for is thrown away — with no
        partial output to show for it, unlike the streaming case.
        """
        agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=True
        )

        assert waits, (
            "the watchdog killed a call the backend reported as ACTIVE — it "
            "never asked, or asked and ignored the answer"
        )
        assert kills == []
        assert self._stale_streak(agent) == 0, (
            "an extended call must not count toward the give-up breaker"
        )
        # It asked about the endpoint and model actually in flight.
        assert probes and probes[0][1:] == (
            "http://10.10.20.211:8080/v1",
            "deepseek-v4-flash-w2",
        )
        # And it asked at the threshold, not before it.
        assert self._seconds(waits[0], "No response for") == 240

    def test_idle_backend_is_still_killed_promptly(self, monkeypatch, tmp_path):
        """A backend that reports nothing in flight is genuinely wedged."""
        agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=False
        )

        assert waits == []
        assert kills, "a wedged backend was never killed"
        assert self._kill_after(kills) == 240
        assert probes, "killed without asking"
        assert self._stale_streak(agent) >= 1

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
        assert kills, "a backend with no progress signal must still be killed"
        assert self._kill_after(kills) == 240
        assert self._stale_streak(agent) >= 1

    def test_remote_endpoint_is_never_asked(self, monkeypatch, tmp_path):
        """Only a local endpoint can have a gate — never probe a remote API.

        ``backend_active=True`` here would extend forever if the guard were
        missing, so this pins the guard and not just the probe's None default.
        """
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
        assert kills, "a remote provider's stale call must still be killed"

    def test_a_backend_that_never_stops_saying_busy_is_still_bounded(
        self, monkeypatch, tmp_path
    ):
        """The ceiling is the backstop for a gate that lies (or is stuck).

        The invariant: a request can never outlast the ceiling by more than
        the poll interval that discovers it.
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
        assert ceiling <= killed_after <= ceiling + self.POLL_SECONDS, (
            f"request ran {killed_after}s against a {ceiling:.0f}s ceiling"
        )

    def test_local_reasoning_model_is_asked_too(self, monkeypatch, tmp_path):
        """The second reachable population, and the least obvious one.

        A local model in the reasoning-floor allowlist is NOT DFlash, so it
        looks like the generic local case that resolves to ``inf`` — but the
        floor returns ``uses_implicit_default=False``, and that is exactly the
        flag the ``inf`` short-circuit tests. So a plain llama.cpp serving
        deepseek-r1 behind a gate gets a finite 600s budget and reaches the
        kill, where every other generic local model does not.
        """
        local_r1 = {
            "base_url": "http://10.10.20.211:8080/v1",
            "provider": "taro",
            "model": "deepseek-r1-distill-qwen-32b",
        }
        agent, kills, waits, probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=True, **local_r1
        )

        assert agent._compute_non_stream_stale_timeout(
            {"messages": [{"role": "user", "content": "hi"}]}
        ) == 600.0
        assert waits, "a local reasoning model was killed without asking"
        assert kills == []
        assert probes[0][1:] == (
            "http://10.10.20.211:8080/v1",
            "deepseek-r1-distill-qwen-32b",
        )

        # Same call, no progress signal: the 600s floor still kills, so the
        # extension above is the probe's doing and not a widened threshold.
        _agent, kills, waits, _probes = self._drive(
            monkeypatch, tmp_path, polls=20, backend_active=None, **local_r1
        )
        assert waits == []
        assert len(kills) == 1, kills
        assert self._kill_after(kills) == 660

    def test_operator_pinned_threshold_is_asked_about(self, monkeypatch, tmp_path):
        """The third reachable population: a pinned threshold on any local lane.

        ``HERMES_API_CALL_STALE_TIMEOUT`` makes the base explicit, which also
        skips the ``inf`` short-circuit — so the generic local endpoint that
        could not be killed in the first test can be killed here, and is now
        asked about first.
        """
        _agent, kills, waits, probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=10,
            backend_active=True,
            base_url="http://10.10.20.211:8080/v1",
            provider="taro",
            model="llama3.3-70b",
            env={"HERMES_API_CALL_STALE_TIMEOUT": "300"},
        )

        assert probes, "a pinned local threshold killed without asking"
        assert waits
        assert kills == []
        assert self._seconds(waits[0], "No response for") == 360

    def test_pinned_threshold_above_the_ceiling_is_honoured_unchanged(
        self, monkeypatch, tmp_path
    ):
        """The probe may only ever extend a deadline, never shorten one.

        An operator who pins a threshold above the ceiling has already chosen
        to wait longer than the probe would ever grant. The guard is false the
        first time the branch fires, so the probe never runs and the pinned
        figure decides on its own — the pre-change behavior exactly.
        """
        _agent, kills, waits, probes = self._drive(
            monkeypatch,
            tmp_path,
            polls=40,
            backend_active=True,
            model="llama3.3-70b",
            env={
                "HERMES_API_CALL_STALE_TIMEOUT": "1500",
                "HERMES_DFLASH_FIRST_CHUNK_CEILING": "600",
            },
        )

        assert probes == [], "probed past a ceiling it had already blown"
        assert waits == []
        assert len(kills) == 1, kills
        assert self._kill_after(kills) == 1560
