"""Mid-turn context-usage events.

Usage used to cross the wire only at turn boundaries, so a single long agentic
turn — every ``hermes desktop spawn --delegated`` run is one — left the desktop
context meter frozen for its whole duration while the window really filled.
``_on_tool_complete`` now also pushes a ``token.usage`` frame.

Tool completion alone still lags the meter: it reports the request that
*produced* the tool call, so while the next (larger) request is in flight the
gauge shows the previous round, and a reasoning-only round reports nothing at
all. The agent loop's per-API-call hook closes that gap; the frames it adds are
deduplicated, and tool-round frames carry a minimum interval so a parallel batch
cannot thrash the status bar.
"""

import time
import types

from tui_gateway import server


def _canonical(*, prompt_tokens: int, output_tokens: int = 200):
    """One provider response's usage. ``prompt_tokens`` is derived from the
    input/cache buckets, so it is the input bucket that carries the count."""
    from agent.usage_pricing import CanonicalUsage

    return CanonicalUsage(input_tokens=prompt_tokens, output_tokens=output_tokens)


def _hold_the_floor_open(monkeypatch, seconds: float = 3_600.0) -> None:
    """Widen the interval so suppression is a decision, not a race.

    At the shipped 1s these tests would still pass, but only because two
    in-process calls land close together — a loaded runner could drift past it
    and turn a real assertion into an occasional false green.
    """
    monkeypatch.setattr(server, "_TOKEN_USAGE_MIN_INTERVAL_S", seconds)


def _age_last_emit(sid: str, seconds: float) -> None:
    """Backdate this session's throttle stamp so the floor has expired.

    Rebases the recorded instant rather than patching ``time.monotonic``:
    sleeping would make the interval assertions depend on a quiet runner, and
    replacing the process clock reaches well beyond the code under test.
    """
    server._sessions[sid]["token_usage_emitted_at"] = time.monotonic() - seconds


class _Compressor:
    """The slice of the ContextEngine contract these paths touch.

    ``update_from_response`` is the ABC's one required hook and the only way a
    provider's real prompt count reaches occupancy — a stub without it silently
    swallows the update behind the recorder's except, which is precisely the
    class of bug this file guards.
    """

    def __init__(self, last_prompt_tokens: int, context_length: int, compressions: int):
        self.last_prompt_tokens = last_prompt_tokens
        self.context_length = context_length
        self.compression_count = compressions
        self.last_prompt_messages_len = 0

    def update_from_response(self, usage: dict) -> None:
        self.last_prompt_tokens = usage.get("prompt_tokens", 0)


def _agent(*, last_prompt_tokens: int, context_length: int = 262_144, compressions: int = 0):
    """Minimal stand-in shaped the way ``_get_usage`` reads an agent."""
    return types.SimpleNamespace(
        model="deepseek-v4-flash-0731-ds4",
        session_input_tokens=1_000,
        session_output_tokens=500,
        session_total_tokens=1_500,
        session_api_calls=3,
        context_compressor=_Compressor(last_prompt_tokens, context_length, compressions),
    )


def _capture(monkeypatch) -> list[tuple[str, str, dict]]:
    events: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(
        server, "_emit", lambda event_type, sid, payload: events.append((event_type, sid, payload))
    )
    return events


def _session(monkeypatch, sid: str, agent, *, tool_progress_mode: str = "all") -> None:
    monkeypatch.setitem(
        server._sessions,
        sid,
        {"agent": agent, "tool_progress_mode": tool_progress_mode, "tool_started_at": {}},
    )


def test_tool_complete_also_pushes_current_context_occupancy(monkeypatch):
    events = _capture(monkeypatch)
    _session(monkeypatch, "mid-turn", _agent(last_prompt_tokens=95_377))

    server._on_tool_complete("mid-turn", "tool-1", "terminal", {"command": "pwd"}, "done")

    kinds = [event for event, _sid, _payload in events]
    # Usage leads: `tool.complete` is the one unguarded emit here, so a tool
    # result that fails to serialize must not take the usage frame down with it.
    assert kinds == ["token.usage", "tool.complete"]

    _event, sid, payload = events[0]
    assert sid == "mid-turn"
    # The desktop's token.usage handler reads a FLAT shape that differs from
    # _get_usage's own key names — a rename here silently stops the meter.
    assert payload["context_tokens"] == 95_377
    assert payload["context_length"] == 262_144
    assert payload["context_pct"] == 36
    assert payload["compressions"] == 0
    assert payload["total_tokens"] == 1_500


def test_occupancy_is_pushed_even_when_tool_cards_are_off(monkeypatch):
    """The meter must track a long turn for a client that streams no tool cards."""
    events = _capture(monkeypatch)
    _session(monkeypatch, "quiet", _agent(last_prompt_tokens=42_000), tool_progress_mode="off")

    server._on_tool_complete("quiet", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert [event for event, _sid, _payload in events] == ["token.usage"]


def test_compression_count_rides_along(monkeypatch):
    """Without it the renderer's monotonic guard rejects a post-compression drop."""
    events = _capture(monkeypatch)
    _session(monkeypatch, "compressed", _agent(last_prompt_tokens=30_000, compressions=2))

    server._on_tool_complete("compressed", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert _usage_frames(events)[-1]["compressions"] == 2


def test_no_gauge_is_emitted_while_the_window_is_still_unknown(monkeypatch):
    """A fresh compressor reports 0; _get_usage omits the gauge rather than
    fabricating 0%, and a partial frame would blank a meter mid-turn."""
    events = _capture(monkeypatch)
    _session(monkeypatch, "fresh", _agent(last_prompt_tokens=0))

    server._on_tool_complete("fresh", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert [event for event, _sid, _payload in events] == ["tool.complete"]


def test_session_without_a_built_agent_stays_silent(monkeypatch):
    events = _capture(monkeypatch)
    monkeypatch.setitem(
        server._sessions, "lazy", {"agent": None, "tool_progress_mode": "all", "tool_started_at": {}}
    )

    server._on_tool_complete("lazy", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert [event for event, _sid, _payload in events] == ["tool.complete"]


def test_a_broken_agent_cannot_take_down_the_tool_callback(monkeypatch):
    events = _capture(monkeypatch)

    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("agent is mid-teardown")

    _session(monkeypatch, "broken", Exploding())

    server._on_tool_complete("broken", "tool-1", "terminal", {"command": "pwd"}, "done")

    assert [event for event, _sid, _payload in events] == ["tool.complete"]


def _usage_frames(events) -> list[dict]:
    return [payload for event, _sid, payload in events if event == "token.usage"]


def test_a_parallel_tool_batch_reports_its_round_once(monkeypatch):
    """Every tool in one batch reads the same occupancy — resending it is pure
    churn on the status bar, so an unchanged gauge is dropped."""
    events = _capture(monkeypatch)
    _session(monkeypatch, "batch", _agent(last_prompt_tokens=55_800))

    for i in range(4):
        server._on_tool_complete("batch", f"tool-{i}", "terminal", {"command": "pwd"}, "done")

    assert [frame["context_tokens"] for frame in _usage_frames(events)] == [55_800]


def test_a_moved_gauge_still_waits_out_the_tool_round_floor(monkeypatch):
    """A fast loop must not push a frame per round trip at the status bar.
    Dropping one is safe: the value is still climbing, so a later frame
    supersedes it and message.complete carries the turn's final number."""
    events = _capture(monkeypatch)
    _hold_the_floor_open(monkeypatch)
    agent = _agent(last_prompt_tokens=55_800)
    _session(monkeypatch, "fast", agent)

    server._on_tool_complete("fast", "tool-1", "terminal", {"command": "pwd"}, "done")
    agent.context_compressor.last_prompt_tokens = 60_000
    server._on_tool_complete("fast", "tool-2", "terminal", {"command": "pwd"}, "done")

    assert [frame["context_tokens"] for frame in _usage_frames(events)] == [55_800]

    _age_last_emit("fast", server._TOKEN_USAGE_MIN_INTERVAL_S + 0.5)
    agent.context_compressor.last_prompt_tokens = 65_700
    server._on_tool_complete("fast", "tool-3", "terminal", {"command": "pwd"}, "done")

    assert [frame["context_tokens"] for frame in _usage_frames(events)] == [55_800, 65_700]


def test_a_compaction_does_not_wait_out_the_floor(monkeypatch):
    """The one moment the gauge legitimately falls. It is also the drop the
    client's monotonic guard is watching `compressions` to authorize, so sitting
    on it would leave the meter pinned high for as long as the floor lasts."""
    events = _capture(monkeypatch)
    _hold_the_floor_open(monkeypatch)
    agent = _agent(last_prompt_tokens=95_000)
    _session(monkeypatch, "compacting", agent)

    server._on_tool_complete("compacting", "tool-1", "terminal", {"command": "pwd"}, "done")
    agent.context_compressor.last_prompt_tokens = 30_000
    agent.context_compressor.compression_count = 1
    server._on_tool_complete("compacting", "tool-2", "terminal", {"command": "pwd"}, "done")

    frames = _usage_frames(events)
    assert [frame["context_tokens"] for frame in frames] == [95_000, 30_000]
    assert frames[-1]["compressions"] == 1


def test_a_real_provider_count_reaches_the_wire(monkeypatch):
    """The contract between the two halves, driven through the real recorder.

    ``record_canonical_usage`` is the one place that sees the provider's own
    prompt count land on the compressor, which is what makes it the honest
    sampling point: the pre-request estimate reads ~13% low against the server
    on dense agentic transcripts. The hook it offers went unassigned for so long
    that the meter only ever moved when a TOOL finished — a round trip behind,
    and silent for a reasoning-only round.
    """
    from agent.session_telemetry import record_canonical_usage

    events = _capture(monkeypatch)
    agent = _agent(last_prompt_tokens=55_800)
    _session(monkeypatch, "served", agent)
    agent._emit_token_usage = lambda **_kwargs: server._emit_token_usage("served")

    record_canonical_usage(agent, _canonical(prompt_tokens=65_700))

    assert agent.context_compressor.last_prompt_tokens == 65_700
    # No tool anywhere in this round: the frame came from the response alone,
    # which is the case the old tool-completion sampling could not report.
    assert [event for event, _sid, _payload in events] == ["token.usage"]
    assert [frame["context_tokens"] for frame in _usage_frames(events)] == [65_700]


def test_the_goal_judge_is_accounted_without_touching_the_gauge(monkeypatch):
    """A goal session runs a judge against every turn's answer. It is a small
    call — a few thousand tokens beside a window in the tens of thousands — so
    were it to reach the meter it would read as the context collapsing.

    Driven through the real auxiliary chokepoint rather than a stand-in, and the
    recorded row is asserted too: without it this would pass just as happily if
    the aux call had never been accounted at all.
    """
    from agent.aux_accounting import (
        record_aux_usage,
        reset_accounting_context,
        set_accounting_context,
    )

    events = _capture(monkeypatch)
    agent = _agent(last_prompt_tokens=65_700)
    _session(monkeypatch, "goal", agent)
    agent._emit_token_usage = lambda **_kwargs: server._emit_token_usage("goal")

    recorded: list[tuple] = []
    session_db = types.SimpleNamespace(
        record_auxiliary_usage=lambda sid, task, **kw: recorded.append((sid, task, kw))
    )
    judge_response = types.SimpleNamespace(
        model="deepseek-v4-flash-0731-ds4",
        usage=types.SimpleNamespace(prompt_tokens=3_100, completion_tokens=40, total_tokens=3_140),
    )

    token = set_accounting_context(session_db, "session-under-test")
    try:
        record_aux_usage(judge_response, "goal_judge")
    finally:
        reset_accounting_context(token)

    assert [task for _sid, task, _kw in recorded] == ["goal_judge"]
    assert _usage_frames(events) == []
    assert agent.context_compressor.last_prompt_tokens == 65_700


def test_an_agent_without_the_hook_cannot_move_this_sessions_gauge(monkeypatch):
    """Only the session's own agent is wired, and the hook is set after
    construction rather than passed as an AIAgent kwarg. A delegate subagent
    builds its own AIAgent (and `delegate_tool` forwards constructor kwargs from
    the parent, so a kwarg is exactly how one could inherit this), while an
    auxiliary call — goal judge, title, compaction — accounts through
    `record_aux_usage` and never reaches this recorder at all. Neither can push
    its much smaller window onto the session's gauge."""
    from agent.session_telemetry import record_canonical_usage

    events = _capture(monkeypatch)
    _session(monkeypatch, "parent", _agent(last_prompt_tokens=65_700))
    child = _agent(last_prompt_tokens=4_100)

    record_canonical_usage(child, _canonical(prompt_tokens=4_800))

    assert _usage_frames(events) == []
