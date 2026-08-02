"""Mid-turn context-usage events.

Usage used to cross the wire only at turn boundaries, so a single long agentic
turn — every ``hermes desktop spawn --delegated`` run is one — left the desktop
context meter frozen for its whole duration while the window really filled.
``_on_tool_complete`` now also pushes a ``token.usage`` frame.
"""

import types

from tui_gateway import server


def _agent(*, last_prompt_tokens: int, context_length: int = 262_144, compressions: int = 0):
    """Minimal stand-in shaped the way ``_get_usage`` reads an agent."""
    return types.SimpleNamespace(
        model="deepseek-v4-flash-0731-ds4",
        session_input_tokens=1_000,
        session_output_tokens=500,
        session_total_tokens=1_500,
        session_api_calls=3,
        context_compressor=types.SimpleNamespace(
            last_prompt_tokens=last_prompt_tokens,
            context_length=context_length,
            compression_count=compressions,
        ),
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
    assert kinds == ["tool.complete", "token.usage"]

    _event, sid, payload = events[1]
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

    assert events[-1][2]["compressions"] == 2


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
