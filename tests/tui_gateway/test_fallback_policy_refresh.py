import contextlib
import threading

from tui_gateway import server


class _ImmediateThread:
    def __init__(self, target=None, daemon=None):
        self._target = target
        self._alive = False

    def start(self):
        self._alive = True
        try:
            self._target()
        finally:
            self._alive = False

    def is_alive(self):
        return self._alive


class _Agent:
    model = "deepseek-v4-flash-w2"
    provider = "custom"
    _fallback_activated = False

    def __init__(self, order):
        self._order = order
        self._fallback_policy = "any"

    def clear_interrupt(self):
        return None

    def _refresh_fallback_policy(self):
        self._order.append("refresh")
        self._fallback_policy = "local-only"
        return self._fallback_policy

    def run_conversation(self, *_args, **_kwargs):
        self._order.append("request")
        return {
            "completed": True,
            "final_response": "ok",
            "messages": [
                {"role": "user", "content": "run"},
                {"role": "assistant", "content": "ok"},
            ],
        }


def test_cached_policy_refresh_is_visible_before_message_start_and_request(monkeypatch):
    order = []
    agent = _Agent(order)
    session = {
        "agent": agent,
        "attached_images": [],
        "cols": 80,
        "cwd": "/tmp",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "inflight_turn": None,
        "profile_home": "/tmp/profile-mesh",
        "running": True,
        "session_key": "stored-session",
        "transport": None,
    }

    def emit(event, _sid, payload):
        if event == "session.info":
            order.append(f"session.info:{payload['fallback_policy']}")
        elif event == "message.start":
            order.append("message.start")

    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_emit", emit)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server,
        "_session_db",
        lambda _session: contextlib.nullcontext(None),
    )
    monkeypatch.setattr(server, "_get_usage", lambda _agent: {})
    monkeypatch.setattr(server, "_sync_agent_model_with_config", lambda *_args: None)
    monkeypatch.setattr(server, "_sync_session_key_after_compress", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *_args: None)
    monkeypatch.setattr(server, "_wire_callbacks", lambda *_args: None)
    monkeypatch.setattr(server, "_set_session_context", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(server, "_clear_session_context", lambda *_args: None)
    monkeypatch.setattr(
        server,
        "set_hermes_home_override",
        lambda profile_home: order.append(f"profile:{profile_home}") or object(),
    )
    monkeypatch.setattr(server, "reset_hermes_home_override", lambda _token: None)
    monkeypatch.setattr(
        server,
        "_session_info",
        lambda live_agent, _session: {
            "fallback_policy": live_agent._fallback_policy,
        },
    )
    monkeypatch.setattr(server, "make_stream_renderer", lambda _cols: None)
    monkeypatch.setattr(server, "render_message", lambda _text, _cols: None)
    monkeypatch.setattr(server, "_voice_tts_enabled", lambda: False)
    monkeypatch.setattr(server, "_drain_queued_prompt", lambda *_args: False)

    server._start_inflight_turn(session, "run")
    server._run_prompt_submit("request-1", "runtime-session", session, "run")

    assert order.index("profile:/tmp/profile-mesh") < order.index("refresh")
    assert order.index("refresh") < order.index("session.info:local-only")
    assert order.index("session.info:local-only") < order.index("message.start")
    assert order.index("message.start") < order.index("request")
