"""The request prefix a long session sends must only ever grow at the end.

Observed 2026-08-02 on a local DeepSeek-V4 lane: deep sessions re-prefilled
from scratch instead of reusing the server's KV cache, and there was no way to
tell whether the client had rewritten the prefix or something had evicted the
server's copy. ``agent/prefix_probe.py`` is the instrument that answers it;
this file pins both the instrument and the contract it measures.

The contract: for two consecutive API calls in one session, every prefix
element of the earlier request must be byte-identical in the later one. A turn
may append; it may not rewrite. That has to hold across a FRESH agent too —
the compute host crashes, the desktop app auto-updates, and the gateway builds
a new agent per turn, so "stable because the object stayed in memory" is not
the property worth having.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.prefix_probe import (
    divergence_index,
    prefix_elements,
    prefix_probe_enabled,
    record_request_prefix,
    rolling_hashes,
)
from hermes_state import SessionDB


# ---------------------------------------------------------------------------
# The instrument itself
# ---------------------------------------------------------------------------

class TestPrefixProbeUnit:
    def test_off_unless_explicitly_enabled(self, monkeypatch):
        monkeypatch.delenv("HERMES_PREFIX_PROBE", raising=False)
        assert prefix_probe_enabled() is False
        monkeypatch.setenv("HERMES_PREFIX_PROBE", "0")
        assert prefix_probe_enabled() is False
        monkeypatch.setenv("HERMES_PREFIX_PROBE", "1")
        assert prefix_probe_enabled() is True

    def test_pure_append_reports_no_divergence(self):
        base = {"tools": [{"name": "t"}], "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "one"},
        ]}
        grown = {"tools": [{"name": "t"}], "messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": "two"},
        ]}
        before = rolling_hashes(prefix_elements(base))
        after = rolling_hashes(prefix_elements(grown))
        assert divergence_index(before, after) is None

    def test_rewriting_an_earlier_message_is_located_exactly(self):
        base = {"messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "one"},
            {"role": "assistant", "content": "ok"},
        ]}
        mutated = {"messages": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "one EDITED"},
            {"role": "assistant", "content": "ok"},
        ]}
        idx = divergence_index(
            rolling_hashes(prefix_elements(base)),
            rolling_hashes(prefix_elements(mutated)),
        )
        assert idx == 1

    def test_a_toolset_change_invalidates_from_byte_zero(self):
        msgs = [{"role": "system", "content": "SYS"}]
        before = rolling_hashes(prefix_elements({"tools": [{"name": "a"}], "messages": msgs}))
        after = rolling_hashes(prefix_elements({"tools": [{"name": "a"}, {"name": "b"}],
                                                "messages": msgs}))
        assert divergence_index(before, after) == 0

    def test_dict_key_order_is_not_a_divergence(self):
        """Python does not guarantee dict order across rebuilds and no chat
        template depends on it — key order must never read as real drift."""
        a = prefix_elements({"tools": [{"x": 1, "y": 2}],
                             "messages": [{"role": "user", "content": "hi"}]})
        b = prefix_elements({"messages": [{"content": "hi", "role": "user"}],
                             "tools": [{"y": 2, "x": 1}]})
        assert divergence_index(rolling_hashes(a), rolling_hashes(b)) is None

    def test_sampling_params_are_not_part_of_the_prefix(self):
        msgs = [{"role": "user", "content": "hi"}]
        a = prefix_elements({"messages": msgs, "temperature": 0.0, "max_tokens": 10})
        b = prefix_elements({"messages": msgs, "temperature": 0.9, "max_tokens": 4096})
        assert divergence_index(rolling_hashes(a), rolling_hashes(b)) is None

    def test_responses_style_input_is_understood(self):
        els = prefix_elements({"input": [{"role": "user", "content": "hi"}]})
        assert [label for label, _ in els] == ["user"]

    def test_state_survives_a_new_agent_object(self, tmp_path, monkeypatch):
        """The crash / app-update / fresh-per-turn-agent case: the comparison
        must still happen when the agent that made call N is gone."""
        monkeypatch.setenv("HERMES_PREFIX_PROBE", "1")

        class _Agent:
            logs_dir = str(tmp_path)
            session_id = "sess-1"

        first = record_request_prefix(_Agent(), {"messages": [{"role": "user", "content": "a"}]})
        assert first["divergence_index"] is None and first["call"] == 1

        # A brand-new agent object, same session.
        second = record_request_prefix(_Agent(), {"messages": [
            {"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]})
        assert second["call"] == 2
        assert second["divergence_index"] is None

        third = record_request_prefix(_Agent(), {"messages": [
            {"role": "user", "content": "CHANGED"}, {"role": "assistant", "content": "b"}]})
        assert third["divergence_index"] == 0

    def test_never_raises_into_the_request_path(self, monkeypatch):
        monkeypatch.setenv("HERMES_PREFIX_PROBE", "1")

        class _Exploding:
            @property
            def logs_dir(self):
                raise RuntimeError("boom")
            session_id = "s"

        # Unserializable payload + an agent whose every attribute explodes.
        assert record_request_prefix(_Exploding(), {"messages": [{"role": "user",
                                                                  "content": object()}]}) is None

    def test_disabled_probe_does_no_work(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HERMES_PREFIX_PROBE", raising=False)

        class _Agent:
            logs_dir = str(tmp_path)
            session_id = "sess-off"

        assert record_request_prefix(_Agent(), {"messages": [{"role": "user", "content": "a"}]}) is None
        assert list(tmp_path.glob("prefix_probe_*")) == []


# ---------------------------------------------------------------------------
# End-to-end: a real AIAgent, a real HTTP provider, many turns
# ---------------------------------------------------------------------------

class _MockHandler(BaseHTTPRequestHandler):
    captured_requests: list = []
    response_queue: list = []

    def do_POST(self):  # noqa: N802 (http.server API)
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length).decode())
        type(self).captured_requests.append(req)
        resp = (type(self).response_queue.pop(0)
                if type(self).response_queue else _text_resp("DONE"))
        msg = resp["choices"][0]["message"]
        if req.get("stream") is True:
            content = msg.get("content") or ""
            tcs = msg.get("tool_calls")
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [{"id": "m", "choices": [{"index": 0, "delta": {
                "role": "assistant", "content": ""}, "finish_reason": None}]}]
            if content:
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {
                    "content": content}, "finish_reason": None}]})
            for ti, tc in enumerate(tcs or []):
                chunks.append({"id": "m", "choices": [{"index": 0, "delta": {"tool_calls": [{
                    "index": ti, "id": tc["id"], "type": "function",
                    "function": {"name": tc["function"]["name"],
                                 "arguments": tc["function"]["arguments"]}}]},
                    "finish_reason": None}]})
            chunks.append({"id": "m", "choices": [{"index": 0, "delta": {},
                          "finish_reason": "tool_calls" if tcs else "stop"}]})
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        else:
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def log_message(self, *a, **kw):
        pass


def _tc_resp(name: str, args: str = "{}") -> dict:
    return {"id": "m", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "call_1", "type": "function",
                        "function": {"name": name, "arguments": args}}]},
        "finish_reason": "tool_calls"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}


def _text_resp(text: str) -> dict:
    return {"id": "m", "choices": [{"index": 0, "message": {
        "role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}}


@pytest.fixture()
def wire_env():
    """Mock provider + isolated HERMES_HOME + a shared SessionDB.

    ``make_agent()`` builds a fresh AIAgent bound to the same DB/session, so
    calling it per turn models the process-restart path rather than a
    long-lived in-memory object.
    """
    _MockHandler.captured_requests = []
    _MockHandler.response_queue = []
    srv = HTTPServer(("127.0.0.1", 0), _MockHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    test_home = tempfile.mkdtemp(prefix="hermes_prefix_stability_")
    os.makedirs(os.path.join(test_home, ".hermes"))
    prev_home = os.environ.get("HERMES_HOME")
    os.environ["HERMES_HOME"] = os.path.join(test_home, ".hermes")

    from run_agent import AIAgent

    db = SessionDB(db_path=Path(test_home) / "state.db")
    sid = "sess-prefix"

    def make_agent():
        agent = AIAgent(
            api_key="test-key", base_url=f"http://127.0.0.1:{port}/v1",
            provider="openai-compat", model="test-model",
            max_iterations=10, enabled_toolsets=[],
            quiet_mode=True, skip_context_files=True, skip_memory=True,
            save_trajectories=False, platform="cli",
            session_db=db, session_id=sid,
        )
        agent.valid_tool_names = {"read_file"}
        return agent

    try:
        yield make_agent, _MockHandler, db, sid
    finally:
        srv.shutdown()
        db.close()
        shutil.rmtree(test_home, ignore_errors=True)
        if prev_home is None:
            os.environ.pop("HERMES_HOME", None)
        else:
            os.environ["HERMES_HOME"] = prev_home


def _chat_requests(handler) -> list:
    # The model context-length probe also hits the mock; keep only
    # chat-completions payloads.
    return [r for r in handler.captured_requests if "messages" in r]


def _first_divergence(earlier: dict, later: dict):
    return divergence_index(
        rolling_hashes(prefix_elements(earlier)),
        rolling_hashes(prefix_elements(later)),
    )


class TestMultiTurnPrefixIsAppendOnly:
    TURNS = 12

    def _run(self, make_agent, handler, db, sid, *, fresh_agent_each_turn):
        """Drive TURNS turns the way the product actually does.

        ``fresh_agent_each_turn`` picks the path: a rebuilt agent reloading
        history from the store (desktop/gateway — and the crash/update case),
        or one long-lived agent carrying the returned message list (CLI).
        """
        agent = None
        history: list = []
        for turn in range(self.TURNS):
            # Every other turn does a tool call, so the transcript grows by
            # assistant+tool rows too — not just user/assistant pairs.
            handler.response_queue = (
                [_tc_resp("read_file", '{"file_path": "/x"}'), _text_resp(f"reply {turn}")]
                if turn % 2 == 0 else [_text_resp(f"reply {turn}")]
            )
            if fresh_agent_each_turn or agent is None:
                agent = make_agent()
            if fresh_agent_each_turn:
                history = db.get_messages_as_conversation(sid)
            with patch("model_tools.handle_function_call",
                       return_value=json.dumps({"success": True, "content": f"file {turn}"})):
                result = agent.run_conversation(
                    f"turn {turn} please",
                    conversation_history=history,
                    task_id=f"t{turn}",
                )
            if not fresh_agent_each_turn:
                history = list(result.get("messages") or [])
        return _chat_requests(handler)

    @pytest.mark.parametrize("fresh_agent_each_turn", [False, True],
                             ids=["one-long-lived-agent", "fresh-agent-per-turn"])
    def test_every_request_extends_the_previous_one(
        self, wire_env, fresh_agent_each_turn
    ):
        """(a) The serialized prefix through call N-1 is byte-identical when
        call N is built — for a long-lived agent AND for a rebuilt one."""
        make_agent, handler, db, sid = wire_env
        requests = self._run(make_agent, handler, db, sid,
                             fresh_agent_each_turn=fresh_agent_each_turn)

        assert len(requests) >= self.TURNS, requests
        offenders = []
        for i in range(1, len(requests)):
            idx = _first_divergence(requests[i - 1], requests[i])
            if idx is not None:
                label = prefix_elements(requests[i])[idx][0]
                offenders.append(f"call {i}: diverged at element {idx} ({label})")
        assert not offenders, "\n".join(offenders)

    def test_the_transcript_actually_grew(self, wire_env):
        """Guard against the assertion above passing vacuously."""
        make_agent, handler, db, sid = wire_env
        requests = self._run(make_agent, handler, db, sid, fresh_agent_each_turn=True)
        first = len(prefix_elements(requests[0]))
        last = len(prefix_elements(requests[-1]))
        assert last > first + self.TURNS, (first, last)

    def test_system_prompt_bytes_are_identical_across_turns(self, wire_env):
        """Hypothesis (a) directly: no per-turn drift in the system payload —
        no timestamp, git snapshot, probe race or reordered tool schema."""
        make_agent, handler, db, sid = wire_env
        requests = self._run(make_agent, handler, db, sid, fresh_agent_each_turn=True)
        systems = {
            json.dumps(r["messages"][0], sort_keys=True)
            for r in requests if r.get("messages")
        }
        assert len(systems) == 1, f"{len(systems)} distinct system messages across turns"

    def test_tool_schema_bytes_are_identical_across_turns(self, wire_env):
        make_agent, handler, db, sid = wire_env
        requests = self._run(make_agent, handler, db, sid, fresh_agent_each_turn=True)
        schemas = {json.dumps(r.get("tools"), sort_keys=True) for r in requests}
        assert len(schemas) == 1, f"{len(schemas)} distinct tool schemas across turns"


class TestGoalContinuationOnlyAppends:
    def test_continuation_prompt_enters_as_a_new_trailing_message(self, wire_env):
        """Hypothesis (c): the goal loop's continuation must not touch history.

        Uses the real template rather than a stand-in string, so a change to
        how the continuation is composed is caught here.
        """
        from hermes_cli.goals import CONTINUATION_PROMPT_TEMPLATE

        make_agent, handler, db, sid = wire_env
        agent = make_agent()

        handler.response_queue = [_text_resp("did a step")]
        agent.run_conversation("start the goal", conversation_history=[], task_id="g1")

        continuation = CONTINUATION_PROMPT_TEMPLATE.format(goal="ship the parser")
        assert continuation.startswith("[Continuing toward your standing goal]")

        handler.response_queue = [_text_resp("did another step")]
        make_agent().run_conversation(
            continuation,
            conversation_history=db.get_messages_as_conversation(sid),
            task_id="g2",
        )

        requests = _chat_requests(handler)
        assert _first_divergence(requests[-2], requests[-1]) is None
        assert requests[-1]["messages"][-1]["role"] == "user"
        assert "standing goal" in json.dumps(requests[-1]["messages"][-1]["content"])
