"""Regression tests for the finish_reason="length" continuation livelock.

The bug: on ``finish_reason="length"`` with no tool calls the loop appended the
partial assistant message plus a "continue" user prompt and re-issued the call,
up to four times. That is correct when the OUTPUT cap was the binding
constraint — but when the CONTEXT WINDOW is the binding constraint every round
made it strictly worse (two more messages into an already-full window), the
model returned another empty/tiny truncation, and the turn died after burning
four paid attempts with "Response remained truncated after 4 continuation
attempts". Nothing coupled the truncation counter to compaction, and nothing
detected that the continuations were producing no new content.

Pins the fix:

- head-room guard: with the window essentially full, the loop compacts through
  the existing context-compressor instead of scaffolding another round;
- non-convergence guard: continuations that return ~no new content stop the
  loop as soon as that is proven, instead of burning the remaining attempts;
- when compaction is unavailable, the turn fails fast with an honest,
  actionable message (/compress, /new) rather than pretending success;
- the legitimate case — a real max-output truncation with plenty of context
  left — still gets its normal continuations.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from run_agent import AIAgent


CONTEXT_LENGTH = 131_072
# ~89% of the window is already prompt — the state the livelock was reported in.
PROMPT_TOKENS_AT_89_PCT = int(CONTEXT_LENGTH * 0.89)


def _mock_response(content, finish_reason="length"):
    msg = SimpleNamespace(
        content=content,
        tool_calls=None,
        reasoning=None,
        reasoning_content=None,
        reasoning_details=None,
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(message=msg, finish_reason=finish_reason)],
        model="test/model",
        usage=None,
    )


@pytest.fixture()
def agent():
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI"),
    ):
        a = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://example.com/v1",
            model="test/model",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    a.client = MagicMock()
    a._cached_system_prompt = "You are helpful."
    a._use_prompt_caching = False
    a.tool_delay = 0
    a.save_trajectories = False
    a.tools = []
    return a


def _pin_context(agent, *, prompt_tokens, context_length=CONTEXT_LENGTH):
    """Make the compressor report a real, nearly-full context window."""
    compressor = agent.context_compressor
    compressor.context_length = context_length
    compressor.last_real_prompt_tokens = prompt_tokens
    compressor.last_prompt_tokens = prompt_tokens
    compressor.last_completion_tokens = 0
    return compressor


_HISTORY = [
    {"role": "user", "content": "earlier question"},
    {"role": "assistant", "content": "earlier answer"},
]


class TestLengthContinuationLivelock:
    def test_no_progress_continuation_compacts_instead_of_burning_attempts(self, agent):
        """~89% context + always-truncating provider with tiny content →
        the loop compacts via the existing compressor rather than scaffolding
        four continuation rounds."""
        _pin_context(agent, prompt_tokens=PROMPT_TOKENS_AT_89_PCT)
        agent.compression_enabled = True

        agent.client.chat.completions.create.side_effect = [
            _mock_response("tok", finish_reason="length"),
            _mock_response("tok", finish_reason="length"),
            _mock_response("tok", finish_reason="length"),
            _mock_response("all done", finish_reason="stop"),
        ]

        compressed = []

        def _fake_compress(messages, system_message, **kwargs):
            compressed.append(list(messages))
            # The real compressor returns (messages, system_prompt).
            return (
                [{"role": "user", "content": "[compacted history] earlier question"}],
                system_message,
            )

        with (
            patch.object(agent, "_compress_context", side_effect=_fake_compress),
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_HISTORY))

        assert len(compressed) == 1, (
            "The truncation path must route to the existing compaction "
            "machinery once the continuations stop converging."
        )
        # The compacted transcript must not carry the half-finished
        # continuation scaffolding (partial assistant + "[System: ...continue]").
        rolled_back = compressed[0]
        assert not any(
            isinstance(m.get("content"), str)
            and m["content"].startswith("[System: Your previous response was truncated")
            for m in rolled_back
        ), "continuation scaffolding must be rolled back before compaction"

        # Livelock signature (4 truncation attempts then a hard error) is gone.
        assert result.get("error") != (
            "Response remained truncated after 4 continuation attempts"
        )
        assert result["completed"] is True
        assert result["final_response"] == "all done"

    def test_fails_fast_and_honestly_when_compaction_unavailable(self, agent):
        """Same livelock, but with compression off: fail fast with an honest,
        actionable message instead of burning all four attempts."""
        _pin_context(agent, prompt_tokens=PROMPT_TOKENS_AT_89_PCT)
        agent.compression_enabled = False

        agent.client.chat.completions.create.return_value = _mock_response(
            "tok", finish_reason="length"
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_HISTORY))

        # Initial truncation + two non-converging continuations — not four.
        assert result["api_calls"] == 3
        assert result["completed"] is False
        assert result["partial"] is True
        assert "no new content" in result["error"]
        # Honest and actionable, not a silent success.
        assert "/compress" in result["final_response"]
        assert "/new" in result["final_response"]

    def test_window_exhausted_refuses_the_very_first_continuation(self, agent):
        """When the window has no room left at all, do not scaffold even the
        first continuation round — compaction is the only move that helps."""
        # 1k tokens of head-room: below _LENGTH_CONTINUE_MIN_HEADROOM_TOKENS.
        _pin_context(agent, prompt_tokens=CONTEXT_LENGTH - 1_000)
        agent.compression_enabled = False

        agent.client.chat.completions.create.return_value = _mock_response(
            "a genuinely long partial answer that is still cut off mid-sentence",
            finish_reason="length",
        )

        with (
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_HISTORY))

        assert result["api_calls"] == 1, (
            "with no head-room left, a continuation round cannot make progress"
        )
        assert result["completed"] is False
        assert "Context window exhausted" in result["error"]
        assert "/compress" in result["final_response"]
        # The partial text the model did produce is still surfaced.
        assert "cut off mid-sentence" in result["final_response"]

    def test_real_truncation_with_headroom_still_continues_normally(self, agent):
        """Guard against over-firing: a model that hit its OUTPUT cap with
        plenty of context left must still get its normal continuation."""
        _pin_context(agent, prompt_tokens=10_000)  # ~121k tokens of head-room
        agent.compression_enabled = True

        agent.client.chat.completions.create.side_effect = [
            _mock_response("Part 1 " + "x" * 500, finish_reason="length"),
            _mock_response("Part 2", finish_reason="stop"),
        ]

        with (
            patch.object(agent, "_compress_context") as compress,
            patch.object(agent, "_persist_session"),
            patch.object(agent, "_save_trajectory"),
            patch.object(agent, "_cleanup_task_resources"),
        ):
            result = agent.run_conversation("hello", conversation_history=list(_HISTORY))

        compress.assert_not_called()
        assert result["completed"] is True
        assert result["api_calls"] == 2
        assert result["final_response"] == "Part 1 " + "x" * 500 + "Part 2"

        second_call = agent.client.chat.completions.create.call_args_list[1]
        assert second_call.kwargs["messages"][-1]["role"] == "user"
        assert "truncated by the output length limit" in (
            second_call.kwargs["messages"][-1]["content"]
        )
