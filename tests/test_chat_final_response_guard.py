"""Tests for AIAgent.chat() final_response guard.

run_agent.py:chat() must honor its ``-> str`` contract even when
run_conversation returns an early-exit / error dict that omits
``"final_response"`` (interrupted, retries exhausted, policy/billing bail).
Before the guard this raised ``KeyError: 'final_response'``, which crashed the
dispatched ``hermes -z`` goal worker and was misclassified upstream as a
harness fault instead of a clean "no final response → failed run".
"""

from unittest.mock import patch

from run_agent import AIAgent


def _agent_without_init() -> AIAgent:
    # Bypass __init__ (heavy: providers/config/IO). We only exercise chat()'s
    # handling of the run_conversation return dict.
    return AIAgent.__new__(AIAgent)


def test_chat_returns_empty_string_when_final_response_key_missing():
    agent = _agent_without_init()
    with patch.object(agent, "run_conversation", return_value={"completed": False, "api_calls": 1}):
        assert agent.chat("hi") == ""


def test_chat_returns_empty_string_when_final_response_is_none():
    agent = _agent_without_init()
    with patch.object(agent, "run_conversation", return_value={"final_response": None}):
        assert agent.chat("hi") == ""


def test_chat_returns_final_response_when_present():
    agent = _agent_without_init()
    with patch.object(agent, "run_conversation", return_value={"final_response": "the answer"}):
        assert agent.chat("hi") == "the answer"
