"""Shared fixtures and doubles for the auxiliary-client test package."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""

import base64
import json
from unittest.mock import MagicMock

import pytest



@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip provider env vars so each test starts clean.

    This is the union of two ``_clean_env`` fixtures that used to live in
    ``test_auxiliary_client.py``. The second definition shadowed the first
    for the entire module, so the whole file silently ran without the
    unhealthy-cache scrubbing below — which the first one documents as
    load-bearing. Keeping them merged is what stops that from recurring.
    """
    for key in (
        "OPENROUTER_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_KEY",
        "OPENAI_MODEL", "LLM_MODEL", "NOUS_INFERENCE_BASE_URL",
        "ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN",
        "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    # Module-level unhealthy cache (10-min TTL) leaks between tests;
    # earlier tests that call _mark_provider_unhealthy() poison the
    # cache for later ones, causing _resolve_auto to skip providers
    # that the test patched to return valid clients.
    import agent.auxiliary_client as _aux_mod
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()
    yield
    _aux_mod._aux_unhealthy_until.clear()
    _aux_mod._aux_unhealthy_logged_at.clear()


@pytest.fixture
def codex_auth_dir(tmp_path, monkeypatch):
    """Provide a writable ~/.codex/ directory with a valid auth.json."""
    codex_dir = tmp_path / ".codex"
    codex_dir.mkdir()
    auth_file = codex_dir / "auth.json"
    auth_file.write_text(json.dumps({
        "tokens": {
            "access_token": "codex-test-token-abc123",
            "refresh_token": "codex-refresh-xyz",
        }
    }))
    monkeypatch.setattr(
        "agent.auxiliary_client._read_codex_access_token",
        lambda: "codex-test-token-abc123",
    )
    return codex_dir


def _jwt_with_claims(claims: dict) -> str:
    header = base64.urlsafe_b64encode(b'{"alg":"none","typ":"JWT"}').decode().rstrip("=")
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).decode().rstrip("=")
    return f"{header}.{payload}.sig"


class _FakeAnthropicStream:
    def __init__(self, final_message):
        self._final_message = final_message

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_final_message(self):
        return self._final_message


class _AuxAuth401(Exception):
    status_code = 401

    def __init__(self, message="Provided authentication token is expired"):
        super().__init__(message)


class _DummyResponse:
    def __init__(self, text="ok"):
        self.choices = [MagicMock(message=MagicMock(content=text))]


class _FailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("sync-ok")


class _AsyncFailingThenSuccessCompletions:
    def __init__(self):
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls == 1:
            raise _AuxAuth401()
        return _DummyResponse("async-ok")
