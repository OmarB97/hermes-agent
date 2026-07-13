"""Pre-flight output-cap clamp (prompt + max_tokens must fit the window).

Regression coverage for the deepseek-v4-flash-w2 death-loop: vLLM rejects any
request where prompt_tokens + max_tokens exceeds the context window and reports
the offending input as a max_tokens-dependent LOWER BOUND, so a reactive
shrink-and-retry never converges.  The fix sizes the output cap to fit BEFORE
sending (and anchors the reactive retry to the same estimate).
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import _preflight_clamp_output_tokens
from agent.model_metadata import estimate_messages_tokens_rough, output_tokens_that_fit


# A prompt large enough that a half-window output cap overflows a 128K window.
# estimate_messages_tokens_rough assumes ~4 chars/token, so 232k chars -> ~58k.
_BIG = [{"role": "user", "content": "x" * 232000}]
_SMALL = [{"role": "user", "content": "hello there"}]
_CTX = 131072


def _agent(max_tokens=None, ephemeral=None, ctx=_CTX):
    return SimpleNamespace(
        max_tokens=max_tokens,
        _ephemeral_max_output_tokens=ephemeral,
        context_compressor=SimpleNamespace(context_length=ctx),
        _buffer_vprint=lambda _m: None,
    )


class TestOutputTokensThatFit:
    def test_none_window_returns_none(self):
        assert output_tokens_that_fit(None, _BIG) is None
        assert output_tokens_that_fit(0, _BIG) is None

    def test_fit_is_strictly_inside_window_against_real_tokenization(self):
        est = estimate_messages_tokens_rough(_BIG)
        fit = output_tokens_that_fit(_CTX, _BIG)
        assert fit is not None and fit > 0
        # Server tokenizes denser than our ~4 chars/token estimate; even a ~15%
        # higher real input count must still leave the clamped request in-window.
        server_input = int(est * 1.15)
        assert server_input + fit < _CTX

    def test_small_prompt_leaves_most_of_window(self):
        assert output_tokens_that_fit(_CTX, _SMALL) > 100_000


class TestPreflightClamp:
    def test_oversized_cap_is_clamped(self):
        a = _agent(max_tokens=65536)
        _preflight_clamp_output_tokens(a, _BIG)
        assert a._ephemeral_max_output_tokens is not None
        assert a._ephemeral_max_output_tokens < 65536

    def test_fitting_cap_is_left_untouched(self):
        a = _agent(max_tokens=65536)
        _preflight_clamp_output_tokens(a, _SMALL)
        assert a._ephemeral_max_output_tokens is None  # provider still gets 65536

    def test_provider_default_is_never_forced(self):
        a = _agent(max_tokens=None)
        _preflight_clamp_output_tokens(a, _BIG)
        assert a._ephemeral_max_output_tokens is None

    def test_existing_ephemeral_is_reclamped(self):
        a = _agent(max_tokens=8192, ephemeral=65471)
        _preflight_clamp_output_tokens(a, _BIG)
        assert a._ephemeral_max_output_tokens < 65471

    def test_unknown_window_is_noop(self):
        a = _agent(max_tokens=65536, ctx=None)
        _preflight_clamp_output_tokens(a, _BIG)
        assert a._ephemeral_max_output_tokens is None


class TestReactiveRetryConverges:
    """A lower-bound overflow error (available_out tracks the sent cap) must
    still converge because we anchor to the local fit estimate."""

    def test_converges_in_one_step(self):
        # vLLM lower bound: available_out == requested - 1 for each retry.
        for requested in (65536, 65471, 65406):
            available_out = requested - 1
            safe_out = max(1, available_out - 64)
            local_fit = output_tokens_that_fit(_CTX, _BIG)
            safe_out = max(1, min(safe_out, local_fit))
            # Without the anchor this would be requested-65 (barely shrinks);
            # with it, it drops straight to the fitting cap.
            assert safe_out <= local_fit
            assert safe_out < requested - 1000
