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

# ~115k tokens in a 131,072 window: past the point where the multiplicative
# input reservation (1.2x + 1536) exhausts the window, but the prompt itself
# still leaves ~16k tokens of genuine headroom.  This is the regime where the
# old ``min_output`` floor turned "nothing fits" into "1 token fits".
_NEAR_FULL = [{"role": "user", "content": "x" * 460000}]


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


class TestNearFullWindowNeverYieldsAUselessCap:
    """A near-full window must report "no usable fit", never a 1-token floor.

    ``output_tokens_that_fit`` reserves the input multiplicatively (1.2x), so
    once the estimate passes ~82.7% of the window the reservation swallows it
    whole.  Returning the ``min_output`` floor there manufactured a "1 output
    token fits" budget out of "nothing fits", and both callers consumed it as a
    real number: the provider accepts a max_tokens=1 retry and returns a single
    truncated token, so the turn *succeeds* with output the user cannot use.
    """

    def test_prompt_itself_still_leaves_real_headroom(self):
        # Guard the fixture: this must be the "reservation exhausted the window"
        # case, not the "prompt genuinely does not fit" case.
        est = estimate_messages_tokens_rough(_NEAR_FULL)
        assert est < _CTX
        assert _CTX - est > 10_000

    def test_reports_none_rather_than_a_floor(self):
        assert output_tokens_that_fit(_CTX, _NEAR_FULL) is None

    def test_every_reported_fit_is_a_real_fit(self):
        # Sweep the whole fill range and hold every non-None answer to the same
        # safety property the function documents: a server tokenizing ~15%
        # denser than our estimate must still leave the clamped request
        # in-window.  A fabricated floor fails this — it is precisely a number
        # that does NOT fit — so this catches the regression at its root rather
        # than by pattern-matching the literal 1.
        for tokens in range(1_000, _CTX, 2_500):
            msgs = [{"role": "user", "content": "x" * (tokens * 4)}]
            fit = output_tokens_that_fit(_CTX, msgs)
            if fit is None:
                continue
            server_input = int(estimate_messages_tokens_rough(msgs) * 1.15)
            assert server_input + fit < _CTX, (
                f"{tokens=} reported fit={fit} that does not actually fit"
            )

    def test_preflight_leaves_cap_alone_instead_of_clamping_to_one(self):
        a = _agent(max_tokens=65536)
        _preflight_clamp_output_tokens(a, _NEAR_FULL)
        # No clamp at all: the provider gets the real cap, 400s, and the
        # reactive path then uses the provider's authoritative budget.
        assert a._ephemeral_max_output_tokens is None

    def test_reactive_retry_keeps_provider_authoritative_cap(self):
        # Mirrors the arithmetic in conversation_loop's output-cap retry.
        provider_available = 25_000
        local_available_out = _CTX - estimate_messages_tokens_rough(_NEAR_FULL)
        safe_out = max(1, min(provider_available, local_available_out) - 64)
        local_fit = output_tokens_that_fit(_CTX, _NEAR_FULL)
        if local_fit is not None:
            safe_out = max(1, min(safe_out, local_fit))
        # The retry must stay anchored to what the provider said actually fits.
        assert safe_out > 1_000
        assert safe_out == max(1, min(provider_available, local_available_out) - 64)
