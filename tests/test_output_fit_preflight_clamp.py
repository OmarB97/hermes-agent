"""Pre-flight output-cap clamp (prompt + max_tokens must fit the window).

Regression coverage for the deepseek-v4-flash-w2 death-loop: vLLM rejects any
request where prompt_tokens + max_tokens exceeds the context window and reports
the offending input as a max_tokens-dependent LOWER BOUND, so a reactive
shrink-and-retry never converges.  The fix sizes the output cap to fit BEFORE
sending (and anchors the reactive retry to the same estimate).
"""
from types import SimpleNamespace

from agent.chat_completion_helpers import _preflight_clamp_output_tokens
from agent.model_metadata import (
    _OUTPUT_FIT_MIN_USABLE,
    _OUTPUT_FIT_WINDOW_MARGIN,
    estimate_messages_tokens_rough,
    output_tokens_that_fit,
)


# A prompt large enough that a half-window output cap overflows a 128K window.
# estimate_messages_tokens_rough assumes ~4 chars/token, so 232k chars -> ~58k.
_BIG = [{"role": "user", "content": "x" * 232000}]
_SMALL = [{"role": "user", "content": "hello there"}]
_CTX = 131072

# ~115k tokens in a 131,072 window (87.7% fill): past the 1/1.15 ceiling, so the
# safety contract itself admits no usable cap — 1.15 * 115,008 already exceeds
# the whole window.  This is the regime where the old ``min_output`` floor turned
# "nothing fits" into "1 token fits".
_NEAR_FULL = [{"role": "user", "content": "x" * 460000}]

# A 200,000-token window, where the fill bands below are wide enough to name
# individual estimates.  ``_HIGH_FILL_TOKENS`` is the case from the report: the
# preferred 1.2x cushion admits nothing at all there, while the 1.15x safety
# contract still has thousands of output tokens to give.
_CTX_200K = 200_000
_HIGH_FILL_TOKENS = 166_008


def _msgs(tokens):
    """Messages whose rough estimate is ~``tokens`` (~4 chars/token)."""
    return [{"role": "user", "content": "x" * (tokens * 4)}]


def _fits_for_a_denser_server(msgs, fit, ctx):
    """The documented safety contract: a server tokenizing ~15% denser than our
    estimate must still leave the clamped request inside the window."""
    return int(estimate_messages_tokens_rough(msgs) * 1.15) + fit < ctx


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

    Past ~87% fill (the ``1/1.15`` ceiling) even the safety contract admits
    nothing: any positive cap would put a denser-tokenizing server over the
    window.  Returning the ``min_output`` floor there manufactured a "1 output
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


class TestHighFillBandDegradesGracefully:
    """The 82.7%-100% fill band: no cliff from a healthy cap straight to ``None``.

    The 1.2x reservation is an *uncertainty allowance* for servers that tokenize
    denser than the ~4 chars/token heuristic, not a real token cost — and near a
    full window that allowance (0.2 * est) exceeds the entire remaining headroom.
    Holding it there reported "nothing fits" while thousands of output tokens
    genuinely did, which switched the pre-flight clamp off for every turn in the
    band: each one sent the full configured max_tokens, ate a provider 400, and
    only then got a usable cap from the reactive retry.  A guaranteed extra
    round-trip per turn.

    So the reservation now degrades to the 1.15x safety contract instead of
    vanishing.  ``None`` is still the answer past the ``1/1.15`` ceiling, where
    nothing usable genuinely fits.
    """

    def test_reported_fit_where_the_preferred_cushion_admits_none(self):
        # The case from the report: at est=166,008 in a 200,000 window the 1.2x
        # reservation overruns the window outright, while the 1.15x contract
        # still admits ~9,000 output tokens.
        msgs = _msgs(_HIGH_FILL_TOKENS)
        est = estimate_messages_tokens_rough(msgs)
        assert int(est * 1.2) + 1536 > _CTX_200K  # preferred cushion: nothing
        fit = output_tokens_that_fit(_CTX_200K, msgs)
        assert fit is not None and fit > 5_000
        assert _fits_for_a_denser_server(msgs, fit, _CTX_200K)

    def test_band_is_covered_without_a_hole(self):
        # Sweep the whole band the old formula gave up on.  Every fill level
        # from the 1.2x limit (~82.7%) up to the 1.15x ceiling (~86.9%) must
        # report a real cap, and the caps must taper rather than jump to None.
        for pct in range(83, 87):
            msgs = _msgs(int(_CTX_200K * pct / 100))
            fit = output_tokens_that_fit(_CTX_200K, msgs)
            assert fit is not None, f"no cap reported at {pct}% fill"
            assert fit >= _OUTPUT_FIT_MIN_USABLE
            assert _fits_for_a_denser_server(msgs, fit, _CTX_200K)

    def test_none_only_when_nothing_usable_genuinely_fits(self):
        # The complement of the sweep above, and the property that keeps this
        # honest: wherever the function still declines to answer, the safety
        # contract really does admit nothing usable.  A cliff — declining while
        # thousands of tokens fit — fails here.  Stepped finely enough to land
        # inside the transition rather than skipping over it.
        #
        # The bound allows for ``_OUTPUT_FIT_WINDOW_MARGIN``: that slack is
        # deliberately held back from every answer, so the contract admitting
        # one margin's worth of tokens on top of the usable floor is still
        # "nothing to offer".  The bound is tight — swept over every estimate in
        # the window, the largest headroom ever declined is exactly this.
        slack = _OUTPUT_FIT_MIN_USABLE + _OUTPUT_FIT_WINDOW_MARGIN
        for tokens in range(int(_CTX_200K * 0.80), _CTX_200K, 500):
            msgs = _msgs(tokens)
            if output_tokens_that_fit(_CTX_200K, msgs) is not None:
                continue
            est = estimate_messages_tokens_rough(msgs)
            contract_max = _CTX_200K - int(est * 1.15) - 1
            assert contract_max <= slack, (
                f"est={est} reported None while {contract_max} tokens fit"
            )

    def test_no_reported_cap_is_too_small_to_use(self):
        # The original defect, swept across the band: a cap of a handful of
        # tokens is a truncated response the provider accepts, not a working
        # request.  Below the usable floor the answer must be None.
        for tokens in range(1_000, _CTX_200K, 500):
            fit = output_tokens_that_fit(_CTX_200K, _msgs(tokens))
            assert fit is None or fit >= _OUTPUT_FIT_MIN_USABLE

    def test_low_fill_reservation_is_unchanged(self):
        # The degraded tier must not loosen the band that already works: below
        # the 1.2x limit the answer is still exactly the preferred reservation.
        for pct in (10, 40, 70, 80):
            msgs = _msgs(int(_CTX_200K * pct / 100))
            est = estimate_messages_tokens_rough(msgs)
            expected = _CTX_200K - (int(est * 1.2) + 1024) - 512
            assert output_tokens_that_fit(_CTX_200K, msgs) == expected

    def test_preflight_clamp_fires_in_the_band(self):
        # The point of the exercise: the clamp is back on above ~82.7% fill, so
        # the turn no longer spends a round-trip discovering its own cap.
        a = _agent(max_tokens=65536, ctx=_CTX_200K)
        _preflight_clamp_output_tokens(a, _msgs(_HIGH_FILL_TOKENS))
        assert a._ephemeral_max_output_tokens is not None
        assert a._ephemeral_max_output_tokens >= _OUTPUT_FIT_MIN_USABLE
        assert a._ephemeral_max_output_tokens < 65536

    def test_reactive_retry_still_converges_in_one_step_in_the_band(self):
        # #271: vLLM reports the offending input as a max_tokens-dependent LOWER
        # bound, so available_out tracks whatever cap we just sent and a plain
        # shrink-and-retry never converges.  In the band the local anchor used to
        # be None; now it exists and pulls the retry straight to a fitting cap.
        msgs = _msgs(_HIGH_FILL_TOKENS)
        local_fit = output_tokens_that_fit(_CTX_200K, msgs)
        assert local_fit is not None
        for requested in (65536, 65471, 65406):
            available_out = requested - 1  # the lower bound, not the truth
            safe_out = max(1, available_out - 64)
            safe_out = max(1, min(safe_out, local_fit))
            assert safe_out == local_fit  # one step, from any starting cap
            assert _fits_for_a_denser_server(msgs, safe_out, _CTX_200K)
