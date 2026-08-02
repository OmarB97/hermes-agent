"""Auxiliary-client error-shape predicates (_is_*_error)."""
"""Tests for agent.auxiliary_client resolution chain, provider overrides, and model overrides."""



from agent.auxiliary_client import (
    _is_payment_error,
    _is_rate_limit_error,
    _is_model_not_found_error,
    _is_model_incompatible_error,
)



class TestIsPaymentError:
    """_is_payment_error detects 402 and credit-related errors."""

    def test_402_status_code(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_402_with_credits_message(self):
        exc = Exception("You requested up to 65535 tokens, but can only afford 8029")
        exc.status_code = 402
        assert _is_payment_error(exc) is True

    def test_429_with_credits_message(self):
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_404_free_tier_model_block_is_payment(self):
        exc = Exception(
            "Model 'gpt-5' is not available on the Free Tier. "
            "Upgrade at https://portal.nousresearch.com or pick a free model."
        )
        exc.status_code = 404
        assert _is_payment_error(exc) is True

    def test_403_subscription_required_is_payment(self):
        exc = Exception(
            "this model requires a subscription, upgrade for access: "
            "https://ollama.com/upgrade"
        )
        setattr(exc, "status_code", 403)
        assert _is_payment_error(exc) is True

    def test_429_session_usage_limit_is_payment(self):
        exc = Exception(
            "you have reached your session usage limit, upgrade for higher limits"
        )
        setattr(exc, "status_code", 429)
        assert _is_payment_error(exc) is True

    def test_404_generic_not_found_is_not_payment(self):
        exc = Exception("Not Found")
        exc.status_code = 404
        assert _is_payment_error(exc) is False

    def test_429_without_credits_message_is_not_payment(self):
        """Normal rate limits should NOT be treated as payment errors."""
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_payment_error(exc) is False

    def test_generic_500_is_not_payment(self):
        exc = Exception("Internal server error")
        exc.status_code = 500
        assert _is_payment_error(exc) is False

    def test_no_status_code_with_billing_message(self):
        exc = Exception("billing: payment required for this request")
        assert _is_payment_error(exc) is True

    def test_no_status_code_no_message(self):
        exc = Exception("connection reset")
        assert _is_payment_error(exc) is False

    # ── Daily / monthly quota exhaustion (#26803) ────────────────────────────

    def test_429_quota_exceeded(self):
        """Cloud provider quota exhaustion (e.g. Vertex AI) is a payment error."""
        exc = Exception("RESOURCE_EXHAUSTED: quota exceeded for project")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_too_many_tokens_per_day(self):
        """Bedrock / LiteLLM daily token limit is a payment error."""
        exc = Exception("Too many tokens per day: 1000000 used, 1000000 limit")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_daily_limit_phrase(self):
        """Generic 'daily limit' phrasing is a payment error."""
        exc = Exception("You have exceeded your daily limit.")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_resource_exhausted_grpc(self):
        """Vertex AI gRPC RESOURCE_EXHAUSTED maps to payment error."""
        exc = Exception("resource exhausted")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_daily_quota_phrase(self):
        """'daily quota' phrasing is a payment error."""
        exc = Exception("Daily quota of 500 requests reached.")
        exc.status_code = 429
        assert _is_payment_error(exc) is True

    def test_429_transient_rate_limit_not_quota(self):
        """Transient 429 rate limit without quota keywords is NOT a payment error."""
        exc = Exception("Rate limit exceeded. Retry after 10s.")
        exc.status_code = 429
        assert _is_payment_error(exc) is False


class TestIsModelNotFoundError:
    """_is_model_not_found_error detects stale/invalid model 404s, distinct
    from payment errors."""

    def test_nous_openrouter_catalog_404(self):
        """The exact incident error: a Portal-recommended model dropped from
        the Nous → OpenRouter catalog."""
        exc = Exception(
            "Model 'gpt-5.4-mini' not found. The requested model does not "
            "exist in our configuration or OpenRouter catalog."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is True

    def test_openai_style_model_does_not_exist(self):
        exc = Exception("The model `gpt-9-turbo` does not exist")
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is True

    def test_invalid_model_id_400(self):
        exc = Exception("openrouter/foo/bar is not a valid model ID")
        exc.status_code = 400
        assert _is_model_not_found_error(exc) is True

    def test_no_such_model(self):
        exc = Exception("no such model: phantom-v1")
        exc.status_code = 400
        assert _is_model_not_found_error(exc) is True

    def test_billing_404_is_not_model_not_found(self):
        """Free-tier / credit 404s belong to _is_payment_error, not here —
        the two predicates must not overlap."""
        exc = Exception(
            "Model 'gpt-5' is not available on the free tier. Upgrade."
        )
        exc.status_code = 404
        assert _is_model_not_found_error(exc) is False
        assert _is_payment_error(exc) is True

    def test_out_of_funds_404_is_not_model_not_found(self):
        exc = Exception(
            "Your API key is blocked or out of funds. model_not_found"
        )
        exc.status_code = 404
        # billing keyword wins — payment owns it
        assert _is_model_not_found_error(exc) is False

    def test_rate_limit_is_not_model_not_found(self):
        exc = Exception("rate limit exceeded, retry after 5s")
        exc.status_code = 429
        assert _is_model_not_found_error(exc) is False

    def test_500_is_not_model_not_found(self):
        exc = Exception("model does not exist")  # right phrase, wrong status
        exc.status_code = 500
        assert _is_model_not_found_error(exc) is False


class TestIsModelIncompatibleError:
    """_is_model_incompatible_error detects 400s where the route cannot run
    the model at all (capability mismatch), distinct from not-found and
    payment errors."""

    def test_codex_chatgpt_account_model_gating(self):
        """The exact incident: an openai-codex/ChatGPT-account fallback asked
        to compress a glm-5.2 conversation."""
        exc = Exception(
            "Error code: 400 - {'detail': \"The 'glm-5.2' model is not "
            "supported when using Codex with a ChatGPT account.\"}"
        )
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_model_is_not_supported_phrasing(self):
        exc = Exception("This model is not supported for this endpoint")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_unsupported_model_keyword(self):
        exc = Exception("unsupported model for this account tier")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is True

    def test_not_found_is_not_incompatible(self):
        """A model-does-not-exist 400 belongs to _is_model_not_found_error —
        the two predicates must not overlap."""
        exc = Exception("openrouter/foo/bar is not a valid model ID")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False
        assert _is_model_not_found_error(exc) is True

    def test_payment_400_is_not_incompatible(self):
        """A billing 400 that also contains capability-ish phrasing must be
        rejected here — billing keywords win so the payment path owns it and
        the two buckets don't overlap."""
        exc = Exception("insufficient credits: model is not supported on free tier")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False

    def test_wrong_status_is_not_incompatible(self):
        exc = Exception("model is not supported")  # right phrase, wrong status
        exc.status_code = 500
        assert _is_model_incompatible_error(exc) is False

    def test_generic_400_is_not_incompatible(self):
        """A plain request-validation 400 without capability phrasing must not
        trigger fallback (we respect explicit-provider choice for those)."""
        exc = Exception("invalid value for parameter temperature")
        exc.status_code = 400
        assert _is_model_incompatible_error(exc) is False


class TestIsRateLimitError:
    """_is_rate_limit_error detects 429 rate-limit errors warranting fallback."""

    def test_429_with_rate_limit_message(self):
        exc = Exception("Rate limit exceeded, try again in 2 seconds")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_resets_in_message(self):
        """Nous-style 429: 'resets in 3508s'."""
        exc = Exception("Hold up for a bit, you've exceeded the rate limit on your API key")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_too_many_requests(self):
        exc = Exception("Too many requests")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_without_billing_keywords_is_rate_limit(self):
        """Generic 429 without billing keywords = likely a rate limit."""
        exc = Exception("Something went wrong")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is True

    def test_429_with_credits_message_is_not_rate_limit(self):
        """Billing-related 429 should NOT be classified as rate limit."""
        exc = Exception("insufficient credits remaining")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_429_with_billing_message_is_not_rate_limit(self):
        exc = Exception("you can only afford 1000 tokens")
        exc.status_code = 429
        assert _is_rate_limit_error(exc) is False

    def test_402_is_not_rate_limit(self):
        exc = Exception("Payment Required")
        exc.status_code = 402
        assert _is_rate_limit_error(exc) is False

    def test_500_is_not_rate_limit(self):
        exc = Exception("Internal Server Error")
        exc.status_code = 500
        assert _is_rate_limit_error(exc) is False

    def test_openai_ratelimiterror_classname(self):
        """OpenAI SDK RateLimitError may omit .status_code — detect by class name."""
        class RateLimitError(Exception):
            pass
        exc = RateLimitError("rate limit exceeded")
        # No status_code set, but class name matches
        assert _is_rate_limit_error(exc) is True

    def test_no_status_code_no_keywords_is_not_rate_limit(self):
        exc = Exception("connection reset")
        assert _is_rate_limit_error(exc) is False


class TestIsTimeoutError:
    """_is_timeout_error distinguishes a full-budget timeout from a fast
    connection drop."""

    def test_timed_out_string(self):
        from agent.auxiliary_client import _is_timeout_error
        assert _is_timeout_error(Exception("Request timed out.")) is True

    def test_timeout_typename(self):
        from agent.auxiliary_client import _is_timeout_error

        class ReadTimeout(Exception):
            pass

        assert _is_timeout_error(ReadTimeout("slow")) is True

    def test_streaming_close_is_not_timeout(self):
        from agent.auxiliary_client import _is_timeout_error
        err = Exception("peer closed connection (incomplete chunked read)")
        assert _is_timeout_error(err) is False

    def test_5xx_is_not_timeout(self):
        from agent.auxiliary_client import _is_timeout_error

        class _Err503(Exception):
            status_code = 503

        assert _is_timeout_error(_Err503("upstream")) is False


class TestIsConnectionError:
    """Tests for _is_connection_error detection."""

    def test_connection_refused(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Connection refused")
        assert _is_connection_error(err) is True

    def test_timeout(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Request timed out.")
        assert _is_connection_error(err) is True

    def test_dns_failure(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Name or service not known")
        assert _is_connection_error(err) is True

    def test_normal_api_error_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Bad Request: invalid model")
        err.status_code = 400
        assert _is_connection_error(err) is False

    def test_500_not_connection(self):
        from agent.auxiliary_client import _is_connection_error
        err = Exception("Internal Server Error")
        err.status_code = 500
        assert _is_connection_error(err) is False
