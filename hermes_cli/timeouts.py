from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _recover_custom_provider_key(
    provider_id: str, base_url: str | None
) -> str | None:
    """Map a bare-``custom`` runtime provider back to its ``providers:`` key.

    ``resolve_runtime_provider()`` reports EVERY user-declared endpoint as the
    bare string ``"custom"`` — that is the resolved *billing class*, not a
    routable identity (see ``runtime_provider.canonical_custom_identity``,
    whose docstring already requires every persist/restore path to run a bare
    ``custom`` back through it).

    Timeout resolution is such a path and was missing the step, so a session on
    a declared endpoint looked up ``providers["custom"]`` — a key that by
    construction never exists — and reported "nothing configured". The user's
    ``providers.<name>.stale_timeout_seconds`` was silently unreachable on
    exactly the local endpoints whose long cold prefills are the reason the
    knob exists.

    Returns the bare entry name (``"ai-router"``), or ``None`` when no
    configured entry owns the endpoint — in which case the caller keeps
    treating this as a genuinely ad-hoc endpoint with no configured timeouts.

    Attribution is by ENDPOINT IDENTITY whenever a base_url is available, and
    only falls back to ``config.model.provider`` when there is no endpoint to
    match on. ``canonical_custom_identity`` always allows that fallback, which
    is right for credential recovery but wrong here: a genuinely ad-hoc
    endpoint would silently inherit an unrelated provider's timeouts.
    """
    normalized = provider_id.strip().lower()
    if normalized != "custom" and not normalized.startswith("custom:"):
        return None

    # A ``custom:<name>`` runtime id already carries the entry name.
    if normalized.startswith("custom:"):
        name = normalized.split(":", 1)[1].strip()
        return name or None

    try:
        if base_url:
            from hermes_cli.runtime_provider import find_custom_provider_identity

            identity = find_custom_provider_identity(base_url)
        else:
            # No endpoint to match on — the configured provider is the only
            # durable identity left (the Desktop/TUI regression vector that
            # canonical_custom_identity exists to absorb).
            from hermes_cli.runtime_provider import canonical_custom_identity

            identity = canonical_custom_identity()
    except Exception:
        return None

    if not identity:
        return None
    return identity.split(":", 1)[1] if ":" in identity else identity


def _provider_config(
    provider_id: str, base_url: str | None
) -> dict[str, object] | None:
    """Return the ``providers:`` entry that owns this runtime provider."""
    if not provider_id:
        return None

    try:
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly()
    except Exception:
        return None

    providers = config.get("providers", {}) if isinstance(config, dict) else {}
    if not isinstance(providers, dict):
        return None

    provider_config = providers.get(provider_id)
    if isinstance(provider_config, dict):
        return provider_config

    # Only pay the reverse lookup when the direct key missed AND the runtime
    # provider is the bare custom billing class. Named and built-in providers
    # keep their existing single-dict-lookup cost on every API turn.
    recovered = _recover_custom_provider_key(provider_id, base_url)
    if recovered:
        recovered_config = providers.get(recovered)
        if isinstance(recovered_config, dict):
            return recovered_config
    return None


def get_provider_request_timeout(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> float | None:
    """Return a configured provider request timeout in seconds, if any.

    ``base_url`` lets a bare-``custom`` runtime provider be attributed to the
    ``providers:`` entry that owns the endpoint; pass the agent's live base_url
    wherever one is available.
    """
    provider_config = _provider_config(provider_id, base_url)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("request_timeout_seconds"))


def get_provider_stale_timeout(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> float | None:
    """Return a configured non-stream stale timeout in seconds, if any.

    ``base_url`` lets a bare-``custom`` runtime provider be attributed to the
    ``providers:`` entry that owns the endpoint; pass the agent's live base_url
    wherever one is available.
    """
    provider_config = _provider_config(provider_id, base_url)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("stale_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("stale_timeout_seconds"))


def get_provider_first_chunk_timeout(
    provider_id: str, model: str | None = None, base_url: str | None = None
) -> float | None:
    """Return a configured pre-first-chunk stream timeout in seconds, if any.

    ``stale_timeout_seconds`` governs the gap between chunks. The wait BEFORE
    the first chunk is a different cost: it is dominated by queue admission,
    a possible model load, and prefill of the whole prompt, so a lane can
    legitimately need minutes there while still wanting a tight inter-chunk
    watchdog. This knob separates the two.

    Reads ``providers.<id>.models.<model>.first_chunk_timeout_seconds`` first,
    then ``providers.<id>.first_chunk_timeout_seconds``. ``base_url`` lets a
    bare-``custom`` runtime provider be attributed to the ``providers:`` entry
    that owns the endpoint; pass the agent's live base_url wherever one is
    available.
    """
    provider_config = _provider_config(provider_id, base_url)
    if provider_config is None:
        return None

    model_config = _get_model_config(provider_config, model)
    if model_config is not None:
        timeout = _coerce_timeout(model_config.get("first_chunk_timeout_seconds"))
        if timeout is not None:
            return timeout

    return _coerce_timeout(provider_config.get("first_chunk_timeout_seconds"))


def _get_model_config(
    provider_config: dict[str, object], model: str | None
) -> dict[str, object] | None:
    if not model:
        return None

    models = provider_config.get("models", {})
    model_config = models.get(model, {}) if isinstance(models, dict) else {}
    if isinstance(model_config, dict):
        return model_config
    return None
