from __future__ import annotations


def _coerce_timeout(raw: object) -> float | None:
    try:
        timeout = float(raw)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _normalize_url(value: object) -> str:
    """Normalize an endpoint URL for comparison.

    Same rule as ``runtime_provider._normalize_base_url_for_match`` so the fast
    path and the general helper can never disagree about whether two URLs are
    the same endpoint.
    """
    return str(value or "").strip().rstrip("/").lower()


def _is_bare_custom_provider_id(provider_id: object) -> bool:
    """Return whether *provider_id* is the bare ``custom`` billing class.

    ``custom`` and ``custom:<name>`` are the only runtime provider ids that do
    NOT already name their own ``providers:`` key, so they are the only ones a
    reverse lookup can say anything about. Shared by the recovery helper and
    its public wrapper so the two can never disagree about which ids need
    recovering.
    """
    normalized = str(provider_id or "").strip().lower()
    return normalized == "custom" or normalized.startswith("custom:")


def _loaded_providers_section() -> dict | None:
    """Return the live ``providers:`` mapping, or ``None`` if unavailable.

    Uses ``load_config_readonly`` (no defensive deepcopy) because this is read
    on API-turn paths; the result is never mutated here.
    """
    try:
        from hermes_cli.config import load_config_readonly

        config = load_config_readonly()
    except Exception:
        return None
    providers = config.get("providers") if isinstance(config, dict) else None
    return providers if isinstance(providers, dict) else None


def _recover_custom_provider_key(
    provider_id: str, base_url: str | None, providers: dict | None = None
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
    normalized = str(provider_id or "").strip().lower()
    if not _is_bare_custom_provider_id(normalized):
        return None

    # A ``custom:<name>`` runtime id already carries the entry name.
    if normalized.startswith("custom:"):
        name = normalized.split(":", 1)[1].strip()
        return name or None

    # Fast path: match the endpoint against the providers dict the caller has
    # already loaded. `find_custom_provider_identity` below is the general
    # helper, but it calls `load_config()`, which deepcopies the whole config —
    # and load_config's own docstring names `get_provider_request_timeout` as
    # the per-API-turn hot spot that must not pay that. A timeout can only be
    # read from a `providers:` entry anyway, so when one owns this URL there is
    # nothing left for the general helper to find.
    if base_url and isinstance(providers, dict):
        target = _normalize_url(base_url)
        if target:
            owners = [
                name
                for name, entry in providers.items()
                if isinstance(entry, dict)
                and _normalize_url(
                    entry.get("api") or entry.get("url") or entry.get("base_url") or ""
                ) == target
            ]
            # Exactly one owner, matching the rule #330 established: rows with
            # distinct credentials can share an endpoint, and none of them can
            # claim it alone.
            if len(owners) == 1:
                return owners[0]
            if len(owners) > 1:
                return None

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


def resolve_provider_config_key(
    provider_id: str, base_url: str | None = None, providers: dict | None = None
) -> str | None:
    """Map a runtime provider id back to the config-entry name that owns it.

    Public wrapper around :func:`_recover_custom_provider_key` for callers
    outside this module that need the same identity recovery — anything keyed
    on a ``providers:`` / ``custom_providers:`` ENTRY NAME rather than on the
    resolved billing class. ``resolve_runtime_provider()`` reports every
    user-declared endpoint as the bare string ``"custom"``, so code that
    compares ``agent.provider`` against entry names matches nothing at runtime;
    run the provider through here first.

    Returns ``None`` for a named or built-in provider id (``"anthropic"``,
    ``"openrouter"``, …). That is not "no entry exists" — those ids ARE their
    own config key already, so there is nothing to recover and no config read
    is performed.

    Attribution is by endpoint identity: pass the agent's live ``base_url``.
    With no ``base_url`` the underlying helper falls back to
    ``config.model.provider``, which is right for credential/timeout recovery
    but wrong for anything that must not be granted to an unidentified
    endpoint — such callers should refuse to recover without a ``base_url``.

    ``providers`` is loaded from config when not supplied. Supplying it also
    engages the exactly-one-owner rule (#330): when two entries declare the
    same endpoint, neither owns it and the result is ``None``.
    """
    if not _is_bare_custom_provider_id(provider_id):
        return None
    if providers is None:
        providers = _loaded_providers_section()
    return _recover_custom_provider_key(str(provider_id), base_url, providers)


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
    recovered = _recover_custom_provider_key(provider_id, base_url, providers)
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
