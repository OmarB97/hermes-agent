"""Helpers for reading and enforcing the fallback provider policy."""

from __future__ import annotations

import os
from typing import Any


FALLBACK_POLICIES = ("off", "local-only", "any")
DEFAULT_FALLBACK_POLICY = "any"
_BUILTIN_LOCAL_PROVIDER_IDS = frozenset({"custom", "local", "lmstudio"})


def get_fallback_policy(config: dict[str, Any] | None) -> str:
    """Return the configured fallback policy, failing closed on invalid input.

    ``any`` is the compatibility default for configs written before the policy
    existed.  Invalid explicit values are treated as ``off`` at runtime; config
    validation reports the mistake so a typo can never silently widen routing.
    """
    config = config or {}
    if "fallback_policy" not in config:
        return DEFAULT_FALLBACK_POLICY
    value = config.get("fallback_policy")
    if isinstance(value, str) and value in FALLBACK_POLICIES:
        return value
    return "off"


def _normalized_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().rstrip("/")


def _iter_fallback_entries(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        candidates = [raw]
    elif isinstance(raw, list):
        candidates = raw
    else:
        return []

    entries: list[dict[str, Any]] = []
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        provider = str(entry.get("provider") or "").strip()
        model = str(entry.get("model") or "").strip()
        if not provider or not model:
            continue

        normalized = dict(entry)
        normalized["provider"] = provider
        normalized["model"] = model

        base_url = _normalized_base_url(entry.get("base_url"))
        if base_url:
            normalized["base_url"] = base_url

        entries.append(normalized)
    return entries


def _entry_identity(entry: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(entry.get("provider") or "").strip().lower(),
        str(entry.get("model") or "").strip().lower(),
        _normalized_base_url(entry.get("base_url")).lower(),
    )


def _fallback_promotions_config(config: dict[str, Any]) -> dict[str, Any]:
    raw = config.get("fallback_promotions")
    if not isinstance(raw, dict):
        return {}
    return raw


def _fallback_promotions_enabled(config: dict[str, Any]) -> bool:
    """Return whether dynamic free-promotion fallbacks should be merged.

    The feature is opt-in through DEFAULT_CONFIG's ``fallback_promotions`` key.
    Tests and callers that pass a small ad-hoc config without that key keep the
    old static-only behavior.
    """
    env_override = os.getenv("HERMES_FALLBACK_PROMOTIONS", "").strip().lower()
    if env_override in {"0", "false", "no", "off"}:
        return False
    if env_override in {"1", "true", "yes", "on"}:
        return True

    promotions = _fallback_promotions_config(config)
    return promotions.get("enabled") is True


def _provider_promotions_enabled(config: dict[str, Any], provider: str) -> bool:
    promotions = _fallback_promotions_config(config)
    providers = promotions.get("providers", ["nous"])
    if isinstance(providers, str) and providers.strip().lower() in {"*", "all"}:
        return True
    if not isinstance(providers, list):
        return False
    wanted = {str(p).strip().lower() for p in providers if str(p).strip()}
    return provider.strip().lower() in wanted


def _promotion_position(config: dict[str, Any]) -> str:
    promotions = _fallback_promotions_config(config)
    position = str(promotions.get("position") or "prepend").strip().lower()
    if position not in {"prepend", "append"}:
        return "prepend"
    return position


def _provider_has_auth(provider: str) -> bool:
    """Cheap auth gate so startup does not hit remote promo endpoints uselessly."""
    provider = provider.strip().lower()
    if provider == "nous":
        if os.getenv("NOUS_API_KEY", "").strip():
            return True
        try:
            from hermes_cli.auth import get_provider_auth_state

            return bool(get_provider_auth_state("nous"))
        except Exception:
            return False
    return False


def _free_nous_promotion_entries() -> list[dict[str, Any]]:
    """Return free Nous Portal promo models currently advertised by Portal."""
    try:
        from hermes_cli.models import (
            _resolve_nous_portal_url,
            fetch_nous_recommended_models,
        )
    except Exception:
        return []

    try:
        payload = fetch_nous_recommended_models(
            _resolve_nous_portal_url(),
            timeout=1.5,
        )
    except Exception:
        return []

    free_models = payload.get("freeRecommendedModels") if isinstance(payload, dict) else None
    if not isinstance(free_models, list):
        return []

    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in free_models:
        if not isinstance(item, dict):
            continue
        model = str(item.get("modelName") or "").strip()
        if not model or model.lower() in seen:
            continue
        seen.add(model.lower())
        entries.append(
            {
                "provider": "nous",
                "model": model,
                "supports_tools": True,
                "source": "dynamic-free-promotion",
            }
        )
    return entries


def _dynamic_free_promotion_entries(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not _fallback_promotions_enabled(config):
        return []

    entries: list[dict[str, Any]] = []
    if _provider_promotions_enabled(config, "nous") and _provider_has_auth("nous"):
        entries.extend(_free_nous_promotion_entries())
    return entries


def _resolve_fallback_provider_definition(
    entry: dict[str, Any],
    config: dict[str, Any],
):
    provider = str(entry.get("provider") or "").strip()
    if not provider:
        return None
    try:
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.providers import resolve_provider_full

        return resolve_provider_full(
            provider,
            user_providers=config.get("providers")
            if isinstance(config.get("providers"), dict)
            else None,
            custom_providers=get_compatible_custom_providers(config),
        )
    except Exception:
        return None


def fallback_entry_endpoint(
    entry: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> str:
    """Resolve an entry's endpoint from explicit or provider metadata.

    This intentionally never infers locality from a model name.  An explicit
    ``base_url`` wins, followed by a provider-specific environment override,
    then the canonical provider definition (including user providers).
    """
    explicit = _normalized_base_url(entry.get("base_url"))
    if explicit:
        return explicit

    config = config or {}
    resolved = _resolve_fallback_provider_definition(entry, config)
    if resolved is None:
        return ""

    env_name = str(getattr(resolved, "base_url_env_var", "") or "").strip()
    if env_name:
        env_url = _normalized_base_url(os.getenv(env_name))
        if env_url:
            return env_url
    return _normalized_base_url(getattr(resolved, "base_url", ""))


def fallback_entry_is_local(
    entry: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> bool:
    """Return whether provider metadata proves that an entry is local."""
    config = config or {}
    endpoint = fallback_entry_endpoint(entry, config)
    if not endpoint:
        return False
    try:
        from agent.model_metadata import is_local_endpoint

        if not is_local_endpoint(endpoint):
            return False

        # A built-in cloud provider remains a remote route even if a stale or
        # hostile environment variable points its client at localhost. Users
        # who deliberately redefine that provider under ``providers:`` get a
        # ``user-config`` definition and are judged by that endpoint instead.
        resolved = _resolve_fallback_provider_definition(entry, config)
        if (
            resolved is not None
            and getattr(resolved, "source", "") != "user-config"
            and str(getattr(resolved, "id", "") or "").strip().lower()
            not in _BUILTIN_LOCAL_PROVIDER_IDS
        ):
            return False
        canonical_endpoint = _normalized_base_url(
            getattr(resolved, "base_url", "") if resolved is not None else ""
        )
        if (
            canonical_endpoint
            and getattr(resolved, "source", "") != "user-config"
            and not is_local_endpoint(canonical_endpoint)
        ):
            return False
        return True
    except Exception:
        return False


def get_configured_fallback_chain(
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Return the ordered configured chain before policy eligibility filtering.

    ``fallback_providers`` remains the primary source of truth and keeps its
    relative order. Legacy ``fallback_model`` entries are appended afterwards
    unless they target the same provider/model/base_url route as an earlier
    entry. When ``fallback_promotions.enabled`` is true, currently-free
    provider promotions are merged in memory under the compatibility ``any``
    policy so short-lived free models do not depend on a static YAML edit.
    Promotions are never consulted under ``off`` or ``local-only`` because
    they are remote routes. Explicitly configured entries win on duplicates.
    The returned list always contains fresh dict copies.
    """

    config = config or {}
    chain: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    for key in ("fallback_providers", "fallback_model"):
        for entry in _iter_fallback_entries(config.get(key)):
            identity = _entry_identity(entry)
            if identity in seen:
                continue
            seen.add(identity)
            chain.append(entry)

    promotion_entries: list[dict[str, Any]] = []
    dynamic_entries = (
        _dynamic_free_promotion_entries(config)
        if get_fallback_policy(config) == "any"
        else []
    )
    for entry in dynamic_entries:
        identity = _entry_identity(entry)
        if identity in seen:
            continue
        seen.add(identity)
        promotion_entries.append(entry)

    if promotion_entries and _promotion_position(config) == "prepend":
        return promotion_entries + chain
    if promotion_entries:
        return chain + promotion_entries
    return chain


def get_fallback_chain(config: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Return the chain eligible under ``off | local-only | any``.

    ``off`` never returns a route. ``local-only`` admits only entries whose
    endpoint is proven local by explicit URL or provider metadata; unknown
    endpoints fail closed. ``any`` preserves the historical configured order.
    """
    config = config or {}
    policy = get_fallback_policy(config)
    if policy == "off":
        return []

    chain = get_configured_fallback_chain(config)
    if policy == "local-only":
        return [entry for entry in chain if fallback_entry_is_local(entry, config)]
    return chain
