"""``auxiliary.<task>.provider: <entry>`` with no ``model:`` — which model ships.

WHY THIS FILE EXISTS AS AN END-TO-END SUITE

The bug it pins is invisible to any test that stubs the client boundary: the
provider, the base_url and the api_key were all resolved correctly, so a
resolution-level assertion on "did we reach the right endpoint" passed while
the request carried the wrong ``model`` id.  ``resolve_provider_client``'s
universal fallback (``if not model and provider != "auto"``) filled the blank
from ``_read_main_model()``, so a lane pinned at a local endpoint received the
MAIN chat model's slug — a 404 from a backend that serves a different model
set, not an answer.

So these tests drive the REAL resolution path against a temp ``HERMES_HOME``
with a real ``config.yaml`` and assert on the ``model`` kwarg that reaches
``.chat.completions.create()``.  ``auxiliary.route`` already resolved its lane
this way (see ``test_auxiliary_route.py``); this is the per-task twin, which
PR #340 deliberately left out of scope.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    _named_provider_default_model,
    _resolve_task_provider_model,
    call_llm,
    get_text_auxiliary_client,
    resolve_vision_provider_client,
)

LANE_URL = "http://127.0.0.1:8080/v1"
LANE_MODEL = "qwen3-4b-local"
MAIN = {"provider": "deepseek", "default": "deepseek-chat"}


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a real config.yaml the loaders will read."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _clean_client_cache(monkeypatch):
    """The client cache is keyed without the model — a leaked entry would let
    one test's resolved model answer for the next one."""
    for key in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENROUTER_API_KEY",
                "DEEPSEEK_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()
    yield
    aux._client_cache.clear()
    aux._aux_unhealthy_until.clear()


def write_config(home: Path, data: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def lane_config(task_block: dict, *, entry: dict | None = None,
                legacy: bool = False) -> dict:
    """Config with a main model plus one user-declared provider entry.

    ``legacy`` writes the ``custom_providers:`` list shape (``model:``)
    instead of the ``providers:`` dict shape (``default_model:``).
    """
    cfg: dict = {
        "model": dict(MAIN),
        "approvals": {"mode": "smart", "timeout": 60},
        "auxiliary": task_block,
    }
    if legacy:
        cfg["custom_providers"] = [
            {"name": "my-local-lane", "base_url": LANE_URL,
             **({"model": LANE_MODEL} if entry is None else entry)},
        ]
    else:
        cfg["providers"] = {
            "my-local-lane": {
                "api": LANE_URL,
                **({"default_model": LANE_MODEL} if entry is None else entry),
            },
        }
    return cfg


def reply(text: str = "APPROVE"):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def wire_model(task: str = "approval") -> str:
    """Run one real ``call_llm`` and return the model id that hit the wire.

    Only the OpenAI client construction is stubbed, so task config →
    ``_resolve_task_provider_model`` → ``resolve_provider_client`` →
    ``_build_call_kwargs`` all run for real.
    """
    client = MagicMock()
    client.base_url = LANE_URL
    client.api_key = "no-key-required"
    client.chat.completions.create.return_value = reply()
    # The client cache is keyed without the model, so a second call in the
    # same test would be served from the first call's entry and this stub
    # would never see a request.
    aux._client_cache.clear()
    with patch.object(aux, "_create_openai_client", return_value=client):
        call_llm(messages=[{"role": "user", "content": "hi"}], task=task)
    return client.chat.completions.create.call_args.kwargs["model"]


# ──────────────────────────────────────────────────────────────────────
# The headline: a named lane with no model pinned
# ──────────────────────────────────────────────────────────────────────


class TestNamedLaneWithoutAModel:
    def test_entrys_default_model_is_sent_not_the_main_chat_model(self, hermes_home):
        """THE regression test.

        ``auxiliary.approval.provider: my-local-lane`` with no ``model:`` used
        to send ``deepseek-chat`` — the MAIN chat model — to a local endpoint
        that has never heard of it.
        """
        write_config(hermes_home, lane_config({"approval": {"provider": "my-local-lane"}}))

        assert wire_model() == LANE_MODEL
        assert wire_model() != MAIN["default"]

    def test_resolution_reports_the_entrys_model(self, hermes_home):
        write_config(hermes_home, lane_config({"approval": {"provider": "my-local-lane"}}))

        provider, model, base_url, _key, _mode = _resolve_task_provider_model("approval")
        assert provider == "my-local-lane"
        assert model == LANE_MODEL
        assert base_url is None  # the entry owns the endpoint, not the task

    def test_the_client_is_built_against_the_lane_with_that_model(self, hermes_home):
        """The endpoint was always right; only the model id was wrong."""
        write_config(hermes_home, lane_config({"compression": {"provider": "my-local-lane"}}))

        client, model = get_text_auxiliary_client("compression")
        assert client is not None
        assert str(client.base_url).rstrip("/") == LANE_URL
        assert model == LANE_MODEL

    def test_custom_colon_name_spelling_resolves_the_same_way(self, hermes_home):
        """``custom:<name>`` is the canonical menu key for the same entry."""
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "custom:my-local-lane"}}))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model == LANE_MODEL

    def test_legacy_custom_providers_list_shape_is_covered(self, hermes_home):
        """The legacy list shape spells it ``model:``, the dict shape
        ``default_model:`` — both arrive as ``entry["model"]``."""
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "my-local-lane"}}, legacy=True))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model == LANE_MODEL
        assert wire_model() == LANE_MODEL

    @pytest.mark.parametrize("model_key", ["model", "default_model"])
    def test_both_legacy_model_spellings_resolve(self, hermes_home, model_key):
        """``_normalize_custom_provider_entry`` accepts either spelling on a
        list entry, so neither may be dropped here."""
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "my-local-lane"}},
            entry={"base_url": LANE_URL, model_key: LANE_MODEL},
            legacy=True))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model == LANE_MODEL


# ──────────────────────────────────────────────────────────────────────
# What must NOT change
# ──────────────────────────────────────────────────────────────────────


class TestUnchangedBehaviour:
    def test_a_task_level_model_pin_still_wins(self, hermes_home):
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "my-local-lane", "model": "pinned-tiny"}}))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model == "pinned-tiny"
        assert wire_model() == "pinned-tiny"

    def test_an_explicit_call_model_still_wins(self, hermes_home):
        write_config(hermes_home, lane_config({"approval": {"provider": "my-local-lane"}}))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model(
            "approval", model="caller-choice")
        assert model == "caller-choice"

    @pytest.mark.parametrize("sentinel", ["auto", "main", "custom"])
    def test_main_lane_sentinels_keep_inheriting_the_main_model(
        self, hermes_home, sentinel,
    ):
        """``auto`` / ``main`` / bare ``custom`` all mean "the lane the main
        runtime resolved".  Even with an entry literally named ``custom`` on
        disk, their blank model must still fall through to
        ``_read_main_model()`` in resolve_provider_client."""
        cfg = lane_config({"approval": {"provider": sentinel}})
        cfg["providers"]["custom"] = {"api": "http://127.0.0.1:9999/v1",
                                      "default_model": "shadow-model"}
        write_config(hermes_home, cfg)

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model is None

    def test_model_auto_sentinel_still_drops_to_none_for_a_named_lane(self, hermes_home):
        """``model: auto`` is not a model id.  It is nulled before the lane
        lookup, so the lane's own default fills the slot instead of the
        literal string reaching the wire."""
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "my-local-lane", "model": "auto"}}))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model == LANE_MODEL

    def test_entry_without_a_default_model_falls_back_as_before(self, hermes_home):
        """No ``default_model`` on the entry → nothing to prefer, so the old
        chain (provider catalog default, then the main model) still runs."""
        write_config(hermes_home, lane_config(
            {"approval": {"provider": "my-local-lane"}}, entry={"api": LANE_URL}))

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert model is None
        assert wire_model() == MAIN["default"]

    def test_a_first_class_provider_is_untouched(self, hermes_home):
        """``_get_named_custom_provider`` defers to canonical built-ins, so a
        task pinned at one resolves exactly as before."""
        write_config(hermes_home, lane_config({"approval": {"provider": "nous"}}))

        provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert provider == "nous"
        assert model is None

    def test_a_task_that_pins_nothing_is_untouched(self, hermes_home):
        write_config(hermes_home, lane_config({"approval": {}}))

        assert _resolve_task_provider_model("approval") == ("auto", None, None, None, None)


# ──────────────────────────────────────────────────────────────────────
# The other two doors into the same resolver
# ──────────────────────────────────────────────────────────────────────


class TestVisionAndAsyncPaths:
    def test_vision_lane_uses_the_entrys_default_model(self, hermes_home):
        """``resolve_vision_provider_client`` funnels through the same
        resolver, so the vision lane carried the main model too."""
        write_config(hermes_home, lane_config({"vision": {"provider": "my-local-lane"}}))

        provider, client, model = resolve_vision_provider_client()
        assert provider == "my-local-lane"
        assert client is not None
        assert model == LANE_MODEL
        assert model != MAIN["default"]

    def test_vision_provider_main_still_inherits_the_main_model(self, hermes_home):
        """The sentinel guard has to hold on the vision path as well —
        ``_normalize_vision_provider`` resolves ``main`` away before
        ``resolve_provider_client`` ever sees the raw name."""
        cfg = lane_config({"vision": {"provider": "main"}})
        cfg["model"] = {"provider": "custom:my-local-lane", "default": "deepseek-chat"}
        write_config(hermes_home, cfg)

        _provider, model, _base, _key, _mode = _resolve_task_provider_model("vision")
        assert model is None

    def test_async_client_gets_the_entrys_default_model(self, hermes_home):
        from agent.auxiliary_client import get_async_text_auxiliary_client

        write_config(hermes_home, lane_config({"compression": {"provider": "my-local-lane"}}))

        client, model = get_async_text_auxiliary_client("compression")
        assert client is not None
        assert model == LANE_MODEL


# ──────────────────────────────────────────────────────────────────────
# The shared helper (also feeds _auxiliary_route_target)
# ──────────────────────────────────────────────────────────────────────


class TestNamedProviderDefaultModelHelper:
    def test_reads_the_providers_dict_shape(self, hermes_home):
        write_config(hermes_home, lane_config({}))
        assert _named_provider_default_model("my-local-lane") == LANE_MODEL

    def test_reads_the_legacy_list_shape(self, hermes_home):
        write_config(hermes_home, lane_config({}, legacy=True))
        assert _named_provider_default_model("my-local-lane") == LANE_MODEL

    def test_returns_none_for_an_unknown_name(self, hermes_home):
        write_config(hermes_home, lane_config({}))
        assert _named_provider_default_model("not-configured") is None

    def test_returns_none_for_blank_input(self, hermes_home):
        write_config(hermes_home, lane_config({}))
        assert _named_provider_default_model("") is None
        assert _named_provider_default_model(None) is None

    def test_survives_a_broken_config_loader(self, hermes_home):
        """Model resolution must never be the thing that takes a turn down."""
        write_config(hermes_home, lane_config({}))
        with patch("hermes_cli.runtime_provider._get_named_custom_provider",
                   side_effect=RuntimeError("config exploded")):
            assert _named_provider_default_model("my-local-lane") is None
