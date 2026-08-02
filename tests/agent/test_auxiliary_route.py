"""``auxiliary.route`` — the single global auxiliary lane.

WHY THIS FILE EXISTS AS AN END-TO-END SUITE

Every existing approval / goal test stubs at the ``_smart_approve`` /
``judge_goal`` / ``call_llm`` boundary, so none of them can observe a routing
regression.  That is exactly how the trap this feature has to defeat survived
the first attempt: a routed lane that is UP but returns 401 raises out of
``call_llm``, ``tools/approval.py::_smart_approve`` swallows it and returns
``"escalate"``, and every smart auto-approval silently becomes a human prompt.
So these tests drive the REAL resolution path against a temp ``HERMES_HOME``
with a real ``config.yaml``, and only mock at the HTTP client.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import agent.auxiliary_client as aux
from agent.auxiliary_client import (
    _aux_call_budget_seconds,
    _auxiliary_route_target,
    _normalize_aux_provider,
    _resolve_task_provider_model,
    _route_is_active_for_task,
    auxiliary_route_is_configured,
    call_llm,
    preflight_auxiliary_route,
)

ROUTE_BASE_URL = "http://aux-lane.test/v1"
MAIN = {"provider": "deepseek", "default": "deepseek-chat"}

# Tasks that must follow the route when they pin nothing of their own.
ROUTED_TASKS = ("approval", "goal_judge", "title_generation", "compression",
                "web_extract", "mcp", "skills_hub")


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a real config.yaml the loaders will read."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture(autouse=True)
def _clean_route_state():
    """Module-level caches leak across tests (10-min TTLs, client cache)."""
    aux._aux_route_unhealthy_until.clear()
    aux._aux_unhealthy_until.clear()
    aux._aux_failure_notified_at.clear()
    aux._client_cache.clear()
    aux.set_auxiliary_failure_notifier(None)
    yield
    aux._aux_route_unhealthy_until.clear()
    aux._aux_unhealthy_until.clear()
    aux._aux_failure_notified_at.clear()
    aux._client_cache.clear()
    aux.set_auxiliary_failure_notifier(None)


def write_config(home: Path, data: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


def route_config(**route) -> dict:
    """Config with a main model plus an ``auxiliary.route`` lane."""
    return {
        "model": dict(MAIN),
        "approvals": {"mode": "smart", "timeout": 60},
        "auxiliary": {"route": route},
    }


def auth_error(message: str = "Error code: 401 - invalid api key") -> Exception:
    err = RuntimeError(message)
    err.status_code = 401
    return err


def connection_error() -> Exception:
    return RuntimeError("Connection refused")


def reply(text: str):
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


# ──────────────────────────────────────────────────────────────────────
# Default: no route configured
# ──────────────────────────────────────────────────────────────────────


class TestRouteDefaultOff:
    @pytest.mark.parametrize("task", ROUTED_TASKS)
    def test_unset_route_resolves_exactly_as_before(self, hermes_home, task):
        """No route on disk → every task still resolves to plain "auto"."""
        write_config(hermes_home, {"model": dict(MAIN)})

        assert _resolve_task_provider_model(task) == ("auto", None, None, None, None)
        assert _route_is_active_for_task(task) is False
        assert auxiliary_route_is_configured() is False
        assert _auxiliary_route_target() is None

    def test_default_config_route_block_is_inert(self, hermes_home):
        """DEFAULT_CONFIG ships the route block; every field is blank, so the
        merged view still reports "no route"."""
        from hermes_cli.config import DEFAULT_CONFIG, load_config

        write_config(hermes_home, {})
        assert "route" in DEFAULT_CONFIG["auxiliary"]
        merged_route = load_config()["auxiliary"]["route"]
        assert not any(str(v).strip() for v in merged_route.values() if v not in (0,))
        assert _auxiliary_route_target() is None

    def test_preflight_is_a_no_op_without_a_route(self, hermes_home):
        write_config(hermes_home, {"model": dict(MAIN)})
        assert preflight_auxiliary_route() is None


# ──────────────────────────────────────────────────────────────────────
# Route on: who wins
# ──────────────────────────────────────────────────────────────────────


class TestRoutePrecedence:
    @pytest.mark.parametrize("task", ROUTED_TASKS)
    def test_route_owns_tasks_that_pin_nothing(self, hermes_home, task):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key", model="small-lane"))

        assert _resolve_task_provider_model(task) == (
            "custom", "small-lane", ROUTE_BASE_URL, "lane-key", None)
        assert _route_is_active_for_task(task) is True

    def test_task_provider_beats_the_route(self, hermes_home):
        cfg = route_config(base_url=ROUTE_BASE_URL, api_key="lane-key")
        cfg["auxiliary"]["compression"] = {"provider": "nous"}
        write_config(hermes_home, cfg)

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model("compression")
        assert provider == "nous"
        assert base_url is None
        assert _route_is_active_for_task("compression") is False
        # Sibling tasks are untouched by one task's opt-out.
        assert _resolve_task_provider_model("approval")[2] == ROUTE_BASE_URL

    def test_task_base_url_beats_the_route(self, hermes_home):
        cfg = route_config(base_url=ROUTE_BASE_URL, api_key="lane-key")
        cfg["auxiliary"]["web_extract"] = {
            "base_url": "http://task-pinned.test/v1", "api_key": "task-key"}
        write_config(hermes_home, cfg)

        assert _resolve_task_provider_model("web_extract") == (
            "custom", None, "http://task-pinned.test/v1", "task-key", None)

    def test_task_model_pin_keeps_todays_lane(self, hermes_home):
        """A model pinned while the provider is "auto" was pinned FOR the main
        lane. Model ids are lane-specific, so carrying it onto a different
        endpoint would 404 rather than answer — that task keeps "auto"."""
        cfg = route_config(base_url=ROUTE_BASE_URL, api_key="lane-key")
        cfg["auxiliary"]["title_generation"] = {"provider": "auto", "model": "gpt-4o-mini"}
        write_config(hermes_home, cfg)

        assert _resolve_task_provider_model("title_generation") == (
            "auto", "gpt-4o-mini", None, None, None)
        assert _route_is_active_for_task("title_generation") is False

    def test_provider_main_opts_a_task_back_to_the_main_lane(self, hermes_home):
        """``provider: main`` is the documented one-line per-task escape now
        that "auto" means "the route"."""
        cfg = route_config(base_url=ROUTE_BASE_URL, api_key="lane-key")
        cfg["auxiliary"]["approval"] = {"provider": "main"}
        write_config(hermes_home, cfg)

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model("approval")
        assert provider == "main"
        assert base_url is None
        assert _normalize_aux_provider(provider) == MAIN["provider"]

    def test_vision_is_never_routed(self, hermes_home):
        """An image payload needs a multimodal model; the vision path has its
        own capability-aware chain. ``auxiliary.vision`` remains the pin."""
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key", model="small-lane"))

        assert _resolve_task_provider_model("vision") == ("auto", None, None, None, None)
        assert _route_is_active_for_task("vision") is False

    def test_raw_auxiliary_vision_config_is_untouched(self, hermes_home):
        """image_routing / vision_routing read RAW config to choose
        native-vs-aux image handling; materialising the route as
        ``auxiliary.vision`` would flip that decision."""
        from agent.image_routing import _explicit_aux_vision_override
        from hermes_cli.config import load_config

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key"))

        assert _explicit_aux_vision_override(load_config()) is False

    def test_explicit_call_argument_beats_the_route(self, hermes_home):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key"))

        provider, _model, base_url, _key, _mode = _resolve_task_provider_model(
            "approval", provider="nous")
        assert provider == "nous"
        assert base_url is None

    def test_named_provider_route_uses_that_entrys_default_model(self, hermes_home, monkeypatch):
        """Without this the shared resolver fills a blank model from
        ``_read_main_model()`` — i.e. it would send the MAIN model's id to the
        auxiliary endpoint, the one thing this lane must not do."""
        monkeypatch.setenv("AUX_LANE_KEY", "sk-lane")
        cfg = route_config(provider="auxlane")
        cfg["providers"] = {"auxlane": {
            "api": ROUTE_BASE_URL, "key_env": "AUX_LANE_KEY",
            "default_model": "small-lane"}}
        write_config(hermes_home, cfg)

        provider, model, _base, _key, _mode = _resolve_task_provider_model("approval")
        assert provider == "auxlane"
        assert model == "small-lane"
        assert model != MAIN["default"]


# ──────────────────────────────────────────────────────────────────────
# G1 — the headline: a routed lane that 401s must NOT strand approvals
# ──────────────────────────────────────────────────────────────────────


class TestRoutedFailureFallsBackToMain:
    def test_routed_401_does_not_escalate_smart_approval(self, hermes_home):
        """THE regression test.

        An aux endpoint that is UP but returns 401 yields ``_is_auth_error``,
        which is NOT a capacity error, so the old ``is_auto or
        is_capacity_error`` gate skipped the entire fallback block, the
        exception reached ``_smart_approve``, and every flagged command became
        a silent human prompt (60 s of dead time each, in an unattended run).
        A provider that came from the global route is a routing DEFAULT, not
        the explicit per-task pin that gate exists to respect.
        """
        from tools.approval import _smart_approve

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="stale", model="small-lane"))

        routed = MagicMock()
        routed.base_url = ROUTE_BASE_URL
        routed.chat.completions.create.side_effect = auth_error()
        main_client = MagicMock()
        main_client.base_url = "https://api.deepseek.com/v1"
        main_client.chat.completions.create.return_value = reply("APPROVE")

        with patch.object(aux, "_get_cached_client", return_value=(routed, "small-lane")), \
             patch.object(aux, "resolve_provider_client",
                          return_value=(main_client, MAIN["default"])), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            verdict = _smart_approve("python -c \"print('hi')\"",
                                     "script execution via -c flag")

        assert verdict == "approve"
        assert routed.chat.completions.create.call_count == 1
        assert main_client.chat.completions.create.call_count == 1

    def test_explicit_task_pin_401_still_respects_the_users_choice(self, hermes_home):
        """The route must not blanket-open the gate: a provider the user
        pinned ON THE TASK keeps today's fail-closed behaviour."""
        from tools.approval import _smart_approve

        cfg = route_config(base_url=ROUTE_BASE_URL, api_key="lane-key")
        cfg["auxiliary"]["approval"] = {
            "base_url": "http://task-pinned.test/v1", "api_key": "stale"}
        write_config(hermes_home, cfg)

        pinned = MagicMock()
        pinned.base_url = "http://task-pinned.test/v1"
        pinned.chat.completions.create.side_effect = auth_error()
        main_client = MagicMock()
        main_client.chat.completions.create.return_value = reply("APPROVE")

        with patch.object(aux, "_get_cached_client", return_value=(pinned, "pinned-model")), \
             patch.object(aux, "resolve_provider_client",
                          return_value=(main_client, MAIN["default"])), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            verdict = _smart_approve("rm -rf /tmp/x", "recursive delete")

        assert verdict == "escalate"
        assert main_client.chat.completions.create.call_count == 0

    def test_routed_failure_is_reported_to_the_operator(self, hermes_home):
        """G4: the failure reaches the agent's chat-visible warning channel
        instead of vanishing into a debug log."""
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="stale", model="small-lane"))

        seen: list[tuple[str, str]] = []
        aux.set_auxiliary_failure_notifier(
            lambda task, exc: seen.append((task, str(exc))))

        routed = MagicMock()
        routed.base_url = ROUTE_BASE_URL
        routed.chat.completions.create.side_effect = auth_error()
        main_client = MagicMock()
        main_client.chat.completions.create.return_value = reply("hello")

        with patch.object(aux, "_get_cached_client", return_value=(routed, "small-lane")), \
             patch.object(aux, "resolve_provider_client",
                          return_value=(main_client, MAIN["default"])), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            call_llm(task="title_generation",
                     messages=[{"role": "user", "content": "hi"}])

        assert len(seen) == 1
        assert "title_generation" in seen[0][0]
        assert "401" in seen[0][1]


# ──────────────────────────────────────────────────────────────────────
# G2 — a dead routed lane is quarantined
# ──────────────────────────────────────────────────────────────────────


class TestRouteQuarantine:
    def test_connection_error_quarantines_the_route(self, hermes_home):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key", model="small-lane"))
        assert _route_is_active_for_task("title_generation") is True

        routed = MagicMock()
        routed.base_url = ROUTE_BASE_URL
        routed.chat.completions.create.side_effect = connection_error()
        main_client = MagicMock()
        main_client.chat.completions.create.return_value = reply("a title")

        with patch.object(aux, "_get_cached_client", return_value=(routed, "small-lane")), \
             patch.object(aux, "resolve_provider_client",
                          return_value=(main_client, MAIN["default"])), \
             patch.object(aux, "_transient_retry_count", return_value=0), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            response = call_llm(task="title_generation",
                                messages=[{"role": "user", "content": "hi"}])

        assert response.choices[0].message.content == "a title"
        # The dead lane pays its penalty once per TTL: subsequent resolution
        # skips the route entirely instead of paying another doomed RTT.
        assert _route_is_active_for_task("title_generation") is False
        assert _resolve_task_provider_model("title_generation") == (
            "auto", None, None, None, None)

    def test_quarantine_does_not_touch_the_shared_provider_health_cache(
        self, hermes_home
    ):
        """``_try_main_agent_model_fallback`` bails when the MAIN provider is
        marked unhealthy, so a route quarantine that wrote into that shared
        cache could disable the very fallback that keeps unattended turns
        moving. The route quarantine is deliberately separate."""
        write_config(hermes_home, route_config(
            provider="custom", base_url=ROUTE_BASE_URL, api_key="k", model="m"))

        aux._mark_route_unhealthy("connection error")

        assert aux._route_is_quarantined() is True
        assert aux._aux_unhealthy_until == {}
        assert aux._is_provider_unhealthy("local/custom") is False


# ──────────────────────────────────────────────────────────────────────
# G3 — a routed approval must decide before the human prompt gives up
# ──────────────────────────────────────────────────────────────────────


class TestRoutedApprovalBudget:
    def test_routed_approval_budget_is_under_the_human_deadline(self, hermes_home):
        from tools.approval import _get_approval_timeout

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key", model="small-lane"))

        human_deadline = _get_approval_timeout()
        assert human_deadline == 60

        routed_budget = _aux_call_budget_seconds("approval", route_derived=True)
        assert routed_budget < human_deadline

        # And the un-routed budget it replaces does NOT fit: the default
        # auxiliary.approval.timeout of 30 s x 3 attempts + 3 s backoff = 93 s,
        # which OUT-WAITS approvals.timeout. That is the stall this bounds.
        assert _aux_call_budget_seconds("approval") > human_deadline

    def test_route_timeout_and_retries_are_configurable(self, hermes_home):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key",
            timeout=7, transient_retries=1))

        assert aux._effective_aux_timeout("web_extract", None, route_derived=True) == 7.0
        assert aux._transient_retry_count(route_derived=True) == 1
        # Un-routed calls keep the task's own timeout and the global retries.
        assert aux._effective_aux_timeout("web_extract", None) == 360.0
        assert aux._transient_retry_count() == 2

    def test_routed_lanes_do_not_retry_by_default(self, hermes_home):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="lane-key"))

        assert aux._transient_retry_count(route_derived=True) == 0


# ──────────────────────────────────────────────────────────────────────
# G5 — the route's credentials actually resolve, and preflight says so
# ──────────────────────────────────────────────────────────────────────


class TestRoutePreflight:
    def _providers_config(self, **entry) -> dict:
        cfg = route_config(provider="auxlane")
        cfg["providers"] = {"auxlane": {"api": ROUTE_BASE_URL,
                                        "default_model": "small-lane", **entry}}
        return cfg

    def test_providers_entry_key_env_resolves_the_key(self, hermes_home, monkeypatch):
        """The new-style ``providers:`` path resolved ``key_env`` internally but
        never returned it, so the second-chance ``os.getenv`` in
        resolve_provider_client could not rescue such an entry and it fell
        through to the ``no-key-required`` placeholder — a silent 401."""
        from hermes_cli.runtime_provider import _get_named_custom_provider

        monkeypatch.setenv("AUX_LANE_KEY", "sk-lane")
        write_config(hermes_home, self._providers_config(key_env="AUX_LANE_KEY"))

        entry = _get_named_custom_provider("auxlane")
        assert entry["api_key"] == "sk-lane"
        assert entry["key_env"] == "AUX_LANE_KEY"

        client, model = aux.resolve_provider_client("auxlane", model="small-lane")
        assert client is not None
        assert client.api_key == "sk-lane"
        assert client.api_key != "no-key-required"
        assert model == "small-lane"

        assert preflight_auxiliary_route() is None

    def test_providers_entry_accepts_the_api_key_env_spelling(self, hermes_home, monkeypatch):
        from hermes_cli.runtime_provider import _get_named_custom_provider

        monkeypatch.setenv("AUX_LANE_KEY", "sk-lane")
        write_config(hermes_home, self._providers_config(api_key_env="AUX_LANE_KEY"))

        entry = _get_named_custom_provider("auxlane")
        assert entry["api_key"] == "sk-lane"
        assert entry["key_env"] == "AUX_LANE_KEY"

    def test_preflight_warns_when_the_key_cannot_be_resolved(self, hermes_home, monkeypatch):
        monkeypatch.delenv("AUX_LANE_KEY", raising=False)
        write_config(hermes_home, self._providers_config(key_env="AUX_LANE_KEY"))

        warning = preflight_auxiliary_route()
        assert warning is not None
        assert "no-key-required" in warning
        assert "auxlane" in warning

    def test_preflight_warns_when_the_route_resolves_to_no_client(self, hermes_home):
        write_config(hermes_home, route_config(provider="definitely-not-a-provider"))

        warning = preflight_auxiliary_route()
        assert warning is not None
        assert "definitely-not-a-provider" in warning

    def test_route_change_rebuilds_the_cached_client(self, hermes_home):
        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="k", model="m"))
        first = aux._client_cache_key("custom", async_mode=False)

        write_config(hermes_home, route_config(
            base_url="http://other-lane.test/v1", api_key="k", model="m"))
        second = aux._client_cache_key("custom", async_mode=False)

        assert first != second

    def test_cache_key_is_unchanged_without_a_route(self, hermes_home):
        write_config(hermes_home, {"model": dict(MAIN)})
        key = aux._client_cache_key("custom", async_mode=False)
        assert key[-1] == ""


# ──────────────────────────────────────────────────────────────────────
# Goal judge — unchanged semantics on a routed lane
# ──────────────────────────────────────────────────────────────────────


class TestGoalJudgeUnchanged:
    def test_judge_still_fails_open_when_route_and_main_both_fail(self, hermes_home):
        """The judge is a gate on continuing, so an unreachable judge must
        return "continue" — never stall the goal loop."""
        from hermes_cli.goals import judge_goal

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="stale", model="small-lane"))

        routed = MagicMock()
        routed.base_url = ROUTE_BASE_URL
        routed.chat.completions.create.side_effect = auth_error()

        with patch.object(aux, "_get_cached_client", return_value=(routed, "small-lane")), \
             patch.object(aux, "resolve_provider_client", return_value=(None, None)), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            verdict, reason, parse_failed, wait, transport_failed = judge_goal(
                "ship the feature", "still working on it")

        assert verdict == "continue"
        assert parse_failed is False
        assert transport_failed is True
        assert wait is None
        assert "judge error" in reason

    def test_judge_answers_from_the_main_lane_when_the_route_is_a_dud(self, hermes_home):
        from hermes_cli.goals import judge_goal

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="stale", model="small-lane"))

        routed = MagicMock()
        routed.base_url = ROUTE_BASE_URL
        routed.chat.completions.create.side_effect = auth_error()
        main_client = MagicMock()
        main_client.chat.completions.create.return_value = reply(
            '{"verdict": "done", "reason": "shipped"}')

        with patch.object(aux, "_get_cached_client", return_value=(routed, "small-lane")), \
             patch.object(aux, "resolve_provider_client",
                          return_value=(main_client, MAIN["default"])), \
             patch.object(aux, "_refresh_provider_credentials", return_value=False):
            verdict, _reason, parse_failed, _wait, transport_failed = judge_goal(
                "ship the feature", "shipped it")

        assert verdict == "done"
        assert parse_failed is False
        assert transport_failed is False

    def test_consecutive_transport_failures_still_auto_pause(self, hermes_home):
        """The 5-in-a-row auto-pause is what stops a permanently broken judge
        from burning the whole turn budget. Routing must not defuse it."""
        from hermes_cli import goals
        from hermes_cli.goals import (
            DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES,
            GoalManager,
        )

        write_config(hermes_home, route_config(
            base_url=ROUTE_BASE_URL, api_key="stale", model="small-lane"))
        goals._DB_CACHE.clear()
        try:
            mgr = GoalManager(session_id="route-judge-sid", default_max_turns=50)
            mgr.set("ship the feature")

            with patch.object(
                goals, "judge_goal",
                return_value=("continue", "judge error: RuntimeError", False, None, True),
            ):
                for _ in range(DEFAULT_MAX_CONSECUTIVE_TRANSPORT_FAILURES - 1):
                    decision = mgr.evaluate_after_turn("still going")
                    assert decision["should_continue"] is True
                    assert mgr.state.status == "active"

                decision = mgr.evaluate_after_turn("still going")

            assert decision["should_continue"] is False
            assert decision["status"] == "paused"
            assert mgr.state.status == "paused"
        finally:
            goals._DB_CACHE.clear()
