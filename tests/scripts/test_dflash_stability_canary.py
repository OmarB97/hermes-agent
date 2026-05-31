from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dflash_stability_canary.py"
SPEC = importlib.util.spec_from_file_location("dflash_stability_canary", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
canary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = canary
SPEC.loader.exec_module(canary)


def test_incomplete_tail_classifier_catches_short_connector_fragment():
    assert canary.looks_like_incomplete_tail("I see a lot of discord-res tasks (digest Discord content) and some")
    assert canary.looks_like_incomplete_tail("Now let me read the STATUS.md to find a task I can pick up")
    assert not canary.looks_like_incomplete_tail("CANARY_ONBOARD_OK")


def test_classify_result_requires_exact_marker():
    ok = canary.CommandResult(returncode=0, stdout="CANARY_ONBOARD_OK\n", stderr="", elapsed_s=1.0)
    noisy = canary.CommandResult(returncode=0, stdout="Done. CANARY_ONBOARD_OK\n", stderr="", elapsed_s=1.0)

    assert canary.classify_result(ok, "CANARY_ONBOARD_OK") is None
    assert canary.classify_result(noisy, "CANARY_ONBOARD_OK") == "marker-mismatch"
    assert canary.classify_result(noisy, "CANARY_ONBOARD_OK", strict_marker=False) is None


def test_classify_result_flags_auth_empty_timeout_and_nonzero():
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=0, stdout="", stderr="", elapsed_s=1.0),
            "MARKER",
        )
        == "empty-final"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=0, stdout="MARKER", stderr="Error code: 401 Invalid API key", elapsed_s=1.0),
            "MARKER",
        )
        == "auth-error"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=124, stdout="", stderr="", elapsed_s=180.0, timed_out=True),
            "MARKER",
        )
        == "timeout"
    )
    assert (
        canary.classify_result(
            canary.CommandResult(returncode=2, stdout="MARKER", stderr="", elapsed_s=1.0),
            "MARKER",
        )
        == "nonzero-exit"
    )


def test_build_command_includes_model_provider_toolsets():
    case = canary.CanaryCase(name="unit", marker="OK", prompt="reply OK")
    cmd = canary.build_command("hermes", case, model="dflash", provider="taro", toolsets="terminal,file")

    assert cmd == [
        "hermes",
        "--provider",
        "taro",
        "--model",
        "dflash",
        "--toolsets",
        "terminal,file",
        "-z",
        "reply OK",
    ]


def test_run_case_uses_runner_and_sanitizes_prompt_from_logged_command(tmp_path):
    case = canary.CanaryCase(name="unit", marker="OK", prompt="private prompt")

    def runner(cmd, cwd, timeout_s):
        assert cmd[-1] == "private prompt"
        assert cwd == tmp_path
        assert timeout_s == 5.0
        return canary.CommandResult(returncode=0, stdout="OK\n", stderr="", elapsed_s=0.25)

    record = canary.run_case(
        case,
        cwd=tmp_path,
        hermes_bin="hermes",
        model="dflash",
        provider="taro",
        toolsets="terminal,file",
        timeout_s=5.0,
        strict_marker=True,
        runner=runner,
    )

    assert record["ok"] is True
    assert record["failure"] is None
    assert record["cmd"][-1] == "<prompt>"
    assert record["stdout"] == "OK\n"
