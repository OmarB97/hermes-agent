"""Integration coverage for profile-local MCP discovery in slash workers."""

from __future__ import annotations

import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import textwrap
import threading
from typing import NoReturn

import pytest
import yaml

pytest.importorskip("mcp.server.fastmcp")

# Answering /tools costs two cold Python interpreter starts (the slash worker,
# then the probe MCP server it spawns) plus an MCP handshake, so this test's
# wall clock tracks machine load rather than anything it asserts — against a
# runner that deliberately runs -j 32 on fewer cores. Two bounds used to assume
# a quiet box, and they fail differently as load climbs: the response bound
# (was 10s) expires first and reports "produced no /tools response", while
# under heavier load the worker answers in time but the discovery bound has
# already expired, so /tools is correct yet omits the probe server and the tool
# assertion fails instead. Both exist to stop a hang, not to assert a speed,
# and both are released the instant the work completes — so keep them generous.
# 120s matches the sibling slash-worker subprocess test.
RESPONSE_TIMEOUT_S = 120
DISCOVERY_TIMEOUT_S = 60


def test_profile_local_mcp_tool_is_visible_in_slash_worker(tmp_path):
    profile_home = tmp_path / "profile-home"
    profile_home.mkdir()
    marker = "profile-local-61922"
    server = tmp_path / "fastmcp_probe.py"
    server.write_text(
        textwrap.dedent(
            f"""
            from mcp.server.fastmcp import FastMCP

            mcp = FastMCP("profileprobe")

            @mcp.tool()
            def hermes_61922_profile_probe() -> str:
                return {marker!r}

            if __name__ == "__main__":
                mcp.run(transport="stdio")
            """
        ),
        encoding="utf-8",
    )
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump(
            {
                # The worker only waits mcp_discovery_timeout for a server to
                # report its tools, and the 1.5s default is tuned for
                # interactive startup latency, not for a probe server that is
                # itself a cold interpreter. When it expires the worker answers
                # /tools normally, just without this server's tools — so the
                # test failed on the tool assertion rather than on a timeout.
                # Discovery joins the instant it completes, so a generous bound
                # costs nothing when the server is healthy.
                "mcp_discovery_timeout": DISCOVERY_TIMEOUT_S,
                "mcp_servers": {
                    "profileprobe": {
                        "enabled": True,
                        "command": sys.executable,
                        "args": [str(server)],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in list(env):
        if key.endswith("_API_KEY") or key.endswith("_TOKEN"):
            env.pop(key)
    env["HERMES_HOME"] = str(profile_home)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2])
    env["HERMES_SLASH_WATCHDOG_GRACE_S"] = "0"
    env["HERMES_SLASH_WATCHDOG_POLL_S"] = "0.05"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-m",
            "tui_gateway.slash_worker",
            "--session-key",
            "agent:main:tui:dm:mcp-profile-test",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    output: queue.Queue[str] = queue.Queue()
    try:
        assert proc.stdin is not None
        assert proc.stdout is not None
        assert proc.stderr is not None
        stdout = proc.stdout
        stderr = proc.stderr
        threading.Thread(
            target=lambda: output.put(stdout.readline()),
            daemon=True,
        ).start()
        # Drain stderr so a crashed or hung worker can say why. Every failure
        # below otherwise presents as "no response", which is what an import
        # error, a bad config, and a genuine hang all look like from here.
        stderr_chunks: list[str] = []
        stderr_reader = threading.Thread(
            target=lambda: stderr_chunks.append(stderr.read()),
            daemon=True,
        )
        stderr_reader.start()

        def _fail(reason: str) -> NoReturn:
            # Close the worker's pipes first, or stderr.read() never returns.
            proc.kill()
            stderr_reader.join(timeout=10)
            captured = "".join(stderr_chunks).strip()
            pytest.fail(f"{reason}\nworker stderr:\n{captured or '<empty>'}")

        proc.stdin.write(json.dumps({"id": 1, "command": "/tools"}) + "\n")
        proc.stdin.flush()
        try:
            line = output.get(timeout=RESPONSE_TIMEOUT_S)
        except queue.Empty:
            _fail(f"slash worker produced no /tools response within {RESPONSE_TIMEOUT_S}s")
        if not line.strip():
            _fail("slash worker exited without answering /tools")
        response = json.loads(line)
        assert response["ok"] is True
        assert "mcp__profileprobe__hermes_61922_profile_probe" in response["output"], (
            "profile-local MCP tool missing from /tools — the worker answered, so "
            f"discovery did not register the probe server within {DISCOVERY_TIMEOUT_S}s\n"
            f"{response['output']}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
