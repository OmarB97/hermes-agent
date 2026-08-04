"""End-to-end client tests against the in-process mock LSP server.

Spins up :file:`_mock_lsp_server.py` as an actual subprocess, drives
it through real LSP traffic, and asserts diagnostic flow.  This is
the closest thing we have to integration coverage without requiring
pyright/gopls/etc. to be installed in CI.
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from agent.lsp.client import LSPClient


MOCK_SERVER = str(Path(__file__).parent / "_mock_lsp_server.py")


def _client(workspace: Path, script: str = "clean", **env_extra: str) -> LSPClient:
    env = {
        "MOCK_LSP_SCRIPT": script,
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
        **env_extra,
    }
    return LSPClient(
        server_id=f"mock-{script}",
        workspace_root=str(workspace),
        command=[sys.executable, MOCK_SERVER],
        env=env,
        cwd=str(workspace),
    )


@pytest.mark.asyncio
async def test_client_lifecycle_clean(tmp_path: Path):
    """Full lifecycle: spawn, initialize, open, get clean diagnostics, shutdown."""
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean")
    await client.start()
    try:
        assert client.is_running
        version = await client.open_file(str(f), language_id="python")
        assert version == 0
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert diags == []
    finally:
        await client.shutdown()
    assert not client.is_running


@pytest.mark.asyncio
async def test_client_receives_published_errors(tmp_path: Path):
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "errors")
    await client.start()
    try:
        version = await client.open_file(str(f), language_id="python")
        await client.wait_for_diagnostics(str(f), version, mode="document")
        diags = client.diagnostics_for(str(f))
        assert len(diags) == 1
        d = diags[0]
        assert d["severity"] == 1
        assert d["code"] == "MOCK001"
        assert d["source"] == "mock-lsp"
        assert "synthetic error" in d["message"]
    finally:
        await client.shutdown()


@pytest.mark.asyncio
async def test_clean_shutdown_waits_for_exit_instead_of_signalling(
    tmp_path: Path, monkeypatch
):
    """A server still quitting after ``exit`` must not be SIGTERMed.

    ``proc.returncode is None`` only says we haven't *observed* an exit,
    so escalating straight to signals cuts a well-behaved server off
    mid-cleanup — and when the child is already gone it means signalling
    an unreaped zombie, which is how every LSP e2e test could die inside
    ``finally: await client.shutdown()`` on a loaded runner.
    """
    f = tmp_path / "x.py"
    f.write_text("print('hi')\n")

    client = _client(tmp_path, "clean", MOCK_LSP_EXIT_DELAY="0.3")
    await client.start()
    proc = client._proc
    assert proc is not None

    # Record signals but still deliver them, so a regression fails the
    # assertion below rather than hanging on a process nothing can kill.
    signalled: list[str] = []
    for name in ("terminate", "kill"):
        real = getattr(proc, name)

        def spy(_name=name, _real=real):
            signalled.append(_name)
            _real()

        monkeypatch.setattr(proc, name, spy)

    await client.shutdown()

    assert signalled == [], "graceful `exit` must be honoured before signals"
    assert proc.returncode == 0, "server should exit on its own terms"
