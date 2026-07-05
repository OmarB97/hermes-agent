import re
from pathlib import Path

import pytest

CHAT_SIDEBAR = Path(__file__).resolve().parent.parent / "web/src/components/ChatSidebar.tsx"


def test_sidecar_session_create_requests_close_on_disconnect():
    """The sidecar must opt its session into close_on_disconnect so the gateway
    reaps the slash_worker on WS disconnect (the #21370/#21467 leak)."""
    source = CHAT_SIDEBAR.read_text(encoding="utf-8")
    call = re.search(r'"session\.create",\s*\{(.*?)\}', source, re.DOTALL)
    assert call, "sidecar session.create call not found"
    assert re.search(r"close_on_disconnect:\s*true", call.group(1))


@pytest.mark.xfail(
    reason="Fork's web/src/components/ChatSidebar.tsx diverged from upstream (fork added a "
    "reasoning-effort picker / session switcher) and has no dashboard profile switcher, so its "
    "session.create has no profile to scope. Remove this marker if the fork adds web profile scoping.",
)
def test_sidecar_session_create_scopes_profile():
    """The sidecar must pass the dashboard's selected profile so model/credential
    info matches the PTY child under profile-scoped chat."""
    source = CHAT_SIDEBAR.read_text(encoding="utf-8")
    assert '"session.create"' in source
    assert re.search(
        r"close_on_disconnect:\s*true,\s*\.\.\.\(profile\s*\?\s*\{\s*profile\s*\}\s*:\s*\{\}\)",
        source,
        re.DOTALL,
    )
