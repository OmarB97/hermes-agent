"""A telemetry bind must not project a row it just created over live counters.

_ensure_db_session hydrates so a RESUMED row's durable counters survive
reset_session_state()'s zeros. But it also runs on the row-creating bind, and a
row created by that same call is all-zero — projecting it erases whatever the
live runtime already accumulated.

That is reachable in one turn. When the turn-setup _ensure_db_session hits a
transient create failure it logs "will retry next turn" and returns BEFORE
binding, leaving _session_db_created False. The retry then fires mid-turn from
_flush_messages_to_session_db (run_agent.py: `if not self._session_db_created`),
creates the row fresh, and first-bind hydration zeroes the turn's tokens, cost,
api-call count and compression_count. Nothing had been persisted to fall back on
either: with no row, record_canonical_usage's UPDATE matched no rows.

Under compression the loss becomes durable — persist_current_telemetry_snapshot
writes the zeroed counters into the rotated child row, so the next turn hydrates
them back as 0.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch


def _make_agent(session_db, session_id="original-session", event_callback=None):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
            event_callback=event_callback,
        )
        agent.compression_in_place = False
        return agent


def _flaky_db(tmpdir, *, fail_first_create):
    """A SessionDB whose first create_session raises, like a contended write."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=Path(tmpdir) / "test.db")
    real_create = db.create_session
    calls = {"n": 0}

    def flaky_create(*a, **kw):
        calls["n"] += 1
        if fail_first_create and calls["n"] == 1:
            raise RuntimeError("database is locked")
        return real_create(*a, **kw)

    db.create_session = flaky_create
    return db


def _accumulate_turn_telemetry(agent):
    """Stand in for a turn that made API calls before the create retry."""
    compressor = MagicMock()
    compressor.compression_count = 3
    compressor.last_prompt_tokens = 8000
    compressor.last_completion_tokens = 250
    compressor.context_length = 128000
    agent.context_compressor = compressor
    agent.session_total_tokens = 8250
    agent.session_api_calls = 4
    agent.session_estimated_cost_usd = 0.0731


def test_create_retry_does_not_erase_the_turns_telemetry():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _flaky_db(tmpdir, fail_first_create=True)
        agent = _make_agent(db)

        # Turn setup: the create faults, so no row exists and no bind happened.
        agent._ensure_db_session()
        assert agent._session_db_created is False
        assert getattr(agent, "_session_telemetry_hydrated_session_id", None) is None

        _accumulate_turn_telemetry(agent)

        # The retry that _flush_messages_to_session_db performs mid-turn.
        agent._ensure_db_session()
        assert agent._session_db_created is True

        assert agent.session_total_tokens == 8250
        assert agent.session_api_calls == 4
        assert agent.session_estimated_cost_usd == 0.0731
        assert agent.context_compressor.compression_count == 3
        assert agent.context_compressor.last_prompt_tokens == 8000
        assert agent.context_compressor.last_completion_tokens == 250

        # The bind must be recorded even though no row was applied, or a later
        # call in the same turn re-projects the partially written row.
        assert (
            getattr(agent, "_session_telemetry_hydrated_session_id", None)
            == agent.session_id
        )
        agent._ensure_db_session()
        assert agent.session_api_calls == 4
        assert agent.context_compressor.compression_count == 3


def test_resumed_row_is_still_authoritative():
    """The restore path this guard must not break."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _flaky_db(tmpdir, fail_first_create=False)
        db.create_session(session_id="resumed-session", source="cli")
        db.set_compression_count("resumed-session", 5)

        agent = _make_agent(db, session_id="resumed-session")
        # What reset_session_state() leaves behind on a session switch.
        agent.context_compressor = MagicMock()
        agent.context_compressor.compression_count = 0
        agent.context_compressor.context_length = 0
        agent.session_api_calls = 0

        agent._ensure_db_session()

        assert agent.context_compressor.compression_count == 5, (
            "a pre-existing row must still win over the reset zeros"
        )


def test_compression_count_survives_a_create_retry_into_the_child_row():
    """End-to-end: the durable loss the in-memory clobber used to cause."""
    events = []
    with tempfile.TemporaryDirectory() as tmpdir:
        db = _flaky_db(tmpdir, fail_first_create=True)
        agent = _make_agent(db, event_callback=lambda et, ctx: events.append((et, ctx)))

        agent._ensure_db_session()  # transient failure, no row, no bind
        assert agent._session_db_created is False

        # The compressor increments in memory only — context_compressor.py's
        # `self.compression_count += 1` never writes to SQLite.
        compressor = MagicMock()
        compressor.compress.return_value = [
            {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
            {"role": "user", "content": "tail"},
        ]
        compressor.compression_count = 1
        compressor.last_prompt_tokens = 0
        compressor.last_completion_tokens = 0
        compressor._last_summary_error = None
        compressor._last_compress_aborted = False
        agent.context_compressor = compressor

        agent._compress_context(
            [{"role": "user", "content": f"m{i}"} for i in range(10)],
            "sys",
            approx_tokens=10_000,
        )

        compress_events = [e for e in events if e[0] == "session:compress"]
        assert compress_events, f"session:compress not emitted, got {events!r}"
        assert compress_events[-1][1]["compression_count"] == 1

        child = db.get_session(agent.session_id)
        assert child.get("compression_count") == 1, (
            "the rotated child row must carry the count forward"
        )
