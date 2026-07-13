import sqlite3
import sys

import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    database = SessionDB(tmp_path / "state.db")
    yield database
    database.close()


TELEMETRY_COLUMNS = {
    "last_prompt_tokens",
    "last_completion_tokens",
    "pending_prompt_tokens",
    "pending_generation",
    "total_tokens",
    "context_length",
    "compression_count",
    "api_call_count",
}


def test_new_sessions_have_explicit_zero_telemetry(db):
    db.create_session("fresh", "cli")

    session = db.get_session("fresh")

    assert TELEMETRY_COLUMNS <= set(session.keys())
    assert {name: session[name] for name in TELEMETRY_COLUMNS} == {
        name: 0 for name in TELEMETRY_COLUMNS
    }
    assert session["pending_owner"] is None
    assert session["pending_started_at"] is None


def test_existing_database_reconciles_all_telemetry_columns(tmp_path):
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parent_session_id TEXT,
            started_at REAL NOT NULL
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            tool_calls TEXT,
            timestamp REAL NOT NULL
        );
        INSERT INTO sessions (id, source, started_at)
        VALUES ('legacy', 'cli', 1.0);
        """
    )
    connection.close()

    database = SessionDB(db_path)
    try:
        session = database.get_session("legacy")
        assert TELEMETRY_COLUMNS <= set(session.keys())
        assert {name: session[name] for name in TELEMETRY_COLUMNS} == {
            name: 0 for name in TELEMETRY_COLUMNS
        }
        assert session["pending_owner"] is None
        assert session["pending_started_at"] is None
    finally:
        database.close()


def test_success_commit_accumulates_totals_but_sets_latest_snapshot(db):
    db.create_session("session", "cli")
    first_generation = db.begin_pending_request(
        "session", tokens=8_000, owner="owner-a", started_at=1.0
    )
    db.update_token_counts(
        "session",
        input_tokens=100,
        output_tokens=50,
        api_call_count=1,
        last_prompt_tokens=8_000,
        last_completion_tokens=200,
        context_length=65_536,
        compression_count=1,
    )
    # Token commits cannot erase a matching or newer in-flight marker; only
    # the generation-fenced lifecycle clear owns that transition.
    assert db.get_session("session")["pending_prompt_tokens"] == 8_000
    assert db.clear_pending_request(
        "session", generation=first_generation, owner="owner-a"
    )
    second_generation = db.begin_pending_request(
        "session", tokens=12_000, owner="owner-a", started_at=2.0
    )
    db.update_token_counts(
        "session",
        input_tokens=150,
        output_tokens=75,
        api_call_count=1,
        last_prompt_tokens=12_000,
        last_completion_tokens=300,
        context_length=65_536,
        compression_count=2,
    )
    assert db.get_session("session")["pending_prompt_tokens"] == 12_000
    assert db.clear_pending_request(
        "session", generation=second_generation, owner="owner-a"
    )

    session = db.get_session("session")
    assert session["input_tokens"] == 250
    assert session["output_tokens"] == 125
    assert session["total_tokens"] == 375
    assert session["api_call_count"] == 2
    assert session["last_prompt_tokens"] == 12_000
    assert session["last_completion_tokens"] == 300
    assert session["context_length"] == 65_536
    assert session["compression_count"] == 2
    assert session["pending_prompt_tokens"] == 0


def test_omitted_snapshot_preserves_value_but_explicit_zero_writes(db):
    db.update_token_counts(
        "session",
        last_prompt_tokens=8_000,
        last_completion_tokens=200,
        context_length=65_536,
        compression_count=3,
    )

    db.update_token_counts("session", input_tokens=10)
    preserved = db.get_session("session")
    assert preserved["last_prompt_tokens"] == 8_000
    assert preserved["last_completion_tokens"] == 200
    assert preserved["context_length"] == 65_536
    assert preserved["compression_count"] == 3

    db.update_token_counts(
        "session",
        last_prompt_tokens=0,
        last_completion_tokens=0,
        context_length=0,
        compression_count=0,
    )
    cleared = db.get_session("session")
    assert cleared["last_prompt_tokens"] == 0
    assert cleared["last_completion_tokens"] == 0
    assert cleared["context_length"] == 0
    assert cleared["compression_count"] == 0


def test_absolute_mode_sets_cumulative_and_latest_values(db):
    generation = db.begin_pending_request(
        "gateway", tokens=20_000, owner="gateway-owner", started_at=1.0
    )
    db.update_token_counts(
        "gateway",
        input_tokens=500,
        output_tokens=200,
        api_call_count=4,
        last_prompt_tokens=20_000,
        last_completion_tokens=400,
        context_length=65_536,
        compression_count=1,
        absolute=True,
    )
    assert db.get_session("gateway")["pending_prompt_tokens"] == 20_000
    assert db.clear_pending_request(
        "gateway", generation=generation, owner="gateway-owner"
    )
    db.update_token_counts(
        "gateway",
        input_tokens=700,
        output_tokens=350,
        api_call_count=5,
        last_prompt_tokens=22_000,
        last_completion_tokens=500,
        context_length=131_072,
        compression_count=2,
        absolute=True,
    )

    session = db.get_session("gateway")
    assert session["input_tokens"] == 700
    assert session["output_tokens"] == 350
    assert session["total_tokens"] == 1_050
    assert session["api_call_count"] == 5
    assert session["last_prompt_tokens"] == 22_000
    assert session["last_completion_tokens"] == 500
    assert session["context_length"] == 131_072
    assert session["compression_count"] == 2
    assert session["pending_prompt_tokens"] == 0


def test_provider_reported_total_override_round_trips_exactly(db):
    db.update_token_counts(
        "reported-total",
        input_tokens=80,
        output_tokens=25,
        cache_read_tokens=20,
        reasoning_tokens=5,
        total_tokens=130,
    )

    session = db.get_session("reported-total")
    assert session["input_tokens"] == 80
    assert session["output_tokens"] == 25
    assert session["cache_read_tokens"] == 20
    assert session["reasoning_tokens"] == 5
    assert session["total_tokens"] == 130


def test_reconciled_legacy_total_seeds_before_new_delta_and_restart(tmp_path):
    db_path = tmp_path / "legacy-total.db"
    connection = sqlite3.connect(db_path)
    connection.executescript(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL);
        INSERT INTO schema_version (version) VALUES (1);

        CREATE TABLE sessions (
            id TEXT PRIMARY KEY,
            source TEXT NOT NULL,
            parent_session_id TEXT,
            started_at REAL NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_write_tokens INTEGER DEFAULT 0,
            reasoning_tokens INTEGER DEFAULT 0
        );
        CREATE TABLE messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT,
            tool_name TEXT,
            tool_calls TEXT,
            timestamp REAL NOT NULL
        );
        INSERT INTO sessions (
            id, source, started_at, input_tokens, output_tokens,
            cache_read_tokens, cache_write_tokens, reasoning_tokens
        ) VALUES ('legacy-total', 'cli', 1.0, 80, 25, 20, 0, 5);
        """
    )
    connection.close()

    database = SessionDB(db_path)
    try:
        reconciled = database.get_session("legacy-total")
        assert reconciled["total_tokens"] == 0
        database.update_token_counts(
            "legacy-total",
            input_tokens=4,
            output_tokens=1,
            total_tokens=5,
        )
        assert database.get_session("legacy-total")["total_tokens"] == 130
    finally:
        database.close()

    restarted = SessionDB(db_path)
    try:
        session = restarted.get_session("legacy-total")
        assert session["input_tokens"] == 84
        assert session["output_tokens"] == 26
        assert session["reasoning_tokens"] == 5
        assert session["total_tokens"] == 130
    finally:
        restarted.close()


def test_legacy_positional_absolute_argument_keeps_original_index(db):
    db.update_token_counts(
        "legacy-positional", input_tokens=10, output_tokens=5, api_call_count=1
    )

    # Exact pre-telemetry-extension positional shape: the final True has always
    # meant ``absolute=True`` immediately after api_call_count.
    db.update_token_counts(
        "legacy-positional",
        100,
        50,
        None,
        0,
        0,
        0,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        2,
        True,
    )

    session = db.get_session("legacy-positional")
    assert session["input_tokens"] == 100
    assert session["output_tokens"] == 50
    assert session["api_call_count"] == 2
    assert session["total_tokens"] == 150
    assert session["last_prompt_tokens"] == 0


def test_startup_reconciles_invalid_legacy_cost_values(tmp_path):
    db_path = tmp_path / "legacy-cost.db"
    database = SessionDB(db_path)
    database.create_session("legacy-cost", "cli")
    database._conn.execute(
        """UPDATE sessions
           SET estimated_cost_usd = ?, actual_cost_usd = ?
           WHERE id = ?""",
        (float("inf"), -1.0, "legacy-cost"),
    )
    database._conn.commit()
    database.close()

    restarted = SessionDB(db_path)
    try:
        session = restarted.get_session("legacy-cost")
        assert session["estimated_cost_usd"] == 0.0
        assert session["actual_cost_usd"] == 0.0
    finally:
        restarted.close()


@pytest.mark.parametrize("absolute", [False, True])
def test_none_api_call_count_persists_as_numeric_zero(db, absolute):
    session_id = f"none-api-count-{absolute}"
    db.update_token_counts(
        session_id,
        api_call_count=None,
        absolute=absolute,
    )

    session = db.get_session(session_id)
    assert session["api_call_count"] == 0
    assert isinstance(session["api_call_count"], int)


def test_cumulative_finite_cost_addition_saturates_before_overflow(db):
    maximum = sys.float_info.max
    db.update_token_counts(
        "cost-overflow",
        estimated_cost_usd=maximum,
        actual_cost_usd=maximum,
    )
    db.update_token_counts(
        "cost-overflow",
        estimated_cost_usd=maximum,
        actual_cost_usd=maximum,
    )

    session = db.get_session("cost-overflow")
    assert session["estimated_cost_usd"] == maximum
    assert session["actual_cost_usd"] == maximum


def test_older_generation_cannot_clear_newer_pending_request(db):
    first = db.begin_pending_request(
        "overlap", tokens=4_000, owner="owner-a", started_at=1.0
    )
    second = db.begin_pending_request(
        "overlap", tokens=9_000, owner="owner-b", started_at=2.0
    )

    assert second == first + 1
    assert not db.clear_pending_request(
        "overlap", generation=first, owner="owner-a"
    )
    pending = db.get_session("overlap")
    assert pending["pending_prompt_tokens"] == 9_000
    assert pending["pending_generation"] == second
    assert pending["pending_owner"] == "owner-b"
    assert pending["pending_started_at"] == 2.0

    assert db.clear_pending_request(
        "overlap", generation=second, owner="owner-b"
    )
    cleared = db.get_session("overlap")
    assert cleared["pending_prompt_tokens"] == 0
    assert cleared["pending_generation"] == second
    assert cleared["pending_owner"] is None
    assert cleared["pending_started_at"] is None


@pytest.mark.parametrize(
    ("setter", "column", "first", "second"),
    [
        ("set_pending_prompt_tokens", "pending_prompt_tokens", 5_000, 0),
        ("set_context_length", "context_length", 65_536, 131_072),
        ("set_compression_count", "compression_count", 1, 3),
    ],
)
def test_setters_create_missing_row_and_overwrite(db, setter, column, first, second):
    getattr(db, setter)("lazy", first)
    assert db.get_session("lazy")[column] == first

    getattr(db, setter)("lazy", second)
    assert db.get_session("lazy")[column] == second
