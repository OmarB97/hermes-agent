"""Shared durable session-telemetry lifecycle helpers.

SQLite is the canonical store.  The in-memory agent fields and optional JSON
snapshot are compatibility projections used by status surfaces; failures in
either projection must never block a provider request.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import math
import os
from pathlib import Path
import socket
import sys
import threading
import time
import uuid
from typing import Any, Dict, Iterator, MutableMapping, Optional

logger = logging.getLogger(__name__)

_PENDING_STALE_AFTER_SECONDS = 24 * 60 * 60
_PROJECTION_THREAD_LOCKS: Dict[str, threading.RLock] = {}
_PROJECTION_THREAD_LOCKS_GUARD = threading.Lock()
_PROJECTION_LOCK_LOCAL = threading.local()
_MAX_TELEMETRY_FLOAT = sys.float_info.max


def telemetry_counter(value: Any) -> int:
    """Return a stable non-negative integer for external telemetry."""
    try:
        return max(int(value or 0), 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _telemetry_float(value: Any) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(numeric) or numeric < 0:
        return 0.0
    return numeric


def _add_telemetry_cost(left: Any, right: Any) -> float:
    """Add normalized costs without overflowing a finite telemetry field."""
    base = _telemetry_float(left)
    delta = _telemetry_float(right)
    if base > _MAX_TELEMETRY_FLOAT - delta:
        return _MAX_TELEMETRY_FLOAT
    return base + delta


def _projection_lock_path(agent: Any) -> Optional[Path]:
    session_db = getattr(agent, "_session_db", None)
    db_path = getattr(session_db, "db_path", None)
    # Mocks and arbitrary objects may advertise ``__fspath__``; accepting them
    # here creates bogus ``MagicMock/...`` directories during dry/test paths.
    # SessionDB itself stores a concrete pathlib.Path (or a caller-supplied
    # string), which are the only forms this lock needs to support.
    if not isinstance(db_path, (str, Path)):
        return None
    path = Path(db_path)
    return path.with_name(f".{path.name}.session-telemetry.lock")


def _thread_projection_lock(key: str) -> threading.RLock:
    with _PROJECTION_THREAD_LOCKS_GUARD:
        return _PROJECTION_THREAD_LOCKS.setdefault(key, threading.RLock())


@contextmanager
def pending_projection_guard(agent: Any) -> Iterator[None]:
    """Serialize canonical pending mutations with their JSON projection.

    SQLite generation/owner predicates prevent a stale process from clearing a
    newer request, but the JSON compatibility file is a second resource.  A
    stale process could otherwise read the newer marker, pause, then overwrite
    a terminal zero after the newer process cleared SQLite.  This per-database
    cross-process lock makes the DB transition and JSON write one serialized
    projection unit.  The thread-local count keeps nested ``_save_session_log``
    calls re-entrant.
    """
    lock_path = _projection_lock_path(agent)
    if lock_path is None:
        yield
        return

    key = str(lock_path.resolve())
    held = getattr(_PROJECTION_LOCK_LOCAL, "held", None)
    if held is None:
        held = {}
        _PROJECTION_LOCK_LOCAL.held = held
    if held.get(key, 0) > 0:
        held[key] += 1
        try:
            yield
        finally:
            held[key] -= 1
        return

    with _thread_projection_lock(key):
        handle = None
        locked = False
        try:
            lock_path.parent.mkdir(parents=True, exist_ok=True)
            handle = open(lock_path, "a+b")
            if os.name == "nt":
                import msvcrt

                handle.seek(0, os.SEEK_END)
                if handle.tell() == 0:
                    handle.write(b" ")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            locked = True
        except Exception:
            logger.warning(
                "session telemetry projection lock unavailable at %s",
                lock_path,
                exc_info=True,
            )

        held[key] = 1
        try:
            yield
        finally:
            held.pop(key, None)
            if handle is not None and locked:
                try:
                    if os.name == "nt":
                        import msvcrt

                        handle.seek(0)
                        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                    else:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except Exception:
                    logger.debug(
                        "session telemetry projection unlock failed",
                        exc_info=True,
                    )
            if handle is not None:
                handle.close()


def sync_pending_from_canonical(agent: Any) -> bool:
    """Refresh the in-memory pending projection from canonical SQLite."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return False
    try:
        session = session_db.get_session(session_id)
    except Exception:
        logger.debug(
            "canonical pending telemetry read failed (session=%s)",
            session_id,
            exc_info=True,
        )
        return False
    if not isinstance(session, dict):
        return False
    agent.pending_prompt_tokens = telemetry_counter(
        session.get("pending_prompt_tokens")
    )
    agent._pending_generation = telemetry_counter(
        session.get("pending_generation")
    )
    agent._pending_owner = session.get("pending_owner")
    agent._pending_started_at = _telemetry_float(
        session.get("pending_started_at")
    )
    return True


def _apply_canonical_telemetry(
    agent: Any,
    session: Dict[str, Any],
) -> None:
    """Replace the in-memory compatibility projection from one SQLite row.

    The currently resolved compressor cap is the one exception: unlike the
    cumulative counters, a persisted cap has no model/provider provenance and
    can belong to the model used before a restart or live switch.  A positive
    current cap therefore wins; persisted context is only a fallback when the
    runtime has not resolved one.
    """
    input_tokens = telemetry_counter(session.get("input_tokens"))
    output_tokens = telemetry_counter(session.get("output_tokens"))
    cache_read_tokens = telemetry_counter(session.get("cache_read_tokens"))
    cache_write_tokens = telemetry_counter(session.get("cache_write_tokens"))
    reasoning_tokens = telemetry_counter(session.get("reasoning_tokens"))
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens

    agent.session_input_tokens = input_tokens
    agent.session_output_tokens = output_tokens
    agent.session_cache_read_tokens = cache_read_tokens
    agent.session_cache_write_tokens = cache_write_tokens
    agent.session_reasoning_tokens = reasoning_tokens
    agent.session_prompt_tokens = prompt_tokens
    agent.session_completion_tokens = output_tokens
    derived_total_tokens = prompt_tokens + output_tokens
    persisted_total_tokens = telemetry_counter(session.get("total_tokens"))
    agent.session_total_tokens = (
        persisted_total_tokens
        if persisted_total_tokens > 0 or derived_total_tokens == 0
        else derived_total_tokens
    )
    agent.session_api_calls = telemetry_counter(session.get("api_call_count"))
    agent.session_estimated_cost_usd = _telemetry_float(
        session.get("estimated_cost_usd")
    )
    if session.get("cost_status"):
        agent.session_cost_status = session["cost_status"]
    if session.get("cost_source"):
        agent.session_cost_source = session["cost_source"]

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        last_prompt = telemetry_counter(session.get("last_prompt_tokens"))
        last_completion = telemetry_counter(
            session.get("last_completion_tokens")
        )
        compressor.last_prompt_tokens = last_prompt
        compressor.last_real_prompt_tokens = last_prompt
        compressor.last_completion_tokens = last_completion
        compressor.last_total_tokens = last_prompt + last_completion
        compressor.compression_count = telemetry_counter(
            session.get("compression_count")
        )
        current_context_length = telemetry_counter(
            getattr(compressor, "context_length", 0)
        )
        persisted_context_length = telemetry_counter(
            session.get("context_length")
        )
        if current_context_length <= 0 and persisted_context_length > 0:
            compressor.context_length = persisted_context_length

    agent.pending_prompt_tokens = telemetry_counter(
        session.get("pending_prompt_tokens")
    )
    agent._pending_generation = telemetry_counter(
        session.get("pending_generation")
    )
    agent._pending_owner = session.get("pending_owner")
    agent._pending_started_at = _telemetry_float(
        session.get("pending_started_at")
    )


def sync_telemetry_from_canonical(agent: Any) -> bool:
    """Refresh every JSON/result telemetry field from one canonical row."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return False
    try:
        session = session_db.get_session(session_id)
    except Exception:
        logger.debug(
            "canonical session telemetry read failed (session=%s)",
            session_id,
            exc_info=True,
        )
        return False
    if not isinstance(session, dict):
        return False
    _apply_canonical_telemetry(agent, session)
    return True


def new_pending_owner() -> str:
    """Return a process-instance identity suitable for pending-request fencing."""
    host = socket.gethostname().replace(":", "_") or "unknown"
    try:
        import psutil

        process_started_at = psutil.Process(os.getpid()).create_time()
    except Exception:
        process_started_at = time.time()
    return (
        f"{host}:{os.getpid()}:{process_started_at:.6f}:"
        f"{uuid.uuid4().hex}"
    )


def _pending_owner_process_alive(owner: str) -> Optional[bool]:
    """Return local owner liveness, or ``None`` for an unprovable owner."""
    try:
        host, pid_text, started_text, _nonce = owner.split(":", 3)
        if host != (socket.gethostname().replace(":", "_") or "unknown"):
            return None
        pid = int(pid_text)
        expected_started_at = float(started_text)
    except (AttributeError, TypeError, ValueError):
        return None

    try:
        import psutil

        if not psutil.pid_exists(pid):
            return False
        actual_started_at = psutil.Process(pid).create_time()
        # PID reuse must not keep a dead request looking live. Process creation
        # timestamps are platform-rounded, so allow a one-second tolerance.
        return abs(actual_started_at - expected_started_at) <= 1.0
    except Exception:
        return None


def _pending_marker_is_stale(
    session: Dict[str, Any],
    *,
    current_owner: str,
    now: Optional[float] = None,
) -> bool:
    pending_tokens = telemetry_counter(session.get("pending_prompt_tokens"))
    owner = session.get("pending_owner")
    started_at = _telemetry_float(session.get("pending_started_at"))
    if pending_tokens <= 0 and not owner and started_at <= 0:
        return False

    if not owner or started_at <= 0:
        # Legacy pending markers had no identity and could never be cleared
        # safely after a crash. They are stale at the first fenced startup.
        return True

    current_time = time.time() if now is None else now
    if current_time - started_at >= _PENDING_STALE_AFTER_SECONDS:
        return True
    if owner == current_owner:
        return False

    owner_alive = _pending_owner_process_alive(str(owner))
    return owner_alive is False


def _hydrate_agent_session_telemetry_locked(agent: Any) -> bool:
    """Hydrate one bound session before its first JSON/provider operation.

    ``reset_session_state`` intentionally zeros runtime counters for a session
    switch.  Resuming an existing SQLite row must then restore its cumulative
    counters and latest snapshots before turn-start persistence can project
    explicit zeros into JSON.  The hydration marker is session-id scoped so it
    runs once per bind and is re-armed by reset/resume.
    """
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return False
    if getattr(agent, "_session_telemetry_hydrated_session_id", None) == session_id:
        return True

    try:
        session = session_db.get_session(session_id)
    except Exception:
        logger.debug(
            "session telemetry hydration read failed (session=%s)",
            session_id,
            exc_info=True,
        )
        return False
    if not isinstance(session, dict):
        return False

    _apply_canonical_telemetry(agent, session)

    owner = getattr(agent, "_session_telemetry_owner", None)
    if not owner:
        owner = new_pending_owner()
        agent._session_telemetry_owner = owner
    pending_generation = agent._pending_generation
    pending_owner = agent._pending_owner
    pending_started_at = agent._pending_started_at
    pending_tokens = agent.pending_prompt_tokens

    if _pending_marker_is_stale(session, current_owner=owner):
        try:
            cleared = session_db.clear_pending_request(
                session_id,
                generation=pending_generation,
                owner=pending_owner,
            )
        except Exception:
            cleared = False
            logger.debug(
                "stale pending telemetry reconciliation failed (session=%s)",
                session_id,
                exc_info=True,
            )
        if cleared:
            pending_tokens = 0
            pending_owner = None
            pending_started_at = 0.0

    agent.pending_prompt_tokens = pending_tokens
    agent._pending_generation = pending_generation
    agent._pending_owner = pending_owner
    agent._pending_started_at = pending_started_at
    agent._session_telemetry_hydrated_session_id = session_id
    return True


def hydrate_agent_session_telemetry(agent: Any) -> bool:
    """Hydrate and reconcile pending state under the projection guard."""
    with pending_projection_guard(agent):
        return _hydrate_agent_session_telemetry_locked(agent)


def persist_current_telemetry_snapshot(agent: Any) -> bool:
    """Seed a newly rotated session with the cumulative runtime counters.

    Legacy compression rotation creates a continuation row while the logical
    conversation keeps its cumulative telemetry in memory.  Persisting that
    memory as one absolute snapshot before the first child JSON write keeps the
    canonical row and compatibility projection on the same boundary.
    """
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if not session_db or not session_id:
        return False
    compressor = getattr(agent, "context_compressor", None)
    total_tokens = telemetry_counter(
        getattr(agent, "session_total_tokens", 0)
    )
    try:
        session_db.update_token_counts(
            session_id,
            input_tokens=telemetry_counter(
                getattr(agent, "session_input_tokens", 0)
            ),
            output_tokens=telemetry_counter(
                getattr(agent, "session_output_tokens", 0)
            ),
            cache_read_tokens=telemetry_counter(
                getattr(agent, "session_cache_read_tokens", 0)
            ),
            cache_write_tokens=telemetry_counter(
                getattr(agent, "session_cache_write_tokens", 0)
            ),
            reasoning_tokens=telemetry_counter(
                getattr(agent, "session_reasoning_tokens", 0)
            ),
            total_tokens=total_tokens if total_tokens > 0 else None,
            estimated_cost_usd=_telemetry_float(
                getattr(agent, "session_estimated_cost_usd", 0.0)
            ),
            cost_status=getattr(agent, "session_cost_status", None),
            cost_source=getattr(agent, "session_cost_source", None),
            billing_provider=getattr(agent, "provider", None),
            billing_base_url=getattr(agent, "base_url", None),
            billing_mode=getattr(agent, "api_mode", None),
            model=getattr(agent, "model", None),
            api_call_count=telemetry_counter(
                getattr(agent, "session_api_calls", 0)
            ),
            absolute=True,
            last_prompt_tokens=telemetry_counter(
                getattr(
                    compressor,
                    "last_real_prompt_tokens",
                    getattr(compressor, "last_prompt_tokens", 0),
                )
            ),
            last_completion_tokens=telemetry_counter(
                getattr(compressor, "last_completion_tokens", 0)
            ),
            context_length=telemetry_counter(
                getattr(compressor, "context_length", 0)
            ),
            compression_count=telemetry_counter(
                getattr(compressor, "compression_count", 0)
            ),
        )
    except Exception:
        logger.debug(
            "rotated session telemetry snapshot failed (session=%s)",
            session_id,
            exc_info=True,
        )
        return False
    agent._session_telemetry_hydrated_session_id = session_id
    return True


def _refresh_json_projection(agent: Any) -> None:
    if not getattr(agent, "_session_json_enabled", False):
        return
    try:
        agent._save_session_log()
    except Exception:
        logger.debug(
            "pending JSON telemetry projection failed (session=%s)",
            getattr(agent, "session_id", None),
            exc_info=True,
        )


def _recover_interrupted_pending_begin(
    agent: Any,
    *,
    session_db: Any,
    session_id: Optional[str],
    owner: str,
    value: int,
    started: float,
    previous_generation: Optional[int],
    begin_invoked: bool,
) -> None:
    """Best-effort rollback and canonical reprojection after BaseException."""
    canonical_synced = False
    if session_db and session_id:
        session = None
        try:
            session = session_db.get_session(session_id)
        except BaseException:
            logger.debug(
                "interrupted pending begin recovery read failed (session=%s)",
                session_id,
                exc_info=True,
            )

        committed_by_this_begin = bool(
            begin_invoked
            and isinstance(session, dict)
            and session.get("pending_owner") == owner
            and telemetry_counter(session.get("pending_prompt_tokens")) == value
            and session.get("pending_started_at") == started
            and (
                previous_generation is None
                or telemetry_counter(session.get("pending_generation"))
                > previous_generation
            )
        )
        if committed_by_this_begin:
            try:
                session_db.clear_pending_request(
                    session_id,
                    generation=telemetry_counter(
                        session.get("pending_generation")
                    ),
                    owner=owner,
                )
            except BaseException:
                logger.debug(
                    "interrupted pending begin rollback failed (session=%s)",
                    session_id,
                    exc_info=True,
                )

        try:
            canonical_synced = sync_pending_from_canonical(agent)
        except BaseException:
            logger.debug(
                "interrupted pending begin canonical sync failed (session=%s)",
                session_id,
                exc_info=True,
            )

    if not canonical_synced:
        # No durable projection is available. Clear only this aborted owner's
        # local marker; never erase a newer owner already adopted in memory.
        local_owner = getattr(agent, "_pending_owner", None)
        if local_owner in (None, owner):
            agent.pending_prompt_tokens = 0
            agent._pending_owner = None
            agent._pending_started_at = 0.0

    # The first JSON write may already have completed its atomic replacement
    # before the control-flow exception arrived. Reproject the recovered
    # canonical state; cleanup errors must never replace the original exception.
    try:
        _refresh_json_projection(agent)
    except BaseException:
        logger.debug(
            "interrupted pending begin JSON recovery failed (session=%s)",
            session_id,
            exc_info=True,
        )


def _begin_pending_request_locked(
    agent: Any,
    tokens: int,
    *,
    started_at: Optional[float] = None,
) -> int:
    """Publish a generation-fenced pending request and return its generation."""
    value = telemetry_counter(tokens)
    owner = getattr(agent, "_session_telemetry_owner", None)
    if not owner:
        owner = new_pending_owner()
        agent._session_telemetry_owner = owner
    started = time.time() if started_at is None else float(started_at)
    generation = telemetry_counter(getattr(agent, "_pending_generation", 0)) + 1

    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    previous_generation: Optional[int] = None
    begin_invoked = False
    try:
        if session_db and session_id:
            try:
                previous = session_db.get_session(session_id)
                previous_generation = telemetry_counter(
                    previous.get("pending_generation")
                    if isinstance(previous, dict)
                    else 0
                )
            except Exception:
                logger.debug(
                    "pending telemetry pre-begin read failed (session=%s)",
                    session_id,
                    exc_info=True,
                )
            try:
                begin_invoked = True
                generation = session_db.begin_pending_request(
                    session_id,
                    tokens=value,
                    owner=owner,
                    started_at=started,
                )
            except Exception:
                logger.debug(
                    "pending telemetry begin failed (session=%s)",
                    session_id,
                    exc_info=True,
                )

        agent.pending_prompt_tokens = value
        agent._pending_generation = telemetry_counter(generation)
        agent._pending_owner = owner
        agent._pending_started_at = started
        _refresh_json_projection(agent)
    except BaseException:
        _recover_interrupted_pending_begin(
            agent,
            session_db=session_db,
            session_id=session_id,
            owner=owner,
            value=value,
            started=started,
            previous_generation=previous_generation,
            begin_invoked=begin_invoked,
        )
        raise
    return agent._pending_generation


def begin_pending_request(
    agent: Any,
    tokens: int,
    *,
    started_at: Optional[float] = None,
) -> int:
    """Publish a pending generation and its JSON projection atomically."""
    with pending_projection_guard(agent):
        return _begin_pending_request_locked(
            agent,
            tokens,
            started_at=started_at,
        )


def _clear_pending_request_locked(agent: Any, generation: int) -> bool:
    """Clear only the matching pending generation.

    An older request's ``finally`` may run after a newer request has published
    its marker.  Both SQLite and the in-memory/JSON projection are therefore
    fenced by generation and owner; the older request cannot erase the newer.
    """
    expected_generation = telemetry_counter(generation)
    owner = getattr(agent, "_session_telemetry_owner", None)
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    clear_errors: list[BaseException] = []
    db_clear_matched: Optional[bool] = None
    if session_db and session_id:
        try:
            db_clear_matched = bool(
                session_db.clear_pending_request(
                    session_id,
                    generation=expected_generation,
                    owner=owner,
                )
            )
        except BaseException as exc:
            clear_errors.append(exc)
            logger.debug(
                "pending telemetry clear failed (session=%s generation=%s)",
                session_id,
                expected_generation,
                exc_info=True,
            )

            # The clear may have committed before its wrapper raised. Re-read
            # canonical state before deciding whether to retry or report the
            # outcome. Only the exact generation/owner can be retried.
            canonical = None
            try:
                canonical = session_db.get_session(session_id)
            except BaseException as recovery_exc:
                clear_errors.append(recovery_exc)
                logger.debug(
                    "pending telemetry clear recovery read failed "
                    "(session=%s generation=%s)",
                    session_id,
                    expected_generation,
                    exc_info=True,
                )

            if isinstance(canonical, dict):
                canonical_generation = telemetry_counter(
                    canonical.get("pending_generation")
                )
                canonical_tokens = telemetry_counter(
                    canonical.get("pending_prompt_tokens")
                )
                canonical_owner = canonical.get("pending_owner")
                if (
                    canonical_generation == expected_generation
                    and canonical_tokens == 0
                    and canonical_owner is None
                ):
                    db_clear_matched = True
                elif (
                    canonical_generation == expected_generation
                    and canonical_tokens > 0
                    and canonical_owner == owner
                ):
                    try:
                        db_clear_matched = bool(
                            session_db.clear_pending_request(
                                session_id,
                                generation=expected_generation,
                                owner=owner,
                            )
                        )
                    except BaseException as retry_exc:
                        clear_errors.append(retry_exc)
                        logger.debug(
                            "pending telemetry clear retry failed "
                            "(session=%s generation=%s)",
                            session_id,
                            expected_generation,
                            exc_info=True,
                        )
                else:
                    db_clear_matched = False

    control_flow_error = next(
        (error for error in clear_errors if not isinstance(error, Exception)),
        None,
    )

    canonical_synced = False
    if session_db and session_id:
        try:
            canonical_synced = sync_pending_from_canonical(agent)
        except BaseException as sync_exc:
            if control_flow_error is None and not isinstance(sync_exc, Exception):
                control_flow_error = sync_exc
            logger.debug(
                "pending telemetry post-clear sync failed "
                "(session=%s generation=%s)",
                session_id,
                expected_generation,
                exc_info=True,
            )

    if db_clear_matched is not True and session_db and session_id:
        # Another process may have published a newer generation after this
        # process began its request, or the canonical clear outcome is still
        # unknown. Never clear the compatibility projections or claim success
        # unless SQLite confirms terminal zero.
        if not canonical_synced:
            logger.error(
                "pending telemetry clear could not confirm canonical state "
                "(session=%s generation=%s)",
                session_id,
                expected_generation,
            )
        try:
            _refresh_json_projection(agent)
        except BaseException as exc:
            if control_flow_error is None:
                control_flow_error = exc
        if control_flow_error is not None:
            raise control_flow_error
        return False

    if not canonical_synced:
        agent.pending_prompt_tokens = 0
        agent._pending_owner = None
        agent._pending_started_at = 0.0
    elif (
        telemetry_counter(getattr(agent, "_pending_generation", 0))
        != expected_generation
        or telemetry_counter(getattr(agent, "pending_prompt_tokens", 0)) != 0
        or getattr(agent, "_pending_owner", None) is not None
    ):
        if control_flow_error is not None:
            raise control_flow_error
        return False
    try:
        _refresh_json_projection(agent)
    except BaseException as exc:
        if control_flow_error is None:
            control_flow_error = exc
    if control_flow_error is not None:
        raise control_flow_error
    return True


def clear_pending_request(agent: Any, generation: int) -> bool:
    """Conditionally clear a pending generation and serialize its projection."""
    with pending_projection_guard(agent):
        return _clear_pending_request_locked(agent, generation)


def _pending_attempt_still_live(agent: Any, generation: int) -> bool:
    return bool(
        telemetry_counter(getattr(agent, "_pending_generation", 0))
        == telemetry_counter(generation)
        and telemetry_counter(getattr(agent, "pending_prompt_tokens", 0)) > 0
        and getattr(agent, "_pending_owner", None)
        == getattr(agent, "_session_telemetry_owner", None)
    )


def clear_pending_request_on_success(agent: Any, generation: int) -> bool:
    """Clear a completed request, failing loud if its own marker survives."""
    cleared = clear_pending_request(agent, generation)
    if not cleared and _pending_attempt_still_live(agent, generation):
        raise RuntimeError(
            "canonical pending telemetry could not be cleared "
            f"(session={getattr(agent, 'session_id', None)} "
            f"generation={telemetry_counter(generation)})"
        )
    return cleared


def clear_pending_request_during_unwind(agent: Any, generation: int) -> bool:
    """Best-effort clear that never replaces an already-active exception."""
    try:
        cleared = clear_pending_request(agent, generation)
    except BaseException:
        logger.error(
            "pending telemetry cleanup failed during exception unwind "
            "(session=%s generation=%s)",
            getattr(agent, "session_id", None),
            telemetry_counter(generation),
            exc_info=True,
        )
        return False
    if not cleared and _pending_attempt_still_live(agent, generation):
        logger.error(
            "canonical pending telemetry survived exception unwind "
            "(session=%s generation=%s)",
            getattr(agent, "session_id", None),
            telemetry_counter(generation),
        )
    return cleared


def record_call_without_usage(agent: Any, *, accepted: bool = True) -> None:
    """Record an accepted provider call that omitted token usage."""
    if not accepted:
        return
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db and session_id:
        try:
            if (
                not getattr(agent, "_session_db_created", False)
                or getattr(agent, "_session_telemetry_hydrated_session_id", None)
                != session_id
            ):
                agent._ensure_db_session()
        except Exception:
            logger.debug(
                "accepted API-call telemetry bind failed (session=%s)",
                session_id,
                exc_info=True,
            )

    agent.session_api_calls = telemetry_counter(
        getattr(agent, "session_api_calls", 0)
    ) + 1
    if not session_db or not session_id:
        return
    try:
        compressor = getattr(agent, "context_compressor", None)
        session_db.update_token_counts(
            session_id,
            model=getattr(agent, "model", None),
            api_call_count=1,
            context_length=telemetry_counter(
                getattr(compressor, "context_length", 0)
            ),
            compression_count=telemetry_counter(
                getattr(compressor, "compression_count", 0)
            ),
        )
    except Exception:
        logger.debug(
            "accepted API-call persistence without usage failed (session=%s)",
            session_id,
            exc_info=True,
        )


def record_canonical_usage(
    agent: Any,
    canonical_usage: Any,
    *,
    accepted: bool = True,
    priced_usage: Any = None,
    pricing_model: Optional[str] = None,
    pricing_provider: Optional[str] = None,
    pricing_base_url: Optional[str] = None,
    extra_cost_usd: Optional[float] = None,
    total_tokens_override: Optional[int] = None,
    messages_len: Optional[int] = None,
    context_length: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply one normalized usage record to memory and canonical SQLite.

    ``accepted`` controls only the successful-call counter. A runtime may
    still report billable token usage for a terminal error/interruption; those
    tokens remain durable without misclassifying the call as accepted.
    """
    from agent.usage_pricing import estimate_usage_cost

    prompt_tokens = telemetry_counter(canonical_usage.prompt_tokens)
    completion_tokens = telemetry_counter(canonical_usage.output_tokens)
    canonical_total = telemetry_counter(canonical_usage.total_tokens)
    total_tokens = (
        telemetry_counter(total_tokens_override)
        if total_tokens_override is not None
        else canonical_total
    )
    usage_dict = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "input_tokens": telemetry_counter(canonical_usage.input_tokens),
        "output_tokens": telemetry_counter(canonical_usage.output_tokens),
        "cache_read_tokens": telemetry_counter(
            canonical_usage.cache_read_tokens
        ),
        "cache_write_tokens": telemetry_counter(
            canonical_usage.cache_write_tokens
        ),
        "reasoning_tokens": telemetry_counter(canonical_usage.reasoning_tokens),
    }

    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None)
    if session_db and session_id:
        try:
            if (
                not getattr(agent, "_session_db_created", False)
                or getattr(agent, "_session_telemetry_hydrated_session_id", None)
                != session_id
            ):
                agent._ensure_db_session()
        except Exception:
            logger.debug(
                "canonical usage telemetry bind failed (session=%s)",
                session_id,
                exc_info=True,
            )

    compressor = getattr(agent, "context_compressor", None)
    if compressor is not None:
        try:
            if messages_len is None:
                compressor.update_from_response(usage_dict)
            else:
                compressor.update_from_response(
                    usage_dict, messages_len=messages_len
                )
            if context_length is not None and telemetry_counter(context_length) > 0:
                compressor.context_length = telemetry_counter(context_length)
        except Exception:
            logger.debug("context usage update failed", exc_info=True)

    if accepted:
        agent.session_api_calls = telemetry_counter(
            getattr(agent, "session_api_calls", 0)
        ) + 1
    agent.session_prompt_tokens = telemetry_counter(
        getattr(agent, "session_prompt_tokens", 0)
    ) + prompt_tokens
    agent.session_completion_tokens = telemetry_counter(
        getattr(agent, "session_completion_tokens", 0)
    ) + completion_tokens
    agent.session_total_tokens = telemetry_counter(
        getattr(agent, "session_total_tokens", 0)
    ) + total_tokens
    agent.session_input_tokens = telemetry_counter(
        getattr(agent, "session_input_tokens", 0)
    ) + usage_dict["input_tokens"]
    agent.session_output_tokens = telemetry_counter(
        getattr(agent, "session_output_tokens", 0)
    ) + usage_dict["output_tokens"]
    agent.session_cache_read_tokens = telemetry_counter(
        getattr(agent, "session_cache_read_tokens", 0)
    ) + usage_dict["cache_read_tokens"]
    agent.session_cache_write_tokens = telemetry_counter(
        getattr(agent, "session_cache_write_tokens", 0)
    ) + usage_dict["cache_write_tokens"]
    agent.session_reasoning_tokens = telemetry_counter(
        getattr(agent, "session_reasoning_tokens", 0)
    ) + usage_dict["reasoning_tokens"]

    priced = priced_usage if priced_usage is not None else canonical_usage
    cost_result = estimate_usage_cost(
        pricing_model or getattr(agent, "model", None),
        priced,
        provider=pricing_provider or getattr(agent, "provider", None),
        base_url=pricing_base_url or getattr(agent, "base_url", None),
        api_key=getattr(agent, "api_key", ""),
    )
    cost_delta: Optional[float] = None
    if cost_result.amount_usd is not None:
        cost_delta = _telemetry_float(cost_result.amount_usd)
    if extra_cost_usd is not None:
        cost_delta = _add_telemetry_cost(cost_delta or 0.0, extra_cost_usd)
    agent.session_estimated_cost_usd = _telemetry_float(
        getattr(agent, "session_estimated_cost_usd", 0.0)
    )
    if cost_delta is not None:
        cost_delta = _telemetry_float(cost_delta)
        agent.session_estimated_cost_usd = _add_telemetry_cost(
            getattr(agent, "session_estimated_cost_usd", 0.0),
            cost_delta,
        )
    agent.session_cost_status = cost_result.status
    agent.session_cost_source = cost_result.source

    if session_db and session_id:
        try:
            session_db.update_token_counts(
                session_id,
                input_tokens=usage_dict["input_tokens"],
                output_tokens=usage_dict["output_tokens"],
                cache_read_tokens=usage_dict["cache_read_tokens"],
                cache_write_tokens=usage_dict["cache_write_tokens"],
                reasoning_tokens=usage_dict["reasoning_tokens"],
                total_tokens=total_tokens,
                estimated_cost_usd=cost_delta,
                cost_status=cost_result.status,
                cost_source=cost_result.source,
                billing_provider=getattr(agent, "provider", None),
                billing_base_url=getattr(agent, "base_url", None),
                billing_mode="subscription_included"
                if cost_result.status == "included" else None,
                model=getattr(agent, "model", None),
                api_call_count=1 if accepted else 0,
                last_prompt_tokens=prompt_tokens,
                last_completion_tokens=completion_tokens,
                context_length=telemetry_counter(
                    getattr(compressor, "context_length", 0)
                ),
                compression_count=telemetry_counter(
                    getattr(compressor, "compression_count", 0)
                ),
            )
        except Exception:
            logger.debug(
                "canonical token persistence failed (session=%s tokens=%d)",
                session_id,
                total_tokens,
                exc_info=True,
            )

    return {
        **usage_dict,
        "last_prompt_tokens": prompt_tokens,
        "last_completion_tokens": completion_tokens,
        "estimated_cost_usd": cost_delta,
        "cost_status": cost_result.status,
        "cost_source": cost_result.source,
    }


def telemetry_result_fields(agent: Any) -> Dict[str, Any]:
    """Return the stable numeric telemetry contract for every turn result."""
    compressor = getattr(agent, "context_compressor", None)
    input_tokens = telemetry_counter(getattr(agent, "session_input_tokens", 0))
    output_tokens = telemetry_counter(getattr(agent, "session_output_tokens", 0))
    estimated_cost_usd = _telemetry_float(
        getattr(agent, "session_estimated_cost_usd", 0.0)
    )
    api_call_count = telemetry_counter(getattr(agent, "session_api_calls", 0))
    agent.session_api_calls = api_call_count
    agent.session_estimated_cost_usd = estimated_cost_usd
    return {
        "api_call_count": api_call_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "session_input_tokens": input_tokens,
        "session_output_tokens": output_tokens,
        "cache_read_tokens": telemetry_counter(
            getattr(agent, "session_cache_read_tokens", 0)
        ),
        "cache_write_tokens": telemetry_counter(
            getattr(agent, "session_cache_write_tokens", 0)
        ),
        "reasoning_tokens": telemetry_counter(
            getattr(agent, "session_reasoning_tokens", 0)
        ),
        "prompt_tokens": telemetry_counter(
            getattr(agent, "session_prompt_tokens", 0)
        ),
        "completion_tokens": telemetry_counter(
            getattr(agent, "session_completion_tokens", 0)
        ),
        "total_tokens": telemetry_counter(
            getattr(agent, "session_total_tokens", 0)
        ),
        "last_prompt_tokens": telemetry_counter(
            getattr(compressor, "last_prompt_tokens", 0)
        ),
        "last_real_prompt_tokens": telemetry_counter(
            getattr(
                compressor,
                "last_real_prompt_tokens",
                getattr(compressor, "last_prompt_tokens", 0),
            )
        ),
        "last_completion_tokens": telemetry_counter(
            getattr(compressor, "last_completion_tokens", 0)
        ),
        "pending_prompt_tokens": telemetry_counter(
            getattr(agent, "pending_prompt_tokens", 0)
        ),
        "pending_generation": telemetry_counter(
            getattr(agent, "_pending_generation", 0)
        ),
        "pending_started_at": _telemetry_float(
            getattr(agent, "_pending_started_at", 0.0)
        ),
        "compression_count": telemetry_counter(
            getattr(compressor, "compression_count", 0)
        ),
        "context_length": telemetry_counter(
            getattr(compressor, "context_length", 0)
        ),
        "estimated_cost_usd": estimated_cost_usd,
    }


def telemetry_json_fields(agent: Any) -> Dict[str, Any]:
    """Return the minimal JSON compatibility projection."""
    fields = telemetry_result_fields(agent)
    return {
        "api_call_count": fields["api_call_count"],
        "session_input_tokens": fields["session_input_tokens"],
        "session_output_tokens": fields["session_output_tokens"],
        "last_prompt_tokens": fields["last_real_prompt_tokens"],
        "last_completion_tokens": fields["last_completion_tokens"],
        "pending_prompt_tokens": fields["pending_prompt_tokens"],
        "pending_generation": fields["pending_generation"],
        "pending_owner": getattr(agent, "_pending_owner", None),
        "pending_started_at": fields["pending_started_at"],
        "compression_count": fields["compression_count"],
        "compression_epoch": fields["compression_count"],
        "context_length": fields["context_length"],
        "estimated_cost_usd": fields["estimated_cost_usd"],
    }


def apply_telemetry_result_fields(
    result: MutableMapping[str, Any], agent: Any
) -> MutableMapping[str, Any]:
    """Attach stable numeric telemetry to any terminal result shape."""
    result["api_calls"] = telemetry_counter(result.get("api_calls"))
    result.update(telemetry_result_fields(agent))
    return result


__all__ = [
    "apply_telemetry_result_fields",
    "begin_pending_request",
    "clear_pending_request",
    "clear_pending_request_during_unwind",
    "clear_pending_request_on_success",
    "hydrate_agent_session_telemetry",
    "new_pending_owner",
    "pending_projection_guard",
    "persist_current_telemetry_snapshot",
    "record_call_without_usage",
    "record_canonical_usage",
    "sync_pending_from_canonical",
    "sync_telemetry_from_canonical",
    "telemetry_counter",
    "telemetry_json_fields",
    "telemetry_result_fields",
]
