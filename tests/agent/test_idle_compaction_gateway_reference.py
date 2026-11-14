"""Idle compaction on the messaging gateway route measures the pause from the newest prior row.

The gateway loads the transcript from state.db and converts it with
``_build_gateway_agent_history`` before the agent sees it. When that conversion kept
``timestamp`` on user rows only, the idle reference was the start of the previous turn, so a
reply sent right after a turn that began more than ``idle_compact_after_seconds`` ago
compacted a conversation whose prompt cache was still warm.

Each test stores rows in a real ``state.db`` under ``tmp_path``, loads them the way
``SessionStore.load_transcript`` does, converts them with the real gateway builder, applies the
real cached-agent reset and runs the real turn prologue.
"""

from __future__ import annotations

import time
from pathlib import Path

from hermes_state import SessionDB

from tests.agent.test_idle_compaction_lock_and_guards import _prep_idle_agent, _run_prologue

IDLE_AFTER = 3600


def _agent(tmp_path: Path, sid: str):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session(sid, source="cli")
    agent = _prep_idle_agent(db, sid, idle_after=IDLE_AFTER)
    agent.context_compressor.emit_automatic_compaction_status = False
    return db, agent


def _store_long_turn(db: SessionDB, sid: str, *, turn_started_ago: float, turn_ended_ago: float) -> None:
    """Older small talk, then one long turn: the user asked ``turn_started_ago`` seconds ago and
    the assistant's last row landed ``turn_ended_ago`` seconds ago."""
    now = time.time()
    for i in range(8):
        db.append_message(sid, "user" if i % 2 == 0 else "assistant", f"old{i}",
                          timestamp=now - turn_started_ago - 600 + i)
    db.append_message(sid, "user", "please do the long job", timestamp=now - turn_started_ago)
    db.append_message(
        sid, "assistant", "",
        tool_calls=[{"id": "call_1", "type": "function",
                     "function": {"name": "terminal", "arguments": "{}"}}],
        timestamp=now - turn_started_ago + 5)
    db.append_message(sid, "tool", "job output", tool_call_id="call_1", tool_name="terminal",
                      timestamp=now - turn_ended_ago - 5)
    db.append_message(sid, "assistant", "the long job is done", timestamp=now - turn_ended_ago)


def _gateway_history(db: SessionDB, sid: str) -> list:
    from gateway.run import _build_gateway_agent_history

    stored = db.get_messages_as_conversation(sid, repair_alternation=True)
    history, _observed = _build_gateway_agent_history(stored)
    return history


def test_gateway_history_keeps_the_stored_timestamp_on_every_row(tmp_path: Path) -> None:
    db, _unused = _agent(tmp_path, "GW_SHAPE")
    _store_long_turn(db, "GW_SHAPE", turn_started_ago=4000, turn_ended_ago=60)
    stored = db.get_messages_as_conversation("GW_SHAPE", repair_alternation=True)

    history = _gateway_history(db, "GW_SHAPE")

    assert [m["role"] for m in history] == [m["role"] for m in stored]
    assert [m.get("timestamp") for m in history] == [m["timestamp"] for m in stored]
    assert {m["role"] for m in history} == {"user", "assistant", "tool"}


def test_gateway_reply_one_minute_after_a_long_turn_does_not_compact(tmp_path: Path) -> None:
    from gateway.run import GatewayRunner

    db, agent = _agent(tmp_path, "GW_WARM")
    _store_long_turn(db, "GW_WARM", turn_started_ago=4000, turn_ended_ago=60)
    history = _gateway_history(db, "GW_WARM")
    GatewayRunner._init_cached_agent_for_turn(agent, 0)

    _run_prologue(agent, history)

    assert agent.context_compressor.compress.call_count == 0, (
        "idle compaction fired although the previous turn's last row is 60 s old"
    )


def test_gateway_turn_after_a_real_pause_compacts(tmp_path: Path) -> None:
    from gateway.run import GatewayRunner

    db, agent = _agent(tmp_path, "GW_IDLE")
    _store_long_turn(db, "GW_IDLE", turn_started_ago=7300, turn_ended_ago=7200)
    history = _gateway_history(db, "GW_IDLE")
    GatewayRunner._init_cached_agent_for_turn(agent, 0)

    _run_prologue(agent, history)

    assert agent.context_compressor.compress.call_count == 1
