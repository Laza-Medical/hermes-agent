"""A Kanban owner's ``owner_admitted`` binding survives a wall-clock step on Linux.

The gateway that admits a card writes its own ``(pid, birth)`` into the ``owner_admitted`` event,
and the dispatcher's reclaim fences re-prove it: ``owner_reclaim_paused`` (a running admission
whose owner is still that process does not pause max-runtime and TTL reclaim) and
``_owner_process_gone`` (part of ``owner_attempt_provably_over``). psutil's Linux
``create_time()`` adds the CURRENT ``/proc/stat`` btime to the process's boot-relative start
ticks, so a stepped host clock (WSL2 resume, NTP) shifts that value for a process that never
changed. Compared exactly against a wall-clock birth, the live owner read as a recycled pid: the
pause held for a running admission and the owner looked gone. The binding is the boot-relative
start fingerprint at the writer and at both readers.

The owner here is this process, admitted through the real ``run_task``; a step moves psutil's
btime five seconds later in this process only, past the 2 s tolerance of
``process_identity._same_incarnation`` (a measured WSL2 step). Real Kanban SQLite, real owner ledger.
"""
import asyncio
from contextlib import closing
import json
import sqlite3
import subprocess
import sys
from types import SimpleNamespace

import psutil
import pytest


def _admit_this_process(tmp_path, monkeypatch):
    """Run the real owner admission for one card; return (board conn factory, task id, receipt)."""
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import Principal
    from gateway.session_kanban import run_task
    monkeypatch.setenv('HERMES_KANBAN_HOME', str(tmp_path))
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env',
                        lambda name: str(tmp_path) if name == 'default' else str(tmp_path / name))
    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        task_id = kb.create_task(conn, title='Owned', assignee='default', workspace_kind='dir',
                                 workspace_path=str(workspace))
        kb.recompute_ready(conn)
        task = kb.claim_task(conn, task_id)
    config = {'model': {'default': 'loop-model', 'provider': 'custom'}, 'platform_toolsets': {'cli': ['terminal']}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda: config)
    monkeypatch.setattr('gateway.session_local.create_local_session', lambda *a, **k: SimpleNamespace(
        session_id='owner-session', profile_id=str(tmp_path)))
    monkeypatch.setattr('gateway.session_kanban.asdict', lambda receipt: {})

    async def submit(actor, submission, **kwargs):
        return SimpleNamespace(admission_id='a')

    async def resume(ref, params):
        pass
    db = SimpleNamespace(get_session=lambda sid: None, db_path=str(tmp_path / 'state.db'))
    authority = SimpleNamespace(profile_id=str(tmp_path), db=db, submit=submit)
    actor = Principal('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=authority, actor=actor, native_owner=True, resume=resume)
    asyncio.run(run_task(connection, dict(board='owned', task_id=task_id, run_id=task.current_run_id,
                                          claim_lock=task.claim_lock)))
    with closing(connect(board='owned')) as conn:
        receipt = json.loads(conn.execute(
            "SELECT payload FROM task_events WHERE task_id=? AND kind='owner_admitted'", (task_id,)).fetchone()[0])
    # The owner's turn is in flight: its admission is started, neither queued nor terminal.
    with closing(sqlite3.connect(tmp_path / 'state.db')) as store, store:
        store.execute('CREATE TABLE session_admissions (target_session_id TEXT, request_id TEXT, status TEXT, outcome TEXT)')
        store.execute("INSERT INTO session_admissions VALUES (?, ?, 'started', NULL)",
                      (receipt['session_id'], receipt['request_id']))
    return (lambda: connect(board='owned')), task_id, receipt


def _rebind(connect, task_id, receipt):
    with closing(connect()) as conn:
        conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='owner_admitted'",
                     (json.dumps(receipt), task_id))
        conn.commit()


def _paused(connect, task_id):
    from hermes_cli.kanban_owner_recovery import owner_reclaim_paused
    with closing(connect()) as conn:
        return owner_reclaim_paused(conn, task_id)


def _step_wall_clock(monkeypatch):
    """One WSL2 timesync step: btime five seconds later, every live process unchanged."""
    import psutil._pslinux as linux
    before = psutil.Process().create_time()
    current = linux.boot_time
    monkeypatch.setattr(linux, 'boot_time', lambda: current() + 5.0)
    assert psutil.Process().create_time() != before  # this process's wall-clock view really moved


@pytest.mark.platforms("linux")
def test_live_owner_binding_survives_a_btime_step_and_a_recycled_pid_still_fails(tmp_path, monkeypatch):
    from hermes_cli.kanban_owner_recovery import _owner_process_gone
    connect, task_id, receipt = _admit_this_process(tmp_path, monkeypatch)
    assert (_paused(connect, task_id), _owner_process_gone(receipt)) == (False, False)

    _step_wall_clock(monkeypatch)
    # The same live owner, the same running admission: reclaim stays unpaused, the owner alive.
    assert (_paused(connect, task_id), _owner_process_gone(receipt)) == (False, False)

    # Identity still refuses another process holding the number, and a pid that is gone.
    child = subprocess.Popen([sys.executable, '-c', 'input()'], stdin=subprocess.PIPE)
    try:
        for pid in (child.pid, 2 ** 22 + 1):
            other = dict(receipt, pid=pid)
            _rebind(connect, task_id, other)
            assert (_paused(connect, task_id), _owner_process_gone(other)) == (True, True), pid
    finally:
        child.kill()
        child.wait()
