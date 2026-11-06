"""A managed worker's ``(pid, birth)`` identity survives wall-clock steps.

The owner re-proves a worker's identity at reservation, on every worker RPC, inside delegation
dispatch and before it signals the worker at shutdown. psutil's Linux ``create_time()`` adds the
CURRENT ``/proc/stat`` btime to the process's boot-relative start ticks, so a host whose clock is
stepped (WSL2 timesync moves btime about every 33 s) shifts that value for a process that never
changed. Compared exactly against a wall-clock birth, the same live worker was refused
(``permission_denied``), its store latched the refusal and the session ended ``unknown``. The
identity is the kernel start since boot (``hermes_cli.process_identity.worker_birth``) at the
producer and at every check; the delegation ledger keeps ``gateway.status.get_process_start_time``.

The owner here is this process: a step moves psutil's btime forward one second in it only. The
child reports its identity through the product's own ``agent.managed_worker.hello``. Real
processes, real SQLite.
"""
from contextlib import closing, suppress
import json
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from hermes_state import SessionDB
from hermes_state_runtime import RuntimeStoreError, admit_session_input, begin_runtime_epoch, claim_session_input

ROOT = Path(__file__).resolve().parents[2]
# Introduces itself exactly as a managed worker does, then echoes each stdin line. Default SIGTERM
# disposition: once the owner has signalled it, it can never echo again.
_WORKER = ('import json, sys\n'
           'from agent.managed_worker import hello\n'
           'print(json.dumps(hello()), flush=True)\n'
           'for _ in sys.stdin:\n'
           '    print("ECHO", flush=True)\n')


def _spawn():
    child = subprocess.Popen([sys.executable, '-c', _WORKER], cwd=ROOT, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    ident = json.loads(child.stdout.readline())
    assert ident['pid'] == child.pid
    return child, ident


def _end(child):
    with suppress(BrokenPipeError):
        child.stdin.close()
    if child.poll() is None:
        child.kill()
    child.wait(timeout=10)


def _answers(child):
    try:
        child.stdin.write('ping\n')
        child.stdin.flush()
    except BrokenPipeError:
        return False
    return child.stdout.readline() == 'ECHO\n'


def _step_wall_clock(monkeypatch, pid):
    """One WSL2 timesync step: btime one second later, every live process unchanged."""
    import psutil._pslinux as linux
    before = psutil.Process(pid).create_time()
    current = linux.boot_time
    monkeypatch.setattr(linux, 'boot_time', lambda: current() + 1.0)
    assert psutil.Process(pid).create_time() != before  # the owner's wall-clock view really moved


def _owner(tmp_path, monkeypatch):
    from gateway.config import Platform
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor, SessionRef
    from tools.async_delegation import _initialize_schema

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    db = SessionDB(tmp_path / 'state.db')
    db.create_session('owned', 'cli')
    with closing(sqlite3.connect(tmp_path / 'state.db')) as conn:
        _initialize_schema(conn)
    runner = SimpleNamespace(_draining=False, config=SimpleNamespace(multiplex_profiles=False),
                             _adapter_for_source=lambda source: None)
    authority = SessionAuthority(runner, profile_id='p', instance_id='owner', db=db,
                                 epoch=begin_runtime_epoch(db, instance_id='owner'))
    authority.sessions['owned'] = LiveSession(SimpleNamespace(platform=Platform.TELEGRAM, user_id='human'), 'route')
    worker = SimpleNamespace(authority=authority, actor=SessionActor('human', 'p', frozenset({'worker:adopt'}), 't'))
    return db, authority, worker, SessionRef('p', 'owned')


def _scope(ident, **extra):
    return {'profile_id': 'p', 'session_id': 'owned', 'execution_id': f"compute-{ident['pid']}",
            'generation': 0, 'pid': ident['pid'], 'birth': ident['birth'], 'secret': 'w' * 32, **extra}


def _persist(authority, ident, sequence, operation='session.tail_role', payload=None):
    return _scope(ident, epoch=authority.epoch, sequence=sequence, operation=operation,
                  payload={} if payload is None else payload)


def _delegation(delegation_id):
    return dict(delegation_id=delegation_id, task={'goal': 'inert'}, dispatched_at=time.time(),
                parent_session_id='owned', session_key='', origin_ui_session_id='', origin_session_id='')


def _writes(db):
    return (db._read_one('SELECT last_sequence FROM worker_executions')[0],
            db._read_one('SELECT COUNT(*) FROM worker_receipts')[0],
            db._read_one('SELECT COUNT(*) FROM async_delegations')[0])


@pytest.mark.platforms("linux")
@pytest.mark.asyncio
async def test_live_worker_registers_persists_and_dispatches_across_wall_clock_steps(tmp_path, monkeypatch):
    """The live session from the incident: the same worker keeps proving itself after btime moves,
    while a forged birth or a dead worker is refused before the ledger changes."""
    from gateway.session_worker import worker_request
    from gateway.session_worker_delegation import persist_worker_delegation
    db, authority, worker, ref = _owner(tmp_path, monkeypatch)
    child, ident = _spawn()
    forged = dict(ident, birth=ident['birth'] + 7)  # another incarnation of this pid
    try:
        _step_wall_clock(monkeypatch, child.pid)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await worker_request(worker, ref, _scope(forged, kind='compute'), operation='register')
        assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 0
        registered = await worker_request(worker, ref, _scope(ident, kind='compute'), operation='register')
        assert registered['status'] == 'registered'
        assert 'value' in await worker_request(worker, ref, _persist(authority, ident, 1), operation='persist')
        _step_wall_clock(monkeypatch, child.pid)
        assert 'value' in await worker_request(worker, ref, _persist(authority, ident, 2), operation='persist')
        dispatch = _persist(authority, ident, 3, 'delegation.dispatch', _delegation('deleg_live'))
        assert await worker_request(worker, ref, dispatch, operation='persist') == {'value': None}
        from gateway.status import get_process_start_time
        assert db._read_one('SELECT owner_pid, owner_started_at FROM async_delegations')[:] == (
            ident['pid'], get_process_start_time(ident['pid']))
        written = _writes(db)
        assert written == (3, 3, 1)
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            await worker_request(worker, ref, _persist(authority, forged, 4), operation='persist')
        # The dispatch handler proves the identity again inside the receipt transaction.
        with pytest.raises(RuntimeStoreError, match='permission_denied'):
            persist_worker_delegation(db, epoch=authority.epoch, execution_id=f"compute-{ident['pid']}",
                session_id='owned', generation=0, sequence=4, operation='delegation.dispatch',
                payload=_delegation('deleg_forged'), worker_pid=ident['pid'], worker_birth=forged['birth'])
        assert _writes(db) == written
    finally:
        _end(child)
    try:
        with pytest.raises(RuntimeStoreError, match='worker_not_live|permission_denied'):
            await worker_request(worker, ref, _persist(authority, ident, 4), operation='persist')
        with pytest.raises(RuntimeStoreError, match='worker_not_live|permission_denied'):
            persist_worker_delegation(db, epoch=authority.epoch, execution_id=f"compute-{ident['pid']}",
                session_id='owned', generation=0, sequence=4, operation='delegation.dispatch',
                payload=_delegation('deleg_dead'), worker_pid=ident['pid'], worker_birth=ident['birth'])
        assert _writes(db) == written
    finally:
        db.close()


@pytest.mark.platforms("linux")
def test_reservation_accepts_the_workers_hello_across_a_wall_clock_step(tmp_path, monkeypatch):
    from gateway.session_worker_reservation import reserve_admission_worker
    child, ident = _spawn()
    try:
        with closing(SessionDB(tmp_path / 'state.db')) as db:
            db.create_session('owned', 'cli')
            epoch = begin_runtime_epoch(db, instance_id='owner')
            authority = SimpleNamespace(db=db, epoch=epoch, profile_id=str(tmp_path), _require_admission_open=lambda: None)
            admitted = admit_session_input(db, epoch=epoch, principal_id='human', session_id='owned',
                                           request_id='input', payload={})
            claim_session_input(db, epoch=epoch, session_id='owned')
            kwargs = dict(admission_id=admitted['admission_id'], process=child, principal_id='human')
            hello = {'type': 'hello', **ident}
            _step_wall_clock(monkeypatch, child.pid)
            with pytest.raises(RuntimeStoreError, match='permission_denied'):
                reserve_admission_worker(authority, hello=dict(hello, birth=ident['birth'] + 7), **kwargs)
            assert db._read_one('SELECT COUNT(*) FROM worker_executions')[0] == 0
            scope = reserve_admission_worker(authority, hello=hello, **kwargs)
            assert (scope['pid'], scope['birth']) == (ident['pid'], ident['birth'])
    finally:
        _end(child)


def _close_naming(worker_pid, birth):
    """The owner's real shutdown of a handle whose verified interpreter is a different process."""
    from gateway.session_managed_worker import ManagedWorker
    handle = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    managed = ManagedWorker(handle)
    managed.worker = (worker_pid, birth)
    managed.close()
    assert handle.poll() is not None


@pytest.mark.platforms("linux")
def test_shutdown_signals_the_verified_worker_across_a_wall_clock_step(monkeypatch):
    child, ident = _spawn()
    try:
        _step_wall_clock(monkeypatch, child.pid)
        _close_naming(ident['pid'], ident['birth'])
        assert child.wait(timeout=10) in (-signal.SIGTERM, -signal.SIGKILL)
    finally:
        _end(child)


def test_shutdown_never_signals_another_incarnation_of_the_pid():
    child, ident = _spawn()
    try:
        _close_naming(ident['pid'], ident['birth'] + 1)
        assert _answers(child)
    finally:
        _end(child)


def test_shutdown_never_signals_a_worker_whose_start_cannot_be_read(monkeypatch):
    """A start fingerprint that cannot be read (the pid exited or became unreadable) proves nothing."""
    from hermes_cli import process_identity
    child, ident = _spawn()

    def unreadable(proc):
        raise psutil.AccessDenied(proc.pid)
    try:
        monkeypatch.setattr(process_identity, 'worker_birth', unreadable)
        _close_naming(ident['pid'], ident['birth'])
        assert _answers(child)
    finally:
        _end(child)
