"""``/compress`` through ``session.mutate`` honours the configured summary length.

This path builds its own ``ContextCompressor`` instead of using the session agent's, so
``compression.summary_tokens_ceiling`` and ``compression.summary_session_log`` have to reach it
here or a manual ``/compress`` from a gateway surface still waits on the long default summary.
"""
import re
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agent.context_compressor import _LEAN_SESSION_LOG_BUDGET_TOKENS, _LEAN_SESSION_LOG_HEADING
from gateway.session_authority import SessionAuthority
from gateway.session_controls import AuthorityConnection
from gateway.session_local import create_local_session
import hermes_state_runtime as rt

SUMMARY_BODY = (
    "## Historical Task Snapshot\nUser asked: 'question 0'\n\n"
    "## Goal\nAnswer the questions.\n\n## Completed Actions\n1. ANSWER question 0 - done\n"
)


async def _compress_prompt(tmp_path, monkeypatch, compression):
    """The prompt one committed ``session.mutate(compress)`` sends to the summary model."""
    from agent import context_compressor
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore

    monkeypatch.setattr(run, '_load_gateway_config', lambda: {'compression': dict(compression)})
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = SUMMARY_BODY
    call_llm = MagicMock(return_value=response)
    monkeypatch.setattr(context_compressor, 'call_llm', call_llm)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    db = store._db
    epoch = rt.begin_runtime_epoch(db, instance_id='owner')
    runner = SimpleNamespace(session_store=store, _session_db=db, adapters={}, _draining=False,
                             _evict_cached_agent=lambda route: None, _cached_agent_for=lambda route: None,
                             _resolve_session_agent_runtime=lambda **k: ('frozen', {}))
    authority = SessionAuthority(runner, profile_id=str(tmp_path), instance_id='owner', db=db, epoch=epoch)
    owner = AuthorityConnection(authority, object(), {'user_id': 'human'})
    ref = create_local_session(authority, owner.actor, dict(request_id='budget', source='cli', cwd=str(tmp_path),
                                                              model='frozen', toolsets=[]))
    # About 100K tokens of history: 20% of it is above the default 10,000 ceiling.
    for i in range(50):
        db.append_message(ref.session_id, 'user', f'question {i}')
        db.append_message(ref.session_id, 'assistant', f'answer {i}: ' + ('detail ' * 1200))
    before = db.get_session(ref.session_id)
    try:
        reply = await owner.dispatch({'id': 1, 'method': 'session.mutate', 'params': {
            'session_id': ref.session_id, 'request_id': 'compress', 'expected_revision': before['runtime_revision'],
            'expected_generation': before['runtime_generation'], 'operation': 'compress', 'payload': {}}})
    finally:
        await owner.close()
    assert reply.get('result', {}).get('operation') == 'compress', reply
    assert call_llm.call_count == 1
    return call_llm.call_args.kwargs['messages'][0]['content']


def _target(prompt):
    targets = re.findall(r'Target ~(\d+) tokens', prompt)
    assert len(targets) == 1, targets
    return int(targets[0])


@pytest.mark.asyncio
@pytest.mark.parametrize(('ceiling', 'session_log'), [(3000, False), ('3000', 'false')], ids=['typed', 'yaml-strings'])
async def test_compress_mutation_sends_the_configured_summary_target(tmp_path, monkeypatch, ceiling, session_log):
    prompt = await _compress_prompt(tmp_path, monkeypatch, {
        'summary_tokens_ceiling': ceiling, 'summary_session_log': session_log})
    assert _target(prompt) == 3000
    assert _LEAN_SESSION_LOG_HEADING not in prompt


@pytest.mark.asyncio
async def test_compress_mutation_keeps_the_session_log_when_only_the_ceiling_is_set(tmp_path, monkeypatch):
    prompt = await _compress_prompt(tmp_path, monkeypatch, {'summary_tokens_ceiling': 3000})
    assert _LEAN_SESSION_LOG_HEADING in prompt
    assert _target(prompt) == 3000 + _LEAN_SESSION_LOG_BUDGET_TOKENS
