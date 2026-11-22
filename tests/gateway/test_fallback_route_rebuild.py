"""A cold local gateway turn must retain its primary after accounting and route sync."""
import json
from types import SimpleNamespace

import pytest

from persistence.conversation.usage import BillingRoute, TokenCounts, UsageDelta
from tests.fixtures.storage_read import session_row

from gateway.session_execution import GatewaySessionAgentMixin
from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from tests.gateway.test_session_follow_profile_config import _owner

pytestmark = pytest.mark.usefixtures('authority_cwd_in_tmp')

PRIMARY = 'claude-fable-5-2'
BEDROCK = 'global.anthropic.claude-fable-5-2'
CHAIN = [{'provider': 'bedrock', 'model': BEDROCK}, {'provider': 'openai-codex', 'model': 'gpt-6.2-astra'}]


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['nested', 'provider-only', 'billing-only', 'pick', 'switch'])
async def test_cold_local_constructor_after_fallback_sync(monkeypatch, shape):
    await cold_turn(monkeypatch, shape)


async def cold_turn(monkeypatch, shape, resume=None):
    from gateway import run
    from gateway.session_local_recovery import restore_local_session
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    config = {'model': {'default': PRIMARY, 'provider': 'anthropic'}, 'fallback_providers': CHAIN}
    (home / 'config.yaml').write_text(json.dumps(config))
    monkeypatch.setattr(run, '_hermes_home', home)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'offline-test')
    store, runner, authority, owner = _owner(home)
    try:
        response = await owner.dispatch({'id': '1', 'method': 'session.create', 'params': {
            'request_id': '1', 'source': 'tui', 'cwd': str(home), 'toolsets': []}})
        assert 'result' in response, response
        sid = response['result']['session_id']
        db = authority.persistence
        expected = PRIMARY
        if shape in ('pick', 'switch'):
            expected = 'claude-sonnet-5'
            runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
            snap = await owner.dispatch(dict(id='attach', method='session.resume', params={'session_id': sid}))
            result = await owner.dispatch(dict(id='pick', method='session.mutate', params=dict(
                session_id=sid, operation='model', request_id='pick', payload={'model': expected, 'provider': 'anthropic'},
                expected_revision=snap['result']['revision'], expected_generation=snap['result']['execution_generation'])))
            assert 'result' in result, result
        original = db.bindings.receipt(sid).policy
        assert original['model'] == json.loads(original['config_json'])['model']['default'] == expected
        db.usage.queue_usage(sid, UsageDelta(tokens=TokenCounts(input_tokens=5), api_call_count=1,
                                            route=BillingRoute(model=BEDROCK, billing_provider='bedrock')))
        assert db.usage.flush_usage()
        serving = SimpleNamespace(model=BEDROCK, provider='bedrock', _fallback_activated=True,
                                  api_mode='anthropic_messages', base_url='https://bedrock-runtime.us-east-1.amazonaws.com')
        runner.persistence = db
        if shape != 'billing-only':
            db.sessions.patch_model_config(sid, {'provider': 'anthropic'})
        if shape == 'switch':
            serving = SimpleNamespace(model=expected, provider='anthropic', _fallback_activated=False)
        if shape in ('nested', 'pick', 'switch'):
            GatewayTurnPrepareMixin._sync_session_model_from_agent(runner, sid, serving)
        row = session_row(db, sid)
        assert row['model'] == (expected if shape == 'switch' else BEDROCK)
        if shape == 'nested':
            assert json.loads(row['model_config'])['gateway_runtime']['fallback_active'] is True
        source = authority.sessions[sid].source
        key = store._generate_session_key(source)
        runner._agent_cache = {key: (serving,)}
        ctx = SimpleNamespace(session_key=key, session_id=sid, source=source, AIAgent=SimpleNamespace,
                              enabled_toolsets=[], disabled_toolsets=[], user_config=config)
        turn = GatewaySessionAgentMixin()
        turn._ctx, turn._runner = ctx, runner
        assert turn._pop_cached_agent_for_eviction() is serving
        adapter = runner._adapter_for_source(source)
        adapter.forget(source, key)
        authority.sessions.pop(sid)
        if resume:
            runner._resolve_session_agent_runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime.__get__(runner)
            await resume(owner, sid, row)
        restore_local_session(authority, sid)
        restored = db.bindings.receipt(sid).policy
        if not resume:
            assert restored == original
        ctx.source = source = authority.sessions[sid].source
        model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=source)
        runner._service_tier = None
        runner._refresh_fallback_model = run.GatewayRunner._refresh_fallback_model.__get__(runner)
        route = GatewayTurnPrepareMixin._resolve_turn_agent_config(runner, 'notice', model, runtime)
        built = turn._build_fresh_agent(route, 'tui', '', 10, None, {}, True)
        assert (built.model, built.provider) == (expected, 'anthropic')
        assert built.fallback_model == CHAIN
        assert restored['model'] == json.loads(restored['config_json'])['model']['default'] == expected
        from agent.anthropic_adapter import build_anthropic_kwargs, create_anthropic_message
        kwargs = build_anthropic_kwargs(built.model, [{'role': 'user', 'content': 'test'}], None, 32, None)
        client = SimpleNamespace(messages=SimpleNamespace(create=lambda **request: request['model']))
        assert create_anthropic_message(client, kwargs) == expected
    finally:
        await owner.close()
        store.close_all_db_handles()
