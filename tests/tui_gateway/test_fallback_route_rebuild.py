"""The billing record must not override a TUI session's primary route."""
import json
from types import SimpleNamespace

import pytest

from gateway.run_turn_prepare import GatewayTurnPrepareMixin
from hermes_constants import get_hermes_home
from hermes_state import SessionDB

PRIMARY = ('claude-fable-5-2', 'anthropic')
FALLBACK = ('global.anthropic.claude-fable-5-2', 'bedrock')
PICK = ('claude-sonnet-5', 'anthropic')


@pytest.fixture(params=['nested', 'provider-only', 'billing-only', 'pick', 'switch'])
def saved_route(request):
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'model': {'default': PRIMARY[0], 'provider': PRIMARY[1]},
        'fallback_providers': [{'model': FALLBACK[0], 'provider': FALLBACK[1]}]}))
    db = SessionDB(db_path=home / 'state.db')
    shape = request.param
    config = {} if shape == 'billing-only' else {'provider': PRIMARY[1]}
    if shape == 'pick':
        config['model'] = PICK[0]
    db.create_session('chat', source='tui', model=PRIMARY[0], model_config=config)
    db.update_token_counts('chat', input_tokens=4, model=FALLBACK[0], billing_provider=FALLBACK[1], api_call_count=1)
    if shape in ('nested', 'pick', 'switch'):
        serving = SimpleNamespace(model=FALLBACK[0], provider=FALLBACK[1], _fallback_activated=shape != 'switch')
        GatewayTurnPrepareMixin._sync_session_model_from_agent(SimpleNamespace(_session_db=SimpleNamespace(_db=db)),
                                                              'chat', serving)
    try:
        yield db.get_session('chat'), {'pick': PICK, 'switch': FALLBACK}.get(shape, PRIMARY)
    finally:
        db.close()


def test_tui_resume_primary(saved_route):
    from tui_gateway.server import _stored_session_runtime_overrides
    row, expected = saved_route
    override = _stored_session_runtime_overrides(row).get('model_override') or {}
    assert (override.get('model', PRIMARY[0]), override.get('provider', PRIMARY[1])) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('shape', ['provider-only', 'billing-only', 'nested'])
async def test_tui_resume_preserves_receipt(monkeypatch, shape):
    from tests.gateway.test_fallback_route_rebuild import cold_turn

    async def resume(owner, sid, row):
        response = await owner.dispatch({'id': 'resume', 'method': 'session.resume', 'params': {'session_id': sid}})
        assert 'result' in response, response

    await cold_turn(monkeypatch, shape, resume)


def test_tui_resume_primary_when_chain_no_longer_lists_the_fallback():
    # A top-level provider that differs from the billing provider marks a fallback without the chain.
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    (home / 'config.yaml').write_text(json.dumps({'model': {'default': PRIMARY[0], 'provider': PRIMARY[1]}}))
    db = SessionDB(db_path=home / 'state.db')
    try:
        db.create_session('chat', source='tui', model=PRIMARY[0], model_config={'provider': PRIMARY[1]})
        db.update_token_counts('chat', input_tokens=4, model=FALLBACK[0], billing_provider=FALLBACK[1], api_call_count=1)
        from tui_gateway.server import _stored_session_runtime_overrides
        override = _stored_session_runtime_overrides(db.get_session('chat')).get('model_override') or {}
        assert (override.get('model', PRIMARY[0]), override.get('provider', PRIMARY[1])) == PRIMARY
    finally:
        db.close()
