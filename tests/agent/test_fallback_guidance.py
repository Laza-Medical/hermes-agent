"""Model guidance follows a fallback request without changing the session's stored prefix."""
from unittest.mock import MagicMock, patch

import pytest

from agent.conversation_loop import _sync_failover_system_message
from agent.prompt_builder import (
    GOOGLE_MODEL_OPERATIONAL_GUIDANCE as GOOGLE,
    TOOL_USE_ENFORCEMENT_GUIDANCE as ENFORCEMENT,
    execution_guidance_text,
)
from hermes_state import SessionDB
from tests.agent.test_fallback_429_after_timeout import RateLimitError, _mock_response
from tests.agent.test_primary_runtime_restore import _make_agent, _mock_resolve


ASTRA = 'gpt-6.2-astra'
FABLE = 'claude-fable-5-2'
BEDROCK = 'global.anthropic.claude-fable-5-2'


@pytest.fixture
def agent():
    a = _make_agent()
    a.model = a._primary_runtime['model'] = FABLE
    a._tool_use_enforcement = a._execution_guidance = 'auto'
    a.valid_tool_names = {'read_file'}
    a._cached_system_prompt = f'Model: decoy\nProvider: decoy\n\nModel: {FABLE}\nProvider: {a.provider}\n'
    with patch('agent.process_bootstrap.OpenAI', return_value=MagicMock()), patch(
        'agent.model_metadata.get_model_context_length', return_value=200000
    ):
        yield a


def chain(agent, *models):
    agent._fallback_chain = [{'provider': 'openrouter', 'model': m} for m in models]
    return patch('agent.auxiliary_client.resolve_provider_client', return_value=(_mock_resolve(), None))


@pytest.mark.parametrize('model,enforce,execute', [(ASTRA, 'auto', 'auto'), ('gemma-test', True, True),
                                                   (ASTRA, False, ['astra']), (ASTRA, ['astra'], False)])
def test_p1_retry_gets_gated_sections(agent, model, enforce, execute, caplog):
    agent._tool_use_enforcement, agent._execution_guidance = enforce, execute
    before = agent._cached_system_prompt
    messages = [{'role': 'system', 'content': before}]
    with chain(agent, model), caplog.at_level('INFO'):
        assert agent._try_activate_fallback()
    _sync_failover_system_message(agent, messages, before)
    sections = {ENFORCEMENT: bool(enforce), GOOGLE: model == 'gemma-test',
                execution_guidance_text(agent.valid_tool_names): bool(execute)}
    for prompt in [agent._cached_system_prompt, messages[0]['content']]:
        for text, expected in sections.items():
            assert prompt.count(text) == int(expected)
    headings = [text.splitlines()[0].lstrip('# ') for text, wanted in sections.items() if wanted]
    assert len([r for r in caplog.records if r.levelname == 'INFO' and model in r.message
                and all(h in r.message for h in headings)]) == 1


def test_p2_restore_replays_original_bytes(agent):
    agent._cached_system_prompt = agent._cached_system_prompt.replace(FABLE, FABLE + '  ')
    before = agent._cached_system_prompt
    with chain(agent, ASTRA):
        assert agent._try_activate_fallback()
    assert agent._restore_primary_runtime()
    assert agent._cached_system_prompt == before


@pytest.mark.parametrize('missing', [None, 'valid_tool_names', '_tool_use_enforcement', '_execution_guidance'])
def test_p3_ungated_matches_base_identity_rewrite(agent, missing):
    # Base changes the last matching line of each kind, preserving even the final newline.
    model = ASTRA if missing else BEDROCK
    if missing:
        delattr(agent, missing)
    with chain(agent, model):
        assert agent._try_activate_fallback()
    expected = f'Model: decoy\nProvider: decoy\n\nModel: {agent.model}\nProvider: openrouter\n'
    assert agent._cached_system_prompt == expected


def test_p4_chain_and_repeat_are_idempotent(agent):
    with chain(agent, BEDROCK, ASTRA), patch('agent.bedrock_adapter.bind_bedrock_runtime'):
        agent._fallback_chain[0]['provider'] = 'bedrock'
        agent._fallback_chain[1]['provider'] = 'openai-codex'
        assert agent._try_activate_fallback()
        assert agent.model == BEDROCK
        assert agent._try_activate_fallback()
        first = agent._cached_system_prompt
        assert first.count(ENFORCEMENT) == first.count(execution_guidance_text(agent.valid_tool_names)) == 1
        # The routing layer skips the backend already bound. Replay its prompt seam directly.
        from agent.chat_completion_helpers import rewrite_prompt_model_identity
        rewrite_prompt_model_identity(agent, agent.model, agent.provider, fallback_from=(agent.model, agent.provider))
        assert agent._cached_system_prompt == first


def test_p5_primary_and_rebuilt_sections_are_not_owned_by_fallback(agent):
    agent._tool_use_enforcement = True
    agent._cached_system_prompt = agent._build_system_prompt(None)
    assert agent._cached_system_prompt.count(ENFORCEMENT) == 1
    with chain(agent, ASTRA, ASTRA):
        assert agent._try_activate_fallback()
        assert agent._cached_system_prompt.count(ENFORCEMENT) == 1
        rebuilt = agent._build_system_prompt(None)
        agent._cached_system_prompt = rebuilt
        assert agent._try_activate_fallback()
        assert agent._cached_system_prompt == rebuilt
    assert agent._restore_primary_runtime()
    assert agent._cached_system_prompt.count(execution_guidance_text(agent.valid_tool_names)) == 1


def test_p6_fallback_served_turn_keeps_primary_row(agent, tmp_path):
    db = SessionDB(db_path=tmp_path / 'state.db')
    try:
        agent._session_db = db
        db.create_session(agent.session_id, source='cli', model=agent.model)
        agent._cached_system_prompt = None
        agent.compression_enabled = False
        primary = []

        def respond(kwargs):
            if agent.model == FABLE:
                primary.append(agent._cached_system_prompt)
                raise RateLimitError()
            assert agent.model.rsplit('/', 1)[-1] == ASTRA
            return _mock_response('Served by fallback')

        with chain(agent, ASTRA), patch.object(agent, '_interruptible_api_call', side_effect=respond):
            result = agent.run_conversation('hello')
        assert result['final_response'] == 'Served by fallback'
        assert ENFORCEMENT in agent._cached_system_prompt
        assert primary and ENFORCEMENT not in primary[0]
        assert db.get_session(agent.session_id)['system_prompt'] == primary[0]
    finally:
        db.close()


def test_p7_switch_discards_old_fallback_guidance(agent):
    with chain(agent, ASTRA):
        assert agent._try_activate_fallback()
    assert ENFORCEMENT in agent._cached_system_prompt
    agent.switch_model(FABLE, 'custom', api_key='test-key', base_url='http://localhost/v1')
    assert not agent._fallback_activated
    assert ENFORCEMENT not in (agent._cached_system_prompt or agent._build_system_prompt(None))
