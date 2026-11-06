"""A managed worker's agent keeps the session's own reasoning level across a provider fallback.

``hermes -z --reasoning high`` (or a committed ``/reasoning``) is the session's choice and outranks
``agent.reasoning_overrides``. Bypass sessions (``--ignore-user-config`` / ``--safe-mode``), Kanban runs
and the ``gateway.managed_workers`` opt-in build their agent in ``agent.managed_worker`` instead of the
in-process turn runner, so that construction must hand the pin to the agent too; a session without a
level still resolves the fallback model's config.
"""
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_constants import get_hermes_home
from tests.gateway.test_managed_worker_route_key import _constructed_agent


def _frame(tmp_path, monkeypatch, **launch):
    from gateway.session_local import _bypass_policy
    from gateway.session_managed_worker import _bootstrap
    policy = _bypass_policy({'cwd': str(tmp_path), 'model': 'primary-model', 'provider': 'custom',
                             'base_url': 'http://127.0.0.1:9/v1', 'ignore_user_config': True, **launch},
                            private_secrets={})
    config = json.loads(policy.config_json)
    config['fallback_providers'] = [{'provider': 'openai', 'model': 'fb-model'}]
    policy = replace(policy, config_json=json.dumps(config))
    monkeypatch.setattr('gateway.session_policy.launch_key', lambda *args: None)
    live = SimpleNamespace(source=SimpleNamespace(user_id='u', chat_id='c'), route='managed-reasoning')
    authority = SimpleNamespace(sessions={'sid': live}, runner=None, profile_id=str(tmp_path))
    row = {'payload': {'text': 'hello', 'finite': True, 'unattended': False}}
    scope = {'profile_id': str(tmp_path), 'session_id': 'sid', 'execution_id': 'x', 'generation': 1,
             'pid': 1, 'birth': 0, 'secret': 's', 'epoch': 1}
    return _bootstrap(authority, SimpleNamespace(session_id='sid'), row, policy, scope)


@pytest.fixture
def fallback_config():
    """The config a fallback re-resolves reasoning from: the fallback model's own level is ``low``."""
    (get_hermes_home() / 'config.yaml').write_text(
        'agent:\n  reasoning_effort: medium\n  reasoning_overrides:\n    fb-model: low\n', encoding='utf-8')


def _level_after_fallback(agent):
    client = MagicMock()
    client.base_url, client.api_key = 'https://api.openai.com/v1', 'fb-key'
    with patch('agent.auxiliary_client.resolve_provider_client', return_value=(client, 'fb-model')):
        assert agent._try_activate_fallback() is True
    return agent.model, agent.reasoning_config['effort']


def test_launch_level_survives_fallback_in_a_managed_worker(tmp_path, monkeypatch, fallback_config):
    agent = _constructed_agent(_frame(tmp_path, monkeypatch, reasoning='high'), monkeypatch)
    assert _level_after_fallback(agent) == ('fb-model', 'high')


def test_without_a_level_the_worker_resolves_the_fallback_models_config(tmp_path, monkeypatch, fallback_config):
    agent = _constructed_agent(_frame(tmp_path, monkeypatch), monkeypatch)
    assert _level_after_fallback(agent) == ('fb-model', 'low')
