"""A mid-turn route change names the serving route through the REAL ``local_session_info``.

``tests/gateway/test_local_viewer_route_change.py`` replaces ``local_session_info`` with a stub, so
it cannot tell whether the real builder reports the running agent's route or a stale one. Here the
stored session row and the cached agent both still name the primary, the running agent has moved
to the fallback, and the frame goes through the real authority's ``publish_execution`` onto the
session's event stream.
"""
from __future__ import annotations

import types
from types import SimpleNamespace

import pytest

from agent.route_binding import notify_route_change
from tests.fixtures.storage_seed import seed_session
from gateway import local_viewer_events
from gateway.config import GatewayConfig, Platform
from gateway.run_turn_runner import TurnRunner

PRIMARY = ('global.anthropic.claude-fable-5-2', 'bedrock')
FALLBACK = ('gpt-6-astra-900k', 'openai-codex')


def _agent(model, provider):
    return SimpleNamespace(model=model, provider=provider, route_change_callback=None,
                           tool_gen_callback=None, _print_fn=None, tools=[], _cached_system_prompt='')


async def _serving(tmp_path, monkeypatch, running, cached):
    """A real authority whose session row stores the primary; returns ``(turn, frames)``."""
    from gateway.session import SessionSource, SessionStore
    from gateway.session_authority import LiveSession, initialize_session_authority
    from gateway.slash_commands import GatewaySlashCommandsMixin

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr(local_viewer_events, '_config_default_launch', lambda: False)
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    runner = SimpleNamespace(persistence=store.persistence, session_store=store, _draining=False, config=None,
                             _adapter_for_source=lambda source: None,
                             _running_agents={'route-s': running} if running is not None else {},
                             _cached_agent_for=lambda key: cached if key == 'route-s' else None)
    # The production lookup: the running agent first, else the cached one.
    runner._resident_agent_for = types.MethodType(GatewaySlashCommandsMixin._resident_agent_for, runner)
    authority = await initialize_session_authority(runner, profile_id='p', instance_id='owner')
    seed_session(authority.persistence, 's', 'telegram', model=PRIMARY[0])
    assert authority.persistence.sessions.get('s').model == PRIMARY[0]
    source = SessionSource(platform=Platform.TELEGRAM, chat_id='c')
    live = authority.sessions['s'] = LiveSession(source, 'route-s')
    authority.executing['s'] = 1
    frames = []
    publish = live.event_stream.publish

    def recording(session_id, payload, event_type=None):
        frames.append((event_type, payload))
        return publish(session_id, payload, event_type=event_type)
    monkeypatch.setattr(live.event_stream, 'publish', recording)
    turn = object.__new__(TurnRunner)
    turn._ctx = SimpleNamespace(source=source, mute_notification_reply=False, session_id='s',
                                inbound_message_id=None)
    turn._runner = runner
    turn._approval_owner = (authority, 's', 1)
    return turn, frames


def _route_infos(frames):
    return [(p.get('model'), p.get('provider')) for kind, p in frames if kind == 'session.info']


@pytest.mark.asyncio
async def test_route_change_frame_names_the_running_agents_route_not_the_stored_model(tmp_path, monkeypatch):
    running, stale_cached = _agent(*PRIMARY), _agent(*PRIMARY)
    turn, frames = await _serving(tmp_path, monkeypatch, running, stale_cached)
    previous = local_viewer_events.wire(turn, running)
    running.model, running.provider = FALLBACK
    assert notify_route_change(running) is True
    assert _route_infos(frames) == [FALLBACK]
    # The restore to the primary publishes again, on the primary.
    running.model, running.provider = PRIMARY
    assert notify_route_change(running) is True
    assert _route_infos(frames) == [FALLBACK, PRIMARY]
    local_viewer_events.restore(running, previous)
    assert running.route_change_callback is None


@pytest.mark.asyncio
async def test_a_late_callback_after_the_turn_stopped_executing_publishes_nothing(tmp_path, monkeypatch):
    running = _agent(*PRIMARY)
    turn, frames = await _serving(tmp_path, monkeypatch, running, None)
    local_viewer_events.wire(turn, running)
    turn._approval_owner[0].executing.pop('s')
    running.model, running.provider = FALLBACK
    assert notify_route_change(running) is True
    assert frames == []
