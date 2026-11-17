"""In-process TUI gateway: the route-change ``session.info`` comes from the REAL ``_session_info``.

``tests/tui_gateway/test_make_agent_route_change.py`` stubs ``_session_info``. Here the real builder
runs, so the emitted model and provider are what it reads off the agent after the switch.
"""

import types

from agent.route_binding import notify_route_change
from tui_gateway import server

FALLBACK = ("gpt-6-astra-900k", "openai-codex")


def _built_agent(monkeypatch):
    def fake_agent(**kwargs):
        return types.SimpleNamespace(model=kwargs.get("model"), provider=kwargs.get("provider"))

    for name in ("HERMES_MODEL", "HERMES_INFERENCE_MODEL", "HERMES_TUI_PROVIDER", "HERMES_DESKTOP"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": {"default": "claude-fable-5-2", "provider": "anthropic"}})
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, target_model=None: {
            "provider": "anthropic", "base_url": "https://api.anthropic.com", "api_key": "token",
            "api_mode": "anthropic_messages", "credential_pool": None,
        },
    )
    monkeypatch.setattr("run_agent.AIAgent", fake_agent)
    monkeypatch.setattr(server, "_load_enabled_toolsets", lambda platform=None: ["file"])
    monkeypatch.setattr(server, "_get_db", lambda: None)
    return server._make_agent("sid-1", "session-key")


def test_route_change_emits_the_agents_new_route_through_the_real_session_info(monkeypatch):
    agent = _built_agent(monkeypatch)
    assert (agent.model, agent.provider) == ("claude-fable-5-2", "anthropic")
    monkeypatch.setitem(server._sessions, "sid-1", {"agent": agent, "session_key": "session-key"})
    emitted = []
    monkeypatch.setattr(server, "_emit", lambda event, sid, payload=None: emitted.append((event, sid, payload)))
    agent.model, agent.provider = FALLBACK
    assert notify_route_change(agent) is True
    assert [(event, sid) for event, sid, _ in emitted] == [("session.info", "sid-1")]
    assert (emitted[0][2]["model"], emitted[0][2]["provider"]) == FALLBACK
    assert emitted[0][2]["stored_session_id"] == "session-key"
