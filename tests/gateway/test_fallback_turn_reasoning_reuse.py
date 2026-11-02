"""A reused gateway agent drops the previous turn's reasoning pin.

The gateway caches one agent per session and rewires it every turn. A turn whose API request names a
reasoning level pins it, so a provider fallback in that turn keeps it. The next turn on the same
cached agent carries no level of its own; when it falls back, the fallback model's configured level
applies. A turn that fell back evicts the cached agent at its end, so the pinned turn here stays on
the primary and the later turn falls back.
"""

import asyncio
import threading
from collections import OrderedDict

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.session_api_turn import api_execution
from gateway.session_cli_turn import cli_turn_scope
from tests.gateway.test_fallback_turn_reasoning_overrides import (
    _ADMISSION, _CONFIG, _FALLBACK, _FallsBackAgent, _runner, gateway_home,
)


class _FallsBackOnRequest(_FallsBackAgent):
    """The real agent. A "fall back" message runs the parent's fallback turn; any other message
    stays on the primary and reports the level and pin it ran with. ``runs`` records which agent
    object served each turn."""

    runs: list = []

    def run_conversation(self, user_message, conversation_history=None, task_id=None, **kwargs):
        type(self).runs.append(self)
        if "fall back" in user_message:
            return super().run_conversation(user_message, conversation_history, task_id, **kwargs)
        pinned = bool(getattr(self, "reasoning_pinned", False))
        return {"final_response": f"{self.model}:{(self.reasoning_config or {}).get('effort')} pinned={pinned}",
                "messages": [], "api_calls": 1}


def _turn(runner, message, model_options):
    """One gateway turn through ``GatewayRunner._run_agent``, sent as an API-server request with
    *model_options* on the session's own route (so both turns share one agent signature)."""
    from types import SimpleNamespace
    adapter = SimpleNamespace(_model_routes={}, _select_agent_runtime=lambda _rt, model, **_k: (model, None, None, None))
    token = api_execution.set({"adapter": adapter, "settings": {"model_options": model_options}, "history": None})
    source = SessionSource(platform=Platform.TELEGRAM, user_id="12345", chat_id="67890", user_name="testuser")
    try:
        with cli_turn_scope(_ADMISSION, None):
            return asyncio.run(runner._run_agent(
                message=message, context_prompt="", history=[], source=source, session_id="session-1",
                session_key=runner._session_key_for_source(source), inbound_message_id=_ADMISSION))
    finally:
        api_execution.reset(token)


def _cached_agent(runner):
    entries = [entry[0] for entry in runner._agent_cache.values() if isinstance(entry, tuple)]
    assert len(entries) == 1, f"expected one cached agent, found {len(entries)}"
    return entries[0]


def test_reused_agent_drops_the_previous_turns_pin(gateway_home, monkeypatch):
    # A configured primary, so the turn that stays on it keeps its cached agent (the post-turn
    # fallback check evicts an agent whose model differs from the configured one).
    (gateway_home / "config.yaml").write_text(_CONFIG + "model:\n  default: primary-model\n  provider: openrouter\n",
                                              encoding="utf-8")
    monkeypatch.setattr("run_agent.AIAgent", _FallsBackOnRequest)
    monkeypatch.setattr(_FallsBackOnRequest, "runs", [])
    runner = _runner()
    # The cache GatewayRunner._init_runtime_caches builds; _resolve_turn_agent fills and reads it.
    runner._agent_cache, runner._agent_cache_lock = OrderedDict(), threading.Lock()

    first = _turn(runner, "ping", {"reasoning": {"effort": "high"}})
    cached = _cached_agent(runner)
    second = _turn(runner, "please fall back", {})

    assert first["final_response"] == "primary-model:high pinned=True"
    runs = _FallsBackOnRequest.runs
    assert len(runs) == 2 and runs[0] is runs[1] is cached, "the second turn must reuse the cached agent"
    # Unpinned again, so the fallback takes fb-model's configured low instead of keeping a level.
    assert second["final_response"] == f"medium>{_FALLBACK['model']}:low pinned=False"
