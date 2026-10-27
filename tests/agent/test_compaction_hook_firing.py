import pytest
from types import SimpleNamespace
from hermes_cli.plugins import PluginManager
from agent.conversation_compression import _finish_compaction_boundary
from agent.codex_runtime import _record_codex_app_server_compaction
from agent.transports.codex_app_server_session import TurnResult
from agent.context_compressor import ContextCompressor

class DummyAgent:
    def __init__(self):
        self.session_id = "test-session"
        self.platform = "cli"
        self.log_prefix = ""
        self.context_compressor = ContextCompressor(model="test-model", quiet_mode=True)
        self.tools = []
        self._memory_manager = None
        self._last_compaction_in_place = False
        self._last_compression_attempt_in_place = False
        self.events = []
        self.event_callback = lambda name, payload: self.events.append((name, payload))

    def _emit_status(self, *a, **kw):
        pass

    def _emit_diagnostic_status(self, *a, **kw):
        pass

def test_chat_completions_compaction_fires_on_compaction_complete_hook(monkeypatch):
    mgr = PluginManager()
    events = []
    def on_compaction(**kwargs):
        events.append(kwargs)

    mgr._hooks.setdefault("on_compaction_complete", []).append(on_compaction)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)
    monkeypatch.setattr("agent.conversation_compression.invoke_hook", mgr.invoke_hook, raising=False)

    agent = DummyAgent()
    agent.context_compressor.compression_count = 1

    _finish_compaction_boundary(
        agent=agent,
        compressed=[{"role": "user", "content": "hello"}],
        compacted_in_place=True,
        compression_made_progress=True,
        compression_used_fallback=False,
        compression_feasibility_skip=False,
        new_system_prompt="sys",
        task_id="t1",
        in_place=True,
        old_session_id=None,
        session_commit_succeeded=True,
        defer_context_engine_notification=False,
    )

    assert len(events) == 1
    assert events[0]["session_id"] == "test-session"
    assert events[0]["compression_count"] == 1
    assert events[0]["in_place"] is True
    assert events[0]["runtime"] == "chat_completions"

def test_codex_compaction_fires_on_compaction_complete_hook(monkeypatch):
    mgr = PluginManager()
    events = []
    def on_compaction(**kwargs):
        events.append(kwargs)

    mgr._hooks.setdefault("on_compaction_complete", []).append(on_compaction)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)
    monkeypatch.setattr("agent.codex_runtime.invoke_hook", mgr.invoke_hook, raising=False)

    agent = DummyAgent()
    turn = TurnResult(thread_id="th-1", turn_id="tu-1", compacted=True)
    turn.compaction_count = 2

    assert _record_codex_app_server_compaction(agent, turn) is True
    assert len(events) == 1
    assert events[0]["session_id"] == "test-session"
    assert events[0]["compression_count"] == 2
    assert events[0]["in_place"] is False
    assert events[0]["runtime"] == "codex_app_server"
    assert events[0]["thread_id"] == "th-1"
    assert events[0]["turn_id"] == "tu-1"

def test_codex_midturn_and_finish_does_not_fire_hook_twice_for_final_compaction(monkeypatch):
    """B3 requirement: In a healthy turn with 2 compactions, hook events must fire exactly
    once per distinct compaction: [1, 2], not [1, 2, 2]."""
    from agent.codex_runtime import _on_codex_app_server_compaction_event

    mgr = PluginManager()
    events = []
    def on_compaction(**kwargs):
        events.append(kwargs)

    mgr._hooks.setdefault("on_compaction_complete", []).append(on_compaction)
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)
    monkeypatch.setattr("agent.codex_runtime.invoke_hook", mgr.invoke_hook, raising=False)

    agent = DummyAgent()

    # 2 mid-turn compactions occur during the turn
    _on_codex_app_server_compaction_event(agent, 1)
    _on_codex_app_server_compaction_event(agent, 2)

    assert [e["compression_count"] for e in events] == [1, 2]

    # Now turn finishes and calls _record_codex_app_server_compaction
    turn = TurnResult(thread_id="th-1", turn_id="tu-1", compacted=True)
    turn.compaction_count = 2
    _record_codex_app_server_compaction(agent, turn)

    # Must still be exactly [1, 2] - no duplicate event for count=2!
    counts = [e["compression_count"] for e in events]
    assert counts == [1, 2], f"Expected distinct counts [1, 2], got {counts}"


def test_codex_session_is_built_with_the_midturn_compaction_callback(monkeypatch):
    """_ensure_codex_session hands the session an on_compaction callback, so a compaction inside a
    running turn fires on_compaction_complete before the turn ends (in-turn checkpoint delivery)."""
    from agent import codex_runtime
    import agent.transports.codex_app_server_session as session_module

    built = {}

    class _RecordingSession:
        def __init__(self, **kwargs):
            built.update(kwargs)

    monkeypatch.setattr(session_module, "CodexAppServerSession", _RecordingSession)
    for name, value in (("_codex_developer_instructions", lambda agent: "system"),
                        ("_codex_model_provider", lambda agent: None),
                        ("_stored_codex_thread_id", lambda agent: None),
                        ("make_codex_app_server_event_bridge", lambda agent: None),
                        ("_codex_wire_model", lambda agent, provider: "gpt-5")):
        monkeypatch.setattr(codex_runtime, name, value)
    mgr = PluginManager()
    counts = []
    mgr._hooks.setdefault("on_compaction_complete", []).append(lambda **kw: counts.append(kw["compression_count"]))
    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", mgr.invoke_hook)
    agent = DummyAgent()

    codex_runtime._ensure_codex_session(agent, [])
    built["on_compaction"](1)

    assert counts == [1]
    assert agent.context_compressor.compression_count == 1
