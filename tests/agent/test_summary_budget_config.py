"""``compression.summary_tokens_ceiling`` and ``compression.summary_session_log``.

A compaction waits on the summary model writing its whole target, so the target has to be reachable
from config. The ceiling replaces the fixed 10,000-token cap on the summary budget (the 2,000 floor
stays); the flag drops the lean "Detailed Session Log" section and its extra allowance. Every path
that asks for a summary (fresh, forced, iterative update, prepared copy, a live TUI session after a
config edit) must send the summariser the configured target, and the defaults must send what an
unconfigured compressor sends.
"""

from __future__ import annotations

import contextlib
import io
import re
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.context_compressor import _LEAN_SESSION_LOG_HEADING, ContextCompressor
from hermes_cli.config_defaults import DEFAULT_CONFIG

WINDOW = 1_000_000  # 5% of the window is far above every ceiling used here

SUMMARY_BODY = (
    "## Historical Task Snapshot\nUser asked: 'Please fix PR #12345 in agent/foo.py'\n\n"
    "## Goal\nFix the bug.\n\n## Completed Actions\n1. EDIT agent/foo.py - done [tool: patch]\n"
)


def _compressor(**settings) -> ContextCompressor:
    c = ContextCompressor(
        model="unset-test-model", config_context_length=WINDOW, quiet_mode=True,
        protect_first_n=2, protect_last_n=2, **settings,
    )
    c._session_id = "sess-summary-budget"
    return c


def _large_middle(n_rounds: int = 60, tool_chars: int = 6_000) -> list[dict]:
    """About 90K tokens: 20% of it is above the default ceiling."""
    turns = [{"role": "user", "content": "Please fix PR #12345 in agent/foo.py"}]
    for i in range(n_rounds):
        turns.append({"role": "assistant", "content": f"Working on step {i}: editing agent/foo.py line {i}"})
        turns.append({"role": "tool", "tool_call_id": f"tc{i}", "tool_name": "terminal",
                      "content": f"round {i} output: " + ("x" * tool_chars)})
    return turns


def _llm_response(text: str = SUMMARY_BODY):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message = MagicMock()
    resp.choices[0].message.content = text
    return resp


def _summary_prompt(compressor, turns=None, **kwargs) -> str:
    """The prompt one summary request sends to the summary model."""
    with patch("agent.context_compressor.call_llm", return_value=_llm_response()) as call:
        assert compressor._generate_summary(turns or _large_middle(), **kwargs) is not None
    assert call.call_count == 1
    return call.call_args.kwargs["messages"][0]["content"]


def _target(prompt: str) -> int:
    targets = re.findall(r"Target ~(\d+) tokens", prompt)
    assert len(targets) == 1, targets
    return int(targets[0])


# ── the budget ───────────────────────────────────────────────────────────────


def test_ceiling_caps_the_summary_target_on_a_large_middle():
    prompt = _summary_prompt(_compressor(summary_tokens_ceiling=4000, summary_session_log=False))
    assert _target(prompt) == 4000


def test_ceiling_can_raise_the_target_above_the_default():
    default = _target(_summary_prompt(_compressor(summary_session_log=False)))
    raised = _target(_summary_prompt(_compressor(summary_tokens_ceiling=50_000, summary_session_log=False)))
    assert raised > default


@pytest.mark.parametrize("ceiling", [1999, 500, 0, -1])
def test_ceiling_under_the_floor_acts_as_the_floor(ceiling):
    floor = _target(_summary_prompt(_compressor(summary_tokens_ceiling=2000, summary_session_log=False)))
    low = _compressor(summary_tokens_ceiling=ceiling, summary_session_log=False)
    assert _target(_summary_prompt(low)) == floor == 2000
    # The micro-compaction request cap is derived from the same value and must stay positive.
    assert low.max_summary_tokens > 0


def test_default_ceiling_is_the_documented_default_and_matches_an_unconfigured_compressor():
    explicit = _compressor(summary_tokens_ceiling=DEFAULT_CONFIG["compression"]["summary_tokens_ceiling"],
                           summary_session_log=DEFAULT_CONFIG["compression"]["summary_session_log"])
    assert _summary_prompt(_compressor()) == _summary_prompt(explicit)
    assert _target(_summary_prompt(_compressor(summary_session_log=False))) == 10_000


# ── the session log section ──────────────────────────────────────────────────


def test_session_log_off_drops_the_section_and_its_allowance():
    on = _summary_prompt(_compressor(summary_tokens_ceiling=4000))
    off = _summary_prompt(_compressor(summary_tokens_ceiling=4000, summary_session_log=False))
    assert _LEAN_SESSION_LOG_HEADING in on
    assert _LEAN_SESSION_LOG_HEADING not in off
    assert "session log" not in off.lower()
    assert _target(off) == 4000 < _target(on)
    # Nothing else is removed: the prompt without the flag is the prompt with it, minus the
    # section and the allowance in the target line.
    section_start = on.index("\n\n" + _LEAN_SESSION_LOG_HEADING)
    section_end = on.index("\n\n", on.index(_LEAN_SESSION_LOG_HEADING))
    without_section = on[:section_start] + on[section_end:]
    assert without_section.replace(f"Target ~{_target(on)} tokens", "Target ~4000 tokens") == off


def test_session_log_off_keeps_the_code_appended_parts_of_the_summary():
    from agent.context_compressor import _LEAN_ANCHOR_HEADING, _LEAN_RECOVERY_HEADING

    c = _compressor(summary_session_log=False)
    with patch("agent.context_compressor.call_llm", return_value=_llm_response()):
        summary = c._generate_summary(_large_middle(n_rounds=5))
    assert _LEAN_ANCHOR_HEADING in summary
    assert _LEAN_RECOVERY_HEADING in summary


def test_legacy_tail_mode_prompt_is_the_same_with_the_flag_on_or_off():
    on = _summary_prompt(_compressor(tail_mode="legacy", summary_session_log=True))
    off = _summary_prompt(_compressor(tail_mode="legacy", summary_session_log=False))
    assert on == off
    assert _LEAN_SESSION_LOG_HEADING not in on


# ── every path that builds a summary ─────────────────────────────────────────


@pytest.mark.parametrize("force", [False, True], ids=["automatic", "forced"])
def test_compaction_sends_the_configured_target(force):
    c = _compressor(summary_tokens_ceiling=4000, summary_session_log=False)
    # Plain text turns: tool output would be pruned before the budget is computed.
    messages = [{"role": "system", "content": "sys"}]
    for i in range(120):
        messages.append({"role": "user", "content": f"Request {i} about agent/foo.py: " + f"detail-{i} " * 400})
        messages.append({"role": "assistant", "content": f"Answer {i}: " + f"finding-{i} " * 400})
    with patch("agent.context_compressor.call_llm", return_value=_llm_response()) as call:
        compressed = c.compress(messages, current_tokens=600_000, force=force)
    assert any(m.get("_compressed_summary") for m in compressed)
    prompt = call.call_args.kwargs["messages"][0]["content"]
    assert _target(prompt) == 4000
    assert _LEAN_SESSION_LOG_HEADING not in prompt


def test_iterative_update_sends_the_configured_target():
    c = _compressor(summary_tokens_ceiling=4000, summary_session_log=False)
    c._previous_summary = "## Goal\nEarlier work on PR #12345."
    prompt = _summary_prompt(c)
    assert "PREVIOUS SUMMARY:" in prompt
    assert _target(prompt) == 4000
    assert _LEAN_SESSION_LOG_HEADING not in prompt


def test_prepared_pass_copy_sends_the_configured_target():
    from agent.prepared_compaction import _worker_copy

    c = _compressor(summary_tokens_ceiling=4000, summary_session_log=False)
    prompt = _summary_prompt(_worker_copy(c))
    assert _target(prompt) == 4000
    assert _LEAN_SESSION_LOG_HEADING not in prompt


def test_model_switch_keeps_the_configured_ceiling():
    c = _compressor(summary_tokens_ceiling=4000, summary_session_log=False)
    c.update_model("other-test-model", 600_000)
    assert _target(_summary_prompt(c)) == 4000


# ── config: the real init path ───────────────────────────────────────────────


def _make_agent(monkeypatch, tmp_path, **compression_keys):
    from hermes_cli import config as config_mod
    from hermes_state import SessionDB
    from run_agent import AIAgent

    cfg = {
        "compression": {"enabled": True, "threshold": 0.50, **compression_keys},
        "prompt_caching": {"cache_ttl": "5m"}, "sessions": {}, "bedrock": {},
    }
    monkeypatch.setattr(config_mod, "load_config", lambda: cfg)
    monkeypatch.setattr(config_mod, "load_config_readonly", lambda: cfg)
    with contextlib.redirect_stdout(io.StringIO()):
        return AIAgent(
            base_url="https://chatgpt.com/backend-api/codex", api_key="test-key", provider="openai-codex",
            model="gpt-5.5", enabled_toolsets=[], disabled_toolsets=[], quiet_mode=True, skip_memory=True,
            session_db=SessionDB(db_path=tmp_path / "state.db"), session_id="summary-budget-config-test",
        )


def test_config_keys_reach_the_compressor_and_its_prompt(monkeypatch, tmp_path):
    agent = _make_agent(monkeypatch, tmp_path, summary_tokens_ceiling=4000, summary_session_log=False)
    prompt = _summary_prompt(agent.context_compressor)
    assert _target(prompt) == 4000
    assert _LEAN_SESSION_LOG_HEADING not in prompt


def test_absent_config_keys_build_the_default_compressor(monkeypatch, tmp_path):
    cc = _make_agent(monkeypatch, tmp_path).context_compressor
    assert cc.summary_tokens_ceiling == DEFAULT_CONFIG["compression"]["summary_tokens_ceiling"]
    assert cc.summary_session_log is DEFAULT_CONFIG["compression"]["summary_session_log"] is True
    prompt = _summary_prompt(cc)
    assert _LEAN_SESSION_LOG_HEADING in prompt


@pytest.mark.parametrize(("raw", "expected"), [("4000", 4000), (4000.0, 4000), (True, 10_000), ("lots", 10_000)])
def test_ceiling_config_values_are_parsed_strictly(monkeypatch, tmp_path, raw, expected):
    agent = _make_agent(monkeypatch, tmp_path, summary_tokens_ceiling=raw)
    assert agent.context_compressor.summary_tokens_ceiling == expected


@pytest.mark.parametrize(("raw", "expected"), [("false", False), ("off", False), (0, False), ("true", True)])
def test_session_log_config_values_are_read_as_flags(monkeypatch, tmp_path, raw, expected):
    agent = _make_agent(monkeypatch, tmp_path, summary_session_log=raw)
    assert agent.context_compressor.summary_session_log is expected


# ── config: a live Desktop/TUI session ───────────────────────────────────────


def _live_session(compressor):
    agent = SimpleNamespace(
        model="unset-test-model", provider="", base_url="", context_compressor=compressor,
        compression_enabled=True, compression_idle_compact_after_seconds=0,
        codex_responses_native_compaction=False, codex_responses_compact_threshold=200_000,
    )
    return {"agent": agent, "session_key": "session-summary-budget"}


def test_live_tui_session_adopts_and_then_drops_the_keys(monkeypatch):
    from tui_gateway import server

    compressor = _compressor()
    session = _live_session(compressor)
    model = {"context_length": WINDOW}
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": model, "compression": {}})
    server._sync_agent_compression_with_config("sid", session)
    default_prompt = _summary_prompt(compressor)
    assert _LEAN_SESSION_LOG_HEADING in default_prompt
    compressor._previous_summary = None  # keep each request a fresh summary, so the prompts compare

    edited = {"model": model, "compression": {"summary_tokens_ceiling": 4000, "summary_session_log": False}}
    monkeypatch.setattr(server, "_load_cfg", lambda: edited)
    server._sync_agent_compression_with_config("sid", session)
    prompt = _summary_prompt(compressor)
    assert _target(prompt) == 4000
    assert _LEAN_SESSION_LOG_HEADING not in prompt
    compressor._previous_summary = None

    # Removing the keys restores what a fresh build sends.
    monkeypatch.setattr(server, "_load_cfg", lambda: {"model": model, "compression": {}})
    server._sync_agent_compression_with_config("sid", session)
    assert _summary_prompt(compressor) == default_prompt
