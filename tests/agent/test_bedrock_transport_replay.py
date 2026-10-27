"""Production Bedrock transport/history replay regressions."""

import sys
from types import ModuleType

import pytest


class _FakeAgent:
    verbose_logging = False
    reasoning_callback = None
    stream_delta_callback = None
    _stream_callback = None

    @staticmethod
    def _extract_reasoning(message):
        return getattr(message, "reasoning_content", None)

    @staticmethod
    def _strip_think_blocks(text):
        return text

    @staticmethod
    def _needs_thinking_reasoning_pad():
        return False

    @staticmethod
    def _split_responses_tool_id(value):
        return value, None

    @staticmethod
    def _deterministic_call_id(name, args, index):
        return f"call-{index}"

    @staticmethod
    def _derive_responses_function_call_id(value, response_item_id=None):
        return value


def _raw_response():
    return {
        "output": {"message": {"role": "assistant", "content": [
            {"reasoningContent": {"redactedContent": b"r1"}},
            {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}},
            {"reasoningContent": {"redactedContent": b"r2"}},
            {"toolUse": {"toolUseId": "t2", "name": "two", "input": {"n": 2}}},
        ]}},
        "stopReason": "tool_use",
    }


def _stream_response():
    return {"stream": [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {
            "reasoningContent": {"redactedContent": b"r1"},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
        {"contentBlockStart": {"contentBlockIndex": 1, "start": {
            "toolUse": {"toolUseId": "t1", "name": "one"},
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 1, "delta": {
            "toolUse": {"input": '{"n":1}'},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 1}},
        {"contentBlockStart": {"contentBlockIndex": 2, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 2, "delta": {
            "reasoningContent": {"redactedContent": b"r2"},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 2}},
        {"contentBlockStart": {"contentBlockIndex": 3, "start": {
            "toolUse": {"toolUseId": "t2", "name": "two"},
        }}},
        {"contentBlockDelta": {"contentBlockIndex": 3, "delta": {
            "toolUse": {"input": '{"n":2}'},
        }}},
        {"contentBlockStop": {"contentBlockIndex": 3}},
        {"messageStop": {"stopReason": "tool_use"}},
    ]}


@pytest.mark.parametrize("streaming", [False, True])
def test_bedrock_transport_preserves_reasoning_and_order(streaming):
    from agent.bedrock_adapter import normalize_converse_response, normalize_converse_stream_events, convert_messages_to_converse
    from agent.transports.bedrock import BedrockTransport

    raw = _stream_response() if streaming else _raw_response()
    adapter_response = (
        normalize_converse_stream_events(raw)
        if streaming else normalize_converse_response(raw)
    )
    normalized = BedrockTransport().normalize_response(adapter_response)
    assert normalized.provider_data["reasoning_details"] == [
        {"type": "redacted_thinking", "data": "cjE="},
        {"type": "redacted_thinking", "data": "cjI="},
    ]

    # Exercise the real history builder against the transport's canonical
    # NormalizedResponse, rather than the adapter-only SimpleNamespace.
    sys.modules.setdefault("requests", ModuleType("requests"))
    from agent.chat_completion_helpers import build_assistant_message
    history = build_assistant_message(_FakeAgent(), normalized, "tool_calls")
    assert history["reasoning_details"] == normalized.provider_data["reasoning_details"]
    assert history["bedrock_content_blocks"] == normalized.provider_data["bedrock_content_blocks"]
    _system, messages = convert_messages_to_converse([
        {"role": "user", "content": "go"}, history,
    ])
    blocks = messages[1]["content"]
    assert [next(iter(block)) for block in blocks] == [
        "reasoningContent", "toolUse", "reasoningContent", "toolUse"
    ]
    assert [block["reasoningContent"]["redactedContent"] for block in blocks if "reasoningContent" in block] == [b"r1", b"r2"]


@pytest.mark.parametrize("streaming", [False, True])
def test_tool_call_dropped_after_capture_is_not_replayed(streaming):
    """The model emits the same call twice; the post-call dedup drops the second from
    ``tool_calls``, but the ordered-block sidecar still holds its toolUse. Replaying it sends a
    toolUse with no toolResult -> ValidationException "Expected toolResult blocks"."""
    from agent.bedrock_adapter import normalize_converse_response, normalize_converse_stream_events, convert_messages_to_converse
    from agent.transports.bedrock import BedrockTransport

    sys.modules.setdefault("requests", ModuleType("requests"))
    from agent.chat_completion_helpers import build_assistant_message
    from run_agent import AIAgent

    raw = _stream_response() if streaming else _raw_response()
    # Make the second call identical to the first, as the model did.
    if streaming:
        for event in raw["stream"]:
            start = event.get("contentBlockStart", {}).get("start", {}).get("toolUse")
            if start and start["toolUseId"] == "t2":
                start["name"] = "one"
            delta = event.get("contentBlockDelta", {})
            if delta.get("contentBlockIndex") == 3:
                delta["delta"]["toolUse"]["input"] = '{"n":1}'
    else:
        raw["output"]["message"]["content"][3]["toolUse"].update(name="one", input={"n": 1})
    adapter_response = normalize_converse_stream_events(raw) if streaming else normalize_converse_response(raw)
    normalized = BedrockTransport().normalize_response(adapter_response)

    normalized.tool_calls = AIAgent._deduplicate_tool_calls(normalized.tool_calls)
    history = build_assistant_message(_FakeAgent(), normalized, "tool_calls")
    assert [tc["id"] for tc in history["tool_calls"]] == ["t1"]
    results = [{"role": "tool", "tool_call_id": "t1", "content": "ok"}]
    _system, messages = convert_messages_to_converse(
        AIAgent._sanitize_api_messages([{"role": "user", "content": "go"}, history, *results])
    )

    blocks = messages[1]["content"]
    assert [next(iter(block)) for block in blocks] == ["reasoningContent", "toolUse", "reasoningContent"]
    tool_uses = [b["toolUse"]["toolUseId"] for m in messages for b in m["content"] if "toolUse" in b]
    tool_results = [b["toolResult"]["toolUseId"] for m in messages for b in m["content"] if "toolResult" in b]
    assert tool_uses == tool_results == ["t1"]


def _opus_duplicate_terminal_stream():
    """Opus 5.5 on ConverseStream: signed thinking, then the same ``terminal`` call twice."""
    events = [
        {"messageStart": {"role": "assistant"}},
        {"contentBlockStart": {"contentBlockIndex": 0, "start": {}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "run it"}}}},
        {"contentBlockDelta": {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "sig"}}}},
        {"contentBlockStop": {"contentBlockIndex": 0}},
    ]
    for index, tool_use_id in ((1, "tooluse_kept"), (2, "tooluse_dropped")):
        events += [
            {"contentBlockStart": {"contentBlockIndex": index, "start": {
                "toolUse": {"toolUseId": tool_use_id, "name": "terminal"}}}},
            {"contentBlockDelta": {"contentBlockIndex": index, "delta": {
                "toolUse": {"input": '{"command": "bash run.sh", "timeout": 900}'}}}},
            {"contentBlockStop": {"contentBlockIndex": index}},
        ]
    return {"stream": [*events, {"messageStop": {"stopReason": "tool_use"}}]}


@pytest.mark.parametrize("replay", ["live", "reloaded"])
def test_live_duplicate_terminal_call_through_turn_order(replay, tmp_path):
    """The t_6e34693f crash: ``turn_tool_round`` expands local batches, then drops the duplicate with
    ``_deduplicate_tool_calls(_cap_delegate_task_calls(...))``, then stages the row. The next ConverseStream
    request must carry one toolUse, live (the ``bedrock_content_blocks`` sidecar) and after a reload from
    state.db (no sidecar: ``reasoning_details`` plus the persisted ``tool_calls``)."""
    from agent.agent_init import _CALLBACK_PARAMS
    from agent.bedrock_adapter import convert_messages_to_converse, normalize_converse_stream_events
    from agent.message_sanitization import normalize_provider_tool_call_ids
    from agent.session_persistence import _db_flush_row
    from agent.tool_call_batches import expand_local_tool_batches
    from agent.transports.bedrock import BedrockTransport
    from agent.turn_tool_round import stage_tool_call_message
    from hermes_state import SessionDB
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent.provider, agent.model, agent.session_id, agent.tools = "bedrock", "global.anthropic.claude-opus-5-5", "s1", []
    agent.log_prefix, agent.quiet_mode, agent.verbose_logging = "", True, False
    agent._mute_post_response, agent._base_url, agent.api_mode = False, "", "bedrock_converse"
    agent._current_streamed_assistant_text, agent._stream_callback = "", None
    for name in _CALLBACK_PARAMS:
        setattr(agent, name, None)

    assistant_message = BedrockTransport().normalize_response(
        normalize_converse_stream_events(_opus_duplicate_terminal_stream()))
    # turn_tool_round's order, verbatim.
    assistant_message.tool_calls = expand_local_tool_batches(
        assistant_message.tool_calls, provider_data=getattr(assistant_message, "provider_data", None))
    assistant_message.tool_calls = agent._deduplicate_tool_calls(
        agent._cap_delegate_task_calls(assistant_message.tool_calls))
    normalize_provider_tool_call_ids(assistant_message.tool_calls)
    messages = [{"role": "user", "content": "go"}]
    staged, _ = stage_tool_call_message(
        agent, assistant_message=assistant_message, finish_reason="tool_calls", messages=messages)
    assert [tc["id"] for tc in staged["tool_calls"]] == ["tooluse_kept"]
    messages += [staged, {"role": "tool", "tool_call_id": "tooluse_kept", "content": "ok"}]

    if replay == "reloaded":
        db = SessionDB(db_path=tmp_path / "state.db")
        try:
            db.create_session("s1", source="cli")
            db.append_messages_batch("s1", [_db_flush_row(agent, m, False) for m in messages])
            messages = db.get_messages_as_conversation("s1")
        finally:
            db.close()
        assert "bedrock_content_blocks" not in messages[1]

    _system, converse = convert_messages_to_converse(AIAgent._sanitize_api_messages(messages), model=agent.model)
    assert [next(iter(block)) for block in converse[1]["content"]] == ["reasoningContent", "toolUse"]
    assert converse[1]["content"][0]["reasoningContent"]["reasoningText"] == {"text": "run it", "signature": "sig"}
    tool_uses = [b["toolUse"]["toolUseId"] for m in converse for b in m["content"] if "toolUse" in b]
    tool_results = [b["toolResult"]["toolUseId"] for m in converse for b in m["content"] if "toolResult" in b]
    assert tool_uses == tool_results == ["tooluse_kept"]


def test_tool_call_missing_from_sidecar_is_still_sent():
    """The inverse of the case above: the sidecar lost a toolUse that ``tool_calls`` still holds. Its
    toolResult is sent regardless, so the toolUse must be too, or Converse rejects the turn with
    "number of toolResult blocks exceeds the number of toolUse blocks"."""
    from agent.bedrock_adapter import convert_messages_to_converse

    calls = [{"id": i, "type": "function", "function": {"name": "one", "arguments": '{"n":1}'}} for i in ("t1", "t2")]
    history = {
        "role": "assistant", "content": None, "tool_calls": calls,
        "bedrock_content_blocks": [{"text": "working"}, {"toolUse": {"toolUseId": "t1", "name": "one", "input": {"n": 1}}}],
    }
    results = [{"role": "tool", "tool_call_id": i, "content": "ok"} for i in ("t1", "t2")]
    _system, messages = convert_messages_to_converse([{"role": "user", "content": "go"}, history, *results])

    assert messages[1]["content"][0] == {"text": "working"}
    tool_uses = [b["toolUse"]["toolUseId"] for m in messages for b in m["content"] if "toolUse" in b]
    tool_results = [b["toolResult"]["toolUseId"] for m in messages for b in m["content"] if "toolResult" in b]
    assert tool_uses == tool_results == ["t1", "t2"]
