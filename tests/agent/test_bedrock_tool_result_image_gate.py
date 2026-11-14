"""Bedrock Converse tool results carry image blocks only for models that accept them.

Claude and Nova read an ``image`` block inside ``toolResult``. OpenAI GPT models on Bedrock refuse the
whole request with a ValidationException, so every other model id, and a missing one, must keep the
single ``json.dumps`` text block that worked before image blocks existed.
"""

import json

import pytest

from agent.bedrock_adapter import build_converse_kwargs, convert_messages_to_converse

_IMG = "data:image/png;base64,iVBORw0KGgo="
_PNG_BYTES = b"\x89PNG\r\n\x1a\n"
_PARTS = [{"type": "text", "text": "a cat"}, {"type": "image_url", "image_url": {"url": _IMG}}]
_ENVELOPE = {"_multimodal": True, "content": _PARTS, "text_summary": "a cat"}
_SHAPES = pytest.mark.parametrize("content", [_PARTS, _ENVELOPE], ids=["list", "envelope"])

_IMAGE_MODELS = [
    "global.anthropic.claude-fable-5-2",
    "us.anthropic.claude-fable-5-2",
    "eu.anthropic.claude-fable-5-2",
    "global.anthropic.claude-haiku-5-5",
    "us.amazon.nova-pro-v1:0",
    "US.ANTHROPIC.CLAUDE-FABLE-5-2",
    "US.AMAZON.NOVA-PRO-V1:0",
]
_TEXT_MODELS = [
    "us.openai.gpt-6.1-sol",
    "global.openai.gpt-6-astra",
    "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcd1234efgh",
    None,
]


def _messages(content):
    return [
        {"role": "user", "content": "look at the screen"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "t1", "type": "function", "function": {"name": "screenshot", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": content},
    ]


def _tool_result_content(converse_messages):
    results = [b["toolResult"] for turn in converse_messages for b in turn["content"] if "toolResult" in b]
    assert len(results) == 1
    return results[0]["content"]


def _block_kinds(converse_messages):
    """Every key of every content block in the request, toolResult contents included."""
    kinds = []
    for turn in converse_messages:
        for block in turn["content"]:
            kinds.extend(block)
            for inner in block.get("toolResult", {}).get("content", []):
                kinds.extend(inner)
    return kinds


@_SHAPES
@pytest.mark.parametrize("model", _IMAGE_MODELS)
def test_image_capable_model_gets_image_block_in_tool_result(model, content):
    _, msgs = convert_messages_to_converse(_messages(content), model=model)
    blocks = _tool_result_content(msgs)
    assert blocks[0] == {"text": "a cat"}
    assert blocks[1] == {"image": {"format": "png", "source": {"bytes": _PNG_BYTES}}}
    assert len(blocks) == 2


@_SHAPES
@pytest.mark.parametrize("model", _TEXT_MODELS)
def test_other_models_keep_tool_result_as_one_json_text_block(model, content):
    _, msgs = convert_messages_to_converse(_messages(content), model=model)
    assert _tool_result_content(msgs) == [{"text": json.dumps(content)}]


def test_no_model_argument_keeps_tool_result_as_one_json_text_block():
    _, msgs = convert_messages_to_converse(_messages(_PARTS))
    assert _tool_result_content(msgs) == [{"text": json.dumps(_PARTS)}]


@_SHAPES
@pytest.mark.parametrize("model", ["us.openai.gpt-6.1-sol", "global.openai.gpt-6-astra"])
def test_gpt_request_has_no_image_block(model, content):
    kwargs = build_converse_kwargs(model=model, messages=_messages(content))
    assert kwargs["modelId"] == model
    assert "image" not in _block_kinds(kwargs["messages"])
    assert _tool_result_content(kwargs["messages"]) == [{"text": json.dumps(content)}]


@_SHAPES
def test_claude_request_carries_the_image_block(content):
    kwargs = build_converse_kwargs(model="global.anthropic.claude-fable-5-2", messages=_messages(content))
    assert _tool_result_content(kwargs["messages"])[1] == {
        "image": {"format": "png", "source": {"bytes": _PNG_BYTES}}}


@pytest.mark.parametrize("content", ["", "  \n"], ids=["empty", "blank"])
@pytest.mark.parametrize("model", _TEXT_MODELS)
def test_other_models_never_send_a_blank_tool_result_text(model, content):
    """Converse rejects a blank text block, so an empty tool result still goes out as visible text."""
    _, msgs = convert_messages_to_converse(_messages(content), model=model)
    blocks = _tool_result_content(msgs)
    assert len(blocks) == 1
    assert blocks[0]["text"].strip()
