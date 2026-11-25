"""Choose a bounded continuation for a text-only stop."""
from __future__ import annotations

import re
from typing import Any, NamedTuple, Optional

_NEXT_STEP_NUDGE = (
    "[System: Your last message named a next step and made no tool call. If that step is yours to do now, "
    "and it waits on no answer from the user and on no result that has not arrived, do it now with tool calls. "
    "Otherwise send your last message again exactly as it was, and if a step waits, add one sentence that says "
    "what it waits on.]"
)

_USER_ASK = re.compile(
    r"(?:^|,\s+so\s+)(?:tell me|say|let me know|confirm|approve|pick|choose|reply|send me)\b|"
    r"\b(?:your (?:call|word|approval|answer)|you decide|waiting on you)\b"
)
# "any that turn out to be your call" names a decision that may come later, not one the reply waits on.
_LATER_ASK = re.compile(r"\bany(?:thing)? (?:that|which) [^,;]*?\byour (?:call|word|approval|answer)\b")
_WAIT = re.compile(r"\b(?:when|once|after|until|as soon as|if|unless)\b")
_WORK_VERB = r"apply|bring|build|check|cut|draft|edit|inspect|pick|read|review|run|test|update|write"
_FUTURE = r"\bi(?: will|'ll)\s+"
_NOT_OR_WAIT = r"(?!(?:will\s+|now\s+)?(?:not|wait)\b|won't\b)"
_COMMITMENT = re.compile(
    r"\bnext[,:]?\s+i(?:'ll)?\s+" + _NOT_OR_WAIT + r"\w+|"
    r"\bnow[,:]?\s+i(?:'ll| will)\s+" + _NOT_OR_WAIT + r"\w+|"
    + _FUTURE + rf"(?:{_WORK_VERB})\b|"
    + _FUTURE + _NOT_OR_WAIT + r"(?:now\b|[^,;.!?]*?\bnext\b)|"
    r"\bi(?:'m| am)\s+(?:starting|continuing)\b(?!\s+to\b)|"
    r"\bthe (?:[\w-]+\s+)+is next\b"
)


class Continuation(NamedTuple):
    kind: str
    nudge: str
    stall_text: str
    tool_rows: int


def _commits(clause: str) -> bool:
    if _WAIT.search(clause):
        return False
    return bool(_COMMITMENT.search(clause))


def announced_next_step_after_work(text: str) -> bool:
    tail = text.strip()[-600:]
    paragraph = re.split(r"\n[^\S\n]*\n", tail)[-1].lower().replace("’", "'")
    if paragraph.endswith("?"):
        return False
    paragraph = re.sub(r"(?m)^\s*(?:\d+[.)]|[-*•])\s+", ". ", paragraph)
    paragraph = " ".join(paragraph.split())
    sentences = [s.strip() for s in re.split(r"[.!?]+(?:\s+|$)", paragraph)]
    if any(_USER_ASK.search(_LATER_ASK.sub(" ", s)) for s in sentences):
        return False
    return any(_commits(c) for s in sentences for c in re.split(r",\s+(?:and|but)\s+|;", s))


def choose_continuation(
    agent: Any, *, final_response: Any, promoted: Any, messages: Any, user_message: Any,
    ack_count: int,
) -> Optional[Continuation]:
    from agent.agent_runtime_helpers import (
        intent_ack_continuation_mode, looks_like_degenerate_final, promoted_reasoning_announces_action,
        tool_results_this_turn, trailing_continue_intent,
    )
    from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE, _DEGENERATE_FINAL_NUDGE

    if ack_count >= 2:
        return None
    ack_mode = intent_ack_continuation_mode(agent)
    stall_text = agent._strip_think_blocks(final_response or "")
    guards = bool(getattr(agent, "_stall_guards", True))
    tool_rows = tool_results_this_turn(messages)
    if guards and agent.valid_tool_names and (
        trailing_continue_intent(stall_text)
        or (bool(promoted) and promoted_reasoning_announces_action(stall_text))
    ):
        return Continuation("stall", _CODEX_ACK_CONTINUATION_NUDGE, stall_text, tool_rows)
    if (
        guards and agent.valid_tool_names and tool_rows > 0 and not promoted
        and announced_next_step_after_work(stall_text)
    ):
        return Continuation("next_step", _NEXT_STEP_NUDGE, stall_text, tool_rows)
    if (
        guards and ack_mode != "off" and tool_rows > 0
        and looks_like_degenerate_final(stall_text, user_message=user_message)
    ):
        return Continuation("degenerate", _DEGENERATE_FINAL_NUDGE, stall_text, tool_rows)
    if ack_mode != "off" and agent.valid_tool_names and agent._looks_like_codex_intermediate_ack(
        user_message=user_message, assistant_content=final_response, messages=messages,
        require_workspace=(ack_mode == "codex_only"),
    ):
        return Continuation("ack", _CODEX_ACK_CONTINUATION_NUDGE, stall_text, tool_rows)
    return None
