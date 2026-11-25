"""Continuation contracts for progress reports after tool work."""
from types import SimpleNamespace

import pytest

from agent.agent_runtime_helpers import looks_like_degenerate_final, tool_results_this_turn, trailing_continue_intent
from agent.context_compressor import ContextCompressor
from agent.conversation_loop import _CODEX_ACK_CONTINUATION_NUDGE
from agent.turn_final_continuation import _NEXT_STEP_NUDGE, announced_next_step_after_work, choose_continuation
from tests.agent import test_degenerate_final_recovery as recovery

loop_agent = recovery.loop_agent

FIRE = [
    'Next I cut the coder card for the build from the frozen spec.',
    'No card is out yet and nothing is committed. Next I will write the hook trim contract and cut its coder card, then write the spec for the second component. The prompt hook is approved but not yet a plan step.',
    'The last fold is paused at the point you interrupted. The second review said to freeze after its listed edits, and I will pick it up from there.',
    'The deb-check spec is next, starting with its design review.',
    'The findings are in the probe folder. The spec is next, followed by its review, and no build card is cut before that.',
    "2. The next change is the sandbox checks skill, and I'll draft it for you before cutting any card.",
    'Next I will write the contract and cut the coder card for the researcher change, which you already chose.',
    'When the card reports I check its result. Meanwhile I will draft the eight asks of Step 6.',
]
NO_FIRE = [
    'All 41 tests pass and the change is committed as abc1234.',
    'Next I will cut the coder card. Should it include the docs?',
    'When the card reports I will read its result and tell you.',
    "Once the build finishes, I'll run the suite.",
    'Tell me which option you want and I will apply it.',
    'Say "pop up" for the form.',
    'If the report shows a shape neither guard catches, I will bring you a source fix.',
    'The reviewer card is next, and it needs your approval first.',
    'Next I will run the suite.\n\nThe suite passed, 12 of 12.',
    'I will not cut the card before the review.', '', '   ',
    'Both cards notify me when they finish.', 'The next read of the prediction is on Nov 29.',
]


@pytest.mark.parametrize('text', FIRE, ids=[f'F{i}' for i in range(1, 9)])
def test_fire(text):
    assert text.strip() and '\n\n' not in text
    assert announced_next_step_after_work(text)


@pytest.mark.parametrize('text', FIRE, ids=[f'F{i}' for i in range(1, 9)])
def test_long_prefix(text):
    report = 'x' * 2000 + '\n\n' + text
    assert len(report) > 2000 and report.split('\n\n')[-1] == text
    assert announced_next_step_after_work(report)


@pytest.mark.parametrize('text', NO_FIRE, ids=[f'G{i}' for i in range(1, 12)] + ['G11-spaces', 'G12', 'G13'])
def test_no_fire(text):
    assert text not in FIRE
    assert not announced_next_step_after_work(text)


def test_window():
    text = 'Next I cut the card. ' + 'x' * 600
    assert text[-600:] == 'x' * 600
    assert not announced_next_step_after_work(text)


def test_wait_sentence():
    waiting = NO_FIRE[3]
    assert waiting.startswith('Once') and waiting.endswith('.')
    assert not announced_next_step_after_work(waiting)
    report = waiting + ' Meanwhile I will draft the notes.'
    assert report.startswith(waiting + ' ')
    assert announced_next_step_after_work(report)


@pytest.mark.parametrize('text,expected', [
    ('Now I have the full picture.', False), ('For now I recommend option B.', False),
    ('Now I will run the suite.', True), ("Now I'll run the suite.", True), ('I will now run the suite.', True),
])
def test_now_marker(text, expected):
    assert announced_next_step_after_work(text) is expected


@pytest.mark.parametrize('text', [
    'Next I wait for the review card, then file the result.',
    'Next I will wait for the review card, then file the result.',
    'I will wait for the next report.',
])
def test_waiting_verb(text):
    acting = text.replace('wait for', 'read')
    assert acting != text and announced_next_step_after_work(acting)
    assert not announced_next_step_after_work(text)


@pytest.mark.parametrize('text', [
    'The patch is staged, so say "go" and I\'ll run the deploy.',
    "Send me the token and I'll run the deploy.",
    'Reply with the option you want and I will apply it.',
])
def test_ask_inside_sentence(text):
    commitment = 'I' + text.rsplit(' and I', 1)[1]
    assert commitment != text and announced_next_step_after_work(commitment)
    assert not announced_next_step_after_work(text)


def choose(text, gate='open'):
    rows = [{'role': 'user', 'content': 'Build the workbook.'}, {'role': 'tool', 'content': 'ok'}]
    if gate == 'no-result':
        rows.append({'role': 'user', 'content': _NEXT_STEP_NUDGE})
    assert (rows[-1]['role'] == 'tool') == (gate != 'no-result')
    assert tool_results_this_turn(rows) == int(gate != 'no-result')
    agent = SimpleNamespace(_strip_think_blocks=str.strip, _stall_guards=gate != 'disabled',
                            valid_tool_names=set() if gate == 'no-tools' else {'read_file'},
                            _intent_ack_continuation=True, _looks_like_codex_intermediate_ack=lambda **kw: False)
    return choose_continuation(agent, final_response=text, messages=rows, user_message=rows[0]['content'],
                               promoted=gate == 'promoted', ack_count=2 if gate == 'budget' else 0)


@pytest.mark.parametrize('gate', ['open', 'no-result', 'no-tools', 'promoted', 'budget', 'disabled'])
def test_gates(gate):
    text = FIRE[3]
    assert announced_next_step_after_work(text) and not trailing_continue_intent(text)
    result = choose(text, gate)
    if gate == 'open':
        assert result is not None
        assert (result.kind, result.nudge, result.stall_text, result.tool_rows) == ('next_step', _NEXT_STEP_NUDGE, text, 1)
    else:
        assert result is None


def test_precedence():
    text = 'Next, I will draft it.'
    assert trailing_continue_intent(text) and announced_next_step_after_work(text)
    assert choose(text).kind == 'stall'
    fragment = '?warming'
    assert looks_like_degenerate_final(fragment) and not announced_next_step_after_work(fragment)
    assert choose(fragment).kind == 'degenerate'
    overlap = ',I will draft'
    assert looks_like_degenerate_final(overlap) and announced_next_step_after_work(overlap)
    assert choose(overlap).kind == 'next_step'


def test_synthetic_row():
    row = {'role': 'user', 'content': _NEXT_STEP_NUDGE}
    assert row['content'].startswith('[System: ')
    assert ContextCompressor._is_synthetic_compression_user_turn(row)


def test_two_replies(loop_agent, caplog):
    first, second = FIRE[3], NO_FIRE[3]
    loop_agent.valid_tool_names = {'web_search'}
    assert announced_next_step_after_work(first) and not announced_next_step_after_work(second)
    with caplog.at_level('INFO'):
        result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(first), recovery._final(second)])
    rows = result['messages']
    interim = next(i for i, m in enumerate(rows) if m.get('content') == first)
    assert tool_results_this_turn(rows[:interim]) == 1
    joined = first + '\n\n' + second
    assert [(m['role'], m['content']) for m in rows[interim:]] == [
        ('assistant', first), ('user', _NEXT_STEP_NUDGE), ('assistant', joined)]
    assert rows[-1]['api_content'] == second
    assert tool_results_this_turn(rows) == 0
    assert result['final_response'] == joined
    assert loop_agent.client.chat.completions.create.call_count == 3
    assert any(r.levelname == 'INFO' and r.message.startswith('Next-step guard:') for r in caplog.records)


def test_report_delivery_and_log(loop_agent, caplog):
    first, second = FIRE[3], NO_FIRE[3]
    loop_agent.valid_tool_names = {'web_search'}
    seen = []
    loop_agent.interim_assistant_callback = lambda text, **kw: seen.append(text)
    with caplog.at_level('INFO'):
        result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(first), recovery._final(second)])
    assert seen.count(first) == 1 and second not in seen
    assert result['final_response'] == second
    assert result['messages'][-1]['content'] == second
    assert 'api_content' not in result['messages'][-1]
    assert [m['role'] for m in result['messages']] == ['user', 'assistant', 'tool', 'assistant', 'user', 'assistant']
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith('Next-step guard:')]
    assert lines == [f'Next-step guard: {len(first)}-char reply named a next step after 1 tool result(s) — re-prompting (1/2)']


@pytest.mark.parametrize('text', [
    'Now I will ship it.', 'I will now ship it.', 'I will ship it next.', "Next I'll ship it.", 'Next, I will ship it.',
])
def test_next_markers(text):
    assert announced_next_step_after_work(text)


@pytest.mark.parametrize('word', ['when', 'once', 'after', 'until', 'as soon as', 'if', 'unless'])
def test_wait_words(word):
    assert announced_next_step_after_work('I will run the suite.')
    assert not announced_next_step_after_work(f'I will run the suite {word} the build ends.')


@pytest.mark.parametrize('joint', [', and ', ', but ', '; '])
def test_coordinate_clauses(joint):
    assert not announced_next_step_after_work('The card reports when it ends and I will draft the notes.')
    assert announced_next_step_after_work(f'The card reports when it ends{joint}I will draft the notes.')


@pytest.mark.parametrize('ask', [
    'Tell me', 'Say', 'Let me know', 'Confirm', 'Approve', 'Pick', 'Choose',
    'It is your call on', 'I need your word on', 'It needs your approval on', 'I need your answer on',
    'Here you decide', 'I am waiting on you for',
])
def test_user_asks(ask):
    assert announced_next_step_after_work('I will draft the notes.')
    assert not announced_next_step_after_work(f'{ask} which one. I will draft the notes.')


def test_resent_reply(loop_agent, caplog):
    report = FIRE[3]
    resent = report.replace(' is next', '\n  is next') + ' It waits on the review card.'
    loop_agent.valid_tool_names = {'web_search'}
    assert announced_next_step_after_work(resent) and not trailing_continue_intent(resent)
    with caplog.at_level('INFO'):
        result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(report), recovery._final(resent)])
    assert result['final_response'] == resent
    assert [m['content'] for m in result['messages'] if m['role'] == 'user'][1:] == [_NEXT_STEP_NUDGE]
    assert loop_agent.client.chat.completions.create.call_count == 3
    assert sum(r.getMessage().startswith('Next-step guard:') for r in caplog.records) == 1


def test_tool_call_after_nudge(loop_agent):
    loop_agent.valid_tool_names = {'web_search'}
    reply = 'The workbook has three sheets.'
    result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(FIRE[3]),
                                        recovery._tool_round('call_2'), recovery._final(reply)])
    assert recovery._user_rows_sent(loop_agent, 2)[-1] == (
        '[System: Your last message named a next step and made no tool call. If that step is yours to do now, '
        'and it waits on no answer from the user and on no result that has not arrived, do it now with tool calls. '
        'Otherwise reply with one sentence that says what the step waits on, or that the work is finished. '
        'Do not repeat your last message. It is kept, and your sentence is added after it.]')
    assert result['final_response'] == result['messages'][-1]['content'] == reply
    assert 'api_content' not in result['messages'][-1]


def test_second_nudge(loop_agent):
    loop_agent.valid_tool_names = {'web_search'}
    report, reply = FIRE[3], 'The workbook is finished.'
    result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(report),
                                        recovery._final('Next, I will draft it.'), recovery._final(reply)])
    assert recovery._user_rows_sent(loop_agent, 3)[-2:] == [_NEXT_STEP_NUDGE, _CODEX_ACK_CONTINUATION_NUDGE]
    assert result['final_response'] == result['messages'][-1]['content'] == report + '\n\n' + reply
    assert result['messages'][-1]['api_content'] == reply


def test_prior_turn_nudge_does_not_join(loop_agent):
    history = [{'role': 'user', 'content': 'Inspect the workbook.'},
               {'role': 'assistant', 'content': 'The workbook contains two sheets.'},
               {'role': 'user', 'content': _NEXT_STEP_NUDGE},
               {'role': 'assistant', 'content': 'The inspection is finished.'}]
    reply = 'The second sheet contains totals.'
    loop_agent.client.chat.completions.create.side_effect = [recovery._final(reply)]
    result = loop_agent.run_conversation('What does the second sheet contain?', conversation_history=history)
    assert result['final_response'] == result['messages'][-1]['content'] == reply
    assert 'api_content' not in result['messages'][-1]


def test_disabled_guard_keeps_plain_report(loop_agent):
    loop_agent.valid_tool_names = {'web_search'}
    loop_agent._stall_guards = False
    result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(FIRE[3])])
    assert loop_agent.client.chat.completions.create.call_count == 2
    assert result['final_response'] == result['messages'][-1]['content'] == FIRE[3]
    assert 'api_content' not in result['messages'][-1]


def test_transform_receives_joined_report(loop_agent, monkeypatch):
    loop_agent.valid_tool_names = {'web_search'}
    report, reply = FIRE[3], 'The workbook is finished.'
    seen = []

    def hook(name, logger, **payload):
        if name == 'transform_llm_output':
            seen.append(payload['response_text'])
            return ['Edited workbook summary.']
        return []

    monkeypatch.setattr('agent.turn_finalizer._invoke_hook_safely', hook)
    result = recovery._run(loop_agent, [recovery._tool_round('call_1'), recovery._final(report), recovery._final(reply)])
    assert seen == [report + '\n\n' + reply]
    assert result['final_response'] == result['messages'][-1]['content'] == 'Edited workbook summary.'
    assert 'api_content' not in result['messages'][-1]


# Two replies that ended a turn with the work still undone. Each defers a possible later decision
# ("any that are your call"), and the second reports progress in the present tense.
LATER_ASK = [
    'Next I fold the findings into draft 3, bring you any that turn out to be your call, and send one follow-up '
    'review. The fold is a long read, and this session is close to its compaction point, so I saved the state in '
    'the workbench Trail first. I am starting the fold now.',
    'Draft 2 is kept unchanged beside it. I am continuing with the remaining findings and will come back with any '
    'that are your call.',
]


@pytest.mark.parametrize('text', LATER_ASK, ids=['next-fold', 'continuing'])
def test_later_ask_does_not_veto(text):
    assert 'your call' in text
    assert announced_next_step_after_work(text)


@pytest.mark.parametrize('text,expected', [
    ('I am starting the fold now.', True), ("I'm continuing with the remaining findings.", True),
    ('I am starting to think the spec is wrong.', False), ('I am continuing to see the same failure.', False),
    ('I am not continuing the fold.', False),
    ('The second finding is your call. I am continuing with the fold.', False),
    ('I am continuing with the fold, and the rename is your call.', False),
])
def test_present_progress(text, expected):
    assert announced_next_step_after_work(text) is expected
