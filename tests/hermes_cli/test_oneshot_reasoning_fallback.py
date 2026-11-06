"""``hermes -z --reasoning X``: the run's explicit level survives a mid-run provider fallback.

``--reasoning`` outranks ``agent.reasoning_overrides`` and the global ``agent.reasoning_effort``; a
fallback activated during the run must keep it, and a run without it still resolves the fallback
model's config.

The one-shot run is a gateway session (upstream 936f3465f8 removed the in-process runner): the parsed
flag becomes a ``session.create`` field (``hermes_cli.gateway_chat._requested_policy``), the session's
policy owns it (``LocalSessionPolicy.reasoning_pinned``) and the turn hands it to the agent. These tests
start from the real ``hermes -z`` argv, so a flag dropped on the way to the session fails here.
"""

import sys

import pytest

from tests.gateway.test_fallback_session_reasoning import _runner, _turn, gateway_home

_CFG = {"agent": {"reasoning_effort": "medium", "reasoning_overrides": {"fb-model": "low"}}}


def _oneshot_create_params(monkeypatch, *flags):
    """The ``session.create`` fields ``hermes -z ping <flags>`` sends (the launcher is stubbed)."""
    import hermes_cli.main as main_mod
    from hermes_cli.gateway_chat import _requested_policy

    seen = []
    monkeypatch.setattr(sys, "argv", ["hermes", "-z", "ping", *flags])
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda _args: None)
    monkeypatch.setattr("hermes_cli.gateway_chat_startup.launch_gateway_chat",
                        lambda args: seen.append(_requested_policy(args)) or 0)
    with pytest.raises(SystemExit):
        main_mod.main()
    assert len(seen) == 1
    return seen[0]


def _oneshot_turn(monkeypatch, tmp_path, *flags):
    from gateway.config import Platform
    from gateway.session import SessionSource
    from gateway.session_policy import build_policy

    params = _oneshot_create_params(monkeypatch, *flags)
    # A local endpoint keeps the turn off the network, and ignore_user_config keeps it off the owner's
    # prefill store, which needs a session authority. Neither touches the reasoning fields.
    policy = build_policy({**params, "cwd": str(tmp_path), "model": "primary-model", "provider": "custom",
                           "base_url": "http://127.0.0.1:1234/v1", "ignore_user_config": True}, _CFG)
    monkeypatch.setattr("gateway.session_policy.policy_for_source", lambda _runner, _source: policy)
    source = SessionSource(platform=Platform.LOCAL, chat_id="local-1", chat_type="dm", user_id="user-1")
    return _turn(_runner(), source)["final_response"]


def test_reasoning_flag_survives_fallback(gateway_home, tmp_path, monkeypatch):
    assert _oneshot_turn(monkeypatch, tmp_path, "--reasoning", "high") == "fb-model:high"


def test_without_the_flag_fallback_resolves_the_fallback_models_config(gateway_home, tmp_path, monkeypatch):
    assert _oneshot_turn(monkeypatch, tmp_path) == "fb-model:low"


def test_an_unparsed_flag_is_refused_rather_than_run_as_the_sessions_level(tmp_path, monkeypatch):
    """The gateway refuses an unknown level at creation, so no run carries it as its own."""
    from gateway.session_policy import build_policy
    from hermes_state_runtime import RuntimeStoreError

    params = _oneshot_create_params(monkeypatch, "--reasoning", "bogus")
    assert params["reasoning"] == "bogus"
    with pytest.raises(RuntimeStoreError) as caught:
        build_policy({**params, "cwd": str(tmp_path)}, _CFG)
    assert (caught.value.field, caught.value.reason) == ("reasoning", "invalid_params")
