"""``providers.bedrock.request_timeout_seconds`` must reach the boto3 clients Hermes builds.

Without it botocore's 60 s default read timeout kills a long Converse stream that is silent while
the model thinks. Real boto3 clients against temp homes with synthetic credentials and instance
metadata disabled; the only network use is a loopback server in this process. Upstream fix: PR
#133700 (zhangchi47).
"""

from __future__ import annotations

import http.server
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("boto3")

from agent import bedrock_adapter as ba
from agent.secret_scope import build_profile_secret_scope, reset_secret_scope, set_secret_scope
from hermes_constants import reset_hermes_home_override, set_hermes_home_override

_BOTOCORE_DEFAULT_TIMEOUT = 60
_GETTERS = {"runtime": ba._get_bedrock_runtime_client, "control": ba._get_bedrock_control_client}


def _write_home(home: Path, name: str, timeout: float | None) -> None:
    home.mkdir(parents=True, exist_ok=True)
    (home / ".env").write_text(
        f"AWS_ACCESS_KEY_ID=AKIA{name * 16}\nAWS_SECRET_ACCESS_KEY=secret-{name}\n", encoding="utf-8")
    body = "" if timeout is None else f"providers:\n  bedrock:\n    request_timeout_seconds: {timeout}\n"
    (home / "config.yaml").write_text(body, encoding="utf-8")


def _under(home: Path, fn):
    home_token = set_hermes_home_override(str(home))
    scope_token = set_secret_scope(build_profile_secret_scope(home))
    try:
        return fn()
    finally:
        reset_secret_scope(scope_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def homes(tmp_path, monkeypatch):
    for var in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
                "AWS_CONFIG_FILE", "AWS_SHARED_CREDENTIALS_FILE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-aws-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-aws-credentials"))
    a = tmp_path / ".hermes"
    b = a / "profiles" / "b"
    monkeypatch.setenv("HERMES_HOME", str(a))
    ba.reset_client_cache()
    yield a, b
    ba.reset_client_cache()


def _build(home: Path, scoped: bool, kind: str):
    if scoped:
        return _under(home, lambda: _GETTERS[kind]("us-east-1"))
    return _GETTERS[kind]("us-east-1")


def _unscoped_env(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "AKIA" + "U" * 16)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret-U")


@pytest.mark.parametrize("kind", ["runtime", "control"])
@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
@pytest.mark.parametrize("timeout,connect", [(900, 60), (25, 25)])
def test_client_timeout_reaches_boto3(homes, monkeypatch, kind, scoped, timeout, connect):
    """B1: both constructor paths (boto3.client, Session.client) carry the configured read timeout."""
    a, _ = homes
    _write_home(a, "A", timeout)
    if not scoped:
        _unscoped_env(monkeypatch)
    client = _build(a, scoped, kind)
    assert client.meta.config.read_timeout == timeout
    assert client.meta.config.connect_timeout == connect
    assert client.meta.region_name == "us-east-1"


@pytest.mark.parametrize("kind", ["runtime", "control"])
@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
@pytest.mark.parametrize("raw", [None, 0, -5])
def test_unconfigured_client_defaults(homes, monkeypatch, kind, scoped, raw):
    """B2: no (or a non-positive) timeout keeps botocore's defaults, and the cache still reuses."""
    a, _ = homes
    _write_home(a, "A", raw)
    if not scoped:
        _unscoped_env(monkeypatch)
    client = _build(a, scoped, kind)
    assert client.meta.config.read_timeout == _BOTOCORE_DEFAULT_TIMEOUT
    assert client.meta.config.connect_timeout == _BOTOCORE_DEFAULT_TIMEOUT
    assert _build(a, scoped, kind) is client


def test_scoped_timeout_isolation(homes):
    """B3: two routed profiles keep their own timeout, cached client and credentials."""
    a, b = homes
    _write_home(a, "A", 900)
    _write_home(b, "B", 300)
    client_a = _under(a, lambda: ba._get_bedrock_runtime_client("us-east-1"))
    client_b = _under(b, lambda: ba._get_bedrock_runtime_client("us-east-1"))
    assert client_a is not client_b
    assert (client_a.meta.config.read_timeout, client_b.meta.config.read_timeout) == (900, 300)
    keys = [c._request_signer._credentials.get_frozen_credentials().access_key for c in (client_a, client_b)]
    assert keys == ["AKIA" + "A" * 16, "AKIA" + "B" * 16]
    assert _under(a, lambda: ba._get_bedrock_runtime_client("us-east-1")) is client_a
    assert _under(b, lambda: ba._get_bedrock_runtime_client("us-east-1")) is client_b


class _SlowModels(http.server.BaseHTTPRequestHandler):
    delay = 2.0

    def do_GET(self):
        time.sleep(self.delay)
        body = b'{"modelSummaries": []}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


@pytest.fixture
def slow_endpoint(monkeypatch):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _SlowModels)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("AWS_ENDPOINT_URL_BEDROCK", f"http://127.0.0.1:{server.server_address[1]}")
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "1")
    yield
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("scoped", [False, True], ids=["unscoped", "scoped"])
def test_configured_read_timeout_governs_the_socket(homes, monkeypatch, slow_endpoint, scoped):
    """B5: a response delayed 2 s times out at 1 s and succeeds with a 10 s read timeout."""
    from botocore.exceptions import ReadTimeoutError

    a, _ = homes
    if not scoped:
        _unscoped_env(monkeypatch)
    _write_home(a, "A", 1)
    with pytest.raises(ReadTimeoutError):
        _build(a, scoped, "control").list_foundation_models()
    ba.reset_client_cache()
    _write_home(a, "A", 10)
    assert _build(a, scoped, "control").list_foundation_models()["modelSummaries"] == []
