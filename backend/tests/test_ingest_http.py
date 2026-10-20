import ssl
import subprocess

import httpx
import pytest

from matchpredict.ingest import http as mhttp
from matchpredict.ingest.http import (
    FetchError,
    NotFoundError,
    fetch_bytes,
    fetch_with_curl,
)

URL = "https://www.football-data.co.uk/fixtures.csv"
NO_SLEEP = lambda _s: None


def _client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler), follow_redirects=True)


def _forbid_curl(monkeypatch):
    def fail(*_a, **_k):
        raise AssertionError("curl must not be called")

    monkeypatch.setattr(mhttp.subprocess, "run", fail)


def test_httpx_success_returns_content(monkeypatch):
    _forbid_curl(monkeypatch)
    client = _client(lambda req: httpx.Response(200, content=b"\xef\xbb\xbfDiv\n"))
    assert fetch_bytes(URL, client=client, sleep=NO_SLEEP) == b"\xef\xbb\xbfDiv\n"


def test_httpx_retries_transient_errors_then_succeeds(monkeypatch):
    _forbid_curl(monkeypatch)
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        if attempts["n"] < 3:
            return httpx.Response(503)
        return httpx.Response(200, content=b"ok")

    assert fetch_bytes(URL, client=_client(handler), sleep=NO_SLEEP) == b"ok"
    assert attempts["n"] == 3


def test_not_found_raises_without_fallback(monkeypatch):
    _forbid_curl(monkeypatch)
    with pytest.raises(NotFoundError):
        fetch_bytes(URL, client=_client(lambda r: httpx.Response(404)), sleep=NO_SLEEP)


def test_tls_failure_falls_back_to_curl(monkeypatch):
    def handler(req):
        raise httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]", request=req)

    seen = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return subprocess.CompletedProcess(args, 0, stdout=b"from-curl", stderr=b"")

    monkeypatch.setattr(mhttp.shutil, "which", lambda _n: "/usr/bin/curl")
    monkeypatch.setattr(mhttp.subprocess, "run", fake_run)
    assert fetch_bytes(URL, client=_client(handler), retries=2, sleep=NO_SLEEP) == b"from-curl"
    assert seen["args"][0] == "/usr/bin/curl"
    assert "-sSL" in seen["args"] and seen["args"][-1] == URL
    assert "shell" not in seen["kwargs"]


def test_ssl_oserror_also_falls_back(monkeypatch):
    def handler(req):
        raise ssl.SSLEOFError("EOF occurred in violation of protocol")

    monkeypatch.setattr(mhttp, "fetch_with_curl", lambda url, timeout: b"curl")
    assert fetch_bytes(URL, client=_client(handler), retries=1, sleep=NO_SLEEP) == b"curl"


def test_fallback_disabled_raises(monkeypatch):
    _forbid_curl(monkeypatch)
    with pytest.raises(FetchError):
        fetch_bytes(
            URL,
            client=_client(lambda r: httpx.Response(500)),
            retries=1,
            sleep=NO_SLEEP,
            use_curl_fallback=False,
        )


def test_client_error_is_not_retried(monkeypatch):
    attempts = {"n": 0}

    def handler(req):
        attempts["n"] += 1
        return httpx.Response(403)

    monkeypatch.setattr(mhttp, "fetch_with_curl", lambda url, timeout: b"curl")
    assert fetch_bytes(URL, client=_client(handler), sleep=NO_SLEEP) == b"curl"
    assert attempts["n"] == 1


def test_curl_failure_raises_fetch_error(monkeypatch):
    monkeypatch.setattr(mhttp.shutil, "which", lambda _n: "curl")
    monkeypatch.setattr(
        mhttp.subprocess,
        "run",
        lambda args, **k: subprocess.CompletedProcess(args, 35, stdout=b"", stderr=b"ssl error"),
    )
    with pytest.raises(FetchError, match="curl exit 35"):
        fetch_with_curl(URL)


def test_curl_missing_or_timeout(monkeypatch):
    monkeypatch.setattr(mhttp.shutil, "which", lambda _n: None)
    with pytest.raises(FetchError, match="not available"):
        fetch_with_curl(URL)

    def timeout(args, **k):
        raise subprocess.TimeoutExpired(args, 1)

    monkeypatch.setattr(mhttp.shutil, "which", lambda _n: "curl")
    monkeypatch.setattr(mhttp.subprocess, "run", timeout)
    with pytest.raises(FetchError, match="timed out"):
        fetch_with_curl(URL)


def test_default_client_is_created_and_closed(monkeypatch):
    created = []
    real = httpx.Client

    def factory(**kwargs):
        assert kwargs["follow_redirects"] is True
        assert "User-Agent" in kwargs["headers"]
        client = real(transport=httpx.MockTransport(lambda r: httpx.Response(200, content=b"x")))
        created.append(client)
        return client

    monkeypatch.setattr(mhttp.httpx, "Client", factory)
    assert fetch_bytes(URL, sleep=NO_SLEEP) == b"x"
    assert created[0].is_closed
