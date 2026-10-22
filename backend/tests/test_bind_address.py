"""Where the API listens: local default vs hosting platforms that set $PORT (Render, Railway...)."""

from matchpredict import _bind_address
from matchpredict.config import Settings


def _settings(**kw) -> Settings:
    return Settings(**{"host": "127.0.0.1", "port": 8000, **kw})


def test_local_run_keeps_loopback_and_configured_port():
    assert _bind_address(_settings(), environ={}) == ("127.0.0.1", 8000)


def test_platform_port_binds_all_interfaces_on_that_port():
    assert _bind_address(_settings(), environ={"PORT": "10000"}) == ("0.0.0.0", 10000)


def test_explicit_mp_host_wins_over_platform_default():
    env = {"PORT": "10000", "MP_HOST": "127.0.0.1"}
    assert _bind_address(_settings(host="127.0.0.1"), environ=env) == ("127.0.0.1", 10000)


def test_explicit_mp_port_wins_over_platform_port():
    env = {"PORT": "10000", "MP_PORT": "9000"}
    assert _bind_address(_settings(port=9000), environ=env) == ("0.0.0.0", 9000)


def test_invalid_platform_port_is_ignored():
    assert _bind_address(_settings(), environ={"PORT": "abc"}) == ("127.0.0.1", 8000)
