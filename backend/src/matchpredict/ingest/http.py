"""HTTP fetching with retries and a ``curl`` fallback.

Python's ssl stack fails the TLS handshake with football-data.co.uk on some
machines while ``curl`` succeeds, so every download goes through httpx first and
falls back to ``curl -sSL`` (argument list, never a shell).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

USER_AGENT = "MatchPredict/0.1 (+https://github.com/; football prediction app)"
DEFAULT_TIMEOUT = 30.0
DEFAULT_RETRIES = 3
BACKOFF_SECONDS = 1.5

Fetcher = Callable[[str], bytes]


class FetchError(RuntimeError):
    """Raised when a resource cannot be downloaded."""


class NotFoundError(FetchError):
    """The server answered 404 (no fallback is attempted)."""


def _fetch_httpx(
    url: str,
    client: httpx.Client,
    retries: int,
    sleep: Callable[[float], None],
    headers: dict[str, str] | None = None,
) -> bytes:
    last: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.get(url, headers=headers)
            if resp.status_code == 404:
                raise NotFoundError(f"404 for {url}")
            resp.raise_for_status()
            return resp.content
        except NotFoundError:
            raise
        except httpx.HTTPStatusError as exc:
            last = exc
            if 400 <= exc.response.status_code < 500 and exc.response.status_code != 429:
                raise FetchError(f"HTTP {exc.response.status_code} for {url}") from exc
        except (httpx.HTTPError, OSError) as exc:  # transport / TLS errors
            last = exc
        log.warning("fetch attempt %d/%d failed for %s: %s", attempt, retries, url, last)
        if attempt < retries:
            sleep(BACKOFF_SECONDS * attempt)
    raise FetchError(f"httpx failed for {url}: {last}") from last


def fetch_with_curl(url: str, timeout: float = DEFAULT_TIMEOUT) -> bytes:
    curl = shutil.which("curl")
    if curl is None:
        raise FetchError("curl is not available for the fallback")
    args = [curl, "-sSL", "--fail", "--max-time", str(int(timeout)), "-A", USER_AGENT, url]
    try:
        proc = subprocess.run(args, capture_output=True, timeout=timeout + 5, check=False)
    except subprocess.TimeoutExpired as exc:
        raise FetchError(f"curl timed out for {url}") from exc
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise FetchError(f"curl exit {proc.returncode} for {url}: {msg}")
    return proc.stdout


def fetch_bytes(
    url: str,
    *,
    client: httpx.Client | None = None,
    retries: int = DEFAULT_RETRIES,
    timeout: float = DEFAULT_TIMEOUT,
    sleep: Callable[[float], None] = time.sleep,
    use_curl_fallback: bool = True,
) -> bytes:
    """Download ``url``; httpx with retries, then curl. Raises FetchError."""
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
    try:
        return _fetch_httpx(url, client, retries, sleep)
    except NotFoundError:
        raise
    except FetchError as exc:
        if not use_curl_fallback:
            raise
        log.warning("httpx failed (%s); falling back to curl", exc)
        return fetch_with_curl(url, timeout)
    finally:
        if owns_client:
            client.close()
