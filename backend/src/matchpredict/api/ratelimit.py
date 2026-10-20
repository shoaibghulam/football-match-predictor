"""Per-client-IP rate limiting (moving window, in-memory) using the ``limits`` library."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import Request, Response
from limits import parse_many
from limits.storage import MemoryStorage
from limits.strategies import MovingWindowRateLimiter

from matchpredict.api.errors import envelope


class RateLimiter:
    def __init__(self, spec: str, exempt_paths: frozenset[str] = frozenset()) -> None:
        self.limits = parse_many(spec)
        self.exempt_paths = exempt_paths
        self._limiter = MovingWindowRateLimiter(MemoryStorage())

    def allow(self, key: str) -> bool:
        return all(self._limiter.hit(limit, key) for limit in self.limits)

    async def __call__(self, request: Request, call_next: Callable) -> Response:
        if request.method == "OPTIONS" or request.url.path in self.exempt_paths:
            return await call_next(request)
        key = request.client.host if request.client else "anonymous"
        if not self.allow(key):
            return envelope(429, "RATE_LIMITED", f"rate limit exceeded ({self.limits[0]})")
        return await call_next(request)
