"""MatchPredict: football match prediction backend."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matchpredict.config import Settings

ALL_INTERFACES = "0.0.0.0"


def _bind_address(settings: Settings, environ: Mapping[str, str] | None = None) -> tuple[str, int]:
    """Host and port to listen on.

    Hosting platforms (Render, Railway, Heroku, Fly...) tell the app its port in ``$PORT`` and
    only reach it on all interfaces. When ``$PORT`` is set, listen there on 0.0.0.0 unless
    ``MP_HOST`` / ``MP_PORT`` say otherwise. Local runs keep the safe 127.0.0.1 default.
    """
    env = os.environ if environ is None else environ
    raw = env.get("PORT", "")
    if not raw.isdigit():
        return settings.host, settings.port
    host = settings.host if "MP_HOST" in env else ALL_INTERFACES
    port = settings.port if "MP_PORT" in env else int(raw)
    return host, port


def main() -> None:
    """Run the API server (``uv run matchpredict``)."""
    import uvicorn

    from matchpredict.config import get_settings

    host, port = _bind_address(get_settings())
    uvicorn.run("matchpredict.api.app:app", host=host, port=port)
