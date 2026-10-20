"""MatchPredict: football match prediction backend."""


def main() -> None:
    """Run the API server (``uv run matchpredict``)."""
    import uvicorn

    from matchpredict.config import get_settings

    settings = get_settings()
    uvicorn.run("matchpredict.api.app:app", host=settings.host, port=settings.port)
