"""Shared async HTTP client with retry/backoff for all sources."""

from __future__ import annotations

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

USER_AGENT = "DailyAIBot - AI news digest"

DEFAULT_TIMEOUT = 15
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(headers={"User-Agent": USER_AGENT})

    return _client


async def aclose() -> None:
    """Close and reset the shared client so the next run rebinds to its own loop."""
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS_CODES

    return isinstance(exc, httpx.TransportError)


@retry(
    retry=retry_if_exception(_is_retryable),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5),
    reraise=True,
)
async def get(url: str, *, timeout: int = DEFAULT_TIMEOUT, **kwargs) -> httpx.Response:
    """GET via the shared client; retries transport errors and 429/5xx."""
    r = await _get_client().get(url, timeout=timeout, **kwargs)
    r.raise_for_status()

    return r
