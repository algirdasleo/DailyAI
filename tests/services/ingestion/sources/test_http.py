import httpx
import pytest

from dailyai.services.ingestion.sources import http


@pytest.fixture(autouse=True)
def _reset_client():
    http._client = None
    yield
    http._client = None


def test_get_client_sets_user_agent_and_is_reused():
    c1 = http._get_client()
    c2 = http._get_client()
    assert c1 is c2
    assert c1.headers["User-Agent"] == http.USER_AGENT


@pytest.mark.asyncio
async def test_get_raises_on_non_retryable_http_error(monkeypatch):
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(404, request=request)

    class FakeClient:
        async def get(self, url, timeout=None, **kwargs):
            return response

    monkeypatch.setattr(http, "_get_client", lambda: FakeClient())

    with pytest.raises(httpx.HTTPStatusError):
        await http.get("https://example.com")


@pytest.mark.asyncio
async def test_get_retries_then_succeeds_on_retryable_status(monkeypatch):
    request = httpx.Request("GET", "https://example.com")
    responses = [
        httpx.Response(503, request=request),
        httpx.Response(200, request=request),
    ]

    class FakeClient:
        async def get(self, url, timeout=None, **kwargs):
            return responses.pop(0)

    monkeypatch.setattr(http, "_get_client", lambda: FakeClient())

    r = await http.get("https://example.com")
    assert r.status_code == 200
    assert responses == []


@pytest.mark.asyncio
async def test_get_passes_default_timeout_and_forwards_kwargs(monkeypatch):
    captured = {}
    request = httpx.Request("GET", "https://example.com")
    response = httpx.Response(200, request=request)

    class FakeClient:
        async def get(self, url, timeout=None, **kwargs):
            captured["url"] = url
            captured["timeout"] = timeout
            captured["kwargs"] = kwargs
            return response

    monkeypatch.setattr(http, "_get_client", lambda: FakeClient())

    await http.get("https://example.com", params={"a": 1})

    assert captured == {
        "url": "https://example.com",
        "timeout": http.DEFAULT_TIMEOUT,
        "kwargs": {"params": {"a": 1}},
    }
