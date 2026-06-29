import asyncio

import httpx

from dailyai.models import Article, SourceCategory
from dailyai.services import fetching
from dailyai.services.fetching import ArticleFetchService, ArticleText
from datetime import datetime, timezone

_HTML = """
<html><head><title>T</title></head><body>
<nav>menu junk</nav>
<article><p>OpenAI shipped a new model today.</p>
<p>It scores higher on benchmarks.</p></article>
<script>var x = 1;</script>
</body></html>
"""


def _article(url="https://ex.com/a") -> Article:
    return Article(
        title="A title",
        url=url,
        source="Ex",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.FEED,
    )


def test_extract_strips_boilerplate_and_keeps_body():
    svc = ArticleFetchService()
    text = svc._extract(_HTML)
    assert "OpenAI shipped a new model today." in text
    assert "var x = 1" not in text
    assert "menu junk" not in text


def test_fetch_one_returns_article_text(monkeypatch):
    async def fake_get(url, *, timeout=15, **kw):
        return httpx.Response(200, text=_HTML, request=httpx.Request("GET", url))

    monkeypatch.setattr(fetching.http, "get", fake_get)
    svc = ArticleFetchService(respect_robots=False)
    out = asyncio.run(svc.fetch_one(_article()))
    assert isinstance(out, ArticleText)
    assert out.url == "https://ex.com/a"
    assert "benchmarks" in out.full_text


def test_fetch_one_fails_soft_on_http_error(monkeypatch):
    async def boom(url, *, timeout=15, **kw):
        raise httpx.ConnectError("down")

    monkeypatch.setattr(fetching.http, "get", boom)
    svc = ArticleFetchService(respect_robots=False)
    assert asyncio.run(svc.fetch_one(_article())) is None
