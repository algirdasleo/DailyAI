from datetime import datetime, timedelta, timezone

import httpx
import pytest

from dailyai.models import SourceCategory
from dailyai.services.ingestion.sources import http
from dailyai.services.ingestion.sources.scrape_source import ScrapeSource

NOW = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=24)


def _fresh_iso(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).isoformat()


def _patch_html(monkeypatch, html: str) -> None:
    request = httpx.Request("GET", "https://example.com/news")
    response = httpx.Response(200, request=request, text=html)

    async def fake_get(url, **kwargs):
        return response

    monkeypatch.setattr(http, "get", fake_get)


def _source(
    *,
    name: str = "Example News",
    url: str = "https://example.com/news",
    item: str = "article.card",
    title: str = "h2",
    link: str | None = "a",
    date: str = "time",
    authority: float = 0.6,
    ai_relevant: str | None = "always",
) -> ScrapeSource:
    return ScrapeSource(
        name=name,
        url=url,
        item=item,
        title=title,
        link=link,
        date=date,
        authority=authority,
        ai_relevant=ai_relevant,
    )


@pytest.mark.asyncio
async def test_extracts_item_within_window(monkeypatch):
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>New AI model released</h2>
          <a href="https://example.com/news/ai-model">read</a>
          <time datetime="{_fresh_iso(2)}">2h ago</time>
        </article>
        """,
    )
    articles = await _source().fetch(SINCE)

    assert len(articles) == 1
    a = articles[0]
    assert a.title == "New AI model released"
    assert a.url == "https://example.com/news/ai-model"
    assert a.source == "Example News"
    assert a.authority == 0.6
    assert a.source_type == SourceCategory.SCRAPE
    assert a.published == NOW - timedelta(hours=2)


@pytest.mark.asyncio
async def test_drops_items_older_than_since(monkeypatch):
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>Fresh</h2><a href="/a">x</a>
          <time datetime="{_fresh_iso(5)}">x</time>
        </article>
        <article class="card">
          <h2>Stale</h2><a href="/b">x</a>
          <time datetime="{_fresh_iso(30)}">x</time>
        </article>
        """,
    )
    articles = await _source().fetch(SINCE)

    assert [a.title for a in articles] == ["Fresh"]


@pytest.mark.asyncio
async def test_drops_items_with_no_parseable_date(monkeypatch):
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>No date element</h2><a href="/a">x</a>
        </article>
        <article class="card">
          <h2>Unparseable date</h2><a href="/b">x</a>
          <time datetime="yesterday">x</time>
        </article>
        <article class="card">
          <h2>Good</h2><a href="/c">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        """,
    )
    articles = await _source().fetch(SINCE)

    assert [a.title for a in articles] == ["Good"]


@pytest.mark.asyncio
async def test_dedupes_items_with_the_same_url(monkeypatch):
    # Some sites render each entry twice (e.g. separate desktop/mobile
    # markup) - the same article must only be ingested once.
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>Duplicated post</h2><a href="/dup">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        <article class="card">
          <h2>Duplicated post</h2><a href="/dup">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        """,
    )
    articles = await _source().fetch(SINCE)

    assert len(articles) == 1


@pytest.mark.asyncio
async def test_resolves_relative_urls_against_base(monkeypatch):
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>Relative link</h2><a href="/news/post-1">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        """,
    )
    articles = await _source().fetch(SINCE)

    assert articles[0].url == "https://example.com/news/post-1"


@pytest.mark.asyncio
async def test_relevance_filter_drops_non_ai_titles(monkeypatch):
    _patch_html(
        monkeypatch,
        f"""
        <article class="card">
          <h2>New AI model released</h2><a href="/a">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        <article class="card">
          <h2>Local bakery wins award</h2><a href="/b">x</a>
          <time datetime="{_fresh_iso(1)}">x</time>
        </article>
        """,
    )
    articles = await _source(ai_relevant="filter").fetch(SINCE)

    assert [a.title for a in articles] == ["New AI model released"]


def test_requires_a_date_selector():
    with pytest.raises(ValueError):
        ScrapeSource(
            name="No date",
            url="https://example.com/news",
            item="article.card",
            title="h2",
            link="a",
            date="",
        )
