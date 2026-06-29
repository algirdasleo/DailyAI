from datetime import datetime, timedelta, timezone

import pytest

from dailyai.models import SourceCategory
from dailyai.services.ingestion.sources.rss_source import RSSSource

NOW = datetime.now(timezone.utc)
SINCE = NOW - timedelta(hours=24)


def _rfc822(hours_ago: float) -> str:
    return (NOW - timedelta(hours=hours_ago)).strftime("%a, %d %b %Y %H:%M:%S GMT")


def _feed(items: str) -> str:
    return f"""<?xml version="1.0"?>
<rss version="2.0"><channel><title>Feed</title>
{items}
</channel></rss>"""


def _item(title: str, link: str, hours_ago: float, summary: str = "") -> str:
    return f"""<item>
<title>{title}</title>
<link>{link}</link>
<summary>{summary}</summary>
<pubDate>{_rfc822(hours_ago)}</pubDate>
</item>"""


def _source(
    *,
    name: str = "Example Feed",
    url: str = "",
    authority: float = 0.6,
    ai_relevant: str | None = "always",
) -> RSSSource:
    return RSSSource(name=name, url=url, authority=authority, ai_relevant=ai_relevant)


@pytest.mark.asyncio
async def test_extracts_entry_within_window():
    xml = _feed(_item("New AI model released", "https://example.com/a", 2))
    articles = await _source(url=xml).fetch(SINCE)

    assert len(articles) == 1
    a = articles[0]
    assert a.title == "New AI model released"
    assert a.url == "https://example.com/a"
    assert a.source == "Example Feed"
    assert a.authority == 0.6
    assert a.source_type == SourceCategory.FEED


@pytest.mark.asyncio
async def test_drops_entries_older_than_since():
    xml = _feed(
        _item("Fresh", "https://example.com/a", 5)
        + _item("Stale", "https://example.com/b", 30)
    )
    articles = await _source(url=xml).fetch(SINCE)

    assert [a.title for a in articles] == ["Fresh"]


@pytest.mark.asyncio
async def test_drops_entries_with_no_title_or_link():
    xml = _feed(_item("", "https://example.com/a", 1) + _item("No link", "", 1))
    articles = await _source(url=xml).fetch(SINCE)

    assert articles == []


@pytest.mark.asyncio
async def test_relevance_filter_drops_non_ai_titles_when_not_trusted():
    xml = _feed(
        _item("New AI model released", "https://example.com/a", 1)
        + _item("Local bakery wins award", "https://example.com/b", 1)
    )
    articles = await _source(url=xml, ai_relevant="filter").fetch(SINCE)

    assert [a.title for a in articles] == ["New AI model released"]


@pytest.mark.asyncio
async def test_trusts_all_entries_when_ai_relevant_always():
    xml = _feed(_item("Local bakery wins award", "https://example.com/a", 1))
    articles = await _source(url=xml, ai_relevant="always").fetch(SINCE)

    assert len(articles) == 1
