from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import pytest

from dailyai.models import SourceCategory
from dailyai.services.ingestion.sources import api_source, http
from dailyai.services.ingestion.sources.api_source import (
    HackerNewsSource,
    HuggingFacePapersSource,
)

NOW = datetime(2026, 6, 27, 12, 0, tzinfo=timezone.utc)
SINCE = NOW - timedelta(hours=24)


def _ts(hours_ago: float) -> int:
    return int((NOW - timedelta(hours=hours_ago)).timestamp())


def _patch_hits(monkeypatch, hits: list[dict]) -> None:
    request = httpx.Request("GET", "https://hn.algolia.com/api/v1/search")
    response = httpx.Response(200, request=request, json={"hits": hits})

    async def fake_get(url, **kwargs):
        return response

    monkeypatch.setattr(http, "get", fake_get)


@pytest.mark.asyncio
async def test_extracts_ai_relevant_hit_within_window(monkeypatch):
    _patch_hits(
        monkeypatch,
        [
            {
                "title": "New AI model released",
                "url": "https://example.com/ai-model",
                "created_at_i": _ts(2),
                "objectID": "1",
            }
        ],
    )
    articles = await HackerNewsSource().fetch(SINCE)

    assert len(articles) == 1
    a = articles[0]
    assert a.title == "New AI model released"
    assert a.url == "https://example.com/ai-model"
    assert a.source == "Hacker News"
    assert a.source_type == SourceCategory.API
    assert a.published == NOW - timedelta(hours=2)


@pytest.mark.asyncio
async def test_drops_hits_older_than_since(monkeypatch):
    _patch_hits(
        monkeypatch,
        [
            {
                "title": "Fresh AI news",
                "url": "https://x.com/a",
                "created_at_i": _ts(5),
                "objectID": "1",
            },
            {
                "title": "Stale AI news",
                "url": "https://x.com/b",
                "created_at_i": _ts(30),
                "objectID": "2",
            },
        ],
    )
    articles = await HackerNewsSource().fetch(SINCE)

    assert [a.title for a in articles] == ["Fresh AI news"]


@pytest.mark.asyncio
async def test_drops_hits_with_no_title(monkeypatch):
    _patch_hits(
        monkeypatch,
        [
            {
                "title": "",
                "url": "https://x.com/a",
                "created_at_i": _ts(1),
                "objectID": "1",
            },
            {"url": "https://x.com/b", "created_at_i": _ts(1), "objectID": "2"},
        ],
    )
    articles = await HackerNewsSource().fetch(SINCE)

    assert articles == []


@pytest.mark.asyncio
async def test_drops_non_ai_relevant_titles(monkeypatch):
    _patch_hits(
        monkeypatch,
        [
            {
                "title": "New AI model released",
                "url": "https://x.com/a",
                "created_at_i": _ts(1),
                "objectID": "1",
            },
            {
                "title": "Local bakery wins award",
                "url": "https://x.com/b",
                "created_at_i": _ts(1),
                "objectID": "2",
            },
        ],
    )
    articles = await HackerNewsSource().fetch(SINCE)

    assert [a.title for a in articles] == ["New AI model released"]


@pytest.mark.asyncio
async def test_falls_back_to_hn_permalink_when_no_url(monkeypatch):
    _patch_hits(
        monkeypatch,
        [{"title": "Ask HN: AI advice?", "created_at_i": _ts(1), "objectID": "999"}],
    )
    articles = await HackerNewsSource().fetch(SINCE)

    assert articles[0].url == "https://news.ycombinator.com/item?id=999"


def _paper(title: str, hours_ago: float, upvotes: int | None):
    return SimpleNamespace(
        title=title,
        submitted_at=NOW - timedelta(hours=hours_ago),
        published_at=None,
        upvotes=upvotes,
        id=title.replace(" ", "-"),
        ai_summary=f"summary of {title}",
        summary="",
    )


def _patch_papers(monkeypatch, papers: list) -> None:
    class FakeApi:
        def list_daily_papers(self, **kwargs):
            return papers

    monkeypatch.setattr(api_source, "HfApi", FakeApi)


@pytest.mark.asyncio
async def test_ranks_by_upvotes_and_takes_top_k(monkeypatch):
    _patch_papers(
        monkeypatch,
        [
            _paper("Mild paper", hours_ago=2, upvotes=5),
            _paper("Breaking paper", hours_ago=3, upvotes=50),
            _paper("Notable paper", hours_ago=4, upvotes=20),
        ],
    )
    articles = await HuggingFacePapersSource(top_k=2).fetch(SINCE)

    assert [a.title for a in articles] == ["Breaking paper", "Notable paper"]


@pytest.mark.asyncio
async def test_drops_papers_older_than_since(monkeypatch):
    _patch_papers(
        monkeypatch,
        [
            _paper("Fresh paper", hours_ago=5, upvotes=10),
            _paper("Stale paper", hours_ago=30, upvotes=999),
        ],
    )
    articles = await HuggingFacePapersSource().fetch(SINCE)

    assert [a.title for a in articles] == ["Fresh paper"]


@pytest.mark.asyncio
async def test_drops_papers_with_no_title(monkeypatch):
    bad = _paper("", hours_ago=1, upvotes=100)
    good = _paper("Real AI paper", hours_ago=1, upvotes=1)
    _patch_papers(monkeypatch, [bad, good])

    articles = await HuggingFacePapersSource().fetch(SINCE)

    assert [a.title for a in articles] == ["Real AI paper"]


@pytest.mark.asyncio
async def test_missing_upvotes_treated_as_zero(monkeypatch):
    no_votes = _paper("Unvoted paper", hours_ago=1, upvotes=None)
    voted = _paper("Voted paper", hours_ago=1, upvotes=3)
    _patch_papers(monkeypatch, [no_votes, voted])

    articles = await HuggingFacePapersSource().fetch(SINCE)

    assert [a.title for a in articles] == ["Voted paper", "Unvoted paper"]
