"""Fetch articles from APIs: Hacker News, Hugging Face."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from huggingface_hub import HfApi

from dailyai.models import Article, SourceCategory
from dailyai.services.ingestion.relevance import is_ai_relevant
from dailyai.services.ingestion.sources import http
from dailyai.services.ingestion.sources.base import Source

logger = logging.getLogger(__name__)


class HackerNewsSource(Source):
    """Fetch front-page stories from Hacker News API."""

    source_type = SourceCategory.API
    name = "Hacker News"
    BASE_URL = "https://hn.algolia.com/api/v1/search"

    def __init__(self, authority: float = 0.5):
        self.authority = authority

    async def fetch(self, since: datetime) -> list[Article]:
        r = await http.get(
            self.BASE_URL, params={"tags": "front_page", "hitsPerPage": 50}
        )
        hits = r.json().get("hits", [])
        cutoff = since.timestamp()

        out: list[Article] = []
        for h in hits:
            title = (h.get("title") or "").strip()
            if not title:
                continue

            published_ts = h.get("created_at_i", 0)
            if published_ts < cutoff:
                continue

            if not is_ai_relevant(title):
                continue

            permalink = f"https://news.ycombinator.com/item?id={h['objectID']}"
            out.append(
                Article(
                    title=title,
                    url=h.get("url") or permalink,
                    source=self.name,
                    authority=self.authority,
                    published=datetime.fromtimestamp(published_ts, tz=timezone.utc),
                    source_type=self.source_type,
                )
            )

        return out


class HuggingFacePapersSource(Source):
    """Fetch trending papers from Hugging Face; filter by window then rank by upvotes."""

    source_type = SourceCategory.API
    name = "HuggingFace Papers"
    BASE_URL = "https://huggingface.co/papers"

    def __init__(self, top_k: int = 15, pool: int = 60, authority: float = 0.75):
        self.top_k = top_k
        self.pool = pool
        self.authority = authority

    async def fetch(self, since: datetime) -> list[Article]:
        papers = await asyncio.to_thread(
            lambda: list(HfApi().list_daily_papers(sort="trending", limit=self.pool))
        )

        fresh = []
        for p in papers:
            published = p.submitted_at or p.published_at
            if not p.title or not published or published < since:
                continue
            fresh.append((p, published))

        fresh.sort(key=lambda pair: pair[0].upvotes or 0, reverse=True)

        out: list[Article] = []
        for p, published in fresh[: self.top_k]:
            out.append(
                Article(
                    title=(p.title or "").strip(),
                    url=f"{self.BASE_URL}/{p.id}",
                    source=self.name,
                    authority=self.authority,
                    published=published,
                    summary=(p.ai_summary or p.summary or "")[:500],
                    source_type=self.source_type,
                )
            )

        return out
