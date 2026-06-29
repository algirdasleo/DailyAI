"""Fetch articles from RSS feeds."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from time import mktime

import feedparser
from bs4 import BeautifulSoup

from dailyai.models import Article, SourceCategory
from dailyai.services.ingestion.relevance import is_ai_relevant
from dailyai.services.ingestion.sources.base import Source

logger = logging.getLogger(__name__)


class RSSSource(Source):
    """Fetch articles from an RSS feed."""

    source_type = SourceCategory.FEED

    def __init__(
        self,
        name: str,
        url: str,
        authority: float = 0.5,
        ai_relevant: str | None = None,
    ):
        self.name = name
        self.url = url
        self.authority = float(authority)
        self.trust_all = ai_relevant == "always"

    async def fetch(self, since: datetime) -> list[Article]:
        parsed = await asyncio.to_thread(feedparser.parse, self.url)

        out: list[Article] = []
        for e in parsed.entries:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if not title or not link:
                continue

            published = self._parse_published(e)
            if not published or published < since:
                continue

            summary = self._clean_summary(getattr(e, "summary", ""))
            if not self.trust_all and not is_ai_relevant(title, summary):
                continue

            out.append(
                Article(
                    title=title,
                    url=link,
                    source=self.name,
                    authority=self.authority,
                    published=published,
                    summary=summary,
                    source_type=self.source_type,
                )
            )

        return out

    @staticmethod
    def _clean_summary(html: str, limit: int = 500) -> str:

        return BeautifulSoup(html or "", "html.parser").get_text(" ", strip=True)[
            :limit
        ]

    @staticmethod
    def _parse_published(entry: feedparser.FeedParserDict) -> datetime | None:
        for attr in ("published_parsed", "updated_parsed"):
            t = getattr(entry, attr, None)
            if t:
                return datetime.fromtimestamp(mktime(t), tz=timezone.utc)

        return None
