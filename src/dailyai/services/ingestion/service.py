"""Fetch articles from all configured sources."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from dailyai.models import Article
from dailyai.services.ingestion.sources import http
from dailyai.services.ingestion.sources.base import Source

logger = logging.getLogger(__name__)


class IngestionService:
    """Fetch articles from all sources within a time window."""

    def __init__(self, sources: list[Source], window_hours: int):
        self.sources = sources
        self.since = datetime.now(timezone.utc) - timedelta(hours=window_hours)

    async def _fetch_sources(self, source: Source) -> list[Article]:
        try:
            return await source.fetch(self.since)
        except Exception:
            logger.exception("%s failed - skipped", source.name)
            return []

    async def ingest_async(self) -> list[Article]:
        try:
            sources_results = await asyncio.gather(
                *(self._fetch_sources(s) for s in self.sources)
            )
        finally:
            await http.aclose()
        all_articles: list[Article] = []
        for articles in sources_results:
            all_articles.extend(articles)

        return all_articles

    def ingest(self) -> list[Article]:
        return asyncio.run(self.ingest_async())
