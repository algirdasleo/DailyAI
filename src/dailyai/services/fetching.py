"""Fetch and extract full article text for the brief stories."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
from pydantic import BaseModel

from dailyai.models import Article, Story
from dailyai.services.ingestion.sources import http

logger = logging.getLogger(__name__)

_STRUCTURAL_TAGS = ("script", "style", "nav", "header", "footer", "aside", "form")


class ArticleText(BaseModel):
    """Extracted full text with citation."""

    url: str
    title: str
    source: str
    full_text: str


class ArticleFetchService:
    """Fetch and extract full article bodies."""

    def __init__(
        self,
        timeout: int = 15,
        max_chars: int = 8000,
        respect_robots: bool = True,
    ):
        self.timeout = timeout
        self.max_chars = max_chars
        self.respect_robots = respect_robots

    async def fetch_one(self, article: Article) -> ArticleText | None:
        try:
            if not await self._can_fetch(article.url):
                return None

            r = await http.get(article.url, timeout=self.timeout)
            text = await asyncio.to_thread(self._extract, r.text)
            if not text:
                return None

            return ArticleText(
                url=article.url,
                title=article.title,
                source=article.source,
                full_text=text,
            )
        except Exception:
            logger.exception("%s failed - using snippet", article.url)
            return None

    def fetch_articles(self, stories: Sequence[Story]) -> list[ArticleText]:
        """Fetch full text for each story, falling back to its snippet on failure."""

        async def _fetch_all() -> list[ArticleText | None]:
            try:
                return await asyncio.gather(
                    *[self.fetch_one(s.main_article) for s in stories]
                )
            finally:
                await http.aclose()

        results = asyncio.run(_fetch_all())
        fetched = {text.url: text for text in results if text is not None}

        return [
            fetched.get(s.main_article.url)
            or ArticleText(
                url=s.main_article.url,
                title=s.main_article.title,
                source=s.main_article.source,
                full_text=s.main_article.summary or s.main_article.title,
            )
            for s in stories
        ]

    async def _can_fetch(self, url: str) -> bool:
        """Check robots.txt; fail-open if unreachable."""
        if not self.respect_robots:
            return True
        try:
            parsed = urlparse(url)
            robots_url = urljoin(f"{parsed.scheme}://{parsed.netloc}", "/robots.txt")
            r = await http.get(robots_url, timeout=self.timeout)
            rp = RobotFileParser()
            rp.parse(r.text.splitlines())

            return rp.can_fetch(http.USER_AGENT, url)
        except Exception:
            logger.debug("robots check skipped for %s", url)
            return True

    @staticmethod
    def _extract(html: str) -> str:
        """Extract article text from HTML, stripping structural elements."""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRUCTURAL_TAGS):
            tag.decompose()
        root = soup.find("article") or soup.body or soup
        text = root.get_text(" ", strip=True)
        return text
