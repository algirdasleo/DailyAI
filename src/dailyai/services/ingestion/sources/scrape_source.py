"""Scrape articles from listing pages via CSS selectors."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from dailyai.models import Article, SourceCategory
from dailyai.services.ingestion.relevance import is_ai_relevant
from dailyai.services.ingestion.sources import http
from dailyai.services.ingestion.sources.base import Source

logger = logging.getLogger(__name__)


def _parse_date(raw: str) -> datetime | None:
    """Parse an ISO-8601 string to a tz-aware UTC datetime, or None."""
    try:
        dt = datetime.fromisoformat(raw.strip())
    except (ValueError, AttributeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


class ScrapeSource(Source):
    """Scrape articles from listing pages using CSS selectors."""

    source_type = SourceCategory.SCRAPE

    def __init__(
        self,
        name: str,
        url: str,
        item: str,
        title: str,
        date: str,
        link: str | None = None,
        authority: float = 0.5,
        ai_relevant: str | None = None,
    ):
        if not date:
            raise ValueError(
                f"ScrapeSource {name!r}: a `date` selector is required - "
                "only sources that expose item dates may be scraped."
            )
        self.name = name
        self.url = url
        self.item_sel = item
        self.title_sel = title
        self.date_sel = date
        self.link_sel = link
        self.authority = float(authority)
        self.trust_all = ai_relevant == "always"

    async def fetch(self, since: datetime) -> list[Article]:
        r = await http.get(self.url)
        soup = await asyncio.to_thread(BeautifulSoup, r.text, "html.parser")

        cards = soup.select(self.item_sel)
        out: list[Article] = []
        seen_urls: set[str] = set()
        for card in cards:
            date_el = card.select_one(self.date_sel)
            if not date_el:
                continue

            raw = date_el.get("datetime") or date_el.get_text(strip=True)
            published = _parse_date(str(raw))
            if not published or published < since:
                continue

            title_el = card.select_one(self.title_sel)
            title = title_el.get_text(" ", strip=True) if title_el else ""
            if not title:
                continue

            link_el = (
                card.select_one(self.link_sel) if self.link_sel else card.find("a")
            )
            href = str(link_el.get("href")) if link_el and link_el.get("href") else ""
            if not href:
                continue
            url = urljoin(self.url, href.strip())

            if url in seen_urls:
                continue
            seen_urls.add(url)

            if not self.trust_all and not is_ai_relevant(title):
                continue

            out.append(
                Article(
                    title=title,
                    url=url,
                    source=self.name,
                    authority=self.authority,
                    published=published,
                    source_type=self.source_type,
                )
            )

        return out
