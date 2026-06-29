"""Registry of source connectors (feed, API, scraper)."""

from __future__ import annotations

from enum import StrEnum

from dailyai.services.ingestion.sources.api_source import (
    HackerNewsSource,
    HuggingFacePapersSource,
)
from dailyai.services.ingestion.sources.base import Source
from dailyai.services.ingestion.sources.rss_source import RSSSource
from dailyai.services.ingestion.sources.scrape_source import ScrapeSource


class SourceType(StrEnum):
    """Source types available for configuration."""

    RSS = "rss"
    SCRAPE = "scrape"
    HACKERNEWS = "hackernews"
    HF_PAPERS = "hf_papers"


SOURCE_REGISTRY: dict[SourceType, type[Source]] = {
    SourceType.RSS: RSSSource,
    SourceType.SCRAPE: ScrapeSource,
    SourceType.HACKERNEWS: HackerNewsSource,
    SourceType.HF_PAPERS: HuggingFacePapersSource,
}


def _init_source(entry: dict) -> Source:
    """Initialize a Source subclass with the kwargs from a feeds.yaml entry."""
    kwargs = {k: v for k, v in entry.items() if k != "type"}

    return SOURCE_REGISTRY[SourceType(entry["type"])](**kwargs)


def init_sources(entries: list[dict]) -> list[Source]:
    """Build every Source listed in feeds.yaml via SOURCE_REGISTRY."""

    return [_init_source(entry) for entry in entries]
