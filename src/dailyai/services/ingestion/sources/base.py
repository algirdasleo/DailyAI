"""Base class for article sources (RSS, API, scraper)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from dailyai.models import Article, SourceCategory


class Source(ABC):
    """Abstract source connector (RSS, API, scraper)."""

    name: str = "source"
    source_type: SourceCategory = SourceCategory.FEED

    @abstractmethod
    async def fetch(self, since: datetime) -> list[Article]:
        """Fetch normalized articles since the given datetime."""
        raise NotImplementedError
