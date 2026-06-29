"""An article from a news source; unit of ingestion, embedding, and clustering."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SourceCategory(StrEnum):
    """Source type: RSS feed, API, or scraper."""

    FEED = "feed"
    API = "api"
    SCRAPE = "scrape"


class Article(BaseModel):
    """News article with title, URL, source, and embedding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    url: str
    source: str
    authority: float = Field(ge=0.0, le=1.0)
    published: datetime
    summary: str = ""
    source_type: SourceCategory = SourceCategory.FEED
    embedding: Optional[np.ndarray] = None

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n\n{self.summary}".strip()
