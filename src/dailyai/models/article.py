"""Article covers a news Story and is the unit of ingestion, embedding, and clustering."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Optional

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class SourceCategory(StrEnum):
    """How an Article was obtained - recorded on every Article for traceability."""

    FEED = "feed"
    API = "api"
    SCRAPE = "scrape"


class Article(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    title: str
    url: str
    source: str  # human name, e.g. "TechCrunch"
    authority: float = Field(ge=0.0, le=1.0)  # from the source registry
    published: datetime  # tz-aware UTC
    summary: str = ""  # plain-text snippet (HTML stripped)
    source_type: SourceCategory = SourceCategory.FEED  # for traceability
    embedding: Optional[np.ndarray] = None

    @property
    def embed_text(self) -> str:
        return f"{self.title}\n\n{self.summary}".strip()
