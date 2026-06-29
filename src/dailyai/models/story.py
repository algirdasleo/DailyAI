"""Group of articles covering the same news story."""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field

from dailyai.models.article import Article


class RankMetadata(BaseModel):
    """Ranking metadata: source count, authority, and recency."""

    source_count: int = Field(
        description="Number of distinct sources covering the story"
    )
    authority: float = Field(description="Highest source authority score in the story")
    recency_hours: float = Field(
        description="Hours since the newest article was published"
    )


class Story(BaseModel):
    """Articles grouped by topic."""

    articles: list[Article] = Field(default_factory=list)

    @property
    def main_article(self) -> Article:
        """Primary article: highest authority, then earliest."""
        if not self.articles:
            raise ValueError("Cannot get canonical article from empty story")
        return sorted(self.articles, key=lambda a: (-a.authority, a.published))[0]

    @property
    def sources(self) -> list[str]:
        return sorted({a.source for a in self.articles})

    @property
    def source_count(self) -> int:
        """Number of distinct sources covering this story."""
        return len(self.sources)

    @property
    def centroid(self) -> "np.ndarray | None":
        """Normalized mean of member embeddings; None if none are embedded."""
        vecs = [a.embedding for a in self.articles if a.embedding is not None]
        if not vecs:
            return None
        c = np.mean(vecs, axis=0)
        n = np.linalg.norm(c)
        if not n:
            return None
        return c / n


class RankedStory(Story):
    """Story with ranking metadata guaranteed."""

    rank_metadata: RankMetadata
