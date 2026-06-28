"""Story = group of articles covering the same story."""

from __future__ import annotations

from pydantic import BaseModel, Field

from dailyai.models.article import Article


class Story(BaseModel):
    articles: list[Article] = Field(default_factory=list)
    score: float = 0.0
    signals: dict = Field(default_factory=dict)

    @property
    def canonical(self) -> Article:
        """The article cited as the primary link: highest authority, then earliest."""
        return sorted(self.articles, key=lambda a: (-a.authority, a.published))[0]

    @property
    def sources(self) -> list[str]:
        return sorted({a.source for a in self.articles})

    @property
    def source_count(self) -> int:
        """Number of distinct sources covering this story."""
        return len(self.sources)
