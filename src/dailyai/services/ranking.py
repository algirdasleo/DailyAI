"""Order stories by importance for the brief."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, Field

from dailyai.llm_client import LLMClient
from dailyai.models import Story
from dailyai.models.story import RankedStory, RankMetadata

RANKING_SYSTEM_PROMPT = """You order today's AI news stories by importance for a mixed \
business audience (engineering, sales, marketing, HR, finance, design). You are \
given numbered story digests, each with hints: how many sources corroborate it, \
its top source authority, and how recent it is. Weigh genuine consequence over \
mere volume — a single-source scoop from an authoritative first party can outrank \
a widely-reposted minor item. Return the story numbers in the order they should \
appear, most important first, including every number exactly once."""


class RankingStrategy(ABC):
    @abstractmethod
    def rank(self, stories: list[Story]) -> list[RankedStory]:
        """Return stories sorted most-important-first, with rank_metadata guaranteed."""


class RankingOrder(BaseModel):
    """LLM output: story indices ordered by importance."""

    order: list[int] = Field(default_factory=list)


class LLMRanker(RankingStrategy):
    """Order stories via LLM with ranking metadata as hints; falls back to baseline."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def rank(self, stories: list[Story]) -> list[RankedStory]:
        if not stories:
            return []

        ranked_stories = [
            RankedStory(
                articles=s.articles, rank_metadata=self._calculate_rank_metadata(s)
            )
            for s in stories
        ]

        if len(ranked_stories) == 1:
            return ranked_stories

        def sort_key(s: RankedStory) -> tuple:
            m = s.rank_metadata
            return (m.source_count, m.authority, -m.recency_hours)

        baseline = sorted(ranked_stories, key=sort_key, reverse=True)

        ranking_prompt = "\n".join(
            f"{i}. {s.main_article.title}\n"
            f"   sources={s.rank_metadata.source_count} "
            f"authority={s.rank_metadata.authority} "
            f"recency_hours={s.rank_metadata.recency_hours}"
            for i, s in enumerate(baseline, 1)
        )

        result = self.llm.generate_structured(
            RANKING_SYSTEM_PROMPT, ranking_prompt, RankingOrder
        )
        if not result:
            return baseline

        seen: set[int] = set()
        ordered = [
            baseline[idx - 1]
            for idx in result.order
            if 1 <= idx <= len(baseline) and not (idx in seen or seen.add(idx))
        ]
        ordered.extend(s for i, s in enumerate(baseline, 1) if i not in seen)

        return ordered

    @staticmethod
    def _calculate_rank_metadata(story: Story) -> RankMetadata:
        newest = max(a.published for a in story.articles)
        age_h = (datetime.now(timezone.utc) - newest).total_seconds() / 3600

        return RankMetadata(
            source_count=story.source_count,
            authority=round(max(a.authority for a in story.articles), 3),
            recency_hours=round(age_h, 1),
        )
