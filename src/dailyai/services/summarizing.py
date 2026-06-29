"""Summarize each article into structured facts and key points."""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel, Field

from dailyai.llm_client import LLMClient
from dailyai.services.fetching import ArticleText

logger = logging.getLogger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You are a precise news summarizer. You receive ONE \
article's title and full text. Produce a faithful, compact summary for a busy \
business reader: 2-3 sentences capturing what actually happened (events, \
numbers, names), plus a few key factual points. Do NOT add anything not present \
in the text. Do NOT include opinions or a 'why it matters' angle - just the \
facts as reported."""

SUMMARIZER_USER_PROMPT_TEMPLATE = (
    "=== TITLE ===\n{title}\n\n=== ARTICLE ===\n{full_text}"
)


class SummaryDraft(BaseModel):
    """LLM output: summary text and key points."""

    summary: str = Field(description="2-3 sentence faithful summary")
    key_points: list[str] = Field(default_factory=list)


class ArticleSummary(BaseModel):
    """Article summary with citation for drafting and evaluation."""

    url: str
    source: str
    title: str
    summary: str
    key_points: list[str] = Field(default_factory=list)


class SummarizerService:
    """Generate per-article summaries via LLM."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize(self, texts: list[ArticleText]) -> list[ArticleSummary]:
        """Generate summaries for multiple articles in parallel via LLM."""

        async def _generate_summary(text: ArticleText) -> ArticleSummary | None:
            user_prompt = SUMMARIZER_USER_PROMPT_TEMPLATE.format(
                title=text.title, full_text=text.full_text
            )
            summary_draft = await asyncio.to_thread(
                self.llm.generate_structured,
                SUMMARIZER_SYSTEM_PROMPT,
                user_prompt,
                SummaryDraft,
            )
            if summary_draft is None:
                return None
            return ArticleSummary(
                url=text.url,
                source=text.source,
                title=text.title,
                **summary_draft.model_dump(),
            )

        async def _generate_all() -> list[ArticleSummary | None]:
            return await asyncio.gather(*[_generate_summary(text) for text in texts])

        summaries = asyncio.run(_generate_all())
        return [s for s in summaries if s is not None]
