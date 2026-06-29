"""Compose the final brief from per-article summaries."""

from __future__ import annotations

from dailyai.llm_client import LLMClient
from dailyai.services.summarizing import ArticleSummary

_EDITORIAL_PROMPT = """You are the editor of a daily AI executive brief for a company whose \
readers span engineering, sales, marketing, HR, finance and design. Write like \
Morning Brew: punchy, plain-language, lightly witty, but factually precise. \
Explain any jargon in-line. The whole brief must be skimmable in 2-4 minutes — \
a reader should get the gist from the headers and bold text alone.

Structure, in GitHub-flavored Markdown, EXACTLY this shape:
- `# ☕ Daily AI Brief`
- A one-line `> ` blockquote TL;DR naming the 2-3 biggest threads.
- `## The big story` — 2-3 sentences on the single most important item, \
**bolding the key number, name, or figure**, ending with why the reader cares, \
then `[Source](url)` on its own line.
- `---`
- Then the rest of the stories, most important first. Each is EXACTLY:
  - `### <specific, punchy one-line takeaway>`  (no "Story N:" prefix)
  - One or two tight sentences of what happened, **bolding the standout \
number, name, or fact**.
  - Blank line.
  - `**Why it matters:** <one sentence on the business/practical consequence>`
  - Blank line.
  - `[Source](url)` on its own line.
- Cite each story's source EXACTLY ONCE, only in that dedicated `[Source](url)` \
line — never link or name the source again inline in the body sentences.
- Keep each story under ~55 words so the page stays scannable.
- Cite ONLY the URLs provided; never invent a URL or a fact. If a story is thin \
or sources disagree, say so in one clause rather than embellish."""

DRAFT_SYSTEM_PROMPT = (
    _EDITORIAL_PROMPT
    + "\n\nYou are now writing from verified per-article summaries — these are "
    "the ONLY facts you may use."
)


class DraftingService:
    """Draft the brief from per-article summaries."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def draft(self, summaries: list[ArticleSummary]) -> str:
        story_blocks = []
        for i, summary in enumerate(summaries, 1):
            lines = [
                f"### Story {i}: {summary.title}  ({summary.source} | {summary.url})",
                summary.summary,
            ]
            lines.extend(f"- {point}" for point in summary.key_points)
            story_blocks.append("\n".join(lines))

        user = (
            "Here are today's verified story summaries, most important first. "
            "Write the draft brief using ONLY these facts and citing each story's "
            "URL inline.\n\n" + "\n\n".join(story_blocks)
        )

        return str(self.llm.generate(DRAFT_SYSTEM_PROMPT, user))
