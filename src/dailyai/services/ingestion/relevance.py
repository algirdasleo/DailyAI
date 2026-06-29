"""Filter articles for AI relevance."""

from __future__ import annotations

import re

_RELEVANT_TERMS = (
    r"ai",
    r"a\.i\.",
    r"artificial intelligence",
    r"machine learning",
    r"ml",
    r"llms?",
    r"gpt",
    r"claude",
    r"gemini",
    r"openai",
    r"anthropic",
    r"deepmind",
    r"mistral",
    r"neural network",
    r"genai",
    r"generative ai",
    r"inference",
    r"ai agents?",
    r"chatbots?",
    r"transformer",
    r"diffusion model",
    r"hugging ?face",
    r"large language model",
    r"fine-?tun\w*",
    r"copilot",
)
_RELEVANCE_REGEX = re.compile(
    r"\b(" + "|".join(_RELEVANT_TERMS) + r")\b", re.IGNORECASE
)


def is_ai_relevant(title: str, summary: str = "") -> bool:

    return bool(_RELEVANCE_REGEX.search(f"{title} {summary}"))
