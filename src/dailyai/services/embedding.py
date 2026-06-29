"""Embed articles with caching; reuse cached embeddings, batch API calls."""

from __future__ import annotations

import numpy as np

from dailyai.llm_client import LLMClient
from dailyai.models import Article
from dailyai.repositories.base import ArticleRepository


class EmbeddingService:
    """Generate vector embeddings for articles with caching and persistence."""

    def __init__(
        self, llm: LLMClient, article_repo: ArticleRepository | None = None
    ):
        self.llm = llm
        self.repo = article_repo

    def embed(self, articles: list[Article]) -> None:
        """Embed articles in place, reusing existing embeddings and persisting new ones."""
        if not articles:
            return

        # Load existing embeddings from repository to avoid redundant API calls.
        existing = (
            self.repo.load_embeddings([a.url for a in articles]) if self.repo else {}
        )
        for a in articles:
            if a.embedding is None and a.url in existing:
                a.embedding = existing[a.url]

        # Call embeddings API only for articles without embeddings.
        to_embed = [a for a in articles if a.embedding is None]
        if to_embed:
            vectors = self.llm.embed([a.embed_text for a in to_embed])
            for art, vec in zip(to_embed, vectors):
                v = np.asarray(vec, dtype=np.float32)
                n = np.linalg.norm(v)
                art.embedding = v / n if n else v

        # Persist embeddings to the repository.
        if self.repo:
            self.repo.save_embeddings(articles)
