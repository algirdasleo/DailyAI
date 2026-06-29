"""Remove duplicate and already-published content."""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from urllib.parse import urlparse, urlunparse

import numpy as np
from rapidfuzz import fuzz

from dailyai.models import Article, Story
from dailyai.repositories import ArticleRepository

logger = logging.getLogger(__name__)


class FilterService:
    """Remove duplicate articles and already-published stories."""

    def __init__(self, fuzzy_threshold: int = 92, repeat_threshold: float = 0.0):
        self.fuzzy_threshold = fuzzy_threshold
        self.repeat_threshold = repeat_threshold

    def dedupe_articles(self, articles: list[Article]) -> list[Article]:
        """Remove duplicate articles by URL and similar titles."""
        seen_urls = set()
        seen_titles = []
        kept_articles = []

        for article in articles:
            normalized_url = self._canonical_url(article.url)
            if normalized_url in seen_urls:
                continue

            normalized_title = re.sub(r"\W+", " ", article.title.lower()).strip()
            if any(
                fuzz.token_sort_ratio(normalized_title, t) >= self.fuzzy_threshold
                for t in seen_titles
            ):
                continue

            seen_urls.add(normalized_url)
            seen_titles.append(normalized_title)
            kept_articles.append(article)

        return kept_articles

    def drop_already_published(
        self,
        stories: list[Story],
        article_repo: ArticleRepository,
        stale_after_days: int,
    ) -> list[Story]:
        """Keep stories not shown in recent briefs."""

        cutoff = date.today() - timedelta(days=stale_after_days)
        prior_centroids = article_repo.recent_cluster_centroids(cutoff)
        if not prior_centroids:
            return list(stories)

        new_stories = []
        for story in stories:
            centroid = story.centroid
            if centroid is not None and any(
                float(np.dot(centroid, prior)) >= self.repeat_threshold
                for prior in prior_centroids
            ):
                continue

            new_stories.append(story)

        return new_stories

    @staticmethod
    def _canonical_url(url: str) -> str:
        parsed = urlparse(url)
        query = "&".join(
            s
            for s in parsed.query.split("&")
            if s and not s.lower().startswith(("utm_", "fbclid", "ref"))
        )
        return urlunparse(
            (
                parsed.scheme.lower(),
                parsed.netloc.lower(),
                parsed.path.rstrip("/"),
                "",
                query,
                "",
            )
        )
