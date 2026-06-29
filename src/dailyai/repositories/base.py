"""Repository abstractions for articles, briefs, and metrics."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date

import numpy as np

from dailyai.models import Article, Brief, PipelineMetrics


class ArticleRepository(ABC):
    """Store and retrieve articles and embeddings."""

    @abstractmethod
    def upsert_many(self, articles: list[Article]) -> int:
        """Persist articles, ignoring ones already stored. Returns count newly added."""

    @abstractmethod
    def recent_cluster_centroids(self, since: date) -> list[np.ndarray]:
        """Centroids of clusters published in briefs on/after `since` - the
        history a new story is checked against to detect a repeat."""

    @abstractmethod
    def add_published_clusters(
        self, centroids: list[np.ndarray], brief_date: date
    ) -> None:
        """Persist the centroids of the clusters shown in `brief_date`'s brief."""

    @abstractmethod
    def load_embeddings(self, urls: list[str]) -> dict[str, np.ndarray]:
        """Return stored embeddings keyed by URL, for the given URLs that have one."""

    @abstractmethod
    def save_embeddings(self, articles: list["Article"]) -> None:
        """Persist each article's embedding (by URL); articles with no embedding are skipped."""


class BriefRepository(ABC):
    """Store generated briefs."""

    @abstractmethod
    def save(self, brief: Brief) -> None:
        """Persist a brief with its evaluation report and metadata."""


class MetricsRepository(ABC):
    """Store evaluation metrics for each brief."""

    @abstractmethod
    def save(self, metrics: PipelineMetrics, detail_json: str | None = None) -> None:
        """Persist one per-brief metrics payload (one row per brief).

        `detail_json` is an optional JSON blob of per-item diagnostics
        (fabricated URLs, unsupported claims, cluster verdicts) behind the
        scalar metrics."""
