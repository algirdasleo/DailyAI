"""Group articles into stories using agglomerative clustering."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
from sklearn.cluster import AgglomerativeClustering

from dailyai.models import Article, Story


class ClusteringStrategy(ABC):
    @abstractmethod
    def cluster(self, articles: list[Article]) -> list[Story]:
        """Group articles into stories (one cluster = one story)."""


class AgglomerativeClusterer(ClusteringStrategy):
    """Groups articles using hierarchical clustering based on embedding similarity."""

    def __init__(self, threshold: float):
        self.threshold = threshold

    def cluster(self, articles: list[Article]) -> list[Story]:
        embedded = [a for a in articles if a.embedding is not None]
        if len(embedded) <= 1:
            return [Story(articles=embedded)] if embedded else []

        vecs = np.stack([a.embedding for a in embedded if a.embedding is not None])
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            distance_threshold=1 - self.threshold,
            linkage="average",
        )
        labels = clusterer.fit_predict(vecs)

        groups: dict[int, list[Article]] = defaultdict(list)
        for article, label in zip(embedded, labels):
            groups[label].append(article)

        return [Story(articles=members) for members in groups.values()]
