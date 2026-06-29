from datetime import datetime, timezone

import numpy as np

from dailyai.models import Article, SourceCategory
from dailyai.services.clustering import AgglomerativeClusterer


def _article(vec) -> Article:
    a = Article(
        title="t",
        url="https://x.com/a",
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )
    if vec is not None:
        a.embedding = np.array(vec, dtype=np.float32)
    return a


def test_agglomerative_groups_similar_separates_dissimilar():
    clusterer = AgglomerativeClusterer(threshold=0.55)
    stories = clusterer.cluster(
        [_article([1, 0]), _article([0.8, 0.6]), _article([0, 1])]
    )  # first two cosine 0.8 (merge), third orthogonal (separate)
    sizes = sorted(len(s.articles) for s in stories)
    assert sizes == [1, 2]


def test_agglomerative_single_article():
    clusterer = AgglomerativeClusterer(threshold=0.55)
    stories = clusterer.cluster([_article([1, 0])])
    assert len(stories) == 1
    assert len(stories[0].articles) == 1


def test_agglomerative_empty_and_unembedded():
    clusterer = AgglomerativeClusterer(threshold=0.55)
    assert clusterer.cluster([]) == []
    assert clusterer.cluster([_article(None)]) == []
