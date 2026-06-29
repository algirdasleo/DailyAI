from datetime import datetime, timezone

import numpy as np

from dailyai.models import Article, SourceCategory, Story


def _article(vec) -> Article:
    a = Article(
        title="t",
        url="https://x.com/a",
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )
    a.embedding = np.array(vec, dtype=np.float32)
    return a


def test_centroid_is_normalized_mean():
    s = Story(articles=[_article([2, 0]), _article([0, 2])])
    assert s.centroid is not None
    assert np.allclose(s.centroid, [0.70710677, 0.70710677])


def test_centroid_none_when_unembedded():
    unembedded = Article(
        title="t",
        url="https://x.com/b",
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )  # embedding defaults to None
    s = Story(articles=[unembedded])
    assert s.centroid is None
