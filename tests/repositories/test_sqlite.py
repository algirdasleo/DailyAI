from datetime import date, datetime, timezone

import numpy as np

from dailyai.models import Article, SourceCategory
from dailyai.repositories import SQLiteArticleRepository, connect


def _repo() -> SQLiteArticleRepository:
    return SQLiteArticleRepository(connect(":memory:"))


def _article(url, vec=None) -> Article:
    a = Article(
        title="t",
        url=url,
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )
    if vec is not None:
        a.embedding = np.array(vec, dtype=np.float32)
    return a


def test_no_centroids_initially():
    assert _repo().recent_cluster_centroids(date(2026, 6, 1)) == []


def test_add_and_read_centroids_roundtrip():
    repo = _repo()
    c1 = np.array([1.0, 0.0], dtype=np.float32)
    c2 = np.array([0.0, 1.0], dtype=np.float32)
    repo.add_published_clusters([c1, c2], date(2026, 6, 28))

    got = repo.recent_cluster_centroids(date(2026, 6, 1))

    assert len(got) == 2
    assert any(np.allclose(g, c1) for g in got)
    assert any(np.allclose(g, c2) for g in got)


def test_recent_centroids_respects_lookback():
    repo = _repo()
    old = np.array([1.0, 0.0], dtype=np.float32)
    fresh = np.array([0.0, 1.0], dtype=np.float32)
    repo.add_published_clusters([old], date(2026, 6, 1))
    repo.add_published_clusters([fresh], date(2026, 6, 28))

    got = repo.recent_cluster_centroids(date(2026, 6, 20))

    assert len(got) == 1
    assert np.allclose(got[0], fresh)


def test_embedding_roundtrip():
    repo = _repo()
    art = _article("https://x.com/1")
    repo.upsert_many([art])
    art.embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    repo.save_embeddings([art])

    got = repo.load_embeddings(["https://x.com/1"])

    assert np.allclose(got["https://x.com/1"], [0.1, 0.2, 0.3])


def test_load_embeddings_empty_when_none_stored():
    repo = _repo()
    repo.upsert_many([_article("https://x.com/1")])
    assert repo.load_embeddings(["https://x.com/1"]) == {}


def test_save_embeddings_ignores_unembedded():
    repo = _repo()
    repo.upsert_many([_article("https://x.com/1")])
    repo.save_embeddings([_article("https://x.com/1")])  # embedding is None
    assert repo.load_embeddings(["https://x.com/1"]) == {}
