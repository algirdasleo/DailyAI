from datetime import datetime, timezone

import numpy as np

from dailyai.llm_client import LLMClient
from dailyai.models import Article, SourceCategory
from dailyai.repositories import ArticleRepository
from dailyai.services.embedding import EmbeddingService


def _article(url) -> Article:
    return Article(
        title="t",
        url=url,
        source="s",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.API,
    )


class _FakeLLM(LLMClient):
    def __init__(self):
        self.inputs = None

    def embed(self, texts):
        self.inputs = list(texts)
        return [[1.0, 0.0] for _ in texts]


class _FakeRepo(ArticleRepository):
    def __init__(self, cached=None):
        self._cached = dict(cached or {})
        self.saved = None

    def upsert_many(self, articles):
        raise NotImplementedError

    def recent_cluster_centroids(self, since):
        raise NotImplementedError

    def add_published_clusters(self, centroids, brief_date):
        raise NotImplementedError

    def load_embeddings(self, urls):
        return {u: v for u, v in self._cached.items() if u in urls}

    def save_embeddings(self, articles):
        self.saved = [a for a in articles if a.embedding is not None]


def test_embeds_only_missing_articles():
    llm = _FakeLLM()
    repo = _FakeRepo()
    svc = EmbeddingService(llm, repo)
    a, b = _article("https://x.com/a"), _article("https://x.com/b")
    a.embedding = np.array([0.0, 1.0], dtype=np.float32)  # already embedded

    svc.embed([a, b])

    assert llm.inputs == [b.embed_text]  # only the missing one is sent
    assert b.embedding is not None


def test_uses_cache_and_skips_api_when_all_cached():
    llm = _FakeLLM()
    repo = _FakeRepo(cached={"https://x.com/a": np.array([0.0, 1.0], dtype=np.float32)})
    svc = EmbeddingService(llm, repo)
    a = _article("https://x.com/a")

    svc.embed([a])

    assert llm.inputs is None  # API never called
    assert a.embedding is not None
    assert np.allclose(a.embedding, [0.0, 1.0])


def test_writes_new_embeddings_back():
    llm = _FakeLLM()
    repo = _FakeRepo()
    svc = EmbeddingService(llm, repo)
    a = _article("https://x.com/a")

    svc.embed([a])

    assert repo.saved == [a]


def test_works_without_a_repo():
    llm = _FakeLLM()
    svc = EmbeddingService(llm)  # no repo
    a = _article("https://x.com/a")

    svc.embed([a])

    assert a.embedding is not None
