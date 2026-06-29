from datetime import datetime, timezone

import numpy as np
import pytest

from dailyai.models import (
    Article,
    EvaluationReport,
    PipelineMetrics,
    SourceCategory,
    Story,
)
from dailyai.models.story import RankedStory, RankMetadata
from dailyai.pipeline import EmptyBriefError, Pipeline
from dailyai.repositories import ArticleRepository, BriefRepository, MetricsRepository
from dailyai.services.clustering import ClusteringStrategy
from dailyai.services.drafting import DraftingService
from dailyai.services.embedding import EmbeddingService
from dailyai.services.evaluation import (
    CoherenceReport,
    EvaluationService,
    FullEvaluation,
    BriefReview,
)
from dailyai.services.fetching import ArticleFetchService, ArticleText
from dailyai.services.filtering import FilterService
from dailyai.services.ingestion import IngestionService
from dailyai.services.ranking import RankingStrategy
from dailyai.services.summarizing import ArticleSummary, SummarizerService


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


class _FakeIngestion(IngestionService):
    def __init__(self, articles):
        self._articles = articles

    def ingest(self):
        return list(self._articles)


class _FakeFilter(FilterService):
    """Passthrough dedupe (test articles intentionally share a URL); the real
    drop_already_published so the restatement logic under test runs for real."""

    def __init__(self, repeat_threshold):
        self._real = FilterService(repeat_threshold=repeat_threshold)

    def dedupe_articles(self, articles):
        return articles

    def drop_already_published(self, stories, article_repo, stale_after_days):
        return self._real.drop_already_published(
            stories, article_repo, stale_after_days
        )


class _FakeEmbed(EmbeddingService):
    def __init__(self):
        pass

    def embed(self, articles):
        pass


class _FakeCluster(ClusteringStrategy):
    """One article per story (embeddings already set on the articles)."""

    def cluster(self, articles):
        return [Story(articles=[a]) for a in articles]


class _FakeRank(RankingStrategy):
    def rank(self, stories):
        dummy_metadata = RankMetadata(source_count=1, authority=0.5, recency_hours=0.0)
        return [
            RankedStory(articles=s.articles, rank_metadata=dummy_metadata)
            for s in stories
        ]


class _FakeArticleRepo(ArticleRepository):
    def __init__(self, prior_centroids=()):
        self._prior = list(prior_centroids)
        self.upserted: list[Article] = []
        self.added: list[np.ndarray] = []

    def upsert_many(self, articles):
        self.upserted = list(articles)
        return len(articles)

    def recent_cluster_centroids(self, since):
        return list(self._prior)

    def add_published_clusters(self, centroids, brief_date):
        self.added = list(centroids)

    def load_embeddings(self, urls):
        raise NotImplementedError

    def save_embeddings(self, articles):
        raise NotImplementedError


class _FakeBriefRepo(BriefRepository):
    def save(self, brief):
        self.saved = brief


class _FakeFetch(ArticleFetchService):
    def __init__(self):
        pass

    def fetch_articles(self, stories):
        return [
            ArticleText(
                url=s.main_article.url, title="T", source="s", full_text="body"
            )
            for s in stories
        ]


class _FakeSummarizer(SummarizerService):
    def __init__(self):
        pass

    def summarize(self, texts):
        return [
            ArticleSummary(
                url=t.url, source=t.source, title=t.title, summary="sum", key_points=[]
            )
            for t in texts
        ]


class _FakeDraft(DraftingService):
    def __init__(self):
        pass

    def draft(self, summaries):
        return "# draft"


class _RichEval(EvaluationService):
    def __init__(self):
        pass

    def evaluate(self, brief_md, stories, summaries=None, full_texts=None):
        return EvaluationReport(
            urls_cited=1, urls_grounded=1, n_claims=2, support_rate=1.0
        )

    def coherence_review(self, stories):
        return CoherenceReport(coherent_rate=1.0)

    def brief_review(self, brief_md):
        return BriefReview(
            scannable=True, stories_with_takeaway=1, unexplained_jargon_terms=0
        )

    def evaluate_all(self, brief_md, stories, summaries=None, full_texts=None):
        return FullEvaluation(
            report=self.evaluate(brief_md, stories, summaries, full_texts),
            coherence=self.coherence_review(stories),
            review=self.brief_review(brief_md),
        )


class _FakeMetricsRepo(MetricsRepository):
    def __init__(self):
        self.saved = None

    def save(self, metrics, detail_json=None):
        self.saved = metrics
        self.saved_detail = detail_json


def _full_pipeline(articles, article_repo, metrics_repo):
    return Pipeline(
        ingestion_service=_FakeIngestion(articles),
        filter_service=_FakeFilter(repeat_threshold=0.9),
        embedding_service=_FakeEmbed(),
        clustering_service=_FakeCluster(),
        ranking_service=_FakeRank(),
        fetch_service=_FakeFetch(),
        summarizer_service=_FakeSummarizer(),
        drafting_service=_FakeDraft(),
        evaluation_service=_RichEval(),
        article_repo=article_repo,
        brief_repo=_FakeBriefRepo(),
        metrics_repo=metrics_repo,
        top_stories=10,
        stale_after_days=14,
    )


def _pipeline(articles, article_repo):
    return _full_pipeline(articles, article_repo, _FakeMetricsRepo())


def test_run_uses_draft_and_persists_metrics():
    fresh = _article([0, 1])
    repo = _FakeArticleRepo()
    metrics = _FakeMetricsRepo()

    brief = _full_pipeline([fresh], repo, metrics).run()

    assert brief.markdown == "# draft"
    assert isinstance(metrics.saved, PipelineMetrics)
    assert metrics.saved.support_rate == 1.0
    assert metrics.saved.cluster_coherence == 1.0
    assert metrics.saved.is_scannable is True


def test_skips_story_matching_a_recent_cluster():
    recurring = _article([1, 0])  # cosine 1.0 to the prior centroid below
    fresh = _article([0, 1])
    repo = _FakeArticleRepo(prior_centroids=[np.array([1, 0], dtype=np.float32)])

    _pipeline([recurring, fresh], repo).run()

    assert len(repo.added) == 1
    assert np.allclose(repo.added[0], [0, 1])


def test_records_shown_clusters_when_no_history():
    fresh = _article([0, 1])
    repo = _FakeArticleRepo()

    _pipeline([fresh], repo).run()

    assert len(repo.added) == 1
    assert np.allclose(repo.added[0], [0, 1])


def test_raises_when_everything_is_a_restatement():
    recurring = _article([1, 0])
    repo = _FakeArticleRepo(prior_centroids=[np.array([1, 0], dtype=np.float32)])

    with pytest.raises(EmptyBriefError):
        _pipeline([recurring], repo).run()
