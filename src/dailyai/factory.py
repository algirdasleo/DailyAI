"""Build the pipeline with all services and repositories."""

from __future__ import annotations

from dailyai.config import Settings, load_sources
from dailyai.llm_client import LLMClient
from dailyai.pipeline import Pipeline
from dailyai.repositories import (
    SQLiteArticleRepository,
    SQLiteBriefRepository,
    SQLiteMetricsRepository,
    connect,
)
from dailyai.services import (
    AgglomerativeClusterer,
    ArticleFetchService,
    DraftingService,
    EmbeddingService,
    EvaluationService,
    FilterService,
    IngestionService,
    LLMRanker,
    RankingStrategy,
    SummarizerService,
)
from dailyai.services.ingestion.registry import init_sources


def _build_ranker(settings: Settings) -> RankingStrategy:
    return LLMRanker(LLMClient(settings.rank_model, settings.openai_api_key))


def build_pipeline(settings: Settings, feeds_path: str = "feeds.yaml") -> Pipeline:
    conn = connect(settings.db_path)
    article_repo = SQLiteArticleRepository(conn)

    return Pipeline(
        ingestion_service=IngestionService(
            init_sources(load_sources(feeds_path)), settings.window_hours
        ),
        filter_service=FilterService(repeat_threshold=settings.repeat_threshold),
        embedding_service=EmbeddingService(
            LLMClient(settings.embed_model, settings.openai_api_key), article_repo
        ),
        clustering_service=AgglomerativeClusterer(settings.cluster_threshold),
        ranking_service=_build_ranker(settings),
        fetch_service=ArticleFetchService(
            timeout=settings.fetch_timeout,
            max_chars=settings.fetch_max_chars,
        ),
        summarizer_service=SummarizerService(
            LLMClient(settings.summarizer_model, settings.openai_api_key)
        ),
        drafting_service=DraftingService(
            LLMClient(settings.brief_model, settings.openai_api_key)
        ),
        evaluation_service=EvaluationService(
            LLMClient(settings.eval_model, settings.openai_api_key)
        ),
        article_repo=article_repo,
        brief_repo=SQLiteBriefRepository(conn),
        metrics_repo=SQLiteMetricsRepository(conn),
        top_stories=settings.top_stories,
        stale_after_days=settings.stale_after_days,
    )
