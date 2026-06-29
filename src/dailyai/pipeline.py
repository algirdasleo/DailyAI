"""Run the news-to-brief pipeline: ingest → dedup → embed → cluster → rank → fetch → summarize → draft → evaluate."""

from __future__ import annotations

from datetime import date

from dailyai.models import Brief
from dailyai.repositories import ArticleRepository, BriefRepository, MetricsRepository
from dailyai.services import (
    ArticleFetchService,
    ClusteringStrategy,
    DraftingService,
    EmbeddingService,
    EvaluationService,
    FilterService,
    IngestionService,
    RankingStrategy,
    SummarizerService,
)


class EmptyBriefError(RuntimeError):
    """Raised when there is nothing to brief: no fresh articles, or all already published."""


class Pipeline:
    """Orchestrate the end-to-end brief generation pipeline."""

    def __init__(
        self,
        *,
        ingestion_service: IngestionService,
        filter_service: FilterService,
        embedding_service: EmbeddingService,
        clustering_service: ClusteringStrategy,
        ranking_service: RankingStrategy,
        fetch_service: ArticleFetchService,
        summarizer_service: SummarizerService,
        drafting_service: DraftingService,
        evaluation_service: EvaluationService,
        article_repo: ArticleRepository,
        brief_repo: BriefRepository,
        metrics_repo: MetricsRepository,
        top_stories: int,
        stale_after_days: int,
    ):
        self.ingestion_service = ingestion_service
        self.filter_service = filter_service
        self.embedding_service = embedding_service
        self.clustering_service = clustering_service
        self.ranking_service = ranking_service
        self.fetch_service = fetch_service
        self.summarizer_service = summarizer_service
        self.drafting_service = drafting_service
        self.evaluation_service = evaluation_service
        self.article_repo = article_repo
        self.brief_repo = brief_repo
        self.metrics_repo = metrics_repo
        self.top_stories = top_stories
        self.stale_after_days = stale_after_days

    def run(self) -> Brief:
        # 1. Ingest articles from all sources
        articles = self.ingestion_service.ingest()

        # 2. Remove duplicate articles
        distinct_articles = self.filter_service.dedupe_articles(articles)
        if not distinct_articles:
            raise EmptyBriefError(
                "No articles in window - widen window_hours or add more sources."
            )

        # 3. Save articles and compute embeddings
        self.article_repo.upsert_many(distinct_articles)
        self.embedding_service.embed(distinct_articles)

        # 4. Cluster articles by story
        stories = self.clustering_service.cluster(distinct_articles)

        # 5. Remove previously published stories
        new_stories = self.filter_service.drop_already_published(
            stories, self.article_repo, self.stale_after_days
        )
        if not new_stories:
            raise EmptyBriefError(
                "Nothing new to brief - all stories in window already published in a recent brief."
            )

        # 6. Rank stories by importance and keep top K
        ranked_stories = self.ranking_service.rank(new_stories)[: self.top_stories]

        # 7. Fetch full text or use feed description as fallback
        story_texts = self.fetch_service.fetch_articles(ranked_stories)

        # 8. Summarize stories and draft brief
        summaries = self.summarizer_service.summarize(story_texts)
        markdown = self.drafting_service.draft(summaries)

        # 9. Evaluate brief quality
        evaluation = self.evaluation_service.evaluate_all(
            markdown, ranked_stories, summaries, story_texts
        )

        # 10. Save brief and metrics
        brief = Brief(
            brief_date=date.today(), markdown=markdown, evaluation=evaluation.report
        )
        self.brief_repo.save(brief)
        self.metrics_repo.save(
            evaluation.to_metrics(brief.brief_date),
            evaluation.to_detail().model_dump_json(),
        )

        # 11. Save centroids to detect future story repeats
        centroids = [c for c in (s.centroid for s in ranked_stories) if c is not None]
        self.article_repo.add_published_clusters(centroids, brief.brief_date)

        return brief
