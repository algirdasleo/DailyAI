"""Verify the brief: URL grounding, claim support, cluster coherence, review."""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date

from pydantic import BaseModel, Field

from dailyai.llm_client import LLMClient
from dailyai.models import (
    ClusterCoherence,
    EvaluationDetail,
    EvaluationReport,
    PipelineMetrics,
    Story,
)
from dailyai.services.fetching import ArticleText
from dailyai.services.summarizing import ArticleSummary

_URL_RE = re.compile(r"\((https?://[^)\s]+)\)")
_MAX_FULL_TEXT_CHARS = 2000

JUDGE_SYSTEM_PROMPT = """You are a strict fact-checker for a news brief. You receive \
(A) the brief and (B) the ONLY allowed source material (titles, snippets, URLs).

For every FACTUAL claim in the brief (what happened, numbers, names, who did \
what), decide if it is SUPPORTED by B or UNSUPPORTED (cannot be derived from B). \
Ignore opinions, predictions, and "why it matters" commentary."""

COHERENCE_SYSTEM_PROMPT = """You check news CLUSTERING quality. Each cluster is a group \
of articles we believe cover the SAME story. For each cluster, decide if all \
members really are the same story (coherent) or if unrelated items were merged \
(incoherent). Internal consistency only — there is no answer key. Return one \
verdict per cluster index given."""

BRIEF_REVIEW_SYSTEM_PROMPT = """You audit a daily AI executive brief and return: \
1. scannable (boolean: true if tight, 2-4 min read) \
2. stories_with_takeaway (count: how many stories have a clear business takeaway) \
3. unexplained_jargon_terms (count: how many technical terms lack inline explanation). \
Be strict and precise with counts."""


class ClaimCheck(BaseModel):
    """One factual claim from the brief and whether the sources support it."""

    claim: str = Field(description="the factual claim, paraphrased short")
    supported: bool = Field(description="True iff derivable from the allowed sources")


class JudgeVerdict(BaseModel):
    """Structured-output schema the judge model must conform to."""

    claims: list[ClaimCheck]
    notes: str = ""


class ClusterVerdicts(BaseModel):
    """Coherence verdicts for all clusters."""

    verdicts: list[ClusterCoherence] = Field(default_factory=list)


class CoherenceReport(BaseModel):
    """Clustering coherence evaluation results."""

    clusters: list[ClusterCoherence] = Field(default_factory=list)
    coherent_rate: float = 1.0
    failed: bool = False


class BriefReview(BaseModel):
    """LLM quality review (counts are actual, not pass/fail)."""

    scannable: bool = False
    stories_with_takeaway: int = 0
    unexplained_jargon_terms: int = 0
    failed: bool = False


class FullEvaluation(BaseModel):
    """Bundle of all three evaluation passes for one brief."""

    report: EvaluationReport
    coherence: CoherenceReport
    review: BriefReview

    def to_detail(self) -> EvaluationDetail:
        """Collect the per-item diagnostics each pass produced but the
        scalar metrics drop."""
        return EvaluationDetail(
            fabricated_urls=self.report.fabricated_urls,
            unsupported_claims=self.report.unsupported_claims,
            cluster_verdicts=self.coherence.clusters,
            claim_check_failed=self.report.claim_check_failed,
            coherence_failed=self.coherence.failed,
            review_failed=self.review.failed,
        )

    def to_metrics(self, brief_date: date) -> PipelineMetrics:
        """Convert evaluation results to pipeline metrics."""
        return PipelineMetrics(
            brief_date=brief_date,
            support_rate=self.report.support_rate,
            total_claims=self.report.n_claims,
            grounded_urls=self.report.urls_grounded,
            cluster_coherence=None
            if self.coherence.failed
            else self.coherence.coherent_rate,
            is_scannable=None if self.review.failed else self.review.scannable,
            stories_with_takeaway=None
            if self.review.failed
            else self.review.stories_with_takeaway,
            unexplained_jargon_terms=None
            if self.review.failed
            else self.review.unexplained_jargon_terms,
        )


class EvaluationService:
    """Verify brief faithfulness, coherence, and quality review."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def evaluate(
        self,
        brief_md: str,
        stories: Sequence[Story],
        summaries: Sequence[ArticleSummary],
        full_texts: Sequence[ArticleText] | None = None,
    ) -> EvaluationReport:
        """Check brief grounding and claim support against source material."""
        grounding = self._check_grounding(brief_md, stories)
        full_texts_dict = (
            {text.url: text for text in full_texts} if full_texts else None
        )
        source_block = self._source_block(summaries, full_texts_dict)
        prompt = (
            f"=== BRIEF ===\n{brief_md}\n\n=== ALLOWED SOURCES (B) ===\n{source_block}"
        )

        verdict = self.llm.generate_structured(
            JUDGE_SYSTEM_PROMPT, prompt, JudgeVerdict
        )
        if verdict is None:
            return EvaluationReport(**grounding, claim_check_failed=True)

        unsupported = [claim for claim in verdict.claims if not claim.supported]
        n_claims = len(verdict.claims)
        support_rate = (n_claims - len(unsupported)) / max(n_claims, 1)

        return EvaluationReport(
            **grounding,
            n_claims=n_claims,
            support_rate=round(support_rate, 3),
            unsupported_claims=[claim.claim for claim in unsupported],
            notes=verdict.notes,
        )

    def coherence_review(self, stories: Sequence[Story]) -> CoherenceReport:
        """Evaluate clustering coherence: whether multi-article clusters are thematically related."""
        multi_article_stories = [story for story in stories if len(story.articles) > 1]
        if not multi_article_stories:
            return CoherenceReport(coherent_rate=1.0)

        clusters_text = []
        for index, story in enumerate(multi_article_stories):
            article_lines = "\n".join(
                f"  - [{article.source}] {article.title}" for article in story.articles
            )
            clusters_text.append(f"Cluster {index}:\n{article_lines}")
        prompt = "\n\n".join(clusters_text)

        cluster_verdicts = self.llm.generate_structured(
            COHERENCE_SYSTEM_PROMPT, prompt, ClusterVerdicts
        )
        clusters = cluster_verdicts.verdicts if cluster_verdicts else []
        if not clusters:
            return CoherenceReport(failed=True)

        coherence_rate = sum(cluster.coherent for cluster in clusters) / len(clusters)
        return CoherenceReport(
            clusters=clusters, coherent_rate=round(coherence_rate, 3)
        )

    def brief_review(self, brief_md: str) -> BriefReview:
        """Evaluate brief quality: scannability, takeaways, and jargon clarity."""
        prompt = f"=== BRIEF ===\n{brief_md}"
        verdict = self.llm.generate_structured(
            BRIEF_REVIEW_SYSTEM_PROMPT, prompt, BriefReview
        )

        return verdict if verdict else BriefReview(failed=True)

    def evaluate_all(
        self,
        brief_md: str,
        stories: Sequence[Story],
        summaries: Sequence[ArticleSummary],
        full_texts: Sequence[ArticleText] | None = None,
    ) -> FullEvaluation:
        """Run all three evaluation passes (grounding/claims, coherence, review)."""

        return FullEvaluation(
            report=self.evaluate(brief_md, stories, summaries, full_texts),
            coherence=self.coherence_review(stories),
            review=self.brief_review(brief_md),
        )

    @staticmethod
    def _check_grounding(brief_md: str, stories: Sequence[Story]) -> dict:
        """Deterministic citation check: every link in the brief must appear in sources."""
        cited = set(_URL_RE.findall(brief_md))
        allowed = {article.url for story in stories for article in story.articles}
        fabricated = sorted(cited - allowed)

        return {
            "urls_cited": len(cited),
            "urls_grounded": len(cited) - len(fabricated),
            "fabricated_urls": fabricated,
        }

    @staticmethod
    def _source_block(
        summaries: Sequence[ArticleSummary],
        full_texts: dict[str, ArticleText] | None,
    ) -> str:
        """Build source material block: summaries + full text (if available)."""
        full_texts = full_texts or {}
        lines = []
        for summary in summaries:
            lines.append(f"- {summary.title} | {summary.url}")
            lines.append(f"  summary: {summary.summary}")
            lines.extend(f"  - {key_point}" for key_point in summary.key_points)
            article_text = full_texts.get(summary.url)
            if article_text:
                lines.append(
                    f"  full text: {article_text.full_text[:_MAX_FULL_TEXT_CHARS]}"
                )

        return "\n".join(lines)
