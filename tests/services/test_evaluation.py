from dailyai.models import Article, SourceCategory, Story
from dailyai.services.evaluation import (
    ClusterVerdicts,
    EvaluationService,
    JudgeVerdict,
    BriefReview,
)
from dailyai.services.fetching import ArticleText
from dailyai.services.summarizing import ArticleSummary
from datetime import datetime, timezone
from tests.services._fakellm import FakeLLMBase


def _story(url="https://ex.com/a") -> Story:
    a = Article(
        title="T",
        url=url,
        source="Ex",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        summary="snippet text",
        source_type=SourceCategory.FEED,
    )
    return Story(articles=[a])


class _CaptureLLM(FakeLLMBase):
    def __init__(self):
        self.user: str | None = None

    def generate(self, system, user, response_format=None):
        self.user = user
        return JudgeVerdict(claims=[], notes="")


def test_judge_uses_summaries_and_full_text_when_provided():
    llm = _CaptureLLM()
    svc = EvaluationService(llm)
    summaries = [
        ArticleSummary(
            url="https://ex.com/a",
            source="Ex",
            title="T",
            summary="A launched with 99% accuracy.",
            key_points=["99% accuracy"],
        )
    ]
    full = [
        ArticleText(
            url="https://ex.com/a",
            title="T",
            source="Ex",
            full_text="The full article body says A launched.",
        )
    ]
    svc.evaluate("# brief [link](https://ex.com/a)", [_story()], summaries, full)
    assert llm.user is not None
    assert "99% accuracy" in llm.user  # summary is the source of truth
    assert "full article body" in llm.user
    assert "snippet text" not in llm.user  # snippet no longer the source


def _multi_story() -> Story:
    a1 = Article(
        title="A1",
        url="https://ex.com/1",
        source="X",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.FEED,
    )
    a2 = Article(
        title="A2",
        url="https://ex.com/2",
        source="Y",
        authority=0.5,
        published=datetime(2026, 6, 28, tzinfo=timezone.utc),
        source_type=SourceCategory.FEED,
    )
    return Story(articles=[a1, a2])


class _CoherenceLLM(FakeLLMBase):
    def generate(self, system, user, response_format=None):
        from dailyai.services.evaluation import ClusterCoherence

        return ClusterVerdicts(
            verdicts=[ClusterCoherence(cluster_index=0, coherent=False, reason="mixed")]
        )


class _BriefReviewLLM(FakeLLMBase):
    def generate(self, system, user, response_format=None):
        return BriefReview(
            scannable=True,
            stories_with_takeaway=3,
            unexplained_jargon_terms=1,
        )


class _BoomLLM(FakeLLMBase):
    def generate(self, system, user, response_format=None):
        raise RuntimeError("down")


def test_coherence_skips_single_article_clusters():
    # single-article story is trivially coherent; no LLM call needed
    rep = EvaluationService(_BoomLLM()).coherence_review([_story()])
    assert rep.coherent_rate == 1.0
    assert rep.failed is False


def test_coherence_flags_incoherent_cluster():
    rep = EvaluationService(_CoherenceLLM()).coherence_review([_multi_story()])
    assert rep.coherent_rate == 0.0
    assert rep.clusters[0].coherent is False


def test_coherence_fails_soft():
    rep = EvaluationService(_BoomLLM()).coherence_review([_multi_story()])
    assert rep.failed is True


def test_brief_review_returns_checks():
    rep = EvaluationService(_BriefReviewLLM()).brief_review("# brief")
    assert isinstance(rep, BriefReview)
    assert rep.scannable is True
    assert rep.unexplained_jargon_terms == 1


def test_brief_review_fails_soft():
    rep = EvaluationService(_BoomLLM()).brief_review("# brief")
    assert rep.failed is True
