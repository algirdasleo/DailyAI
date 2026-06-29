from datetime import datetime, timedelta, timezone

from dailyai.models import Article, SourceCategory, Story
from dailyai.services.ranking import LLMRanker, RankingOrder
from tests.services._fakellm import FakeLLMBase

_PUB = datetime(2026, 6, 28, 12, tzinfo=timezone.utc)


def _article(title="t", authority=0.5, source="s", url=None, published=_PUB) -> Article:
    return Article(
        title=title,
        url=url or f"https://x.com/{title}-{source}",
        source=source,
        authority=authority,
        published=published,
        source_type=SourceCategory.API,
    )


def _story(articles) -> Story:
    return Story(articles=articles)


class _FakeLLM(FakeLLMBase):
    """Returns a fixed RankingOrder and records that it was called once."""

    def __init__(self, order):
        self._order = order
        self.calls = 0

    def generate(self, system, user, response_format=None):
        self.calls += 1
        return RankingOrder(order=list(self._order))


class _BoomLLM(FakeLLMBase):
    def generate(self, system, user, response_format=None):
        raise RuntimeError("llm down")


def test_empty_returns_empty():
    assert LLMRanker(_FakeLLM(order=[])).rank([]) == []


def test_sets_signals():
    s = _story([_article(authority=0.7)])
    ranked = LLMRanker(_FakeLLM(order=[1])).rank([s])
    assert ranked[0].rank_metadata is not None
    assert {"source_count", "authority", "recency_hours"} <= set(
        ranked[0].rank_metadata.__dict__
    )


def test_llm_orders_all_stories():
    # Equal inputs => stable baseline order = input order, so the LLM-order ids
    # map predictably onto the candidate list.
    a = _story([_article(title="A")])
    b = _story([_article(title="B")])
    c = _story([_article(title="C")])
    ranked = LLMRanker(_FakeLLM(order=[3, 2, 1])).rank([a, b, c])
    assert [s.main_article.title for s in ranked] == ["C", "B", "A"]


def test_llm_appends_omitted_ids():
    a = _story([_article(title="A")])
    b = _story([_article(title="B")])
    c = _story([_article(title="C")])
    ranked = LLMRanker(_FakeLLM(order=[2])).rank([a, b, c])
    assert [s.main_article.title for s in ranked] == ["B", "A", "C"]


def test_llm_ignores_invalid_and_duplicate_ids():
    a = _story([_article(title="A")])
    b = _story([_article(title="B")])
    ranked = LLMRanker(_FakeLLM(order=[99, 2, 2, 1])).rank([a, b])
    assert [s.main_article.title for s in ranked] == ["B", "A"]


def test_falls_back_to_baseline_on_error():
    # More sources, then higher authority, then newer wins the baseline order.
    minor = _story([_article(title="minor", authority=0.5, source="a")])
    big = _story([_article(title="big", source="a"), _article(title="big", source="b")])
    newer = _story(
        [
            _article(
                title="newer",
                authority=0.5,
                source="a",
                published=_PUB + timedelta(hours=5),
            )
        ]
    )
    ranked = LLMRanker(_BoomLLM()).rank([minor, newer, big])
    assert [s.main_article.title for s in ranked] == ["big", "newer", "minor"]
