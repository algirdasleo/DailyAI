from dailyai.services.fetching import ArticleText
from dailyai.services.summarizing import (
    ArticleSummary,
    SummarizerService,
    SummaryDraft,
)
from tests.services._fakellm import FakeLLMBase


def _text(url="https://ex.com/a") -> ArticleText:
    return ArticleText(url=url, title="T", source="Ex", full_text="body about a launch")


class _FakeLLM(FakeLLMBase):
    def __init__(self, draft):
        self._draft = draft
        self.calls = 0

    def generate(self, system, user, response_format=None):
        self.calls += 1
        return self._draft


class _BoomLLM(FakeLLMBase):
    def generate(self, system, user, response_format=None):
        raise RuntimeError("llm down")


def test_summarize_stitches_citation_from_source_not_model():
    draft = SummaryDraft(summary="A new model launched.", key_points=["benchmarks up"])
    out = SummarizerService(_FakeLLM(draft)).summarize([_text("https://ex.com/x")])
    assert len(out) == 1
    s = out[0]
    assert isinstance(s, ArticleSummary)
    assert s.url == "https://ex.com/x"  # url comes from ArticleText, not the model
    assert s.summary == "A new model launched."
    assert s.key_points == ["benchmarks up"]


def test_summarize_one_call_per_article():
    draft = SummaryDraft(summary="s", key_points=[])
    llm = _FakeLLM(draft)
    SummarizerService(llm).summarize(
        [_text("https://ex.com/1"), _text("https://ex.com/2")]
    )
    assert llm.calls == 2


def test_summarize_fails_soft_skips_failed_article():
    out = SummarizerService(_BoomLLM()).summarize([_text()])
    assert out == []
