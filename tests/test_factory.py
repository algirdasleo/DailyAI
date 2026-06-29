from dailyai.config import get_settings
from dailyai.factory import _build_ranker
from dailyai.services import LLMRanker


def test_build_ranker_is_llm(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("RANK_MODEL", "m")
    settings = get_settings()
    assert isinstance(_build_ranker(settings), LLMRanker)


def test_build_pipeline_wires_brief_services(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("DB_PATH", str(tmp_path / "t.db"))
    from dailyai.factory import build_pipeline
    from dailyai.services import ArticleFetchService, DraftingService, SummarizerService

    settings = get_settings()
    pipe = build_pipeline(settings, feeds_path="feeds.yaml")
    assert isinstance(pipe.fetch_service, ArticleFetchService)
    assert isinstance(pipe.summarizer_service, SummarizerService)
    assert isinstance(pipe.drafting_service, DraftingService)
