from dailyai.llm_client import LLMClient
from dailyai.services.drafting import DraftingService
from dailyai.services.summarizing import ArticleSummary


def _summary(url="https://ex.com/a") -> ArticleSummary:
    return ArticleSummary(
        url=url, source="Ex", title="T", summary="A launched.", key_points=["fast"]
    )


class _CaptureLLM(LLMClient):
    def __init__(self):
        self.system: str | None = None
        self.user: str | None = None

    def generate(self, system, user, response_format=None):
        self.system = system
        self.user = user
        return "# Draft brief"


def test_draft_passes_morning_brew_voice_and_citations():
    llm = _CaptureLLM()
    out = DraftingService(llm).draft([_summary("https://ex.com/x")])
    assert out == "# Draft brief"
    assert llm.system is not None
    assert llm.user is not None
    assert "Morning Brew" in llm.system  # reuses the existing voice
    assert "https://ex.com/x" in llm.user  # citation reaches the editor


def test_draft_includes_summary_and_key_points():
    llm = _CaptureLLM()
    DraftingService(llm).draft([_summary()])
    assert llm.user is not None
    assert "A launched." in llm.user
    assert "- fast" in llm.user
